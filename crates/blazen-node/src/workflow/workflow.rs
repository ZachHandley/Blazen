//! JavaScript workflow builder and runner.
//!
//! Provides [`JsWorkflow`] which lets TypeScript/JavaScript users define
//! workflows with step handlers as async functions.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use blazen_core::SessionRefDeserializerFn;
use blazen_core::runtime;
use blazen_core::session_ref::{
    SERIALIZED_SESSION_REFS_META_KEY, SessionPausePolicy as CoreSessionPausePolicy,
};
use blazen_events::{AnyEvent, intern_event_type};
use napi::Status;
use napi::bindgen_prelude::*;
use napi::threadsafe_function::{ThreadsafeFunction, ThreadsafeFunctionCallMode};
use napi_derive::napi;
use tokio_stream::StreamExt;

use super::context::JsContext;
use super::event::{any_event_to_js_value, js_value_to_any_event};
use super::handler::JsWorkflowHandler;
use super::session_ref_serializable::{DESERIALIZER_FN, intern_type_tag};
use crate::error::workflow_error_to_napi;
#[cfg(not(target_os = "wasi"))]
use crate::persist::JsCheckpointStore;

// ---------------------------------------------------------------------------
// SessionPausePolicy enum
// ---------------------------------------------------------------------------

/// Policy applied to live session references when a workflow is paused
/// or snapshotted.
///
/// Mirrors the Rust-side
/// [`SessionPausePolicy`](blazen_core::session_ref::SessionPausePolicy)
/// enum and the Python `SessionPausePolicy` exposed by the pyo3
/// bindings. Configure it on a [`JsWorkflow`] via
/// [`JsWorkflow::set_session_pause_policy`] before calling
/// `run`/`runWithHandler`/`resume`.
///
/// Variants:
///
/// - `PickleOrError`: best-effort pickle each live ref; fail the pause
///   with a descriptive error if a ref cannot be captured. This is the
///   default.
/// - `PickleOrSerialize`: same as `PickleOrError` but additionally
///   honours the [`SessionRefSerializable`](blazen_core::session_ref::SessionRefSerializable)
///   protocol — values registered via
///   `ctx.insertSessionRefSerializable(typeName, bytes)` are captured
///   as opaque bytes in snapshot metadata and reconstructed on resume
///   via [`JsWorkflow::resume_with_serializable_refs`].
/// - `WarnDrop`: log a warning and drop each live ref. Downstream
///   `__blazen_session_ref__` markers carrying dropped UUIDs become
///   unresolved.
/// - `HardError`: fail the pause immediately if any live refs exist.
#[napi(string_enum, js_name = "SessionPausePolicy")]
#[derive(Clone, Copy)]
pub enum JsSessionPausePolicy {
    PickleOrError,
    PickleOrSerialize,
    WarnDrop,
    HardError,
}

impl From<JsSessionPausePolicy> for CoreSessionPausePolicy {
    fn from(p: JsSessionPausePolicy) -> Self {
        match p {
            JsSessionPausePolicy::PickleOrError => Self::PickleOrError,
            JsSessionPausePolicy::PickleOrSerialize => Self::PickleOrSerialize,
            JsSessionPausePolicy::WarnDrop => Self::WarnDrop,
            JsSessionPausePolicy::HardError => Self::HardError,
        }
    }
}

impl From<CoreSessionPausePolicy> for JsSessionPausePolicy {
    fn from(p: CoreSessionPausePolicy) -> Self {
        match p {
            CoreSessionPausePolicy::PickleOrError => Self::PickleOrError,
            CoreSessionPausePolicy::PickleOrSerialize => Self::PickleOrSerialize,
            CoreSessionPausePolicy::WarnDrop => Self::WarnDrop,
            CoreSessionPausePolicy::HardError => Self::HardError,
        }
    }
}

// ---------------------------------------------------------------------------
// Type aliases for ThreadsafeFunction variants
// ---------------------------------------------------------------------------

/// Step handler: takes (event, ctx) and returns a `serde_json::Value`.
///
/// The `CalleeHandled = false` disables the error-first callback convention.
/// Without this, napi-rs prepends a `null` first argument to the JS handler,
/// shifting all parameters by one.
///
/// We use `FnArgs<(A, B)>` instead of a bare tuple `(A, B)` because bare tuples
/// implement `ToNapiValue` (serialised as a JS Array), which means the blanket
/// `JsValuesTupleIntoVec` impl converts them into a **single** Array argument.
/// `FnArgs` has a dedicated `JsValuesTupleIntoVec` impl that **spreads** the
/// elements into separate JS function arguments.
///
/// The `Return` type is `Promise<serde_json::Value>` because JS handlers are
/// `async` functions that return a `Promise`. Using bare `serde_json::Value`
/// would try to serialise the Promise object itself (yielding `{}`).
///
/// `Weak = true` unrefs the TSFN so it does not prevent Node.js from exiting
/// once the workflow completes and the result Promise resolves.
type StepHandlerTsfn = ThreadsafeFunction<
    FnArgs<(serde_json::Value, JsContext)>,
    Promise<serde_json::Value>,
    FnArgs<(serde_json::Value, JsContext)>,
    Status,
    false,
    true,
>;

/// Stream callback: takes a `serde_json::Value`, returns nothing meaningful.
/// We use the default Unknown return type and fire-and-forget via `call`.
/// `CalleeHandled = false` to avoid the error-first callback convention.
/// `Weak = true` so it does not prevent Node.js from exiting.
type StreamCallbackTsfn =
    ThreadsafeFunction<serde_json::Value, Unknown<'static>, serde_json::Value, Status, false, true>;

// ---------------------------------------------------------------------------
// Step registration data
// ---------------------------------------------------------------------------

/// Internal representation of a step before building the workflow.
///
/// The handler is wrapped in `Arc` because `ThreadsafeFunction` does not
/// implement `Clone`, but the workflow engine needs to clone step handlers
/// for concurrent dispatch.
struct JsStepRegistration {
    name: String,
    event_types: Vec<String>,
    handler: Arc<StepHandlerTsfn>,
    /// Per-step wall-clock timeout (mirrors Rust `StepRegistration::timeout`).
    /// `None` means inherit the workflow-level timeout.
    timeout_secs: Option<f64>,
    /// Per-step retry override (mirrors Rust `StepRegistration::retry_config`).
    /// `None` means inherit from the workflow / pipeline / provider scope.
    retry_config: Option<blazen_llm::retry::RetryConfig>,
}

// ---------------------------------------------------------------------------
// JsWorkflowResult
// ---------------------------------------------------------------------------

/// The result of a workflow run.
#[napi(object)]
pub struct JsWorkflowResult {
    /// The event type of the final result (typically "`blazen::StopEvent`").
    #[napi(js_name = "type")]
    pub event_type: String,
    /// The result data as a JSON object.
    pub data: serde_json::Value,
    /// Aggregated token usage across the run, mirroring
    /// [`blazen_core::WorkflowResult::usage_total`].
    #[napi(js_name = "usageTotal")]
    pub usage_total: crate::generated::JsTokenUsage,
    /// Aggregated cost in USD across the run, mirroring
    /// [`blazen_core::WorkflowResult::cost_total_usd`].
    #[napi(js_name = "costTotalUsd")]
    pub cost_total_usd: f64,
}

/// Plain-object descriptor for one branch of a parallel fan-out — every
/// field except `workflowName` is metadata. The actual child `Workflow`
/// instances are passed alongside the spec array in
/// [`JsWorkflow::addParallelSubworkflows`] (napi cannot embed napi-class
/// values inside `#[napi(object)]` shapes).
#[napi(object, js_name = "SubWorkflowBranchSpec")]
pub struct SubWorkflowBranchSpec {
    /// Human-readable name for this branch.
    pub name: String,
    /// Event types this branch's parent step accepts.
    pub accepts: Vec<String>,
    /// Event types this branch's parent step may emit (informational).
    pub emits: Vec<String>,
    /// Optional wall-clock timeout in seconds for this branch.
    #[napi(js_name = "timeoutSecs")]
    pub timeout_secs: Option<f64>,
    /// Optional retry config for this branch.
    #[napi(js_name = "retryConfig")]
    pub retry_config: Option<crate::generated::JsRetryConfig>,
}

// ---------------------------------------------------------------------------
// JsWorkflow
// ---------------------------------------------------------------------------

/// A workflow builder and runner.
///
/// Create a workflow, add steps with async handler functions, then run it.
///
/// ```javascript
/// const workflow = new Workflow("my-workflow");
///
/// workflow.addStep("analyze", ["blazen::StartEvent"], async (event, ctx) => {
///   const text = event.message;
///   return { type: "blazen::StopEvent", result: { analyzed: text } };
/// });
///
/// const result = await workflow.run({ message: "hello" });
/// ```
#[napi(js_name = "Workflow")]
pub struct JsWorkflow {
    name: String,
    steps: Vec<JsStepRegistration>,
    timeout_secs: Option<f64>,
    /// Workflow-level default retry config; threaded into
    /// [`blazen_core::WorkflowBuilder::retry_config`] at build time.
    retry_config: Option<blazen_llm::retry::RetryConfig>,
    /// Sub-workflow steps registered via
    /// [`Self::add_subworkflow_step`] (Wave 5).
    subworkflow_steps: Vec<JsSubWorkflowStepInner>,
    /// Parallel sub-workflow fan-outs registered via
    /// [`Self::add_parallel_subworkflows`] (Wave 5).
    parallel_subworkflow_steps: Vec<JsParallelSubWorkflowsInner>,
    session_pause_policy: JsSessionPausePolicy,
    auto_publish_events: bool,
}

/// Internal record for a sub-workflow step pending registration.
#[derive(Clone)]
struct JsSubWorkflowStepInner {
    name: String,
    accepts: Vec<String>,
    emits: Vec<String>,
    inner_workflow: Arc<blazen_core::Workflow>,
    timeout_secs: Option<f64>,
    retry_config: Option<blazen_llm::retry::RetryConfig>,
}

/// Internal record for a parallel sub-workflows fan-out pending registration.
#[derive(Clone)]
struct JsParallelSubWorkflowsInner {
    name: String,
    accepts: Vec<String>,
    emits: Vec<String>,
    branches: Vec<JsSubWorkflowStepInner>,
    join_strategy: blazen_core::JoinStrategy,
}

#[napi]
#[allow(clippy::must_use_candidate, clippy::missing_errors_doc)]
impl JsWorkflow {
    /// Create a new workflow with the given name.
    #[napi(constructor)]
    pub fn new(name: String) -> Self {
        Self {
            name,
            steps: Vec::new(),
            timeout_secs: Some(300.0), // 5 min default
            retry_config: None,
            subworkflow_steps: Vec::new(),
            parallel_subworkflow_steps: Vec::new(),
            session_pause_policy: JsSessionPausePolicy::PickleOrError,
            // Mirror the Rust `WorkflowBuilder` default (Wave 7).
            auto_publish_events: true,
        }
    }

    /// Start a fluent builder for a workflow with the given name.
    ///
    /// ```javascript
    /// const wf = Workflow.builder("my-wf")
    ///     .addStep("first", ["blazen::StartEvent"], async (ev, ctx) => /* ... */)
    ///     .timeout(60)
    ///     .autoPublishEvents(true)
    ///     .build();
    /// ```
    #[napi]
    pub fn builder(name: String) -> JsWorkflowBuilder {
        JsWorkflowBuilder::new_internal(name)
    }

    /// Configure how live session refs are treated when the workflow
    /// is paused or snapshotted.
    ///
    /// Defaults to `SessionPausePolicy.PickleOrError`. Set this to
    /// `SessionPausePolicy.PickleOrSerialize` to opt into the
    /// `insertSessionRefSerializable` round-trip path.
    #[napi(js_name = "setSessionPausePolicy")]
    pub fn set_session_pause_policy(&mut self, policy: JsSessionPausePolicy) {
        self.session_pause_policy = policy;
    }

    /// Enable or disable automatic publishing of lifecycle events to the
    /// broadcast stream.
    ///
    /// When enabled, the event loop publishes `DynamicEvent`s with type
    /// `"blazen::lifecycle"` at key decision points (event routed, step
    /// started, step completed, step failed). Defaults to `false`.
    #[napi(js_name = "setAutoPublishEvents")]
    pub fn set_auto_publish_events(&mut self, enabled: bool) {
        self.auto_publish_events = enabled;
    }

    /// Register a sub-workflow step that delegates to another `Workflow`.
    /// Mirrors [`blazen_core::WorkflowBuilder::add_subworkflow_step`].
    ///
    /// - `name`: human-readable step name.
    /// - `accepts`: array of event type strings this step handles.
    /// - `emits`: array of event type strings this step may emit (informational).
    /// - `inner`: the child `Workflow` to run.
    /// - `timeoutSecs`: optional wall-clock timeout for the child run.
    /// - `retryConfig`: optional retry policy for the child run.
    #[napi(js_name = "addSubworkflowStep")]
    #[allow(clippy::needless_pass_by_value)]
    pub fn add_subworkflow_step(
        &mut self,
        name: String,
        accepts: Vec<String>,
        emits: Vec<String>,
        inner: &JsWorkflow,
        timeout_secs: Option<f64>,
        retry_config: Option<crate::generated::JsRetryConfig>,
    ) -> Result<()> {
        let inner_workflow = Arc::new(inner.build_workflow()?);
        self.subworkflow_steps.push(JsSubWorkflowStepInner {
            name,
            accepts,
            emits,
            inner_workflow,
            timeout_secs,
            retry_config: retry_config.map(Into::into),
        });
        Ok(())
    }

    /// Register a parallel sub-workflows fan-out step. Mirrors
    /// [`blazen_core::WorkflowBuilder::add_parallel_subworkflows`].
    ///
    /// `branchSpecs` and `branchWorkflows` are parallel arrays of equal
    /// length: each `(spec, wf)` pair becomes one branch. They're split
    /// into two parameters because napi-rs cannot embed napi-class values
    /// (the `Workflow` instances) inside `#[napi(object)]` shapes.
    ///
    /// `joinStrategy` defaults to `WaitAll`.
    #[napi(js_name = "addParallelSubworkflows")]
    #[allow(clippy::needless_pass_by_value)]
    pub fn add_parallel_subworkflows(
        &mut self,
        name: String,
        accepts: Vec<String>,
        emits: Vec<String>,
        branch_specs: Vec<SubWorkflowBranchSpec>,
        branch_workflows: Vec<&JsWorkflow>,
        join_strategy: Option<crate::pipeline::stage::JsJoinStrategy>,
    ) -> Result<()> {
        if branch_specs.len() != branch_workflows.len() {
            return Err(napi::Error::new(
                Status::InvalidArg,
                format!(
                    "addParallelSubworkflows: branchSpecs ({}) and branchWorkflows ({}) must be the same length",
                    branch_specs.len(),
                    branch_workflows.len()
                ),
            ));
        }
        let mut inner_branches = Vec::with_capacity(branch_specs.len());
        for (spec, wf) in branch_specs.into_iter().zip(branch_workflows) {
            let inner_workflow = Arc::new(wf.build_workflow()?);
            inner_branches.push(JsSubWorkflowStepInner {
                name: spec.name,
                accepts: spec.accepts,
                emits: spec.emits,
                inner_workflow,
                timeout_secs: spec.timeout_secs,
                retry_config: spec.retry_config.map(Into::into),
            });
        }
        let join_core = match join_strategy
            .unwrap_or(crate::pipeline::stage::JsJoinStrategy::WaitAll)
        {
            crate::pipeline::stage::JsJoinStrategy::WaitAll => blazen_core::JoinStrategy::WaitAll,
            crate::pipeline::stage::JsJoinStrategy::FirstCompletes => {
                blazen_core::JoinStrategy::FirstCompletes
            }
        };
        self.parallel_subworkflow_steps
            .push(JsParallelSubWorkflowsInner {
                name,
                accepts,
                emits,
                branches: inner_branches,
                join_strategy: join_core,
            });
        Ok(())
    }

    /// Register a pre-built [`SubWorkflowStep`] wrapper.
    ///
    /// Object-form of [`Self::add_subworkflow_step`]: the same step instance
    /// can be reused across multiple workflows since its inner child
    /// workflow is captured in `Arc` form at construction time.
    #[napi(js_name = "addSubworkflowStepObj")]
    #[allow(clippy::needless_pass_by_value)]
    pub fn add_subworkflow_step_obj(
        &mut self,
        step: &crate::workflow::subworkflow_step::JsSubWorkflowStep,
    ) -> Result<()> {
        self.subworkflow_steps.push(JsSubWorkflowStepInner {
            name: step.name.clone(),
            accepts: step.accepts.clone(),
            emits: step.emits.clone(),
            inner_workflow: Arc::clone(&step.inner_workflow),
            timeout_secs: step.timeout_secs,
            retry_config: step.retry_config.clone(),
        });
        Ok(())
    }

    /// Register a pre-built [`ParallelSubWorkflowsStep`] wrapper.
    ///
    /// Object-form of [`Self::add_parallel_subworkflows`]: lifts the
    /// awkward parallel-arrays signature into a single class instance.
    #[napi(js_name = "addParallelSubworkflowsObj")]
    #[allow(clippy::needless_pass_by_value)]
    pub fn add_parallel_subworkflows_obj(
        &mut self,
        step: &crate::workflow::subworkflow_step::JsParallelSubWorkflowsStep,
    ) -> Result<()> {
        let mut branches = Vec::with_capacity(step.branches.len());
        for b in &step.branches {
            branches.push(JsSubWorkflowStepInner {
                name: b.name.clone(),
                accepts: b.accepts.clone(),
                emits: b.emits.clone(),
                inner_workflow: Arc::clone(&b.inner_workflow),
                timeout_secs: b.timeout_secs,
                retry_config: b.retry_config.clone(),
            });
        }
        let join_core = match step.join_strategy {
            crate::pipeline::stage::JsJoinStrategy::WaitAll => blazen_core::JoinStrategy::WaitAll,
            crate::pipeline::stage::JsJoinStrategy::FirstCompletes => {
                blazen_core::JoinStrategy::FirstCompletes
            }
        };
        self.parallel_subworkflow_steps
            .push(JsParallelSubWorkflowsInner {
                name: step.name.clone(),
                accepts: step.accepts.clone(),
                emits: step.emits.clone(),
                branches,
                join_strategy: join_core,
            });
        Ok(())
    }

    /// Add a step to the workflow.
    ///
    /// - `name`: Human-readable step name.
    /// - `eventTypes`: Array of event type strings this step handles.
    /// - `handler`: Async function `(event, ctx) => Event` that processes
    ///   events and returns the next event.
    #[napi(js_name = "addStep")]
    pub fn add_step(
        &mut self,
        name: String,
        event_types: Vec<String>,
        handler: StepHandlerTsfn,
    ) -> Result<()> {
        self.steps.push(JsStepRegistration {
            name,
            event_types,
            handler: Arc::new(handler),
            timeout_secs: None,
            retry_config: None,
        });
        Ok(())
    }

    /// Set the workflow timeout in seconds.
    ///
    /// Set to 0 or negative to disable the timeout.
    #[napi(js_name = "setTimeout")]
    pub fn set_timeout(&mut self, seconds: f64) {
        if seconds <= 0.0 {
            self.timeout_secs = None;
        } else {
            self.timeout_secs = Some(seconds);
        }
    }

    /// Run the workflow with the given input data.
    ///
    /// The input is wrapped in a `StartEvent` automatically.
    /// Returns the final result when the workflow completes via a `StopEvent`.
    #[napi]
    pub async fn run(&self, input: serde_json::Value) -> Result<JsWorkflowResult> {
        let workflow = self.build_workflow()?;

        let handler = run_with_optional_parent_registry(&workflow, input).await?;

        let result = handler.result().await.map_err(workflow_error_to_napi)?;

        Ok(make_result(&result))
    }

    /// Run the workflow with streaming.
    ///
    /// The `onEvent` callback receives intermediate events published via
    /// `ctx.writeEventToStream()` from within step handlers.
    ///
    /// Returns the final result when the workflow completes.
    #[napi(js_name = "runStreaming")]
    pub async fn run_streaming(
        &self,
        input: serde_json::Value,
        on_event: StreamCallbackTsfn,
    ) -> Result<JsWorkflowResult> {
        let workflow = self.build_workflow()?;

        let handler = run_with_optional_parent_registry(&workflow, input).await?;

        // Subscribe to the stream before awaiting the result.
        let mut stream = handler.stream_events();

        // Spawn a task to forward stream events to the JS callback.
        // We use `call` with `NonBlocking` mode (fire-and-forget) because:
        // 1. We don't need the return value from the stream callback.
        // 2. `call_async` returns a future that is not Send-safe.
        let on_event = Arc::new(on_event);
        let on_event_clone = Arc::clone(&on_event);
        let stream_handle = runtime::spawn(async move {
            while let Some(event) = stream.next().await {
                // Stop on the stream-end sentinel (same as Python bindings).
                if event.event_type_id() == "blazen::StreamEnd" {
                    break;
                }
                let js_event = any_event_to_js_value(&*event);
                // Fire-and-forget: call the JS callback without awaiting.
                let _ = on_event_clone.call(js_event, ThreadsafeFunctionCallMode::NonBlocking);
            }
        });

        let result = handler.result().await.map_err(workflow_error_to_napi)?;

        // Wait for the stream consumer to finish.
        let _ = stream_handle.await;

        Ok(make_result(&result))
    }

    /// Run the workflow and return a handler object.
    ///
    /// Unlike `run()` which awaits the result internally, this method
    /// returns a `WorkflowHandler` that gives you control over the
    /// running workflow:
    ///
    /// - Call `handler.result()` to await the final result.
    /// - Call `handler.pause()` to pause and get a snapshot JSON string.
    /// - Call `handler.streamEvents(cb)` to subscribe to intermediate events.
    ///
    /// ```javascript
    /// const handler = await workflow.runWithHandler({ message: "hello" });
    /// // ... later ...
    /// const snapshot = await handler.pause();
    /// ```
    #[napi(js_name = "runWithHandler")]
    pub async fn run_with_handler(&self, input: serde_json::Value) -> Result<JsWorkflowHandler> {
        let workflow = self.build_workflow()?;
        let handler = run_with_optional_parent_registry(&workflow, input).await?;
        Ok(JsWorkflowHandler::new(handler))
    }

    /// Resume a previously paused workflow from a snapshot.
    ///
    /// The snapshot JSON string should have been obtained from a prior
    /// call to `handler.pause()`. The workflow will resume with the same
    /// steps that were registered on this `Workflow` instance.
    ///
    /// Returns a new `WorkflowHandler` for the resumed workflow.
    ///
    /// ```javascript
    /// const snapshot = fs.readFileSync("snapshot.json", "utf-8");
    /// const handler = await workflow.resume(snapshot);
    /// const result = await handler.result();
    /// ```
    #[napi]
    pub async fn resume(&self, snapshot_json: String) -> Result<JsWorkflowHandler> {
        let snapshot = blazen_core::WorkflowSnapshot::from_json(&snapshot_json)
            .map_err(workflow_error_to_napi)?;

        // Build the step registrations from the JS steps.
        let steps: Vec<blazen_core::StepRegistration> =
            self.steps.iter().map(make_step_registration).collect();

        // Use the workflow's configured timeout.
        let timeout = self.timeout_secs.map(Duration::from_secs_f64);

        let handler = blazen_core::Workflow::resume(snapshot, steps, timeout)
            .await
            .map_err(workflow_error_to_napi)?;

        Ok(JsWorkflowHandler::new(handler))
    }

    /// Resume a workflow from a snapshot, rehydrating every
    /// `SessionPausePolicy.PickleOrSerialize` session-ref entry into
    /// the resumed registry under its original [`RegistryKey`].
    ///
    /// This is the resume-side counterpart of pausing a workflow whose
    /// `sessionPausePolicy` is set to `PickleOrSerialize`. The
    /// snapshot's `__blazen_serialized_session_refs` sidecar carries
    /// `(typeName, bytes)` records for every payload that was inserted
    /// via `ctx.insertSessionRefSerializable`. This method walks that
    /// sidecar, registers a no-op rehydrator for each unique type tag,
    /// and lets the core
    /// [`Workflow::resume_with_deserializers`](blazen_core::Workflow::resume_with_deserializers)
    /// path repopulate the registry. After this call, JS code can
    /// retrieve the original bytes via
    /// `ctx.getSessionRefSerializable(key)` and deserialize them
    /// itself.
    ///
    /// Snapshots that do **not** contain any serialized session refs
    /// work fine with the plain [`Self::resume`] entrypoint. Use this
    /// method only when you need the serializable rehydration path.
    ///
    /// ```javascript
    /// const snap = fs.readFileSync("snapshot.json", "utf-8");
    /// const handler = await workflow.resumeWithSerializableRefs(snap);
    /// const result = await handler.result();
    /// ```
    #[napi(js_name = "resumeWithSerializableRefs")]
    pub async fn resume_with_serializable_refs(
        &self,
        snapshot_json: String,
    ) -> Result<JsWorkflowHandler> {
        let snapshot = blazen_core::WorkflowSnapshot::from_json(&snapshot_json)
            .map_err(workflow_error_to_napi)?;

        // Walk the serialized-refs sidecar and register the Node-side
        // trampoline for every unique type tag referenced inside it.
        // The trampoline simply re-wraps the captured bytes in a fresh
        // `NodeSessionRefSerializable`, which is enough for JS callers
        // to read them back via `ctx.getSessionRefSerializable`.
        let mut deserializers: HashMap<&'static str, SessionRefDeserializerFn> = HashMap::new();
        if let Some(serde_json::Value::Object(entries)) =
            snapshot.metadata.get(SERIALIZED_SESSION_REFS_META_KEY)
        {
            for record in entries.values() {
                if let Some(type_tag) = record.get("type_tag").and_then(serde_json::Value::as_str) {
                    let interned = intern_type_tag(type_tag);
                    deserializers.insert(interned, DESERIALIZER_FN);
                }
            }
        }

        // Build the step registrations from the JS steps.
        let steps: Vec<blazen_core::StepRegistration> =
            self.steps.iter().map(make_step_registration).collect();

        // Use the workflow's configured timeout.
        let timeout = self.timeout_secs.map(Duration::from_secs_f64);

        let handler = blazen_core::Workflow::resume_with_deserializers(
            snapshot,
            steps,
            deserializers,
            timeout,
        )
        .await
        .map_err(workflow_error_to_napi)?;

        Ok(JsWorkflowHandler::new(handler))
    }
}

impl JsWorkflow {
    /// Build the internal `Workflow` from the registered steps.
    pub(crate) fn build_workflow(&self) -> Result<blazen_core::Workflow> {
        let mut builder = blazen_core::WorkflowBuilder::new(self.name.clone());

        if let Some(secs) = self.timeout_secs {
            builder = builder.timeout(Duration::from_secs_f64(secs));
        } else {
            builder = builder.no_timeout();
        }

        if let Some(cfg) = self.retry_config.as_ref() {
            builder = builder.retry_config(cfg.clone());
        }

        for step in &self.steps {
            let registration = make_step_registration(step);
            builder = builder.step(registration);
        }

        for sub in &self.subworkflow_steps {
            builder = builder.add_subworkflow_step(make_subworkflow_step(sub));
        }

        for par in &self.parallel_subworkflow_steps {
            builder = builder.add_parallel_subworkflows(make_parallel_subworkflows_step(par));
        }

        builder = builder.session_pause_policy(self.session_pause_policy.into());
        builder = builder.auto_publish_events(self.auto_publish_events);

        builder.build().map_err(workflow_error_to_napi)
    }
}

// ---------------------------------------------------------------------------
// Sub-workflow step helpers (Wave 5)
// ---------------------------------------------------------------------------

fn make_subworkflow_step(sub: &JsSubWorkflowStepInner) -> blazen_core::SubWorkflowStep {
    let accepts: Vec<&'static str> = sub.accepts.iter().map(|s| intern_event_type(s)).collect();
    let emits: Vec<&'static str> = sub.emits.iter().map(|s| intern_event_type(s)).collect();

    // Identity input mapper: pass through the parent event's `to_json()`.
    let input_mapper: blazen_core::SubWorkflowInputMapper = Arc::new(|event| event.to_json());
    // Wrap the child workflow's terminal `StopEvent.result` JSON in a
    // generic `DynamicEvent` so the parent step can consume it.
    let output_mapper: blazen_core::SubWorkflowOutputMapper = Arc::new(|json| {
        Box::new(blazen_events::DynamicEvent {
            event_type: "blazen::StopEvent".to_owned(),
            data: serde_json::json!({ "type": "blazen::StopEvent", "result": json }),
        }) as Box<dyn AnyEvent>
    });

    blazen_core::SubWorkflowStep {
        name: sub.name.clone(),
        accepts,
        emits,
        workflow: Arc::clone(&sub.inner_workflow),
        input_mapper,
        output_mapper,
        timeout: sub.timeout_secs.map(Duration::from_secs_f64),
        retry_config: sub.retry_config.as_ref().map(|c| Arc::new(c.clone())),
    }
}

fn make_parallel_subworkflows_step(
    par: &JsParallelSubWorkflowsInner,
) -> blazen_core::ParallelSubWorkflowsStep {
    let accepts: Vec<&'static str> = par.accepts.iter().map(|s| intern_event_type(s)).collect();
    let emits: Vec<&'static str> = par.emits.iter().map(|s| intern_event_type(s)).collect();
    let branches = par.branches.iter().map(make_subworkflow_step).collect();
    blazen_core::ParallelSubWorkflowsStep {
        name: par.name.clone(),
        accepts,
        emits,
        branches,
        join_strategy: par.join_strategy,
    }
}

// ---------------------------------------------------------------------------
// JsWorkflowBuilder
// ---------------------------------------------------------------------------

/// Internal mutable state for [`JsWorkflowBuilder`]. Wrapped in a
/// [`Mutex`] so the builder can expose chainable `&self` methods (each
/// call mutates the inner state behind the lock).
#[allow(clippy::struct_excessive_bools)]
struct WorkflowBuilderState {
    name: String,
    steps: Vec<JsStepRegistration>,
    timeout_secs: Option<f64>,
    /// Workflow-level default retry configuration (mirrors
    /// `blazen_core::WorkflowBuilder::retry_config`). `None` means inherit
    /// from the next-outer scope.
    retry_config: Option<blazen_llm::retry::RetryConfig>,
    session_pause_policy: JsSessionPausePolicy,
    auto_publish_events: bool,
    /// Mirrors the `with_history` flag on
    /// [`blazen_core::WorkflowBuilder`]. Stored here for forward
    /// compatibility — the underlying call site is gated on the
    /// `telemetry` feature on `blazen-core`, which is not enabled in
    /// this binding's compilation, so the flag is currently a no-op.
    /// Setting it does NOT raise an error so JS code can opt into the
    /// API ahead of the feature being turned on without breaking.
    collect_history: bool,
    /// Mirrors `checkpoint_after_step`. Same forward-compatibility
    /// caveat as `collect_history`.
    checkpoint_after_step: bool,
    /// `true` once `checkpointStore(...)` has been called. Same caveat
    /// as `collect_history` — currently does not flow into the core
    /// builder because the `persist` feature is off here.
    checkpoint_store_set: bool,
}

/// Fluent builder for constructing a [`JsWorkflow`].
///
/// Obtained via [`JsWorkflow::builder`]. All configuration methods
/// take `&self` and return `&Self` so they can be chained from
/// JavaScript. `build()` consumes the builder; calling any other
/// method after `build()` raises a `napi::Error`.
///
/// ```javascript
/// const wf = Workflow.builder("my-wf")
///     .addStep("first", ["blazen::StartEvent"], async (ev, ctx) => {
///         return { type: "blazen::StopEvent", result: ev };
///     })
///     .timeout(60)
///     .autoPublishEvents(true)
///     .sessionPausePolicy(SessionPausePolicy.PickleOrSerialize)
///     .build();
/// ```
#[napi(js_name = "WorkflowBuilder")]
pub struct JsWorkflowBuilder {
    inner: Mutex<Option<WorkflowBuilderState>>,
}

#[napi]
#[allow(
    clippy::must_use_candidate,
    clippy::missing_errors_doc,
    clippy::missing_panics_doc,
    clippy::needless_pass_by_value
)]
impl JsWorkflowBuilder {
    /// Construct a fresh builder. Equivalent to
    /// [`JsWorkflow::builder`] but exposed as a constructor so JS code
    /// can also write `new WorkflowBuilder("name")`.
    #[napi(constructor)]
    pub fn new(name: String) -> Self {
        Self::new_internal(name)
    }

    /// Set the workflow name. Replaces any name passed to the
    /// constructor.
    #[napi]
    pub fn name(&self, name: String) -> Result<&Self> {
        let mut guard = self.inner.lock().expect("poisoned");
        let state = guard
            .as_mut()
            .ok_or_else(JsWorkflowBuilder::consumed_error)?;
        state.name = name;
        Ok(self)
    }

    /// Append a step. The step's `handler` is an `async (event, ctx)`
    /// JavaScript function; the workflow engine routes events whose
    /// `type` matches one of `eventTypes` to it.
    #[napi(js_name = "addStep")]
    pub fn add_step(
        &self,
        name: String,
        event_types: Vec<String>,
        handler: StepHandlerTsfn,
    ) -> Result<&Self> {
        let mut guard = self.inner.lock().expect("poisoned");
        let state = guard
            .as_mut()
            .ok_or_else(JsWorkflowBuilder::consumed_error)?;
        state.steps.push(JsStepRegistration {
            name,
            event_types,
            handler: Arc::new(handler),
            timeout_secs: None,
            retry_config: None,
        });
        Ok(self)
    }

    /// Set the workflow timeout in seconds. A non-positive value
    /// disables the timeout (equivalent to [`Self::no_timeout`]).
    #[napi]
    pub fn timeout(&self, seconds: f64) -> Result<&Self> {
        let mut guard = self.inner.lock().expect("poisoned");
        let state = guard
            .as_mut()
            .ok_or_else(JsWorkflowBuilder::consumed_error)?;
        if seconds <= 0.0 {
            state.timeout_secs = None;
        } else {
            state.timeout_secs = Some(seconds);
        }
        Ok(self)
    }

    /// Disable the workflow timeout — the workflow runs until a
    /// `StopEvent` is emitted.
    #[napi(js_name = "noTimeout")]
    pub fn no_timeout(&self) -> Result<&Self> {
        let mut guard = self.inner.lock().expect("poisoned");
        let state = guard
            .as_mut()
            .ok_or_else(JsWorkflowBuilder::consumed_error)?;
        state.timeout_secs = None;
        Ok(self)
    }

    /// Enable or disable automatic publishing of lifecycle events to
    /// the broadcast stream. See [`JsWorkflow::set_auto_publish_events`].
    #[napi(js_name = "autoPublishEvents")]
    pub fn auto_publish_events(&self, enabled: bool) -> Result<&Self> {
        let mut guard = self.inner.lock().expect("poisoned");
        let state = guard
            .as_mut()
            .ok_or_else(JsWorkflowBuilder::consumed_error)?;
        state.auto_publish_events = enabled;
        Ok(self)
    }

    /// Configure the [`JsSessionPausePolicy`] applied to live session
    /// refs at pause/snapshot time. Defaults to
    /// `SessionPausePolicy.PickleOrError`.
    #[napi(js_name = "sessionPausePolicy")]
    pub fn session_pause_policy(&self, policy: JsSessionPausePolicy) -> Result<&Self> {
        let mut guard = self.inner.lock().expect("poisoned");
        let state = guard
            .as_mut()
            .ok_or_else(JsWorkflowBuilder::consumed_error)?;
        state.session_pause_policy = policy;
        Ok(self)
    }

    /// Set a per-step wall-clock timeout in seconds on the most-recently
    /// added step. Mirrors [`blazen_core::WorkflowBuilder::step_timeout`].
    /// Errors if no step has been registered yet.
    #[napi(js_name = "stepTimeout")]
    pub fn step_timeout(&self, seconds: f64) -> Result<&Self> {
        let mut guard = self.inner.lock().expect("poisoned");
        let state = guard
            .as_mut()
            .ok_or_else(JsWorkflowBuilder::consumed_error)?;
        let step = state.steps.last_mut().ok_or_else(|| {
            napi::Error::new(
                Status::GenericFailure,
                "stepTimeout() called before any step was registered",
            )
        })?;
        step.timeout_secs = Some(seconds);
        Ok(self)
    }

    /// Clear the per-step timeout on the most-recently added step.
    #[napi(js_name = "noStepTimeout")]
    pub fn no_step_timeout(&self) -> Result<&Self> {
        let mut guard = self.inner.lock().expect("poisoned");
        let state = guard
            .as_mut()
            .ok_or_else(JsWorkflowBuilder::consumed_error)?;
        let step = state.steps.last_mut().ok_or_else(|| {
            napi::Error::new(
                Status::GenericFailure,
                "noStepTimeout() called before any step was registered",
            )
        })?;
        step.timeout_secs = None;
        Ok(self)
    }

    /// Set a per-step retry config on the most-recently added step.
    /// Mirrors [`blazen_core::WorkflowBuilder::step_retry`].
    #[napi(js_name = "stepRetry")]
    pub fn step_retry(&self, config: crate::generated::JsRetryConfig) -> Result<&Self> {
        let mut guard = self.inner.lock().expect("poisoned");
        let state = guard
            .as_mut()
            .ok_or_else(JsWorkflowBuilder::consumed_error)?;
        let step = state.steps.last_mut().ok_or_else(|| {
            napi::Error::new(
                Status::GenericFailure,
                "stepRetry() called before any step was registered",
            )
        })?;
        step.retry_config = Some(config.into());
        Ok(self)
    }

    /// Disable retries on the most-recently added step (`maxRetries = 0`).
    #[napi(js_name = "noStepRetry")]
    pub fn no_step_retry(&self) -> Result<&Self> {
        let mut guard = self.inner.lock().expect("poisoned");
        let state = guard
            .as_mut()
            .ok_or_else(JsWorkflowBuilder::consumed_error)?;
        let step = state.steps.last_mut().ok_or_else(|| {
            napi::Error::new(
                Status::GenericFailure,
                "noStepRetry() called before any step was registered",
            )
        })?;
        step.retry_config = Some(blazen_llm::retry::RetryConfig {
            max_retries: 0,
            ..blazen_llm::retry::RetryConfig::default()
        });
        Ok(self)
    }

    /// Set a workflow-level default retry config. Step / per-call
    /// overrides take precedence; pipeline / provider defaults take
    /// lower precedence.
    #[napi(js_name = "retryConfig")]
    pub fn retry_config(&self, config: crate::generated::JsRetryConfig) -> Result<&Self> {
        let mut guard = self.inner.lock().expect("poisoned");
        let state = guard
            .as_mut()
            .ok_or_else(JsWorkflowBuilder::consumed_error)?;
        state.retry_config = Some(config.into());
        Ok(self)
    }

    /// Disable workflow-level retries (`maxRetries = 0`).
    #[napi(js_name = "noRetry")]
    pub fn no_retry(&self) -> Result<&Self> {
        let mut guard = self.inner.lock().expect("poisoned");
        let state = guard
            .as_mut()
            .ok_or_else(JsWorkflowBuilder::consumed_error)?;
        state.retry_config = Some(blazen_llm::retry::RetryConfig {
            max_retries: 0,
            ..blazen_llm::retry::RetryConfig::default()
        });
        Ok(self)
    }

    /// Clear any workflow-level retry config.
    #[napi(js_name = "clearRetryConfig")]
    pub fn clear_retry_config(&self) -> Result<&Self> {
        let mut guard = self.inner.lock().expect("poisoned");
        let state = guard
            .as_mut()
            .ok_or_else(JsWorkflowBuilder::consumed_error)?;
        state.retry_config = None;
        Ok(self)
    }

    /// Enable history collection.
    ///
    /// Mirrors [`blazen_core::WorkflowBuilder::with_history`]. The
    /// underlying call is gated on the `telemetry` feature of
    /// `blazen-core`, which is **not** currently enabled in the Node
    /// binding's compilation. The flag is recorded on the builder for
    /// forward compatibility but does not yet flow into the core
    /// engine — calls to `withHistory()` complete successfully but do
    /// not change runtime behavior. The setting will start taking
    /// effect once the `blazen-core/telemetry` feature is enabled in
    /// `crates/blazen-node/Cargo.toml`.
    #[napi(js_name = "withHistory")]
    pub fn with_history(&self) -> Result<&Self> {
        let mut guard = self.inner.lock().expect("poisoned");
        let state = guard
            .as_mut()
            .ok_or_else(JsWorkflowBuilder::consumed_error)?;
        state.collect_history = true;
        Ok(self)
    }

    /// Enable or disable automatic checkpointing after each step
    /// completes. Same forward-compatibility caveat as
    /// [`Self::with_history`] — the flag is recorded but does not yet
    /// flow into the core engine.
    #[napi(js_name = "checkpointAfterStep")]
    pub fn checkpoint_after_step(&self, enabled: bool) -> Result<&Self> {
        let mut guard = self.inner.lock().expect("poisoned");
        let state = guard
            .as_mut()
            .ok_or_else(JsWorkflowBuilder::consumed_error)?;
        state.checkpoint_after_step = enabled;
        Ok(self)
    }

    /// Validate and produce the [`JsWorkflow`]. Consumes the builder —
    /// any subsequent method call (including a second `build()`) will
    /// raise a `napi::Error`.
    ///
    /// The forward-compat fields (`collect_history`,
    /// `checkpoint_after_step`, `checkpoint_store_set`) are read and
    /// dropped here — they will start flowing into the core engine
    /// once the matching `blazen-core/{telemetry,persist}` features
    /// are enabled in the Node binding's `Cargo.toml`.
    #[napi]
    pub fn build(&self) -> Result<JsWorkflow> {
        let mut guard = self.inner.lock().expect("poisoned");
        let state = guard.take().ok_or_else(JsWorkflowBuilder::consumed_error)?;

        let _ = state.collect_history;
        let _ = state.checkpoint_after_step;
        let _ = state.checkpoint_store_set;

        Ok(JsWorkflow {
            name: state.name,
            steps: state.steps,
            timeout_secs: state.timeout_secs,
            retry_config: state.retry_config,
            subworkflow_steps: Vec::new(),
            parallel_subworkflow_steps: Vec::new(),
            session_pause_policy: state.session_pause_policy,
            auto_publish_events: state.auto_publish_events,
        })
    }
}

#[cfg(not(target_os = "wasi"))]
#[napi]
#[allow(
    clippy::must_use_candidate,
    clippy::missing_errors_doc,
    clippy::missing_panics_doc,
    clippy::needless_pass_by_value
)]
impl JsWorkflowBuilder {
    /// Attach a checkpoint store to the workflow.
    ///
    /// Mirrors [`blazen_core::WorkflowBuilder::checkpoint_store`]. The
    /// underlying call is gated on the `persist` feature of
    /// `blazen-core`, which is **not** currently enabled in the Node
    /// binding's compilation. The flag is recorded on the builder for
    /// forward compatibility but does not yet flow into the core
    /// engine — pass a [`JsCheckpointStore`] (typically a concrete
    /// subclass like `RedbCheckpointStore` or `ValkeyCheckpointStore`)
    /// so the JS API is stable; the binding will start forwarding the
    /// store once the `blazen-core/persist` feature is enabled in
    /// `crates/blazen-node/Cargo.toml`.
    #[napi(js_name = "checkpointStore")]
    pub fn checkpoint_store(&self, _store: &JsCheckpointStore) -> Result<&Self> {
        let mut guard = self.inner.lock().expect("poisoned");
        let state = guard
            .as_mut()
            .ok_or_else(JsWorkflowBuilder::consumed_error)?;
        state.checkpoint_store_set = true;
        Ok(self)
    }
}

impl JsWorkflowBuilder {
    /// Shared constructor used by both [`Self::new`] and
    /// [`JsWorkflow::builder`].
    pub(crate) fn new_internal(name: String) -> Self {
        Self {
            inner: Mutex::new(Some(WorkflowBuilderState {
                name,
                steps: Vec::new(),
                timeout_secs: Some(300.0),
                retry_config: None,
                session_pause_policy: JsSessionPausePolicy::PickleOrError,
                // Mirror Rust default — Wave 7.
                auto_publish_events: true,
                collect_history: false,
                checkpoint_after_step: false,
                checkpoint_store_set: false,
            })),
        }
    }

    fn consumed_error() -> napi::Error {
        napi::Error::new(
            Status::GenericFailure,
            "WorkflowBuilder already consumed (build() was called)",
        )
    }
}

/// Run the workflow, threading through a parent session-ref registry
/// if one is currently installed.
///
/// This is the Phase 0.6 fix site for the sub-workflow session-ref
/// lifespan bug: if a parent workflow is already in flight (detected
/// via [`blazen_core::session_ref::current_session_registry`]), this
/// child run inherits the parent's `Arc<SessionRefRegistry>` via
/// [`blazen_core::Workflow::run_with_registry`]. Otherwise it takes the
/// normal `Workflow::run` path and gets a fresh registry.
async fn run_with_optional_parent_registry(
    workflow: &blazen_core::Workflow,
    input: serde_json::Value,
) -> Result<blazen_core::WorkflowHandler> {
    let handler =
        if let Some(parent_registry) = blazen_core::session_ref::current_session_registry() {
            workflow.run_with_registry(input, parent_registry).await
        } else {
            workflow.run(input).await
        };
    handler.map_err(workflow_error_to_napi)
}

/// Create a [`StepRegistration`](blazen_core::StepRegistration) from a JS step.
fn make_step_registration(step: &JsStepRegistration) -> blazen_core::StepRegistration {
    let accepts: Vec<&'static str> = step
        .event_types
        .iter()
        .map(|s| intern_event_type(s))
        .collect();

    // Arc clone is cheap -- the ThreadsafeFunction itself is shared.
    let handler_tsfn = Arc::clone(&step.handler);

    let handler: blazen_core::StepFn = Arc::new(
        move |event: Box<dyn AnyEvent>,
              ctx: blazen_core::Context|
              -> std::pin::Pin<
            Box<
                dyn std::future::Future<
                        Output = std::result::Result<
                            blazen_core::StepOutput,
                            blazen_core::WorkflowError,
                        >,
                    > + Send,
            >,
        > {
            let tsfn = Arc::clone(&handler_tsfn);

            Box::pin(async move {
                // Install the session-ref registry as a Tokio task_local so
                // that `current_session_registry()` returns `Some` inside the
                // JS step handler (mirrors the Python binding's wrapper).
                let registry = ctx.session_refs_arc().await;
                let js_ctx = JsContext::new(ctx);

                blazen_core::session_ref::with_session_registry(registry, async move {
                    // Convert the Rust event to a JS-friendly JSON value.
                    let js_event = any_event_to_js_value(&*event);

                    // Call the JavaScript handler function.
                    // ThreadsafeFunction::call_async returns a Future that resolves
                    // to the JS function's return value (serde_json::Value).
                    let result_value: serde_json::Value = tsfn
                        .call_async(FnArgs::from((js_event, js_ctx)))
                        .await
                        .map_err(|e: napi::Error| {
                            blazen_core::WorkflowError::Context(e.to_string())
                        })?
                        .await
                        .map_err(|e: napi::Error| {
                            blazen_core::WorkflowError::Context(e.to_string())
                        })?;

                    // Convert the JS return value back to a Rust event.
                    if result_value.is_null() {
                        return Ok(blazen_core::StepOutput::None);
                    }

                    // Check if it's an array (multiple events).
                    if let serde_json::Value::Array(arr) = &result_value {
                        let events: Vec<Box<dyn AnyEvent>> =
                            arr.iter().map(js_value_to_any_event).collect();
                        return Ok(blazen_core::StepOutput::Multiple(events));
                    }

                    // Single event.
                    let event = js_value_to_any_event(&result_value);
                    Ok(blazen_core::StepOutput::Single(event))
                })
                .await
            })
        },
    );

    let mut reg = blazen_core::StepRegistration::new(
        step.name.clone(),
        accepts,
        vec![], // JS steps don't declare emits statically.
        handler,
        0, // unlimited concurrency
    );
    if let Some(secs) = step.timeout_secs {
        reg = reg.with_timeout(Duration::from_secs_f64(secs));
    }
    if let Some(cfg) = step.retry_config.as_ref() {
        reg = reg.with_retry_config(cfg.clone());
    }
    reg
}

/// Convert a result event to a [`JsWorkflowResult`].
fn make_result(result: &blazen_core::WorkflowResult) -> JsWorkflowResult {
    let event = &*result.event;
    let event_type = event.event_type_id().to_owned();
    let json = event.to_json();

    // For StopEvent, extract the result field.
    let data = if event_type == "blazen::StopEvent" {
        json.get("result")
            .cloned()
            .unwrap_or(serde_json::Value::Null)
    } else {
        json
    };

    JsWorkflowResult {
        event_type,
        data,
        usage_total: result.usage_total.clone().into(),
        cost_total_usd: result.cost_total_usd,
    }
}
