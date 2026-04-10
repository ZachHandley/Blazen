//! JavaScript workflow builder and runner.
//!
//! Provides [`JsWorkflow`] which lets TypeScript/JavaScript users define
//! workflows with step handlers as async functions.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use blazen_core::SessionRefDeserializerFn;
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
    session_pause_policy: JsSessionPausePolicy,
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
            session_pause_policy: JsSessionPausePolicy::PickleOrError,
        }
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

        Ok(make_result(&*result.event))
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
        let stream_handle = tokio::spawn(async move {
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

        Ok(make_result(&*result.event))
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
    fn build_workflow(&self) -> Result<blazen_core::Workflow> {
        let mut builder = blazen_core::WorkflowBuilder::new(self.name.clone());

        if let Some(secs) = self.timeout_secs {
            builder = builder.timeout(Duration::from_secs_f64(secs));
        } else {
            builder = builder.no_timeout();
        }

        for step in &self.steps {
            let registration = make_step_registration(step);
            builder = builder.step(registration);
        }

        builder = builder.session_pause_policy(self.session_pause_policy.into());

        builder.build().map_err(workflow_error_to_napi)
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

    blazen_core::StepRegistration {
        name: step.name.clone(),
        accepts,
        emits: vec![], // JS steps don't declare emits statically.
        handler,
        max_concurrency: 0,
    }
}

/// Convert a result event to a [`JsWorkflowResult`].
fn make_result(event: &dyn AnyEvent) -> JsWorkflowResult {
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

    JsWorkflowResult { event_type, data }
}
