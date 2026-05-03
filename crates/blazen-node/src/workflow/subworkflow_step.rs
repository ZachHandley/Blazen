//! Standalone wrapper classes for [`blazen_core::SubWorkflowStep`] and
//! [`blazen_core::ParallelSubWorkflowsStep`].
//!
//! The Node binding's existing `Workflow.addSubworkflowStep` /
//! `addParallelSubworkflows` methods take plain positional arguments
//! (name, accepts, emits, inner, …). These wrapper classes expose the
//! Rust types as first-class JS classes so callers can construct a step
//! once and pass it to multiple builders. Use them via the new
//! `Workflow.addSubworkflowStepObj` / `addParallelSubworkflowsObj`
//! methods on [`JsWorkflow`](super::workflow::JsWorkflow).

use std::sync::Arc;

use napi::bindgen_prelude::Result;
use napi_derive::napi;

use crate::generated::JsRetryConfig;
use crate::pipeline::stage::JsJoinStrategy;

use super::workflow::JsWorkflow;

// ---------------------------------------------------------------------------
// JsSubWorkflowStep
// ---------------------------------------------------------------------------

/// A workflow step that delegates to another `Workflow`.
///
/// The parent workflow's event loop spawns the child via `Workflow.run()`,
/// converts the parent event to JSON for the child's input, and wraps the
/// child's terminal `StopEvent.result` into a `DynamicEvent` named
/// `"<stepName>::output"` for the parent.
///
/// ```javascript
/// const child = new Workflow("enrich");
/// child.addStep("enrich", ["blazen::StartEvent"], async (ev) => ({ type: "blazen::StopEvent", result: { ok: true } }));
/// const step = new SubWorkflowStep("enrich", ["blazen::StartEvent"], ["enrich::output"], child);
/// const parent = new Workflow("parent");
/// parent.addSubworkflowStepObj(step);
/// ```
#[napi(js_name = "SubWorkflowStep")]
pub struct JsSubWorkflowStep {
    pub(crate) name: String,
    pub(crate) accepts: Vec<String>,
    pub(crate) emits: Vec<String>,
    pub(crate) inner_workflow: Arc<blazen_core::Workflow>,
    pub(crate) timeout_secs: Option<f64>,
    pub(crate) retry_config: Option<blazen_llm::retry::RetryConfig>,
}

#[napi]
#[allow(
    clippy::must_use_candidate,
    clippy::missing_errors_doc,
    clippy::needless_pass_by_value
)]
impl JsSubWorkflowStep {
    /// Create a sub-workflow step.
    ///
    /// `name` / `accepts` / `emits` describe routing. `inner` is the child
    /// workflow whose event loop is spawned for each parent dispatch. The
    /// inner workflow is cloned (and built) at construction time so this
    /// step instance can be reused across builders.
    #[napi(constructor)]
    pub fn new(
        name: String,
        accepts: Vec<String>,
        emits: Vec<String>,
        inner: &JsWorkflow,
        timeout_secs: Option<f64>,
        retry_config: Option<JsRetryConfig>,
    ) -> Result<Self> {
        let inner_workflow = Arc::new(inner.build_workflow()?);
        Ok(Self {
            name,
            accepts,
            emits,
            inner_workflow,
            timeout_secs,
            retry_config: retry_config.map(Into::into),
        })
    }

    /// The step name.
    #[napi(getter)]
    pub fn name(&self) -> String {
        self.name.clone()
    }

    /// Event type identifiers this step accepts.
    #[napi(getter)]
    pub fn accepts(&self) -> Vec<String> {
        self.accepts.clone()
    }

    /// Event type identifiers this step may emit.
    #[napi(getter)]
    pub fn emits(&self) -> Vec<String> {
        self.emits.clone()
    }
}

// ---------------------------------------------------------------------------
// JsParallelSubWorkflowsStep
// ---------------------------------------------------------------------------

/// Fan out into multiple parallel sub-workflow branches.
///
/// Each branch is a `SubWorkflowStep` that runs concurrently. The
/// `joinStrategy` controls whether the parent waits for all branches
/// (`JoinStrategy.WaitAll`) or only the first to complete
/// (`JoinStrategy.FirstCompletes`).
#[napi(js_name = "ParallelSubWorkflowsStep")]
pub struct JsParallelSubWorkflowsStep {
    pub(crate) name: String,
    pub(crate) accepts: Vec<String>,
    pub(crate) emits: Vec<String>,
    pub(crate) branches: Vec<JsBranchSpec>,
    pub(crate) join_strategy: JsJoinStrategy,
}

/// Internal-only: a per-branch snapshot captured at constructor time so we
/// don't need to retain raw `&JsSubWorkflowStep` references (napi-rs doesn't
/// hold them across the JS↔Rust boundary).
#[derive(Clone)]
pub(crate) struct JsBranchSpec {
    pub name: String,
    pub accepts: Vec<String>,
    pub emits: Vec<String>,
    pub inner_workflow: Arc<blazen_core::Workflow>,
    pub timeout_secs: Option<f64>,
    pub retry_config: Option<blazen_llm::retry::RetryConfig>,
}

#[napi]
#[allow(
    clippy::must_use_candidate,
    clippy::missing_errors_doc,
    clippy::needless_pass_by_value
)]
impl JsParallelSubWorkflowsStep {
    /// Create a parallel sub-workflow fan-out step.
    ///
    /// `branches` is an array of already-constructed `SubWorkflowStep`
    /// instances. The branches' inner workflows are captured by reference
    /// so the parent step keeps a stable view even if the originals are
    /// dropped from JS.
    ///
    /// `joinStrategy` defaults to `JoinStrategy.WaitAll`.
    #[napi(constructor)]
    pub fn new(
        name: String,
        accepts: Vec<String>,
        emits: Vec<String>,
        branches: Vec<&JsSubWorkflowStep>,
        join_strategy: Option<JsJoinStrategy>,
    ) -> Self {
        let branch_specs: Vec<JsBranchSpec> = branches
            .iter()
            .map(|b| JsBranchSpec {
                name: b.name.clone(),
                accepts: b.accepts.clone(),
                emits: b.emits.clone(),
                inner_workflow: Arc::clone(&b.inner_workflow),
                timeout_secs: b.timeout_secs,
                retry_config: b.retry_config.clone(),
            })
            .collect();
        Self {
            name,
            accepts,
            emits,
            branches: branch_specs,
            join_strategy: join_strategy.unwrap_or(JsJoinStrategy::WaitAll),
        }
    }

    /// The step name.
    #[napi(getter)]
    pub fn name(&self) -> String {
        self.name.clone()
    }

    /// Event type identifiers this step accepts.
    #[napi(getter)]
    pub fn accepts(&self) -> Vec<String> {
        self.accepts.clone()
    }

    /// Event type identifiers this step may emit.
    #[napi(getter)]
    pub fn emits(&self) -> Vec<String> {
        self.emits.clone()
    }

    /// The join strategy used to combine branch results.
    #[napi(getter, js_name = "joinStrategy")]
    pub fn join_strategy(&self) -> JsJoinStrategy {
        self.join_strategy
    }
}
