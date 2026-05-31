//! Standalone wrapper class for [`blazen_core::SubPipelineStep`].
//!
//! Mirrors [`JsSubWorkflowStep`](super::subworkflow_step::JsSubWorkflowStep)
//! but embeds a `Pipeline` (any [`blazen_core::SubExecutable`]) as a step
//! inside a parent `Workflow`. The parent workflow's event loop spawns the
//! child pipeline via `Pipeline.start()`, converts the parent event to JSON
//! for the child's input, and wraps the child's `PipelineResult.finalOutput`
//! into a `DynamicEvent` named `"<stepName>::output"` for the parent.
//!
//! ```javascript
//! const child = new PipelineBuilder("enrich").stage(stage).build();
//! const step = new SubPipelineStep("enrich", ["blazen::StartEvent"], ["enrich::output"], child);
//! const parent = new Workflow("parent");
//! parent.addSubpipelineStepObj(step);
//! ```

use std::sync::Arc;

use napi::bindgen_prelude::Result;
use napi_derive::napi;

use crate::core::sub_executable::JsSubExecutable;
use crate::generated::JsRetryConfig;
use crate::pipeline::pipeline::JsPipeline;

// ---------------------------------------------------------------------------
// JsSubPipelineStep
// ---------------------------------------------------------------------------

/// A workflow step that delegates to a `Pipeline`.
///
/// The child pipeline is cloned (from a built [`Pipeline`](JsPipeline)) at
/// construction time and stored as an `Arc<dyn SubExecutable>` so this step
/// instance can be reused across multiple parent workflows.
#[napi(js_name = "SubPipelineStep")]
pub struct JsSubPipelineStep {
    pub(crate) name: String,
    pub(crate) accepts: Vec<String>,
    pub(crate) emits: Vec<String>,
    pub(crate) executable: Arc<dyn blazen_core::SubExecutable>,
    pub(crate) timeout_secs: Option<f64>,
    pub(crate) retry_config: Option<blazen_llm::retry::RetryConfig>,
}

#[napi]
#[allow(
    clippy::must_use_candidate,
    clippy::missing_errors_doc,
    clippy::needless_pass_by_value
)]
impl JsSubPipelineStep {
    /// Create a sub-pipeline step.
    ///
    /// `name` / `accepts` / `emits` describe routing. `inner` is the child
    /// pipeline whose stages are run for each parent dispatch. The inner
    /// pipeline is cloned at construction time, so `inner` must not have
    /// been consumed (by `start`/`run`/`resume`) yet and this step instance
    /// can be reused across builders.
    #[napi(constructor)]
    pub fn new(
        name: String,
        accepts: Vec<String>,
        emits: Vec<String>,
        inner: &JsPipeline,
        timeout_secs: Option<f64>,
        retry_config: Option<JsRetryConfig>,
    ) -> Result<Self> {
        let pipeline = inner.clone_inner()?;
        let executable: Arc<dyn blazen_core::SubExecutable> = Arc::new(pipeline);
        Ok(Self {
            name,
            accepts,
            emits,
            executable,
            timeout_secs,
            retry_config: retry_config.map(Into::into),
        })
    }

    /// Create a sub-pipeline step from any [`SubExecutable`](JsSubExecutable)
    /// child runner.
    ///
    /// Unlike [`new`](Self::new) (which embeds a built `Pipeline`), this
    /// accepts a user-defined `SubExecutable` subclass instance, letting an
    /// arbitrary JS-implemented child runner be embedded inside a parent
    /// `Workflow`. The executable handle is cloned, so the instance can be
    /// reused across builders.
    #[napi(factory, js_name = "fromExecutable")]
    pub fn from_executable(
        name: String,
        accepts: Vec<String>,
        emits: Vec<String>,
        executable: &JsSubExecutable,
        timeout_secs: Option<f64>,
        retry_config: Option<JsRetryConfig>,
    ) -> Result<Self> {
        Ok(Self {
            name,
            accepts,
            emits,
            executable: executable.executable(),
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
