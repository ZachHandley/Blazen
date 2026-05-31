use napi::bindgen_prelude::*;
use napi_derive::napi;

use crate::error::pipeline_error_to_napi;
use crate::generated::JsRetryConfig;
use crate::pipeline::handler::JsPipelineHandler;
use crate::pipeline::snapshot::{JsPipelineResult, JsPipelineSnapshot};

/// A validated, ready-to-run pipeline.
#[napi(js_name = "Pipeline")]
pub struct JsPipeline {
    /// Inner Rust pipeline. Wrapped in `Mutex<Option<...>>` because napi
    /// only gives `&self`/`&mut self` access but `Pipeline::start`/`resume`
    /// consume `self`. We swap the pipeline out via `Option::take`.
    inner: std::sync::Mutex<Option<blazen_pipeline::Pipeline>>,
}

impl JsPipeline {
    pub(crate) fn from_inner(pipeline: blazen_pipeline::Pipeline) -> Self {
        Self {
            inner: std::sync::Mutex::new(Some(pipeline)),
        }
    }

    /// Clone the inner pipeline without consuming this `JsPipeline`.
    ///
    /// Used to embed a built pipeline as a [`crate::workflow::subpipeline_step::JsSubPipelineStep`]
    /// inside a parent workflow. Errors if the pipeline has already been
    /// consumed by `start`/`run`/`resume`.
    pub(crate) fn clone_inner(&self) -> Result<blazen_pipeline::Pipeline> {
        let guard = self.inner.lock().expect("poisoned");
        guard.as_ref().cloned().ok_or_else(|| {
            napi::Error::new(
                napi::Status::GenericFailure,
                "Pipeline already consumed (start()/run()/resume() was already called) -- \
                 construct the SubPipelineStep before running the pipeline",
            )
        })
    }
}

#[napi]
#[allow(
    clippy::must_use_candidate,
    clippy::missing_errors_doc,
    clippy::missing_panics_doc,
    clippy::needless_pass_by_value,
    clippy::unused_async
)]
impl JsPipeline {
    /// Execute the pipeline with the given input.
    /// Consumes the pipeline -- calling start/resume a second time errors.
    #[napi]
    pub async fn start(&self, input: serde_json::Value) -> Result<JsPipelineHandler> {
        let pipeline = {
            let mut guard = self.inner.lock().expect("poisoned");
            guard.take().ok_or_else(|| {
                napi::Error::new(
                    napi::Status::GenericFailure,
                    "Pipeline already consumed (start() or resume() was already called)",
                )
            })?
        };
        let handler = pipeline.start(input);
        Ok(JsPipelineHandler::new(handler))
    }

    /// Execute the pipeline and await its final result in one call.
    ///
    /// This is the result-shorthand mirror of [`crate::workflow::JsWorkflow::run`]:
    /// equivalent to `(await pipeline.start(input)).result()`, but without
    /// exposing the intermediate handler. Consumes the pipeline -- calling
    /// `run`/`start`/`resume` a second time errors.
    #[napi]
    pub async fn run(&self, input: serde_json::Value) -> Result<JsPipelineResult> {
        let pipeline = {
            let mut guard = self.inner.lock().expect("poisoned");
            guard.take().ok_or_else(|| {
                napi::Error::new(
                    napi::Status::GenericFailure,
                    "Pipeline already consumed (start() or resume() was already called)",
                )
            })?
        };
        let result = pipeline.run(input).await.map_err(pipeline_error_to_napi)?;
        Ok(JsPipelineResult::from_inner(result))
    }

    /// Inspect the pipeline-level default retry configuration, if any.
    /// Mirrors [`blazen_pipeline::Pipeline::retry_config`] (Wave 2).
    /// Returns `null` after the pipeline has been consumed.
    #[napi(js_name = "retryConfig")]
    pub fn retry_config(&self) -> Option<JsRetryConfig> {
        let guard = self.inner.lock().expect("poisoned");
        let pipeline = guard.as_ref()?;
        pipeline
            .retry_config()
            .map(|arc| arc.as_ref().clone().into())
    }

    /// Resume the pipeline from a previously captured snapshot.
    /// Consumes the pipeline.
    #[napi]
    pub async fn resume(&self, snapshot: &JsPipelineSnapshot) -> Result<JsPipelineHandler> {
        let pipeline = {
            let mut guard = self.inner.lock().expect("poisoned");
            guard.take().ok_or_else(|| {
                napi::Error::new(
                    napi::Status::GenericFailure,
                    "Pipeline already consumed (start() or resume() was already called)",
                )
            })?
        };
        let snap = snapshot.inner.clone();
        let handler = pipeline.resume(snap).map_err(pipeline_error_to_napi)?;
        Ok(JsPipelineHandler::new(handler))
    }
}
