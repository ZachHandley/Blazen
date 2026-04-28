use napi::bindgen_prelude::*;
use napi_derive::napi;

use crate::error::pipeline_error_to_napi;
use crate::pipeline::handler::JsPipelineHandler;
use crate::pipeline::snapshot::JsPipelineSnapshot;

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
