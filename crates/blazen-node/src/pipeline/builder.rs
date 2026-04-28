use std::sync::Mutex;
use std::time::Duration;

use napi::bindgen_prelude::*;
use napi_derive::napi;

use crate::error::pipeline_error_to_napi;
use crate::pipeline::pipeline::JsPipeline;
use crate::pipeline::stage::{JsParallelStage, JsStage};

/// Fluent builder for constructing a Pipeline.
#[napi(js_name = "PipelineBuilder")]
pub struct JsPipelineBuilder {
    inner: Mutex<Option<blazen_pipeline::PipelineBuilder>>,
}

#[napi]
#[allow(
    clippy::must_use_candidate,
    clippy::missing_errors_doc,
    clippy::missing_panics_doc,
    clippy::needless_pass_by_value
)]
impl JsPipelineBuilder {
    #[napi(constructor)]
    pub fn new(name: String) -> Self {
        Self {
            inner: Mutex::new(Some(blazen_pipeline::PipelineBuilder::new(name))),
        }
    }

    /// Append a sequential Stage to the pipeline.
    #[napi]
    pub fn stage(&self, stage: &JsStage) -> Result<&Self> {
        let core_stage = stage.take()?;
        let mut guard = self.inner.lock().expect("poisoned");
        let builder = guard.take().ok_or_else(|| {
            napi::Error::new(
                napi::Status::GenericFailure,
                "PipelineBuilder already consumed (build() was called)",
            )
        })?;
        *guard = Some(builder.stage(core_stage));
        Ok(self)
    }

    /// Append a `ParallelStage` to the pipeline.
    #[napi]
    pub fn parallel(&self, parallel: &JsParallelStage) -> Result<&Self> {
        let core_parallel = parallel.take()?;
        let mut guard = self.inner.lock().expect("poisoned");
        let builder = guard.take().ok_or_else(|| {
            napi::Error::new(
                napi::Status::GenericFailure,
                "PipelineBuilder already consumed (build() was called)",
            )
        })?;
        *guard = Some(builder.parallel(core_parallel));
        Ok(self)
    }

    /// Set a per-stage timeout in seconds. Each stage's workflow gets this duration.
    #[napi(js_name = "timeoutPerStage")]
    pub fn timeout_per_stage(&self, seconds: f64) -> Result<&Self> {
        let mut guard = self.inner.lock().expect("poisoned");
        let builder = guard.take().ok_or_else(|| {
            napi::Error::new(
                napi::Status::GenericFailure,
                "PipelineBuilder already consumed (build() was called)",
            )
        })?;
        *guard = Some(builder.timeout_per_stage(Duration::from_secs_f64(seconds)));
        Ok(self)
    }

    /// Validate and build the pipeline. Throws if no stages or duplicate names.
    #[napi]
    pub fn build(&self) -> Result<JsPipeline> {
        let mut guard = self.inner.lock().expect("poisoned");
        let builder = guard.take().ok_or_else(|| {
            napi::Error::new(
                napi::Status::GenericFailure,
                "PipelineBuilder already consumed",
            )
        })?;
        let pipeline = builder.build().map_err(pipeline_error_to_napi)?;
        Ok(JsPipeline::from_inner(pipeline))
    }
}
