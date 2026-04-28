//! Fluent builder for constructing a [`WasmPipeline`].
//!
//! Mirrors the napi/PyO3 bindings: each builder method takes `&self` and
//! mutates the inner state through a [`Mutex<Option<...>>`] so JS callers
//! can chain calls without re-binding the builder. `build()` consumes the
//! builder and produces a [`WasmPipeline`] ready to `start()` or
//! `resume()`.

use std::sync::Mutex;
use std::time::Duration;

use wasm_bindgen::prelude::*;

use crate::pipeline::error::pipeline_err;
use crate::pipeline::pipeline::WasmPipeline;
use crate::pipeline::stage::{WasmParallelStage, WasmStage};

/// Fluent builder for constructing a [`WasmPipeline`].
///
/// ```typescript
/// const pipeline = new PipelineBuilder("my-pipeline")
///   .stage(new Stage("ingest", wfIngest))
///   .parallel(new ParallelStage("fan-out", [stageA, stageB]))
///   .timeoutPerStage(30)
///   .build();
/// const handler = await pipeline.start({ data: "hello" });
/// const result = await handler.result();
/// ```
#[wasm_bindgen(js_name = "PipelineBuilder")]
pub struct WasmPipelineBuilder {
    /// Inner builder. Wrapped in `Mutex<Option<...>>` because every
    /// chainable method takes `&self` (so the JS-side `this` chain works)
    /// while the underlying `blazen_pipeline::PipelineBuilder` methods
    /// consume `self`.
    inner: Mutex<Option<blazen_pipeline::PipelineBuilder>>,
}

#[wasm_bindgen(js_class = "PipelineBuilder")]
impl WasmPipelineBuilder {
    /// Create a new builder with the given pipeline name.
    #[wasm_bindgen(constructor)]
    #[must_use]
    pub fn new(name: String) -> WasmPipelineBuilder {
        Self {
            inner: Mutex::new(Some(blazen_pipeline::PipelineBuilder::new(name))),
        }
    }

    /// Append a sequential [`Stage`](super::stage::WasmStage) to the
    /// pipeline.
    ///
    /// Consumes the stage's underlying [`blazen_pipeline::Stage`] — the
    /// same `Stage` instance cannot be added to two pipelines.
    ///
    /// # Errors
    ///
    /// Returns a `JsValue` error if the builder has already been consumed
    /// (`build()` was called) or the stage was already consumed by a
    /// previous `stage()` / `parallel()` call.
    #[wasm_bindgen]
    pub fn stage(&self, stage: &WasmStage) -> Result<(), JsValue> {
        let core_stage = stage.take()?;
        let mut guard = self.inner.lock().expect("poisoned");
        let builder = guard.take().ok_or_else(|| {
            JsValue::from_str("PipelineBuilder already consumed (build() was called)")
        })?;
        *guard = Some(builder.stage(core_stage));
        Ok(())
    }

    /// Append a [`ParallelStage`](super::stage::WasmParallelStage) to the
    /// pipeline.
    ///
    /// Consumes the parallel stage's underlying
    /// [`blazen_pipeline::ParallelStage`].
    ///
    /// # Errors
    ///
    /// Returns a `JsValue` error if the builder has already been consumed
    /// or the parallel stage was already consumed.
    #[wasm_bindgen]
    pub fn parallel(&self, parallel: &WasmParallelStage) -> Result<(), JsValue> {
        let core_parallel = parallel.take()?;
        let mut guard = self.inner.lock().expect("poisoned");
        let builder = guard.take().ok_or_else(|| {
            JsValue::from_str("PipelineBuilder already consumed (build() was called)")
        })?;
        *guard = Some(builder.parallel(core_parallel));
        Ok(())
    }

    /// Set a per-stage timeout in seconds. Each stage's workflow is
    /// constrained by this duration; exceeding it surfaces as a stage
    /// failure with [`blazen_core::WorkflowError::Timeout`].
    ///
    /// # Errors
    ///
    /// Returns a `JsValue` error if the builder has already been
    /// consumed.
    #[wasm_bindgen(js_name = "timeoutPerStage")]
    pub fn timeout_per_stage(&self, seconds: f64) -> Result<(), JsValue> {
        let mut guard = self.inner.lock().expect("poisoned");
        let builder = guard.take().ok_or_else(|| {
            JsValue::from_str("PipelineBuilder already consumed (build() was called)")
        })?;
        *guard = Some(builder.timeout_per_stage(Duration::from_secs_f64(seconds)));
        Ok(())
    }

    /// Validate and build the pipeline. Throws if no stages were added or
    /// the stage names are not unique.
    ///
    /// Consumes the builder; subsequent method calls error.
    ///
    /// # Errors
    ///
    /// Returns a `JsValue` error if the builder has already been consumed
    /// or the underlying [`blazen_pipeline::PipelineBuilder::build`] call
    /// fails (e.g. zero stages, duplicate stage names).
    #[wasm_bindgen]
    pub fn build(&self) -> Result<WasmPipeline, JsValue> {
        let mut guard = self.inner.lock().expect("poisoned");
        let builder = guard
            .take()
            .ok_or_else(|| JsValue::from_str("PipelineBuilder already consumed"))?;
        let pipeline = builder.build().map_err(pipeline_err)?;
        Ok(WasmPipeline::from_inner(pipeline))
    }
}
