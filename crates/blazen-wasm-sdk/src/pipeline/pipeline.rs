//! WASM binding for [`blazen_pipeline::Pipeline`].
//!
//! A validated, ready-to-run pipeline. Returned by
//! [`WasmPipelineBuilder::build`](super::builder::WasmPipelineBuilder::build)
//! and consumed by [`Self::start`] or [`Self::resume`], each of which
//! returns a [`WasmPipelineHandler`].

use std::sync::Mutex;

use wasm_bindgen::prelude::*;

use crate::pipeline::error::pipeline_err;
use crate::pipeline::handler::WasmPipelineHandler;
use crate::pipeline::snapshot::{WasmPipelineResult, WasmPipelineSnapshot};

/// A validated, ready-to-run pipeline.
///
/// Single-shot: calling `start()` or `resume()` once consumes the inner
/// pipeline, so a second call returns an error.
#[wasm_bindgen(js_name = "Pipeline")]
pub struct WasmPipeline {
    /// Inner Rust pipeline. Wrapped in `Mutex<Option<...>>` because
    /// `wasm-bindgen` only gives us `&self`/`&mut self` access, but
    /// `Pipeline::start` / `Pipeline::resume` consume `self`. We swap the
    /// pipeline out of the slot via `Option::take`.
    inner: Mutex<Option<blazen_pipeline::Pipeline>>,
}

impl WasmPipeline {
    /// Wrap a built [`blazen_pipeline::Pipeline`].
    #[must_use]
    pub(crate) fn from_inner(pipeline: blazen_pipeline::Pipeline) -> Self {
        Self {
            inner: Mutex::new(Some(pipeline)),
        }
    }

    /// Clone the inner [`blazen_pipeline::Pipeline`] without consuming this
    /// handle.
    ///
    /// Used to embed a built pipeline as a child runner (e.g. via
    /// [`SubPipelineStep`](crate::sub_executable::WasmSubPipelineStep)) while
    /// leaving the original handle runnable.
    ///
    /// # Errors
    ///
    /// Returns a `JsValue` error if the pipeline has already been consumed by
    /// a `start()` / `resume()` call.
    pub(crate) fn clone_inner(&self) -> Result<blazen_pipeline::Pipeline, JsValue> {
        let guard = self.inner.lock().expect("poisoned");
        guard
            .as_ref()
            .cloned()
            .ok_or_else(|| JsValue::from_str("Pipeline already consumed; cannot reuse as a child"))
    }
}

#[wasm_bindgen(js_class = "Pipeline")]
impl WasmPipeline {
    /// Execute the pipeline with the given input.
    ///
    /// Consumes the pipeline; calling `start()` or `resume()` a second
    /// time errors. Returns a [`WasmPipelineHandler`] for awaiting the
    /// result, streaming events, pausing, or aborting.
    ///
    /// # Errors
    ///
    /// Returns a `JsValue` error if:
    /// - the pipeline has already been consumed, or
    /// - the input cannot be deserialised as JSON.
    #[wasm_bindgen]
    // `async` is part of the JS-facing contract (returns a Promise on the
    // TS side) even though the current body is synchronous.
    #[allow(clippy::unused_async)]
    pub async fn start(&self, input: JsValue) -> Result<WasmPipelineHandler, JsValue> {
        let pipeline = {
            let mut guard = self.inner.lock().expect("poisoned");
            guard.take().ok_or_else(|| {
                JsValue::from_str(
                    "Pipeline already consumed (start() or resume() was already called)",
                )
            })?
        };

        let input_json: serde_json::Value = if input.is_undefined() || input.is_null() {
            serde_json::Value::Null
        } else {
            serde_wasm_bindgen::from_value(input)
                .map_err(|e| JsValue::from_str(&format!("input must be JSON-serializable: {e}")))?
        };

        let handler = pipeline.start(input_json);
        Ok(WasmPipelineHandler::new(handler))
    }

    /// Build and dispatch the pipeline, returning the live
    /// [`WasmPipelineHandler`] without awaiting the final result.
    ///
    /// Functionally identical to [`start`](Self::start); exposed under the
    /// JS name `runWithHandler` for naming parity with
    /// [`Workflow.runWithHandler`](crate::workflow). Use either
    /// interchangeably.
    ///
    /// # Errors
    ///
    /// Returns a `JsValue` error under the same conditions as
    /// [`start`](Self::start): the pipeline was already consumed, or the
    /// input is not JSON-serialisable.
    #[wasm_bindgen(js_name = "runWithHandler")]
    pub async fn run_with_handler(&self, input: JsValue) -> Result<WasmPipelineHandler, JsValue> {
        self.start(input).await
    }

    /// Run the pipeline to completion and resolve with the final result.
    ///
    /// Result shorthand mirroring [`Pipeline::run`](blazen_pipeline::Pipeline::run)
    /// and [`Workflow.run`](crate::workflow) — equivalent to
    /// `start(input).result()`. For streaming, pausing, snapshotting, or
    /// human-in-the-loop input, use [`start`](Self::start) /
    /// [`runWithHandler`](Self::run_with_handler) and drive the returned
    /// [`WasmPipelineHandler`].
    ///
    /// Consumes the pipeline; a second `run`/`start`/`resume` call errors.
    ///
    /// # Errors
    ///
    /// Returns a `JsValue` error if:
    /// - the pipeline has already been consumed,
    /// - the input cannot be deserialised as JSON, or
    /// - the pipeline ran to completion with an error
    ///   ([`PipelineError`](blazen_pipeline::PipelineError)).
    #[wasm_bindgen]
    pub async fn run(&self, input: JsValue) -> Result<WasmPipelineResult, JsValue> {
        let pipeline = {
            let mut guard = self.inner.lock().expect("poisoned");
            guard.take().ok_or_else(|| {
                JsValue::from_str(
                    "Pipeline already consumed (start() or resume() was already called)",
                )
            })?
        };

        let input_json: serde_json::Value = if input.is_undefined() || input.is_null() {
            serde_json::Value::Null
        } else {
            serde_wasm_bindgen::from_value(input)
                .map_err(|e| JsValue::from_str(&format!("input must be JSON-serializable: {e}")))?
        };

        // Dispatch, then yield before parking on the result channel so the
        // spawn_local'd executor gets an event-loop turn on workerd. (We
        // can't call `Pipeline::run` directly — it couples `start().result()`
        // with no yield in between, which would hang on workerd. See
        // `crate::handler::yield_to_js`.)
        let handler = pipeline.start(input_json);
        crate::handler::yield_to_js().await;
        let result = handler.result().await.map_err(pipeline_err)?;
        Ok(WasmPipelineResult::from_inner(result))
    }

    /// Resume the pipeline from a previously captured snapshot.
    ///
    /// Consumes the pipeline.
    ///
    /// # Errors
    ///
    /// Returns a `JsValue` error if:
    /// - the pipeline has already been consumed, or
    /// - the snapshot's stage layout doesn't match the pipeline (name
    ///   mismatch, stage-index out of range, completed-stage drift).
    #[wasm_bindgen]
    // `async` is part of the JS-facing contract (returns a Promise on the
    // TS side) even though the current body is synchronous.
    #[allow(clippy::unused_async)]
    pub async fn resume(
        &self,
        snapshot: &WasmPipelineSnapshot,
    ) -> Result<WasmPipelineHandler, JsValue> {
        let pipeline = {
            let mut guard = self.inner.lock().expect("poisoned");
            guard.take().ok_or_else(|| {
                JsValue::from_str(
                    "Pipeline already consumed (start() or resume() was already called)",
                )
            })?
        };

        let snap = snapshot.inner().clone();
        let handler = pipeline.resume(snap).map_err(pipeline_err)?;
        Ok(WasmPipelineHandler::new(handler))
    }
}
