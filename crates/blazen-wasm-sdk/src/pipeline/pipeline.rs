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
use crate::pipeline::snapshot::WasmPipelineSnapshot;

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
