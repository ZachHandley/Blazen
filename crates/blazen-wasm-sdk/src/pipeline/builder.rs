//! Fluent builder for constructing a [`WasmPipeline`].
//!
//! Mirrors the napi/PyO3 bindings: each builder method takes `&self` and
//! mutates the inner state through a [`Mutex<Option<...>>`] so JS callers
//! can chain calls without re-binding the builder. `build()` consumes the
//! builder and produces a [`WasmPipeline`] ready to `start()` or
//! `resume()`.

use std::future::Future;
use std::pin::Pin;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use wasm_bindgen::prelude::*;

use blazen_pipeline::{PersistFn, PersistJsonFn, PipelineError, PipelineSnapshot};

use crate::pipeline::error::pipeline_err;
use crate::pipeline::pipeline::WasmPipeline;
use crate::pipeline::stage::{WasmParallelStage, WasmStage};

/// Newtype wrapping a [`js_sys::Function`] persist callback so the
/// `PersistFn` / `PersistJsonFn` `Send + Sync` bound is satisfied.
///
/// SAFETY: `wasm32-unknown-unknown` is single-threaded; the JS function
/// never crosses a thread boundary. The unsafe `Send`/`Sync` impls are
/// vacuously safe.
struct JsClosure(js_sys::Function);

// SAFETY: wasm32 is single-threaded; nothing crosses threads.
unsafe impl Send for JsClosure {}
// SAFETY: wasm32 is single-threaded; nothing crosses threads.
unsafe impl Sync for JsClosure {}

/// Wrap a non-`Send` future so it can be returned where `Send` is required.
///
/// `wasm_bindgen_futures::JsFuture` is not `Send`, but `PersistFn` and
/// `PersistJsonFn` both require `Pin<Box<dyn Future<Output = ...> + Send>>`.
/// On `wasm32-unknown-unknown` the runtime is single-threaded, so the
/// future never actually crosses threads; this wrapper makes the type
/// system accept the bound. Mirrors the `SendFuture` pattern already used
/// in `workflow.rs`.
struct SendFuture<F>(F);

// SAFETY: wasm32 is single-threaded; nothing crosses threads.
unsafe impl<F> Send for SendFuture<F> {}

impl<F: Future> Future for SendFuture<F> {
    type Output = F::Output;

    fn poll(
        self: Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Self::Output> {
        // SAFETY: structural projection through the wrapper. We never
        // move `F` out of `self`.
        let inner = unsafe { self.map_unchecked_mut(|s| &mut s.0) };
        inner.poll(cx)
    }
}

/// Fluent builder for constructing a [`WasmPipeline`].
///
/// ```typescript
/// const pipeline = new PipelineBuilder("my-pipeline")
///   .stage(new Stage("ingest", wfIngest))
///   .parallel(new ParallelStage("fan-out", [stageA, stageB]))
///   .timeoutPerStage(30)
///   .onPersist(async (snapshot) => { await save(snapshot); })
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
    /// Persist callback (typed `PipelineSnapshot`). Drained into the
    /// underlying builder by `build()`.
    persist_fn: Mutex<Option<PersistFn>>,
    /// Persist callback (JSON-serialized snapshot). Drained into the
    /// underlying builder by `build()`.
    persist_json_fn: Mutex<Option<PersistJsonFn>>,
}

#[wasm_bindgen(js_class = "PipelineBuilder")]
impl WasmPipelineBuilder {
    /// Create a new builder with the given pipeline name.
    #[wasm_bindgen(constructor)]
    #[must_use]
    pub fn new(name: String) -> WasmPipelineBuilder {
        Self {
            inner: Mutex::new(Some(blazen_pipeline::PipelineBuilder::new(name))),
            persist_fn: Mutex::new(None),
            persist_json_fn: Mutex::new(None),
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

    /// Set a total wall-clock timeout in seconds for the whole pipeline run.
    /// Exceeding it surfaces as a pipeline failure.
    ///
    /// # Errors
    ///
    /// Returns a `JsValue` error if the builder has already been consumed.
    #[wasm_bindgen(js_name = "totalTimeout")]
    pub fn total_timeout(&self, seconds: f64) -> Result<(), JsValue> {
        let mut guard = self.inner.lock().expect("poisoned");
        let builder = guard.take().ok_or_else(|| {
            JsValue::from_str("PipelineBuilder already consumed (build() was called)")
        })?;
        *guard = Some(builder.total_timeout(Duration::from_secs_f64(seconds)));
        Ok(())
    }

    /// Disable any total wall-clock pipeline timeout that was previously
    /// configured (the pipeline runs without a wall-clock cap).
    ///
    /// # Errors
    ///
    /// Returns a `JsValue` error if the builder has already been consumed.
    #[wasm_bindgen(js_name = "noTotalTimeout")]
    pub fn no_total_timeout(&self) -> Result<(), JsValue> {
        let mut guard = self.inner.lock().expect("poisoned");
        let builder = guard.take().ok_or_else(|| {
            JsValue::from_str("PipelineBuilder already consumed (build() was called)")
        })?;
        *guard = Some(builder.no_total_timeout());
        Ok(())
    }

    /// Set a pipeline-level [`RetryConfig`](blazen_llm::retry::RetryConfig)
    /// default for every LLM/embed/HTTP call inside the pipeline. Workflow,
    /// step, and per-call overrides take precedence.
    ///
    /// `options` is a plain JS object with optional fields `maxRetries`,
    /// `initialDelayMs`, `maxDelayMs`, `honorRetryAfter`, `jitter`.
    ///
    /// # Errors
    ///
    /// Returns a `JsValue` error if the builder has already been consumed.
    #[wasm_bindgen(js_name = "retryConfig")]
    pub fn retry_config(&self, options: JsValue) -> Result<(), JsValue> {
        let cfg = crate::decorators::build_retry_config(&options);
        let mut guard = self.inner.lock().expect("poisoned");
        let builder = guard.take().ok_or_else(|| {
            JsValue::from_str("PipelineBuilder already consumed (build() was called)")
        })?;
        *guard = Some(builder.retry_config(cfg));
        Ok(())
    }

    /// Disable any pipeline-level retry default that was previously
    /// configured. Equivalent to `retry_config(RetryConfig { max_retries: 0,
    /// .. })` but more explicit.
    ///
    /// # Errors
    ///
    /// Returns a `JsValue` error if the builder has already been consumed.
    #[wasm_bindgen(js_name = "noRetry")]
    pub fn no_retry(&self) -> Result<(), JsValue> {
        let mut guard = self.inner.lock().expect("poisoned");
        let builder = guard.take().ok_or_else(|| {
            JsValue::from_str("PipelineBuilder already consumed (build() was called)")
        })?;
        *guard = Some(builder.no_retry());
        Ok(())
    }

    /// Clear any pipeline-level retry config so resolution falls back to
    /// (workflow / step / provider) precedence.
    ///
    /// # Errors
    ///
    /// Returns a `JsValue` error if the builder has already been consumed.
    #[wasm_bindgen(js_name = "clearRetryConfig")]
    pub fn clear_retry_config(&self) -> Result<(), JsValue> {
        let mut guard = self.inner.lock().expect("poisoned");
        let builder = guard.take().ok_or_else(|| {
            JsValue::from_str("PipelineBuilder already consumed (build() was called)")
        })?;
        *guard = Some(builder.clear_retry_config());
        Ok(())
    }

    /// Set a persist callback that receives a typed `PipelineSnapshot`
    /// (serialized to a JS object via `serde-wasm-bindgen`) after each
    /// stage completes. The callback may return a `Promise`; the engine
    /// awaits it before continuing.
    ///
    /// # Errors
    ///
    /// Returns a `JsValue` error if the builder has already been
    /// consumed.
    #[wasm_bindgen(js_name = "onPersist")]
    pub fn on_persist(&self, callback: js_sys::Function) -> Result<(), JsValue> {
        if self.inner.lock().expect("poisoned").is_none() {
            return Err(JsValue::from_str(
                "PipelineBuilder already consumed (build() was called)",
            ));
        }
        let wrapper = Arc::new(JsClosure(callback));
        let persist: PersistFn = Arc::new(move |snapshot: PipelineSnapshot| {
            let wrapper = Arc::clone(&wrapper);
            Box::pin(SendFuture(async move {
                let snapshot_js = serde_wasm_bindgen::to_value(&snapshot)
                    .map_err(|e| PipelineError::PersistFailed(e.to_string()))?;
                let promise_val = wrapper
                    .0
                    .call1(&JsValue::NULL, &snapshot_js)
                    .map_err(|e| PipelineError::PersistFailed(format!("{e:?}")))?;
                if promise_val.is_undefined() || promise_val.is_null() {
                    return Ok(());
                }
                let promise = js_sys::Promise::from(promise_val);
                wasm_bindgen_futures::JsFuture::from(promise)
                    .await
                    .map_err(|e| PipelineError::PersistFailed(format!("{e:?}")))?;
                Ok(())
            })) as Pin<Box<dyn Future<Output = Result<(), PipelineError>> + Send>>
        });
        *self.persist_fn.lock().expect("poisoned") = Some(persist);
        Ok(())
    }

    /// Set a persist callback that receives the snapshot serialized as a
    /// JSON string after each stage completes. The callback may return a
    /// `Promise`; the engine awaits it before continuing.
    ///
    /// # Errors
    ///
    /// Returns a `JsValue` error if the builder has already been
    /// consumed.
    #[wasm_bindgen(js_name = "onPersistJson")]
    pub fn on_persist_json(&self, callback: js_sys::Function) -> Result<(), JsValue> {
        if self.inner.lock().expect("poisoned").is_none() {
            return Err(JsValue::from_str(
                "PipelineBuilder already consumed (build() was called)",
            ));
        }
        let wrapper = Arc::new(JsClosure(callback));
        let persist: PersistJsonFn = Arc::new(move |json: String| {
            let wrapper = Arc::clone(&wrapper);
            Box::pin(SendFuture(async move {
                let json_js = JsValue::from_str(&json);
                let promise_val = wrapper
                    .0
                    .call1(&JsValue::NULL, &json_js)
                    .map_err(|e| PipelineError::PersistFailed(format!("{e:?}")))?;
                if promise_val.is_undefined() || promise_val.is_null() {
                    return Ok(());
                }
                let promise = js_sys::Promise::from(promise_val);
                wasm_bindgen_futures::JsFuture::from(promise)
                    .await
                    .map_err(|e| PipelineError::PersistFailed(format!("{e:?}")))?;
                Ok(())
            })) as Pin<Box<dyn Future<Output = Result<(), PipelineError>> + Send>>
        });
        *self.persist_json_fn.lock().expect("poisoned") = Some(persist);
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
        let mut builder = guard
            .take()
            .ok_or_else(|| JsValue::from_str("PipelineBuilder already consumed"))?;
        if let Some(persist) = self.persist_fn.lock().expect("poisoned").take() {
            builder = builder.on_persist(persist);
        }
        if let Some(persist_json) = self.persist_json_fn.lock().expect("poisoned").take() {
            builder = builder.on_persist_json(persist_json);
        }
        let pipeline = builder.build().map_err(pipeline_err)?;
        Ok(WasmPipeline::from_inner(pipeline))
    }
}
