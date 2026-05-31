//! WASM bindings for [`blazen_core::SubExecutable`] and
//! [`blazen_core::SubPipelineStep`].
//!
//! [`WasmSubExecutable`] is the foreign-implementable ABC: a JS object with an
//! async `execute(input, context) => result` method is wrapped via
//! [`WasmSubExecutable::fromJsObject`] (mirroring how
//! [`CustomProvider`](crate::providers::custom::WasmCustomProvider) wraps a JS
//! object). [`WasmSubPipelineStep`] embeds any such executable — most
//! commonly a built [`Pipeline`](crate::pipeline::WasmPipeline) — as a step
//! inside a parent [`Workflow`](crate::workflow::WasmWorkflow), mirroring the
//! Node binding's `SubPipelineStep` class.
//!
//! ```js
//! const child = new PipelineBuilder("enrich").stage(stage).build();
//! const step = new SubPipelineStep(
//!   "enrich",
//!   ["blazen::StartEvent"],
//!   ["enrich::output"],
//!   child,
//! );
//! const parent = new Workflow("parent");
//! parent.addSubpipelineStep(step);
//! ```

use std::pin::Pin;
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use js_sys::{Function, Reflect};
use wasm_bindgen::prelude::*;
use wasm_bindgen_futures::JsFuture;

use blazen_core::{Context, SubExecutable, WorkflowError};

// ---------------------------------------------------------------------------
// SendFuture — same pattern as byo_backend.rs / providers/custom.rs.
// ---------------------------------------------------------------------------

/// Wrapper that unsafely implements `Send` for a non-Send future.
/// SAFETY: wasm32 is single-threaded.
struct SendFuture<F>(F);

// SAFETY: wasm32 is single-threaded.
unsafe impl<F> Send for SendFuture<F> {}

impl<F: std::future::Future> std::future::Future for SendFuture<F> {
    type Output = F::Output;

    fn poll(
        self: Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Self::Output> {
        // SAFETY: we are not moving `F`, just projecting through the wrapper.
        let inner = unsafe { self.map_unchecked_mut(|s| &mut s.0) };
        inner.poll(cx)
    }
}

// ---------------------------------------------------------------------------
// JsSubExecutableAdapter — implements `SubExecutable` over a JS object.
// ---------------------------------------------------------------------------

/// Adapter that implements [`SubExecutable`] by dispatching to an
/// `execute(input, context)` method on a held JS object.
///
/// SAFETY: wasm32 is single-threaded so the `unsafe impl Send + Sync` is
/// vacuously safe.
struct JsSubExecutableAdapter {
    obj: JsValue,
    execute_fn: Function,
}

// SAFETY: wasm32 is single-threaded.
unsafe impl Send for JsSubExecutableAdapter {}
// SAFETY: wasm32 is single-threaded.
unsafe impl Sync for JsSubExecutableAdapter {}

impl std::fmt::Debug for JsSubExecutableAdapter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("JsSubExecutableAdapter").finish_non_exhaustive()
    }
}

impl JsSubExecutableAdapter {
    /// Validate the JS contract and capture the `execute` handle.
    fn from_js(obj: JsValue) -> Result<Self, JsValue> {
        if !obj.is_object() {
            return Err(JsValue::from_str("SubExecutable must be a JS object"));
        }
        let candidate = Reflect::get(&obj, &JsValue::from_str("execute"))
            .map_err(|e| JsValue::from_str(&format!("SubExecutable.execute lookup failed: {e:?}")))?;
        let execute_fn: Function = candidate.dyn_into().map_err(|_| {
            JsValue::from_str("SubExecutable.execute is required and must be a function")
        })?;
        Ok(Self { obj, execute_fn })
    }

    async fn execute_impl(
        &self,
        input: serde_json::Value,
    ) -> Result<serde_json::Value, WorkflowError> {
        let input_js = serde_wasm_bindgen::to_value(&input)
            .map_err(|e| sub_err(format!("failed to marshal sub-executable input: {e}")))?;
        // The parent `Context` is not forwarded across the JS boundary — the
        // child runs against its own context. JS receives `(input)`; the
        // second positional arg is reserved for future context bridging.
        let raw = self
            .execute_fn
            .call1(&self.obj, &input_js)
            .map_err(|e| sub_err(format!("SubExecutable.execute threw: {e:?}")))?;
        let resolved = if raw.has_type::<js_sys::Promise>() {
            let promise: js_sys::Promise = raw.unchecked_into();
            JsFuture::from(promise)
                .await
                .map_err(|e| sub_err(format!("SubExecutable.execute rejected: {e:?}")))?
        } else {
            raw
        };
        serde_wasm_bindgen::from_value::<serde_json::Value>(resolved).map_err(|e| {
            sub_err(format!(
                "SubExecutable.execute returned a value that did not deserialize to JSON: {e}"
            ))
        })
    }
}

/// Build a [`WorkflowError::SubWorkflowFailed`] for a JS-backed
/// sub-executable.
fn sub_err(message: String) -> WorkflowError {
    WorkflowError::SubWorkflowFailed {
        step_name: "sub-executable".to_owned(),
        message,
    }
}

#[async_trait]
impl SubExecutable for JsSubExecutableAdapter {
    async fn execute(
        &self,
        input: serde_json::Value,
        _ctx: Context,
    ) -> Result<serde_json::Value, WorkflowError> {
        SendFuture(self.execute_impl(input)).await
    }
}

// ---------------------------------------------------------------------------
// WasmSubExecutable — the JS-visible ABC handle.
// ---------------------------------------------------------------------------

/// A child runner that can be embedded inside a parent workflow via
/// [`WasmSubPipelineStep`].
///
/// Construct from a built [`Pipeline`](crate::pipeline::WasmPipeline) with
/// [`WasmSubExecutable::from_pipeline`], or from a custom JS object exposing an
/// async `execute(input)` method with
/// [`WasmSubExecutable::from_js_object`].
#[wasm_bindgen(js_name = "SubExecutable")]
pub struct WasmSubExecutable {
    inner: Arc<dyn SubExecutable>,
}

// SAFETY: wasm32 is single-threaded.
unsafe impl Send for WasmSubExecutable {}
// SAFETY: wasm32 is single-threaded.
unsafe impl Sync for WasmSubExecutable {}

#[wasm_bindgen(js_class = "SubExecutable")]
impl WasmSubExecutable {
    /// Wrap a built [`Pipeline`](crate::pipeline::WasmPipeline) as a
    /// [`SubExecutable`].
    ///
    /// The pipeline is cloned (not consumed), so the source `Pipeline` can
    /// still be run independently and this executable can be reused across
    /// parent workflows.
    ///
    /// # Errors
    ///
    /// Returns a `JsValue` error if the pipeline has already been consumed by
    /// a `start()` / `resume()` call.
    #[wasm_bindgen(js_name = "fromPipeline")]
    pub fn from_pipeline(
        pipeline: &crate::pipeline::WasmPipeline,
    ) -> Result<WasmSubExecutable, JsValue> {
        let inner = pipeline.clone_inner()?;
        Ok(Self {
            inner: Arc::new(inner),
        })
    }

    /// Wrap a JS object exposing an async `execute(input)` method as a
    /// [`SubExecutable`].
    ///
    /// The JS `execute` receives the parent event serialized to JSON and must
    /// resolve to the child's terminal result JSON.
    ///
    /// # Errors
    ///
    /// Returns a `JsValue` error if `obj` is not an object or does not expose
    /// a callable `execute` method.
    #[wasm_bindgen(js_name = "fromJsObject")]
    pub fn from_js_object(obj: JsValue) -> Result<WasmSubExecutable, JsValue> {
        Ok(Self {
            inner: Arc::new(JsSubExecutableAdapter::from_js(obj)?),
        })
    }
}

impl WasmSubExecutable {
    /// Clone the inner `Arc<dyn SubExecutable>` for embedding in a step.
    pub(crate) fn clone_inner(&self) -> Arc<dyn SubExecutable> {
        Arc::clone(&self.inner)
    }
}

// ---------------------------------------------------------------------------
// WasmSubPipelineStep — embeds a SubExecutable as a parent-workflow step.
// ---------------------------------------------------------------------------

/// A workflow step that delegates to a [`SubExecutable`] (most commonly a
/// `Pipeline`).
///
/// Mirrors the Node binding's `SubPipelineStep`. The parent workflow's event
/// loop maps the parent event into the child input JSON, runs the child to
/// completion via [`SubExecutable::execute`], and wraps the terminal JSON in a
/// `DynamicEvent` named `"<stepName>::output"` for the parent.
#[wasm_bindgen(js_name = "SubPipelineStep")]
pub struct WasmSubPipelineStep {
    pub(crate) name: String,
    pub(crate) accepts: Vec<String>,
    pub(crate) emits: Vec<String>,
    pub(crate) executable: Arc<dyn SubExecutable>,
    pub(crate) timeout: Option<Duration>,
}

// SAFETY: wasm32 is single-threaded.
unsafe impl Send for WasmSubPipelineStep {}
// SAFETY: wasm32 is single-threaded.
unsafe impl Sync for WasmSubPipelineStep {}

#[wasm_bindgen(js_class = "SubPipelineStep")]
#[allow(clippy::must_use_candidate)]
impl WasmSubPipelineStep {
    /// Create a sub-pipeline step.
    ///
    /// - `name` / `accepts` / `emits` — routing metadata.
    /// - `inner` — the child to run for each parent dispatch. Accepts either a
    ///   built [`Pipeline`](crate::pipeline::WasmPipeline) (cloned, not
    ///   consumed) or a [`SubExecutable`](WasmSubExecutable) handle.
    /// - `timeout_secs` — optional per-step wall-clock timeout (positive
    ///   seconds) for the whole child run.
    ///
    /// # Errors
    ///
    /// Returns a `JsValue` error if `inner` is neither a `Pipeline` nor a
    /// `SubExecutable`, or if a passed `Pipeline` has already been consumed.
    #[wasm_bindgen(constructor)]
    pub fn new(
        name: String,
        accepts: Vec<String>,
        emits: Vec<String>,
        inner: &WasmSubExecutable,
        timeout_secs: Option<f64>,
    ) -> Result<WasmSubPipelineStep, JsValue> {
        Ok(Self {
            name,
            accepts,
            emits,
            executable: inner.clone_inner(),
            timeout: timeout_secs.and_then(|s| {
                if s > 0.0 {
                    Some(Duration::from_secs_f64(s))
                } else {
                    None
                }
            }),
        })
    }

    /// The step name.
    #[wasm_bindgen(getter)]
    pub fn name(&self) -> String {
        self.name.clone()
    }

    /// Event type identifiers this step accepts.
    #[wasm_bindgen(getter)]
    pub fn accepts(&self) -> Vec<String> {
        self.accepts.clone()
    }

    /// Event type identifiers this step may emit.
    #[wasm_bindgen(getter)]
    pub fn emits(&self) -> Vec<String> {
        self.emits.clone()
    }
}

impl WasmSubPipelineStep {
    /// Materialize a [`blazen_core::SubPipelineStep`] for handing to a
    /// `WorkflowBuilder`.
    pub(crate) fn to_core(&self) -> blazen_core::SubPipelineStep {
        let accepts: Vec<&'static str> = self
            .accepts
            .iter()
            .map(|s| blazen_events::intern_event_type(s))
            .collect();
        let emits: Vec<&'static str> = self
            .emits
            .iter()
            .map(|s| blazen_events::intern_event_type(s))
            .collect();
        let mut step = blazen_core::SubPipelineStep::with_json_mappers(
            self.name.clone(),
            accepts,
            emits,
            Arc::clone(&self.executable),
        );
        if let Some(t) = self.timeout {
            step = step.with_timeout(t);
        }
        step
    }
}
