//! WASM binding for [`blazen_pipeline::PipelineHandler`].
//!
//! Mirrors the napi binding's surface: `result()`, `pause()`, `snapshot()`,
//! `streamEvents(callback)`, `resumeInPlace()`, `abort()`. Operations that
//! consume the upstream handler take the inner value out of an `Option`
//! slot; control-only operations (`abort`, `resumeInPlace`) borrow it.
//!
//! # `streamEvents` shape
//!
//! Mirrors `WorkflowHandler::stream_events` in this crate: instead of
//! returning a JS `AsyncIterable` (wasm-bindgen's `Symbol.asyncIterator`
//! support is too thin to wire from Rust today), we accept a JS callback
//! `(event) => void` and call it once per event in a `spawn_local` task.
//! JS callers can wrap repeated `Promise`-resolving subscribers if they
//! need iterator semantics.

use std::cell::RefCell;
use std::pin::Pin;
use std::rc::Rc;

use futures_util::StreamExt;
use serde::Serialize;
use wasm_bindgen::prelude::*;

use crate::pipeline::error::pipeline_err;
use crate::pipeline::event::WasmPipelineEvent;
use crate::pipeline::snapshot::{WasmPipelineResult, WasmPipelineSnapshot};

// ---------------------------------------------------------------------------
// Marshalling helpers
// ---------------------------------------------------------------------------

/// Convert a `Serialize` value into a `JsValue` with maps marshalled as
/// plain JS objects, matching the SDK-wide convention.
fn marshal_to_js<T: Serialize + ?Sized>(value: &T) -> Result<JsValue, JsValue> {
    let serializer = serde_wasm_bindgen::Serializer::new().serialize_maps_as_objects(true);
    value
        .serialize(&serializer)
        .map_err(|e| JsValue::from_str(&format!("marshal failed: {e}")))
}

/// Type-erased pipeline event stream sourced from
/// [`PipelineHandler::stream_events`](blazen_pipeline::PipelineHandler::stream_events).
///
/// Boxed so it can live in a struct field across `await` points without
/// the caller naming the concrete combinator type.
type PipelineEventStream =
    Pin<Box<dyn futures_util::Stream<Item = blazen_pipeline::PipelineEvent> + 'static>>;

// ---------------------------------------------------------------------------
// WasmPipelineHandler
// ---------------------------------------------------------------------------

/// JavaScript-facing wrapper around [`blazen_pipeline::PipelineHandler`].
///
/// `Rc<RefCell<...>>` is used rather than `Arc<Mutex<...>>` because wasm32
/// is single-threaded; the handler cannot cross threads, and a mutex would
/// only add overhead and panic-on-reentry hazards.
#[wasm_bindgen(js_name = "PipelineHandler")]
pub struct WasmPipelineHandler {
    /// Inner handler. `None` once a consuming method (`result`, `pause`)
    /// has taken it.
    inner: Rc<RefCell<Option<blazen_pipeline::PipelineHandler>>>,
}

impl WasmPipelineHandler {
    /// Wrap a real handler. Crate-internal because callers
    /// ([`WasmPipeline::start`](super::pipeline::WasmPipeline::start),
    /// [`WasmPipeline::resume`](super::pipeline::WasmPipeline::resume))
    /// construct this from the engine's
    /// [`PipelineHandler`](blazen_pipeline::PipelineHandler).
    #[must_use]
    pub(crate) fn new(handler: blazen_pipeline::PipelineHandler) -> Self {
        Self {
            inner: Rc::new(RefCell::new(Some(handler))),
        }
    }
}

#[wasm_bindgen(js_class = "PipelineHandler")]
impl WasmPipelineHandler {
    /// Await the final pipeline result.
    ///
    /// Consumes the handler. Returns a [`WasmPipelineResult`] containing
    /// the final output and all stage results.
    ///
    /// # Errors
    ///
    /// Returns a `JsValue` error if the handler was already consumed, or
    /// the pipeline ran to completion with an error
    /// ([`PipelineError`](blazen_pipeline::PipelineError)).
    #[wasm_bindgen]
    pub async fn result(&self) -> Result<WasmPipelineResult, JsValue> {
        let handler = self
            .inner
            .borrow_mut()
            .take()
            .ok_or_else(|| JsValue::from_str("PipelineHandler already consumed"))?;
        let result = handler.result().await.map_err(pipeline_err)?;
        Ok(WasmPipelineResult::from_inner(result))
    }

    /// Pause the pipeline and return a snapshot.
    ///
    /// Consumes the handler since the pipeline is no longer running after
    /// a pause.
    ///
    /// # Errors
    ///
    /// Returns a `JsValue` error if the handler was already consumed or
    /// the pipeline has already terminated.
    #[wasm_bindgen]
    pub async fn pause(&self) -> Result<WasmPipelineSnapshot, JsValue> {
        let handler = self
            .inner
            .borrow_mut()
            .take()
            .ok_or_else(|| JsValue::from_str("PipelineHandler already consumed"))?;
        let snapshot = handler.pause().await.map_err(pipeline_err)?;
        Ok(WasmPipelineSnapshot::from_inner(snapshot))
    }

    /// Resume a paused pipeline in place.
    ///
    /// Currently a no-op at the upstream pipeline level (see
    /// [`PipelineHandler::resume_in_place`](blazen_pipeline::PipelineHandler::resume_in_place));
    /// kept on the JS surface so callers can wire it up ahead of upstream
    /// support landing.
    ///
    /// # Errors
    ///
    /// Returns a `JsValue` error if the handler was already consumed or
    /// the pipeline has already terminated.
    #[wasm_bindgen(js_name = "resumeInPlace")]
    pub fn resume_in_place(&self) -> Result<(), JsValue> {
        let inner = self.inner.borrow();
        let handler = inner
            .as_ref()
            .ok_or_else(|| JsValue::from_str("PipelineHandler already consumed"))?;
        handler.resume_in_place().map_err(pipeline_err)
    }

    /// Capture a snapshot without stopping the pipeline.
    ///
    /// Mirrors [`PipelineHandler::snapshot`](blazen_pipeline::PipelineHandler::snapshot),
    /// which is currently stubbed upstream and always returns
    /// `ChannelClosed`. Surfaced here for forward-compatibility so JS
    /// callers can adopt the API now.
    ///
    /// # Errors
    ///
    /// Currently always returns a `JsValue` error wrapping
    /// [`PipelineError::ChannelClosed`](blazen_pipeline::PipelineError);
    /// also returns an error if the handler has already been consumed.
    #[wasm_bindgen]
    pub async fn snapshot(&self) -> Result<WasmPipelineSnapshot, JsValue> {
        // `PipelineHandler::snapshot` only borrows `&self`, so take the
        // handler out of its slot for the duration of the `.await` to
        // avoid holding a `RefCell` borrow across the await point
        // (clippy `await_holding_refcell_ref` lint), then restore it.
        // Single-threaded wasm makes this safe — no other caller can
        // observe the empty slot during the await.
        let handler = self
            .inner
            .borrow_mut()
            .take()
            .ok_or_else(|| JsValue::from_str("PipelineHandler already consumed"))?;

        let snapshot_result = handler.snapshot().await;

        *self.inner.borrow_mut() = Some(handler);

        let snapshot = snapshot_result.map_err(pipeline_err)?;
        Ok(WasmPipelineSnapshot::from_inner(snapshot))
    }

    /// Abort the pipeline.
    ///
    /// Sends an abort signal to the execution loop. Best-effort; returns
    /// an error if the loop has already exited.
    ///
    /// # Errors
    ///
    /// Returns a `JsValue` error if the handler was already consumed or
    /// the pipeline has already terminated.
    #[wasm_bindgen]
    pub fn abort(&self) -> Result<(), JsValue> {
        let inner = self.inner.borrow();
        let handler = inner
            .as_ref()
            .ok_or_else(|| JsValue::from_str("PipelineHandler already consumed"))?;
        handler.abort().map_err(pipeline_err)
    }

    /// Subscribe to intermediate events from pipeline stages.
    ///
    /// `callback` is a JS function `(event: PipelineEvent) => void`. It
    /// is invoked once per [`PipelineEvent`](blazen_pipeline::PipelineEvent),
    /// receiving a plain JS object with shape
    /// `{ stageName, branchName, workflowRunId, event }`. The `event`
    /// field is the underlying workflow event's `to_json()` representation.
    ///
    /// Returns immediately. The subscription runs in a `spawn_local` task
    /// that pumps events until the broadcast stream closes (typically
    /// when the pipeline completes, pauses, or aborts). The task is
    /// detached; there's no explicit unsubscribe — callers should
    /// `abort()` the pipeline to terminate the stream.
    ///
    /// # Errors
    ///
    /// Returns a `JsValue` error if the handler has already been
    /// consumed.
    #[wasm_bindgen(js_name = "streamEvents")]
    pub fn stream_events(&self, callback: js_sys::Function) -> Result<(), JsValue> {
        let stream: PipelineEventStream = {
            let inner = self.inner.borrow();
            let handler = inner
                .as_ref()
                .ok_or_else(|| JsValue::from_str("PipelineHandler already consumed"))?;
            Box::pin(handler.stream_events())
        };

        wasm_bindgen_futures::spawn_local(pump_events(stream, callback));

        Ok(())
    }
}

/// Drain a pipeline event stream, invoking `callback` for each event.
///
/// Lives outside `WasmPipelineHandler` so that `Drop` of the handler
/// doesn't interrupt event delivery: the spawned task owns the boxed
/// stream and runs to completion when the upstream broadcast closes.
async fn pump_events(mut stream: PipelineEventStream, callback: js_sys::Function) {
    while let Some(event) = stream.next().await {
        let view = WasmPipelineEvent::from_native(&event);
        let js_payload = match marshal_to_js(&view) {
            Ok(value) => value,
            Err(_) => {
                // Marshalling errors at the boundary are non-recoverable
                // for this event; skip and keep pumping so a single
                // malformed event doesn't kill the subscription.
                continue;
            }
        };
        // Ignore JS-side throws: a misbehaving callback should not tear
        // down the subscription or surface as a Rust panic.
        let _ = callback.call1(&JsValue::NULL, &js_payload);
    }
}
