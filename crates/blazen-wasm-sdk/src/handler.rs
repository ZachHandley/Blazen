//! `wasm-bindgen` wrapper for [`blazen_core::WorkflowHandler`].
//!
//! Stage 2 of the WASM rollout (`please-investigate-if-blazen-swift-pnueli`,
//! agent A2.3): expose the real workflow handle to JavaScript with a small,
//! ergonomic surface that matches what JS callers expect from a
//! `WorkflowHandler` returned by `Workflow.run()`.
//!
//! # Exposed JS API
//!
//! - `awaitResult(): Promise<JsValue>` — waits for the workflow's terminal
//!   event, downcasts to [`StopEvent`], and returns the result payload as a
//!   plain JS object (maps marshalled as objects, not `Map` instances — see
//!   [`marshal_to_js`]).
//! - `pause(): Promise<JsValue>` — parks the event loop and resolves with a
//!   serialised [`WorkflowSnapshot`]. Agent A2.5 will retype this as a
//!   `tsify`-exposed `WorkflowSnapshot` interface; for now JS sees a plain
//!   object literal.
//! - `cancel(): void` — calls [`WorkflowHandler::abort`] on the inner
//!   handler. Named `cancel` per the plan so JS callers don't need to know
//!   about Tokio's `abort` terminology.
//! - `nextEvent(): Promise<JsValue | null>` — returns the next event from
//!   the stream channel, or `null` when the stream closes. JS consumers wrap
//!   this in their own async generator to get an `AsyncIterable`.
//!   wasm-bindgen's [`Symbol.asyncIterator`] support is not stable enough to
//!   wire a real async iterable from Rust today, so the convention is to let
//!   the JS layer do the wrapping.
//! - `runId(): Promise<string>` — resolves with the run id by capturing a
//!   transient snapshot the first time it's called and caching the result.
//!   The handler doesn't expose `run_id` directly today (see
//!   `crates/blazen-core/src/handler.rs`), but the snapshot does, so this is
//!   a side-effect-free way to read it without touching `blazen-core`.
//!
//! # Map → Object marshalling
//!
//! `serde_wasm_bindgen::to_value` defaults to emitting Rust `HashMap`s and
//! `BTreeMap`s as JS `Map` instances. `JSON.stringify(new Map())` returns
//! `"{}"`, which silently breaks any JS caller that JSON-stringifies the
//! result. We force `serialize_maps_as_objects(true)` at the SDK boundary
//! via [`marshal_to_js`] so JS callers see plain object literals.

use std::cell::RefCell;
use std::pin::Pin;
use std::rc::Rc;

use blazen_core::WorkflowHandler;
use blazen_events::{AnyEvent, DynamicEvent, InputResponseEvent, StopEvent};
use futures_util::StreamExt;
use serde::Serialize;
use wasm_bindgen::prelude::*;

/// Newtype wrapping a [`js_sys::Function`] so it can cross `await` points
/// inside futures that the runtime treats as `Send`.
///
/// Wasm32 is single-threaded, so the `Send + Sync` impls are vacuously
/// safe — there is no other thread that could observe the wrapped JS
/// function. The wrapper exists purely to satisfy `SendFuture`-style
/// patterns used elsewhere in this crate (see `capability_providers.rs`).
struct JsCallback(js_sys::Function);

// SAFETY: wasm32 is single-threaded; no other thread exists to observe
// the wrapped JS function, so Send/Sync are vacuously satisfied.
unsafe impl Send for JsCallback {}
// SAFETY: see above.
unsafe impl Sync for JsCallback {}

/// Convert any [`Serialize`] value into a [`JsValue`] using the SDK-wide
/// convention that Rust maps marshal as plain JS objects (not `Map`
/// instances). Always prefer this over `serde_wasm_bindgen::to_value`
/// directly so the output round-trips cleanly through `JSON.stringify`.
fn marshal_to_js<T: Serialize + ?Sized>(value: &T) -> Result<JsValue, JsValue> {
    let serializer = serde_wasm_bindgen::Serializer::new().serialize_maps_as_objects(true);
    value
        .serialize(&serializer)
        .map_err(|e| JsValue::from_str(&format!("marshal failed: {e}")))
}

/// Yield once to the JS event loop using a `setTimeout(0)` macrotask.
///
/// # Why this exists
///
/// When `WasmWorkflow::runHandler` returns a `WasmWorkflowHandler` to JS,
/// the workflow's event loop is driven by a `wasm_bindgen_futures::spawn_local`
/// future scheduled inside `blazen-core`'s wasm32 runtime shim
/// (`crates/blazen-core/src/runtime.rs`). In Node.js's V8 host the microtask
/// queue is flushed aggressively between every Promise resolution, so the
/// spawned event-loop future is polled "for free" and any subsequent JS
/// `await handler.awaitResult()` resolves promptly.
///
/// **workerd is not Node.** Cloudflare Workers' I/O context only drives
/// microtasks until the next external I/O wait. When JS code does
/// `const h = await wf.runHandler(...); await h.awaitResult();`, the second
/// `await` returns a Promise from a fresh wasm-bindgen export call. If that
/// call enters a Rust `async fn` and immediately `.await`s a oneshot/channel
/// receiver, no JS microtask runs in between — the spawn_local'd event loop
/// never gets a chance to make progress, and the worker hangs at workerd's
/// "your Worker's code hung" timeout.
///
/// Awaiting a `setTimeout(0)` macrotask forces workerd to fully drain the
/// microtask queue (which is where wasm-bindgen-futures wake-ups live) AND
/// run a turn of the event loop, giving the spawn_local'd event-loop future
/// a chance to advance state before we park on a oneshot/broadcast wait.
///
/// We use `setTimeout(0)` rather than `Promise.resolve()` because the latter
/// is only a microtask — in workerd, awaiting a microtask Promise from
/// inside a wasm-bindgen export does NOT cause the JS engine to run other
/// microtasks; it just resumes the same async chain. A macrotask
/// (`setTimeout`) forces a full event-loop turn, which is what we need.
async fn yield_to_js() {
    use js_sys::Promise;
    use wasm_bindgen_futures::JsFuture;

    let promise = Promise::new(&mut |resolve, _reject| {
        let global = js_sys::global();
        let set_timeout = js_sys::Reflect::get(&global, &JsValue::from_str("setTimeout"))
            .expect("setTimeout missing from global scope");
        let set_timeout: js_sys::Function =
            set_timeout.dyn_into().expect("setTimeout not a function");
        // setTimeout(resolve, 0) — a 0ms macrotask. Workerd treats this as
        // a real I/O wait and drains the microtask queue first.
        let _ = set_timeout.call2(&JsValue::NULL, &resolve, &JsValue::from(0));
    });
    let _ = JsFuture::from(promise).await;
}

/// Type-erased event stream sourced from
/// [`WorkflowHandler::stream_events`]. Boxed and pinned so it can live
/// inside the `WasmWorkflowHandler` struct across `await` points without
/// the caller needing to name the concrete combinator type.
type EventStream = Pin<Box<dyn futures_util::Stream<Item = Box<dyn AnyEvent>> + 'static>>;

/// JavaScript-facing wrapper around [`blazen_core::WorkflowHandler`].
///
/// The inner handler is held in an [`Option`] so that `awaitResult()` can
/// `take()` it (the underlying `result()` consumes `self`) while still
/// leaving the wrapper struct around for the `Drop` glue and any other
/// already-resolved methods.
///
/// `Rc<RefCell<...>>` wrappers are used (rather than the more familiar
/// `Arc<Mutex<...>>`) because wasm32 is single-threaded: the handler isn't
/// `Send` once it crosses the wasm-bindgen boundary, and `Mutex` would only
/// add overhead and panic-on-reentry hazards we don't need.
#[wasm_bindgen(js_name = "WorkflowHandler")]
pub struct WasmWorkflowHandler {
    /// The inner handler. `None` once `awaitResult()` has consumed it.
    inner: Rc<RefCell<Option<WorkflowHandler>>>,
    /// Lazily-populated event stream. Built on first `nextEvent()` call so
    /// that subscribers don't miss events emitted between handler
    /// construction and first poll. (`stream_events()` only delivers events
    /// published *after* subscription.)
    stream: Rc<RefCell<Option<EventStream>>>,
    /// Cached run id, populated on first `runId()` call via a snapshot.
    cached_run_id: Rc<RefCell<Option<String>>>,
}

impl WasmWorkflowHandler {
    /// Wrap a real handler. Crate-internal because callers (agent A2.4's
    /// `WasmWorkflow::run`) construct this from the engine's
    /// [`WorkflowHandler`]; JS code receives a finished `WasmWorkflowHandler`
    /// from the workflow's `run()` Promise.
    ///
    /// `prefetched_run_id` lets callers (`WasmWorkflow::run_handler` and
    /// `WasmWorkflow::resume_from_snapshot`) capture the run id before
    /// returning the handler to JS. This is important on workerd: by the
    /// time JS code calls `handler.runId()`, the spawn_local'd event loop
    /// has typically already run to completion (because `awaitResult`'s
    /// `setTimeout(0)` yield gives the loop a full event-loop turn before
    /// parking on the result oneshot), so a fresh snapshot request would
    /// hit a closed control channel. Pre-fetching while the loop is still
    /// alive avoids that race.
    #[allow(dead_code)] // Wired up by A2.4 in `WasmWorkflow::run`.
    #[must_use]
    pub(crate) fn new(handler: WorkflowHandler) -> Self {
        Self {
            inner: Rc::new(RefCell::new(Some(handler))),
            stream: Rc::new(RefCell::new(None)),
            cached_run_id: Rc::new(RefCell::new(None)),
        }
    }
}

#[wasm_bindgen(js_class = "WorkflowHandler")]
impl WasmWorkflowHandler {
    /// Await the workflow's terminal event and return the result payload as
    /// a plain JS value.
    ///
    /// Downcasts the terminal event to [`StopEvent`] (the canonical
    /// terminator) and returns its `result` JSON. If the workflow stopped
    /// with a [`DynamicEvent`] instead (e.g. a custom terminal event in
    /// JS-defined workflows), returns that event's `data` payload. Otherwise
    /// falls back to [`AnyEvent::to_json`] so callers always get *something*
    /// representable.
    ///
    /// # Errors
    ///
    /// Rejects with a stringified error if:
    ///
    /// - `awaitResult()` was already called (the inner handler was
    ///   consumed),
    /// - the workflow itself produced a [`WorkflowError`](blazen_core::WorkflowError),
    /// - JS marshalling of the payload failed.
    #[wasm_bindgen(js_name = "awaitResult")]
    pub async fn await_result(&self) -> Result<JsValue, JsValue> {
        // Give the spawn_local'd event loop a chance to make progress
        // before we park on the result oneshot. Required for workerd —
        // see `yield_to_js` for the full explanation.
        yield_to_js().await;

        let handler = self
            .inner
            .borrow_mut()
            .take()
            .ok_or_else(|| JsValue::from_str("handler already consumed"))?;

        let result = handler
            .result()
            .await
            .map_err(|e| JsValue::from_str(&format!("workflow failed: {e}")))?;

        let payload = if let Some(stop) = result.event.as_any().downcast_ref::<StopEvent>() {
            stop.result.clone()
        } else if let Some(dynamic) = result.event.as_any().downcast_ref::<DynamicEvent>() {
            dynamic.data.clone()
        } else {
            result.event.to_json()
        };

        marshal_to_js(&payload)
    }

    /// Park the event loop and return a [`WorkflowSnapshot`] of the current
    /// state, serialised to a plain JS object.
    ///
    /// Mirrors the native pattern: `pause` stops dispatching events, then
    /// `snapshot` captures a quiescent view. JS callers can later persist
    /// the snapshot and feed it back to `Workflow.resume()` (agent A2.5).
    ///
    /// # Errors
    ///
    /// Rejects if the handler was already consumed, the event loop has
    /// exited, or marshalling fails.
    #[wasm_bindgen(js_name = "pause")]
    pub async fn pause(&self) -> Result<JsValue, JsValue> {
        let handler = self
            .inner
            .borrow_mut()
            .take()
            .ok_or_else(|| JsValue::from_str("handler already consumed"))?;

        // **Send pause synchronously, BEFORE yielding to JS.** The pause
        // control message is non-async (just an mpsc::send), so we can
        // queue it on the control channel before giving the event loop
        // any opportunity to run. If we yielded first, on workerd the
        // `setTimeout(0)` macrotask would give the spawn_local'd loop a
        // full event-loop turn — long enough for a trivial workflow to
        // run to completion and close the control channel before pause
        // arrives. Sending pause first guarantees the loop sees the
        // Pause command in its control channel before processing further
        // events.
        let pause_send = handler
            .pause()
            .map_err(|e| JsValue::from_str(&format!("pause failed: {e}")));

        // Now yield to give the loop a chance to *process* the pause
        // and reach a quiescent state before we ask for the snapshot.
        // See `yield_to_js` for the workerd-specific microtask /
        // macrotask explanation.
        yield_to_js().await;

        let result = async {
            pause_send?;
            handler
                .snapshot()
                .await
                .map_err(|e| JsValue::from_str(&format!("snapshot failed: {e}")))
        }
        .await;

        // Restore the handler regardless of success so subsequent calls
        // (e.g. `resume_in_place` via a future API, or `awaitResult`)
        // still see a populated slot.
        *self.inner.borrow_mut() = Some(handler);

        let snapshot = result?;
        marshal_to_js(&snapshot)
    }

    /// Tear down the event loop. Best-effort: returns an error string if
    /// the loop has already exited.
    ///
    /// Wraps [`WorkflowHandler::abort`] under the JS-friendly name `cancel`
    /// per the plan, so JS callers don't have to learn Tokio's task-abort
    /// terminology.
    ///
    /// # Errors
    ///
    /// Returns a stringified [`WorkflowError`](blazen_core::WorkflowError)
    /// if the underlying handler is already gone.
    #[wasm_bindgen(js_name = "cancel")]
    pub fn cancel(&self) -> Result<(), JsValue> {
        let inner = self.inner.borrow();
        let handler = inner
            .as_ref()
            .ok_or_else(|| JsValue::from_str("handler already consumed"))?;

        handler
            .abort()
            .map_err(|e| JsValue::from_str(&format!("cancel failed: {e}")))
    }

    /// Pull the next event from the workflow's broadcast stream.
    ///
    /// Returns the event (with `event_type` and `data` fields) on the
    /// first poll, then continues yielding events until the workflow ends,
    /// at which point it resolves with `null`.
    ///
    /// JS callers should wrap repeated calls in an async generator to get
    /// an `AsyncIterable`:
    ///
    /// ```js
    /// async function* streamEvents(handler) {
    ///   while (true) {
    ///     const ev = await handler.nextEvent();
    ///     if (ev === null) return;
    ///     yield ev;
    ///   }
    /// }
    /// ```
    ///
    /// Why not return a real `AsyncIterable` directly? wasm-bindgen's
    /// `Symbol.asyncIterator` story isn't stable enough to wire from Rust
    /// without hand-rolled JS glue, and the JS-side wrapper above is two
    /// lines. Once wasm-bindgen's iterable support firms up we can revisit.
    ///
    /// # Errors
    ///
    /// Rejects with a stringified marshalling error if the event payload
    /// can't be encoded to JS. Stream-side errors (lagged subscribers) are
    /// silently filtered out by the underlying [`tokio_stream`] adapter, so
    /// this method never surfaces a "stream error" — only a clean `null`
    /// when the stream closes.
    #[wasm_bindgen(js_name = "nextEvent")]
    pub async fn next_event(&self) -> Result<JsValue, JsValue> {
        // Give the spawn_local'd event loop a chance to make progress
        // before we park on the broadcast stream. Required for workerd —
        // see `yield_to_js` for the full explanation.
        yield_to_js().await;

        // Lazily build the stream on first call. Subscribing here (rather
        // than in `new()`) is fine because `stream_events()` returns a fresh
        // broadcast subscription and JS callers are expected to start
        // pulling events synchronously after `run()` returns.
        if self.stream.borrow().is_none() {
            let inner = self.inner.borrow();
            let handler = inner
                .as_ref()
                .ok_or_else(|| JsValue::from_str("handler already consumed"))?;

            let stream = handler.stream_events();
            // The native return type is `impl Stream + Send + Unpin`; we
            // erase to a boxed dyn so the stream can live in a struct field
            // across `await` points without naming the combinator type.
            let boxed: EventStream = Box::pin(stream);
            *self.stream.borrow_mut() = Some(boxed);
        }

        // Take the stream out of its slot to avoid holding a `RefCell`
        // borrow across `next().await`. Single-threaded wasm guarantees no
        // concurrent caller can observe the empty slot, so this is safe;
        // the alternative (holding `RefMut` across the await) trips
        // clippy's `await_holding_refcell_ref` lint.
        let mut stream = self
            .stream
            .borrow_mut()
            .take()
            .ok_or_else(|| JsValue::from_str("stream slot was unexpectedly empty"))?;

        let next = stream.next().await;

        // Restore the stream so subsequent `nextEvent` calls keep pulling
        // from the same subscription (re-subscribing would lose any events
        // emitted between calls).
        *self.stream.borrow_mut() = Some(stream);

        match next {
            Some(event) => {
                // Serialise the event as `{ event_type, data }` so JS gets
                // a discriminant + payload pair regardless of whether the
                // underlying type is a static `#[derive(Event)]` struct or
                // a `DynamicEvent`. `to_json()` is the canonical
                // representation defined on the `Event` trait.
                let envelope = serde_json::json!({
                    "event_type": event.event_type_id(),
                    "data": event.to_json(),
                });
                marshal_to_js(&envelope)
            }
            None => Ok(JsValue::NULL),
        }
    }

    /// Return this run's UUID as a string.
    ///
    /// The native [`WorkflowHandler`] doesn't expose `run_id` directly, so
    /// the first call captures a snapshot to read `snapshot.run_id`, then
    /// caches the value for subsequent calls. The snapshot is intentionally
    /// not paired with a `pause()` here because callers who haven't asked
    /// for a pause shouldn't have their workflow parked as a side effect of
    /// reading the id; `WorkflowHandler::snapshot` works on a live loop and
    /// just captures whatever state it sees.
    ///
    /// # Errors
    ///
    /// Rejects if the handler was already consumed by `awaitResult()` or
    /// the event loop has exited.
    #[wasm_bindgen(js_name = "runId")]
    pub async fn run_id(&self) -> Result<JsValue, JsValue> {
        // Fast-path: return cached id without touching the handler. Both
        // entry points that construct a `WasmWorkflowHandler`
        // (`WasmWorkflow::run_handler` and `WasmWorkflow::resume_from_snapshot`)
        // pre-fetch the run id and pass it to `WasmWorkflowHandler::new`,
        // so this fast-path covers the overwhelmingly common case. Cloning
        // out of the borrow keeps the `Ref` from spanning any potential
        // await below.
        let cached = self.cached_run_id.borrow().clone();
        if let Some(id) = cached {
            return Ok(JsValue::from_str(&id));
        }

        // Slow path: no pre-fetched id. Yield to give the spawn_local'd
        // event loop a chance to make progress, then issue a snapshot.
        // See `yield_to_js` for the full explanation.
        yield_to_js().await;

        // Take the handler out of its slot to issue the snapshot without
        // holding a `RefCell` borrow across `snapshot().await`. Restored
        // afterwards so other methods see the populated slot.
        let handler = self
            .inner
            .borrow_mut()
            .take()
            .ok_or_else(|| JsValue::from_str("handler already consumed"))?;

        let snapshot_result = handler.snapshot().await;

        *self.inner.borrow_mut() = Some(handler);

        let snapshot =
            snapshot_result.map_err(|e| JsValue::from_str(&format!("snapshot failed: {e}")))?;

        let id = snapshot.run_id.to_string();
        *self.cached_run_id.borrow_mut() = Some(id.clone());
        Ok(JsValue::from_str(&id))
    }

    /// Deliver a human-in-the-loop response to a workflow that auto-parked
    /// on an `InputRequestEvent`.
    ///
    /// Mirrors [`JsWorkflowHandler::respond_to_input`] in the Node bindings:
    /// the workflow's event loop unparks and injects the response as a
    /// routable event. JS callers pass the matching `request_id` (from the
    /// original `InputRequestEvent`) and any JSON-serialisable response
    /// value.
    ///
    /// # Errors
    ///
    /// Rejects if the handler was already consumed, the response payload
    /// can't be deserialised, or the event loop has already exited.
    #[wasm_bindgen(js_name = "respondToInput")]
    pub fn respond_to_input(&self, request_id: String, response: JsValue) -> Result<(), JsValue> {
        let inner = self.inner.borrow();
        let handler = inner
            .as_ref()
            .ok_or_else(|| JsValue::from_str("handler already consumed"))?;

        // Convert the JS value into the JSON shape expected by
        // `InputResponseEvent.response`. `JsValue::UNDEFINED` collapses to
        // `null` so callers can omit a payload entirely if they want.
        let response_json: serde_json::Value = if response.is_undefined() {
            serde_json::Value::Null
        } else {
            serde_wasm_bindgen::from_value(response)
                .map_err(|e| JsValue::from_str(&format!("invalid response payload: {e}")))?
        };

        let event = InputResponseEvent {
            request_id,
            response: response_json,
        };

        handler
            .respond_to_input(event)
            .map_err(|e| JsValue::from_str(&format!("respondToInput failed: {e}")))
    }

    /// Capture the workflow's current snapshot **without** halting the
    /// event loop.
    ///
    /// Unlike [`pause`](Self::pause), this does not send a Pause control
    /// message — it just asks the loop for a snapshot and lets execution
    /// continue. Use this when you want a JSON-serialisable view of the
    /// run's state for logging or telemetry.
    ///
    /// For a quiescent snapshot (no in-flight steps), pair `pause()` →
    /// `snapshot()` → `resumeInPlace()`.
    ///
    /// # Errors
    ///
    /// Rejects if the handler was already consumed, the event loop has
    /// exited, or marshalling fails.
    #[wasm_bindgen(js_name = "snapshot")]
    pub async fn snapshot(&self) -> Result<JsValue, JsValue> {
        // Yield first so the spawn_local'd event loop can advance to a
        // point where it can service the snapshot control message. See
        // `yield_to_js` for the workerd-specific explanation.
        yield_to_js().await;

        // Take the handler out of its slot so we don't hold a `RefCell`
        // borrow across the await; restored unconditionally below.
        let handler = self
            .inner
            .borrow_mut()
            .take()
            .ok_or_else(|| JsValue::from_str("handler already consumed"))?;

        let snapshot_result = handler.snapshot().await;

        *self.inner.borrow_mut() = Some(handler);

        let snapshot =
            snapshot_result.map_err(|e| JsValue::from_str(&format!("snapshot failed: {e}")))?;

        marshal_to_js(&snapshot)
    }

    /// Resume a paused event loop in place.
    ///
    /// Mirrors [`JsWorkflowHandler::resume_in_place`] in the Node bindings.
    /// After a successful [`pause`](Self::pause), this unparks the loop so
    /// it resumes dispatching events to steps. The same handler instance
    /// continues to be valid for [`await_result`](Self::await_result),
    /// [`next_event`](Self::next_event), etc.
    ///
    /// # Errors
    ///
    /// Rejects if the handler was already consumed or the event loop has
    /// already exited.
    #[wasm_bindgen(js_name = "resumeInPlace")]
    pub fn resume_in_place(&self) -> Result<(), JsValue> {
        let inner = self.inner.borrow();
        let handler = inner
            .as_ref()
            .ok_or_else(|| JsValue::from_str("handler already consumed"))?;

        handler
            .resume_in_place()
            .map_err(|e| JsValue::from_str(&format!("resumeInPlace failed: {e}")))
    }

    /// Subscribe to the workflow's broadcast event stream and forward each
    /// event to a JS callback until the stream closes.
    ///
    /// Mirrors [`JsWorkflowHandler::stream_events`] in the Node bindings.
    /// The returned Promise resolves when the stream closes (either the
    /// workflow completed or was aborted). Unlike [`next_event`](Self::next_event),
    /// this is a single Promise that drives the whole subscription, so JS
    /// callers don't need to wrap repeated calls in their own loop.
    ///
    /// Events emitted before this call are not replayed — `stream_events()`
    /// only delivers events published *after* subscription. Call this
    /// before [`await_result`](Self::await_result) to avoid races.
    ///
    /// # Errors
    ///
    /// Rejects if `callback` is not a function, the handler was already
    /// consumed, or marshalling an event payload fails.
    #[wasm_bindgen(js_name = "streamEvents")]
    pub async fn stream_events(&self, callback: js_sys::Function) -> Result<(), JsValue> {
        // Yield so the loop can publish any already-queued events into the
        // broadcast channel before we subscribe. See `yield_to_js` for the
        // workerd-specific explanation.
        yield_to_js().await;

        let mut stream: EventStream = {
            let inner = self.inner.borrow();
            let handler = inner
                .as_ref()
                .ok_or_else(|| JsValue::from_str("handler already consumed"))?;
            Box::pin(handler.stream_events())
        };

        let cb = JsCallback(callback);

        while let Some(event) = stream.next().await {
            // Drop the stream-end sentinel for parity with the Node
            // bindings — it's an internal signal, not a user-facing event.
            if event.event_type_id() == "blazen::StreamEnd" {
                break;
            }

            let envelope = serde_json::json!({
                "event_type": event.event_type_id(),
                "data": event.to_json(),
            });
            let js_event = marshal_to_js(&envelope)?;

            // Ignore callback errors so a faulty subscriber can't tear
            // down the whole subscription; matching Node's NonBlocking
            // semantics.
            let _ = cb.0.call1(&JsValue::NULL, &js_event);
        }

        Ok(())
    }

    /// Tear down the event loop. Pure alias for [`cancel`](Self::cancel),
    /// matching [`JsWorkflowHandler::abort`] in the Node bindings so JS
    /// callers can use whichever name reads better in context.
    ///
    /// # Errors
    ///
    /// Returns a stringified error if the underlying handler is already
    /// gone.
    #[wasm_bindgen(js_name = "abort")]
    pub fn abort(&self) -> Result<(), JsValue> {
        self.cancel()
    }
}
