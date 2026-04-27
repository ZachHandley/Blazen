//! `wasm-bindgen` wrapper for the real Blazen workflow engine.
//!
//! Stage 2 of the WASM rollout (`please-investigate-if-blazen-swift-pnueli`):
//! this module replaces the previous hand-rolled simplified event loop with
//! a thin wrapper around [`blazen_core::WorkflowBuilder`]. JS-registered
//! step callbacks are buffered as [`PendingStep`]s until `run()` is invoked;
//! at that point they are translated into real [`StepRegistration`]s whose
//! handlers marshal events and a [`WasmContext`] wrapper into JS, invoke the
//! caller's JS function, await any returned `Promise`, and marshal the
//! returned `{ type, ...data }` object back into a [`DynamicEvent`] for the
//! engine.
//!
//! **Status (this agent A2.4)**: full JS-callback adapter wired into the
//! real engine. `run()` builds the workflow, dispatches it, awaits the
//! terminal event, and resolves with the [`StopEvent`] payload (or the data
//! field of a terminal [`DynamicEvent`], or the raw event JSON as a final
//! fallback). The richer "return a `WorkflowHandler` for streaming/pause"
//! shape is intentionally not exposed yet — the smoke-test JS surface
//! returns the payload directly and that's what existing callers expect.

use std::pin::Pin;
use std::sync::Arc;

use blazen_core::{
    Context, StepFn, StepOutput, StepRegistration, Workflow, WorkflowBuilder, WorkflowError,
    WorkflowSnapshot,
};
use blazen_events::{AnyEvent, DynamicEvent, StopEvent};
use serde::Serialize;
use wasm_bindgen::prelude::*;

// ---------------------------------------------------------------------------
// WasmWorkflow
// ---------------------------------------------------------------------------

/// A workflow exposed to JavaScript that wraps a real
/// [`blazen_core::WorkflowBuilder`].
///
/// Steps are JavaScript callback functions that receive an event and a
/// context object and return the next event (or `null` to stop). The
/// callbacks are buffered until `run()` is called, at which point they are
/// adapted into native step registrations and dispatched through the real
/// engine.
///
/// ```js
/// const wf = new Workflow('my-flow');
/// wf.addStep('process', ['StartEvent'], (event, ctx) => {
///   return { type: 'StopEvent', result: event.data };
/// });
/// const result = await wf.run({ data: 'hello' });
/// ```
#[wasm_bindgen(js_name = "Workflow")]
pub struct WasmWorkflow {
    /// Workflow name. Cached separately from `builder` so the `name` getter
    /// remains cheap and survives `builder` being moved into `build()`.
    name: String,
    /// The real engine builder. Held in an `Option` so that `run()` can
    /// move it out via `Option::take` when calling `build()`, which
    /// consumes `self`.
    builder: Option<WorkflowBuilder>,
    /// Pending JS step registrations. Drained on `run()` and adapted into
    /// native [`StepRegistration`]s whose handlers call back into the JS
    /// function.
    pending_steps: Vec<PendingStep>,
}

/// A JS step registration that has not yet been adapted into a native
/// [`blazen_core::StepRegistration`].
///
/// Drained in `run()`. The shape is intentionally minimal: a unique step
/// name, the event type strings the step accepts, and the JS callback to
/// invoke.
pub(crate) struct PendingStep {
    /// Unique step identifier.
    pub(crate) name: String,
    /// Event type strings this step responds to (e.g. `["StartEvent"]`).
    pub(crate) event_types: Vec<String>,
    /// The JS callback. Signature is `(event, context) => Event | Promise<Event> | null`.
    pub(crate) handler: js_sys::Function,
}

#[wasm_bindgen(js_class = "Workflow")]
impl WasmWorkflow {
    /// Create a new workflow with the given name.
    #[wasm_bindgen(constructor)]
    #[must_use]
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_owned(),
            builder: Some(WorkflowBuilder::new(name)),
            pending_steps: Vec::new(),
        }
    }

    /// The workflow name.
    #[wasm_bindgen(getter)]
    #[must_use]
    pub fn name(&self) -> String {
        self.name.clone()
    }

    /// Register a step handler.
    ///
    /// - `name` -- unique step identifier
    /// - `event_types` -- array of event type strings this step responds to
    ///   (e.g. `["StartEvent"]`)
    /// - `handler` -- a function `(event, context) => Event | null`.
    ///   Returning `null` or an object with `type: "StopEvent"` ends the
    ///   workflow. The handler may be async (return a `Promise`).
    ///
    /// The registration is buffered as a [`PendingStep`] and adapted into
    /// the real engine on [`run`](Self::run).
    ///
    /// # Errors
    ///
    /// Returns a `JsValue` error if `event_types` is not an array of strings.
    #[wasm_bindgen(js_name = "addStep")]
    pub fn add_step(
        &mut self,
        name: &str,
        event_types: JsValue,
        handler: js_sys::Function,
    ) -> Result<(), JsValue> {
        let types_array = js_sys::Array::from(&event_types);
        let mut event_types_vec = Vec::with_capacity(types_array.length() as usize);
        for i in 0..types_array.length() {
            let t = types_array
                .get(i)
                .as_string()
                .ok_or_else(|| JsValue::from_str("event_types must be an array of strings"))?;
            event_types_vec.push(t);
        }

        self.pending_steps.push(PendingStep {
            name: name.to_owned(),
            event_types: event_types_vec,
            handler,
        });

        Ok(())
    }

    /// Execute the workflow with the given input data.
    ///
    /// Returns a `Promise` that resolves to the final result (the `result`
    /// field of the [`StopEvent`], or the `data` field of a terminal
    /// [`DynamicEvent`], or the raw event JSON as a fallback).
    ///
    /// Each `WasmWorkflow` is single-shot: calling `run()` consumes the
    /// internal builder, so a second `run()` returns an error rather than
    /// silently re-running with stale state.
    ///
    /// # Errors
    ///
    /// Returns a `JsValue` error if:
    ///
    /// - `run()` was already called on this workflow,
    /// - the input cannot be JSON-serialised,
    /// - the workflow build fails (e.g. duplicate step names),
    /// - any step throws, rejects, or returns a non-JSON-serialisable value,
    /// - the workflow itself errors during execution.
    #[wasm_bindgen]
    pub async fn run(&mut self, input: JsValue) -> Result<JsValue, JsValue> {
        let mut builder = self
            .builder
            .take()
            .ok_or_else(|| JsValue::from_str("Workflow already run; reuse not supported"))?;

        let workflow_name = self.name.clone();

        // Drain pending steps and convert each into a real StepRegistration
        // whose handler calls back into the JS function.
        let pending = std::mem::take(&mut self.pending_steps);
        for pending_step in pending {
            let registration = step_registration_from_js(pending_step, workflow_name.clone());
            builder = builder.step(registration);
        }

        let workflow = builder
            .no_timeout()
            .build()
            .map_err(|e| JsValue::from_str(&format!("workflow build failed: {e}")))?;

        // Convert the input JsValue to a serde_json::Value for `Workflow::run`.
        // Treat `undefined` / `null` as an empty object so callers can pass
        // nothing (`workflow.run()` from JS) without an error.
        let input_json: serde_json::Value = if input.is_undefined() || input.is_null() {
            serde_json::Value::Null
        } else {
            serde_wasm_bindgen::from_value(input)
                .map_err(|e| JsValue::from_str(&format!("input must be JSON-serializable: {e}")))?
        };

        let handler = workflow
            .run(input_json)
            .await
            .map_err(|e| JsValue::from_str(&format!("workflow run failed: {e}")))?;

        let result = handler
            .result()
            .await
            .map_err(|e| JsValue::from_str(&format!("workflow result failed: {e}")))?;

        // Pull the payload off the terminal event in the same order of
        // preference as `WasmWorkflowHandler::await_result` so JS callers
        // see a consistent shape regardless of which entry point ran the
        // workflow.
        let payload = if let Some(stop) = result.event.as_any().downcast_ref::<StopEvent>() {
            stop.result.clone()
        } else if let Some(dynamic) = result.event.as_any().downcast_ref::<DynamicEvent>() {
            dynamic.data.clone()
        } else {
            result.event.to_json()
        };

        marshal_to_js(&payload)
    }

    /// Build and dispatch the workflow, returning the live
    /// [`WasmWorkflowHandler`] instead of awaiting the terminal event.
    ///
    /// Same plumbing as [`run`](Self::run) up to the point at which the
    /// engine's [`WorkflowHandler`](blazen_core::WorkflowHandler) becomes
    /// available, but stops there so JS callers can drive the handle
    /// themselves — calling `awaitResult()` to get the final payload,
    /// `pause()`/`snapshot` to capture mid-flight state, `nextEvent()` to
    /// stream events, or `cancel()` to tear the loop down. The simpler
    /// "fire and forget" [`run`](Self::run) entry point is unchanged for
    /// callers that only need the result.
    ///
    /// Like [`run`](Self::run), each `WasmWorkflow` is single-shot: the
    /// internal builder is consumed by `runHandler`, so a second call to
    /// either `run` or `runHandler` returns an error.
    ///
    /// # Errors
    ///
    /// Returns a `JsValue` error under the same conditions as
    /// [`run`](Self::run): the workflow was already run, the input is not
    /// JSON-serialisable, the build failed, or the engine rejected the
    /// initial dispatch.
    #[wasm_bindgen(js_name = "runHandler")]
    pub async fn run_handler(
        &mut self,
        input: JsValue,
    ) -> Result<crate::handler::WasmWorkflowHandler, JsValue> {
        let mut builder = self
            .builder
            .take()
            .ok_or_else(|| JsValue::from_str("Workflow already run; reuse not supported"))?;

        let workflow_name = self.name.clone();

        let pending = std::mem::take(&mut self.pending_steps);
        for pending_step in pending {
            let registration = step_registration_from_js(pending_step, workflow_name.clone());
            builder = builder.step(registration);
        }

        let workflow = builder
            .no_timeout()
            .build()
            .map_err(|e| JsValue::from_str(&format!("workflow build failed: {e}")))?;

        let input_json: serde_json::Value = if input.is_undefined() || input.is_null() {
            serde_json::Value::Null
        } else {
            serde_wasm_bindgen::from_value(input)
                .map_err(|e| JsValue::from_str(&format!("input must be JSON-serializable: {e}")))?
        };

        let handler = workflow
            .run(input_json)
            .await
            .map_err(|e| JsValue::from_str(&format!("workflow run failed: {e}")))?;

        Ok(crate::handler::WasmWorkflowHandler::new(handler))
    }

    /// Resume a workflow from a previously-captured [`WorkflowSnapshot`]
    /// (typically produced by [`WasmWorkflowHandler::pause`]).
    ///
    /// JS callers reconstruct a fresh `Workflow` with the same `addStep`
    /// calls as the original (step handler functions can't be serialised,
    /// so they have to be re-registered), then call `resumeFromSnapshot`
    /// instead of `run`. The workflow's pending events, context state,
    /// collected events, and metadata are restored from the snapshot, and
    /// the engine continues dispatching from where the original loop was
    /// paused.
    ///
    /// Like [`run`](Self::run), this method consumes the workflow: the
    /// internal builder slot is moved out of, and pending steps are
    /// drained, so a second call returns an error.
    ///
    /// # Errors
    ///
    /// Returns a `JsValue` error if:
    ///
    /// - the workflow was already run or resumed,
    /// - the snapshot can't be deserialised from JS (must round-trip
    ///   through [`WorkflowSnapshot`]'s serde representation),
    /// - the step set rejects rebuild (e.g. duplicate names),
    /// - the engine fails to re-inject pending events.
    #[wasm_bindgen(js_name = "resumeFromSnapshot")]
    pub async fn resume_from_snapshot(
        &mut self,
        snapshot_js: JsValue,
    ) -> Result<crate::handler::WasmWorkflowHandler, JsValue> {
        // Mark the builder as consumed so `run`/`runHandler`/another
        // `resumeFromSnapshot` on the same instance fail loudly. The
        // builder itself is not used on the resume path — `Workflow::resume`
        // rebuilds its own registry from the supplied step list — but
        // taking it here preserves the single-shot contract.
        let _builder = self
            .builder
            .take()
            .ok_or_else(|| JsValue::from_str("Workflow already run; reuse not supported"))?;

        let snapshot: WorkflowSnapshot =
            serde_wasm_bindgen::from_value(snapshot_js).map_err(|e| {
                JsValue::from_str(&format!("snapshot deserialize failed: {e}"))
            })?;

        let workflow_name = self.name.clone();

        let pending = std::mem::take(&mut self.pending_steps);
        let mut steps = Vec::with_capacity(pending.len());
        for pending_step in pending {
            steps.push(step_registration_from_js(pending_step, workflow_name.clone()));
        }

        let handler = Workflow::resume(snapshot, steps, None)
            .await
            .map_err(|e| JsValue::from_str(&format!("workflow resume failed: {e}")))?;

        Ok(crate::handler::WasmWorkflowHandler::new(handler))
    }
}

// ---------------------------------------------------------------------------
// JS-callback step adapter
// ---------------------------------------------------------------------------

/// A `Send + Sync` wrapper around a [`js_sys::Function`].
///
/// `StepFn` requires `Send + Sync` because the workflow engine expects step
/// handlers to be safe to invoke from any thread. `js_sys::Function` is
/// neither, but on `wasm32-unknown-unknown` there is only ever one OS thread,
/// so the `Send`/`Sync` bounds are satisfiable in practice.
///
/// SAFETY: this `unsafe impl` is sound on `wasm32` because:
///
/// 1. The runtime is strictly single-threaded — no two threads can ever
///    observe this value concurrently.
/// 2. The wrapper exposes no API for mutation; the inner `Function` is only
///    `clone`d and `call*`-ed.
///
/// On any non-wasm32 target the SDK isn't compiled (the crate's
/// `crate-type = ["cdylib", "rlib"]` is built for `wasm32-unknown-unknown`
/// in CI), so no real cross-thread handoff is possible.
struct SendFn(js_sys::Function);

// SAFETY: see comment on `SendFn` — wasm32 is single-threaded.
unsafe impl Send for SendFn {}
// SAFETY: see comment on `SendFn` — wasm32 is single-threaded.
unsafe impl Sync for SendFn {}

/// Wrap a non-`Send` future so it can be returned from a `StepFn`.
///
/// `wasm_bindgen_futures::JsFuture` is not `Send`, but `StepFn` requires
/// `Pin<Box<dyn Future<Output = …> + Send>>`. On `wasm32-unknown-unknown` the
/// runtime is single-threaded so a future never actually crosses threads.
/// This wrapper mirrors `blazen_llm::http_fetch::SendFuture` — the same
/// pattern is already used elsewhere in the SDK.
///
/// SAFETY: wasm32 is single-threaded; nothing crosses threads.
struct SendFuture<F>(F);

// SAFETY: see comment on `SendFuture` — wasm32 is single-threaded.
unsafe impl<F> Send for SendFuture<F> {}

impl<F: std::future::Future> std::future::Future for SendFuture<F> {
    type Output = F::Output;

    fn poll(
        self: Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Self::Output> {
        // SAFETY: structural projection through the wrapper. We never move
        // `F` out of `self`.
        let inner = unsafe { self.map_unchecked_mut(|s| &mut s.0) };
        inner.poll(cx)
    }
}

/// Convert any [`Serialize`] value into a [`JsValue`] using the SDK-wide
/// convention that Rust maps marshal as plain JS objects (not `Map`
/// instances). Mirrors the `marshal_to_js` helper in `handler.rs` so JSON
/// stringification of step return values round-trips cleanly.
fn marshal_to_js<T: Serialize + ?Sized>(value: &T) -> Result<JsValue, JsValue> {
    let serializer = serde_wasm_bindgen::Serializer::new().serialize_maps_as_objects(true);
    value
        .serialize(&serializer)
        .map_err(|e| JsValue::from_str(&format!("marshal failed: {e}")))
}

/// Marshal a type-erased event into a `JsValue` shaped as the event's JSON
/// representation. Used to feed each step's JS callback the same `{ ...data }`
/// shape regardless of whether the underlying event is a static
/// `#[derive(Event)]` struct or a [`DynamicEvent`].
fn marshal_event_to_js(event: &dyn AnyEvent) -> Result<JsValue, WorkflowError> {
    let json = event.to_json();
    let serializer = serde_wasm_bindgen::Serializer::new().serialize_maps_as_objects(true);
    json.serialize(&serializer)
        .map_err(|e| WorkflowError::Context(format!("event marshal failed: {e}")))
}

/// Convert a buffered [`PendingStep`] into a real
/// [`blazen_core::StepRegistration`] whose handler dispatches into the
/// supplied JS callback.
///
/// The returned registration does the following on each invocation:
///
/// 1. Marshals the type-erased event into a `JsValue` via [`marshal_event_to_js`].
/// 2. Wraps the live [`Context`] in a [`WasmContext`] (exported to JS as
///    `WorkflowContext`) and converts it to a `JsValue`.
/// 3. Invokes the JS handler with `(event, context)`.
/// 4. If the handler returned a `Promise`, awaits it via [`wasm_bindgen_futures::JsFuture`].
/// 5. Marshals the resolved value back into a [`DynamicEvent`] (or treats
///    `null`/`undefined` as [`StepOutput::None`]).
///
/// `StepRegistration::accepts` requires `&'static str`, but our event-type
/// strings come in as owned `String`s from JS. We leak them to obtain
/// `'static` lifetimes. This is acceptable because:
///
/// - Each `WasmWorkflow` is single-shot (consumed by `run()`).
/// - The number of distinct workflow builds in a page lifetime is bounded by
///   the number of times JS code calls `new Workflow(...)` — typically a
///   small number per session.
/// - The leaked strings are tiny (event type names, e.g. `"StartEvent"`).
fn step_registration_from_js(
    pending_step: PendingStep,
    workflow_name: String,
) -> StepRegistration {
    let PendingStep {
        name,
        event_types,
        handler,
    } = pending_step;

    let handler = Arc::new(SendFn(handler));

    // Leak the event-type strings to obtain `'static` references. See doc
    // comment above for the rationale.
    let accepts: Vec<&'static str> = event_types
        .into_iter()
        .map(|s| &*Box::leak(s.into_boxed_str()))
        .collect();

    let step_name = name.clone();

    let step_handler: StepFn = Arc::new(move |event: Box<dyn AnyEvent>, ctx: Context| {
        let handler = Arc::clone(&handler);
        let step_name = step_name.clone();
        let workflow_name = workflow_name.clone();
        Box::pin(SendFuture(dispatch_js_step(
            handler,
            step_name,
            workflow_name,
            event,
            ctx,
        )))
    });

    // `emits` is informational; the engine routes by the runtime
    // `event_type_id()` of the boxed event, not this list, so leaving it
    // empty is fine for JS-defined steps.
    StepRegistration::new(name, accepts, vec![], step_handler, 0)
}

/// Body of a JS-callback step invocation. Lifted out of the closure inside
/// [`step_registration_from_js`] to keep that function under clippy's
/// `too_many_lines` threshold and to make the step lifecycle (marshal →
/// invoke → await → marshal back) easier to follow at a glance.
async fn dispatch_js_step(
    handler: Arc<SendFn>,
    step_name: String,
    workflow_name: String,
    event: Box<dyn AnyEvent>,
    ctx: Context,
) -> Result<StepOutput, WorkflowError> {
    // 1. Marshal the event into a JS value.
    let event_js = marshal_event_to_js(&*event)?;

    // 2. Wrap the live context in a WasmContext and convert to JS.
    let wasm_ctx = crate::context::WasmContext::from_inner(ctx, workflow_name);
    let ctx_js = JsValue::from(wasm_ctx);

    // 3. Invoke the JS handler with (event, ctx).
    let result_js = handler
        .0
        .call2(&JsValue::NULL, &event_js, &ctx_js)
        .map_err(|e| {
            WorkflowError::Context(format!("JS step handler '{step_name}' threw: {e:?}"))
        })?;

    // 4. If the handler returned a Promise, await it.
    let resolved_js = if result_js.has_type::<js_sys::Promise>() {
        let promise: js_sys::Promise = result_js.unchecked_into();
        wasm_bindgen_futures::JsFuture::from(promise)
            .await
            .map_err(|e| {
                WorkflowError::Context(format!(
                    "JS step handler '{step_name}' rejected: {e:?}"
                ))
            })?
    } else {
        result_js
    };

    // 5. Marshal the resolved value back into an event for the engine.
    // `null` / `undefined` means "no further events" — the engine treats
    // `StepOutput::None` as a terminal step, matching the documented JS
    // contract on `addStep`.
    if resolved_js.is_null() || resolved_js.is_undefined() {
        return Ok(StepOutput::None);
    }

    let event_obj: serde_json::Value = serde_wasm_bindgen::from_value(resolved_js).map_err(|e| {
        WorkflowError::Context(format!(
            "JS step handler '{step_name}' return value not JSON-serializable: {e}"
        ))
    })?;

    let event_type = event_obj
        .get("type")
        .and_then(|v| v.as_str())
        .ok_or_else(|| {
            WorkflowError::Context(format!(
                "JS step handler '{step_name}' return value missing 'type' field"
            ))
        })?
        .to_owned();

    // Special-case StopEvent so the engine's terminal-event detection
    // recognises it. The canonical native StopEvent carries a single
    // `result` field, so route the JS object's `result` (or `data`, for
    // symmetry with the marshal-back path) into it.
    if event_type == "StopEvent" {
        let stop_payload = event_obj
            .get("result")
            .or_else(|| event_obj.get("data"))
            .cloned()
            .unwrap_or(serde_json::Value::Null);
        let stop_evt = StopEvent {
            result: stop_payload,
        };
        return Ok(StepOutput::Single(Box::new(stop_evt)));
    }

    // For all other event types the engine routes via interned
    // `event_type_id`, so a `DynamicEvent` is the right carrier.
    let dynamic = DynamicEvent {
        event_type,
        data: event_obj,
    };
    Ok(StepOutput::Single(Box::new(dynamic)))
}
