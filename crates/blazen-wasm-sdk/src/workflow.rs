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

use std::cell::RefCell;
use std::collections::HashMap;
use std::pin::Pin;
use std::sync::Arc;

use blazen_core::{
    Context, SERIALIZED_SESSION_REFS_META_KEY, SessionPausePolicy, SessionRefDeserializerFn,
    SessionRefError, SessionRefSerializable, StepFn, StepOutput, StepRegistration, Workflow,
    WorkflowBuilder, WorkflowError, WorkflowSnapshot,
};
use blazen_events::{AnyEvent, DynamicEvent, StopEvent};
use futures_util::StreamExt;
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
    /// Whether the caller customised timeout via [`WasmWorkflowBuilder`].
    /// `true` means `run()`/`runHandler()` must NOT clobber the configured
    /// timeout with a hardcoded `no_timeout()` call. `false` (the default
    /// for the direct `new()` constructor) preserves the legacy behavior
    /// where every invocation runs without a timeout.
    timeout_user_configured: bool,
    /// Session-ref pause policy. `None` leaves the engine default
    /// (`PickleOrError`) in place; `Some(policy)` applies the chosen
    /// policy on `run`/`runHandler`/`build_workflow`. Mirrors the
    /// `setSessionPausePolicy` knob on the Node binding.
    session_pause_policy: Option<SessionPausePolicy>,
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
            timeout_user_configured: false,
            session_pause_policy: None,
        }
    }

    /// Construct a fluent [`WasmWorkflowBuilder`] for the given workflow
    /// name.
    ///
    /// Equivalent to constructing a `Workflow` directly and calling
    /// `addStep` repeatedly, but returns a chainable builder so JS callers
    /// can configure timeout / auto-publish / step set up-front before
    /// producing the `Workflow`. Mirrors
    /// [`blazen_core::WorkflowBuilder`].
    ///
    /// ```js
    /// const wf = Workflow.builder('my-flow')
    ///   .addStep('process', ['StartEvent'], (event, ctx) => ({
    ///     type: 'StopEvent',
    ///     result: event.data,
    ///   }))
    ///   .noTimeout()
    ///   .build();
    /// const result = await wf.run({ data: 'hello' });
    /// ```
    #[wasm_bindgen(js_name = "builder")]
    #[must_use]
    pub fn builder(name: String) -> WasmWorkflowBuilder {
        WasmWorkflowBuilder::new_internal(name)
    }

    /// The workflow name.
    #[wasm_bindgen(getter)]
    #[must_use]
    pub fn name(&self) -> String {
        self.name.clone()
    }

    /// Register a sub-workflow step that delegates to a pre-built
    /// [`crate::subworkflow_step::WasmSubWorkflowStep`].
    ///
    /// Mirrors `blazen_core::WorkflowBuilder::add_subworkflow_step`. The
    /// sub-workflow's child workflow is captured by the wrapper at its
    /// construction time, so this method just hands a finalised
    /// [`SubWorkflowStep`] to the underlying builder.
    ///
    /// # Errors
    ///
    /// Returns a `JsValue` error if the workflow has already been run.
    #[wasm_bindgen(js_name = "addSubworkflowStep")]
    pub fn add_subworkflow_step(
        &mut self,
        step: &crate::subworkflow_step::WasmSubWorkflowStep,
    ) -> Result<(), JsValue> {
        let builder = self
            .builder
            .take()
            .ok_or_else(|| JsValue::from_str("Workflow already run; reuse not supported"))?;
        self.builder = Some(builder.add_subworkflow_step(step.to_core()));
        Ok(())
    }

    /// Register a parallel-sub-workflows fan-out step.
    ///
    /// Mirrors `blazen_core::WorkflowBuilder::add_parallel_subworkflows`.
    ///
    /// # Errors
    ///
    /// Returns a `JsValue` error if the workflow has already been run.
    #[wasm_bindgen(js_name = "addParallelSubworkflows")]
    pub fn add_parallel_subworkflows(
        &mut self,
        step: &crate::subworkflow_step::WasmParallelSubWorkflowsStep,
    ) -> Result<(), JsValue> {
        let builder = self
            .builder
            .take()
            .ok_or_else(|| JsValue::from_str("Workflow already run; reuse not supported"))?;
        self.builder = Some(builder.add_parallel_subworkflows(step.to_core()));
        Ok(())
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

    /// Configure how live session refs are treated when this workflow is
    /// paused or snapshotted. Mirrors the Node binding's
    /// `setSessionPausePolicy`.
    ///
    /// `policy` must be one of: `"pickle_or_error"` (default),
    /// `"pickle_or_serialize"`, `"warn_drop"`, `"hard_error"`. The
    /// `PascalCase` spellings used by the Node binding's enum
    /// (`PickleOrError`, `PickleOrSerialize`, `WarnDrop`, `HardError`)
    /// are also accepted.
    ///
    /// The policy is applied to the engine builder when the workflow is
    /// dispatched via `run` / `runHandler` / `runStreaming` /
    /// `runWithHandler` / `resumeFromSnapshot` /
    /// `resumeWithSerializableRefs`.
    ///
    /// # Errors
    ///
    /// Returns a `JsValue` error if `policy` is not one of the
    /// supported strings.
    #[wasm_bindgen(js_name = "setSessionPausePolicy")]
    pub fn set_session_pause_policy(&mut self, policy: &str) -> Result<(), JsValue> {
        self.session_pause_policy = Some(parse_session_pause_policy(policy)?);
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

        // Preserve the legacy "no timeout by default" behavior for the
        // direct `new()` constructor, but respect the timeout configured
        // on a `WasmWorkflowBuilder` when the user opted into one.
        if !self.timeout_user_configured {
            builder = builder.no_timeout();
        }

        if let Some(policy) = self.session_pause_policy {
            builder = builder.session_pause_policy(policy);
        }

        let workflow = builder
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

        // Required for workerd: give the spawn_local'd event-loop future a
        // chance to advance before parking on the result oneshot. Without
        // this, the second await sits inside the same wasm-bindgen export
        // call and workerd never re-enters the JS event loop, so the loop
        // never gets polled and the request hits "code had hung" timeout.
        // See `crate::handler::yield_to_js` for the full explanation.
        crate::handler::yield_to_js().await;

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

        // Same legacy-preserving timeout policy as `run()`: the direct
        // `new()` constructor disables the timeout to match the prior
        // behavior, while a `WasmWorkflowBuilder`-configured timeout is
        // honored when set.
        if !self.timeout_user_configured {
            builder = builder.no_timeout();
        }

        if let Some(policy) = self.session_pause_policy {
            builder = builder.session_pause_policy(policy);
        }

        let workflow = builder
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

        let snapshot: WorkflowSnapshot = serde_wasm_bindgen::from_value(snapshot_js)
            .map_err(|e| JsValue::from_str(&format!("snapshot deserialize failed: {e}")))?;

        let workflow_name = self.name.clone();

        let pending = std::mem::take(&mut self.pending_steps);
        let mut steps = Vec::with_capacity(pending.len());
        for pending_step in pending {
            steps.push(step_registration_from_js(
                pending_step,
                workflow_name.clone(),
            ));
        }

        let handler = Workflow::resume(snapshot, steps, None)
            .await
            .map_err(|e| JsValue::from_str(&format!("workflow resume failed: {e}")))?;

        Ok(crate::handler::WasmWorkflowHandler::new(handler))
    }

    /// Run the workflow and forward each event published by the engine
    /// to a JS callback as it occurs, resolving with the terminal
    /// payload once the workflow completes.
    ///
    /// Mirrors the Node binding's `runStreaming(input, onEvent)`. The
    /// callback is invoked with a single `{ event_type, data }` JS
    /// object per event. Stream events are subscribed *before* the
    /// engine begins dispatching, so no events are missed between
    /// dispatch and subscription.
    ///
    /// Like `run`/`runHandler`, this method consumes the workflow's
    /// internal builder; calling any `run*` method on the same
    /// instance afterwards returns the standard "already run" error.
    ///
    /// # Errors
    ///
    /// Returns a `JsValue` error under the same conditions as
    /// [`run`](Self::run): the workflow was already run, the input is
    /// not JSON-serialisable, the build failed, or the engine rejected
    /// the initial dispatch. Errors raised synchronously by the JS
    /// callback are swallowed so a misbehaving listener does not abort
    /// the run; the run still resolves with the terminal payload.
    #[wasm_bindgen(js_name = "runStreaming")]
    pub async fn run_streaming(
        &mut self,
        input: JsValue,
        callback: js_sys::Function,
    ) -> Result<JsValue, JsValue> {
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

        if !self.timeout_user_configured {
            builder = builder.no_timeout();
        }
        if let Some(policy) = self.session_pause_policy {
            builder = builder.session_pause_policy(policy);
        }

        let workflow = builder
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

        // Subscribe to the broadcast stream before kicking off the
        // forwarding task so no events emitted between dispatch and
        // subscription are dropped.
        let mut stream = handler.stream_events();

        // Spawn a single-threaded local task to drain the stream into
        // the JS callback. wasm32 has only one OS thread, so
        // `spawn_local` is the correct primitive here. The task
        // captures the callback and runs concurrently with the result
        // future below.
        wasm_bindgen_futures::spawn_local(async move {
            while let Some(event) = stream.next().await {
                // Match the Node binding's stream-end sentinel so JS
                // listeners stop receiving events at the same point
                // across bindings.
                if event.event_type_id() == "blazen::StreamEnd" {
                    break;
                }
                let envelope = serde_json::json!({
                    "event_type": event.event_type_id(),
                    "data": event.to_json(),
                });
                let serializer =
                    serde_wasm_bindgen::Serializer::new().serialize_maps_as_objects(true);
                let Ok(event_js) = envelope.serialize(&serializer) else {
                    continue;
                };
                // Fire-and-forget. Errors raised synchronously by the
                // JS callback are intentionally ignored so a buggy
                // listener can't abort the run.
                let _ = callback.call1(&JsValue::NULL, &event_js);
            }
        });

        // Same workerd yield pattern as `run()`: the spawn_local'd loop
        // (and the forwarding task above) need a JS event-loop turn
        // before this fn parks on the result oneshot. See
        // `crate::handler::yield_to_js` for the full rationale.
        crate::handler::yield_to_js().await;

        let result = handler
            .result()
            .await
            .map_err(|e| JsValue::from_str(&format!("workflow result failed: {e}")))?;

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
    /// [`WasmWorkflowHandler`] without awaiting the terminal event.
    ///
    /// Functionally equivalent to [`run_handler`](Self::run_handler);
    /// exposed under the JS name `runWithHandler` for naming parity
    /// with the Node binding's `runWithHandler` entry point. Use
    /// either method interchangeably.
    ///
    /// # Errors
    ///
    /// Returns a `JsValue` error under the same conditions as
    /// [`run`](Self::run): the workflow was already run, the input is
    /// not JSON-serialisable, the build failed, or the engine
    /// rejected the initial dispatch.
    #[wasm_bindgen(js_name = "runWithHandler")]
    pub async fn run_with_handler(
        &mut self,
        input: JsValue,
    ) -> Result<crate::handler::WasmWorkflowHandler, JsValue> {
        self.run_handler(input).await
    }

    /// Resume a workflow from a snapshot whose
    /// `__blazen_serialized_session_refs` sidecar carries
    /// JS-serialized session refs, optionally invoking a per-tag JS
    /// deserializer callback on the captured bytes before resuming.
    ///
    /// `deserializers` is an object mapping `type_tag` strings to
    /// `(bytes: Uint8Array) => unknown` JS functions. For every entry
    /// in the snapshot's serialized-refs sidecar whose `type_tag`
    /// appears in the object, the corresponding callback is invoked
    /// synchronously with the captured bytes; the callback's return
    /// value is ignored (callbacks should populate any application
    /// state they need to expose to step handlers themselves). The
    /// snapshot's bytes are then handed to the engine's
    /// [`Workflow::resume_with_deserializers`] under a built-in
    /// trampoline that re-wraps each payload as opaque bytes — so JS
    /// callers can additionally retrieve them via
    /// `ctx.getSessionRefSerializable(key)` from inside step handlers
    /// after resume, mirroring the Node binding's resume path.
    ///
    /// Snapshots that do not contain any serialized session refs work
    /// fine with the plain [`resume_from_snapshot`](Self::resume_from_snapshot)
    /// entry point; this method is only required when the original
    /// pause used [`SessionPausePolicy::PickleOrSerialize`].
    ///
    /// Like the other run/resume entry points, this method consumes
    /// the workflow's internal builder; reuse returns an error.
    ///
    /// # Errors
    ///
    /// Returns a `JsValue` error if:
    ///
    /// - the workflow was already run or resumed,
    /// - the snapshot can't be deserialised from JS,
    /// - `deserializers` is provided but is not a JS object,
    /// - a value in `deserializers` is not callable,
    /// - the step set rejects rebuild (e.g. duplicate names),
    /// - the engine fails to rehydrate the registry from the
    ///   serialized-refs sidecar.
    #[wasm_bindgen(js_name = "resumeWithSerializableRefs")]
    pub async fn resume_with_serializable_refs(
        &mut self,
        snapshot_js: JsValue,
        deserializers: JsValue,
    ) -> Result<crate::handler::WasmWorkflowHandler, JsValue> {
        let _builder = self
            .builder
            .take()
            .ok_or_else(|| JsValue::from_str("Workflow already run; reuse not supported"))?;

        let snapshot: WorkflowSnapshot = serde_wasm_bindgen::from_value(snapshot_js)
            .map_err(|e| JsValue::from_str(&format!("snapshot deserialize failed: {e}")))?;

        // Parse the user-provided JS deserializer map (if any) into a
        // tag -> Function lookup. `undefined` / `null` is treated as
        // an empty map so callers can omit the second argument from
        // JS.
        let user_callbacks: HashMap<String, js_sys::Function> =
            if deserializers.is_undefined() || deserializers.is_null() {
                HashMap::new()
            } else {
                let obj = deserializers
                    .dyn_ref::<js_sys::Object>()
                    .ok_or_else(|| JsValue::from_str("deserializers must be a JS object"))?;
                let mut map = HashMap::new();
                let entries = js_sys::Object::entries(obj);
                for i in 0..entries.length() {
                    let pair: js_sys::Array = entries.get(i).unchecked_into();
                    let key = pair
                        .get(0)
                        .as_string()
                        .ok_or_else(|| JsValue::from_str("deserializer key must be a string"))?;
                    let value = pair.get(1);
                    let func: js_sys::Function = value.dyn_into().map_err(|_| {
                        JsValue::from_str(&format!("deserializer for '{key}' must be a function"))
                    })?;
                    map.insert(key, func);
                }
                map
            };

        // Walk the snapshot's serialized-refs sidecar to discover
        // every tag we need to register a trampoline for, and to
        // eagerly invoke each user-provided JS deserializer with the
        // raw bytes captured in the snapshot. This gives JS callers a
        // chance to do app-level rehydration (e.g. populating an
        // external cache) before the engine takes over.
        let mut tags_seen: Vec<String> = Vec::new();
        if let Some(serde_json::Value::Object(entries)) =
            snapshot.metadata.get(SERIALIZED_SESSION_REFS_META_KEY)
        {
            for record in entries.values() {
                let Some(type_tag) = record.get("type_tag").and_then(serde_json::Value::as_str)
                else {
                    continue;
                };
                if !tags_seen.iter().any(|t| t == type_tag) {
                    tags_seen.push(type_tag.to_owned());
                }
                // If the JS caller supplied a deserializer for this
                // tag, fire it with the captured bytes. The bytes
                // live under either `bytes` (raw byte array) or
                // `payload` (snake_case mirror of the trait wire
                // format) depending on how the snapshot was
                // produced; we accept both for robustness.
                if let Some(callback) = user_callbacks.get(type_tag) {
                    let raw = record
                        .get("bytes")
                        .or_else(|| record.get("payload"))
                        .cloned()
                        .unwrap_or(serde_json::Value::Null);
                    let serializer =
                        serde_wasm_bindgen::Serializer::new().serialize_maps_as_objects(true);
                    if let Ok(bytes_js) = raw.serialize(&serializer) {
                        // Errors raised synchronously by the JS
                        // callback are swallowed: the resume still
                        // proceeds because the engine's own
                        // trampoline re-wraps the bytes regardless,
                        // and surfacing partial JS-side failures
                        // here would make resume non-atomic.
                        let _ = callback.call1(&JsValue::NULL, &bytes_js);
                    }
                }
            }
        }

        // Register the built-in trampoline for every observed tag so
        // the core resume path can repopulate the registry. This
        // mirrors the Node binding's strategy: the trampoline wraps
        // the raw bytes in an opaque `WasmSessionRefSerializable`
        // adapter; JS step handlers can pull them back out via
        // `ctx.getSessionRefSerializable(key)`.
        let mut core_deserializers: HashMap<&'static str, SessionRefDeserializerFn> =
            HashMap::new();
        for tag in &tags_seen {
            let interned: &'static str = Box::leak(tag.clone().into_boxed_str());
            core_deserializers.insert(interned, WASM_SESSION_REF_DESERIALIZER);
        }

        let workflow_name = self.name.clone();

        let pending = std::mem::take(&mut self.pending_steps);
        let mut steps = Vec::with_capacity(pending.len());
        for pending_step in pending {
            steps.push(step_registration_from_js(
                pending_step,
                workflow_name.clone(),
            ));
        }

        let handler =
            Workflow::resume_with_deserializers(snapshot, steps, core_deserializers, None)
                .await
                .map_err(|e| JsValue::from_str(&format!("workflow resume failed: {e}")))?;

        // Hold the JS callback map alive for the duration of the
        // resumed run so any captured closures stay valid. Stash it
        // in a thread-local so the handler keeps it alive without
        // requiring `WasmWorkflowHandler` to grow a new field.
        ACTIVE_RESUME_CALLBACKS.with(|cell| {
            cell.borrow_mut().push(user_callbacks);
        });

        Ok(crate::handler::WasmWorkflowHandler::new(handler))
    }
}

impl WasmWorkflow {
    /// Materialize the buffered builder + pending JS steps into a real
    /// [`blazen_core::Workflow`] without running it.
    ///
    /// Used by the pipeline binding to embed a `WasmWorkflow` inside a
    /// [`blazen_pipeline::Stage`]. Like `run()` / `runHandler()`, this
    /// consumes the workflow's internal builder; calling `run()` on the
    /// same instance afterwards returns the standard "already run" error.
    ///
    /// # Errors
    ///
    /// Returns a `JsValue` error if:
    /// - the workflow was already consumed (run/runHandler/build_workflow),
    /// - the underlying [`WorkflowBuilder::build`] fails (e.g. duplicate step
    ///   names).
    pub(crate) fn build_workflow(&mut self) -> Result<Workflow, JsValue> {
        let mut builder = self
            .builder
            .take()
            .ok_or_else(|| JsValue::from_str("Workflow already consumed; reuse not supported"))?;

        let workflow_name = self.name.clone();

        let pending = std::mem::take(&mut self.pending_steps);
        for pending_step in pending {
            let registration = step_registration_from_js(pending_step, workflow_name.clone());
            builder = builder.step(registration);
        }

        if !self.timeout_user_configured {
            builder = builder.no_timeout();
        }

        if let Some(policy) = self.session_pause_policy {
            builder = builder.session_pause_policy(policy);
        }

        builder
            .build()
            .map_err(|e| JsValue::from_str(&format!("workflow build failed: {e}")))
    }
}

// ---------------------------------------------------------------------------
// WasmWorkflowBuilder — fluent JS surface mirroring blazen_core::WorkflowBuilder
// ---------------------------------------------------------------------------

/// Fluent JS-facing builder mirroring [`blazen_core::WorkflowBuilder`].
///
/// Buffers pending steps and configuration knobs until [`build`](Self::build)
/// is called, at which point the configuration is applied to a real
/// [`WorkflowBuilder`] and a [`WasmWorkflow`] is produced.
///
/// All chainable methods consume `self` and return `Self` so JS callers can
/// fluently chain configuration without intermediate variables.
#[wasm_bindgen(js_name = "WorkflowBuilder")]
pub struct WasmWorkflowBuilder {
    /// Workflow name. Cached separately so [`name`](Self::name) stays cheap
    /// and survives [`build`](Self::build).
    name: String,
    /// Buffered step registrations. Drained on [`build`](Self::build) into
    /// the produced [`WasmWorkflow`]'s pending-step list, so the same
    /// JS-callback adapter as the direct `Workflow::addStep` path is reused.
    pending_steps: Vec<PendingStep>,
    /// Optional execution timeout. `None` means use the
    /// [`WorkflowBuilder`] default; `Some(None)` means disable the timeout
    /// entirely (mirrors [`WorkflowBuilder::no_timeout`]); `Some(Some(d))`
    /// means apply [`WorkflowBuilder::timeout`].
    timeout: Option<Option<std::time::Duration>>,
    /// Whether to enable automatic broadcast publishing (mirrors
    /// [`WorkflowBuilder::auto_publish_events`]). `None` leaves the engine
    /// default in place.
    auto_publish_events: Option<bool>,
    /// Session-ref pause policy. Mirrors
    /// [`WorkflowBuilder::session_pause_policy`]. `None` leaves the
    /// engine default (`PickleOrError`) in place.
    session_pause_policy: Option<SessionPausePolicy>,
}

impl WasmWorkflowBuilder {
    /// Internal constructor used by [`WasmWorkflow::builder`].
    fn new_internal(name: String) -> Self {
        Self {
            name,
            pending_steps: Vec::new(),
            timeout: None,
            auto_publish_events: None,
            session_pause_policy: None,
        }
    }
}

#[wasm_bindgen(js_class = "WorkflowBuilder")]
impl WasmWorkflowBuilder {
    /// Construct a fresh builder with the given workflow name.
    #[wasm_bindgen(constructor)]
    #[must_use]
    pub fn new(name: String) -> WasmWorkflowBuilder {
        Self::new_internal(name)
    }

    /// The workflow name.
    #[wasm_bindgen(getter)]
    #[must_use]
    pub fn name(&self) -> String {
        self.name.clone()
    }

    /// Register a step handler. Mirrors [`WasmWorkflow::add_step`].
    ///
    /// Returns the builder so the call can be chained:
    ///
    /// ```js
    /// Workflow.builder('flow')
    ///   .addStep('a', ['StartEvent'], handler1)
    ///   .addStep('b', ['MidEvent'],   handler2)
    ///   .build();
    /// ```
    ///
    /// # Errors
    ///
    /// Returns a `JsValue` error if `event_types` is not an array of
    /// strings.
    #[wasm_bindgen(js_name = "addStep")]
    pub fn add_step(
        mut self,
        name: &str,
        event_types: JsValue,
        handler: js_sys::Function,
    ) -> Result<WasmWorkflowBuilder, JsValue> {
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

        Ok(self)
    }

    /// Set the maximum execution time for the workflow, in milliseconds.
    ///
    /// Mirrors [`WorkflowBuilder::timeout`].
    #[wasm_bindgen(js_name = "timeoutMs")]
    #[must_use]
    pub fn timeout_ms(mut self, ms: u64) -> WasmWorkflowBuilder {
        self.timeout = Some(Some(std::time::Duration::from_millis(ms)));
        self
    }

    /// Disable the execution timeout (workflow runs until `StopEvent`).
    ///
    /// Mirrors [`WorkflowBuilder::no_timeout`].
    #[wasm_bindgen(js_name = "noTimeout")]
    #[must_use]
    pub fn no_timeout(mut self) -> WasmWorkflowBuilder {
        self.timeout = Some(None);
        self
    }

    /// Enable or disable automatic publishing of lifecycle events to the
    /// broadcast stream.
    ///
    /// Mirrors [`WorkflowBuilder::auto_publish_events`].
    #[wasm_bindgen(js_name = "autoPublishEvents")]
    #[must_use]
    pub fn auto_publish_events(mut self, enabled: bool) -> WasmWorkflowBuilder {
        self.auto_publish_events = Some(enabled);
        self
    }

    /// Configure how live session refs are treated when the workflow is
    /// paused or snapshotted. Mirrors the Node binding's
    /// `setSessionPausePolicy`.
    ///
    /// `policy` must be one of: `"pickle_or_error"` (default),
    /// `"pickle_or_serialize"`, `"warn_drop"`, `"hard_error"`. The
    /// `PickleOrError` / `PickleOrSerialize` / `WarnDrop` /
    /// `HardError` `PascalCase` spellings used by the Node binding's
    /// enum are also accepted.
    ///
    /// # Errors
    ///
    /// Returns a `JsValue` error if `policy` is not one of the
    /// supported strings.
    #[wasm_bindgen(js_name = "setSessionPausePolicy")]
    pub fn set_session_pause_policy(
        mut self,
        policy: &str,
    ) -> Result<WasmWorkflowBuilder, JsValue> {
        self.session_pause_policy = Some(parse_session_pause_policy(policy)?);
        Ok(self)
    }

    /// Finalise the builder and return a [`WasmWorkflow`] ready to `run`.
    ///
    /// Applies the buffered configuration to a real
    /// [`blazen_core::WorkflowBuilder`] and stuffs the pending steps into
    /// the returned `WasmWorkflow` so the existing JS-callback adapter
    /// inside `run` / `runHandler` is reused unchanged.
    ///
    /// # Errors
    ///
    /// Returns a `JsValue` error if the underlying engine builder rejects
    /// the configuration (currently only validation that a [`WasmWorkflow`]
    /// has at least one step, which is deferred until `run`).
    #[wasm_bindgen(js_name = "build")]
    pub fn build(self) -> Result<WasmWorkflow, JsValue> {
        let Self {
            name,
            pending_steps,
            timeout,
            auto_publish_events,
            session_pause_policy,
        } = self;

        let timeout_user_configured = timeout.is_some();
        let mut core_builder = WorkflowBuilder::new(&name);
        match timeout {
            Some(Some(d)) => {
                core_builder = core_builder.timeout(d);
            }
            Some(None) => {
                core_builder = core_builder.no_timeout();
            }
            None => {}
        }
        if let Some(enabled) = auto_publish_events {
            core_builder = core_builder.auto_publish_events(enabled);
        }
        if let Some(policy) = session_pause_policy {
            core_builder = core_builder.session_pause_policy(policy);
        }

        Ok(WasmWorkflow {
            name,
            builder: Some(core_builder),
            pending_steps,
            timeout_user_configured,
            session_pause_policy,
        })
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

/// Parse a JS-supplied policy string into a [`SessionPausePolicy`].
///
/// Accepts the canonical `snake_case` names matching the Rust serde
/// representation (`pickle_or_error`, `pickle_or_serialize`, `warn_drop`,
/// `hard_error`) as well as the `camelCase` / `PascalCase` aliases used
/// by the Node binding (`PickleOrError`, `PickleOrSerialize`, `WarnDrop`,
/// `HardError`) so JS callers can use whichever spelling matches the
/// surrounding code style.
fn parse_session_pause_policy(s: &str) -> Result<SessionPausePolicy, JsValue> {
    match s {
        "pickle_or_error" | "PickleOrError" | "pickleOrError" => {
            Ok(SessionPausePolicy::PickleOrError)
        }
        "pickle_or_serialize" | "PickleOrSerialize" | "pickleOrSerialize" => {
            Ok(SessionPausePolicy::PickleOrSerialize)
        }
        "warn_drop" | "WarnDrop" | "warnDrop" => Ok(SessionPausePolicy::WarnDrop),
        "hard_error" | "HardError" | "hardError" => Ok(SessionPausePolicy::HardError),
        other => Err(JsValue::from_str(&format!(
            "unknown SessionPausePolicy '{other}' (expected one of: pickle_or_error, \
             pickle_or_serialize, warn_drop, hard_error)"
        ))),
    }
}

/// Convert a buffered [`PendingStep`] into a real
/// [`blazen_core::StepRegistration`] whose handler dispatches into the
/// supplied JS callback.
///
/// The returned registration does the following on each invocation:
///
/// 1. Marshals the type-erased event into a `JsValue` via [`marshal_event_to_js`].
/// 2. Wraps the live [`Context`] in a [`WasmContext`] (exported to JS as
///    `Context`, with a deprecated `WorkflowContext` type alias for
///    backward compatibility) and converts it to a `JsValue`.
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
fn step_registration_from_js(pending_step: PendingStep, workflow_name: String) -> StepRegistration {
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
                WorkflowError::Context(format!("JS step handler '{step_name}' rejected: {e:?}"))
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

    let event_obj: serde_json::Value =
        serde_wasm_bindgen::from_value(resolved_js).map_err(|e| {
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

// ---------------------------------------------------------------------------
// Serializable session-ref support for `resume_with_serializable_refs`
// ---------------------------------------------------------------------------

thread_local! {
    /// JS-supplied deserializer callbacks pinned alive for the
    /// lifetime of the wasm module. Each call to
    /// [`WasmWorkflow::resume_with_serializable_refs`] appends its
    /// callback map here so any JS closures captured by the user
    /// remain valid across awaited engine progress. The vector is
    /// never drained — the wasm module's lifetime bounds the total
    /// size, and the cost (one `js_sys::Function` per registered tag
    /// per resume) is proportional to the number of resumes performed
    /// in a session.
    static ACTIVE_RESUME_CALLBACKS: RefCell<Vec<HashMap<String, js_sys::Function>>> =
        const { RefCell::new(Vec::new()) };
}

/// Adapter that satisfies [`SessionRefSerializable`] by holding the
/// raw bytes captured in a snapshot's serialized-refs sidecar
/// alongside the originating type tag.
///
/// Identical in shape to the Node binding's
/// `NodeSessionRefSerializable`: the serialized wire format is
/// `[4-byte BE tag_len][tag bytes][user bytes]` so the deserializer
/// trampoline can recover both halves from the bytes alone.
struct WasmSessionRefSerializable {
    type_tag: &'static str,
    user_bytes: Vec<u8>,
}

// SAFETY: wasm32 is single-threaded — `Send`/`Sync` are required by
// the core `SessionRefRegistry` map's `Arc<dyn Any + Send + Sync>`
// element type but are never actually exercised across threads on
// this target.
unsafe impl Send for WasmSessionRefSerializable {}
// SAFETY: see comment on the `Send` impl above.
unsafe impl Sync for WasmSessionRefSerializable {}

impl SessionRefSerializable for WasmSessionRefSerializable {
    fn blazen_serialize(&self) -> Result<Vec<u8>, SessionRefError> {
        let tag_bytes = self.type_tag.as_bytes();
        let tag_len: u32 =
            u32::try_from(tag_bytes.len()).map_err(|_| SessionRefError::SerializationFailed {
                type_tag: self.type_tag.to_owned(),
                source: Box::<dyn std::error::Error + Send + Sync>::from(
                    "type tag longer than u32::MAX bytes",
                ),
            })?;
        let mut out = Vec::with_capacity(4 + tag_bytes.len() + self.user_bytes.len());
        out.extend_from_slice(&tag_len.to_be_bytes());
        out.extend_from_slice(tag_bytes);
        out.extend_from_slice(&self.user_bytes);
        Ok(out)
    }

    fn blazen_type_tag(&self) -> &'static str {
        self.type_tag
    }
}

/// Parse the self-describing prefix produced by
/// [`WasmSessionRefSerializable::blazen_serialize`] and return
/// `(type_tag, user_bytes)` slices. Mirrors the Node binding's
/// `split_prefix` helper so both bindings agree on the wire format.
fn wasm_split_prefix(bytes: &[u8]) -> Result<(&str, &[u8]), SessionRefError> {
    if bytes.len() < 4 {
        return Err(SessionRefError::SerializationFailed {
            type_tag: "<unknown>".to_owned(),
            source: Box::<dyn std::error::Error + Send + Sync>::from(
                "payload too short to contain type tag prefix",
            ),
        });
    }
    let mut tag_len_bytes = [0_u8; 4];
    tag_len_bytes.copy_from_slice(&bytes[..4]);
    let tag_len = u32::from_be_bytes(tag_len_bytes) as usize;
    if bytes.len() < 4 + tag_len {
        return Err(SessionRefError::SerializationFailed {
            type_tag: "<unknown>".to_owned(),
            source: Box::<dyn std::error::Error + Send + Sync>::from(
                "payload shorter than declared type tag length",
            ),
        });
    }
    let tag = std::str::from_utf8(&bytes[4..4 + tag_len]).map_err(|e| {
        SessionRefError::SerializationFailed {
            type_tag: "<unknown>".to_owned(),
            source: Box::new(e),
        }
    })?;
    Ok((tag, &bytes[4 + tag_len..]))
}

/// Static-fn-pointer trampoline registered with the core resume
/// path. Re-wraps the bytes captured in the snapshot's
/// serialized-refs sidecar in a fresh
/// [`WasmSessionRefSerializable`] so the engine can repopulate the
/// session-ref registry; JS step handlers retrieve the raw bytes
/// afterwards via `ctx.getSessionRefSerializable(key)`.
fn wasm_session_ref_deserializer(
    bytes: &[u8],
) -> Result<Arc<dyn SessionRefSerializable>, SessionRefError> {
    let (type_tag, user_bytes) = wasm_split_prefix(bytes)?;
    let interned: &'static str = Box::leak(type_tag.to_owned().into_boxed_str());
    Ok(Arc::new(WasmSessionRefSerializable {
        type_tag: interned,
        user_bytes: user_bytes.to_vec(),
    }))
}

/// Static reference to [`wasm_session_ref_deserializer`] coerced to
/// the core [`SessionRefDeserializerFn`] alias. Re-used by every tag
/// observed in a `resumeWithSerializableRefs` call.
const WASM_SESSION_REF_DESERIALIZER: SessionRefDeserializerFn = wasm_session_ref_deserializer;
