//! Live-`JsValue` identity passthrough for the WASM workflow bridge.
//!
//! # Why this module exists
//!
//! When a step handler returns an event, the event crosses the Rust↔JS
//! boundary on its way to the *next* step. The naive bridge marshals the
//! return value to a [`serde_json::Value`] and rebuilds a
//! [`DynamicEvent`](blazen_events::DynamicEvent) with
//! [`DynamicEvent::from_json`] — which **destroys JS object identity**. A
//! class instance, a closure, or any object carrying non-JSON state (a
//! `Map`, a `WeakRef`, a DOM node, a model handle) returned by step *A*
//! arrives at step *B* as a fresh, structurally-equal-but-not-identical
//! plain object, and any non-serialisable members are silently dropped.
//!
//! [`DynamicEvent::with_native`] exists precisely to keep an event live
//! step-to-step, but its contract requires the native handle to be
//! `Send + Sync + 'static`. A raw [`JsValue`] is **neither `Send` nor
//! `Sync`**, and `blazen-events` explicitly forbids stuffing a `!Send`
//! host object into `with_native` (see the doc on
//! [`DynamicEvent::with_native`]). The mandated route for `!Send` host
//! objects is *store id-indirection*: keep the live object on the JS heap
//! in a thread-local store keyed by a small integer id, and let the
//! `Send`-able [`JsHandle`] carry only that id across the (single-threaded,
//! never-actually-cross-thread) `Arc<dyn Any + Send + Sync>` boundary.
//!
//! # How it works
//!
//! 1. **Outbound** (a JS step returns a value): [`stash_js_event`] stores
//!    the live [`JsValue`] in [`JS_VALUE_STORE`] under a fresh id, snapshots
//!    its JSON form once, and produces a
//!    [`DynamicEvent::with_native`]-backed event whose native handle is a
//!    [`JsHandle`] `{ id, json_snapshot }`.
//! 2. **Inbound** (the next step receives the event):
//!    [`marshal_event_to_js`] checks
//!    [`AnyEvent::native_handle`](blazen_events::AnyEvent::native_handle);
//!    if it downcasts to a [`JsHandle`] whose id is still live in the
//!    store, the **original** [`JsValue`] is handed back to the JS callback
//!    — identity survives the hop.
//! 3. **Snapshot / serialize** (a native-backed event is serialised, e.g.
//!    for a workflow snapshot or for a step that did not carry a live
//!    handle): the [`NativeSerializerFn`] registered at init by
//!    [`register`] downcasts the handle and returns its cached
//!    `json_snapshot`, so the wire payload is real data, not `null`.
//!
//! The store is a thread-local because wasm32 is strictly single-threaded;
//! ids are never observed from another thread. Entries are reaped when the
//! handle's last [`Arc`] drops (see [`JsHandle::drop`]), so the store does
//! not grow unbounded across a long-lived workflow.

use std::any::Any;
use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::Arc;

use blazen_events::{AnyEvent, DynamicEvent};
use serde::Serialize;
use wasm_bindgen::prelude::*;

thread_local! {
    /// Live `JsValue`s kept on the JS heap, keyed by the integer id carried
    /// by their [`JsHandle`]. Single-threaded (`wasm32`) so a plain
    /// `RefCell<HashMap<…>>` is sufficient — no lock is needed and no entry
    /// is ever touched from a second thread.
    static JS_VALUE_STORE: RefCell<HashMap<u64, JsValue>> =
        RefCell::new(HashMap::new());

    /// Monotonic id source for [`JS_VALUE_STORE`] keys. Wraps at `u64::MAX`,
    /// which is unreachable in any realistic page lifetime.
    static NEXT_ID: RefCell<u64> = const { RefCell::new(1) };
}

/// Allocate the next store id.
fn next_id() -> u64 {
    NEXT_ID.with(|cell| {
        let mut id = cell.borrow_mut();
        let out = *id;
        *id = id.wrapping_add(1);
        out
    })
}

/// `Send + Sync` native handle for a live `JsValue` parked in
/// [`JS_VALUE_STORE`].
///
/// Carries only the integer `id` (which *is* `Send`) plus a cheap JSON
/// snapshot taken at stash time. The actual `JsValue` never moves off the
/// JS heap — this handle is pure indirection.
///
/// SAFETY: the `unsafe impl Send`/`Sync` below are sound on `wasm32`
/// because the runtime is strictly single-threaded; no two threads ever
/// observe a `JsHandle` concurrently, and the only `!Send` data (the live
/// `JsValue`) lives in a thread-local, reached *only* by id from the
/// thread that created it.
pub(crate) struct JsHandle {
    /// Key into [`JS_VALUE_STORE`].
    id: u64,
    /// JSON snapshot of the live value, captured once at stash time. Used by
    /// the [`NativeSerializerFn`] and as the fallback when the live entry has
    /// been reaped.
    json_snapshot: serde_json::Value,
}

// SAFETY: see the doc comment on `JsHandle` — wasm32 is single-threaded and
// the live `JsValue` stays in a thread-local reached only by id.
unsafe impl Send for JsHandle {}
// SAFETY: see the `Send` impl above.
unsafe impl Sync for JsHandle {}

impl JsHandle {
    /// Pull the live `JsValue` back out of the store, if its id is still
    /// resident. Returns `None` if the entry was reaped (e.g. the event was
    /// round-tripped through a snapshot, which drops the native handle).
    pub(crate) fn live_value(&self) -> Option<JsValue> {
        JS_VALUE_STORE.with(|store| store.borrow().get(&self.id).cloned())
    }
}

impl Drop for JsHandle {
    fn drop(&mut self) {
        // Reap the live entry when the last handle referencing it drops, so
        // the store does not accumulate every event ever emitted across a
        // long-running workflow.
        JS_VALUE_STORE.with(|store| {
            store.borrow_mut().remove(&self.id);
        });
    }
}

/// Convention key under which `stash_js_event` records the originating event
/// type inside the JSON snapshot when the returned value was not an object
/// (so the snapshot can still round-trip through `from_json`). Internal only.
const SNAPSHOT_TYPE_KEY: &str = "type";

/// Stash a live JS step-return value and wrap it in a native-backed
/// [`DynamicEvent`] so its identity survives the hop to the next step.
///
/// `event_type` is the routing discriminator (the `type` field of the JS
/// object). `value` is the live `JsValue` the handler returned; it is parked
/// in [`JS_VALUE_STORE`] and a JSON snapshot is taken for the serialize /
/// snapshot lane. The returned event reports `event_type` via
/// `event_type_id()` and yields the live value back through
/// [`marshal_event_to_js`] on the inbound side.
///
/// `json_snapshot` is the already-parsed JSON form of `value` (the caller
/// has it in hand from the `type`-field probe, so we avoid re-serialising).
pub(crate) fn stash_js_event(
    event_type: String,
    value: JsValue,
    json_snapshot: serde_json::Value,
) -> DynamicEvent {
    let id = next_id();
    JS_VALUE_STORE.with(|store| {
        store.borrow_mut().insert(id, value);
    });
    let handle = Arc::new(JsHandle { id, json_snapshot });
    DynamicEvent::with_native(event_type, handle)
}

/// Stash a live JS event object, computing its (lossy) JSON snapshot
/// internally via [`json_snapshot_of`].
///
/// Convenience wrapper over [`stash_js_event`] used by `Context.sendEvent` /
/// `Context.writeEventToStream`, where the caller has the live `JsValue` but
/// not a pre-parsed snapshot. Identity is preserved the same way as a step's
/// return value: the live object is parked on the JS heap and threaded
/// through as a native-backed event.
pub(crate) fn stash_js_object(event_type: String, value: JsValue) -> DynamicEvent {
    let snapshot = json_snapshot_of(&value);
    stash_js_event(event_type, value, snapshot)
}

/// Marshal a type-erased event into a `JsValue` for a step's JS callback.
///
/// Identity-preserving: if the event carries a live [`JsHandle`] whose id is
/// still resident in [`JS_VALUE_STORE`], the **original** `JsValue` is
/// returned, so JS object identity survives the Rust↔JS hop. Otherwise the
/// event's JSON form (`to_json()`, which already consults the registered
/// native serializer for native-backed events) is marshalled to a fresh JS
/// object — the unchanged behaviour for plain JSON events.
pub(crate) fn marshal_event_to_js(event: &dyn AnyEvent) -> Result<JsValue, JsValue> {
    if let Some(native) = event.native_handle()
        && let Some(handle) = native.downcast_ref::<JsHandle>()
        && let Some(live) = handle.live_value()
    {
        return Ok(live);
    }

    let json = event.to_json();
    let serializer = serde_wasm_bindgen::Serializer::new().serialize_maps_as_objects(true);
    json.serialize(&serializer)
        .map_err(|e| JsValue::from_str(&format!("event marshal failed: {e}")))
}

/// Process-wide [`NativeSerializerFn`] implementation for the WASM binding.
///
/// Registered once at module init by [`register`]. Given the type-erased
/// native handle of a [`DynamicEvent`], downcasts it to a [`JsHandle`] and
/// returns its cached JSON snapshot so
/// [`DynamicEvent::to_json`](blazen_events::DynamicEvent) materialises real
/// data (not the placeholder `Null`) on the serialize / snapshot lane.
/// Returns `None` for any other native type, in which case the engine falls
/// back to the event's cached `data` field.
fn js_native_to_json(native: &Arc<dyn Any + Send + Sync>) -> Option<serde_json::Value> {
    native
        .downcast_ref::<JsHandle>()
        .map(|h| h.json_snapshot.clone())
}

/// Register the WASM native serializer with `blazen-events`.
///
/// Idempotent (the first registration wins, per
/// [`register_native_serializer`](blazen_events::register_native_serializer)).
/// Called from the module [`init`](crate::init) hook so the serialize /
/// snapshot lane can render JS-backed events as their JSON snapshot.
pub(crate) fn register() {
    blazen_events::register_native_serializer(js_native_to_json);
}

/// Probe a resolved JS step-return value and build the next event for the
/// engine, preserving live JS object identity via the native lane.
///
/// Shared by every JS-step dispatch path. Mirrors the previous
/// `dispatch_js_step` tail (extract `type`, special-case `StopEvent`) but
/// keeps the live `JsValue` alive through [`stash_js_event`] for the common
/// non-terminal case so the next step receives the *same* object.
///
/// Returns:
/// - `Ok(None)` when the value is `null` / `undefined` (terminal step).
/// - `Ok(Some((event_type, dynamic_event)))` for a routed event.
///
/// The caller is responsible for `StopEvent` special-casing using the
/// returned `event_type`, since the terminal carrier differs (`StopEvent`
/// vs. `DynamicEvent`) and the engine's terminal detection keys off the
/// concrete `StopEvent` type.
pub(crate) fn parse_js_return(
    value: JsValue,
) -> Result<Option<(String, serde_json::Value, DynamicEvent)>, String> {
    if value.is_null() || value.is_undefined() {
        return Ok(None);
    }

    // Read the `type` discriminator directly off the live object. This must
    // not go through JSON, because the value may carry non-JSON members
    // (functions, `Map`s, class instances) whose presence would make a
    // strict `serde_wasm_bindgen` conversion fail outright — the very
    // identity we are here to preserve.
    let event_type = js_sys::Reflect::get(&value, &JsValue::from_str(SNAPSHOT_TYPE_KEY))
        .ok()
        .and_then(|v| v.as_string())
        .ok_or_else(|| "return value missing 'type' field".to_owned())?;

    // Best-effort JSON snapshot for the serialize / snapshot / stream lane.
    // Lossy by design: it omits non-JSON members (functions etc.) exactly as
    // `JSON.stringify` does, so a live object that *carries* such members
    // still round-trips its JSON-able shape without erroring. The live value
    // retains the full identity regardless.
    let snapshot = json_snapshot_of(&value);

    let dynamic = stash_js_event(event_type.clone(), value, snapshot.clone());
    Ok(Some((event_type, snapshot, dynamic)))
}

/// Take a best-effort, lossy JSON snapshot of a live `JsValue`.
///
/// Uses `JSON.stringify` semantics (via [`js_sys::JSON::stringify`]) so
/// function-valued and otherwise non-serialisable members are dropped rather
/// than producing an error. Returns [`serde_json::Value::Null`] if the value
/// cannot be stringified at all (e.g. a circular structure) — the live
/// handle still carries the real object, and `Null` is a safe placeholder
/// for the rare value that genuinely has no JSON form.
fn json_snapshot_of(value: &JsValue) -> serde_json::Value {
    js_sys::JSON::stringify(value)
        .ok()
        .and_then(|s| s.as_string())
        .and_then(|s| serde_json::from_str(&s).ok())
        .unwrap_or(serde_json::Value::Null)
}
