//! Live-JS-object passthrough across workflow step hops.
//!
//! ## Why this exists
//!
//! By default the napi step bridge marshals a step handler's argument and
//! return value through [`serde_json::Value`]. That flattens a JS object
//! into plain JSON on the way out of the handler and rebuilds a *fresh*
//! object on the way into the next handler — so object identity, class
//! prototype, methods, and any non-JSON-serializable fields are lost
//! between hops. The Python binding avoids this with
//! [`DynamicEvent::with_native`](blazen_events::DynamicEvent::with_native),
//! which keeps a live `Py<PyAny>` attached to the event. That constructor
//! is explicitly **forbidden** for JS values, because napi-rs's JS
//! references are not safe to materialize off the v8 main thread (see the
//! doc-comment on `with_native`). Node therefore takes the
//! registry-marker route the core mandates.
//!
//! ## How it works
//!
//! When a JS handler returns an event object, [`StepEventReturn`] (the
//! `FromNapiValue` return type of the step `ThreadsafeFunction`) runs on
//! the v8 main thread and:
//!
//! 1. creates a napi *reference* ([`JsNativeRef`]) to the returned object,
//!    pinning it on the JS heap so v8 will not collect it;
//! 2. also projects the object to a [`serde_json::Value`] so routing by
//!    `type` and snapshot serialization keep working.
//!
//! The async step closure (`make_step_registration` in
//! [`crate::workflow::workflow`]) then inserts the [`JsNativeRef`] into the
//! run's unified [`SessionRefRegistry`] — the **same** store every other
//! session ref uses — receiving a [`RegistryKey`]. It stamps that key into
//! the event's JSON payload under [`NATIVE_REF_TAG`]. Only the id (a UUID
//! string) crosses into Rust; the JS object never leaves the JS heap.
//!
//! On the inbound side, [`StepEventArg`] (the argument type, also run on
//! the main thread) builds the JS object from JSON, but if the payload
//! carries a [`NATIVE_REF_TAG`] whose key still resolves to a live
//! [`JsNativeRef`] in the registry, it hands the handler back the **same**
//! JS object the previous step returned — full identity preservation —
//! and releases the reference (it has been consumed for this hop).
//!
//! This reuses [`SessionRefRegistry`] / the `NodeSessionRefSerializable`
//! machinery rather than introducing a parallel store, exactly as the
//! cross-binding charter requires.

use std::ptr;
use std::sync::Arc;

use blazen_core::session_ref::{RegistryKey, SessionRefRegistry};
use napi::bindgen_prelude::*;
use napi::sys;
use serde_json::Value;

/// Reserved JSON key embedded in an event payload to point at a live JS
/// object pinned in the [`SessionRefRegistry`].
///
/// Distinct from [`blazen_core::session_ref::SESSION_REF_TAG`]
/// (`__blazen_session_ref__`, used for user-stored `ctx.setObject` refs):
/// this tag specifically marks **whole-event** identity passthrough so an
/// object returned by one step is the literal same object the next step
/// receives. The value is the [`RegistryKey`] UUID string.
pub const NATIVE_REF_TAG: &str = "__blazen_native_ref__";

/// A `Send` wrapper around a napi reference (`napi_ref`) plus the
/// `napi_env` it was created in.
///
/// The underlying object lives on the v8 heap; this struct merely pins it
/// (refcount 1) so it survives until the next step consumes it. napi
/// reference operations are only valid on the env's (main) thread, which
/// is exactly where [`StepEventReturn`] creates the ref and where
/// [`StepEventArg`] resolves and releases it.
///
/// `Send + Sync` is required to live inside the cross-thread
/// [`SessionRefRegistry`]; it is sound here because the raw pointers are
/// only ever *dereferenced* (via napi calls) on the main thread — the
/// async step closure that owns the `Arc` between hops only moves it, it
/// never touches v8.
pub struct JsNativeRef {
    env: sys::napi_env,
    napi_ref: sys::napi_ref,
}

// SAFETY: the contained raw pointers are only used in napi calls that run
// on the v8 main thread (ref creation in `StepEventReturn::from_napi_value`,
// resolution + deletion in `StepEventArg::to_napi_value`). Between those
// points the value is only moved across threads inside an `Arc`, never
// dereferenced. This mirrors the soundness contract napi-patched itself
// asserts for `Ref<T>` / `UnknownRef`.
#[allow(unsafe_code)]
unsafe impl Send for JsNativeRef {}
#[allow(unsafe_code)]
unsafe impl Sync for JsNativeRef {}

#[allow(unsafe_code)]
impl JsNativeRef {
    /// Pin `value` on the JS heap and return a `Send` handle to it.
    ///
    /// # Safety
    /// `env` and `value` must be a valid napi env / value on the current
    /// (main) thread.
    unsafe fn create(env: sys::napi_env, value: sys::napi_value) -> Result<Self> {
        let mut napi_ref = ptr::null_mut();
        check_status!(
            unsafe { sys::napi_create_reference(env, value, 1, &raw mut napi_ref) },
            "Failed to create napi reference for native event passthrough"
        )?;
        Ok(Self { env, napi_ref })
    }

    /// Resolve the live JS value this reference points at.
    ///
    /// # Safety
    /// Must be called on the main (v8) thread with a still-valid env.
    unsafe fn get_value(&self) -> Result<sys::napi_value> {
        let mut result = ptr::null_mut();
        check_status!(
            unsafe { sys::napi_get_reference_value(self.env, self.napi_ref, &raw mut result) },
            "Failed to resolve native event passthrough reference"
        )?;
        Ok(result)
    }

    /// Drop the reference, allowing v8 to collect the object.
    ///
    /// # Safety
    /// Must be called on the main (v8) thread with a still-valid env.
    unsafe fn delete(&self) {
        // Best-effort: unref then delete. Ignore status — if the env is
        // already torn down there is nothing to release.
        unsafe {
            let mut refcount = 0;
            let _ = sys::napi_reference_unref(self.env, self.napi_ref, &raw mut refcount);
            let _ = sys::napi_delete_reference(self.env, self.napi_ref);
        }
    }
}

/// Project a returned JS event value to a routable [`serde_json::Value`].
///
/// The full serde projection ([`Value::from_napi_value`]) fails if the
/// object carries own-enumerable values serde cannot represent — most
/// commonly a method stored as an **own** property (functions are not
/// JSON). Since the live object is pinned separately for identity, the
/// JSON projection only needs to carry enough to route the event. So on
/// failure we fall back to a minimal `{ "type": <type> }` object read
/// directly off the value, preserving routing while letting the
/// non-serializable fields ride along on the live object.
///
/// # Safety
/// `env` / `napi_val` must be a valid object on the main thread.
#[allow(unsafe_code)]
unsafe fn project_event_json(env: sys::napi_env, napi_val: sys::napi_value) -> Result<Value> {
    if let Ok(v) = unsafe { Value::from_napi_value(env, napi_val) } {
        return Ok(v);
    }
    // Fallback: read just the `type` discriminant so the event still routes.
    let obj = unsafe { Object::from_napi_value(env, napi_val)? };
    let event_type: Option<String> = obj.get("type")?;
    let mut map = serde_json::Map::new();
    if let Some(t) = event_type {
        map.insert("type".to_owned(), Value::String(t));
    }
    Ok(Value::Object(map))
}

/// If `data` is a JSON object, stamp the native-ref marker key into it so
/// the inbound side can resolve the live object. Non-object payloads are
/// wrapped as `{ "data": <payload>, "__blazen_native_ref__": "<uuid>" }`
/// so the tag always lives at the top level where the inbound resolver
/// looks for it.
fn stamp_native_ref(mut data: Value, key: RegistryKey) -> Value {
    if let Value::Object(map) = &mut data {
        map.insert(NATIVE_REF_TAG.to_owned(), Value::String(key.to_string()));
        return data;
    }
    let mut obj = serde_json::Map::new();
    obj.insert("data".to_owned(), data);
    obj.insert(NATIVE_REF_TAG.to_owned(), Value::String(key.to_string()));
    Value::Object(obj)
}

/// Extract the [`RegistryKey`] from a `{ "__blazen_native_ref__": "<uuid>"
/// }` marker embedded at the top level of `data`, if present.
#[must_use]
pub fn native_ref_key(data: &Value) -> Option<RegistryKey> {
    let raw = data.as_object()?.get(NATIVE_REF_TAG)?.as_str()?;
    RegistryKey::parse(raw).ok()
}

/// Strip the native-ref marker key from an event payload, returning the
/// cleaned JSON. Used before handing JSON to user-facing JSON projections
/// (results, streaming, snapshots) so the internal tag never leaks out.
#[must_use]
pub fn strip_native_ref(mut data: Value) -> Value {
    if let Value::Object(map) = &mut data {
        map.remove(NATIVE_REF_TAG);
    }
    data
}

// ---------------------------------------------------------------------------
// Registry insertion (outbound) + resolution (inbound)
// ---------------------------------------------------------------------------

/// Insert a captured live JS object into the run's unified
/// [`SessionRefRegistry`] and return the event JSON with the native-ref
/// marker stamped in.
///
/// `registry` is the same store every session ref uses, obtained from the
/// step's [`blazen_core::Context`]. The returned JSON keeps the original
/// `type` and serializable fields (for routing / snapshots) and gains the
/// [`NATIVE_REF_TAG`] so the next inbound step resolves back to the same
/// object.
pub async fn register_native_event(
    registry: &SessionRefRegistry,
    json: Value,
    native: Arc<JsNativeRef>,
) -> Value {
    match registry.insert_native_sync(native).await {
        Ok(key) => stamp_native_ref(json, key),
        // Registry full (MAX_SESSION_REFS_PER_RUN) — degrade gracefully to
        // JSON-only passthrough rather than failing the step. The event
        // still routes correctly; only object identity is lost for this hop.
        Err(_) => json,
    }
}

// ---------------------------------------------------------------------------
// StepEventArg — inbound (Rust -> JS handler argument)
// ---------------------------------------------------------------------------

/// The argument handed to a JS step handler.
///
/// Carries the event JSON plus a handle to the run's unified
/// [`SessionRefRegistry`]. Its [`ToNapiValue`] impl (main thread) rebuilds
/// the JS object from JSON, then — if the payload carries a still-live
/// [`NATIVE_REF_TAG`] — substitutes the original live object so the handler
/// sees the *same* object the previous step returned.
///
/// The registry is captured **by value** (an `Arc`) rather than read from a
/// task-local, because `to_napi_value` runs on the v8 main thread inside
/// the napi threadsafe-function callback — a different thread from the
/// tokio task that built this value and installed the
/// [`with_session_registry`](blazen_core::session_ref::with_session_registry)
/// task-local. The task-local is therefore **not** visible at marshal
/// time; carrying the `Arc` is the only correct way to reach the registry.
pub struct StepEventArg {
    json: Value,
    registry: Arc<SessionRefRegistry>,
}

impl StepEventArg {
    #[must_use]
    pub fn new(json: Value, registry: Arc<SessionRefRegistry>) -> Self {
        Self { json, registry }
    }
}

#[allow(unsafe_code)]
impl ToNapiValue for StepEventArg {
    unsafe fn to_napi_value(env: sys::napi_env, val: Self) -> Result<sys::napi_value> {
        // Fast path: try to resolve a live native ref for full identity
        // preservation. Reads the synchronous native fast-lane (no await),
        // which is safe on the v8 main thread.
        let registry = val.registry;
        if let Some(key) = native_ref_key(&val.json)
            && let Some(any) = registry.get_native_sync(key)
            && let Ok(native) = any.downcast::<JsNativeRef>()
        {
            // Resolve the live object and release the reference (consumed
            // for this hop).
            let resolved = unsafe { native.get_value() };
            unsafe { native.delete() };
            // Best-effort removal of the now-dead entry from the sync
            // fast-lane mirror so it is not resolved twice.
            registry.remove_native_sync(key);
            if let Ok(value) = resolved {
                return Ok(value);
            }
            // If resolution failed (env torn down / GC'd), fall through to
            // the JSON projection below.
        }

        // Default path: build a plain JS object from the (marker-stripped)
        // JSON projection.
        let clean = strip_native_ref(val.json);
        unsafe { Value::to_napi_value(env, clean) }
    }
}

// ---------------------------------------------------------------------------
// StepEventReturn — outbound (JS handler return -> Rust)
// ---------------------------------------------------------------------------

/// The value returned by a JS step handler.
///
/// Its [`FromNapiValue`] impl (main thread) projects the returned JS value
/// to JSON for routing while also pinning the original object(s) on the JS
/// heap via [`JsNativeRef`] so the next step can recover identity.
///
/// Three shapes are supported, matching the existing JSON contract:
///
/// - `null`/`undefined` -> [`StepEventReturn::None`]
/// - an array -> [`StepEventReturn::Multiple`] (one captured ref per event)
/// - any object -> [`StepEventReturn::Single`]
pub enum StepEventReturn {
    None,
    /// A primitive (string/number/bool) event — no live object to pin.
    JsonOnly(Value),
    Single {
        json: Value,
        native: Arc<JsNativeRef>,
    },
    Multiple(Vec<(Value, Arc<JsNativeRef>)>),
}

#[allow(unsafe_code)]
impl FromNapiValue for StepEventReturn {
    unsafe fn from_napi_value(env: sys::napi_env, napi_val: sys::napi_value) -> Result<Self> {
        let ty = type_of!(env, napi_val)?;
        match ty {
            ValueType::Null | ValueType::Undefined => Ok(StepEventReturn::None),
            ValueType::Object => {
                let mut is_arr = false;
                check_status!(
                    unsafe { sys::napi_is_array(env, napi_val, &raw mut is_arr) },
                    "Failed to detect whether the step return value is an array"
                )?;
                if is_arr {
                    // Per-element: project each entry to JSON and pin a ref
                    // to the element object so identity survives the hop.
                    let mut len = 0_u32;
                    check_status!(
                        unsafe { sys::napi_get_array_length(env, napi_val, &raw mut len) },
                        "Failed to read step return array length"
                    )?;
                    let mut out = Vec::with_capacity(len as usize);
                    for i in 0..len {
                        let mut element = ptr::null_mut();
                        check_status!(
                            unsafe { sys::napi_get_element(env, napi_val, i, &raw mut element) },
                            "Failed to read step return array element"
                        )?;
                        let json = unsafe { project_event_json(env, element)? };
                        let native = Arc::new(unsafe { JsNativeRef::create(env, element)? });
                        out.push((json, native));
                    }
                    Ok(StepEventReturn::Multiple(out))
                } else {
                    let json = unsafe { project_event_json(env, napi_val)? };
                    let native = Arc::new(unsafe { JsNativeRef::create(env, napi_val)? });
                    Ok(StepEventReturn::Single { json, native })
                }
            }
            // Primitives (string/number/bool) cannot carry identity; treat
            // them as plain JSON events with no native ref.
            _ => {
                let json = unsafe { Value::from_napi_value(env, napi_val)? };
                if json.is_null() {
                    Ok(StepEventReturn::None)
                } else {
                    Ok(StepEventReturn::JsonOnly(json))
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stamp_into_object_adds_marker_at_top_level() {
        let key = RegistryKey::new();
        let data = serde_json::json!({ "type": "AnalyzeEvent", "text": "hi" });
        let stamped = stamp_native_ref(data, key);
        assert_eq!(stamped["type"], "AnalyzeEvent");
        assert_eq!(stamped["text"], "hi");
        assert_eq!(stamped[NATIVE_REF_TAG], key.to_string());
        // And it round-trips back out through `native_ref_key`.
        assert_eq!(native_ref_key(&stamped), Some(key));
    }

    #[test]
    fn stamp_into_non_object_wraps_under_data() {
        let key = RegistryKey::new();
        let stamped = stamp_native_ref(serde_json::json!("scalar"), key);
        assert_eq!(stamped["data"], "scalar");
        assert_eq!(native_ref_key(&stamped), Some(key));
    }

    #[test]
    fn native_ref_key_rejects_missing_or_malformed() {
        assert!(native_ref_key(&serde_json::json!({ "type": "X" })).is_none());
        assert!(native_ref_key(&serde_json::json!({ NATIVE_REF_TAG: 42 })).is_none());
        assert!(native_ref_key(&serde_json::json!({ NATIVE_REF_TAG: "not-a-uuid" })).is_none());
        assert!(native_ref_key(&serde_json::json!("scalar")).is_none());
    }

    #[test]
    fn strip_removes_only_the_marker_key() {
        let key = RegistryKey::new();
        let stamped = stamp_native_ref(serde_json::json!({ "type": "E", "keep": true }), key);
        let cleaned = strip_native_ref(stamped);
        assert_eq!(cleaned["type"], "E");
        assert_eq!(cleaned["keep"], true);
        assert!(cleaned.as_object().unwrap().get(NATIVE_REF_TAG).is_none());
    }

    #[test]
    fn strip_is_noop_on_non_object() {
        assert_eq!(
            strip_native_ref(serde_json::json!("scalar")),
            serde_json::json!("scalar")
        );
    }

    #[tokio::test]
    async fn registry_sync_lane_round_trips_through_async_insert() {
        // Confirm the sync native fast-lane is populated by the async insert
        // and resolvable without awaiting (the inbound-resolution contract).
        let reg = SessionRefRegistry::new();
        let key = reg
            .insert_native_sync(Arc::new(7_u32))
            .await
            .expect("insert ok");

        // Sync resolution (no await) sees the same Arc.
        let got = reg.get_native_sync(key).expect("present in sync lane");
        assert_eq!(*got.downcast::<u32>().unwrap(), 7);

        // The async unified map sees it too (snapshot / pause accounting).
        assert_eq!(reg.len().await, 1);

        // Removing from the sync lane makes it unresolvable there.
        let removed = reg.remove_native_sync(key).expect("removed");
        assert_eq!(*removed.downcast::<u32>().unwrap(), 7);
        assert!(reg.get_native_sync(key).is_none());
    }
}
