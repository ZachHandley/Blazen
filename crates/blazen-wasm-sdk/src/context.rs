//! WASM-bindgen wrapper around the real [`blazen_core::Context`].
//!
//! Stage 2 of the wasm-bindgen rollout: the WASM SDK now drives the genuine
//! workflow engine. This module wraps [`blazen_core::Context`] and surfaces a
//! synchronous JS API matching the contract in `crates/blazen-wasm-sdk/README.md`
//! ("All `Context` methods are **synchronous** in WASM").
//!
//! ## Storage backing
//!
//! State, session, and metadata all delegate to the underlying
//! [`blazen_core::Context`]. The real context stores data as
//! [`blazen_core::StateValue`] (JSON / bytes / native), which means JS values
//! are round-tripped through `serde-wasm-bindgen` / `serde_json::Value`. This
//! preserves snapshot/resume semantics at the cost of object identity (see the
//! note on [`WasmContext::session`] below).
//!
//! ## Sync over async
//!
//! [`blazen_core::Context`]'s public surface is `async` because it guards an
//! `Arc<RwLock<…>>`. In single-threaded WASM the lock can never be contended
//! across threads (there's only one OS thread), so a `read()` / `write()`
//! future resolves on the first poll — no executor needed. We exploit that
//! with [`block_on_local`], which polls a future exactly once and panics if
//! it returns `Pending`. That guard catches accidental misuse if a future
//! method ever grows real I/O (e.g. distributed peer RPC) and gets called
//! through the sync API.
//!
//! ## What is NOT preserved from the simplified WasmContext
//!
//! - JS object identity through `session.set` / `session.get`. The real
//!   context stores session entries as JSON, so the value you read back is a
//!   structurally-equal JS object, not the same reference.
//! - `BlazenState` field decomposition. The previous simplified context split
//!   marker objects into per-field state entries with metadata stitching. The
//!   real context stores the whole JSON tree under one key. Decomposition is
//!   no longer required for correctness because the JSON serialiser handles
//!   nested values natively.

use std::collections::HashMap;
use std::pin::Pin;
use std::task::{Context as TaskContext, Poll};

use blazen_core::{BytesWrapper, Context, StateValue};
use wasm_bindgen::prelude::*;

// ---------------------------------------------------------------------------
// Block-on-local: poll a future exactly once.
// ---------------------------------------------------------------------------

/// Poll a future to completion on the current thread, panicking if it returns
/// `Pending`.
///
/// Used to bridge [`blazen_core::Context`]'s `async` lock-guarded API to the
/// synchronous JS contract this module exposes. Safe for any operation that
/// only acquires the internal `Arc<RwLock<ContextInner>>` and does not await
/// anything else: in single-threaded WASM the lock cannot be contended.
///
/// # Panics
///
/// Panics if the future yields `Pending`. That signals a real `await` point
/// inside the supposedly-sync method (e.g. a distributed peer RPC). The
/// caller must use the async API instead.
pub(crate) fn block_on_local<F: std::future::Future>(fut: F) -> F::Output {
    let mut fut = Box::pin(fut);
    let waker = noop_waker();
    let mut cx = TaskContext::from_waker(&waker);
    match Pin::as_mut(&mut fut).poll(&mut cx) {
        Poll::Ready(v) => v,
        Poll::Pending => panic!(
            "WasmContext sync method polled future to Pending — only no-I/O \
             operations are allowed through the sync API"
        ),
    }
}

/// Construct a no-op `Waker`. Replaces `futures_util::task::noop_waker` so we
/// don't depend on its specific path.
fn noop_waker() -> std::task::Waker {
    use std::task::{RawWaker, RawWakerVTable, Waker};

    const VTABLE: RawWakerVTable = RawWakerVTable::new(
        |_| RawWaker::new(std::ptr::null(), &VTABLE),
        |_| {},
        |_| {},
        |_| {},
    );

    // SAFETY: the vtable functions are all no-ops and never dereference the
    // data pointer. A null pointer is therefore safe.
    unsafe { Waker::from_raw(RawWaker::new(std::ptr::null(), &VTABLE)) }
}

// ---------------------------------------------------------------------------
// TypeScript type augmentation
// ---------------------------------------------------------------------------

#[wasm_bindgen(typescript_custom_section)]
const TS_STATE_VALUE: &str = r#"
/**
 * A value that can be stored in the workflow context state map.
 *
 * This is the polymorphic JS-side shape — strings / numbers / objects / etc.
 * are all valid. The typed equivalent (a `StateValue` *class* with explicit
 * `Json` / `Bytes` / `Native` variants) is exported separately as
 * `StateValue` from `core_types`.
 */
export type StateValueLike =
  | string
  | number
  | boolean
  | null
  | Uint8Array
  | StateValueLike[]
  | { [key: string]: StateValueLike };
"#;

#[wasm_bindgen(typescript_custom_section)]
const TS_NAMESPACES: &str = r#"
export interface StateNamespace {
  get(key: string, defaultValue?: StateValueLike): StateValueLike | null;
  set(key: string, value: StateValueLike): void;
  keys(): string[];
}
export interface SessionNamespace {
  get(key: string, defaultValue?: unknown): unknown;
  set(key: string, value: unknown): void;
  keys(): string[];
}
export interface MetadataNamespace {
  get(key: string, defaultValue?: unknown): unknown;
  set(key: string, value: unknown): void;
}
"#;

/// Backward-compatible TypeScript alias for the old `WorkflowContext` JS
/// class name. Existing imports of `WorkflowContext` continue to type-check
/// against the new `Context` class. Slated for removal in a future release;
/// new code should import `Context` directly.
#[wasm_bindgen(typescript_custom_section)]
const TS_WORKFLOWCONTEXT_ALIAS: &str = r#"
/** @deprecated Use `Context` instead. Kept for backward compatibility. */
export type WorkflowContext = Context;
"#;

// ---------------------------------------------------------------------------
// Internal key prefixes
// ---------------------------------------------------------------------------

/// Prefix used for `session` namespace entries inside the shared state map.
///
/// Session values do not have a public sync surface on
/// [`blazen_core::Context`] — `session_refs` are Uuid-keyed and `objects`
/// require `Send + Sync`. Storing JS-shaped session data behind this prefix
/// lets us preserve string keys while still riding the snapshotable JSON
/// state map.
const SESSION_PREFIX: &str = "__wasm_session__/";

/// Prefix used for the JS-writable side of the metadata namespace.
///
/// Reads still consult the real `Context` metadata first (engine-seeded
/// `run_id` / `workflow_name`) and fall back to this prefix for keys that
/// JS callers wrote themselves. Writes always land under this prefix because
/// `Context::set_metadata` is `pub(crate)` and not callable from outside the
/// `blazen-core` crate.
const METADATA_PREFIX: &str = "__wasm_metadata__/";

// ---------------------------------------------------------------------------
// JsValue <-> StateValue helpers
// ---------------------------------------------------------------------------

/// Convert a `JsValue` into a [`StateValue`] suitable for the shared state map.
///
/// `Uint8Array` instances become [`StateValue::Bytes`]; everything else is
/// serialised through `serde-wasm-bindgen` to JSON.
fn js_to_state_value(value: JsValue) -> Result<StateValue, JsValue> {
    if value.is_instance_of::<js_sys::Uint8Array>() {
        let arr: js_sys::Uint8Array = value.unchecked_into();
        return Ok(StateValue::Bytes(BytesWrapper(arr.to_vec())));
    }

    let json: serde_json::Value = serde_wasm_bindgen::from_value(value)
        .map_err(|e| JsValue::from_str(&format!("WasmContext: value not JSON-serialisable: {e}")))?;
    Ok(StateValue::Json(json))
}

/// Convert a [`StateValue`] back into a `JsValue` for the JS caller.
///
/// `Bytes` becomes a `Uint8Array`. `Json` is round-tripped through
/// `serde-wasm-bindgen`. `Native` (platform-serialised opaque blobs) is
/// surfaced as a `Uint8Array` — WASM bindings have no native pickle format
/// of their own, so this is the most useful representation.
fn state_value_to_js(value: &StateValue) -> JsValue {
    match value {
        StateValue::Bytes(b) | StateValue::Native(b) => {
            js_sys::Uint8Array::from(b.0.as_slice()).into()
        }
        StateValue::Json(v) => serde_wasm_bindgen::to_value(v).unwrap_or(JsValue::NULL),
    }
}

/// Convert a `serde_json::Value` directly into a `JsValue`.
fn json_to_js(value: &serde_json::Value) -> JsValue {
    serde_wasm_bindgen::to_value(value).unwrap_or(JsValue::NULL)
}

// ---------------------------------------------------------------------------
// WasmContext
// ---------------------------------------------------------------------------

/// JS-facing handle to a real [`blazen_core::Context`].
///
/// Cheaply cloneable: the inner `Context` is itself an `Arc<RwLock<…>>` clone
/// so duplicating this struct only bumps refcounts.
///
/// **JS name:** This type is exported to JS as `Context`. For backward
/// compatibility with the previous SDK surface, a `WorkflowContext` alias is
/// declared in [`TS_WORKFLOWCONTEXT_ALIAS`] so existing TypeScript imports
/// of `WorkflowContext` continue to type-check. The alias will be removed in
/// a future release.
#[wasm_bindgen(js_name = "Context")]
pub struct WasmContext {
    /// Underlying real context. `Context` is itself `Arc`-backed so we can
    /// clone it freely without an outer wrapper.
    inner: Context,
    /// Cached workflow name, plumbed in at construction. The engine also
    /// writes this into the underlying context's metadata under the key
    /// `workflow_name`, but caching here avoids a lock acquisition per read.
    workflow_name: String,
}

impl Clone for WasmContext {
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
            workflow_name: self.workflow_name.clone(),
        }
    }
}

// ---------------------------------------------------------------------------
// Internal (non-wasm_bindgen) API used by sibling modules.
// ---------------------------------------------------------------------------

impl WasmContext {
    /// Wrap an existing [`blazen_core::Context`].
    ///
    /// Used by the JS-callback step adapter (`workflow.rs`'s `addStep`) to
    /// hand the running context to user-supplied step handlers. The caller
    /// is responsible for ensuring `inner` was constructed by the real
    /// workflow engine — this module never builds a fresh `Context` on its
    /// own because the public `Context::new` is `pub(crate)`.
    pub(crate) fn from_inner(inner: Context, workflow_name: String) -> Self {
        Self {
            inner,
            workflow_name,
        }
    }

    /// Borrow the underlying real context.
    ///
    /// Lets the step adapter and handler wrapper reach into the engine for
    /// async-only operations (e.g. emitting events from inside an async
    /// callback, snapshotting state for a pause request).
    #[allow(dead_code)]
    pub(crate) fn inner(&self) -> &Context {
        &self.inner
    }
}

// ---------------------------------------------------------------------------
// Public wasm_bindgen API
// ---------------------------------------------------------------------------

#[wasm_bindgen(js_class = "Context")]
impl WasmContext {
    /// Persistable workflow state. Backed by [`blazen_core::Context`]'s
    /// JSON-typed state map; entries survive snapshotting.
    ///
    /// ```javascript
    /// ctx.state.set("counter", 5);
    /// const count = ctx.state.get("counter");
    /// ```
    #[must_use]
    #[wasm_bindgen(getter)]
    pub fn state(&self) -> WasmStateNamespace {
        WasmStateNamespace { ctx: self.clone() }
    }

    /// In-process key/value scratch space.
    ///
    /// NOTE: Stage 2 wraps the real `blazen_core::Context` whose session
    /// storage is `serde_json::Value`-backed. JS reference identity is NOT
    /// preserved across `session.set` / `session.get` calls. The simplified
    /// `WasmWorkflow`'s identity-preservation behaviour is gone. If users
    /// need it back, file an issue and we'll add a separate JS-only session
    /// store keyed alongside the real one.
    ///
    /// ```javascript
    /// ctx.session.set("token", "abc123");
    /// const token = ctx.session.get("token");
    /// ```
    #[must_use]
    #[wasm_bindgen(getter)]
    pub fn session(&self) -> WasmSessionNamespace {
        WasmSessionNamespace { ctx: self.clone() }
    }

    /// Workflow metadata. The engine seeds this with `run_id` and
    /// `workflow_name` before any step runs; user-written keys are stored in
    /// a separate JS-writable side-channel because `Context::set_metadata`
    /// is `pub(crate)` in `blazen-core`.
    ///
    /// Reads consult the engine-seeded metadata first, then fall back to the
    /// JS-writable side-channel.
    #[must_use]
    #[wasm_bindgen(getter)]
    pub fn metadata(&self) -> WasmMetadataNamespace {
        WasmMetadataNamespace { ctx: self.clone() }
    }

    /// Store raw binary data under `key`.
    ///
    /// Mirrors [`blazen_core::Context::set_bytes`] and the `set_bytes` method
    /// on the Node SDK's `Context`. The data is stored as a
    /// [`blazen_core::StateValue::Bytes`] entry and survives snapshotting.
    #[wasm_bindgen(js_name = "setBytes")]
    pub fn set_bytes(&self, key: String, data: js_sys::Uint8Array) {
        let inner = self.inner.clone();
        let bytes = data.to_vec();
        block_on_local(async move {
            inner.set_bytes(&key, bytes).await;
        });
    }

    /// Retrieve raw binary data previously stored under `key`.
    ///
    /// Returns `null` if the key does not exist or the stored value is a
    /// JSON / native variant rather than bytes. Mirrors
    /// [`blazen_core::Context::get_bytes`].
    #[wasm_bindgen(js_name = "getBytes")]
    #[must_use]
    pub fn get_bytes(&self, key: String) -> JsValue {
        let inner = self.inner.clone();
        let raw: Option<Vec<u8>> = block_on_local(async move { inner.get_bytes(&key).await });
        match raw {
            Some(bytes) => js_sys::Uint8Array::from(bytes.as_slice()).into(),
            None => JsValue::NULL,
        }
    }

    /// Send an event into the workflow's internal routing channel.
    ///
    /// Accepts either a JS object with a `type: string` discriminator (the
    /// canonical shape used by step handlers) or a plain string treated as
    /// the event type with an empty payload.
    #[wasm_bindgen(js_name = "sendEvent")]
    pub fn send_event(&self, event: JsValue) -> Result<(), JsValue> {
        let dynamic = js_to_dynamic_event(&event)?;
        let inner = self.inner.clone();
        block_on_local(async move {
            inner.send_event(dynamic).await;
        });
        Ok(())
    }

    /// Publish an event to the external broadcast stream.
    ///
    /// Same accepted shapes as [`send_event`](Self::send_event), but the
    /// event is delivered ONLY to subscribers of the streaming channel — it
    /// does NOT route through the internal step registry.
    #[wasm_bindgen(js_name = "writeEventToStream")]
    pub fn write_event_to_stream(&self, event: JsValue) -> Result<(), JsValue> {
        let dynamic = js_to_dynamic_event(&event)?;
        let inner = self.inner.clone();
        block_on_local(async move {
            inner.write_event_to_stream(dynamic).await;
        });
        Ok(())
    }

    /// Return the unique run ID for this workflow execution.
    #[wasm_bindgen(js_name = "runId")]
    pub fn run_id(&self) -> String {
        let inner = self.inner.clone();
        block_on_local(async move { inner.run_id().await.to_string() })
    }

    /// The workflow name.
    #[wasm_bindgen(getter, js_name = "workflowName")]
    pub fn workflow_name(&self) -> String {
        self.workflow_name.clone()
    }
}

// ---------------------------------------------------------------------------
// JS event -> blazen_events::DynamicEvent
// ---------------------------------------------------------------------------

/// Turn a JS event object (or string) into a [`blazen_events::DynamicEvent`]
/// suitable for emission through [`blazen_core::Context`].
fn js_to_dynamic_event(event: &JsValue) -> Result<blazen_events::DynamicEvent, JsValue> {
    // Plain string => use as the event type with an empty payload.
    if let Some(s) = event.as_string() {
        return Ok(blazen_events::DynamicEvent {
            event_type: s,
            data: serde_json::Value::Null,
        });
    }

    // Object with a `type` discriminator.
    if event.is_object() {
        let type_val = js_sys::Reflect::get(event, &JsValue::from_str("type"))
            .map_err(|e| JsValue::from_str(&format!("sendEvent: cannot read .type: {e:?}")))?;
        let event_type = type_val
            .as_string()
            .ok_or_else(|| JsValue::from_str("sendEvent: event.type must be a string"))?;

        let data: serde_json::Value = serde_wasm_bindgen::from_value(event.clone())
            .map_err(|e| JsValue::from_str(&format!("sendEvent: payload not JSON: {e}")))?;

        return Ok(blazen_events::DynamicEvent { event_type, data });
    }

    Err(JsValue::from_str(
        "sendEvent: argument must be a string or an object with a `type` field",
    ))
}

// ---------------------------------------------------------------------------
// WasmStateNamespace — persistable workflow state
// ---------------------------------------------------------------------------

/// Namespace handle for the persistable workflow state map.
#[wasm_bindgen(js_name = "StateNamespace")]
pub struct WasmStateNamespace {
    ctx: WasmContext,
}

#[wasm_bindgen(js_class = "StateNamespace")]
impl WasmStateNamespace {
    /// Read a state entry. Returns `defaultValue` (or `null` if omitted) when
    /// the key is missing.
    #[wasm_bindgen]
    pub fn get(&self, key: String, default: JsValue) -> JsValue {
        let inner = self.ctx.inner.clone();
        let stored: Option<StateValue> =
            block_on_local(async move { inner.get_value(&key).await });
        match stored {
            Some(sv) => state_value_to_js(&sv),
            None => {
                if default.is_undefined() {
                    JsValue::NULL
                } else {
                    default
                }
            }
        }
    }

    /// Write a state entry. `Uint8Array` values are stored as raw bytes;
    /// other values are JSON-serialised through `serde-wasm-bindgen`.
    #[wasm_bindgen]
    pub fn set(&self, key: String, value: JsValue) -> Result<(), JsValue> {
        let sv = js_to_state_value(value)?;
        let inner = self.ctx.inner.clone();
        block_on_local(async move {
            inner.set_value(&key, sv).await;
        });
        Ok(())
    }

    /// Return all keys currently present in the state map (excluding
    /// session and metadata side-channel entries).
    #[wasm_bindgen]
    pub fn keys(&self) -> js_sys::Array {
        let inner = self.ctx.inner.clone();
        let snap: HashMap<String, StateValue> =
            block_on_local(async move { inner.snapshot_state().await });
        let arr = js_sys::Array::new();
        for k in snap.keys() {
            if k.starts_with(SESSION_PREFIX) || k.starts_with(METADATA_PREFIX) {
                continue;
            }
            arr.push(&JsValue::from_str(k));
        }
        arr
    }
}

// ---------------------------------------------------------------------------
// WasmSessionNamespace — JSON-backed session scratch space
// ---------------------------------------------------------------------------

/// Namespace handle for the session-scoped scratch space.
#[wasm_bindgen(js_name = "SessionNamespace")]
pub struct WasmSessionNamespace {
    ctx: WasmContext,
}

#[wasm_bindgen(js_class = "SessionNamespace")]
impl WasmSessionNamespace {
    /// Read a session entry. Returns `defaultValue` (or `null` if omitted)
    /// when the key is missing.
    #[wasm_bindgen]
    pub fn get(&self, key: String, default: JsValue) -> JsValue {
        let prefixed = format!("{SESSION_PREFIX}{key}");
        let inner = self.ctx.inner.clone();
        let stored: Option<StateValue> =
            block_on_local(async move { inner.get_value(&prefixed).await });
        match stored {
            Some(sv) => state_value_to_js(&sv),
            None => {
                if default.is_undefined() {
                    JsValue::NULL
                } else {
                    default
                }
            }
        }
    }

    /// Write a session entry. JS reference identity is NOT preserved — the
    /// value is serialised through `serde-wasm-bindgen` and reconstructed on
    /// each `get`. See [`WasmContext::session`] for the rationale.
    #[wasm_bindgen]
    pub fn set(&self, key: String, value: JsValue) -> Result<(), JsValue> {
        let prefixed = format!("{SESSION_PREFIX}{key}");
        let sv = js_to_state_value(value)?;
        let inner = self.ctx.inner.clone();
        block_on_local(async move {
            inner.set_value(&prefixed, sv).await;
        });
        Ok(())
    }

    /// Return all keys currently present in the session scratch space (with
    /// the internal prefix stripped).
    #[wasm_bindgen]
    pub fn keys(&self) -> js_sys::Array {
        let inner = self.ctx.inner.clone();
        let snap: HashMap<String, StateValue> =
            block_on_local(async move { inner.snapshot_state().await });
        let arr = js_sys::Array::new();
        for k in snap.keys() {
            if let Some(stripped) = k.strip_prefix(SESSION_PREFIX) {
                arr.push(&JsValue::from_str(stripped));
            }
        }
        arr
    }
}

// ---------------------------------------------------------------------------
// WasmMetadataNamespace — engine-seeded reads + JS-writable side-channel
// ---------------------------------------------------------------------------

/// Namespace handle for workflow metadata.
#[wasm_bindgen(js_name = "MetadataNamespace")]
pub struct WasmMetadataNamespace {
    ctx: WasmContext,
}

#[wasm_bindgen(js_class = "MetadataNamespace")]
impl WasmMetadataNamespace {
    /// Read a metadata entry.
    ///
    /// Lookup order:
    /// 1. The engine-seeded metadata map (e.g. `run_id`, `workflow_name`).
    /// 2. The JS-writable side-channel under [`METADATA_PREFIX`].
    /// 3. `defaultValue` (or `null` if omitted).
    #[wasm_bindgen]
    pub fn get(&self, key: String, default: JsValue) -> JsValue {
        let inner = self.ctx.inner.clone();
        let lookup_key = key.clone();
        let real_meta: HashMap<String, serde_json::Value> =
            block_on_local(async move { inner.snapshot_metadata().await });

        if let Some(v) = real_meta.get(&lookup_key) {
            return json_to_js(v);
        }

        // Fall back to the JS-writable side-channel.
        let prefixed = format!("{METADATA_PREFIX}{key}");
        let inner = self.ctx.inner.clone();
        let stored: Option<StateValue> =
            block_on_local(async move { inner.get_value(&prefixed).await });
        match stored {
            Some(sv) => state_value_to_js(&sv),
            None => {
                if default.is_undefined() {
                    JsValue::NULL
                } else {
                    default
                }
            }
        }
    }

    /// Write a metadata entry into the JS-writable side-channel.
    ///
    /// Engine-seeded metadata (`run_id`, `workflow_name`) is read-only from
    /// JS because [`blazen_core::Context::set_metadata`] is `pub(crate)`.
    /// Writing under those keys still succeeds but stores into the
    /// side-channel; subsequent reads return the original engine value (the
    /// real metadata wins on lookup).
    #[wasm_bindgen]
    pub fn set(&self, key: String, value: JsValue) -> Result<(), JsValue> {
        let prefixed = format!("{METADATA_PREFIX}{key}");
        let sv = js_to_state_value(value)?;
        let inner = self.ctx.inner.clone();
        block_on_local(async move {
            inner.set_value(&prefixed, sv).await;
        });
        Ok(())
    }
}
