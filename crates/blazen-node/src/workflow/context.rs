//! JavaScript wrapper for the workflow [`Context`](blazen_core::Context).
//!
//! `set()` stores any JSON-serializable value. `setBytes()` stores binary data
//! as `Buffer`. `get()` returns the stored value regardless of variant —
//! `Buffer` for binary data, the original JS value for JSON data.
//!
//! ## Namespaces
//!
//! Two explicit namespaces are exposed alongside the smart-routing shortcuts
//! (`ctx.set` / `ctx.get`):
//!
//! - **`ctx.state`** — persistable values: JSON and raw binary. Survives
//!   `pause()` / `resume()` and checkpoint stores.
//! - **`ctx.session`** — in-process-only values. Excluded from snapshots,
//!   so use this for things whose lifetime should be bounded by the
//!   current workflow run.
//!
//! **Note on JS object identity.** napi-rs's `Reference<T>` is not `Send`
//! because its `Drop` must run on the v8 main thread, and arbitrary JS
//! objects cannot safely cross the napi/tokio boundary. As a result the
//! Node bindings store session values as `serde_json::Value` (like the
//! rest of the JS event system). Full identity preservation of live JS
//! objects through event payloads — the analogue of the Python
//! `Arc<Py<PyAny>>` path — is a follow-up refactor that would need a
//! different threading model.

use std::sync::Arc;

use blazen_core::StateValue;
use blazen_core::session_ref::{RegistryKey, SessionRefSerializable};
use napi::bindgen_prelude::*;
use napi_derive::napi;

use super::event::js_value_to_any_event;
use super::session_ref_serializable::NodeSessionRefSerializable;

/// Payload returned by [`JsContext::get_session_ref_serializable`].
///
/// Carries the type-tag string the JS caller supplied at insertion
/// time alongside the raw bytes captured for that key. JS code is
/// responsible for reconstructing whatever runtime object the bytes
/// represent — see the module docs on
/// [`super::session_ref_serializable`] for the trade-off rationale.
#[napi(object, js_name = "SerializableRefPayload")]
pub struct SerializableRefPayload {
    /// Stable identifier the JS caller passed to
    /// `insertSessionRefSerializable`.
    #[napi(js_name = "typeName")]
    pub type_name: String,
    /// Raw bytes the JS caller passed to
    /// `insertSessionRefSerializable`. Returned as a `Buffer` so the
    /// payload survives the napi boundary unchanged.
    pub bytes: Buffer,
}

/// Shared workflow context accessible by all steps.
///
/// Provides typed key/value storage, event emission, and stream publishing.
/// Use `set()` for JSON-serializable values, `setBytes()` for binary data.
/// `get()` returns the correct type for any stored value.
#[napi(js_name = "Context")]
pub struct JsContext {
    pub(crate) inner: blazen_core::Context,
}

#[napi]
#[allow(clippy::missing_errors_doc)]
impl JsContext {
    /// Store a JSON-serializable value under the given key.
    ///
    /// For binary data (Buffer/Uint8Array), use `setBytes()` instead.
    #[napi(ts_args_type = "key: string, value: Exclude<StateValue, Buffer>")]
    pub async fn set(&self, key: String, value: serde_json::Value) -> Result<()> {
        self.inner.set(&key, value).await;
        Ok(())
    }

    /// Retrieve a value previously stored under the given key.
    ///
    /// Returns the original JS value for JSON data, an array-of-bytes
    /// representation for binary data (use `getBytes` for proper
    /// `Buffer` round-trip), or `defaultValue` if the key is missing
    /// or the stored value is `null`/`Native`.
    #[napi(
        ts_args_type = "key: string, defaultValue?: StateValue",
        ts_return_type = "Promise<StateValue | null>"
    )]
    pub async fn get(
        &self,
        key: String,
        default: Option<serde_json::Value>,
    ) -> Result<serde_json::Value> {
        let val = match self.inner.get_value(&key).await {
            Some(StateValue::Json(v)) => v,
            Some(StateValue::Bytes(b)) => {
                let arr: Vec<serde_json::Value> =
                    b.0.into_iter()
                        .map(|byte| serde_json::Value::Number(byte.into()))
                        .collect();
                serde_json::Value::Array(arr)
            }
            Some(StateValue::Native(_)) | None => serde_json::Value::Null,
        };
        if val.is_null() {
            return Ok(default.unwrap_or(serde_json::Value::Null));
        }
        Ok(val)
    }

    /// Emit an event into the internal routing queue.
    ///
    /// The event will be routed to any step whose `eventTypes` list includes
    /// its event type. The event object must have a `type` field.
    #[napi(js_name = "sendEvent")]
    pub async fn send_event(&self, event: serde_json::Value) -> Result<()> {
        let any_event = js_value_to_any_event(&event);

        let event_type = any_event.event_type_id().to_owned();
        let data = any_event.to_json();

        let dynamic = blazen_events::DynamicEvent { event_type, data };
        self.inner.send_event(dynamic).await;
        Ok(())
    }

    /// Publish an event to the external broadcast stream.
    ///
    /// Consumers that subscribed via streaming will receive this event.
    /// Unlike `sendEvent`, this does NOT route the event through the
    /// internal step registry.
    #[napi(js_name = "writeEventToStream")]
    pub async fn write_event_to_stream(&self, event: serde_json::Value) -> Result<()> {
        let any_event = js_value_to_any_event(&event);

        let event_type = any_event.event_type_id().to_owned();
        let data = any_event.to_json();

        let dynamic = blazen_events::DynamicEvent { event_type, data };
        self.inner.write_event_to_stream(dynamic).await;
        Ok(())
    }

    /// Store raw binary data under the given key.
    ///
    /// Use this for Buffer/Uint8Array data that should be stored as binary
    /// and returned as Buffer from `getBytes()`.
    #[napi(js_name = "setBytes")]
    pub async fn set_bytes(&self, key: String, data: Buffer) -> Result<()> {
        self.inner.set_bytes(&key, data.to_vec()).await;
        Ok(())
    }

    /// Retrieve raw binary data previously stored under the given key.
    ///
    /// Returns `defaultValue` if the key does not exist or the stored
    /// value is not binary data; if no default is provided, returns
    /// `null`.
    #[napi(
        js_name = "getBytes",
        ts_args_type = "key: string, defaultValue?: Buffer",
        ts_return_type = "Promise<Buffer | null>"
    )]
    pub async fn get_bytes(&self, key: String, default: Option<Buffer>) -> Result<Option<Buffer>> {
        Ok(self
            .inner
            .get_bytes(&key)
            .await
            .map(Buffer::from)
            .or(default))
    }

    /// Get the workflow run ID.
    #[napi(js_name = "runId")]
    pub async fn run_id(&self) -> Result<String> {
        let id = self.inner.run_id().await;
        Ok(id.to_string())
    }

    /// Store an opaque, user-serialized payload in the session-ref
    /// registry under a fresh [`RegistryKey`].
    ///
    /// `typeName` is a stable identifier the caller chooses for this
    /// payload (e.g. `"app::EmbeddingHandle"`). The same name must be
    /// used on the resume side to recognise the payload — the type tag
    /// is captured into snapshot metadata along with the bytes when
    /// the workflow is paused under
    /// [`SessionPausePolicy::PickleOrSerialize`](blazen_core::session_ref::SessionPausePolicy).
    ///
    /// Returns the registry key as a string. JS callers can use this
    /// key with [`Self::get_session_ref_serializable`] inside the same
    /// run, or after a snapshot/resume cycle to retrieve the bytes
    /// they originally inserted.
    ///
    /// **Important.** Unlike the Python binding, the Node bindings do
    /// not currently auto-detect a `serialize()` method on JS objects.
    /// JS code must serialize the value itself (typically into a
    /// `Buffer`) before calling this method, and must deserialize the
    /// bytes returned by `getSessionRefSerializable` back into a
    /// runtime object in user code. This limitation is rooted in the
    /// `serde_json::Value`-based step bridge and is tracked separately
    /// from this method.
    #[napi(js_name = "insertSessionRefSerializable")]
    pub async fn insert_session_ref_serializable(
        &self,
        type_name: String,
        bytes: Buffer,
    ) -> Result<String> {
        let registry = self.inner.session_refs_arc().await;
        let serializable = Arc::new(NodeSessionRefSerializable::new(&type_name, bytes.to_vec()));
        let key = registry
            .insert_serializable(serializable)
            .await
            .map_err(|e| napi::Error::from_reason(e.to_string()))?;
        Ok(key.to_string())
    }

    /// Retrieve a payload previously stored via
    /// [`Self::insert_session_ref_serializable`].
    ///
    /// Returns `null` if the registry has no entry under `key`, or if
    /// the entry exists but was inserted via the non-serializable
    /// path (`set` / `setBytes` / language-specific live refs).
    /// Otherwise returns `{ typeName, bytes }` matching the
    /// arguments the caller originally passed in.
    #[napi(js_name = "getSessionRefSerializable")]
    pub async fn get_session_ref_serializable(
        &self,
        key: String,
    ) -> Result<Option<SerializableRefPayload>> {
        let parsed = RegistryKey::parse(&key)
            .map_err(|e| napi::Error::from_reason(format!("invalid registry key `{key}`: {e}")))?;
        let registry = self.inner.session_refs_arc().await;
        let Some(serializable) = registry.get_serializable(parsed).await else {
            return Ok(None);
        };

        // Try to downcast to the concrete `NodeSessionRefSerializable`
        // adapter so we can return the original user bytes (not the
        // length-prefixed wire format produced by `blazen_serialize`).
        let trait_ref: &dyn SessionRefSerializable = &*serializable;
        let any_ref: &dyn std::any::Any = trait_ref;
        if let Some(node_ser) = any_ref.downcast_ref::<NodeSessionRefSerializable>() {
            return Ok(Some(SerializableRefPayload {
                type_name: node_ser.type_tag().to_owned(),
                bytes: Buffer::from(node_ser.user_bytes().to_vec()),
            }));
        }

        // Foreign serializable adapter (e.g. inserted via a different
        // language binding sharing the same registry). Fall back to the
        // wire-format bytes returned by `blazen_serialize` so the
        // caller still has something to work with.
        let wire_bytes = serializable
            .blazen_serialize()
            .map_err(|e| napi::Error::from_reason(e.to_string()))?;
        Ok(Some(SerializableRefPayload {
            type_name: serializable.blazen_type_tag().to_owned(),
            bytes: Buffer::from(wire_bytes),
        }))
    }

    /// Persistable workflow state. Survives `pause()` / `resume()`,
    /// checkpoints, and durable storage.
    ///
    /// ```javascript
    /// await ctx.state.set("counter", 5);
    /// const count = await ctx.state.get("counter");
    /// ```
    #[must_use]
    #[napi(getter)]
    pub fn state(&self) -> JsStateNamespace {
        JsStateNamespace {
            inner: self.inner.clone(),
        }
    }

    /// In-process-only values. Excluded from snapshots — use this for
    /// things that should not survive `pause()` / `resume()`.
    ///
    /// ```javascript
    /// await ctx.session.set("reqId", 42);
    /// const n = await ctx.session.get("reqId");
    /// ```
    #[must_use]
    #[napi(getter)]
    pub fn session(&self) -> JsSessionNamespace {
        JsSessionNamespace {
            inner: self.inner.clone(),
        }
    }
}

impl JsContext {
    /// Create a new `JsContext` wrapping a Rust `Context`.
    pub(crate) fn new(inner: blazen_core::Context) -> Self {
        Self { inner }
    }
}

// ---------------------------------------------------------------------------
// JsStateNamespace — persistable workflow state
// ---------------------------------------------------------------------------

/// Namespace for persistable workflow state.
///
/// Values stored via `state.set` / `state.setBytes` go into the
/// underlying `ContextInner.state` map and survive snapshots,
/// `pause()` / `resume()`, and checkpoint stores.
#[napi(js_name = "StateNamespace")]
pub struct JsStateNamespace {
    inner: blazen_core::Context,
}

#[napi]
#[allow(clippy::missing_errors_doc)]
impl JsStateNamespace {
    /// Store a JSON-serializable value under the given key.
    #[napi(ts_args_type = "key: string, value: Exclude<StateValue, Buffer>")]
    pub async fn set(&self, key: String, value: serde_json::Value) -> Result<()> {
        self.inner.set(&key, value).await;
        Ok(())
    }

    /// Retrieve a value previously stored under the given key.
    /// Returns `defaultValue` if the key is missing.
    #[napi(
        ts_args_type = "key: string, defaultValue?: StateValue",
        ts_return_type = "Promise<StateValue | null>"
    )]
    pub async fn get(
        &self,
        key: String,
        default: Option<serde_json::Value>,
    ) -> Result<serde_json::Value> {
        let val = match self.inner.get_value(&key).await {
            Some(StateValue::Json(v)) => v,
            Some(StateValue::Bytes(b)) => {
                let arr: Vec<serde_json::Value> =
                    b.0.into_iter()
                        .map(|byte| serde_json::Value::Number(byte.into()))
                        .collect();
                serde_json::Value::Array(arr)
            }
            Some(StateValue::Native(_)) | None => serde_json::Value::Null,
        };
        if val.is_null() {
            return Ok(default.unwrap_or(serde_json::Value::Null));
        }
        Ok(val)
    }

    /// Store raw binary data under the given key.
    #[napi(js_name = "setBytes")]
    pub async fn set_bytes(&self, key: String, data: Buffer) -> Result<()> {
        self.inner.set_bytes(&key, data.to_vec()).await;
        Ok(())
    }

    /// Retrieve raw binary data previously stored under the given key.
    /// Returns `defaultValue` if the key is missing.
    #[napi(
        js_name = "getBytes",
        ts_args_type = "key: string, defaultValue?: Buffer",
        ts_return_type = "Promise<Buffer | null>"
    )]
    pub async fn get_bytes(&self, key: String, default: Option<Buffer>) -> Result<Option<Buffer>> {
        Ok(self
            .inner
            .get_bytes(&key)
            .await
            .map(Buffer::from)
            .or(default))
    }
}

// ---------------------------------------------------------------------------
// JsSessionNamespace — in-process-only values (excluded from snapshots)
// ---------------------------------------------------------------------------

/// Namespace for in-process-only workflow values.
///
/// Values stored via `session.set` are kept in the
/// `ContextInner.objects` side-channel and are **excluded** from
/// snapshots. Use this for state that should not survive a
/// `pause()` / `resume()` round-trip (request IDs, rate-limit
/// counters, ephemeral caches, …).
///
/// For `session.set` values, identity preservation of JS class
/// instances through this namespace is **not** supported on the Node
/// bindings (see the module-level note on napi-rs threading). Values
/// are serialised via `serde_json::Value`, so you will get a plain
/// object back on `session.get`.
#[napi(js_name = "SessionNamespace")]
pub struct JsSessionNamespace {
    inner: blazen_core::Context,
}

#[napi]
#[allow(clippy::missing_errors_doc)]
impl JsSessionNamespace {
    /// Store a JSON-serializable value under the given key. The value
    /// is excluded from snapshots.
    #[napi(ts_args_type = "key: string, value: unknown")]
    pub async fn set(&self, key: String, value: serde_json::Value) -> Result<()> {
        // Stored in `ContextInner.objects` which is deliberately excluded
        // from `snapshot_state()`.
        self.inner.set_object(&key, value).await;
        Ok(())
    }

    /// Retrieve a value previously stored under the given key. Returns
    /// `defaultValue` if the key is missing.
    #[napi(
        ts_args_type = "key: string, defaultValue?: unknown",
        ts_return_type = "Promise<unknown>"
    )]
    pub async fn get(
        &self,
        key: String,
        default: Option<serde_json::Value>,
    ) -> Result<serde_json::Value> {
        let val = self.inner.get_object::<serde_json::Value>(&key).await;
        Ok(val.or(default).unwrap_or(serde_json::Value::Null))
    }

    /// Check whether a value exists under the given key.
    #[napi]
    pub async fn has(&self, key: String) -> Result<bool> {
        Ok(self.inner.has_object(&key).await)
    }

    /// Remove the value stored under the given key.
    #[napi]
    pub async fn remove(&self, key: String) -> Result<()> {
        self.inner.remove_object(&key).await;
        Ok(())
    }
}
