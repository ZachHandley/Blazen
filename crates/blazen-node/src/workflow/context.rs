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

use blazen_core::StateValue;
use napi::bindgen_prelude::*;
use napi_derive::napi;

use super::event::js_value_to_any_event;

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
    /// Returns `Buffer` for binary data, the original JS value for JSON data,
    /// or `null` if the key does not exist.
    #[napi(ts_return_type = "Promise<StateValue | null>")]
    pub async fn get(&self, key: String) -> Result<serde_json::Value> {
        match self.inner.get_value(&key).await {
            Some(StateValue::Json(v)) => Ok(v),
            Some(StateValue::Bytes(b)) => {
                // Return bytes as a JSON array so the value isn't silently dropped.
                // For proper binary round-trip, use getBytes().
                let arr: Vec<serde_json::Value> =
                    b.0.into_iter()
                        .map(|byte| serde_json::Value::Number(byte.into()))
                        .collect();
                Ok(serde_json::Value::Array(arr))
            }
            Some(StateValue::Native(_)) | None => Ok(serde_json::Value::Null),
        }
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
    /// Returns `null` if the key does not exist or the stored value is
    /// not binary data.
    #[napi(js_name = "getBytes")]
    pub async fn get_bytes(&self, key: String) -> Result<Option<Buffer>> {
        Ok(self.inner.get_bytes(&key).await.map(Buffer::from))
    }

    /// Get the workflow run ID.
    #[napi(js_name = "runId")]
    pub async fn run_id(&self) -> Result<String> {
        let id = self.inner.run_id().await;
        Ok(id.to_string())
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
    #[napi(ts_return_type = "Promise<StateValue | null>")]
    pub async fn get(&self, key: String) -> Result<serde_json::Value> {
        match self.inner.get_value(&key).await {
            Some(StateValue::Json(v)) => Ok(v),
            Some(StateValue::Bytes(b)) => {
                let arr: Vec<serde_json::Value> =
                    b.0.into_iter()
                        .map(|byte| serde_json::Value::Number(byte.into()))
                        .collect();
                Ok(serde_json::Value::Array(arr))
            }
            Some(StateValue::Native(_)) | None => Ok(serde_json::Value::Null),
        }
    }

    /// Store raw binary data under the given key.
    #[napi(js_name = "setBytes")]
    pub async fn set_bytes(&self, key: String, data: Buffer) -> Result<()> {
        self.inner.set_bytes(&key, data.to_vec()).await;
        Ok(())
    }

    /// Retrieve raw binary data previously stored under the given key.
    #[napi(js_name = "getBytes")]
    pub async fn get_bytes(&self, key: String) -> Result<Option<Buffer>> {
        Ok(self.inner.get_bytes(&key).await.map(Buffer::from))
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
    /// `null` if the key does not exist.
    #[napi(ts_return_type = "Promise<unknown>")]
    pub async fn get(&self, key: String) -> Result<serde_json::Value> {
        Ok(self
            .inner
            .get_object::<serde_json::Value>(&key)
            .await
            .unwrap_or(serde_json::Value::Null))
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
