//! JavaScript wrapper for the workflow [`Context`](blazen_core::Context).
//!
//! `set()` stores any JSON-serializable value. `setBytes()` stores binary data
//! as `Buffer`. `get()` returns the stored value regardless of variant —
//! `Buffer` for binary data, the original JS value for JSON data.

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
}

impl JsContext {
    /// Create a new `JsContext` wrapping a Rust `Context`.
    pub(crate) fn new(inner: blazen_core::Context) -> Self {
        Self { inner }
    }
}
