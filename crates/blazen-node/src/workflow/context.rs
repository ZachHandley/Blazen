//! JavaScript wrapper for the workflow [`Context`](blazen_core::Context).
//!
//! Values are exchanged as `serde_json::Value` which napi-rs automatically
//! converts to/from JavaScript objects via the `serde-json` feature.

use napi::bindgen_prelude::*;
use napi_derive::napi;

use super::event::js_value_to_any_event;

/// Shared workflow context accessible by all steps.
///
/// Provides typed key/value storage, event emission, and stream publishing.
/// All values are stored as JSON internally.
#[napi(js_name = "Context")]
pub struct JsContext {
    pub(crate) inner: blazen_core::Context,
}

#[napi]
#[allow(clippy::missing_errors_doc)]
impl JsContext {
    /// Store a value under the given key.
    ///
    /// The value is stored as JSON internally.
    #[napi]
    pub async fn set(&self, key: String, value: serde_json::Value) -> Result<()> {
        self.inner.set(&key, value).await;
        Ok(())
    }

    /// Retrieve a value previously stored under the given key.
    ///
    /// Returns `null` if the key does not exist.
    #[napi]
    pub async fn get(&self, key: String) -> Result<serde_json::Value> {
        let val: Option<serde_json::Value> = self.inner.get(&key).await;
        Ok(val.unwrap_or(serde_json::Value::Null))
    }

    /// Emit an event into the internal routing queue.
    ///
    /// The event will be routed to any step whose `eventTypes` list includes
    /// its event type. The event object must have a `type` field.
    #[napi(js_name = "sendEvent")]
    pub async fn send_event(&self, event: serde_json::Value) -> Result<()> {
        let any_event = js_value_to_any_event(&event);

        // We need to send the event through the context. Since send_event
        // requires Event + Serialize, and DynamicEvent satisfies both,
        // we convert to DynamicEvent first.
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
    /// Useful for storing files, images, serialized objects, or any binary
    /// data that should not be JSON-serialized. The data persists through
    /// pause/resume snapshots.
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
