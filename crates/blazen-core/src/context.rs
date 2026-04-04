//! Shared workflow state accessible by all steps.
//!
//! [`Context`] wraps an `Arc<RwLock<ContextInner>>` so it can be cheaply
//! cloned and shared across concurrent step executions. It provides:
//!
//! - Typed key/value state storage (backed by JSON for serializability)
//! - Event emission to the internal routing queue
//! - Fan-in event collection
//! - Publishing events to the external streaming channel
//! - Workflow metadata (e.g. run ID)
//! - State snapshotting and restoration for pause/resume/checkpoint

use std::any::Any;
use std::collections::HashMap;
use std::sync::Arc;

use blazen_events::{AnyEvent, Event, EventEnvelope};
use serde::Serialize;
use serde::de::DeserializeOwned;
use tokio::sync::{RwLock, broadcast, mpsc};
use uuid::Uuid;

use crate::value::{BytesWrapper, StateValue};

/// Type alias for the state map (supports both JSON and binary values).
type StateMap = HashMap<String, StateValue>;

/// Internal state behind the `Arc<RwLock<_>>`.
struct ContextInner {
    /// JSON-serialized key/value store shared across all steps.
    state: StateMap,
    /// Sender side of the internal event routing channel.
    event_tx: mpsc::UnboundedSender<EventEnvelope>,
    /// Sender side of the external broadcast channel for streaming.
    stream_tx: broadcast::Sender<Box<dyn AnyEvent>>,
    /// Fan-in accumulator keyed by event type string.
    collected: HashMap<String, Vec<serde_json::Value>>,
    /// Arbitrary JSON metadata (e.g. `run_id`, workflow name).
    metadata: HashMap<String, serde_json::Value>,
    /// Opaque in-process objects (DB connections, file handles, etc.).
    /// NOT serialized — excluded from snapshots. Bindings store platform-specific
    /// types here and downcast on retrieval.
    objects: HashMap<String, Box<dyn Any + Send + Sync>>,
}

/// Shared workflow context.
///
/// Cheaply clonable handle to the shared state. Every step receives a
/// `Context` and can read/write state, emit events, and publish to the
/// external stream.
///
/// State values are stored as JSON internally, enabling serialization for
/// pause/resume/checkpoint functionality. Users can still use ergonomic
/// typed accessors (`set`/`get`) as long as their types implement
/// `Serialize`/`DeserializeOwned`.
#[derive(Clone)]
pub struct Context {
    inner: Arc<RwLock<ContextInner>>,
}

impl Context {
    // -----------------------------------------------------------------
    // Construction (crate-internal)
    // -----------------------------------------------------------------

    /// Create a new context wired to the given channels.
    pub(crate) fn new(
        event_tx: mpsc::UnboundedSender<EventEnvelope>,
        stream_tx: broadcast::Sender<Box<dyn AnyEvent>>,
    ) -> Self {
        Self {
            inner: Arc::new(RwLock::new(ContextInner {
                state: HashMap::new(),
                event_tx,
                stream_tx,
                collected: HashMap::new(),
                metadata: HashMap::new(),
                objects: HashMap::new(),
            })),
        }
    }

    // -----------------------------------------------------------------
    // Public state accessors
    // -----------------------------------------------------------------

    /// Store a typed value under `key`.
    ///
    /// The value is serialized to JSON before storage. Overwrites any
    /// previous value stored under the same key regardless of its type.
    ///
    /// # Panics
    ///
    /// Panics if the value cannot be serialized to JSON. In practice this
    /// should never happen for well-formed serde types.
    pub async fn set<T: Serialize + Send + Sync + 'static>(&self, key: &str, value: T) {
        let json_value =
            serde_json::to_value(&value).expect("Context::set: value must be JSON-serializable");
        let mut inner = self.inner.write().await;
        inner
            .state
            .insert(key.to_owned(), StateValue::Json(json_value));
    }

    /// Retrieve a typed value previously stored under `key`.
    ///
    /// The stored JSON is deserialized back into type `T`. Returns `None`
    /// if the key does not exist or the stored JSON cannot be deserialized
    /// into `T`.
    pub async fn get<T: DeserializeOwned + Send + Sync + Clone + 'static>(
        &self,
        key: &str,
    ) -> Option<T> {
        let inner = self.inner.read().await;
        inner.state.get(key).and_then(|sv| match sv {
            StateValue::Json(v) => serde_json::from_value::<T>(v.clone()).ok(),
            StateValue::Bytes(_) | StateValue::Native(_) => None,
        })
    }

    /// Store a raw [`StateValue`] directly.
    ///
    /// Used by language bindings for polymorphic dispatch (e.g. storing
    /// platform-serialized opaque objects via [`StateValue::Native`]).
    pub async fn set_value(&self, key: &str, value: StateValue) {
        let mut inner = self.inner.write().await;
        inner.state.insert(key.to_owned(), value);
    }

    /// Retrieve the raw [`StateValue`] stored under `key`.
    ///
    /// Returns `None` if the key does not exist. Unlike [`get`](Self::get),
    /// this returns the value regardless of its variant.
    pub async fn get_value(&self, key: &str) -> Option<StateValue> {
        let inner = self.inner.read().await;
        inner.state.get(key).cloned()
    }

    // -----------------------------------------------------------------
    // Opaque object storage (non-serializable, in-process only)
    // -----------------------------------------------------------------

    /// Store a live in-process object under `key`.
    ///
    /// The object is NOT serialized and will NOT survive snapshots or
    /// pause/resume. Use this for DB connections, file handles, and other
    /// resources that must be shared across steps within a single run.
    pub async fn set_object<T: Any + Send + Sync + 'static>(&self, key: &str, value: T) {
        let mut inner = self.inner.write().await;
        inner.objects.insert(key.to_owned(), Box::new(value));
    }

    /// Retrieve a live in-process object previously stored under `key`.
    ///
    /// Returns `None` if the key does not exist or the stored type does
    /// not match `T`.
    pub async fn get_object<T: Any + Send + Sync + Clone + 'static>(&self, key: &str) -> Option<T> {
        let inner = self.inner.read().await;
        inner
            .objects
            .get(key)
            .and_then(|v| v.downcast_ref::<T>())
            .cloned()
    }

    /// Remove a live in-process object stored under `key`.
    pub async fn remove_object(&self, key: &str) {
        let mut inner = self.inner.write().await;
        inner.objects.remove(key);
    }

    /// Check whether an opaque object exists under `key`.
    pub async fn has_object(&self, key: &str) -> bool {
        let inner = self.inner.read().await;
        inner.objects.contains_key(key)
    }

    /// Store raw binary data under `key`.
    ///
    /// Useful for files, images, audio, and other binary artifacts that
    /// should not be JSON-serialized.
    pub async fn set_bytes(&self, key: &str, data: Vec<u8>) {
        let mut inner = self.inner.write().await;
        inner
            .state
            .insert(key.to_owned(), StateValue::Bytes(BytesWrapper(data)));
    }

    /// Retrieve raw binary data previously stored under `key`.
    ///
    /// Returns `None` if the key does not exist or the stored value is
    /// a JSON variant rather than bytes.
    pub async fn get_bytes(&self, key: &str) -> Option<Vec<u8>> {
        let inner = self.inner.read().await;
        inner.state.get(key).and_then(|sv| match sv {
            StateValue::Bytes(b) => Some(b.0.clone()),
            StateValue::Json(_) | StateValue::Native(_) => None,
        })
    }

    // -----------------------------------------------------------------
    // Event emission
    // -----------------------------------------------------------------

    /// Emit an event into the internal routing queue.
    ///
    /// The event will be picked up by the event loop and routed to any
    /// step whose `accepts` list includes its event type.
    pub async fn send_event<E: Event + Serialize>(&self, event: E) {
        let inner = self.inner.read().await;
        let envelope = EventEnvelope::new(Box::new(event), None);
        // Ignore send errors -- the receiver may have been dropped if the
        // workflow already terminated.
        let _ = inner.event_tx.send(envelope);
    }

    /// Publish an event to the external broadcast stream.
    ///
    /// Consumers that called [`crate::WorkflowHandler::stream_events`] will
    /// receive this event. Unlike [`send_event`](Self::send_event), this does
    /// **not** route the event through the internal step registry.
    pub async fn write_event_to_stream<E: Event + Serialize>(&self, event: E) {
        let inner = self.inner.read().await;
        // Ignore send errors -- there may be no active subscribers.
        let _ = inner.stream_tx.send(Box::new(event));
    }

    // -----------------------------------------------------------------
    // Fan-in collection
    // -----------------------------------------------------------------

    /// Accumulate events of type `E` until `expected_count` are available.
    ///
    /// Returns `Some(Vec<E>)` when exactly `expected_count` events have been
    /// collected, or `None` if not enough have arrived yet.
    ///
    /// Once the threshold is reached the internal buffer for this type is
    /// cleared automatically so a subsequent call starts fresh.
    pub async fn collect_events<E: Event + DeserializeOwned>(
        &self,
        expected_count: usize,
    ) -> Option<Vec<E>> {
        let mut inner = self.inner.write().await;
        let type_key = E::event_type().to_owned();

        let collected = inner.collected.entry(type_key).or_default();
        if collected.len() >= expected_count {
            let drained: Vec<serde_json::Value> = collected.drain(..expected_count).collect();
            let mut results = Vec::with_capacity(drained.len());
            for json_val in drained {
                if let Ok(concrete) = serde_json::from_value::<E>(json_val) {
                    results.push(concrete);
                }
            }
            Some(results)
        } else {
            None
        }
    }

    /// Push a type-erased event into the fan-in accumulator.
    ///
    /// The event is serialized to JSON and stored under its event type
    /// string (obtained via `AnyEvent::event_type_id`).
    pub(crate) async fn push_collected(&self, event: &dyn AnyEvent) {
        let mut inner = self.inner.write().await;
        let type_key = event.event_type_id().to_owned();
        let json_val = event.to_json();
        inner.collected.entry(type_key).or_default().push(json_val);
    }

    /// Clear the collection buffer for a specific event type.
    #[allow(dead_code)]
    pub(crate) async fn clear_collected<E: Event>(&self) {
        let mut inner = self.inner.write().await;
        let type_key = E::event_type().to_owned();
        inner.collected.remove(&type_key);
    }

    // -----------------------------------------------------------------
    // Snapshotting & restoration
    // -----------------------------------------------------------------

    /// Returns a clone of the entire state map.
    ///
    /// Useful for checkpointing or pausing a workflow so it can be
    /// resumed later.
    pub async fn snapshot_state(&self) -> HashMap<String, StateValue> {
        let inner = self.inner.read().await;
        inner.state.clone()
    }

    /// Replace the state map wholesale.
    ///
    /// Used to restore state from a previous checkpoint. Any existing
    /// state is discarded.
    pub async fn restore_state(&self, state: HashMap<String, StateValue>) {
        let mut inner = self.inner.write().await;
        inner.state = state;
    }

    /// Returns a clone of the collected events map (serialized as JSON).
    ///
    /// Useful for checkpointing fan-in state alongside the main state map.
    pub async fn snapshot_collected(&self) -> HashMap<String, Vec<serde_json::Value>> {
        let inner = self.inner.read().await;
        inner.collected.clone()
    }

    /// Replace the collected events map wholesale.
    ///
    /// Used to restore fan-in state from a previous checkpoint. Any existing
    /// collected events are discarded.
    pub async fn restore_collected(&self, collected: HashMap<String, Vec<serde_json::Value>>) {
        let mut inner = self.inner.write().await;
        inner.collected = collected;
    }

    /// Returns a clone of the metadata map.
    ///
    /// Useful for checkpointing metadata alongside the main state map.
    pub async fn snapshot_metadata(&self) -> HashMap<String, serde_json::Value> {
        let inner = self.inner.read().await;
        inner.metadata.clone()
    }

    /// Replace the metadata map wholesale.
    ///
    /// Used to restore metadata from a previous checkpoint. Any existing
    /// metadata is discarded.
    pub(crate) async fn restore_metadata(&self, metadata: HashMap<String, serde_json::Value>) {
        let mut inner = self.inner.write().await;
        inner.metadata = metadata;
    }

    // -----------------------------------------------------------------
    // Metadata
    // -----------------------------------------------------------------

    /// Get the workflow run ID from metadata.
    ///
    /// # Panics
    ///
    /// Panics if the `run_id` metadata key was never set (this is always
    /// set by the workflow engine before any step executes).
    pub async fn run_id(&self) -> Uuid {
        let inner = self.inner.read().await;
        inner
            .metadata
            .get("run_id")
            .and_then(|v| v.as_str())
            .and_then(|s| Uuid::parse_str(s).ok())
            .expect("run_id must be set in workflow metadata")
    }

    /// Store a metadata key/value pair.
    pub(crate) async fn set_metadata(&self, key: &str, value: serde_json::Value) {
        let mut inner = self.inner.write().await;
        inner.metadata.insert(key.to_owned(), value);
    }

    /// Send a sentinel event through the broadcast stream to signal that
    /// no more events will be published.
    ///
    /// Consumers that check for `"blazen::StreamEnd"` can use this to
    /// terminate their iteration.
    pub(crate) async fn signal_stream_end(&self) {
        self.write_event_to_stream(blazen_events::DynamicEvent {
            event_type: "blazen::StreamEnd".to_owned(),
            data: serde_json::Value::Null,
        })
        .await;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper to build a context with throw-away channels.
    fn test_context() -> Context {
        let (event_tx, _event_rx) = mpsc::unbounded_channel();
        let (stream_tx, _stream_rx) = broadcast::channel(16);
        Context::new(event_tx, stream_tx)
    }

    #[tokio::test]
    async fn set_and_get_typed_value() {
        let ctx = test_context();
        ctx.set("counter", 42_u64).await;
        assert_eq!(ctx.get::<u64>("counter").await, Some(42));
    }

    #[tokio::test]
    async fn get_wrong_type_returns_none() {
        let ctx = test_context();
        ctx.set("counter", 42_u64).await;
        // JSON number 42 can be deserialized as a String? No -- serde_json
        // will fail to parse a number as a String, so this returns None.
        assert_eq!(ctx.get::<String>("counter").await, None);
    }

    #[tokio::test]
    async fn get_missing_key_returns_none() {
        let ctx = test_context();
        assert_eq!(ctx.get::<u64>("nope").await, None);
    }

    #[tokio::test]
    async fn run_id_roundtrip() {
        let ctx = test_context();
        let id = Uuid::new_v4();
        ctx.set_metadata("run_id", serde_json::Value::String(id.to_string()))
            .await;
        assert_eq!(ctx.run_id().await, id);
    }

    #[tokio::test]
    async fn collect_events_accumulation() {
        use blazen_events::StartEvent;

        let ctx = test_context();
        let e1 = StartEvent {
            data: serde_json::json!(1),
        };
        let e2 = StartEvent {
            data: serde_json::json!(2),
        };

        ctx.push_collected(&e1).await;
        // Not enough yet.
        assert!(ctx.collect_events::<StartEvent>(2).await.is_none());

        ctx.push_collected(&e2).await;
        // Now we have 2.
        let events = ctx.collect_events::<StartEvent>(2).await.unwrap();
        assert_eq!(events.len(), 2);
        assert_eq!(events[0].data, serde_json::json!(1));
        assert_eq!(events[1].data, serde_json::json!(2));
    }

    #[tokio::test]
    async fn snapshot_and_restore_state() {
        let ctx = test_context();
        ctx.set("name", "alice".to_string()).await;
        ctx.set("count", 10_u32).await;

        // Snapshot
        let snap = ctx.snapshot_state().await;
        assert_eq!(snap.len(), 2);
        assert_eq!(
            snap.get("name").unwrap(),
            &StateValue::Json(serde_json::json!("alice"))
        );
        assert_eq!(
            snap.get("count").unwrap(),
            &StateValue::Json(serde_json::json!(10))
        );

        // Modify state
        ctx.set("name", "bob".to_string()).await;
        assert_eq!(ctx.get::<String>("name").await, Some("bob".to_string()));

        // Restore
        ctx.restore_state(snap).await;
        assert_eq!(ctx.get::<String>("name").await, Some("alice".to_string()));
        assert_eq!(ctx.get::<u32>("count").await, Some(10));
    }

    #[tokio::test]
    async fn set_and_get_bytes() {
        let ctx = test_context();
        let data = vec![0xDE, 0xAD, 0xBE, 0xEF];
        ctx.set_bytes("binary", data.clone()).await;

        assert_eq!(ctx.get_bytes("binary").await, Some(data));
        // get<T> should return None for bytes values.
        assert_eq!(ctx.get::<String>("binary").await, None);
    }

    #[tokio::test]
    async fn get_bytes_returns_none_for_json() {
        let ctx = test_context();
        ctx.set("key", "value".to_string()).await;
        assert_eq!(ctx.get_bytes("key").await, None);
    }

    #[tokio::test]
    async fn get_bytes_returns_none_for_missing_key() {
        let ctx = test_context();
        assert_eq!(ctx.get_bytes("nope").await, None);
    }

    #[tokio::test]
    async fn snapshot_collected() {
        use blazen_events::StartEvent;

        let ctx = test_context();
        let e1 = StartEvent {
            data: serde_json::json!("a"),
        };
        ctx.push_collected(&e1).await;

        let snap = ctx.snapshot_collected().await;
        assert_eq!(snap.len(), 1);
        let start_events = snap.get("blazen::StartEvent").unwrap();
        assert_eq!(start_events.len(), 1);
    }

    #[tokio::test]
    async fn set_value_and_get_value() {
        let ctx = test_context();
        let native = StateValue::native(vec![0x80, 0x04, 0x95]);
        ctx.set_value("pickled", native.clone()).await;

        let retrieved = ctx.get_value("pickled").await;
        assert_eq!(retrieved, Some(native));
    }

    #[tokio::test]
    async fn get_value_returns_all_variants() {
        let ctx = test_context();
        ctx.set("json_key", "hello".to_string()).await;
        ctx.set_bytes("bytes_key", vec![1, 2, 3]).await;
        ctx.set_value("native_key", StateValue::native(vec![4, 5, 6]))
            .await;

        assert!(ctx.get_value("json_key").await.unwrap().is_json());
        assert!(ctx.get_value("bytes_key").await.unwrap().is_bytes());
        assert!(ctx.get_value("native_key").await.unwrap().is_native());
        assert!(ctx.get_value("missing").await.is_none());
    }

    #[tokio::test]
    async fn get_returns_none_for_native() {
        let ctx = test_context();
        ctx.set_value("key", StateValue::native(vec![0x80, 0x04]))
            .await;
        assert_eq!(ctx.get::<String>("key").await, None);
    }

    #[tokio::test]
    async fn get_bytes_returns_none_for_native() {
        let ctx = test_context();
        ctx.set_value("key", StateValue::native(vec![0x80, 0x04]))
            .await;
        assert_eq!(ctx.get_bytes("key").await, None);
    }

    #[tokio::test]
    async fn set_and_get_object() {
        let ctx = test_context();
        ctx.set_object("counter", 42_i32).await;
        assert_eq!(ctx.get_object::<i32>("counter").await, Some(42));
    }

    #[tokio::test]
    async fn get_object_wrong_type_returns_none() {
        let ctx = test_context();
        ctx.set_object("counter", 42_i32).await;
        assert_eq!(ctx.get_object::<String>("counter").await, None);
    }

    #[tokio::test]
    async fn get_object_missing_key_returns_none() {
        let ctx = test_context();
        assert_eq!(ctx.get_object::<i32>("nope").await, None);
    }

    #[tokio::test]
    async fn remove_object() {
        let ctx = test_context();
        ctx.set_object("key", "value".to_string()).await;
        assert!(ctx.has_object("key").await);
        ctx.remove_object("key").await;
        assert!(!ctx.has_object("key").await);
    }

    #[tokio::test]
    async fn objects_excluded_from_snapshot() {
        let ctx = test_context();
        ctx.set_object("live", 42_i32).await;
        ctx.set("json_key", "hello".to_string()).await;
        let snap = ctx.snapshot_state().await;
        // Snapshot only contains state map entries, not objects
        assert!(snap.contains_key("json_key"));
        assert!(!snap.contains_key("live"));
    }

    #[tokio::test]
    async fn snapshot_includes_bytes_and_native() {
        let ctx = test_context();
        ctx.set("json_key", "hello".to_string()).await;
        ctx.set_bytes("bytes_key", vec![0xDE, 0xAD]).await;
        ctx.set_value("native_key", StateValue::native(vec![0x80, 0x04]))
            .await;

        let snap = ctx.snapshot_state().await;
        assert!(snap.get("json_key").unwrap().is_json());
        assert!(snap.get("bytes_key").unwrap().is_bytes());
        assert!(snap.get("native_key").unwrap().is_native());

        // Restore into a fresh context and verify
        let ctx2 = test_context();
        ctx2.restore_state(snap).await;
        assert_eq!(
            ctx2.get::<String>("json_key").await,
            Some("hello".to_string())
        );
        assert_eq!(ctx2.get_bytes("bytes_key").await, Some(vec![0xDE, 0xAD]));
        assert_eq!(
            ctx2.get_value("native_key").await.unwrap().as_native(),
            Some([0x80, 0x04].as_slice())
        );
    }

    #[tokio::test]
    async fn set_overwrites_previous_value() {
        let ctx = test_context();
        ctx.set("key", 1_u64).await;
        assert_eq!(ctx.get::<u64>("key").await, Some(1));
        ctx.set("key", 2_u64).await;
        assert_eq!(ctx.get::<u64>("key").await, Some(2));
        // Overwrite JSON with bytes
        ctx.set_bytes("key", vec![1, 2, 3]).await;
        assert_eq!(ctx.get::<u64>("key").await, None);
        assert_eq!(ctx.get_bytes("key").await, Some(vec![1, 2, 3]));
    }
}
