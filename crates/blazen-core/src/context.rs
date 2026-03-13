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

use std::collections::HashMap;
use std::sync::Arc;

use blazen_events::{AnyEvent, Event, EventEnvelope};
use serde::Serialize;
use serde::de::DeserializeOwned;
use tokio::sync::{RwLock, broadcast, mpsc};
use uuid::Uuid;

/// Type alias for the JSON-backed state map.
type StateMap = HashMap<String, serde_json::Value>;

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
        inner.state.insert(key.to_owned(), json_value);
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
        inner
            .state
            .get(key)
            .and_then(|v| serde_json::from_value::<T>(v.clone()).ok())
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
    pub async fn snapshot_state(&self) -> HashMap<String, serde_json::Value> {
        let inner = self.inner.read().await;
        inner.state.clone()
    }

    /// Replace the state map wholesale.
    ///
    /// Used to restore state from a previous checkpoint. Any existing
    /// state is discarded.
    pub async fn restore_state(&self, state: HashMap<String, serde_json::Value>) {
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
        assert_eq!(snap.get("name").unwrap(), &serde_json::json!("alice"));
        assert_eq!(snap.get("count").unwrap(), &serde_json::json!(10));

        // Modify state
        ctx.set("name", "bob".to_string()).await;
        assert_eq!(ctx.get::<String>("name").await, Some("bob".to_string()));

        // Restore
        ctx.restore_state(snap).await;
        assert_eq!(ctx.get::<String>("name").await, Some("alice".to_string()));
        assert_eq!(ctx.get::<u32>("count").await, Some(10));
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
}
