//! Shared workflow state accessible by all steps.
//!
//! [`Context`] wraps an `Arc<RwLock<ContextInner>>` so it can be cheaply
//! cloned and shared across concurrent step executions. It provides:
//!
//! - Typed key/value state storage
//! - Event emission to the internal routing queue
//! - Fan-in event collection
//! - Publishing events to the external streaming channel
//! - Workflow metadata (e.g. run ID)

use std::any::{Any, TypeId};
use std::collections::HashMap;
use std::sync::Arc;

use serde::Serialize;
use tokio::sync::{RwLock, broadcast, mpsc};
use uuid::Uuid;
use zagents_events::{AnyEvent, Event, EventEnvelope};

/// Type alias for the heterogeneous state map.
type StateMap = HashMap<String, Box<dyn Any + Send + Sync>>;

/// Internal state behind the `Arc<RwLock<_>>`.
struct ContextInner {
    /// Typed key/value store shared across all steps.
    state: StateMap,
    /// Sender side of the internal event routing channel.
    event_tx: mpsc::UnboundedSender<EventEnvelope>,
    /// Sender side of the external broadcast channel for streaming.
    stream_tx: broadcast::Sender<Box<dyn AnyEvent>>,
    /// Fan-in accumulator keyed by concrete `TypeId`.
    collected: HashMap<TypeId, Vec<Box<dyn AnyEvent>>>,
    /// Arbitrary JSON metadata (e.g. `run_id`, workflow name).
    metadata: HashMap<String, serde_json::Value>,
}

/// Shared workflow context.
///
/// Cheaply clonable handle to the shared state. Every step receives a
/// `Context` and can read/write state, emit events, and publish to the
/// external stream.
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
    /// Overwrites any previous value stored under the same key regardless
    /// of its type.
    pub async fn set<T: Send + Sync + 'static>(&self, key: &str, value: T) {
        let mut inner = self.inner.write().await;
        inner.state.insert(key.to_owned(), Box::new(value));
    }

    /// Retrieve a typed value previously stored under `key`.
    ///
    /// Returns `None` if the key does not exist or the stored value is not
    /// of type `T`.
    pub async fn get<T: Send + Sync + Clone + 'static>(&self, key: &str) -> Option<T> {
        let inner = self.inner.read().await;
        inner
            .state
            .get(key)
            .and_then(|v| v.downcast_ref::<T>())
            .cloned()
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
    pub async fn collect_events<E: Event>(&self, expected_count: usize) -> Option<Vec<E>> {
        let mut inner = self.inner.write().await;
        let type_id = TypeId::of::<E>();

        let collected = inner.collected.entry(type_id).or_default();
        if collected.len() >= expected_count {
            let drained: Vec<Box<dyn AnyEvent>> = collected.drain(..expected_count).collect();
            let mut results = Vec::with_capacity(drained.len());
            for boxed in drained {
                if let Some(concrete) = boxed.as_any().downcast_ref::<E>() {
                    results.push(concrete.clone());
                }
            }
            Some(results)
        } else {
            None
        }
    }

    /// Push a type-erased event into the fan-in accumulator.
    ///
    /// The event is stored under its concrete `TypeId` (obtained via
    /// `Any::type_id`).
    pub(crate) async fn push_collected(&self, event: Box<dyn AnyEvent>) {
        let mut inner = self.inner.write().await;
        let type_id = event.as_any().type_id();
        inner.collected.entry(type_id).or_default().push(event);
    }

    /// Clear the collection buffer for a specific event type.
    #[allow(dead_code)]
    pub(crate) async fn clear_collected<E: Event>(&self) {
        let mut inner = self.inner.write().await;
        let type_id = TypeId::of::<E>();
        inner.collected.remove(&type_id);
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
        use zagents_events::StartEvent;

        let ctx = test_context();
        let e1 = StartEvent {
            data: serde_json::json!(1),
        };
        let e2 = StartEvent {
            data: serde_json::json!(2),
        };

        ctx.push_collected(Box::new(e1)).await;
        // Not enough yet.
        assert!(ctx.collect_events::<StartEvent>(2).await.is_none());

        ctx.push_collected(Box::new(e2)).await;
        // Now we have 2.
        let events = ctx.collect_events::<StartEvent>(2).await.unwrap();
        assert_eq!(events.len(), 2);
        assert_eq!(events[0].data, serde_json::json!(1));
        assert_eq!(events[1].data, serde_json::json!(2));
    }
}
