//! # `ZAgents` Events
//!
//! Defines the core event traits and built-in event types used for
//! inter-component communication within the `ZAgents` framework.
//!
//! Every piece of data flowing between workflow steps is an [`Event`].
//! Events carry a stable string identifier ([`Event::event_type`]) used for
//! routing, serialization, and cross-language boundaries.
//!
//! The [`AnyEvent`] trait provides type-erased access so that the internal
//! event queue can hold heterogeneous events without generics.

use std::any::Any;
use std::fmt::Debug;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

// ---------------------------------------------------------------------------
// Core traits
// ---------------------------------------------------------------------------

/// The core routing trait.
///
/// Every piece of data flowing between workflow steps implements `Event`.
/// Implementors must be `Send + Sync + Debug + Clone + 'static` so that
/// events can be safely transferred across threads and stored in queues.
pub trait Event: Send + Sync + Debug + Clone + 'static {
    /// Stable string identifier for this event type.
    ///
    /// Used for routing, serialization, and cross-language boundaries.
    /// By convention the string takes the form `"crate::TypeName"`.
    fn event_type() -> &'static str
    where
        Self: Sized;

    /// Instance method version of [`Event::event_type`] for dynamic dispatch.
    fn event_type_id(&self) -> &'static str;

    /// Upcast to [`Any`] for type-erasure in the event queue.
    fn as_any(&self) -> &dyn Any;

    /// Clone into a boxed trait object.
    fn clone_boxed(&self) -> Box<dyn AnyEvent>;

    /// Serialize to JSON for cross-language boundaries and persistence.
    #[must_use]
    fn to_json(&self) -> serde_json::Value;
}

/// Type-erased event for the internal event queue.
///
/// This trait mirrors the instance methods of [`Event`] but drops the
/// `Clone` and `Sized` bounds so it can be used as a trait object.
pub trait AnyEvent: Send + Sync + Debug {
    /// Returns the stable string identifier for this event type.
    fn event_type_id(&self) -> &'static str;

    /// Upcast to [`Any`] for downcasting back to the concrete type.
    fn as_any(&self) -> &dyn Any;

    /// Clone into a new boxed trait object.
    fn clone_boxed(&self) -> Box<dyn AnyEvent>;

    /// Serialize to JSON for cross-language boundaries and persistence.
    #[must_use]
    fn to_json(&self) -> serde_json::Value;
}

// Blanket implementation: anything that is `Event + Serialize` is `AnyEvent`.
impl<T> AnyEvent for T
where
    T: Event + Serialize,
{
    fn event_type_id(&self) -> &'static str {
        Event::event_type_id(self)
    }

    fn as_any(&self) -> &dyn Any {
        Event::as_any(self)
    }

    fn clone_boxed(&self) -> Box<dyn AnyEvent> {
        Event::clone_boxed(self)
    }

    fn to_json(&self) -> serde_json::Value {
        Event::to_json(self)
    }
}

// Downcast helper on `dyn AnyEvent`.
impl dyn AnyEvent {
    /// Attempt to downcast the type-erased event to a concrete type `T`.
    ///
    /// Returns `Some(&T)` if the underlying event is of type `T`, or `None`
    /// otherwise.
    #[must_use]
    pub fn downcast_ref<T: Event>(&self) -> Option<&T> {
        self.as_any().downcast_ref::<T>()
    }
}

// Allow cloning boxed trait objects via `clone_boxed`.
impl Clone for Box<dyn AnyEvent> {
    fn clone(&self) -> Self {
        self.clone_boxed()
    }
}

// ---------------------------------------------------------------------------
// Built-in events
// ---------------------------------------------------------------------------

/// Emitted to kick off a workflow with arbitrary JSON data.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StartEvent {
    /// Arbitrary payload passed into the workflow at start.
    pub data: serde_json::Value,
}

impl Event for StartEvent {
    fn event_type() -> &'static str {
        "zagents::StartEvent"
    }

    fn event_type_id(&self) -> &'static str {
        "zagents::StartEvent"
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn clone_boxed(&self) -> Box<dyn AnyEvent> {
        Box::new(self.clone())
    }

    fn to_json(&self) -> serde_json::Value {
        serde_json::to_value(self).expect("StartEvent serialization should never fail")
    }
}

/// Emitted to signal that a workflow has completed with a result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StopEvent {
    /// The final result of the workflow.
    pub result: serde_json::Value,
}

impl Event for StopEvent {
    fn event_type() -> &'static str {
        "zagents::StopEvent"
    }

    fn event_type_id(&self) -> &'static str {
        "zagents::StopEvent"
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn clone_boxed(&self) -> Box<dyn AnyEvent> {
        Box::new(self.clone())
    }

    fn to_json(&self) -> serde_json::Value {
        serde_json::to_value(self).expect("StopEvent serialization should never fail")
    }
}

// ---------------------------------------------------------------------------
// EventEnvelope
// ---------------------------------------------------------------------------

/// Wraps an event with metadata for the internal queue.
///
/// Each envelope carries a unique id, a timestamp, and an optional source step
/// name so the runtime can trace event provenance.
#[derive(Debug)]
pub struct EventEnvelope {
    /// The type-erased event payload.
    pub event: Box<dyn AnyEvent>,
    /// The name of the step that produced this event, if any.
    pub source_step: Option<String>,
    /// When the envelope was created.
    pub timestamp: DateTime<Utc>,
    /// Unique identifier for this envelope.
    pub id: Uuid,
}

impl EventEnvelope {
    /// Create a new envelope, automatically filling in the timestamp and id.
    #[must_use]
    pub fn new(event: Box<dyn AnyEvent>, source_step: Option<String>) -> Self {
        Self {
            event,
            source_step,
            timestamp: Utc::now(),
            id: Uuid::new_v4(),
        }
    }
}

// ---------------------------------------------------------------------------
// Dynamic event type interning
// ---------------------------------------------------------------------------

/// Thread-safe registry that interns event type names into `&'static str`.
///
/// The [`Event`] trait requires `&'static str` for `event_type_id()`, but
/// dynamic events from foreign language bindings carry runtime type names.
/// We leak a small, bounded number of strings once and reuse them forever.
static EVENT_TYPE_REGISTRY: std::sync::LazyLock<dashmap::DashMap<String, &'static str>> =
    std::sync::LazyLock::new(dashmap::DashMap::new);

/// Intern a dynamic event type name, returning a `&'static str`.
///
/// If the name has been interned before, the same pointer is returned.
/// Otherwise the string is heap-allocated and leaked so it lives for
/// `'static`.
pub fn intern_event_type(name: &str) -> &'static str {
    if let Some(entry) = EVENT_TYPE_REGISTRY.get(name) {
        return entry.value();
    }
    let leaked: &'static str = Box::leak(name.to_string().into_boxed_str());
    EVENT_TYPE_REGISTRY.insert(name.to_string(), leaked);
    leaked
}

// ---------------------------------------------------------------------------
// DynamicEvent
// ---------------------------------------------------------------------------

/// A type-erased event that carries its type name and payload as JSON.
///
/// Used to transport events defined in foreign language bindings (Python,
/// TypeScript) through the Rust workflow engine.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DynamicEvent {
    /// The event type identifier (e.g. `"AnalyzeEvent"`).
    pub event_type: String,
    /// The event data as a JSON object.
    pub data: serde_json::Value,
}

impl Event for DynamicEvent {
    fn event_type() -> &'static str
    where
        Self: Sized,
    {
        // This static method cannot return a dynamic string, but it is only
        // used when you know the concrete type at compile time. For dynamic
        // dispatch the instance method `event_type_id` is used instead.
        "dynamic"
    }

    fn event_type_id(&self) -> &'static str {
        intern_event_type(&self.event_type)
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn clone_boxed(&self) -> Box<dyn AnyEvent> {
        Box::new(self.clone())
    }

    fn to_json(&self) -> serde_json::Value {
        serde_json::json!({
            "event_type": self.event_type,
            "data": self.data,
        })
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn start_event_type_id() {
        assert_eq!(StartEvent::event_type(), "zagents::StartEvent");
        let evt = StartEvent {
            data: serde_json::json!({"key": "value"}),
        };
        assert_eq!(Event::event_type_id(&evt), "zagents::StartEvent");
    }

    #[test]
    fn stop_event_type_id() {
        assert_eq!(StopEvent::event_type(), "zagents::StopEvent");
        let evt = StopEvent {
            result: serde_json::json!(42),
        };
        assert_eq!(Event::event_type_id(&evt), "zagents::StopEvent");
    }

    #[test]
    fn any_event_downcast() {
        let evt = StartEvent {
            data: serde_json::json!(null),
        };
        let boxed: Box<dyn AnyEvent> = Box::new(evt.clone());
        let downcasted = boxed.downcast_ref::<StartEvent>().unwrap();
        assert_eq!(downcasted.data, evt.data);

        // Wrong type returns None.
        assert!(boxed.downcast_ref::<StopEvent>().is_none());
    }

    #[test]
    fn clone_boxed_any_event() {
        let evt = StopEvent {
            result: serde_json::json!("done"),
        };
        let boxed: Box<dyn AnyEvent> = Box::new(evt);
        let cloned = boxed.clone();
        assert_eq!(boxed.event_type_id(), cloned.event_type_id());
        assert_eq!(boxed.to_json(), cloned.to_json());
    }

    #[test]
    fn event_envelope_constructor() {
        let evt = StartEvent {
            data: serde_json::json!({"hello": "world"}),
        };
        let envelope = EventEnvelope::new(Box::new(evt), Some("my_step".to_string()));
        assert_eq!(envelope.source_step.as_deref(), Some("my_step"));
        assert_eq!(envelope.event.event_type_id(), "zagents::StartEvent");
    }

    #[test]
    fn to_json_roundtrip() {
        let start = StartEvent {
            data: serde_json::json!({"nums": [1, 2, 3]}),
        };
        let json = Event::to_json(&start);
        let deserialized: StartEvent = serde_json::from_value(json).unwrap();
        assert_eq!(start.data, deserialized.data);

        let stop = StopEvent {
            result: serde_json::json!("ok"),
        };
        let json = Event::to_json(&stop);
        let deserialized: StopEvent = serde_json::from_value(json).unwrap();
        assert_eq!(stop.result, deserialized.result);
    }

    #[test]
    fn intern_event_type_returns_same_pointer() {
        let a = intern_event_type("TestEventInEvents");
        let b = intern_event_type("TestEventInEvents");
        assert!(std::ptr::eq(a, b));
    }

    #[test]
    fn dynamic_event_roundtrip() {
        let evt = DynamicEvent {
            event_type: "MyEvent".to_owned(),
            data: serde_json::json!({"key": "value"}),
        };
        let json = Event::to_json(&evt);
        assert_eq!(json["event_type"], "MyEvent");
        assert_eq!(json["data"]["key"], "value");
    }

    #[test]
    fn dynamic_event_type_id() {
        let evt = DynamicEvent {
            event_type: "CustomEvent".to_owned(),
            data: serde_json::json!({}),
        };
        assert_eq!(Event::event_type_id(&evt), "CustomEvent");
    }
}
