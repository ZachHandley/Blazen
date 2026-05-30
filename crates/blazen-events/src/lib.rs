//! # `Blazen` Events
//!
//! Defines the core event traits and built-in event types used for
//! inter-component communication within the `Blazen` framework.
//!
//! Every piece of data flowing between workflow steps is an [`Event`].
//! Events carry a stable string identifier ([`Event::event_type`]) used for
//! routing, serialization, and cross-language boundaries.
//!
//! The [`AnyEvent`] trait provides type-erased access so that the internal
//! event queue can hold heterogeneous events without generics.

use std::any::Any;
use std::fmt::Debug;
use std::sync::{Arc, OnceLock};

use chrono::{DateTime, Utc};
use serde::de::{self, Deserializer, MapAccess, Visitor};
use serde::ser::{SerializeStruct, Serializer};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

mod progress_event;
mod usage_event;

pub use progress_event::{ProgressEvent, ProgressKind};
pub use usage_event::{Modality, UsageEvent};

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

    /// Recover the live, in-memory native handle backing this event, if any.
    ///
    /// Most events have no native handle and return `None`. Dynamic events
    /// produced by foreign language bindings may carry a live object (e.g. a
    /// `Py<PyAny>`) so the binding can recover the original instance without a
    /// `DynamicEvent` downcast or a JSON round-trip.
    ///
    /// This is the Send-native live lane of the unified session-ref store:
    /// `Send` natives (e.g. `Py<PyAny>`, which is `Send + Sync + 'static`)
    /// may be held directly here for in-process identity — the `Arc` rides
    /// along with each event clone and is simply `None` after a
    /// snapshot/resume round-trip (dropped, not errored). `!Send` host
    /// objects (JS/wasm handles bound to their host heap) CANNOT live in an
    /// `Arc<dyn Any + Send + Sync>`; they must instead be inserted into the
    /// `SessionRefRegistry` and referenced from the event's JSON payload via
    /// the `{"__blazen_session_ref__": "<uuid>"}` marker, which resolves
    /// through the same unified store on the host thread.
    #[must_use]
    fn native_handle(&self) -> Option<Arc<dyn Any + Send + Sync>> {
        None
    }
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

    /// Recover the live, in-memory native handle backing this event, if any.
    #[must_use]
    fn native_handle(&self) -> Option<Arc<dyn Any + Send + Sync>>;
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

    fn native_handle(&self) -> Option<Arc<dyn Any + Send + Sync>> {
        Event::native_handle(self)
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
        static REGISTER: std::sync::Once = std::sync::Once::new();
        REGISTER.call_once(|| {
            register_event_deserializer("blazen::StartEvent", |value| {
                serde_json::from_value::<StartEvent>(value)
                    .ok()
                    .map(|e| Box::new(e) as _)
            });
        });
        "blazen::StartEvent"
    }

    fn event_type_id(&self) -> &'static str {
        "blazen::StartEvent"
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
        static REGISTER: std::sync::Once = std::sync::Once::new();
        REGISTER.call_once(|| {
            register_event_deserializer("blazen::StopEvent", |value| {
                serde_json::from_value::<StopEvent>(value)
                    .ok()
                    .map(|e| Box::new(e) as _)
            });
        });
        "blazen::StopEvent"
    }

    fn event_type_id(&self) -> &'static str {
        "blazen::StopEvent"
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

/// Emitted by a step to request human input. Triggers auto-pause.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InputRequestEvent {
    /// Unique ID for this request (for matching response).
    pub request_id: String,
    /// The question/prompt to show the human.
    pub prompt: String,
    /// Optional structured metadata (choices, type hints, etc.).
    pub metadata: serde_json::Value,
}

impl Event for InputRequestEvent {
    fn event_type() -> &'static str {
        static REGISTER: std::sync::Once = std::sync::Once::new();
        REGISTER.call_once(|| {
            register_event_deserializer("blazen::InputRequestEvent", |value| {
                serde_json::from_value::<InputRequestEvent>(value)
                    .ok()
                    .map(|e| Box::new(e) as _)
            });
        });
        "blazen::InputRequestEvent"
    }

    fn event_type_id(&self) -> &'static str {
        "blazen::InputRequestEvent"
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn clone_boxed(&self) -> Box<dyn AnyEvent> {
        Box::new(self.clone())
    }

    fn to_json(&self) -> serde_json::Value {
        serde_json::to_value(self).expect("InputRequestEvent serialization should never fail")
    }
}

/// The human's response, injected on resume.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InputResponseEvent {
    /// Matches the `InputRequestEvent.request_id`.
    pub request_id: String,
    /// The human's answer.
    pub response: serde_json::Value,
}

impl Event for InputResponseEvent {
    fn event_type() -> &'static str {
        static REGISTER: std::sync::Once = std::sync::Once::new();
        REGISTER.call_once(|| {
            register_event_deserializer("blazen::InputResponseEvent", |value| {
                serde_json::from_value::<InputResponseEvent>(value)
                    .ok()
                    .map(|e| Box::new(e) as _)
            });
        });
        "blazen::InputResponseEvent"
    }

    fn event_type_id(&self) -> &'static str {
        "blazen::InputResponseEvent"
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn clone_boxed(&self) -> Box<dyn AnyEvent> {
        Box::new(self.clone())
    }

    fn to_json(&self) -> serde_json::Value {
        serde_json::to_value(self).expect("InputResponseEvent serialization should never fail")
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
// Event deserializer registry
// ---------------------------------------------------------------------------

/// Function signature for deserializing JSON into a concrete event type.
pub type EventDeserializerFn = fn(serde_json::Value) -> Option<Box<dyn AnyEvent>>;

/// Global registry mapping event type strings to deserializer functions.
///
/// When a workflow is resumed from a snapshot, pending events are stored as
/// JSON. This registry allows the runtime to reconstruct concrete event types
/// from that JSON, avoiding the `DynamicEvent` wrapper that breaks
/// `downcast_ref`.
static EVENT_DESERIALIZER_REGISTRY: std::sync::LazyLock<
    dashmap::DashMap<&'static str, EventDeserializerFn>,
> = std::sync::LazyLock::new(dashmap::DashMap::new);

/// Register a deserializer function for a given event type string.
///
/// Typically called once per event type, guarded by [`std::sync::Once`],
/// inside the `event_type()` method of each concrete event.
pub fn register_event_deserializer(event_type: &'static str, deserializer: EventDeserializerFn) {
    EVENT_DESERIALIZER_REGISTRY.insert(event_type, deserializer);
}

/// Attempt to deserialize a JSON value into a concrete event type using the
/// registry.
///
/// Returns `Some(boxed_event)` if a deserializer is registered for the given
/// event type and deserialization succeeds, or `None` otherwise.
pub fn try_deserialize_event(
    event_type: &str,
    data: &serde_json::Value,
) -> Option<Box<dyn AnyEvent>> {
    let entry = EVENT_DESERIALIZER_REGISTRY.get(event_type)?;
    let deserializer = *entry.value();
    deserializer(data.clone())
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

/// Function signature for serializing a live native handle to JSON.
///
/// Registered once by the language binding (e.g. `blazen-py`) so that
/// [`DynamicEvent::to_json`] can materialize a native-backed event WITHOUT
/// `blazen-events` depending on `pyo3` (or any binding crate). Returns `None`
/// if the handle cannot be serialized, in which case the cached `data` field is
/// used as a fallback.
pub type NativeSerializerFn = fn(&Arc<dyn Any + Send + Sync>) -> Option<serde_json::Value>;

/// Process-wide hook used to serialize native event handles lazily.
static NATIVE_SERIALIZER: OnceLock<NativeSerializerFn> = OnceLock::new();

/// Register the process-wide native serializer hook.
///
/// Should be called once during binding initialization. Subsequent calls are
/// ignored (the first registration wins).
pub fn register_native_serializer(f: NativeSerializerFn) {
    let _ = NATIVE_SERIALIZER.set(f);
}

/// A type-erased event that carries its type name and payload.
///
/// Used to transport events defined in foreign language bindings (Python,
/// TypeScript) through the Rust workflow engine.
///
/// A `DynamicEvent` may be backed by a live native object (`native`) instead of
/// eagerly-computed JSON. When `native` is `Some`, the `data` field is NOT
/// authoritative; JSON is computed lazily via [`DynamicEvent::to_json`] and
/// cached in `cached_json`. When `native` is `None`, `data` is authoritative.
///
/// `Clone` is cheap: it bumps the `Arc`s for the native handle and the shared
/// JSON cache rather than copying the underlying object.
#[derive(Clone)]
pub struct DynamicEvent {
    /// The event type identifier (e.g. `"AnalyzeEvent"`).
    pub event_type: String,
    /// The event data as JSON. Authoritative ONLY when `native` is `None`.
    pub data: serde_json::Value,
    /// Live in-memory object backing this event (e.g. a `Py<PyAny>`).
    ///
    /// Skipped by serde; materialized to JSON lazily via the registered
    /// [`NativeSerializerFn`].
    pub native: Option<Arc<dyn Any + Send + Sync>>,
    /// Lazily-filled JSON cache, computed from `native` on first `to_json`.
    /// Shared across clones via `Arc`.
    cached_json: Arc<OnceLock<serde_json::Value>>,
}

impl DynamicEvent {
    /// Construct a `DynamicEvent` from an event type and JSON payload.
    ///
    /// No native handle is attached; `data` is authoritative.
    #[must_use]
    pub fn from_json(event_type: impl Into<String>, data: serde_json::Value) -> Self {
        Self {
            event_type: event_type.into(),
            data,
            native: None,
            cached_json: Arc::new(OnceLock::new()),
        }
    }

    /// Construct a `DynamicEvent` backed by a live native handle.
    ///
    /// `data` is left as `Null`; JSON is computed lazily from `native` on the
    /// first call to [`DynamicEvent::to_json`].
    ///
    /// Only `Send + Sync + 'static` natives belong here (e.g. `Py<PyAny>`):
    /// they are held directly for in-process identity and are dropped — not
    /// errored — across a snapshot/resume boundary (`native` becomes `None`
    /// after deserialization). `!Send` host objects (JS/wasm handles) must
    /// NOT use this constructor; insert them into the `SessionRefRegistry`
    /// and reference the registry key from the event payload via the
    /// `{"__blazen_session_ref__": "<uuid>"}` marker instead.
    #[must_use]
    pub fn with_native(event_type: impl Into<String>, native: Arc<dyn Any + Send + Sync>) -> Self {
        Self {
            event_type: event_type.into(),
            data: serde_json::Value::Null,
            native: Some(native),
            cached_json: Arc::new(OnceLock::new()),
        }
    }
}

impl Debug for DynamicEvent {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // `native` (a `dyn Any` handle) and `cached_json` (an internal lazy
        // cache) are intentionally omitted — neither is meaningfully Debuggable
        // and both are derived state. `finish_non_exhaustive()` records that
        // (and satisfies clippy::missing_fields_in_debug under `-D warnings`).
        f.debug_struct("DynamicEvent")
            .field("event_type", &self.event_type)
            .field("data", &self.data)
            .finish_non_exhaustive()
    }
}

impl Serialize for DynamicEvent {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        // Materialize native-backed events first so the wire payload is the
        // real data, not the placeholder `Null`. Wire shape is exactly
        // {event_type, data}.
        let data = Event::to_json(self);
        let mut state = serializer.serialize_struct("DynamicEvent", 2)?;
        state.serialize_field("event_type", &self.event_type)?;
        state.serialize_field("data", &data)?;
        state.end()
    }
}

impl<'de> Deserialize<'de> for DynamicEvent {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        #[derive(serde::Deserialize)]
        #[serde(field_identifier, rename_all = "snake_case")]
        enum Field {
            EventType,
            Data,
        }

        struct DynamicEventVisitor;

        impl<'de> Visitor<'de> for DynamicEventVisitor {
            type Value = DynamicEvent;

            fn expecting(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                f.write_str("struct DynamicEvent with fields event_type and data")
            }

            fn visit_map<A>(self, mut map: A) -> Result<DynamicEvent, A::Error>
            where
                A: MapAccess<'de>,
            {
                let mut event_type: Option<String> = None;
                let mut data: Option<serde_json::Value> = None;
                while let Some(key) = map.next_key::<Field>()? {
                    match key {
                        Field::EventType => {
                            if event_type.is_some() {
                                return Err(de::Error::duplicate_field("event_type"));
                            }
                            event_type = Some(map.next_value()?);
                        }
                        Field::Data => {
                            if data.is_some() {
                                return Err(de::Error::duplicate_field("data"));
                            }
                            data = Some(map.next_value()?);
                        }
                    }
                }
                let event_type =
                    event_type.ok_or_else(|| de::Error::missing_field("event_type"))?;
                let data = data.ok_or_else(|| de::Error::missing_field("data"))?;
                Ok(DynamicEvent::from_json(event_type, data))
            }
        }

        deserializer.deserialize_struct(
            "DynamicEvent",
            &["event_type", "data"],
            DynamicEventVisitor,
        )
    }
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
        match &self.native {
            Some(native) => self
                .cached_json
                .get_or_init(|| {
                    NATIVE_SERIALIZER
                        .get()
                        .and_then(|f| f(native))
                        .unwrap_or_else(|| self.data.clone())
                })
                .clone(),
            None => self.data.clone(),
        }
    }

    fn native_handle(&self) -> Option<Arc<dyn Any + Send + Sync>> {
        self.native.clone()
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
        assert_eq!(StartEvent::event_type(), "blazen::StartEvent");
        let evt = StartEvent {
            data: serde_json::json!({"key": "value"}),
        };
        assert_eq!(Event::event_type_id(&evt), "blazen::StartEvent");
    }

    #[test]
    fn stop_event_type_id() {
        assert_eq!(StopEvent::event_type(), "blazen::StopEvent");
        let evt = StopEvent {
            result: serde_json::json!(42),
        };
        assert_eq!(Event::event_type_id(&evt), "blazen::StopEvent");
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
        assert_eq!(envelope.event.event_type_id(), "blazen::StartEvent");
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
        let evt = DynamicEvent::from_json("MyEvent", serde_json::json!({"key": "value"}));
        let json = Event::to_json(&evt);
        // DynamicEvent::to_json() now returns the flat data directly.
        assert_eq!(json["key"], "value");
    }

    #[test]
    fn dynamic_event_type_id() {
        let evt = DynamicEvent::from_json("CustomEvent", serde_json::json!({}));
        assert_eq!(Event::event_type_id(&evt), "CustomEvent");
    }

    #[test]
    fn input_request_event_type_id() {
        assert_eq!(InputRequestEvent::event_type(), "blazen::InputRequestEvent");
        let evt = InputRequestEvent {
            request_id: "req-1".to_string(),
            prompt: "What is your name?".to_string(),
            metadata: serde_json::json!({"choices": ["Alice", "Bob"]}),
        };
        assert_eq!(Event::event_type_id(&evt), "blazen::InputRequestEvent");
    }

    #[test]
    fn input_response_event_type_id() {
        assert_eq!(
            InputResponseEvent::event_type(),
            "blazen::InputResponseEvent"
        );
        let evt = InputResponseEvent {
            request_id: "req-1".to_string(),
            response: serde_json::json!("Alice"),
        };
        assert_eq!(Event::event_type_id(&evt), "blazen::InputResponseEvent");
    }

    #[test]
    fn input_request_event_roundtrip() {
        let evt = InputRequestEvent {
            request_id: "req-42".to_string(),
            prompt: "Pick a number".to_string(),
            metadata: serde_json::json!({"min": 1, "max": 100}),
        };
        let json = Event::to_json(&evt);
        let deserialized: InputRequestEvent = serde_json::from_value(json).unwrap();
        assert_eq!(evt.request_id, deserialized.request_id);
        assert_eq!(evt.prompt, deserialized.prompt);
        assert_eq!(evt.metadata, deserialized.metadata);
    }

    #[test]
    fn input_response_event_roundtrip() {
        let evt = InputResponseEvent {
            request_id: "req-42".to_string(),
            response: serde_json::json!(77),
        };
        let json = Event::to_json(&evt);
        let deserialized: InputResponseEvent = serde_json::from_value(json).unwrap();
        assert_eq!(evt.request_id, deserialized.request_id);
        assert_eq!(evt.response, deserialized.response);
    }

    #[test]
    fn input_request_event_downcast() {
        let evt = InputRequestEvent {
            request_id: "req-99".to_string(),
            prompt: "Confirm?".to_string(),
            metadata: serde_json::json!(null),
        };
        let boxed: Box<dyn AnyEvent> = Box::new(evt.clone());
        let downcasted = boxed.downcast_ref::<InputRequestEvent>().unwrap();
        assert_eq!(downcasted.request_id, evt.request_id);

        // Wrong type returns None.
        assert!(boxed.downcast_ref::<InputResponseEvent>().is_none());
    }

    #[test]
    fn input_response_event_downcast() {
        let evt = InputResponseEvent {
            request_id: "req-99".to_string(),
            response: serde_json::json!({"answer": true}),
        };
        let boxed: Box<dyn AnyEvent> = Box::new(evt.clone());
        let downcasted = boxed.downcast_ref::<InputResponseEvent>().unwrap();
        assert_eq!(downcasted.request_id, evt.request_id);

        // Wrong type returns None.
        assert!(boxed.downcast_ref::<InputRequestEvent>().is_none());
    }
}
