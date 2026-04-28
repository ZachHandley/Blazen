//! Typed JS wrappers for the [`blazen_events`] event types and the
//! deserializer registry / type-interning free functions.
//!
//! The existing module [`super::event`] only exposes events as plain
//! `serde_json::Value` round-trips. This module adds:
//!
//! - [`JsDynamicEvent`] -- a `DynamicEvent` class with a constructor and
//!   `eventType` / `data` getters so JS callers can build dynamic events
//!   ergonomically without having to remember the `{ type, ... }` flat
//!   object convention used by [`super::event`].
//! - [`JsEventEnvelope`] -- a `#[napi(object)]` mirror of
//!   [`blazen_events::EventEnvelope`] suitable for streaming envelopes
//!   out of the workflow runtime.
//! - [`JsInputRequestEvent`] / [`JsInputResponseEvent`] -- typed
//!   `#[napi(object)]` mirrors of the built-in human-in-the-loop events.
//! - [`register_event_deserializer`] / [`try_deserialize_event`] /
//!   [`intern_event_type`] -- free functions exposing the cross-language
//!   deserializer registry and type-name interning helpers.
//!
//! ## Deserializer bridging
//!
//! [`blazen_events::register_event_deserializer`] stores a bare
//! `fn(serde_json::Value) -> Option<Box<dyn AnyEvent>>` pointer per event
//! type. Bare `fn` pointers cannot capture a per-name JS callback, so the
//! Node binding keeps a parallel map of `ThreadsafeFunction`s keyed by
//! event type name. [`try_deserialize_event`] consults the Rust registry
//! first (so built-in [`StartEvent`] / [`StopEvent`] / [`InputRequestEvent`]
//! / [`InputResponseEvent`] / Rust-side derive-registered types still
//! work) and falls back to the JS map when the Rust registry has no entry
//! for the requested name.

use std::collections::HashMap;
use std::sync::{Arc, LazyLock, Mutex};

use blazen_events::DynamicEvent;
use napi::Status;
use napi::bindgen_prelude::*;
use napi::threadsafe_function::ThreadsafeFunction;
use napi_derive::napi;

// ---------------------------------------------------------------------------
// JsDynamicEvent
// ---------------------------------------------------------------------------

/// A type-erased event that carries its type name and JSON payload.
///
/// Mirrors [`blazen_events::DynamicEvent`]. Use this when you want to
/// build a dynamic event in JS without going through the plain
/// `{ type: "MyEvent", ...data }` flat-object convention used by the
/// step-handler return path.
///
/// ```javascript
/// const ev = new DynamicEvent("MyEvent", { text: "hi", score: 0.9 });
/// ev.eventType; // "MyEvent"
/// ev.data;      // { text: "hi", score: 0.9 }
/// ```
#[napi(js_name = "DynamicEvent")]
pub struct JsDynamicEvent {
    inner: DynamicEvent,
}

#[napi]
#[allow(clippy::must_use_candidate, clippy::missing_errors_doc)]
impl JsDynamicEvent {
    /// Build a new dynamic event.
    #[napi(constructor)]
    pub fn new(event_type: String, data: serde_json::Value) -> Self {
        Self {
            inner: DynamicEvent { event_type, data },
        }
    }

    /// The event type identifier (e.g. `"MyEvent"`).
    #[napi(getter, js_name = "eventType")]
    pub fn event_type(&self) -> String {
        self.inner.event_type.clone()
    }

    /// The event data payload.
    #[napi(getter)]
    pub fn data(&self) -> serde_json::Value {
        self.inner.data.clone()
    }

    /// Convert to the flat JS object form used by step-handler return
    /// values: `{ type, ...data }`.
    #[napi(js_name = "toJSON")]
    pub fn to_json(&self) -> serde_json::Value {
        let mut obj = match &self.inner.data {
            serde_json::Value::Object(map) => map.clone(),
            other => {
                let mut m = serde_json::Map::new();
                m.insert("data".to_owned(), other.clone());
                m
            }
        };
        obj.insert(
            "type".to_owned(),
            serde_json::Value::String(self.inner.event_type.clone()),
        );
        serde_json::Value::Object(obj)
    }
}

// ---------------------------------------------------------------------------
// JsEventEnvelope
// ---------------------------------------------------------------------------

/// Wraps an event payload with metadata: the source step that produced
/// it (if any) and the event data as a JS-friendly JSON value.
///
/// This is a plain JS object, not a class, so it can be freely
/// constructed / destructured on the JS side. It mirrors
/// [`blazen_events::EventEnvelope`] but without the `id` / `timestamp`
/// fields, which are bookkeeping internal to the Rust runtime.
#[napi(object, js_name = "EventEnvelope")]
pub struct JsEventEnvelope {
    /// The event type identifier.
    #[napi(js_name = "eventType")]
    pub event_type: String,
    /// The name of the step that produced this event, if any.
    #[napi(js_name = "sourceStep")]
    pub source_step: Option<String>,
    /// The event payload as JSON.
    #[napi(js_name = "eventData")]
    pub event_data: serde_json::Value,
}

impl JsEventEnvelope {
    /// Build a [`JsEventEnvelope`] from a [`blazen_events::EventEnvelope`].
    #[must_use]
    pub fn from_envelope(envelope: &blazen_events::EventEnvelope) -> Self {
        Self {
            event_type: envelope.event.event_type_id().to_owned(),
            source_step: envelope.source_step.clone(),
            event_data: envelope.event.to_json(),
        }
    }
}

// ---------------------------------------------------------------------------
// JsInputRequestEvent
// ---------------------------------------------------------------------------

/// A request for human input emitted by a workflow step.
///
/// Mirrors [`blazen_events::InputRequestEvent`]. Workflows publish this
/// event to pause and wait for a [`JsInputResponseEvent`] on resume.
#[napi(object, js_name = "InputRequestEvent")]
pub struct JsInputRequestEvent {
    /// Unique ID for this request, used to match the corresponding response.
    #[napi(js_name = "requestId")]
    pub request_id: String,
    /// The question/prompt to show the human, if any.
    pub prompt: Option<String>,
    /// Optional structured metadata (choices, type hints, etc.).
    pub metadata: serde_json::Value,
}

impl JsInputRequestEvent {
    /// Build a [`JsInputRequestEvent`] from a
    /// [`blazen_events::InputRequestEvent`].
    #[must_use]
    pub fn from_event(event: &blazen_events::InputRequestEvent) -> Self {
        let prompt = if event.prompt.is_empty() {
            None
        } else {
            Some(event.prompt.clone())
        };
        Self {
            request_id: event.request_id.clone(),
            prompt,
            metadata: event.metadata.clone(),
        }
    }

    /// Convert to a [`blazen_events::InputRequestEvent`].
    #[must_use]
    pub fn into_event(self) -> blazen_events::InputRequestEvent {
        blazen_events::InputRequestEvent {
            request_id: self.request_id,
            prompt: self.prompt.unwrap_or_default(),
            metadata: self.metadata,
        }
    }
}

// ---------------------------------------------------------------------------
// JsInputResponseEvent
// ---------------------------------------------------------------------------

/// A human's response to a [`JsInputRequestEvent`], injected on resume.
///
/// Mirrors [`blazen_events::InputResponseEvent`].
#[napi(object, js_name = "InputResponseEvent")]
pub struct JsInputResponseEvent {
    /// Matches the `InputRequestEvent.requestId`.
    #[napi(js_name = "requestId")]
    pub request_id: String,
    /// The human's answer.
    pub response: serde_json::Value,
}

impl JsInputResponseEvent {
    /// Build a [`JsInputResponseEvent`] from a
    /// [`blazen_events::InputResponseEvent`].
    #[must_use]
    pub fn from_event(event: &blazen_events::InputResponseEvent) -> Self {
        Self {
            request_id: event.request_id.clone(),
            response: event.response.clone(),
        }
    }

    /// Convert to a [`blazen_events::InputResponseEvent`].
    #[must_use]
    pub fn into_event(self) -> blazen_events::InputResponseEvent {
        blazen_events::InputResponseEvent {
            request_id: self.request_id,
            response: self.response,
        }
    }
}

// ---------------------------------------------------------------------------
// JS-side deserializer registry
// ---------------------------------------------------------------------------

/// Threadsafe JS callback that accepts a JSON value and returns the
/// concrete event payload as JSON (or a Promise thereof).
///
/// Generic parameters mirror the conventions used elsewhere in this
/// crate -- see the doc comment on [`super::workflow`]'s `StepHandlerTsfn`
/// for a detailed walkthrough.
type DeserializerTsfn = ThreadsafeFunction<
    serde_json::Value,
    Promise<serde_json::Value>,
    serde_json::Value,
    Status,
    false,
    true,
>;

/// Per-name JS deserializer callbacks.
///
/// `blazen_events::EVENT_DESERIALIZER_REGISTRY` only accepts bare
/// `fn` pointers, which cannot capture a per-name [`ThreadsafeFunction`].
/// We therefore keep a parallel map and consult it from
/// [`try_deserialize_event`] when the Rust registry has no entry.
static JS_DESERIALIZER_REGISTRY: LazyLock<Mutex<HashMap<String, Arc<DeserializerTsfn>>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

/// Register a JS-side deserializer callback for the given event type
/// name.
///
/// The callback receives the raw event JSON and must return the concrete
/// event payload as JSON (synchronously or as a Promise). Replaces any
/// previously-registered callback for the same name.
///
/// ```javascript
/// registerEventDeserializer("MyEvent", async (json) => {
///   return { ...json, normalized: true };
/// });
/// ```
#[allow(clippy::missing_panics_doc)]
#[napi(js_name = "registerEventDeserializer")]
pub fn register_event_deserializer(name: String, deserializer: DeserializerTsfn) {
    let mut map = JS_DESERIALIZER_REGISTRY
        .lock()
        .expect("JS deserializer registry poisoned");
    map.insert(name, Arc::new(deserializer));
}

/// Attempt to deserialize an event payload using the registry.
///
/// The Rust registry (populated by built-in events and `#[derive(Event)]`
/// types) is consulted first. If no Rust deserializer is registered for
/// the given name, the JS-side registry populated by
/// [`register_event_deserializer`] is consulted. If neither has a
/// deserializer, `null` is returned.
///
/// `jsonStr` is parsed as JSON before being handed to the deserializer.
/// A parse error is reported as a thrown napi error; an unknown event
/// type is *not* an error -- it returns `null`.
#[allow(clippy::missing_panics_doc)]
#[napi(js_name = "tryDeserializeEvent")]
#[allow(clippy::missing_errors_doc)]
pub async fn try_deserialize_event(
    name: String,
    json_str: String,
) -> Result<Option<serde_json::Value>> {
    let value: serde_json::Value = serde_json::from_str(&json_str)
        .map_err(|e| napi::Error::from_reason(format!("invalid JSON: {e}")))?;

    // Consult the Rust registry first.
    if let Some(boxed) = blazen_events::try_deserialize_event(&name, &value) {
        return Ok(Some(boxed.to_json()));
    }

    // Fall back to the JS-side registry.
    let tsfn = {
        let map = JS_DESERIALIZER_REGISTRY
            .lock()
            .expect("JS deserializer registry poisoned");
        map.get(&name).cloned()
    };

    let Some(tsfn) = tsfn else {
        return Ok(None);
    };

    let promise = tsfn
        .call_async(value)
        .await
        .map_err(|e| napi::Error::from_reason(format!("deserializer call failed: {e}")))?;
    let resolved = promise
        .await
        .map_err(|e| napi::Error::from_reason(format!("deserializer promise rejected: {e}")))?;
    Ok(Some(resolved))
}

/// Intern a dynamic event type name, returning the canonical string.
///
/// Repeated calls with the same name return the same interned value.
/// Useful for keeping the JS-side event-type strings in sync with the
/// Rust-side `&'static str` pool used by [`blazen_events::Event`].
#[allow(clippy::needless_pass_by_value)]
#[napi(js_name = "internEventType")]
#[must_use]
pub fn intern_event_type(name: String) -> String {
    blazen_events::intern_event_type(&name).to_owned()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use blazen_events::AnyEvent;

    #[test]
    fn dynamic_event_round_trips_through_to_json() {
        let ev = JsDynamicEvent::new(
            "MyEvent".to_owned(),
            serde_json::json!({"text": "hi", "score": 0.9}),
        );
        let flat = ev.to_json();
        assert_eq!(flat["type"], "MyEvent");
        assert_eq!(flat["text"], "hi");
        assert!((flat["score"].as_f64().unwrap() - 0.9).abs() < 1e-9);
    }

    #[test]
    fn dynamic_event_to_json_wraps_non_object_data() {
        let ev = JsDynamicEvent::new("Tick".to_owned(), serde_json::json!(42));
        let flat = ev.to_json();
        assert_eq!(flat["type"], "Tick");
        assert_eq!(flat["data"], 42);
    }

    #[test]
    fn input_request_event_round_trips() {
        let original = blazen_events::InputRequestEvent {
            request_id: "req-1".to_owned(),
            prompt: "Confirm?".to_owned(),
            metadata: serde_json::json!({"choices": ["yes", "no"]}),
        };
        let js = JsInputRequestEvent::from_event(&original);
        assert_eq!(js.request_id, "req-1");
        assert_eq!(js.prompt.as_deref(), Some("Confirm?"));
        let back = js.into_event();
        assert_eq!(back.request_id, original.request_id);
        assert_eq!(back.prompt, original.prompt);
        assert_eq!(back.metadata, original.metadata);
    }

    #[test]
    fn input_request_event_empty_prompt_becomes_none() {
        let original = blazen_events::InputRequestEvent {
            request_id: "req-2".to_owned(),
            prompt: String::new(),
            metadata: serde_json::Value::Null,
        };
        let js = JsInputRequestEvent::from_event(&original);
        assert!(js.prompt.is_none());
        let back = js.into_event();
        assert_eq!(back.prompt, "");
    }

    #[test]
    fn input_response_event_round_trips() {
        let original = blazen_events::InputResponseEvent {
            request_id: "req-3".to_owned(),
            response: serde_json::json!({"answer": true}),
        };
        let js = JsInputResponseEvent::from_event(&original);
        assert_eq!(js.request_id, "req-3");
        let back = js.into_event();
        assert_eq!(back.request_id, original.request_id);
        assert_eq!(back.response, original.response);
    }

    #[test]
    fn event_envelope_from_envelope_extracts_metadata() {
        let event: Box<dyn AnyEvent> = Box::new(blazen_events::StartEvent {
            data: serde_json::json!({"key": "value"}),
        });
        let envelope = blazen_events::EventEnvelope::new(event, Some("step-a".to_owned()));
        let js = JsEventEnvelope::from_envelope(&envelope);
        assert_eq!(js.event_type, "blazen::StartEvent");
        assert_eq!(js.source_step.as_deref(), Some("step-a"));
        assert_eq!(js.event_data["data"]["key"], "value");
    }

    #[test]
    fn intern_event_type_matches_blazen_events() {
        let interned = intern_event_type("CustomEvent".to_owned());
        assert_eq!(interned, "CustomEvent");
        // Calling again returns the same logical string.
        let again = intern_event_type("CustomEvent".to_owned());
        assert_eq!(interned, again);
    }
}
