//! WASM bindings for the [`blazen_events`] typed surface.
//!
//! Exposes the runtime [`DynamicEvent`](blazen_events::DynamicEvent) wrapper
//! plus the [`EventEnvelope`](blazen_events::EventEnvelope) and
//! [`InputRequestEvent`](blazen_events::InputRequestEvent) /
//! [`InputResponseEvent`](blazen_events::InputResponseEvent) data types so
//! TypeScript can construct, inspect, and round-trip events through the
//! Blazen workflow engine.
//!
//! # Pattern
//!
//! - [`WasmDynamicEvent`] is a [`#[wasm_bindgen]`] class with a constructor
//!   and getters because callers need to build it from a JS-side
//!   `(eventType, data)` pair and read fields back.
//! - [`WasmEventEnvelope`], [`WasmInputRequestEvent`], and
//!   [`WasmInputResponseEvent`] are plain data structs that derive
//!   [`tsify_next::Tsify`] so they appear as TypeScript interfaces in the
//!   generated `.d.ts` and round-trip through `serde-wasm-bindgen` directly.
//! - The free functions [`register_event_deserializer`],
//!   [`try_deserialize_event`], and [`intern_event_type`] are thin wrappers
//!   over the [`blazen_events`] registry helpers, adapted to JS-friendly
//!   signatures (string in, string/`JsValue` out, JS [`js_sys::Function`]
//!   for callbacks).
//!
//! The interned `&'static str` returned by [`intern_event_type`] is owned by
//! the global registry in `blazen-events`; the JS side only ever sees an
//! owned [`String`] copy, so there's no lifetime hazard at the boundary.

use std::cell::RefCell;

use blazen_events::{
    AnyEvent, DynamicEvent, EventEnvelope, InputRequestEvent, InputResponseEvent,
    intern_event_type as blazen_intern_event_type,
    register_event_deserializer as blazen_register_event_deserializer,
    try_deserialize_event as blazen_try_deserialize_event,
};
use serde::{Deserialize, Serialize};
use tsify_next::Tsify;
use wasm_bindgen::prelude::*;

// ---------------------------------------------------------------------------
// DynamicEvent
// ---------------------------------------------------------------------------

/// JavaScript-facing wrapper around [`blazen_events::DynamicEvent`].
///
/// Constructed from JS as `new DynamicEvent(eventType, data)` where `data`
/// is any JSON-serialisable value. Internally the payload is stored as a
/// [`serde_json::Value`] so it round-trips cleanly through the engine's
/// dynamic event registry.
#[wasm_bindgen(js_name = "DynamicEvent")]
pub struct WasmDynamicEvent {
    inner: DynamicEvent,
}

#[wasm_bindgen(js_class = "DynamicEvent")]
impl WasmDynamicEvent {
    /// Build a new dynamic event with the given type identifier and payload.
    ///
    /// `event_type` is the stable string used for routing and serialization
    /// (e.g. `"AnalyzeEvent"`). `data` may be any JS value; it's converted
    /// to a [`serde_json::Value`] via `serde-wasm-bindgen`.
    ///
    /// # Errors
    ///
    /// Rejects with a stringified error if `data` cannot be converted to a
    /// JSON value.
    #[wasm_bindgen(constructor)]
    pub fn new(event_type: String, data: JsValue) -> Result<WasmDynamicEvent, JsValue> {
        let data: serde_json::Value = serde_wasm_bindgen::from_value(data)
            .map_err(|e| JsValue::from_str(&format!("invalid event data: {e}")))?;
        Ok(Self {
            inner: DynamicEvent { event_type, data },
        })
    }

    /// Return the event type identifier.
    #[wasm_bindgen(getter, js_name = "eventType")]
    #[must_use]
    pub fn event_type(&self) -> String {
        self.inner.event_type.clone()
    }

    /// Return the event payload as a plain JS value.
    ///
    /// # Errors
    ///
    /// Rejects with a stringified error if the underlying JSON value cannot
    /// be marshalled back into a [`JsValue`].
    #[wasm_bindgen(getter)]
    pub fn data(&self) -> Result<JsValue, JsValue> {
        let serializer = serde_wasm_bindgen::Serializer::new().serialize_maps_as_objects(true);
        self.inner
            .data
            .serialize(&serializer)
            .map_err(|e| JsValue::from_str(&format!("marshal failed: {e}")))
    }
}

// ---------------------------------------------------------------------------
// Plain data types (Tsify)
// ---------------------------------------------------------------------------

/// Wrapper mirroring [`blazen_events::EventEnvelope`] for the JS boundary.
///
/// The native [`EventEnvelope`] holds a `Box<dyn AnyEvent>` which can't
/// cross the WASM ABI directly, so this struct flattens the event payload
/// to `event_type` + `data` plus the optional `source_step` produced by the
/// runtime.
#[derive(Debug, Clone, Serialize, Deserialize, Tsify)]
#[tsify(into_wasm_abi, from_wasm_abi)]
pub struct WasmEventEnvelope {
    /// Stable string identifier for the wrapped event.
    pub event_type: String,
    /// JSON payload of the wrapped event.
    pub data: serde_json::Value,
    /// Name of the step that produced this envelope, if any.
    pub source_step: Option<String>,
}

impl WasmEventEnvelope {
    /// Build an envelope view from a real [`EventEnvelope`].
    #[must_use]
    pub fn from_native(envelope: &EventEnvelope) -> Self {
        Self {
            event_type: envelope.event.event_type_id().to_owned(),
            data: envelope.event.to_json(),
            source_step: envelope.source_step.clone(),
        }
    }
}

/// TS-facing copy of [`blazen_events::InputRequestEvent`].
#[derive(Debug, Clone, Serialize, Deserialize, Tsify)]
#[tsify(into_wasm_abi, from_wasm_abi)]
pub struct WasmInputRequestEvent {
    /// Unique id used to match a response back to this request.
    pub request_id: String,
    /// Prompt to display to the human.
    pub prompt: String,
    /// Optional structured metadata (choices, type hints, etc.).
    pub metadata: serde_json::Value,
}

impl From<InputRequestEvent> for WasmInputRequestEvent {
    fn from(value: InputRequestEvent) -> Self {
        Self {
            request_id: value.request_id,
            prompt: value.prompt,
            metadata: value.metadata,
        }
    }
}

impl From<WasmInputRequestEvent> for InputRequestEvent {
    fn from(value: WasmInputRequestEvent) -> Self {
        Self {
            request_id: value.request_id,
            prompt: value.prompt,
            metadata: value.metadata,
        }
    }
}

/// TS-facing copy of [`blazen_events::InputResponseEvent`].
#[derive(Debug, Clone, Serialize, Deserialize, Tsify)]
#[tsify(into_wasm_abi, from_wasm_abi)]
pub struct WasmInputResponseEvent {
    /// Matches the originating [`WasmInputRequestEvent::request_id`].
    pub request_id: String,
    /// The human's answer.
    pub response: serde_json::Value,
}

impl From<InputResponseEvent> for WasmInputResponseEvent {
    fn from(value: InputResponseEvent) -> Self {
        Self {
            request_id: value.request_id,
            response: value.response,
        }
    }
}

impl From<WasmInputResponseEvent> for InputResponseEvent {
    fn from(value: WasmInputResponseEvent) -> Self {
        Self {
            request_id: value.request_id,
            response: value.response,
        }
    }
}

// ---------------------------------------------------------------------------
// JS-callback registry adapter
// ---------------------------------------------------------------------------

thread_local! {
    /// Per-thread (single-threaded on wasm32) registry of JS-side
    /// deserializer callbacks keyed by event type.
    ///
    /// The wrapper around them lives as a static `fn` pointer in the native
    /// `blazen-events` registry; that pointer reaches back into this map at
    /// dispatch time to find the right JS [`js_sys::Function`] to call.
    static JS_DESERIALIZERS: RefCell<std::collections::HashMap<String, js_sys::Function>> =
        RefCell::new(std::collections::HashMap::new());

    /// Most recent event type the native trampoline was invoked with.
    ///
    /// `blazen-events` uses a `fn` pointer (no closure environment), so the
    /// trampoline can't capture which event type it was registered for —
    /// instead we set this slot just before dispatching and read it back
    /// inside the trampoline.
    static CURRENT_EVENT_TYPE: RefCell<Option<String>> = const { RefCell::new(None) };
}

/// Register a JS callback as the deserializer for `name`.
///
/// The callback is invoked as `callback(jsonString)` and is expected to
/// return a JS object (or `undefined`/`null` on failure). Calling this
/// twice for the same `name` overwrites the previous binding.
#[wasm_bindgen(js_name = "registerEventDeserializer")]
pub fn register_event_deserializer(name: String, callback: js_sys::Function) {
    JS_DESERIALIZERS.with(|map| {
        map.borrow_mut().insert(name.clone(), callback);
    });

    let interned = blazen_intern_event_type(&name);
    blazen_register_event_deserializer(interned, native_trampoline);
}

/// Native trampoline registered with [`blazen_events`].
///
/// The native registry stores plain `fn` pointers (no closure environment)
/// so we use [`CURRENT_EVENT_TYPE`] as a thread-local to communicate which
/// JS callback to invoke. JS callbacks are intentionally `unimplemented!`
/// at the AnyEvent return path because the native engine can't actually use
/// a JS-built event; this trampoline exists so that
/// [`try_deserialize_event`] can route through the same registry the rest
/// of the engine uses.
fn native_trampoline(value: serde_json::Value) -> Option<Box<dyn AnyEvent>> {
    let event_type = CURRENT_EVENT_TYPE.with(|slot| slot.borrow().clone())?;
    Some(Box::new(DynamicEvent {
        event_type,
        data: value,
    }))
}

/// Attempt to deserialize a JSON string into a JS event object.
///
/// Looks up the JS callback registered via [`register_event_deserializer`]
/// for `name` and calls it with the parsed JSON. Returns the JS value
/// produced by the callback, or `None` if no callback is registered or the
/// JSON is invalid.
#[wasm_bindgen(js_name = "tryDeserializeEvent")]
#[must_use]
pub fn try_deserialize_event(name: String, json_str: String) -> Option<JsValue> {
    let value: serde_json::Value = serde_json::from_str(&json_str).ok()?;

    CURRENT_EVENT_TYPE.with(|slot| {
        *slot.borrow_mut() = Some(name.clone());
    });
    let _native = blazen_try_deserialize_event(&name, &value);
    CURRENT_EVENT_TYPE.with(|slot| {
        *slot.borrow_mut() = None;
    });

    let callback = JS_DESERIALIZERS.with(|map| map.borrow().get(&name).cloned())?;

    let serializer = serde_wasm_bindgen::Serializer::new().serialize_maps_as_objects(true);
    let js_value = value.serialize(&serializer).ok()?;

    let result = callback.call1(&JsValue::NULL, &js_value).ok()?;
    if result.is_undefined() || result.is_null() {
        None
    } else {
        Some(result)
    }
}

/// Intern a dynamic event type name and return an owned copy of the
/// canonical string.
///
/// The interned `&'static str` lives in the engine's global registry; the
/// JS side receives a fresh [`String`] containing the same bytes so it can
/// be passed back across the boundary safely.
#[wasm_bindgen(js_name = "internEventType")]
#[must_use]
pub fn intern_event_type(name: String) -> String {
    blazen_intern_event_type(&name).to_owned()
}
