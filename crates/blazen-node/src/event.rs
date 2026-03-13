//! Event conversion between JavaScript objects and the Rust event system.
//!
//! JavaScript events are plain objects with a `type` field:
//!
//! ```javascript
//! const event = { type: "AnalyzeEvent", text: "hello", score: 0.9 };
//! ```
//!
//! This module converts them to/from [`DynamicEvent`] for the Rust workflow
//! engine, and handles the special `StartEvent`/`StopEvent` types.

use blazen_events::{AnyEvent, DynamicEvent};

/// Convert a JavaScript event (as `serde_json::Value`) to a boxed [`AnyEvent`].
///
/// The JS object is expected to have a `"type"` field containing the event
/// type string. All other fields become the event data.
///
/// Special handling:
/// - `"blazen::StartEvent"` -> [`blazen_events::StartEvent`]
/// - `"blazen::StopEvent"` -> [`blazen_events::StopEvent`]
/// - Everything else -> [`DynamicEvent`]
pub fn js_value_to_any_event(value: &serde_json::Value) -> Box<dyn AnyEvent> {
    let event_type = value
        .get("type")
        .and_then(serde_json::Value::as_str)
        .unwrap_or("unknown")
        .to_owned();

    // Extract the data: everything except the "type" field.
    let data = extract_data(value);

    // Check for built-in event types.
    if event_type == "blazen::StartEvent" {
        return Box::new(blazen_events::StartEvent { data });
    }

    if event_type == "blazen::StopEvent" {
        let result = if let Some(r) = data.get("result") {
            r.clone()
        } else {
            data
        };
        return Box::new(blazen_events::StopEvent { result });
    }

    // Generic dynamic event.
    Box::new(DynamicEvent { event_type, data })
}

/// Convert a boxed [`AnyEvent`] to a JavaScript-friendly `serde_json::Value`.
///
/// The returned JSON object always has a `"type"` field and additional data
/// fields.
pub fn any_event_to_js_value(event: &dyn AnyEvent) -> serde_json::Value {
    let event_type = event.event_type_id().to_owned();
    let json = event.to_json();

    // StartEvent: { type: "blazen::StartEvent", ...data }
    if event_type == "blazen::StartEvent" {
        let data = json.get("data").cloned().unwrap_or(serde_json::Value::Null);
        return merge_type_into_data(&event_type, &data);
    }

    // StopEvent: { type: "blazen::StopEvent", result: ... }
    if event_type == "blazen::StopEvent" {
        let result = json
            .get("result")
            .cloned()
            .unwrap_or(serde_json::Value::Null);
        let mut obj = serde_json::Map::new();
        obj.insert("type".to_owned(), serde_json::Value::String(event_type));
        obj.insert("result".to_owned(), result);
        return serde_json::Value::Object(obj);
    }

    // DynamicEvent: { type: "...", ...data }
    if let Some(dynamic) = event.as_any().downcast_ref::<DynamicEvent>() {
        return merge_type_into_data(&dynamic.event_type, &dynamic.data);
    }

    // Fallback: use the full JSON as data.
    merge_type_into_data(&event_type, &json)
}

/// Extract data fields from a JS value, stripping the `"type"` field.
fn extract_data(value: &serde_json::Value) -> serde_json::Value {
    match value {
        serde_json::Value::Object(map) => {
            let mut data = serde_json::Map::new();
            for (k, v) in map {
                if k != "type" {
                    data.insert(k.clone(), v.clone());
                }
            }
            serde_json::Value::Object(data)
        }
        other => other.clone(),
    }
}

/// Merge a `"type"` field into a data value, returning a flat object.
fn merge_type_into_data(event_type: &str, data: &serde_json::Value) -> serde_json::Value {
    let mut obj = if let serde_json::Value::Object(map) = data {
        map.clone()
    } else {
        let mut m = serde_json::Map::new();
        m.insert("data".to_owned(), data.clone());
        m
    };
    obj.insert(
        "type".to_owned(),
        serde_json::Value::String(event_type.to_owned()),
    );
    serde_json::Value::Object(obj)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn start_event_roundtrip() {
        let js = serde_json::json!({
            "type": "blazen::StartEvent",
            "message": "hello"
        });
        let event = js_value_to_any_event(&js);
        assert_eq!(event.event_type_id(), "blazen::StartEvent");

        let back = any_event_to_js_value(&*event);
        assert_eq!(back["type"], "blazen::StartEvent");
        assert_eq!(back["message"], "hello");
    }

    #[test]
    fn stop_event_roundtrip() {
        let js = serde_json::json!({
            "type": "blazen::StopEvent",
            "result": {"answer": 42}
        });
        let event = js_value_to_any_event(&js);
        assert_eq!(event.event_type_id(), "blazen::StopEvent");

        let back = any_event_to_js_value(&*event);
        assert_eq!(back["type"], "blazen::StopEvent");
        assert_eq!(back["result"]["answer"], 42);
    }

    #[test]
    fn dynamic_event_roundtrip() {
        let js = serde_json::json!({
            "type": "AnalyzeEvent",
            "text": "hello",
            "score": 0.9
        });
        let event = js_value_to_any_event(&js);
        assert_eq!(event.event_type_id(), "AnalyzeEvent");

        let back = any_event_to_js_value(&*event);
        assert_eq!(back["type"], "AnalyzeEvent");
        assert_eq!(back["text"], "hello");
        assert_eq!(back["score"], 0.9);
    }
}
