//! `ProgressEvent` — emitted by Pipeline (per-stage with known total) and
//! Workflow (per-step, total may be `None` when steps are dynamic). Used by
//! callers and bindings to render progress UI.

use std::any::Any;

use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::{AnyEvent, Event, register_event_deserializer};

/// The kind of progress this event describes.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[cfg_attr(feature = "tsify", derive(tsify_next::Tsify))]
#[cfg_attr(feature = "tsify", tsify(into_wasm_abi, from_wasm_abi))]
pub enum ProgressKind {
    /// Top-level pipeline progress.
    Pipeline,
    /// Top-level workflow progress.
    Workflow,
    /// A nested workflow's progress, surfaced under its parent.
    SubWorkflow,
    /// Progress within a single pipeline stage.
    Stage,
}

/// Emitted by Pipeline and Workflow runners to surface progress to callers
/// and bindings for rendering progress UI.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "tsify", derive(tsify_next::Tsify))]
#[cfg_attr(feature = "tsify", tsify(into_wasm_abi, from_wasm_abi))]
pub struct ProgressEvent {
    /// What this progress event describes.
    pub kind: ProgressKind,
    /// Current step / stage index (1-based).
    pub current: u32,
    /// Total number of steps / stages, when known.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub total: Option<u32>,
    /// Progress as a percentage in `0.0..=100.0`, when computable.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub percent: Option<f32>,
    /// Human-readable label for this progress tick (typically the step name).
    pub label: String,
    /// Identifier of the run / pipeline invocation this progress belongs to.
    pub run_id: Uuid,
}

impl Event for ProgressEvent {
    fn event_type() -> &'static str {
        static REGISTER: std::sync::Once = std::sync::Once::new();
        REGISTER.call_once(|| {
            register_event_deserializer("blazen::ProgressEvent", |value| {
                serde_json::from_value::<ProgressEvent>(value)
                    .ok()
                    .map(|e| Box::new(e) as _)
            });
        });
        "blazen::ProgressEvent"
    }

    fn event_type_id(&self) -> &'static str {
        "blazen::ProgressEvent"
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn clone_boxed(&self) -> Box<dyn AnyEvent> {
        Box::new(self.clone())
    }

    fn to_json(&self) -> serde_json::Value {
        serde_json::to_value(self).expect("ProgressEvent serialization should never fail")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_event() -> ProgressEvent {
        ProgressEvent {
            kind: ProgressKind::Pipeline,
            current: 2,
            total: Some(5),
            percent: Some(40.0),
            label: "stage_two".to_string(),
            run_id: Uuid::new_v4(),
        }
    }

    #[test]
    fn progress_event_type_id() {
        assert_eq!(ProgressEvent::event_type(), "blazen::ProgressEvent");
        let evt = sample_event();
        assert_eq!(Event::event_type_id(&evt), "blazen::ProgressEvent");
    }

    #[test]
    fn progress_event_roundtrip() {
        let evt = sample_event();
        let json = Event::to_json(&evt);
        let deserialized: ProgressEvent = serde_json::from_value(json).unwrap();
        assert_eq!(evt.kind, deserialized.kind);
        assert_eq!(evt.current, deserialized.current);
        assert_eq!(evt.total, deserialized.total);
        match (evt.percent, deserialized.percent) {
            (Some(a), Some(b)) => assert!((a - b).abs() < f32::EPSILON),
            (None, None) => {}
            _ => panic!("percent did not roundtrip"),
        }
        assert_eq!(evt.label, deserialized.label);
        assert_eq!(evt.run_id, deserialized.run_id);
    }

    #[test]
    fn progress_event_downcast() {
        let evt = sample_event();
        let boxed: Box<dyn AnyEvent> = Box::new(evt.clone());
        let downcasted = boxed.downcast_ref::<ProgressEvent>().unwrap();
        assert_eq!(downcasted.label, evt.label);
        assert_eq!(downcasted.run_id, evt.run_id);
    }

    #[test]
    fn progress_event_omits_none_total_in_json() {
        let evt = ProgressEvent {
            kind: ProgressKind::Workflow,
            current: 3,
            total: None,
            percent: None,
            label: "dynamic_step".to_string(),
            run_id: Uuid::new_v4(),
        };
        let json = Event::to_json(&evt);
        let obj = json
            .as_object()
            .expect("ProgressEvent should serialize as object");
        assert!(!obj.contains_key("total"));
        assert!(!obj.contains_key("percent"));
        assert_eq!(obj.get("current"), Some(&serde_json::json!(3)));
        assert_eq!(obj.get("label"), Some(&serde_json::json!("dynamic_step")));
    }
}
