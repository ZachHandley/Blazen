//! `UsageEvent` — emitted after each provider call (LLM / embed / image / audio / etc.)
//! to surface tokens, modality-specific quantities, and cost from the
//! provider's actual API response. Used by Pipeline / Workflow rollups.

use std::any::Any;

use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::{AnyEvent, Event, register_event_deserializer};

/// The kind of provider call that produced a [`UsageEvent`].
#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
#[cfg_attr(feature = "tsify", derive(tsify_next::Tsify))]
#[cfg_attr(feature = "tsify", tsify(into_wasm_abi, from_wasm_abi))]
pub enum Modality {
    /// A text completion / chat / structured-output LLM call.
    #[default]
    Llm,
    /// An embedding model call.
    Embedding,
    /// An image generation call.
    ImageGen,
    /// A text-to-speech audio generation call.
    AudioTts,
    /// A speech-to-text transcription call.
    AudioStt,
    /// A video generation call.
    Video,
    /// A 3D asset generation call.
    ThreeD,
    /// A background-removal call.
    BackgroundRemoval,
    /// A user-defined modality.
    Custom(String),
}

/// Emitted after each provider call to surface tokens, modality-specific
/// quantities, and cost from the provider's actual API response.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "tsify", derive(tsify_next::Tsify))]
#[cfg_attr(feature = "tsify", tsify(into_wasm_abi, from_wasm_abi))]
pub struct UsageEvent {
    /// The provider that served the call (e.g. `"openai"`, `"anthropic"`).
    pub provider: String,
    /// The model identifier (e.g. `"gpt-4o-mini"`).
    pub model: String,
    /// The kind of call this usage record describes.
    pub modality: Modality,
    /// Number of prompt / input tokens billed.
    #[serde(default)]
    pub prompt_tokens: u32,
    /// Number of completion / output tokens billed.
    #[serde(default)]
    pub completion_tokens: u32,
    /// Total tokens billed (typically `prompt_tokens + completion_tokens`).
    #[serde(default)]
    pub total_tokens: u32,
    /// Reasoning tokens (e.g. `OpenAI` o-series, Anthropic extended thinking).
    #[serde(default)]
    pub reasoning_tokens: u32,
    /// Tokens served from the provider's prompt cache at a discount.
    #[serde(default)]
    pub cached_input_tokens: u32,
    /// Audio input tokens (multimodal speech-in models).
    #[serde(default)]
    pub audio_input_tokens: u32,
    /// Audio output tokens (multimodal speech-out models).
    #[serde(default)]
    pub audio_output_tokens: u32,
    /// Number of images generated or processed.
    #[serde(default)]
    pub image_count: u32,
    /// Audio duration in seconds (for STT inputs and TTS outputs).
    #[serde(default)]
    pub audio_seconds: f64,
    /// Video duration in seconds (for video generation outputs).
    #[serde(default)]
    pub video_seconds: f64,
    /// Cost in USD as reported (or computed) for this call.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cost_usd: Option<f64>,
    /// Wall-clock latency of the provider call in milliseconds.
    #[serde(default)]
    pub latency_ms: u64,
    /// Identifier of the run / pipeline invocation this usage belongs to.
    pub run_id: Uuid,
}

impl Event for UsageEvent {
    fn event_type() -> &'static str {
        static REGISTER: std::sync::Once = std::sync::Once::new();
        REGISTER.call_once(|| {
            register_event_deserializer("blazen::UsageEvent", |value| {
                serde_json::from_value::<UsageEvent>(value)
                    .ok()
                    .map(|e| Box::new(e) as _)
            });
        });
        "blazen::UsageEvent"
    }

    fn event_type_id(&self) -> &'static str {
        "blazen::UsageEvent"
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn clone_boxed(&self) -> Box<dyn AnyEvent> {
        Box::new(self.clone())
    }

    fn to_json(&self) -> serde_json::Value {
        serde_json::to_value(self).expect("UsageEvent serialization should never fail")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_event() -> UsageEvent {
        UsageEvent {
            provider: "openai".to_string(),
            model: "gpt-4o-mini".to_string(),
            modality: Modality::Llm,
            prompt_tokens: 100,
            completion_tokens: 50,
            total_tokens: 150,
            reasoning_tokens: 0,
            cached_input_tokens: 0,
            audio_input_tokens: 0,
            audio_output_tokens: 0,
            image_count: 0,
            audio_seconds: 0.0,
            video_seconds: 0.0,
            cost_usd: Some(0.000_25),
            latency_ms: 432,
            run_id: Uuid::new_v4(),
        }
    }

    #[test]
    fn usage_event_type_id() {
        assert_eq!(UsageEvent::event_type(), "blazen::UsageEvent");
        let evt = sample_event();
        assert_eq!(Event::event_type_id(&evt), "blazen::UsageEvent");
    }

    #[test]
    fn usage_event_roundtrip() {
        let evt = sample_event();
        let json = Event::to_json(&evt);
        let deserialized: UsageEvent = serde_json::from_value(json).unwrap();
        assert_eq!(evt.provider, deserialized.provider);
        assert_eq!(evt.model, deserialized.model);
        assert_eq!(evt.modality, deserialized.modality);
        assert_eq!(evt.prompt_tokens, deserialized.prompt_tokens);
        assert_eq!(evt.completion_tokens, deserialized.completion_tokens);
        assert_eq!(evt.total_tokens, deserialized.total_tokens);
        assert_eq!(evt.reasoning_tokens, deserialized.reasoning_tokens);
        assert_eq!(evt.cached_input_tokens, deserialized.cached_input_tokens);
        assert_eq!(evt.audio_input_tokens, deserialized.audio_input_tokens);
        assert_eq!(evt.audio_output_tokens, deserialized.audio_output_tokens);
        assert_eq!(evt.image_count, deserialized.image_count);
        assert!((evt.audio_seconds - deserialized.audio_seconds).abs() < f64::EPSILON);
        assert!((evt.video_seconds - deserialized.video_seconds).abs() < f64::EPSILON);
        assert_eq!(evt.cost_usd, deserialized.cost_usd);
        assert_eq!(evt.latency_ms, deserialized.latency_ms);
        assert_eq!(evt.run_id, deserialized.run_id);
    }

    #[test]
    fn usage_event_downcast() {
        let evt = sample_event();
        let boxed: Box<dyn AnyEvent> = Box::new(evt.clone());
        let downcasted = boxed.downcast_ref::<UsageEvent>().unwrap();
        assert_eq!(downcasted.provider, evt.provider);
        assert_eq!(downcasted.run_id, evt.run_id);
    }

    #[test]
    fn modality_serializes() {
        let json = serde_json::to_value(Modality::ImageGen).unwrap();
        assert_eq!(json, serde_json::json!("ImageGen"));

        let json = serde_json::to_value(Modality::Custom("foo".into())).unwrap();
        assert_eq!(json, serde_json::json!({"Custom": "foo"}));
    }
}
