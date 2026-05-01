//! Plain-object napi mirrors for the wave 0 event types (`UsageEvent`,
//! `ProgressEvent`) and their associated enums (`Modality`, `ProgressKind`).
//!
//! These complement the typed class wrappers in [`super::events`] (which
//! cover `StartEvent` / `StopEvent` / `StreamChunkEvent` / etc.) by surfacing
//! the cost / token / progress event shapes that `Pipeline` and `Workflow`
//! emit during a run. They are intentionally `#[napi(object)]` shapes (not
//! classes) so step / stream callbacks can return them directly as plain
//! JS literals.

use napi::bindgen_prelude::*;
use napi_derive::napi;

use blazen_events::{Modality, ProgressEvent, ProgressKind, UsageEvent};

// ---------------------------------------------------------------------------
// Modality
// ---------------------------------------------------------------------------

/// Discriminant for the kind of provider call a [`JsUsageEvent`] describes.
///
/// Mirrors [`blazen_events::Modality`]. The string-enum representation
/// matches the `Modality::*` unit variants. Custom modalities (the Rust
/// `Modality::Custom(String)` variant) are surfaced via the
/// [`JsUsageEvent::modalityCustom`] string field — when `modalityCustom`
/// is non-null, callers should treat `modality` as `Custom` regardless of
/// its value.
#[napi(string_enum, js_name = "Modality")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum JsModality {
    Llm,
    Embedding,
    ImageGen,
    AudioTts,
    AudioStt,
    Video,
    ThreeD,
    BackgroundRemoval,
    Custom,
}

impl From<&Modality> for JsModality {
    fn from(m: &Modality) -> Self {
        match m {
            Modality::Llm => Self::Llm,
            Modality::Embedding => Self::Embedding,
            Modality::ImageGen => Self::ImageGen,
            Modality::AudioTts => Self::AudioTts,
            Modality::AudioStt => Self::AudioStt,
            Modality::Video => Self::Video,
            Modality::ThreeD => Self::ThreeD,
            Modality::BackgroundRemoval => Self::BackgroundRemoval,
            Modality::Custom(_) => Self::Custom,
        }
    }
}

impl JsModality {
    /// Build the canonical Rust [`Modality`] from a JS-side `(modality,
    /// custom)` pair. When `modality == Custom`, the `custom` string is
    /// required; otherwise it is ignored.
    #[must_use]
    pub fn into_modality(self, custom: Option<String>) -> Modality {
        match self {
            Self::Llm => Modality::Llm,
            Self::Embedding => Modality::Embedding,
            Self::ImageGen => Modality::ImageGen,
            Self::AudioTts => Modality::AudioTts,
            Self::AudioStt => Modality::AudioStt,
            Self::Video => Modality::Video,
            Self::ThreeD => Modality::ThreeD,
            Self::BackgroundRemoval => Modality::BackgroundRemoval,
            Self::Custom => Modality::Custom(custom.unwrap_or_default()),
        }
    }
}

// ---------------------------------------------------------------------------
// JsUsageEvent — plain-object mirror of `blazen_events::UsageEvent`
// ---------------------------------------------------------------------------

/// Token / cost / latency snapshot for a single provider call, emitted
/// after each LLM / embedding / image / audio / video / 3D request.
///
/// Pipelines and workflows aggregate these into [`PipelineState.usageTotal`]
/// and [`PipelineState.costTotalUsd`] when a `UsageEmitter` is wired up.
///
/// ```typescript
/// import { UsageEvent, Modality } from 'blazen';
///
/// const ev: UsageEvent = {
///     provider: "openai",
///     model: "gpt-4o-mini",
///     modality: Modality.Llm,
///     promptTokens: 100,
///     completionTokens: 25,
///     totalTokens: 125,
///     reasoningTokens: 0,
///     cachedInputTokens: 0,
///     audioInputTokens: 0,
///     audioOutputTokens: 0,
///     imageCount: 0,
///     audioSeconds: 0,
///     videoSeconds: 0,
///     latencyMs: 432,
///     costUsd: 0.000_25,
///     runId: "...",
/// };
/// ```
#[napi(object, js_name = "UsageEvent")]
pub struct JsUsageEvent {
    /// The provider that served the call (e.g. `"openai"`, `"anthropic"`).
    pub provider: String,
    /// The model identifier.
    pub model: String,
    /// Discriminant for the kind of call.
    pub modality: JsModality,
    /// Free-form custom-modality label. Populated when [`Self::modality`] is
    /// [`JsModality::Custom`]; ignored otherwise.
    #[napi(js_name = "modalityCustom")]
    pub modality_custom: Option<String>,
    /// Number of prompt / input tokens billed.
    #[napi(js_name = "promptTokens")]
    pub prompt_tokens: u32,
    /// Number of completion / output tokens billed.
    #[napi(js_name = "completionTokens")]
    pub completion_tokens: u32,
    /// Total tokens billed (typically `promptTokens + completionTokens`).
    #[napi(js_name = "totalTokens")]
    pub total_tokens: u32,
    /// Reasoning tokens (e.g. `OpenAI` o-series, Anthropic extended thinking).
    #[napi(js_name = "reasoningTokens")]
    pub reasoning_tokens: u32,
    /// Tokens served from the provider's prompt cache at a discount.
    #[napi(js_name = "cachedInputTokens")]
    pub cached_input_tokens: u32,
    /// Audio input tokens (multimodal speech-in models).
    #[napi(js_name = "audioInputTokens")]
    pub audio_input_tokens: u32,
    /// Audio output tokens (multimodal speech-out models).
    #[napi(js_name = "audioOutputTokens")]
    pub audio_output_tokens: u32,
    /// Number of images generated or processed.
    #[napi(js_name = "imageCount")]
    pub image_count: u32,
    /// Audio duration in seconds (for STT inputs and TTS outputs).
    #[napi(js_name = "audioSeconds")]
    pub audio_seconds: f64,
    /// Video duration in seconds.
    #[napi(js_name = "videoSeconds")]
    pub video_seconds: f64,
    /// Cost in USD as reported (or computed) for this call.
    #[napi(js_name = "costUsd")]
    pub cost_usd: Option<f64>,
    /// Wall-clock latency of the provider call in milliseconds.
    #[napi(js_name = "latencyMs")]
    pub latency_ms: f64,
    /// UUID of the run / pipeline invocation this usage belongs to.
    #[napi(js_name = "runId")]
    pub run_id: String,
}

#[allow(clippy::cast_precision_loss)]
impl From<UsageEvent> for JsUsageEvent {
    fn from(ev: UsageEvent) -> Self {
        let modality_custom = match &ev.modality {
            Modality::Custom(s) => Some(s.clone()),
            _ => None,
        };
        Self {
            provider: ev.provider,
            model: ev.model,
            modality: JsModality::from(&ev.modality),
            modality_custom,
            prompt_tokens: ev.prompt_tokens,
            completion_tokens: ev.completion_tokens,
            total_tokens: ev.total_tokens,
            reasoning_tokens: ev.reasoning_tokens,
            cached_input_tokens: ev.cached_input_tokens,
            audio_input_tokens: ev.audio_input_tokens,
            audio_output_tokens: ev.audio_output_tokens,
            image_count: ev.image_count,
            audio_seconds: ev.audio_seconds,
            video_seconds: ev.video_seconds,
            cost_usd: ev.cost_usd,
            latency_ms: ev.latency_ms as f64,
            run_id: ev.run_id.to_string(),
        }
    }
}

#[allow(clippy::cast_sign_loss, clippy::cast_possible_truncation)]
impl From<JsUsageEvent> for UsageEvent {
    fn from(ev: JsUsageEvent) -> Self {
        let run_id = uuid::Uuid::parse_str(&ev.run_id).unwrap_or_else(|_| uuid::Uuid::nil());
        Self {
            provider: ev.provider,
            model: ev.model,
            modality: ev.modality.into_modality(ev.modality_custom),
            prompt_tokens: ev.prompt_tokens,
            completion_tokens: ev.completion_tokens,
            total_tokens: ev.total_tokens,
            reasoning_tokens: ev.reasoning_tokens,
            cached_input_tokens: ev.cached_input_tokens,
            audio_input_tokens: ev.audio_input_tokens,
            audio_output_tokens: ev.audio_output_tokens,
            image_count: ev.image_count,
            audio_seconds: ev.audio_seconds,
            video_seconds: ev.video_seconds,
            cost_usd: ev.cost_usd,
            latency_ms: ev.latency_ms as u64,
            run_id,
        }
    }
}

// ---------------------------------------------------------------------------
// ProgressKind
// ---------------------------------------------------------------------------

/// What a [`JsProgressEvent`] describes. Mirrors
/// [`blazen_events::ProgressKind`].
#[napi(string_enum, js_name = "ProgressKind")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum JsProgressKind {
    Pipeline,
    Workflow,
    SubWorkflow,
    Stage,
}

impl From<&ProgressKind> for JsProgressKind {
    fn from(k: &ProgressKind) -> Self {
        match k {
            ProgressKind::Pipeline => Self::Pipeline,
            ProgressKind::Workflow => Self::Workflow,
            ProgressKind::SubWorkflow => Self::SubWorkflow,
            ProgressKind::Stage => Self::Stage,
        }
    }
}

impl From<JsProgressKind> for ProgressKind {
    fn from(k: JsProgressKind) -> Self {
        match k {
            JsProgressKind::Pipeline => Self::Pipeline,
            JsProgressKind::Workflow => Self::Workflow,
            JsProgressKind::SubWorkflow => Self::SubWorkflow,
            JsProgressKind::Stage => Self::Stage,
        }
    }
}

// ---------------------------------------------------------------------------
// JsProgressEvent
// ---------------------------------------------------------------------------

/// Per-stage / per-step progress tick emitted by Pipeline and Workflow
/// runners. Mirrors [`blazen_events::ProgressEvent`].
///
/// `total` and `percent` are absent (`undefined`) when the step set is
/// dynamic and the total is not known up front.
#[napi(object, js_name = "ProgressEvent")]
pub struct JsProgressEvent {
    /// What this progress event describes.
    pub kind: JsProgressKind,
    /// Current step / stage index (1-based).
    pub current: u32,
    /// Total number of steps / stages, when known.
    pub total: Option<u32>,
    /// Progress as a percentage in `0.0..=100.0`, when computable.
    pub percent: Option<f64>,
    /// Human-readable label for this progress tick (typically the step name).
    pub label: String,
    /// UUID of the run this progress belongs to.
    #[napi(js_name = "runId")]
    pub run_id: String,
}

#[allow(clippy::cast_lossless)]
impl From<ProgressEvent> for JsProgressEvent {
    fn from(ev: ProgressEvent) -> Self {
        Self {
            kind: JsProgressKind::from(&ev.kind),
            current: ev.current,
            total: ev.total,
            percent: ev.percent.map(f64::from),
            label: ev.label,
            run_id: ev.run_id.to_string(),
        }
    }
}

#[allow(clippy::cast_possible_truncation)]
impl From<JsProgressEvent> for ProgressEvent {
    fn from(ev: JsProgressEvent) -> Self {
        let run_id = uuid::Uuid::parse_str(&ev.run_id).unwrap_or_else(|_| uuid::Uuid::nil());
        Self {
            kind: ev.kind.into(),
            current: ev.current,
            total: ev.total,
            percent: ev.percent.map(|v| v as f32),
            label: ev.label,
            run_id,
        }
    }
}

// ---------------------------------------------------------------------------
// Free-function constructors (for ergonomic JS construction without
// remembering every camelCase field). Kept tiny — the canonical surface is
// the plain-object literal.
// ---------------------------------------------------------------------------

/// Build a default [`JsUsageEvent`] for the given provider / model, with
/// every numeric field zeroed and `modality = Llm`. Useful as a starting
/// point for emitter shims that only know a subset of the fields.
#[napi(js_name = "newUsageEvent")]
#[must_use]
pub fn new_usage_event(provider: String, model: String, run_id: String) -> JsUsageEvent {
    JsUsageEvent {
        provider,
        model,
        modality: JsModality::Llm,
        modality_custom: None,
        prompt_tokens: 0,
        completion_tokens: 0,
        total_tokens: 0,
        reasoning_tokens: 0,
        cached_input_tokens: 0,
        audio_input_tokens: 0,
        audio_output_tokens: 0,
        image_count: 0,
        audio_seconds: 0.0,
        video_seconds: 0.0,
        cost_usd: None,
        latency_ms: 0.0,
        run_id,
    }
}

/// Aggregate one [`JsUsageEvent`] into a [`crate::types::JsTokenUsageClass`].
/// Returns a fresh class instance that adds the seven token counters from the
/// event into the existing usage. Mirrors the Rust `TokenUsage::add` /
/// `PipelineState::record_usage` flow at the JS layer.
///
/// # Errors
///
/// Currently never returns an error; the [`Result`] return is reserved for
/// future validation (e.g. parsing the `runId` UUID).
#[napi(js_name = "addUsageToTokenUsage")]
#[allow(clippy::needless_pass_by_value)]
pub fn add_usage_to_token_usage(
    base: &crate::types::JsTokenUsageClass,
    event: JsUsageEvent,
) -> Result<crate::types::JsTokenUsageClass> {
    let mut tu = base.inner.clone();
    let delta = blazen_llm::types::TokenUsage {
        prompt_tokens: event.prompt_tokens,
        completion_tokens: event.completion_tokens,
        total_tokens: event.total_tokens,
        reasoning_tokens: event.reasoning_tokens,
        cached_input_tokens: event.cached_input_tokens,
        audio_input_tokens: event.audio_input_tokens,
        audio_output_tokens: event.audio_output_tokens,
    };
    tu.add(&delta);
    Ok(crate::types::JsTokenUsageClass::from(tu))
}
