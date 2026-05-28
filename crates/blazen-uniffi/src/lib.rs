//! `blazen-uniffi` — multi-language bindings for Blazen via Mozilla UniFFI.
//!
//! This crate is the source of truth for Blazen's Go, Swift, Kotlin, and Ruby
//! bindings. A single set of `#[uniffi::export]` annotations on the public
//! surface here drives four foreign-language bindgens:
//!
//! - Go     via NordSecurity/uniffi-bindgen-go      → `bindings/go/`
//! - Swift  via mozilla/uniffi-rs's swift bindgen   → `bindings/swift/`
//! - Kotlin via mozilla/uniffi-rs's kotlin bindgen  → `bindings/kotlin/`
//! - Ruby   via mozilla/uniffi-rs's ruby bindgen    → `bindings/ruby/`
//!
//! Python is intentionally NOT covered here — Blazen ships `blazen-py` via
//! PyO3 which is more mature and idiomatic than UniFFI's Python output.
//!
//! Async Rust functions are exposed as language-native async (Swift
//! `async`/`await`, Kotlin `suspend fun`) or as blocking calls that compose
//! naturally with the host runtime (Go goroutines, Ruby fibers). Tokio runs
//! invisibly underneath via `uniffi`'s `tokio` feature.

#![allow(unsafe_code)] // UniFFI scaffolding contains generated `extern "C"` thunks.
#![allow(
    clippy::all, // UniFFI-generated scaffolding doesn't pass workspace lints.
    clippy::pedantic,
    dead_code
)]

uniffi::include_scaffolding!("blazen");

pub mod agent;
pub mod batch;
pub mod compute;
pub mod compute_music;
pub mod compute_types;
pub mod compute_vc;
// P4.2: per-engine `#[uniffi::Object]` providers wrapping the
// canonical Rust concretes in `blazen_llm::providers::concrete::*`.
pub mod concrete;
#[cfg(feature = "distributed")]
pub mod controlplane;
pub mod errors;
pub mod llm;
pub(crate) mod llm_adapter;
pub mod local_model;
pub mod manager;
#[cfg(feature = "distributed")]
pub mod model_client;
#[cfg(feature = "distributed")]
pub mod peer;
pub mod persist;
pub mod pipeline;
pub mod pricing;
pub mod provider_api_protocol;
pub mod provider_base;
pub mod provider_custom;
pub mod provider_defaults;
pub mod providers;
pub mod runtime;
pub mod streaming;
pub mod telemetry;
pub mod workflow;

pub use agent::{Agent, AgentResult, ToolHandler};
pub use batch::{BatchItem, BatchResult, complete_batch, complete_batch_blocking};
#[cfg(feature = "triposr")]
pub use compute::ThreeDGenerateResult;
pub use compute::{ImageGenResult, SttResult, TtsResult};
pub use compute_music::{MusicChunk, MusicResult, MusicStreamSink};
pub use compute_types::{
    AudioResult, BackgroundRemovalRequest, Generated3DModel, GeneratedAudio, GeneratedImage,
    GeneratedVideo, ImageRequest, ImageResult, MediaOutput, MusicRequest, RequestTiming,
    SpeechRequest, ThreeDRequest, ThreeDResult, TranscriptionRequest, TranscriptionResult,
    TranscriptionSegment, UpscaleRequest, VideoRequest, VideoResult, VoiceCloneRequest,
    VoiceHandle,
};
pub use compute_vc::{TargetVoice, VcChunk, VcResult, VcStreamSink};
#[cfg(feature = "distributed")]
pub use controlplane::{
    ControlPlaneAdmission, ControlPlaneAdmissionMode, ControlPlaneAssignmentHandler,
    ControlPlaneClient, ControlPlaneRunEvent, ControlPlaneRunEventSubscriber,
    ControlPlaneRunStateSnapshot, ControlPlaneRunStatus, ControlPlaneSubmitRequest,
    ControlPlaneSubscription, ControlPlaneWorker, ControlPlaneWorkerCapability,
    ControlPlaneWorkerInfo,
};
pub use errors::{BlazenError, BlazenResult};
pub use llm::{
    ChatMessage, EmbeddingResponse, Media, ModelRequest, ModelResponse, TokenUsage, Tool, ToolCall,
};
pub use local_model::{
    AdapterHandleRecord, AdapterOptionsRecord, AdapterStatusRecord, ForeignLocalModel,
};
pub use manager::{ModelStatusRecord, PoolStatusRecord, UniffiModelManager};
#[cfg(feature = "distributed")]
pub use model_client::{ModelClient, ModelClientStatusRecord, ModelPool, StatusRecord};
#[cfg(feature = "distributed")]
pub use peer::{PeerClient, PeerServer};
pub use persist::{
    CheckpointStore, PersistedEvent, WorkflowCheckpoint, new_redb_checkpoint_store,
    new_valkey_checkpoint_store,
};
pub use pipeline::{Pipeline, PipelineBuilder};
pub use provider_api_protocol::{ApiProtocol, AuthMethod, KeyValue, OpenAiCompatConfig};
pub use provider_base::LlmProviderDefaults;
pub use provider_custom::{
    CustomProvider, CustomProviderHandle, custom_provider_from_foreign, lm_studio,
    new_openai_compat_config, ollama, openai_compat,
};
pub use provider_defaults::{
    AudioMusicProviderDefaults, AudioSpeechProviderDefaults, BackgroundRemovalProviderDefaults,
    BaseProviderDefaults, EmbeddingProviderDefaults, ImageGenerationProviderDefaults,
    ImageUpscaleProviderDefaults, ProviderDefaults, ThreeDProviderDefaults,
    TranscriptionProviderDefaults, VideoProviderDefaults, VoiceCloningProviderDefaults,
};
pub use runtime::init;
pub use streaming::{CompletionStreamSink, StreamChunk};
#[cfg(feature = "langfuse")]
pub use telemetry::init_langfuse;
#[cfg(feature = "otlp")]
pub use telemetry::init_otlp;
#[cfg(feature = "prometheus")]
pub use telemetry::init_prometheus;
pub use telemetry::{WorkflowHistoryEntry, parse_workflow_history, shutdown_telemetry};
// 3D DTOs + per-engine concretes are re-exported from the canonical
// `concrete::three_d` home (the legacy `threed` module was retired in
// P4.2.x.3.three_d in favor of the unified per-engine layout).
#[cfg(feature = "threed-compat-proxy")]
pub use concrete::three_d::Compat3dProvider;
#[cfg(feature = "threed")]
pub use concrete::three_d::{
    AnimateRequest, AnimateResult, PbrMaps, RefineRequest, RefineResult, RefineStats, RigRequest,
    RigResult, TexturizeRequest, TexturizeResult,
};
pub use workflow::{Event, StepHandler, StepOutput, Workflow, WorkflowBuilder, WorkflowResult};

/// Returns the `blazen-uniffi` crate version baked in at compile time.
///
/// Exposed in the UDL as `string version()` so every foreign binding has a
/// stable way to query the underlying native lib version (useful for
/// diagnosing version-skew issues between a Go/Swift/Kotlin/Ruby module
/// and its embedded native lib).
#[must_use]
pub fn version() -> String {
    env!("CARGO_PKG_VERSION").to_string()
}
