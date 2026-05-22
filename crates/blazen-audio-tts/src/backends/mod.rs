//! Concrete [`TtsBackend`](crate::TtsBackend) implementations.
//!
//! Each backend is gated behind its own feature flag so consumers only
//! pay for what they use:
//!
//! | Module       | Feature   | Notes                                       |
//! |--------------|-----------|---------------------------------------------|
//! | [`anytts`]   | `anytts`  | Local engines (Kokoro / VibeVoice / Qwen3). |
//! | [`bark`]     | `bark`    | Suno-AI Bark (3-stage AR + EnCodec).        |
//! | [`f5`]       | `f5-tts`  | SWivid F5-TTS (flow-matching DiT + Vocos).  |
//! | [`openai`]   | `openai`  | HTTP client for OpenAI-compat servers.      |
//! | [`piper`]    | always on | Reserved stub — see module docs.            |

#[cfg(feature = "anytts")]
pub mod anytts;

#[cfg(feature = "bark")]
pub mod bark;

#[cfg(feature = "f5-tts")]
pub mod f5;

#[cfg(feature = "spark-tts")]
pub mod spark;

#[cfg(feature = "maskgct")]
pub mod maskgct;

#[cfg(feature = "openai")]
pub mod openai;

#[cfg(feature = "openai")]
pub mod openai_types;

pub mod piper;

#[cfg(feature = "anytts")]
pub use anytts::AnyTtsBackend;

#[cfg(feature = "bark")]
pub use bark::{BARK_BACKEND_ID_PREFIX, BarkBackend, BarkConfig};

#[cfg(feature = "f5-tts")]
pub use f5::{F5_BACKEND_ID_PREFIX, F5Backend, F5Config};

#[cfg(feature = "openai")]
pub use openai::{
    DEFAULT_MODEL, DEFAULT_RESPONSE_FORMAT, DEFAULT_VOICE, OpenAiTtsBackend, OpenAiTtsConfig,
    OpenAiTtsSpeechRequest, OpenAiTtsSpeechResponse,
};

#[cfg(feature = "openai")]
pub use openai_types::{
    CloneVoiceResponse, ListVoicesResponse as OpenAiListVoicesResponse, VoiceDto as OpenAiVoiceDto,
};

pub use piper::PiperBackend;
