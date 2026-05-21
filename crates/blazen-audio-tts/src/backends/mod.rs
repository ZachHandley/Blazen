//! Concrete [`TtsBackend`](crate::TtsBackend) implementations.
//!
//! Each backend is gated behind its own feature flag so consumers only
//! pay for what they use:
//!
//! | Module       | Feature   | Notes                                       |
//! |--------------|-----------|---------------------------------------------|
//! | [`anytts`]   | `anytts`  | Local engines (Kokoro / VibeVoice / Qwen3). |
//! | [`openai`]   | `openai`  | HTTP client for OpenAI-compat servers.      |
//! | [`piper`]    | always on | Reserved stub — see module docs.            |

#[cfg(feature = "anytts")]
pub mod anytts;

#[cfg(feature = "openai")]
pub mod openai;

#[cfg(feature = "openai")]
pub mod openai_types;

pub mod piper;

#[cfg(feature = "anytts")]
pub use anytts::AnyTtsBackend;

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
