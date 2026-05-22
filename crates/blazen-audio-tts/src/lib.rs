//! Multi-backend text-to-speech surface for Blazen.
//!
//! This crate provides one capability trait — [`TtsBackend`] — that
//! extends [`blazen_audio::AudioBackend`] with synthesis and voice-
//! management methods, plus a small set of concrete backends under
//! [`backends`]:
//!
//! - [`backends::anytts::AnyTtsBackend`] — fully local engine via
//!   [`any-tts`](https://crates.io/crates/any-tts) (Kokoro-82M,
//!   `VibeVoice`, Qwen3-TTS). Gated by the `anytts` feature flag.
//! - [`backends::openai::OpenAiTtsBackend`] — HTTP client for
//!   OpenAI-compatible TTS servers (`/v1/audio/speech` + the de-facto
//!   `/v1/voices/*` extension surface). Gated by the `openai` feature
//!   flag (enabled by default).
//! - [`backends::piper::PiperBackend`] — reserved slot for a future
//!   Piper-ONNX backend; every method currently returns
//!   [`TtsError::Unsupported`] pending Wave 22 of the PR-AUDIO
//!   restructure.
//!
//! Two provider wrappers compose backends with the same surface:
//!
//! - [`TtsProvider<B>`] — monomorphized over a concrete backend, used
//!   when the backend choice is fixed at compile time.
//! - [`DynTtsProvider`] — type-erased `Arc<dyn TtsBackend>`, used by
//!   the manager / pipeline layer.
//!
//! # Feature flags
//!
//! | Feature   | Default | Description                                       |
//! |-----------|---------|---------------------------------------------------|
//! | `openai`  | yes     | Builds [`backends::openai::OpenAiTtsBackend`].    |
//! | `anytts`  | no      | Builds [`backends::anytts::AnyTtsBackend`].       |
//! | `bark`    | no      | Builds [`backends::bark::BarkBackend`] (Suno-AI   |
//! |           |         | Bark: 3-stage AR transformer + EnCodec).          |
//! | `engine`  | no      | **Deprecated** alias for `anytts`; will be       |
//! |           |         | removed one release after the multi-backend      |
//! |           |         | restructure ships.                               |

#![deny(missing_docs)]

pub mod backends;
mod error;
mod options;
mod provider;
mod traits;

pub use error::TtsError;
pub use options::{TtsModel, TtsOptions};
pub use provider::{DynTtsProvider, TtsProvider};
pub use traits::TtsBackend;

// Re-exports of common backend types so callers don't have to dig
// through `backends::*` for the most-used names.

#[cfg(feature = "anytts")]
pub use backends::AnyTtsBackend;

#[cfg(feature = "bark")]
pub use backends::{BARK_BACKEND_ID_PREFIX, BarkBackend, BarkConfig};

#[cfg(feature = "openai")]
pub use backends::{
    DEFAULT_MODEL, DEFAULT_RESPONSE_FORMAT, DEFAULT_VOICE, OpenAiTtsBackend, OpenAiTtsConfig,
    OpenAiTtsSpeechRequest, OpenAiTtsSpeechResponse,
};

pub use backends::PiperBackend;
