//! Compute request types.
//!
//! These are re-exported from `blazen_llm::compute` and used via
//! `pythonize::depythonize` at the FFI boundary. Python users pass
//! plain dicts that match the serde schema.

// Re-export core types so other modules can reference them by short name.
pub use blazen_llm::compute::{
    BackgroundRemovalRequest, ImageRequest, MusicRequest, SpeechRequest, ThreeDRequest,
    TranscriptionRequest, UpscaleRequest, VideoRequest,
};
