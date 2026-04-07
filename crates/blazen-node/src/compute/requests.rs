//! Compute request types.
//!
//! Re-exports core types. The `Js*` mirror structs are auto-generated
//! by `build.rs` and included via `crate::generated`.

pub use blazen_llm::compute::{
    BackgroundRemovalRequest, ImageRequest, MusicRequest, SpeechRequest, ThreeDRequest,
    TranscriptionRequest, UpscaleRequest, VideoRequest,
};
