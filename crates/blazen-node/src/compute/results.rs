//! Compute result types.
//!
//! Re-exports core types. The `Js*` mirror structs are auto-generated
//! by `build.rs` and included via `crate::generated`.

pub use blazen_llm::compute::{
    AudioResult, ImageResult, ThreeDResult, TranscriptionResult, TranscriptionSegment, VideoResult,
};
