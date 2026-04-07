//! Compute request, result, and job types for the Node.js bindings.

pub mod job;
pub mod requests;
pub mod results;

// Re-export generated Js* mirror types (produced by build.rs from blazen-llm).
pub use crate::generated::{
    JsAudioResult, JsBackgroundRemovalRequest, JsComputeRequest, JsComputeResult, JsImageRequest,
    JsImageResult, JsJobHandle, JsMusicRequest, JsRequestTiming, JsSpeechRequest, JsThreeDRequest,
    JsThreeDResult, JsTranscriptionRequest, JsTranscriptionResult, JsTranscriptionSegment,
    JsUpscaleRequest, JsVideoRequest, JsVideoResult,
};
// JsJobStatus is hand-written (tagged enum with data variant, not auto-generated).
pub use job::JsJobStatus;
