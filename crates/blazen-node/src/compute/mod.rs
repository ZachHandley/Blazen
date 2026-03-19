//! Compute request, result, and job types for the Node.js bindings.

pub mod job;
pub mod requests;
pub mod results;

// Re-export all public types for convenient access.
pub use job::{JsComputeRequest, JsComputeResult, JsComputeTiming, JsJobHandle, JsJobStatus};
pub use requests::{
    JsImageRequest, JsMusicRequest, JsSpeechRequest, JsThreeDRequest, JsTranscriptionRequest,
    JsUpscaleRequest, JsVideoRequest,
};
pub use results::{
    JsAudioResult, JsImageResult, JsThreeDResult, JsTranscriptionResult, JsTranscriptionSegment,
    JsVideoResult,
};
