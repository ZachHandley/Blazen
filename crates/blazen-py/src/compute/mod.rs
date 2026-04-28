//! Python wrappers for compute request types, job types, and related types.

pub mod job;
pub mod request_types;
pub mod requests;
pub mod result_types;

pub use job::{PyCompute, PyComputeRequest, PyJobHandle, PyJobStatus};
pub use requests::{
    BackgroundRemovalRequest, ImageRequest, MusicRequest, SpeechRequest, ThreeDRequest,
    TranscriptionRequest, UpscaleRequest, VideoRequest,
};
pub use result_types::{
    PyAudioResult, PyComputeResult, PyImageResult, PyThreeDResult, PyTranscriptionResult,
    PyTranscriptionSegment, PyVideoResult, PyVoiceHandle,
};
