//! Python wrappers for compute request types, job types, and related types.

pub mod job;
pub mod requests;

pub use job::{PyComputeRequest, PyJobHandle, PyJobStatus};
pub use requests::{
    PyImageRequest, PyMusicRequest, PySpeechRequest, PyThreeDRequest, PyTranscriptionRequest,
    PyUpscaleRequest, PyVideoRequest,
};
