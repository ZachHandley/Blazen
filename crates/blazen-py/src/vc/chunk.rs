//! `VcChunk` Python wrapper.
//!
//! Mirrors the per-chunk payload yielded by
//! [`VoiceConversionBackend::stream_convert`](blazen_audio_vc::VoiceConversionBackend::stream_convert):
//! a slice of f32 PCM samples plus an `is_final` flag, optional latency
//! measurement, and the producing backend's target-voice sample rate.

use pyo3::prelude::*;
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pymethods};

/// A single PCM audio chunk emitted by a [`PyVcModel`](super::PyVcModel).
///
/// Each chunk owns its own f32 sample buffer; consumers can concatenate
/// chunk samples in arrival order to reconstruct the full converted
/// utterance equivalent to a non-streaming `convert_voice` call.
#[gen_stub_pyclass]
#[pyclass(name = "VcChunk", frozen, from_py_object)]
#[derive(Clone)]
pub struct PyVcChunk {
    pub(crate) samples: Vec<f32>,
    pub(crate) is_final: bool,
    pub(crate) latency_seconds: Option<f32>,
    pub(crate) sample_rate: Option<u32>,
}

#[gen_stub_pymethods]
#[pymethods]
impl PyVcChunk {
    /// 32-bit float PCM samples in `[-1.0, 1.0]` at the target voice's
    /// sample rate (mono).
    #[getter]
    fn samples(&self) -> Vec<f32> {
        self.samples.clone()
    }

    /// `True` if this is the last chunk for the conversion call.
    ///
    /// Stream items emitted by [`PyVcStream`](super::PyVcStream) carry
    /// `is_final = false`; end-of-stream is signalled to Python callers
    /// via `StopAsyncIteration`. Final chunks produced by the
    /// non-streaming `convert_voice` path carry `is_final = true`.
    #[getter]
    fn is_final(&self) -> bool {
        self.is_final
    }

    /// Latency-from-call-start in seconds for this chunk, if the backend
    /// measured it. RVC backends today do not surface per-chunk latency
    /// metrics, so this is typically `None`.
    #[getter]
    fn latency_seconds(&self) -> Option<f32> {
        self.latency_seconds
    }

    /// Sample rate (Hz) of the f32 samples carried by this chunk.
    ///
    /// Always set on the final chunk produced by `convert_voice`; may be
    /// `None` on intermediate streamed chunks (consult the producing
    /// `VcModel` via the registered [`PyTargetVoice`](super::PyTargetVoice)
    /// for the rate in that case).
    #[getter]
    fn sample_rate(&self) -> Option<u32> {
        self.sample_rate
    }

    /// Number of f32 PCM samples carried by this chunk.
    #[getter]
    fn sample_count(&self) -> usize {
        self.samples.len()
    }

    fn __repr__(&self) -> String {
        format!(
            "VcChunk(samples={} f32, is_final={}, sample_rate={:?})",
            self.samples.len(),
            self.is_final,
            self.sample_rate,
        )
    }
}

impl PyVcChunk {
    /// Build a streamed intermediate chunk from a backend-yielded f32
    /// sample buffer.
    pub(crate) fn from_streamed_samples(samples: Vec<f32>) -> Self {
        Self {
            samples,
            is_final: false,
            latency_seconds: None,
            sample_rate: None,
        }
    }
}
