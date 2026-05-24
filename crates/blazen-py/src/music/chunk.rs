//! `MusicChunk` Python wrapper.
//!
//! Mirrors [`blazen_audio_music::MusicChunk`]: a slice of f32 PCM samples
//! plus an `is_final` flag and an optional measured chunk latency. The
//! Python class additionally carries the originating backend's sample
//! rate (set on non-streaming `generate_*` results that have a complete
//! [`GeneratedAudio`](blazen_audio::GeneratedAudio) at hand; `None` for
//! intermediate streamed chunks where the rate is implied by the backend).

use blazen_audio_music::MusicChunk;
use pyo3::prelude::*;
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pymethods};

/// A single PCM audio chunk emitted by a [`PyMusicModel`](super::PyMusicModel).
///
/// Each chunk owns its own f32 sample buffer; consumers can concatenate
/// chunk samples in arrival order to reconstruct the full clip equivalent
/// to a non-streaming `generate_music` / `generate_sfx` call.
#[gen_stub_pyclass]
#[pyclass(name = "MusicChunk", frozen, from_py_object)]
#[derive(Clone)]
pub struct PyMusicChunk {
    pub(crate) inner: MusicChunk,
    /// Backend sample rate (Hz). Always set on results produced by the
    /// non-streaming `generate_*` path; `None` on streamed intermediate
    /// chunks (the producing `MusicModel` exposes the rate via its
    /// `sample_rate` accessor).
    pub(crate) sample_rate: Option<u32>,
}

#[gen_stub_pymethods]
#[pymethods]
impl PyMusicChunk {
    /// 32-bit float PCM samples in `[-1.0, 1.0]` at the backend's
    /// `sample_rate` (mono).
    #[getter]
    fn samples(&self) -> Vec<f32> {
        self.inner.samples.clone()
    }

    /// `True` if this is the last chunk for the generation call.
    #[getter]
    fn is_final(&self) -> bool {
        self.inner.is_final
    }

    /// Latency-from-call-start in seconds for this chunk, if the backend
    /// measured it.
    #[getter]
    fn latency_seconds(&self) -> Option<f32> {
        self.inner.latency_seconds
    }

    /// Sample rate (Hz) of the f32 samples carried by this chunk.
    ///
    /// `None` on intermediate streamed chunks (consult the producing
    /// `MusicModel.sample_rate` accessor for those). Always populated on
    /// non-streaming `generate_*` results.
    #[getter]
    fn sample_rate(&self) -> Option<u32> {
        self.sample_rate
    }

    /// Number of f32 PCM samples carried by this chunk.
    #[getter]
    fn sample_count(&self) -> usize {
        self.inner.samples.len()
    }

    fn __repr__(&self) -> String {
        format!(
            "MusicChunk(samples={} f32, is_final={}, sample_rate={:?})",
            self.inner.samples.len(),
            self.inner.is_final,
            self.sample_rate,
        )
    }
}

impl From<MusicChunk> for PyMusicChunk {
    fn from(inner: MusicChunk) -> Self {
        Self {
            inner,
            sample_rate: None,
        }
    }
}

impl PyMusicChunk {
    /// Build a non-streaming final chunk from a fully-rendered
    /// [`GeneratedAudio`](blazen_audio::GeneratedAudio).
    ///
    /// The audio bytes must be raw f32 little-endian PCM samples (which
    /// is what the upstream backends' WAV pipeline carries internally for
    /// the streaming path; for the non-streaming path the bytes are WAV
    /// container and the caller-supplied `samples` slice is preferred).
    pub(crate) fn from_samples_final(samples: Vec<f32>, sample_rate: u32) -> Self {
        Self {
            inner: MusicChunk {
                samples,
                is_final: true,
                latency_seconds: None,
            },
            sample_rate: Some(sample_rate),
        }
    }
}
