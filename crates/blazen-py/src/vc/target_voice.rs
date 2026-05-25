//! `TargetVoice` Python wrapper.
//!
//! Mirrors [`blazen_audio_vc::TargetVoice`] — a registered target
//! speaker descriptor exposed to Python via three getters.

use blazen_audio_vc::TargetVoice;
use pyo3::prelude::*;
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pymethods};

/// A registered target speaker that a [`PyVcModel`](super::PyVcModel)
/// can render source audio into.
#[gen_stub_pyclass]
#[pyclass(name = "TargetVoice", frozen, from_py_object)]
#[derive(Clone)]
pub struct PyTargetVoice {
    pub(crate) inner: TargetVoice,
}

#[gen_stub_pymethods]
#[pymethods]
impl PyTargetVoice {
    /// Backend-scoped identifier for this voice. Pass to
    /// [`PyVcModel.convert_voice`](super::PyVcModel) and
    /// [`PyVcModel.stream_convert_pcm`](super::PyVcModel).
    #[getter]
    fn id(&self) -> &str {
        &self.inner.id
    }

    /// Optional human-readable display name. `None` for voices that ship
    /// without a label in their on-disk profile.
    #[getter]
    fn label(&self) -> Option<&str> {
        self.inner.label.as_deref()
    }

    /// Native sample rate (Hz) the backend renders this voice at.
    /// Callers must resample if a different downstream rate is required.
    #[getter]
    fn sample_rate_hz(&self) -> u32 {
        self.inner.sample_rate_hz
    }

    fn __repr__(&self) -> String {
        format!(
            "TargetVoice(id={:?}, label={:?}, sample_rate_hz={})",
            self.inner.id, self.inner.label, self.inner.sample_rate_hz,
        )
    }
}

impl From<TargetVoice> for PyTargetVoice {
    fn from(inner: TargetVoice) -> Self {
        Self { inner }
    }
}
