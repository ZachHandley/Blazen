//! Python bindings for the provider-root metadata surface:
//! [`CapabilityKind`], [`ProviderMetadata`], and the [`BaseProvider`] ABC.
//!
//! These mirror `blazen_llm::providers::{CapabilityKind, ProviderMetadata,
//! BaseProvider}`. `BaseProvider` is the polymorphic root every capability
//! provider (`LLMProvider`, `TTSProvider`, …) shares; it is exposed here as a
//! subclassable ABC whose single required override, ``metadata()``, drives the
//! convenience ``provider_id`` / ``capability`` accessors.

use pyo3::prelude::*;
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pyclass_enum, gen_stub_pymethods};

use blazen_llm::{CapabilityKind, ProviderMetadata};

// ---------------------------------------------------------------------------
// PyCapabilityKind
// ---------------------------------------------------------------------------

/// What a provider does -- the capability it serves.
#[gen_stub_pyclass_enum]
#[pyclass(name = "CapabilityKind", eq, eq_int, from_py_object)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PyCapabilityKind {
    /// Large language model -- chat / completion / streaming.
    Llm,
    /// Text-to-speech audio synthesis.
    Tts,
    /// Speech-to-text transcription.
    Stt,
    /// Text-to-music / text-to-sfx audio generation.
    Music,
    /// Voice conversion (source utterance + target voice -> re-voiced audio).
    Vc,
    /// 3D mesh generation (image-to-3D, text-to-3D).
    ThreeD,
    /// 2D image generation (text-to-image, image-to-image, upscale).
    ImageGen,
    /// Vector embedding generation (text / image / multi-modal).
    Embedding,
    /// Neural audio codec (PCM <-> discrete codebook tokens).
    Codec,
    /// Background removal on existing images.
    BackgroundRemoval,
    /// Video generation (text-to-video, image-to-video).
    Video,
}

impl From<CapabilityKind> for PyCapabilityKind {
    fn from(k: CapabilityKind) -> Self {
        match k {
            CapabilityKind::Llm => Self::Llm,
            CapabilityKind::Tts => Self::Tts,
            CapabilityKind::Stt => Self::Stt,
            CapabilityKind::Music => Self::Music,
            CapabilityKind::Vc => Self::Vc,
            CapabilityKind::ThreeD => Self::ThreeD,
            CapabilityKind::ImageGen => Self::ImageGen,
            CapabilityKind::Embedding => Self::Embedding,
            CapabilityKind::Codec => Self::Codec,
            CapabilityKind::BackgroundRemoval => Self::BackgroundRemoval,
            CapabilityKind::Video => Self::Video,
        }
    }
}

impl From<PyCapabilityKind> for CapabilityKind {
    fn from(k: PyCapabilityKind) -> Self {
        match k {
            PyCapabilityKind::Llm => Self::Llm,
            PyCapabilityKind::Tts => Self::Tts,
            PyCapabilityKind::Stt => Self::Stt,
            PyCapabilityKind::Music => Self::Music,
            PyCapabilityKind::Vc => Self::Vc,
            PyCapabilityKind::ThreeD => Self::ThreeD,
            PyCapabilityKind::ImageGen => Self::ImageGen,
            PyCapabilityKind::Embedding => Self::Embedding,
            PyCapabilityKind::Codec => Self::Codec,
            PyCapabilityKind::BackgroundRemoval => Self::BackgroundRemoval,
            PyCapabilityKind::Video => Self::Video,
        }
    }
}

#[gen_stub_pymethods]
#[pymethods]
impl PyCapabilityKind {
    /// Stable lower-snake-case string identifier for this capability.
    // `&self` is the pyo3 method-receiver convention; the
    // `trivially_copy_pass_by_ref` lint does not apply to bound methods.
    #[allow(clippy::trivially_copy_pass_by_ref)]
    fn as_str(&self) -> &'static str {
        CapabilityKind::from(*self).as_str()
    }

    #[allow(clippy::trivially_copy_pass_by_ref)]
    fn __str__(&self) -> &'static str {
        CapabilityKind::from(*self).as_str()
    }
}

// ---------------------------------------------------------------------------
// PyProviderMetadata
// ---------------------------------------------------------------------------

/// Static metadata describing a provider instance.
///
/// Carries the canonical provider id (used for routing, telemetry, billing
/// keys), the capability kind, an optional human-readable display name, and an
/// optional version pin (model id / weights revision).
///
/// Example:
/// ```text
///  >>> meta = ProviderMetadata("openai", CapabilityKind.Llm,
///  ...                         display_name="OpenAI", version="gpt-4o")
/// ```
#[gen_stub_pyclass]
#[pyclass(name = "ProviderMetadata", from_py_object)]
#[derive(Clone)]
pub struct PyProviderMetadata {
    pub(crate) inner: ProviderMetadata,
}

#[gen_stub_pymethods]
#[pymethods]
impl PyProviderMetadata {
    /// Build a metadata record.
    ///
    /// Args:
    ///     provider_id: Canonical provider identifier (e.g. ``"openai"``).
    ///     capability: What this provider does.
    ///     display_name: Optional human-readable name shown in UIs / logs.
    ///     version: Optional version pin (model id / weights revision).
    #[new]
    #[pyo3(signature = (provider_id, capability, display_name=None, version=None))]
    fn new(
        provider_id: String,
        capability: PyCapabilityKind,
        display_name: Option<String>,
        version: Option<String>,
    ) -> Self {
        let mut inner = ProviderMetadata::new(provider_id, capability.into());
        if let Some(name) = display_name {
            inner = inner.with_display_name(name);
        }
        if let Some(version) = version {
            inner = inner.with_version(version);
        }
        Self { inner }
    }

    /// Canonical provider identifier.
    #[getter]
    fn provider_id(&self) -> &str {
        &self.inner.provider_id
    }

    /// What this provider does.
    #[getter]
    fn capability(&self) -> PyCapabilityKind {
        self.inner.capability.into()
    }

    /// Optional human-readable display name.
    #[getter]
    fn display_name(&self) -> Option<&str> {
        self.inner.display_name.as_deref()
    }

    /// Optional version pin (model id / weights revision).
    #[getter]
    fn version(&self) -> Option<&str> {
        self.inner.version.as_deref()
    }

    /// The display name when set, otherwise the provider id.
    fn display(&self) -> &str {
        self.inner.display()
    }

    fn __repr__(&self) -> String {
        format!(
            "ProviderMetadata(provider_id={:?}, capability={:?}, display_name={:?}, version={:?})",
            self.inner.provider_id,
            self.inner.capability,
            self.inner.display_name,
            self.inner.version
        )
    }
}

// ---------------------------------------------------------------------------
// PyBaseProvider -- ABC subclassable from Python
// ---------------------------------------------------------------------------

/// Polymorphic root for every Blazen provider, mirroring the Rust
/// :rust:trait:`blazen_llm::providers::BaseProvider` trait.
///
/// Subclass this (or one of the capability ABCs that conceptually extend it,
/// such as :class:`LLMProvider`) and override ``metadata()`` to return a
/// :class:`ProviderMetadata`. The convenience ``provider_id`` / ``capability``
/// accessors delegate to ``metadata()`` by default.
///
/// Example:
/// ```text
///  >>> class MyProvider(BaseProvider):
///  ...     def metadata(self):
///  ...         return ProviderMetadata("mine", CapabilityKind.Llm)
/// ```
#[gen_stub_pyclass]
#[pyclass(name = "BaseProvider", subclass)]
pub struct PyBaseProviderAbc {
    /// Marker -- subclasses store their own state on the Python side.
    _private: (),
}

#[gen_stub_pymethods]
#[pymethods]
impl PyBaseProviderAbc {
    #[new]
    fn new() -> Self {
        Self { _private: () }
    }

    /// Static metadata describing this provider instance.
    ///
    /// The default implementation raises ``NotImplementedError``. Override
    /// this in a subclass to return a :class:`ProviderMetadata`.
    fn metadata(&self) -> PyResult<PyProviderMetadata> {
        Err(pyo3::exceptions::PyNotImplementedError::new_err(
            "BaseProvider.metadata must be overridden by a subclass",
        ))
    }

    /// Canonical provider identifier. Delegates to ``metadata().provider_id``.
    fn provider_id(slf: &Bound<'_, Self>) -> PyResult<String> {
        let meta = slf.call_method0("metadata")?;
        let provider_id = meta.getattr("provider_id")?;
        provider_id.extract::<String>()
    }

    /// Capability kind this provider serves. Delegates to
    /// ``metadata().capability``.
    fn capability(slf: &Bound<'_, Self>) -> PyResult<PyCapabilityKind> {
        let meta = slf.call_method0("metadata")?;
        let capability = meta.getattr("capability")?;
        Ok(capability.extract::<PyCapabilityKind>()?)
    }
}
