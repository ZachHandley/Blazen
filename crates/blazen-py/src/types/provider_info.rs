//! Python wrappers for provider-info types: [`ProviderCapabilities`],
//! [`ProviderConfig`], [`ModelCapabilities`], plus a subclassable
//! [`ProviderInfo`] ABC for user-defined providers that want to expose
//! their identity / endpoint / capabilities to Blazen's discovery layer.

use pyo3::prelude::*;
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pymethods};

use blazen_llm::traits::{ModelCapabilities, ModelPricing, ProviderCapabilities, ProviderConfig};

use super::PyModelPricing;

// ---------------------------------------------------------------------------
// PyProviderCapabilities
// ---------------------------------------------------------------------------

/// Capabilities advertised by a provider as a whole.
///
/// Mirrors [`blazen_llm::traits::ProviderCapabilities`].
#[gen_stub_pyclass]
#[pyclass(name = "ProviderCapabilities", from_py_object)]
#[derive(Clone, Default)]
pub struct PyProviderCapabilities {
    pub(crate) inner: ProviderCapabilities,
}

#[gen_stub_pymethods]
#[pymethods]
impl PyProviderCapabilities {
    #[new]
    #[pyo3(signature = (
        *,
        streaming=false,
        tool_calling=false,
        structured_output=false,
        vision=false,
        model_listing=false,
        embeddings=false,
    ))]
    #[allow(clippy::fn_params_excessive_bools)]
    fn new(
        streaming: bool,
        tool_calling: bool,
        structured_output: bool,
        vision: bool,
        model_listing: bool,
        embeddings: bool,
    ) -> Self {
        Self {
            inner: ProviderCapabilities {
                streaming,
                tool_calling,
                structured_output,
                vision,
                model_listing,
                embeddings,
            },
        }
    }

    #[getter]
    fn streaming(&self) -> bool {
        self.inner.streaming
    }

    #[getter]
    fn tool_calling(&self) -> bool {
        self.inner.tool_calling
    }

    #[getter]
    fn structured_output(&self) -> bool {
        self.inner.structured_output
    }

    #[getter]
    fn vision(&self) -> bool {
        self.inner.vision
    }

    #[getter]
    fn model_listing(&self) -> bool {
        self.inner.model_listing
    }

    #[getter]
    fn embeddings(&self) -> bool {
        self.inner.embeddings
    }

    fn __repr__(&self) -> String {
        format!(
            "ProviderCapabilities(streaming={}, tool_calling={}, vision={}, embeddings={})",
            self.inner.streaming, self.inner.tool_calling, self.inner.vision, self.inner.embeddings,
        )
    }
}

impl From<ProviderCapabilities> for PyProviderCapabilities {
    fn from(inner: ProviderCapabilities) -> Self {
        Self { inner }
    }
}

// ---------------------------------------------------------------------------
// PyModelCapabilities
// ---------------------------------------------------------------------------

/// What a single model can do.
///
/// Mirrors [`blazen_llm::traits::ModelCapabilities`]. Returned as part of
/// [`ModelInfo`] entries.
#[gen_stub_pyclass]
#[pyclass(name = "ModelCapabilities", from_py_object)]
#[derive(Clone, Default)]
pub struct PyModelCapabilities {
    pub(crate) inner: ModelCapabilities,
}

#[gen_stub_pymethods]
#[pymethods]
impl PyModelCapabilities {
    #[new]
    #[pyo3(signature = (
        *,
        chat=false,
        streaming=false,
        tool_use=false,
        structured_output=false,
        vision=false,
        image_generation=false,
        embeddings=false,
        video_generation=false,
        text_to_speech=false,
        speech_to_text=false,
        audio_generation=false,
        three_d_generation=false,
    ))]
    #[allow(clippy::too_many_arguments, clippy::fn_params_excessive_bools)]
    fn new(
        chat: bool,
        streaming: bool,
        tool_use: bool,
        structured_output: bool,
        vision: bool,
        image_generation: bool,
        embeddings: bool,
        video_generation: bool,
        text_to_speech: bool,
        speech_to_text: bool,
        audio_generation: bool,
        three_d_generation: bool,
    ) -> Self {
        Self {
            inner: ModelCapabilities {
                chat,
                streaming,
                tool_use,
                structured_output,
                vision,
                image_generation,
                embeddings,
                video_generation,
                text_to_speech,
                speech_to_text,
                audio_generation,
                three_d_generation,
            },
        }
    }

    #[getter]
    fn chat(&self) -> bool {
        self.inner.chat
    }
    #[getter]
    fn streaming(&self) -> bool {
        self.inner.streaming
    }
    #[getter]
    fn tool_use(&self) -> bool {
        self.inner.tool_use
    }
    #[getter]
    fn structured_output(&self) -> bool {
        self.inner.structured_output
    }
    #[getter]
    fn vision(&self) -> bool {
        self.inner.vision
    }
    #[getter]
    fn image_generation(&self) -> bool {
        self.inner.image_generation
    }
    #[getter]
    fn embeddings(&self) -> bool {
        self.inner.embeddings
    }
    #[getter]
    fn video_generation(&self) -> bool {
        self.inner.video_generation
    }
    #[getter]
    fn text_to_speech(&self) -> bool {
        self.inner.text_to_speech
    }
    #[getter]
    fn speech_to_text(&self) -> bool {
        self.inner.speech_to_text
    }
    #[getter]
    fn audio_generation(&self) -> bool {
        self.inner.audio_generation
    }
    #[getter]
    fn three_d_generation(&self) -> bool {
        self.inner.three_d_generation
    }

    fn __repr__(&self) -> String {
        format!(
            "ModelCapabilities(chat={}, streaming={}, tool_use={}, vision={})",
            self.inner.chat, self.inner.streaming, self.inner.tool_use, self.inner.vision,
        )
    }
}

impl From<ModelCapabilities> for PyModelCapabilities {
    fn from(inner: ModelCapabilities) -> Self {
        Self { inner }
    }
}

// ---------------------------------------------------------------------------
// PyProviderConfig
// ---------------------------------------------------------------------------

/// Configuration metadata for a provider instance.
///
/// Mirrors [`blazen_llm::traits::ProviderConfig`]. Used by custom providers
/// to advertise their identity, endpoint, pricing, and resource information.
#[gen_stub_pyclass]
#[pyclass(name = "ProviderConfig", from_py_object)]
#[derive(Clone, Default)]
pub struct PyProviderConfig {
    pub(crate) inner: ProviderConfig,
}

#[gen_stub_pymethods]
#[pymethods]
impl PyProviderConfig {
    #[new]
    #[pyo3(signature = (
        *,
        name=None,
        model_id=None,
        provider_id=None,
        base_url=None,
        context_length=None,
        max_output_tokens=None,
        memory_estimate_bytes=None,
        pricing=None,
        capabilities=None,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        name: Option<String>,
        model_id: Option<String>,
        provider_id: Option<String>,
        base_url: Option<String>,
        context_length: Option<u64>,
        max_output_tokens: Option<u64>,
        memory_estimate_bytes: Option<u64>,
        pricing: Option<PyModelPricing>,
        capabilities: Option<PyModelCapabilities>,
    ) -> Self {
        let pricing_inner: Option<ModelPricing> = pricing.map(|p| p.inner);
        let caps_inner: Option<ModelCapabilities> = capabilities.map(|c| c.inner);
        Self {
            inner: ProviderConfig {
                name,
                model_id,
                provider_id,
                base_url,
                context_length,
                max_output_tokens,
                memory_estimate_bytes,
                pricing: pricing_inner,
                capabilities: caps_inner,
            },
        }
    }

    #[getter]
    fn name(&self) -> Option<&str> {
        self.inner.name.as_deref()
    }
    #[getter]
    fn model_id(&self) -> Option<&str> {
        self.inner.model_id.as_deref()
    }
    #[getter]
    fn provider_id(&self) -> Option<&str> {
        self.inner.provider_id.as_deref()
    }
    #[getter]
    fn base_url(&self) -> Option<&str> {
        self.inner.base_url.as_deref()
    }
    #[getter]
    fn context_length(&self) -> Option<u64> {
        self.inner.context_length
    }
    #[getter]
    fn max_output_tokens(&self) -> Option<u64> {
        self.inner.max_output_tokens
    }
    #[getter]
    fn memory_estimate_bytes(&self) -> Option<u64> {
        self.inner.memory_estimate_bytes
    }
    #[getter]
    fn pricing(&self) -> Option<PyModelPricing> {
        self.inner
            .pricing
            .clone()
            .map(|p| PyModelPricing { inner: p })
    }
    #[getter]
    fn capabilities(&self) -> Option<PyModelCapabilities> {
        self.inner
            .capabilities
            .clone()
            .map(|c| PyModelCapabilities { inner: c })
    }

    fn __repr__(&self) -> String {
        format!(
            "ProviderConfig(name={:?}, provider_id={:?}, model_id={:?})",
            self.inner.name, self.inner.provider_id, self.inner.model_id,
        )
    }
}

impl From<ProviderConfig> for PyProviderConfig {
    fn from(inner: ProviderConfig) -> Self {
        Self { inner }
    }
}

// ---------------------------------------------------------------------------
// PyProviderInfo (ABC)
// ---------------------------------------------------------------------------

/// Subclassable ABC that mirrors [`blazen_llm::traits::ProviderInfo`].
///
/// Implement to expose a custom provider's identity, endpoint, and
/// capabilities to Blazen's discovery surface. Override every method --
/// the default implementations raise ``NotImplementedError``.
///
/// Example:
///     >>> class MyProviderInfo(ProviderInfo):
///     ...     def provider_name(self) -> str: return "my-provider"
///     ...     def base_url(self) -> str: return "https://api.example.com"
///     ...     def capabilities(self) -> ProviderCapabilities:
///     ...         return ProviderCapabilities(streaming=True, tool_calling=True)
#[gen_stub_pyclass]
#[pyclass(name = "ProviderInfo", subclass)]
pub struct PyProviderInfo;

#[gen_stub_pymethods]
#[pymethods]
impl PyProviderInfo {
    #[new]
    fn new() -> Self {
        Self
    }

    /// The provider's canonical name (e.g. `"openai"`, `"anthropic"`).
    fn provider_name(&self) -> PyResult<String> {
        Err(pyo3::exceptions::PyNotImplementedError::new_err(
            "subclass must override provider_name()",
        ))
    }

    /// The provider's base API URL.
    fn base_url(&self) -> PyResult<String> {
        Err(pyo3::exceptions::PyNotImplementedError::new_err(
            "subclass must override base_url()",
        ))
    }

    /// The provider's capability flags as a [`ProviderCapabilities`] object.
    fn capabilities(&self) -> PyResult<PyProviderCapabilities> {
        Err(pyo3::exceptions::PyNotImplementedError::new_err(
            "subclass must override capabilities()",
        ))
    }
}
