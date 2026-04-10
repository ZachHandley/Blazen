//! Typed option wrappers for provider factories.

use pyo3::prelude::*;
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pyclass_enum, gen_stub_pymethods};

use blazen_llm::types::provider_options::{
    AzureOptions, BedrockOptions, FalLlmEndpointKind, FalOptions, ProviderOptions,
};

#[cfg(any(feature = "fastembed", feature = "mistralrs", feature = "whispercpp"))]
use std::path::PathBuf;

// ---------------------------------------------------------------------------
// ProviderOptions
// ---------------------------------------------------------------------------

/// Base options shared by every provider.
#[gen_stub_pyclass]
#[pyclass(name = "ProviderOptions", from_py_object)]
#[derive(Clone, Default)]
pub struct PyProviderOptions {
    pub(crate) inner: ProviderOptions,
}

#[gen_stub_pymethods]
#[pymethods]
impl PyProviderOptions {
    /// Create a new ProviderOptions.
    ///
    /// Args:
    ///     api_key: Override the provider's API key.
    ///     model: Override the default model identifier.
    ///     base_url: Override the provider's base URL.
    #[new]
    #[pyo3(signature = (*, api_key=None, model=None, base_url=None))]
    fn new(api_key: Option<String>, model: Option<String>, base_url: Option<String>) -> Self {
        Self {
            inner: ProviderOptions {
                api_key,
                model,
                base_url,
            },
        }
    }

    #[getter]
    fn api_key(&self) -> Option<String> {
        self.inner.api_key.clone()
    }
    #[setter]
    fn set_api_key(&mut self, value: Option<String>) {
        self.inner.api_key = value;
    }

    #[getter]
    fn model(&self) -> Option<String> {
        self.inner.model.clone()
    }
    #[setter]
    fn set_model(&mut self, value: Option<String>) {
        self.inner.model = value;
    }

    #[getter]
    fn base_url(&self) -> Option<String> {
        self.inner.base_url.clone()
    }
    #[setter]
    fn set_base_url(&mut self, value: Option<String>) {
        self.inner.base_url = value;
    }

    fn __repr__(&self) -> String {
        format!(
            "ProviderOptions(model={:?}, base_url={:?})",
            self.inner.model, self.inner.base_url
        )
    }
}

// ---------------------------------------------------------------------------
// FalLlmEndpointKind
// ---------------------------------------------------------------------------

/// Simplified fal.ai endpoint family.
#[gen_stub_pyclass_enum]
#[pyclass(name = "FalLlmEndpointKind", eq, eq_int, from_py_object)]
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum PyFalLlmEndpointKind {
    OpenAiChat,
    OpenAiResponses,
    OpenAiEmbeddings,
    OpenRouter,
    AnyLlm,
}

impl From<PyFalLlmEndpointKind> for FalLlmEndpointKind {
    fn from(kind: PyFalLlmEndpointKind) -> Self {
        match kind {
            PyFalLlmEndpointKind::OpenAiChat => Self::OpenAiChat,
            PyFalLlmEndpointKind::OpenAiResponses => Self::OpenAiResponses,
            PyFalLlmEndpointKind::OpenAiEmbeddings => Self::OpenAiEmbeddings,
            PyFalLlmEndpointKind::OpenRouter => Self::OpenRouter,
            PyFalLlmEndpointKind::AnyLlm => Self::AnyLlm,
        }
    }
}

// ---------------------------------------------------------------------------
// FalOptions
// ---------------------------------------------------------------------------

/// Options specific to the fal.ai provider.
#[gen_stub_pyclass]
#[pyclass(name = "FalOptions", from_py_object)]
#[derive(Clone, Default)]
pub struct PyFalOptions {
    pub(crate) inner: FalOptions,
}

#[gen_stub_pymethods]
#[pymethods]
impl PyFalOptions {
    /// Create a new FalOptions.
    ///
    /// Args:
    ///     api_key: Override the provider's API key.
    ///     model: Override the default model identifier.
    ///     base_url: Override the provider's base URL.
    ///     endpoint: The fal endpoint family (defaults to OpenAiChat).
    ///     enterprise: Promote the endpoint to its enterprise / SOC2-eligible variant.
    ///     auto_route_modality: Auto-route to vision/audio/video variant when content has media.
    #[new]
    #[pyo3(signature = (*, api_key=None, model=None, base_url=None, endpoint=None, enterprise=false, auto_route_modality=true))]
    fn new(
        api_key: Option<String>,
        model: Option<String>,
        base_url: Option<String>,
        endpoint: Option<PyFalLlmEndpointKind>,
        enterprise: bool,
        auto_route_modality: bool,
    ) -> Self {
        Self {
            inner: FalOptions {
                base: ProviderOptions {
                    api_key,
                    model,
                    base_url,
                },
                endpoint: endpoint.map(Into::into),
                enterprise,
                auto_route_modality,
            },
        }
    }

    #[getter]
    fn api_key(&self) -> Option<String> {
        self.inner.base.api_key.clone()
    }
    #[setter]
    fn set_api_key(&mut self, value: Option<String>) {
        self.inner.base.api_key = value;
    }

    #[getter]
    fn model(&self) -> Option<String> {
        self.inner.base.model.clone()
    }
    #[setter]
    fn set_model(&mut self, value: Option<String>) {
        self.inner.base.model = value;
    }

    #[getter]
    fn base_url(&self) -> Option<String> {
        self.inner.base.base_url.clone()
    }
    #[setter]
    fn set_base_url(&mut self, value: Option<String>) {
        self.inner.base.base_url = value;
    }

    #[getter]
    fn enterprise(&self) -> bool {
        self.inner.enterprise
    }
    #[setter]
    fn set_enterprise(&mut self, value: bool) {
        self.inner.enterprise = value;
    }

    #[getter]
    fn auto_route_modality(&self) -> bool {
        self.inner.auto_route_modality
    }
    #[setter]
    fn set_auto_route_modality(&mut self, value: bool) {
        self.inner.auto_route_modality = value;
    }

    fn __repr__(&self) -> String {
        format!(
            "FalOptions(model={:?}, enterprise={})",
            self.inner.base.model, self.inner.enterprise
        )
    }
}

// ---------------------------------------------------------------------------
// AzureOptions
// ---------------------------------------------------------------------------

/// Options specific to Azure OpenAI.
#[gen_stub_pyclass]
#[pyclass(name = "AzureOptions", from_py_object)]
#[derive(Clone)]
pub struct PyAzureOptions {
    pub(crate) inner: AzureOptions,
}

#[gen_stub_pymethods]
#[pymethods]
impl PyAzureOptions {
    /// Create a new AzureOptions.
    ///
    /// Args:
    ///     resource_name: Azure resource name (required).
    ///     deployment_name: Azure deployment name (required).
    ///     api_key: Override the provider's API key.
    ///     model: Override the default model identifier.
    ///     base_url: Override the provider's base URL.
    ///     api_version: API version override (e.g. "2024-02-15-preview").
    #[new]
    #[pyo3(signature = (resource_name, deployment_name, *, api_key=None, model=None, base_url=None, api_version=None))]
    fn new(
        resource_name: String,
        deployment_name: String,
        api_key: Option<String>,
        model: Option<String>,
        base_url: Option<String>,
        api_version: Option<String>,
    ) -> Self {
        Self {
            inner: AzureOptions {
                base: ProviderOptions {
                    api_key,
                    model,
                    base_url,
                },
                resource_name,
                deployment_name,
                api_version,
            },
        }
    }

    #[getter]
    fn api_key(&self) -> Option<String> {
        self.inner.base.api_key.clone()
    }
    #[setter]
    fn set_api_key(&mut self, value: Option<String>) {
        self.inner.base.api_key = value;
    }

    #[getter]
    fn model(&self) -> Option<String> {
        self.inner.base.model.clone()
    }
    #[setter]
    fn set_model(&mut self, value: Option<String>) {
        self.inner.base.model = value;
    }

    #[getter]
    fn base_url(&self) -> Option<String> {
        self.inner.base.base_url.clone()
    }
    #[setter]
    fn set_base_url(&mut self, value: Option<String>) {
        self.inner.base.base_url = value;
    }

    #[getter]
    fn resource_name(&self) -> String {
        self.inner.resource_name.clone()
    }
    #[setter]
    fn set_resource_name(&mut self, value: String) {
        self.inner.resource_name = value;
    }

    #[getter]
    fn deployment_name(&self) -> String {
        self.inner.deployment_name.clone()
    }
    #[setter]
    fn set_deployment_name(&mut self, value: String) {
        self.inner.deployment_name = value;
    }

    #[getter]
    fn api_version(&self) -> Option<String> {
        self.inner.api_version.clone()
    }
    #[setter]
    fn set_api_version(&mut self, value: Option<String>) {
        self.inner.api_version = value;
    }

    fn __repr__(&self) -> String {
        format!(
            "AzureOptions(resource_name={:?}, deployment_name={:?}, api_version={:?})",
            self.inner.resource_name, self.inner.deployment_name, self.inner.api_version
        )
    }
}

// ---------------------------------------------------------------------------
// BedrockOptions
// ---------------------------------------------------------------------------

/// Options specific to AWS Bedrock.
#[gen_stub_pyclass]
#[pyclass(name = "BedrockOptions", from_py_object)]
#[derive(Clone)]
pub struct PyBedrockOptions {
    pub(crate) inner: BedrockOptions,
}

#[gen_stub_pymethods]
#[pymethods]
impl PyBedrockOptions {
    /// Create a new BedrockOptions.
    ///
    /// Args:
    ///     region: AWS region (required, e.g. "us-east-1").
    ///     api_key: Override the provider's API key.
    ///     model: Override the default model identifier.
    ///     base_url: Override the provider's base URL.
    #[new]
    #[pyo3(signature = (region, *, api_key=None, model=None, base_url=None))]
    fn new(
        region: String,
        api_key: Option<String>,
        model: Option<String>,
        base_url: Option<String>,
    ) -> Self {
        Self {
            inner: BedrockOptions {
                base: ProviderOptions {
                    api_key,
                    model,
                    base_url,
                },
                region,
            },
        }
    }

    #[getter]
    fn api_key(&self) -> Option<String> {
        self.inner.base.api_key.clone()
    }
    #[setter]
    fn set_api_key(&mut self, value: Option<String>) {
        self.inner.base.api_key = value;
    }

    #[getter]
    fn model(&self) -> Option<String> {
        self.inner.base.model.clone()
    }
    #[setter]
    fn set_model(&mut self, value: Option<String>) {
        self.inner.base.model = value;
    }

    #[getter]
    fn base_url(&self) -> Option<String> {
        self.inner.base.base_url.clone()
    }
    #[setter]
    fn set_base_url(&mut self, value: Option<String>) {
        self.inner.base.base_url = value;
    }

    #[getter]
    fn region(&self) -> String {
        self.inner.region.clone()
    }
    #[setter]
    fn set_region(&mut self, value: String) {
        self.inner.region = value;
    }

    fn __repr__(&self) -> String {
        format!("BedrockOptions(region={:?})", self.inner.region)
    }
}

// ---------------------------------------------------------------------------
// Device
// ---------------------------------------------------------------------------

/// Hardware device selection for compute backends.
///
/// Example:
///     >>> Device.Cpu
///     >>> Device.Metal
///     >>> Device.Cuda
#[gen_stub_pyclass_enum]
#[pyclass(name = "Device", eq, eq_int, from_py_object)]
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum PyDevice {
    Cpu,
    Cuda,
    Metal,
    Vulkan,
    Rocm,
}

impl From<PyDevice> for blazen_llm::Device {
    fn from(d: PyDevice) -> Self {
        match d {
            PyDevice::Cpu => Self::Cpu,
            PyDevice::Cuda => Self::Cuda(0),
            PyDevice::Metal => Self::Metal,
            PyDevice::Vulkan => Self::Vulkan(0),
            PyDevice::Rocm => Self::Rocm(0),
        }
    }
}

// ---------------------------------------------------------------------------
// FastEmbedOptions
// ---------------------------------------------------------------------------

/// Options for the local fastembed embedding backend.
///
/// Example:
///     >>> opts = FastEmbedOptions(model_name="BGESmallENV15")
///     >>> model = EmbeddingModel.fastembed(options=opts)
#[cfg(feature = "fastembed")]
#[gen_stub_pyclass]
#[pyclass(name = "FastEmbedOptions", from_py_object)]
#[derive(Clone, Default)]
pub struct PyFastEmbedOptions {
    pub(crate) inner: blazen_llm::FastEmbedOptions,
}

#[cfg(feature = "fastembed")]
#[gen_stub_pymethods]
#[pymethods]
impl PyFastEmbedOptions {
    /// Create a new FastEmbedOptions.
    ///
    /// Args:
    ///     model_name: Fastembed model variant name (e.g. "BGESmallENV15").
    ///     cache_dir: Model cache directory path.
    ///     max_batch_size: Maximum batch size for embedding.
    ///     show_download_progress: Show download progress when fetching models.
    #[new]
    #[pyo3(signature = (*, model_name=None, cache_dir=None, max_batch_size=None, show_download_progress=None))]
    fn new(
        model_name: Option<String>,
        cache_dir: Option<String>,
        max_batch_size: Option<usize>,
        show_download_progress: Option<bool>,
    ) -> Self {
        Self {
            inner: blazen_llm::FastEmbedOptions {
                model_name,
                cache_dir: cache_dir.map(PathBuf::from),
                max_batch_size,
                show_download_progress,
            },
        }
    }

    #[getter]
    fn model_name(&self) -> Option<String> {
        self.inner.model_name.clone()
    }
    #[setter]
    fn set_model_name(&mut self, value: Option<String>) {
        self.inner.model_name = value;
    }

    #[getter]
    fn cache_dir(&self) -> Option<String> {
        self.inner
            .cache_dir
            .as_ref()
            .map(|p: &PathBuf| p.display().to_string())
    }
    #[setter]
    fn set_cache_dir(&mut self, value: Option<String>) {
        self.inner.cache_dir = value.map(PathBuf::from);
    }

    #[getter]
    fn max_batch_size(&self) -> Option<usize> {
        self.inner.max_batch_size
    }
    #[setter]
    fn set_max_batch_size(&mut self, value: Option<usize>) {
        self.inner.max_batch_size = value;
    }

    #[getter]
    fn show_download_progress(&self) -> Option<bool> {
        self.inner.show_download_progress
    }
    #[setter]
    fn set_show_download_progress(&mut self, value: Option<bool>) {
        self.inner.show_download_progress = value;
    }

    fn __repr__(&self) -> String {
        format!(
            "FastEmbedOptions(model_name={:?}, cache_dir={:?})",
            self.inner.model_name,
            self.inner
                .cache_dir
                .as_ref()
                .map(|p: &PathBuf| p.display().to_string())
        )
    }
}

// ---------------------------------------------------------------------------
// Quantization
// ---------------------------------------------------------------------------

/// Model quantization format.
///
/// Covers IEEE floating-point formats, GGML k-quant levels, and the two
/// most popular GPU quantization schemes (GPTQ, AWQ).
///
/// Example:
///     >>> opts = MistralRsOptions("my-org/model", quantization=Quantization.Q4KM)
#[gen_stub_pyclass_enum]
#[pyclass(name = "Quantization", eq, eq_int, from_py_object)]
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum PyQuantization {
    F32,
    F16,
    BF16,
    Q8_0,
    Q6K,
    Q5KM,
    Q5KS,
    Q4KM,
    Q4KS,
    Q3KM,
    Q2K,
    Gptq4Bit,
    Awq4Bit,
}

impl From<PyQuantization> for blazen_llm::Quantization {
    fn from(q: PyQuantization) -> Self {
        match q {
            PyQuantization::F32 => Self::F32,
            PyQuantization::F16 => Self::F16,
            PyQuantization::BF16 => Self::BF16,
            PyQuantization::Q8_0 => Self::Q8_0,
            PyQuantization::Q6K => Self::Q6K,
            PyQuantization::Q5KM => Self::Q5KM,
            PyQuantization::Q5KS => Self::Q5KS,
            PyQuantization::Q4KM => Self::Q4KM,
            PyQuantization::Q4KS => Self::Q4KS,
            PyQuantization::Q3KM => Self::Q3KM,
            PyQuantization::Q2K => Self::Q2K,
            PyQuantization::Gptq4Bit => Self::Gptq4Bit,
            PyQuantization::Awq4Bit => Self::Awq4Bit,
        }
    }
}

// ---------------------------------------------------------------------------
// MistralRsOptions
// ---------------------------------------------------------------------------

/// Options for the local mistral.rs LLM backend.
///
/// The ``model_id`` argument is required (HuggingFace model ID or local
/// path to a GGUF file). All other arguments are optional.
///
/// Example:
///     >>> opts = MistralRsOptions("mistralai/Mistral-7B-Instruct-v0.3")
///     >>> model = CompletionModel.mistralrs(options=opts)
#[cfg(feature = "mistralrs")]
#[gen_stub_pyclass]
#[pyclass(name = "MistralRsOptions", from_py_object)]
#[derive(Clone)]
pub struct PyMistralRsOptions {
    pub(crate) inner: blazen_llm::MistralRsOptions,
}

#[cfg(feature = "mistralrs")]
#[gen_stub_pymethods]
#[pymethods]
impl PyMistralRsOptions {
    /// Create a new MistralRsOptions.
    ///
    /// Args:
    ///     model_id: HuggingFace model ID or local GGUF path (required).
    ///     quantization: Quantization format enum value (e.g. ``Quantization.Q4KM``).
    ///     device: Hardware device enum value (e.g. ``Device.Cuda``).
    ///     context_length: Maximum context length in tokens.
    ///     max_batch_size: Maximum batch size for concurrent requests.
    ///     chat_template: Jinja2 chat template override.
    ///     cache_dir: Path to cache downloaded models.
    #[new]
    #[pyo3(signature = (model_id, *, quantization=None, device=None, context_length=None, max_batch_size=None, chat_template=None, cache_dir=None))]
    fn new(
        model_id: String,
        quantization: Option<PyQuantization>,
        device: Option<PyDevice>,
        context_length: Option<usize>,
        max_batch_size: Option<usize>,
        chat_template: Option<String>,
        cache_dir: Option<String>,
    ) -> Self {
        Self {
            inner: blazen_llm::MistralRsOptions {
                model_id,
                quantization: quantization.map(|q| {
                    let core: blazen_llm::Quantization = q.into();
                    core.as_gguf_str().to_owned()
                }),
                device: device.map(|d| {
                    let core: blazen_llm::Device = d.into();
                    core.to_string()
                }),
                context_length,
                max_batch_size,
                chat_template,
                cache_dir: cache_dir.map(PathBuf::from),
                // Vision input is not yet surfaced through the Python
                // binding. Users must construct `MistralRsOptions`
                // directly in Rust to enable vision mode.
                vision: false,
            },
        }
    }

    #[getter]
    fn model_id(&self) -> &str {
        &self.inner.model_id
    }
    #[setter]
    fn set_model_id(&mut self, value: String) {
        self.inner.model_id = value;
    }

    #[getter]
    fn quantization(&self) -> Option<String> {
        self.inner.quantization.clone()
    }
    #[setter]
    fn set_quantization(&mut self, value: Option<String>) {
        self.inner.quantization = value;
    }

    #[getter]
    fn device(&self) -> Option<String> {
        self.inner.device.clone()
    }
    #[setter]
    fn set_device(&mut self, value: Option<String>) {
        self.inner.device = value;
    }

    #[getter]
    fn context_length(&self) -> Option<usize> {
        self.inner.context_length
    }
    #[setter]
    fn set_context_length(&mut self, value: Option<usize>) {
        self.inner.context_length = value;
    }

    #[getter]
    fn max_batch_size(&self) -> Option<usize> {
        self.inner.max_batch_size
    }
    #[setter]
    fn set_max_batch_size(&mut self, value: Option<usize>) {
        self.inner.max_batch_size = value;
    }

    #[getter]
    fn chat_template(&self) -> Option<String> {
        self.inner.chat_template.clone()
    }
    #[setter]
    fn set_chat_template(&mut self, value: Option<String>) {
        self.inner.chat_template = value;
    }

    #[getter]
    fn cache_dir(&self) -> Option<String> {
        self.inner
            .cache_dir
            .as_ref()
            .map(|p: &PathBuf| p.display().to_string())
    }
    #[setter]
    fn set_cache_dir(&mut self, value: Option<String>) {
        self.inner.cache_dir = value.map(PathBuf::from);
    }

    fn __repr__(&self) -> String {
        format!(
            "MistralRsOptions(model_id={:?}, quantization={:?}, device={:?})",
            self.inner.model_id, self.inner.quantization, self.inner.device
        )
    }
}

// ---------------------------------------------------------------------------
// WhisperModel
// ---------------------------------------------------------------------------

/// Whisper model size variant for the local whisper.cpp backend.
///
/// Larger models are more accurate but require more memory and are slower.
///
/// | Variant   | Params | RAM   |
/// |-----------|--------|-------|
/// | Tiny      | 39M    | ~1GB  |
/// | Base      | 74M    | ~1GB  |
/// | Small     | 244M   | ~2GB  |
/// | Medium    | 769M   | ~5GB  |
/// | LargeV3   | 1.5B   | ~10GB |
///
/// Example:
///     >>> WhisperModel.Base
///     >>> opts = WhisperOptions(model=WhisperModel.Base)
#[cfg(feature = "whispercpp")]
#[gen_stub_pyclass_enum]
#[pyclass(name = "WhisperModel", eq, eq_int, from_py_object)]
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum PyWhisperModel {
    Tiny,
    Base,
    Small,
    Medium,
    LargeV3,
}

#[cfg(feature = "whispercpp")]
impl From<PyWhisperModel> for blazen_llm::WhisperModel {
    fn from(m: PyWhisperModel) -> Self {
        match m {
            PyWhisperModel::Tiny => Self::Tiny,
            PyWhisperModel::Base => Self::Base,
            PyWhisperModel::Small => Self::Small,
            PyWhisperModel::Medium => Self::Medium,
            PyWhisperModel::LargeV3 => Self::LargeV3,
        }
    }
}

// ---------------------------------------------------------------------------
// WhisperOptions
// ---------------------------------------------------------------------------

/// Options for the local whisper.cpp transcription backend.
///
/// All fields are optional. When ``model`` is omitted, defaults to
/// :class:`WhisperModel.Small`. When ``language`` is omitted, whisper.cpp
/// will auto-detect the spoken language.
///
/// Example:
///     >>> opts = WhisperOptions()
///     >>> opts = WhisperOptions(model=WhisperModel.Base, language="en")
///     >>> transcriber = Transcription.whispercpp(options=opts)
#[cfg(feature = "whispercpp")]
#[gen_stub_pyclass]
#[pyclass(name = "WhisperOptions", from_py_object)]
#[derive(Clone, Default)]
pub struct PyWhisperOptions {
    pub(crate) inner: blazen_llm::WhisperOptions,
}

#[cfg(feature = "whispercpp")]
#[gen_stub_pymethods]
#[pymethods]
impl PyWhisperOptions {
    /// Create a new WhisperOptions.
    ///
    /// Args:
    ///     model: Whisper model size (defaults to ``WhisperModel.Small``).
    ///     device: Hardware device specifier string (e.g. ``"cpu"``,
    ///         ``"cuda:0"``, ``"metal"``). Defaults to ``"cpu"`` when ``None``.
    ///     language: ISO 639-1 language code (e.g. ``"en"``). When ``None``,
    ///         whisper auto-detects the language.
    ///     diarize: Enable speaker diarization. Currently unsupported by the
    ///         whisper.cpp backend; setting ``True`` will cause transcription
    ///         calls to fail.
    ///     cache_dir: Directory to cache downloaded models. When ``None``,
    ///         falls back to ``$BLAZEN_CACHE_DIR`` or
    ///         ``~/.cache/blazen/models``.
    #[new]
    #[pyo3(signature = (*, model=None, device=None, language=None, diarize=None, cache_dir=None))]
    fn new(
        model: Option<PyWhisperModel>,
        device: Option<String>,
        language: Option<String>,
        diarize: Option<bool>,
        cache_dir: Option<String>,
    ) -> Self {
        Self {
            inner: blazen_llm::WhisperOptions {
                model: model.map(Into::into).unwrap_or_default(),
                device,
                language,
                diarize,
                cache_dir: cache_dir.map(PathBuf::from),
            },
        }
    }

    #[getter]
    fn device(&self) -> Option<String> {
        self.inner.device.clone()
    }
    #[setter]
    fn set_device(&mut self, value: Option<String>) {
        self.inner.device = value;
    }

    #[getter]
    fn language(&self) -> Option<String> {
        self.inner.language.clone()
    }
    #[setter]
    fn set_language(&mut self, value: Option<String>) {
        self.inner.language = value;
    }

    #[getter]
    fn diarize(&self) -> Option<bool> {
        self.inner.diarize
    }
    #[setter]
    fn set_diarize(&mut self, value: Option<bool>) {
        self.inner.diarize = value;
    }

    #[getter]
    fn cache_dir(&self) -> Option<String> {
        self.inner
            .cache_dir
            .as_ref()
            .map(|p: &PathBuf| p.display().to_string())
    }
    #[setter]
    fn set_cache_dir(&mut self, value: Option<String>) {
        self.inner.cache_dir = value.map(PathBuf::from);
    }

    fn __repr__(&self) -> String {
        format!(
            "WhisperOptions(model={}, device={:?}, language={:?})",
            self.inner.model, self.inner.device, self.inner.language
        )
    }
}
