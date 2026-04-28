//! Typed option wrappers for provider factories.

use pyo3::prelude::*;
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pyclass_enum, gen_stub_pymethods};

use blazen_llm::types::provider_options::{
    AzureOptions, BedrockOptions, FalLlmEndpointKind, FalOptions, ProviderOptions,
};

#[cfg(any(
    feature = "embed",
    feature = "mistralrs",
    feature = "whispercpp",
    feature = "llamacpp",
    feature = "candle-llm",
    feature = "candle-embed",
    feature = "piper",
    feature = "diffusion",
))]
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
// EmbedOptions
// ---------------------------------------------------------------------------

/// Options for the local embedding backend.
///
/// Example:
///     >>> opts = EmbedOptions(model_name="BGESmallENV15")
///     >>> model = EmbeddingModel.local(options=opts)
#[cfg(feature = "embed")]
#[gen_stub_pyclass]
#[pyclass(name = "EmbedOptions", from_py_object)]
#[derive(Clone, Default)]
pub struct PyEmbedOptions {
    pub(crate) inner: blazen_llm::EmbedOptions,
}

#[cfg(feature = "embed")]
#[gen_stub_pymethods]
#[pymethods]
impl PyEmbedOptions {
    /// Create a new EmbedOptions.
    ///
    /// Args:
    ///     model_name: Model variant name (e.g. "BGESmallENV15").
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
            inner: blazen_llm::EmbedOptions {
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
            "EmbedOptions(model_name={:?}, cache_dir={:?})",
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

// ---------------------------------------------------------------------------
// LlamaCppOptions
// ---------------------------------------------------------------------------

/// Options for the local llama.cpp LLM backend.
///
/// Example:
///     >>> opts = LlamaCppOptions(model_path="/models/llama-3.2-1b-q4_k_m.gguf")
///     >>> provider = LlamaCppProvider(options=opts)
#[cfg(feature = "llamacpp")]
#[gen_stub_pyclass]
#[pyclass(name = "LlamaCppOptions", from_py_object)]
#[derive(Clone, Default)]
pub struct PyLlamaCppOptions {
    pub(crate) inner: blazen_llm::LlamaCppOptions,
}

#[cfg(feature = "llamacpp")]
#[gen_stub_pymethods]
#[pymethods]
impl PyLlamaCppOptions {
    /// Create a new LlamaCppOptions.
    ///
    /// Args:
    ///     model_path: Path to a GGUF model file or HuggingFace model ID.
    ///     device: Hardware device specifier (``"cpu"``, ``"cuda:0"``, ``"metal"``).
    ///     quantization: Quantization format string (e.g. ``"q4_k_m"``).
    ///     context_length: Maximum context length in tokens.
    ///     n_gpu_layers: Number of layers to offload to GPU.
    ///     cache_dir: Path to cache downloaded models.
    #[new]
    #[pyo3(signature = (*, model_path=None, device=None, quantization=None, context_length=None, n_gpu_layers=None, cache_dir=None))]
    fn new(
        model_path: Option<String>,
        device: Option<String>,
        quantization: Option<String>,
        context_length: Option<usize>,
        n_gpu_layers: Option<u32>,
        cache_dir: Option<String>,
    ) -> Self {
        Self {
            inner: blazen_llm::LlamaCppOptions {
                model_path,
                device,
                quantization,
                context_length,
                n_gpu_layers,
                cache_dir: cache_dir.map(PathBuf::from),
            },
        }
    }

    #[getter]
    fn model_path(&self) -> Option<String> {
        self.inner.model_path.clone()
    }
    #[setter]
    fn set_model_path(&mut self, value: Option<String>) {
        self.inner.model_path = value;
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
    fn quantization(&self) -> Option<String> {
        self.inner.quantization.clone()
    }
    #[setter]
    fn set_quantization(&mut self, value: Option<String>) {
        self.inner.quantization = value;
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
    fn n_gpu_layers(&self) -> Option<u32> {
        self.inner.n_gpu_layers
    }
    #[setter]
    fn set_n_gpu_layers(&mut self, value: Option<u32>) {
        self.inner.n_gpu_layers = value;
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
            "LlamaCppOptions(model_path={:?}, device={:?})",
            self.inner.model_path, self.inner.device
        )
    }
}

// ---------------------------------------------------------------------------
// CandleLlmOptions
// ---------------------------------------------------------------------------

/// Options for the local candle LLM backend.
///
/// Example:
///     >>> opts = CandleLlmOptions(model_id="meta-llama/Llama-3.2-1B")
///     >>> provider = CandleLlmProvider(options=opts)
#[cfg(feature = "candle-llm")]
#[gen_stub_pyclass]
#[pyclass(name = "CandleLlmOptions", from_py_object)]
#[derive(Clone, Default)]
pub struct PyCandleLlmOptions {
    pub(crate) inner: blazen_llm::CandleLlmOptions,
}

#[cfg(feature = "candle-llm")]
#[gen_stub_pymethods]
#[pymethods]
impl PyCandleLlmOptions {
    /// Create a new CandleLlmOptions.
    ///
    /// Args:
    ///     model_id: HuggingFace model ID or local path to weights.
    ///     device: Hardware device specifier (``"cpu"``, ``"cuda:0"``, ``"metal"``).
    ///     quantization: Quantization format string (e.g. ``"q4_k_m"``).
    ///     revision: Model revision / branch on HuggingFace.
    ///     context_length: Maximum context length in tokens.
    ///     cache_dir: Path to cache downloaded models.
    #[new]
    #[pyo3(signature = (*, model_id=None, device=None, quantization=None, revision=None, context_length=None, cache_dir=None))]
    fn new(
        model_id: Option<String>,
        device: Option<String>,
        quantization: Option<String>,
        revision: Option<String>,
        context_length: Option<usize>,
        cache_dir: Option<String>,
    ) -> Self {
        Self {
            inner: blazen_llm::CandleLlmOptions {
                model_id,
                device,
                quantization,
                revision,
                context_length,
                cache_dir: cache_dir.map(PathBuf::from),
            },
        }
    }

    #[getter]
    fn model_id(&self) -> Option<String> {
        self.inner.model_id.clone()
    }
    #[setter]
    fn set_model_id(&mut self, value: Option<String>) {
        self.inner.model_id = value;
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
    fn quantization(&self) -> Option<String> {
        self.inner.quantization.clone()
    }
    #[setter]
    fn set_quantization(&mut self, value: Option<String>) {
        self.inner.quantization = value;
    }

    #[getter]
    fn revision(&self) -> Option<String> {
        self.inner.revision.clone()
    }
    #[setter]
    fn set_revision(&mut self, value: Option<String>) {
        self.inner.revision = value;
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
            "CandleLlmOptions(model_id={:?}, device={:?})",
            self.inner.model_id, self.inner.device
        )
    }
}

// ---------------------------------------------------------------------------
// CandleEmbedOptions
// ---------------------------------------------------------------------------

/// Options for the local candle embedding backend.
///
/// Example:
///     >>> opts = CandleEmbedOptions(model_id="BAAI/bge-small-en-v1.5")
///     >>> model = CandleEmbedModel(options=opts)
#[cfg(feature = "candle-embed")]
#[gen_stub_pyclass]
#[pyclass(name = "CandleEmbedOptions", from_py_object)]
#[derive(Clone, Default)]
pub struct PyCandleEmbedOptions {
    pub(crate) inner: blazen_llm::CandleEmbedOptions,
}

#[cfg(feature = "candle-embed")]
#[gen_stub_pymethods]
#[pymethods]
impl PyCandleEmbedOptions {
    /// Create a new CandleEmbedOptions.
    ///
    /// Args:
    ///     model_id: HuggingFace model ID
    ///         (default: ``"sentence-transformers/all-MiniLM-L6-v2"``).
    ///     device: Hardware device specifier (``"cpu"``, ``"cuda:0"``, ``"metal"``).
    ///     revision: Model revision / git ref on HuggingFace.
    ///     cache_dir: Path to cache downloaded models.
    #[new]
    #[pyo3(signature = (*, model_id=None, device=None, revision=None, cache_dir=None))]
    fn new(
        model_id: Option<String>,
        device: Option<String>,
        revision: Option<String>,
        cache_dir: Option<String>,
    ) -> Self {
        Self {
            inner: blazen_llm::CandleEmbedOptions {
                model_id,
                device,
                revision,
                cache_dir: cache_dir.map(PathBuf::from),
            },
        }
    }

    #[getter]
    fn model_id(&self) -> Option<String> {
        self.inner.model_id.clone()
    }
    #[setter]
    fn set_model_id(&mut self, value: Option<String>) {
        self.inner.model_id = value;
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
    fn revision(&self) -> Option<String> {
        self.inner.revision.clone()
    }
    #[setter]
    fn set_revision(&mut self, value: Option<String>) {
        self.inner.revision = value;
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
            "CandleEmbedOptions(model_id={:?}, device={:?})",
            self.inner.model_id, self.inner.device
        )
    }
}

// ---------------------------------------------------------------------------
// PiperOptions
// ---------------------------------------------------------------------------

/// Options for the local Piper TTS backend.
///
/// Example:
///     >>> opts = PiperOptions(model_id="en_US-amy-medium")
///     >>> provider = PiperProvider(options=opts)
#[cfg(feature = "piper")]
#[gen_stub_pyclass]
#[pyclass(name = "PiperOptions", from_py_object)]
#[derive(Clone, Default)]
pub struct PyPiperOptions {
    pub(crate) inner: blazen_llm::PiperOptions,
}

#[cfg(feature = "piper")]
#[gen_stub_pymethods]
#[pymethods]
impl PyPiperOptions {
    /// Create a new PiperOptions.
    ///
    /// Args:
    ///     model_id: Piper voice model identifier (e.g. ``"en_US-amy-medium"``).
    ///     speaker_id: Speaker ID for multi-speaker models.
    ///     sample_rate: Output audio sample rate in Hz.
    ///     cache_dir: Path to cache downloaded voice models.
    #[new]
    #[pyo3(signature = (*, model_id=None, speaker_id=None, sample_rate=None, cache_dir=None))]
    fn new(
        model_id: Option<String>,
        speaker_id: Option<u32>,
        sample_rate: Option<u32>,
        cache_dir: Option<String>,
    ) -> Self {
        Self {
            inner: blazen_llm::PiperOptions {
                model_id,
                speaker_id,
                sample_rate,
                cache_dir: cache_dir.map(PathBuf::from),
            },
        }
    }

    #[getter]
    fn model_id(&self) -> Option<String> {
        self.inner.model_id.clone()
    }
    #[setter]
    fn set_model_id(&mut self, value: Option<String>) {
        self.inner.model_id = value;
    }

    #[getter]
    fn speaker_id(&self) -> Option<u32> {
        self.inner.speaker_id
    }
    #[setter]
    fn set_speaker_id(&mut self, value: Option<u32>) {
        self.inner.speaker_id = value;
    }

    #[getter]
    fn sample_rate(&self) -> Option<u32> {
        self.inner.sample_rate
    }
    #[setter]
    fn set_sample_rate(&mut self, value: Option<u32>) {
        self.inner.sample_rate = value;
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
            "PiperOptions(model_id={:?}, speaker_id={:?})",
            self.inner.model_id, self.inner.speaker_id
        )
    }
}

// ---------------------------------------------------------------------------
// DiffusionScheduler
// ---------------------------------------------------------------------------

/// Noise schedulers available for the diffusion process.
///
/// Different schedulers trade off between generation speed and output quality.
/// :attr:`EulerA` is a good default for most use cases.
#[cfg(feature = "diffusion")]
#[gen_stub_pyclass_enum]
#[pyclass(name = "DiffusionScheduler", eq, eq_int, from_py_object)]
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum PyDiffusionScheduler {
    Euler,
    EulerA,
    Dpm,
    Ddim,
}

#[cfg(feature = "diffusion")]
#[allow(clippy::derivable_impls)]
impl Default for PyDiffusionScheduler {
    fn default() -> Self {
        Self::EulerA
    }
}

#[cfg(feature = "diffusion")]
impl From<PyDiffusionScheduler> for blazen_llm::DiffusionScheduler {
    fn from(s: PyDiffusionScheduler) -> Self {
        match s {
            PyDiffusionScheduler::Euler => Self::Euler,
            PyDiffusionScheduler::EulerA => Self::EulerA,
            PyDiffusionScheduler::Dpm => Self::Dpm,
            PyDiffusionScheduler::Ddim => Self::Ddim,
        }
    }
}

#[cfg(feature = "diffusion")]
impl From<blazen_llm::DiffusionScheduler> for PyDiffusionScheduler {
    fn from(s: blazen_llm::DiffusionScheduler) -> Self {
        match s {
            blazen_llm::DiffusionScheduler::Euler => Self::Euler,
            blazen_llm::DiffusionScheduler::EulerA => Self::EulerA,
            blazen_llm::DiffusionScheduler::Dpm => Self::Dpm,
            blazen_llm::DiffusionScheduler::Ddim => Self::Ddim,
        }
    }
}

// ---------------------------------------------------------------------------
// DiffusionOptions
// ---------------------------------------------------------------------------

/// Options for the local diffusion-rs image generation backend.
///
/// Example:
///     >>> opts = DiffusionOptions(model_id="stabilityai/stable-diffusion-2-1", width=768, height=768)
///     >>> provider = DiffusionProvider(options=opts)
#[cfg(feature = "diffusion")]
#[gen_stub_pyclass]
#[pyclass(name = "DiffusionOptions", from_py_object)]
#[derive(Clone, Default)]
pub struct PyDiffusionOptions {
    pub(crate) inner: blazen_llm::DiffusionOptions,
}

#[cfg(feature = "diffusion")]
#[gen_stub_pymethods]
#[pymethods]
impl PyDiffusionOptions {
    /// Create a new DiffusionOptions.
    ///
    /// Args:
    ///     model_id: HuggingFace model repository ID.
    ///     device: Hardware device specifier (``"cpu"``, ``"cuda:0"``, ``"metal"``).
    ///     width: Output image width in pixels (default 512).
    ///     height: Output image height in pixels (default 512).
    ///     num_inference_steps: Number of denoising steps (default 20).
    ///     guidance_scale: Classifier-free guidance scale (default 7.5).
    ///     scheduler: Noise scheduler enum value (default ``DiffusionScheduler.EulerA``).
    ///     cache_dir: Path to cache downloaded models.
    #[new]
    #[pyo3(signature = (*, model_id=None, device=None, width=None, height=None, num_inference_steps=None, guidance_scale=None, scheduler=None, cache_dir=None))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        model_id: Option<String>,
        device: Option<String>,
        width: Option<u32>,
        height: Option<u32>,
        num_inference_steps: Option<u32>,
        guidance_scale: Option<f32>,
        scheduler: Option<PyDiffusionScheduler>,
        cache_dir: Option<String>,
    ) -> Self {
        Self {
            inner: blazen_llm::DiffusionOptions {
                model_id,
                device,
                width,
                height,
                num_inference_steps,
                guidance_scale,
                scheduler: scheduler.map(Into::into).unwrap_or_default(),
                cache_dir: cache_dir.map(PathBuf::from),
            },
        }
    }

    #[getter]
    fn model_id(&self) -> Option<String> {
        self.inner.model_id.clone()
    }
    #[setter]
    fn set_model_id(&mut self, value: Option<String>) {
        self.inner.model_id = value;
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
    fn width(&self) -> Option<u32> {
        self.inner.width
    }
    #[setter]
    fn set_width(&mut self, value: Option<u32>) {
        self.inner.width = value;
    }

    #[getter]
    fn height(&self) -> Option<u32> {
        self.inner.height
    }
    #[setter]
    fn set_height(&mut self, value: Option<u32>) {
        self.inner.height = value;
    }

    #[getter]
    fn num_inference_steps(&self) -> Option<u32> {
        self.inner.num_inference_steps
    }
    #[setter]
    fn set_num_inference_steps(&mut self, value: Option<u32>) {
        self.inner.num_inference_steps = value;
    }

    #[getter]
    fn guidance_scale(&self) -> Option<f32> {
        self.inner.guidance_scale
    }
    #[setter]
    fn set_guidance_scale(&mut self, value: Option<f32>) {
        self.inner.guidance_scale = value;
    }

    #[getter]
    fn scheduler(&self) -> PyDiffusionScheduler {
        self.inner.scheduler.into()
    }
    #[setter]
    fn set_scheduler(&mut self, value: PyDiffusionScheduler) {
        self.inner.scheduler = value.into();
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
            "DiffusionOptions(model_id={:?}, width={:?}, height={:?})",
            self.inner.model_id, self.inner.width, self.inner.height
        )
    }
}

// ---------------------------------------------------------------------------
// FastEmbedOptions  (alias of EmbedOptions facade on non-musl)
// ---------------------------------------------------------------------------

/// Options for the local fastembed (ONNX Runtime) embedding backend.
///
/// Mirrors :class:`EmbedOptions`; the underlying crate is
/// ``blazen-embed-fastembed``. Only available on non-musl targets where
/// the Microsoft-prebuilt ONNX Runtime binaries can link.
///
/// Example:
///     >>> opts = FastEmbedOptions(model_name="BGESmallENV15")
///     >>> model = FastEmbedModel(options=opts)
#[cfg(all(feature = "embed", not(target_env = "musl")))]
#[gen_stub_pyclass]
#[pyclass(name = "FastEmbedOptions", from_py_object)]
#[derive(Clone, Default)]
pub struct PyFastEmbedOptions {
    pub(crate) inner: blazen_llm::EmbedOptions,
}

#[cfg(all(feature = "embed", not(target_env = "musl")))]
#[gen_stub_pymethods]
#[pymethods]
impl PyFastEmbedOptions {
    /// Create a new FastEmbedOptions.
    ///
    /// Args:
    ///     model_name: Fastembed model variant name (e.g. ``"BGESmallENV15"``).
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
            inner: blazen_llm::EmbedOptions {
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
        format!("FastEmbedOptions(model_name={:?})", self.inner.model_name)
    }
}

// ---------------------------------------------------------------------------
// TractOptions
// ---------------------------------------------------------------------------

/// Options for the local tract (pure-Rust ONNX) embedding backend.
///
/// Mirrors :class:`FastEmbedOptions`; the underlying crate is
/// ``blazen-embed-tract``. Available on every target the ``tract``
/// feature is enabled for -- including musl and other environments
/// where fastembed's prebuilt ONNX Runtime binaries are unavailable.
///
/// Example:
///     >>> opts = TractOptions(model_name="BGESmallENV15")
///     >>> model = TractEmbedModel(options=opts)
#[cfg(feature = "tract")]
#[gen_stub_pyclass]
#[pyclass(name = "TractOptions", from_py_object)]
#[derive(Clone, Default)]
pub struct PyTractOptions {
    pub(crate) inner: blazen_embed_tract::TractOptions,
}

#[cfg(feature = "tract")]
#[gen_stub_pymethods]
#[pymethods]
impl PyTractOptions {
    /// Create a new TractOptions.
    ///
    /// Args:
    ///     model_name: Embedding model variant name (e.g. ``"BGESmallENV15"``).
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
            inner: blazen_embed_tract::TractOptions {
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
        format!("TractOptions(model_name={:?})", self.inner.model_name)
    }
}
