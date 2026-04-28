//! Python wrapper for the CompletionModel type with all provider constructors.

use std::pin::Pin;
use std::sync::Arc;

use pyo3::prelude::*;
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pymethods};
use tokio::sync::Mutex;
use tokio_stream::{Stream, StreamExt};

use blazen_llm::cache::CachedCompletionModel;
use blazen_llm::fallback::FallbackModel;
use blazen_llm::retry::RetryCompletionModel;
use blazen_llm::types::ToolDefinition;
use blazen_llm::{BlazenError, ChatMessage, CompletionModel, CompletionRequest, StreamChunk};

use crate::agent::PyToolDef;
use crate::error::BlazenPyError;
use crate::providers::config::{PyCacheConfig, PyRetryConfig};
use crate::providers::options::{
    PyAzureOptions, PyBedrockOptions, PyFalOptions, PyProviderOptions,
};
use crate::types::{PyChatMessage, PyCompletionResponse};

/// Type alias for the pinned boxed stream returned by `CompletionModel::stream`.
type PinnedChunkStream = Pin<Box<dyn Stream<Item = Result<StreamChunk, BlazenError>> + Send>>;

// ---------------------------------------------------------------------------
// PyCompletionOptions
// ---------------------------------------------------------------------------

/// Options for a chat completion request.
///
/// Example:
///     >>> opts = CompletionOptions(temperature=0.7, max_tokens=1000)
///     >>> response = await model.complete(messages, opts)
#[gen_stub_pyclass]
#[pyclass(name = "CompletionOptions")]
#[derive(Debug, Default)]
pub struct PyCompletionOptions {
    /// Sampling temperature (0.0-2.0).
    #[pyo3(get, set)]
    pub temperature: Option<f32>,
    /// Maximum tokens to generate.
    #[pyo3(get, set)]
    pub max_tokens: Option<u32>,
    /// Nucleus sampling parameter (0.0-1.0).
    #[pyo3(get, set)]
    pub top_p: Option<f32>,
    /// Model override for this request.
    #[pyo3(get, set)]
    pub model: Option<String>,
    /// Tool definitions for function calling.
    #[pyo3(get, set)]
    pub tools: Option<Vec<Py<PyToolDef>>>,
    /// JSON schema dict for structured output.
    #[pyo3(get, set)]
    pub response_format: Option<Py<PyAny>>,
}

#[gen_stub_pymethods]
#[pymethods]
impl PyCompletionOptions {
    #[new]
    #[pyo3(signature = (temperature=None, max_tokens=None, top_p=None, model=None, tools=None, response_format=None))]
    fn new(
        temperature: Option<f32>,
        max_tokens: Option<u32>,
        top_p: Option<f32>,
        model: Option<String>,
        tools: Option<Vec<Py<PyToolDef>>>,
        response_format: Option<Py<PyAny>>,
    ) -> Self {
        Self {
            temperature,
            max_tokens,
            top_p,
            model,
            tools,
            response_format,
        }
    }
}

// ---------------------------------------------------------------------------
// PyCompletionModel
// ---------------------------------------------------------------------------

/// A chat completion model.
///
/// Use the static constructor methods to create a model for a specific
/// provider, then call `complete()` to generate responses.
///
/// Example:
///     >>> model = CompletionModel.openai()
///     >>> model = CompletionModel.anthropic()
///     >>> model = CompletionModel.openrouter()
///     >>>
///     >>> response = await model.complete([
///     ...     ChatMessage.user("What is 2+2?")
///     ... ])
#[gen_stub_pyclass]
#[pyclass(name = "CompletionModel", subclass, from_py_object)]
#[derive(Clone)]
pub struct PyCompletionModel {
    pub(crate) inner: Option<Arc<dyn CompletionModel>>,
    /// Present iff the underlying provider is a local in-process model
    /// (mistral.rs, llama.cpp, candle) that implements
    /// [`blazen_llm::LocalModel`]. `None` for remote HTTP providers.
    /// Populated by the provider factory methods.
    pub(crate) local_model: Option<Arc<dyn blazen_llm::LocalModel>>,
    /// Configuration for subclassed models. `None` for built-in providers
    /// (whose config lives inside the `inner` trait object).
    pub(crate) config: Option<blazen_llm::ProviderConfig>,
}

#[gen_stub_pymethods]
#[pymethods]
impl PyCompletionModel {
    // -----------------------------------------------------------------
    // Subclass constructor
    // -----------------------------------------------------------------

    /// Create a custom completion model by subclassing.
    ///
    /// Override ``complete()`` and optionally ``stream()`` in your
    /// subclass to implement a custom provider.
    ///
    /// Args:
    ///     model_id: The model identifier.
    ///     context_length: Maximum context window in tokens.
    ///     base_url: Base URL for HTTP-based providers.
    ///     pricing: Optional pricing information.
    ///     vram_estimate_bytes: Estimated VRAM footprint in bytes.
    ///     max_output_tokens: Maximum output tokens the model supports.
    #[new]
    #[pyo3(signature = (*, model_id=None, context_length=None, base_url=None, pricing=None, vram_estimate_bytes=None, max_output_tokens=None))]
    fn new(
        model_id: Option<String>,
        context_length: Option<u64>,
        base_url: Option<String>,
        pricing: Option<PyRef<'_, crate::types::pricing::PyModelPricing>>,
        vram_estimate_bytes: Option<u64>,
        max_output_tokens: Option<u64>,
    ) -> Self {
        Self {
            inner: None,
            local_model: None,
            config: Some(blazen_llm::ProviderConfig {
                model_id,
                context_length,
                base_url,
                vram_estimate_bytes,
                max_output_tokens,
                pricing: pricing.map(|p| p.inner.clone()),
                ..Default::default()
            }),
        }
    }

    // -----------------------------------------------------------------
    // Provider constructors
    // -----------------------------------------------------------------
    //
    // Each factory deserializes its native options dict into the typed
    // core struct, then delegates to the provider's `from_options()`
    // method. The construction logic itself lives in `blazen-llm` (see
    // `crates/blazen-llm/src/providers/mod.rs::impl_simple_from_options`).

    /// Create an OpenAI provider.
    ///
    /// Args:
    ///     options: Optional typed ``ProviderOptions`` object.
    #[staticmethod]
    #[pyo3(signature = (*, options=None))]
    fn openai(options: Option<PyRef<'_, PyProviderOptions>>) -> PyResult<Self> {
        let opts = options.map(|o| o.inner.clone()).unwrap_or_default();
        Ok(Self {
            inner: Some(Arc::new(
                blazen_llm::providers::openai::OpenAiProvider::from_options(opts)
                    .map_err(crate::error::blazen_error_to_pyerr)?,
            )),
            local_model: None,
            config: None,
        })
    }

    /// Create an Anthropic provider.
    ///
    /// Args:
    ///     options: Optional typed ``ProviderOptions`` object.
    #[staticmethod]
    #[pyo3(signature = (*, options=None))]
    fn anthropic(options: Option<PyRef<'_, PyProviderOptions>>) -> PyResult<Self> {
        let opts = options.map(|o| o.inner.clone()).unwrap_or_default();
        Ok(Self {
            inner: Some(Arc::new(
                blazen_llm::providers::anthropic::AnthropicProvider::from_options(opts)
                    .map_err(crate::error::blazen_error_to_pyerr)?,
            )),
            local_model: None,
            config: None,
        })
    }

    /// Create a Google Gemini provider.
    ///
    /// Args:
    ///     options: Optional typed ``ProviderOptions`` object.
    #[staticmethod]
    #[pyo3(signature = (*, options=None))]
    fn gemini(options: Option<PyRef<'_, PyProviderOptions>>) -> PyResult<Self> {
        let opts = options.map(|o| o.inner.clone()).unwrap_or_default();
        Ok(Self {
            inner: Some(Arc::new(
                blazen_llm::providers::gemini::GeminiProvider::from_options(opts)
                    .map_err(crate::error::blazen_error_to_pyerr)?,
            )),
            local_model: None,
            config: None,
        })
    }

    /// Create an Azure OpenAI provider.
    ///
    /// Args:
    ///     options: Typed ``AzureOptions`` object with required
    ///         ``resource_name`` and ``deployment_name``, plus optional
    ///         ``api_version``.
    #[staticmethod]
    #[pyo3(signature = (*, options))]
    fn azure(options: PyRef<'_, PyAzureOptions>) -> PyResult<Self> {
        let opts = options.inner.clone();
        Ok(Self {
            inner: Some(Arc::new(
                blazen_llm::providers::azure::AzureOpenAiProvider::from_options(opts)
                    .map_err(crate::error::blazen_error_to_pyerr)?,
            )),
            local_model: None,
            config: None,
        })
    }

    /// Create an OpenRouter provider.
    ///
    /// Args:
    ///     options: Optional typed ``ProviderOptions`` object.
    #[staticmethod]
    #[pyo3(signature = (*, options=None))]
    fn openrouter(options: Option<PyRef<'_, PyProviderOptions>>) -> PyResult<Self> {
        let opts = options.map(|o| o.inner.clone()).unwrap_or_default();
        Ok(Self {
            inner: Some(Arc::new(
                blazen_llm::providers::openrouter::OpenRouterProvider::from_options(opts)
                    .map_err(crate::error::blazen_error_to_pyerr)?,
            )),
            local_model: None,
            config: None,
        })
    }

    /// Create a Groq provider.
    ///
    /// Args:
    ///     options: Optional typed ``ProviderOptions`` object.
    #[staticmethod]
    #[pyo3(signature = (*, options=None))]
    fn groq(options: Option<PyRef<'_, PyProviderOptions>>) -> PyResult<Self> {
        let opts = options.map(|o| o.inner.clone()).unwrap_or_default();
        Ok(Self {
            inner: Some(Arc::new(
                blazen_llm::providers::groq::GroqProvider::from_options(opts)
                    .map_err(crate::error::blazen_error_to_pyerr)?,
            )),
            local_model: None,
            config: None,
        })
    }

    /// Create a Together AI provider.
    ///
    /// Args:
    ///     options: Optional typed ``ProviderOptions`` object.
    #[staticmethod]
    #[pyo3(signature = (*, options=None))]
    fn together(options: Option<PyRef<'_, PyProviderOptions>>) -> PyResult<Self> {
        let opts = options.map(|o| o.inner.clone()).unwrap_or_default();
        Ok(Self {
            inner: Some(Arc::new(
                blazen_llm::providers::together::TogetherProvider::from_options(opts)
                    .map_err(crate::error::blazen_error_to_pyerr)?,
            )),
            local_model: None,
            config: None,
        })
    }

    /// Create a Mistral provider.
    ///
    /// Args:
    ///     options: Optional typed ``ProviderOptions`` object.
    #[staticmethod]
    #[pyo3(signature = (*, options=None))]
    fn mistral(options: Option<PyRef<'_, PyProviderOptions>>) -> PyResult<Self> {
        let opts = options.map(|o| o.inner.clone()).unwrap_or_default();
        Ok(Self {
            inner: Some(Arc::new(
                blazen_llm::providers::mistral::MistralProvider::from_options(opts)
                    .map_err(crate::error::blazen_error_to_pyerr)?,
            )),
            local_model: None,
            config: None,
        })
    }

    /// Create a DeepSeek provider.
    ///
    /// Args:
    ///     options: Optional typed ``ProviderOptions`` object.
    #[staticmethod]
    #[pyo3(signature = (*, options=None))]
    fn deepseek(options: Option<PyRef<'_, PyProviderOptions>>) -> PyResult<Self> {
        let opts = options.map(|o| o.inner.clone()).unwrap_or_default();
        Ok(Self {
            inner: Some(Arc::new(
                blazen_llm::providers::deepseek::DeepSeekProvider::from_options(opts)
                    .map_err(crate::error::blazen_error_to_pyerr)?,
            )),
            local_model: None,
            config: None,
        })
    }

    /// Create a Fireworks AI provider.
    ///
    /// Args:
    ///     options: Optional typed ``ProviderOptions`` object.
    #[staticmethod]
    #[pyo3(signature = (*, options=None))]
    fn fireworks(options: Option<PyRef<'_, PyProviderOptions>>) -> PyResult<Self> {
        let opts = options.map(|o| o.inner.clone()).unwrap_or_default();
        Ok(Self {
            inner: Some(Arc::new(
                blazen_llm::providers::fireworks::FireworksProvider::from_options(opts)
                    .map_err(crate::error::blazen_error_to_pyerr)?,
            )),
            local_model: None,
            config: None,
        })
    }

    /// Create a Perplexity provider.
    ///
    /// Args:
    ///     options: Optional typed ``ProviderOptions`` object.
    #[staticmethod]
    #[pyo3(signature = (*, options=None))]
    fn perplexity(options: Option<PyRef<'_, PyProviderOptions>>) -> PyResult<Self> {
        let opts = options.map(|o| o.inner.clone()).unwrap_or_default();
        Ok(Self {
            inner: Some(Arc::new(
                blazen_llm::providers::perplexity::PerplexityProvider::from_options(opts)
                    .map_err(crate::error::blazen_error_to_pyerr)?,
            )),
            local_model: None,
            config: None,
        })
    }

    /// Create an xAI (Grok) provider.
    ///
    /// Args:
    ///     options: Optional typed ``ProviderOptions`` object.
    #[staticmethod]
    #[pyo3(signature = (*, options=None))]
    fn xai(options: Option<PyRef<'_, PyProviderOptions>>) -> PyResult<Self> {
        let opts = options.map(|o| o.inner.clone()).unwrap_or_default();
        Ok(Self {
            inner: Some(Arc::new(
                blazen_llm::providers::xai::XaiProvider::from_options(opts)
                    .map_err(crate::error::blazen_error_to_pyerr)?,
            )),
            local_model: None,
            config: None,
        })
    }

    /// Create a Cohere provider.
    ///
    /// Args:
    ///     options: Optional typed ``ProviderOptions`` object.
    #[staticmethod]
    #[pyo3(signature = (*, options=None))]
    fn cohere(options: Option<PyRef<'_, PyProviderOptions>>) -> PyResult<Self> {
        let opts = options.map(|o| o.inner.clone()).unwrap_or_default();
        Ok(Self {
            inner: Some(Arc::new(
                blazen_llm::providers::cohere::CohereProvider::from_options(opts)
                    .map_err(crate::error::blazen_error_to_pyerr)?,
            )),
            local_model: None,
            config: None,
        })
    }

    /// Create an AWS Bedrock provider.
    ///
    /// Args:
    ///     options: Typed ``BedrockOptions`` object with required ``region``
    ///         and optional ``model``.
    #[staticmethod]
    #[pyo3(signature = (*, options))]
    fn bedrock(options: PyRef<'_, PyBedrockOptions>) -> PyResult<Self> {
        let opts = options.inner.clone();
        Ok(Self {
            inner: Some(Arc::new(
                blazen_llm::providers::bedrock::BedrockProvider::from_options(opts)
                    .map_err(crate::error::blazen_error_to_pyerr)?,
            )),
            local_model: None,
            config: None,
        })
    }

    /// Create a fal.ai provider.
    ///
    /// Args:
    ///     options: Optional typed ``FalOptions`` object for selecting the
    ///         model, endpoint, enterprise tier, and auto-routing. Defaults to
    ///         the OpenAI-chat endpoint
    ///         (``openrouter/router/openai/v1/chat/completions``).
    #[staticmethod]
    #[pyo3(signature = (*, options=None))]
    fn fal(options: Option<PyRef<'_, PyFalOptions>>) -> PyResult<Self> {
        let opts = options.map(|o| o.inner.clone()).unwrap_or_default();
        Ok(Self {
            inner: Some(Arc::new(
                blazen_llm::providers::fal::FalProvider::from_options(opts)
                    .map_err(crate::error::blazen_error_to_pyerr)?,
            )),
            local_model: None,
            config: None,
        })
    }

    // -----------------------------------------------------------------
    // Model info
    // -----------------------------------------------------------------

    /// Get the model ID.
    ///
    /// Returns:
    ///     The string identifier of the model.
    #[getter]
    fn model_id(&self) -> String {
        if let Some(ref inner) = self.inner {
            inner.model_id().to_owned()
        } else {
            self.config
                .as_ref()
                .and_then(|c| c.model_id.clone())
                .unwrap_or_default()
        }
    }

    // -----------------------------------------------------------------
    // Decorators: retry, fallback, caching
    // -----------------------------------------------------------------

    /// Wrap this model with automatic retry on transient failures.
    ///
    /// Returns a new CompletionModel that retries on rate limits,
    /// timeouts, and server errors with exponential backoff.
    ///
    /// Args:
    ///     config: Optional typed ``RetryConfig`` object. Defaults to
    ///         ``RetryConfig()`` (3 retries, 1s initial, 30s max).
    ///
    /// Returns:
    ///     A new CompletionModel with retry behaviour.
    ///
    /// Example:
    ///     >>> model = CompletionModel.openai(options=ProviderOptions(api_key="sk-...")).with_retry(RetryConfig(max_retries=5))
    #[pyo3(signature = (config=None))]
    fn with_retry(slf: Bound<'_, Self>, config: Option<PyRef<'_, PyRetryConfig>>) -> Self {
        let retry_config = config.map(|c| c.inner.clone()).unwrap_or_default();
        let local_model = slf.borrow().local_model.clone();
        let inner = arc_from_bound(&slf);
        let model = RetryCompletionModel::from_arc(inner, retry_config);
        Self {
            inner: Some(Arc::new(model)),
            local_model,
            config: None,
        }
    }

    /// Create a fallback model that tries multiple providers in order.
    ///
    /// When the first provider fails with a retryable error, the request
    /// is forwarded to the next provider. Non-retryable errors (auth,
    /// validation) short-circuit immediately.
    ///
    /// Args:
    ///     models: A list of CompletionModel instances to try in order.
    ///
    /// Returns:
    ///     A new CompletionModel that falls back through the providers.
    ///
    /// Example:
    ///     >>> primary = CompletionModel.openai(options=ProviderOptions(api_key="sk-..."))
    ///     >>> backup = CompletionModel.anthropic(options=ProviderOptions(api_key="sk-ant-..."))
    ///     >>> model = CompletionModel.with_fallback([primary, backup])
    #[staticmethod]
    fn with_fallback(models: Vec<Bound<'_, PyCompletionModel>>) -> PyResult<Self> {
        if models.is_empty() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "with_fallback requires at least one model",
            ));
        }
        let providers: Vec<Arc<dyn CompletionModel>> = models.iter().map(arc_from_bound).collect();
        let model = FallbackModel::new(providers);
        // A fallback chain is a composition of heterogeneous providers, so
        // there is no single `LocalModel` to forward load/unload to. Callers
        // that need local-model control should apply it to the individual
        // component models before combining them via `with_fallback`.
        Ok(Self {
            inner: Some(Arc::new(model)),
            local_model: None,
            config: None,
        })
    }

    /// Wrap this model with response caching.
    ///
    /// Repeated identical requests are served from an in-memory cache
    /// without hitting the underlying provider. Streaming requests are
    /// never cached.
    ///
    /// Args:
    ///     config: Optional typed ``CacheConfig`` object. Defaults to
    ///         ``CacheConfig()`` (content-hash strategy, 300s TTL, 1000 entries).
    ///
    /// Returns:
    ///     A new CompletionModel with caching enabled.
    ///
    /// Example:
    ///     >>> model = CompletionModel.openai(options=ProviderOptions(api_key="sk-...")).with_cache(CacheConfig(ttl_seconds=600))
    #[pyo3(signature = (config=None))]
    fn with_cache(slf: Bound<'_, Self>, config: Option<PyRef<'_, PyCacheConfig>>) -> Self {
        let cache_config = config.map(|c| c.inner.clone()).unwrap_or_default();
        let local_model = slf.borrow().local_model.clone();
        let inner = arc_from_bound(&slf);
        let model = CachedCompletionModel::from_arc(inner, cache_config);
        Self {
            inner: Some(Arc::new(model)),
            local_model,
            config: None,
        }
    }

    // -----------------------------------------------------------------
    // Completion
    // -----------------------------------------------------------------

    /// Perform a chat completion.
    ///
    /// Args:
    ///     messages: A list of ChatMessage objects.
    ///     options: Optional CompletionOptions for sampling parameters,
    ///         tools, and response format.
    ///
    /// Returns:
    ///     A CompletionResponse with content, model, tool_calls, usage,
    ///     and finish_reason attributes.
    ///
    /// Example:
    ///     >>> response = await model.complete([
    ///     ...     ChatMessage.system("You are helpful."),
    ///     ...     ChatMessage.user("What is 2+2?"),
    ///     ... ])
    ///     >>> print(response.content)
    #[pyo3(signature = (messages, options=None))]
    #[gen_stub(override_return_type(type_repr = "typing.Coroutine[typing.Any, typing.Any, CompletionResponse]", imports = ("typing",)))]
    fn complete<'py>(
        &self,
        py: Python<'py>,
        messages: Vec<PyRef<'py, PyChatMessage>>,
        options: Option<PyRef<'py, PyCompletionOptions>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        if let Some(ref inner) = self.inner {
            // Built-in provider path
            let rust_messages: Vec<ChatMessage> =
                messages.iter().map(|m| m.inner.clone()).collect();
            let request = build_request(py, rust_messages, options.as_deref())?;
            let inner = inner.clone();
            pyo3_async_runtimes::tokio::future_into_py(py, async move {
                let response = inner.complete(request).await.map_err(BlazenPyError::from)?;
                Ok(PyCompletionResponse { inner: response })
            })
        } else {
            // Subclass path -- if we got here, the subclass didn't override complete()
            Err(pyo3::exceptions::PyNotImplementedError::new_err(
                "subclass must override complete()",
            ))
        }
    }

    /// Stream a chat completion.
    ///
    /// Two usage modes are supported:
    ///
    /// 1. **Async iterator** (``on_chunk`` omitted) -- returns a
    ///    [`CompletionStream`] which can be consumed with ``async for``.
    /// 2. **Callback** (``on_chunk`` provided) -- returns a coroutine that
    ///    resolves once the stream is exhausted, invoking ``on_chunk`` once
    ///    per [`StreamChunk`].
    ///
    /// Args:
    ///     messages: A list of ChatMessage objects.
    ///     on_chunk: Optional callback function receiving each chunk. If
    ///         omitted, the method returns an async iterator.
    ///     options: Optional [`CompletionOptions`] for sampling parameters,
    ///         tools, and response format.
    ///
    /// Example (async iterator):
    ///     >>> async for chunk in model.stream([ChatMessage.user("Hi!")]):
    ///     ...     if chunk.delta:
    ///     ...         print(chunk.delta, end="")
    ///
    /// Example (callback):
    ///     >>> def handle_chunk(chunk):
    ///     ...     if chunk.delta:
    ///     ...         print(chunk.delta, end="")
    ///     >>> await model.stream([ChatMessage.user("Hi!")], handle_chunk)
    #[pyo3(signature = (messages, on_chunk=None, options=None))]
    fn stream<'py>(
        &self,
        py: Python<'py>,
        messages: Vec<PyRef<'py, PyChatMessage>>,
        on_chunk: Option<Py<PyAny>>,
        options: Option<PyRef<'py, PyCompletionOptions>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let inner = match self.inner {
            Some(ref inner) => inner.clone(),
            None => {
                // Subclass path -- if we got here, the subclass didn't override stream()
                return Err(pyo3::exceptions::PyNotImplementedError::new_err(
                    "subclass must override stream()",
                ));
            }
        };

        let rust_messages: Vec<ChatMessage> = messages.iter().map(|m| m.inner.clone()).collect();
        let request = build_request(py, rust_messages, options.as_deref())?;

        // Callback mode: return a coroutine that drives the stream and
        // invokes the callback once per chunk.
        //
        // Async iterator mode (`on_chunk` is `None`): return a
        // [`PyLazyCompletionStream`] directly. We intentionally do NOT wrap
        // this in a coroutine because `async for` calls `__aiter__`
        // synchronously on the expression. By deferring stream initialization
        // until the first `__anext__` call, the one-liner form works
        // naturally:
        //
        //     async for chunk in model.stream([ChatMessage.user("Hi!")]):
        //         ...
        if let Some(callback) = on_chunk {
            pyo3_async_runtimes::tokio::future_into_py(py, async move {
                let stream = inner
                    .stream(request)
                    .await
                    .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

                let mut stream = std::pin::pin!(stream);
                while let Some(result) = stream.next().await {
                    match result {
                        Ok(chunk) => {
                            // Call the Python callback
                            tokio::task::block_in_place(|| {
                                Python::attach(|py| {
                                    let py_chunk =
                                        pythonize::pythonize(py, &chunk).map_err(|e| {
                                            pyo3::exceptions::PyRuntimeError::new_err(e.to_string())
                                        })?;
                                    callback.call1(py, (py_chunk,))?;
                                    Ok::<_, PyErr>(())
                                })
                            })?;
                        }
                        Err(e) => {
                            return Err(pyo3::exceptions::PyRuntimeError::new_err(e.to_string()));
                        }
                    }
                }
                Ok(())
            })
        } else {
            let lazy = PyLazyCompletionStream {
                state: Arc::new(Mutex::new(LazyStreamState::NotStarted(Box::new(
                    PendingStream {
                        model: inner,
                        request: Some(request),
                    },
                )))),
            };
            Ok(lazy.into_pyobject(py)?.into_any())
        }
    }

    fn __repr__(&self) -> String {
        format!("CompletionModel(model_id='{}')", self.model_id())
    }

    // -----------------------------------------------------------------
    // Local-model control (only meaningful for in-process providers)
    // -----------------------------------------------------------------

    /// Explicitly load the model weights into memory / VRAM.
    ///
    /// For remote providers (OpenAI, Anthropic, fal, etc.) this raises
    /// ``NotImplementedError`` -- there is no local model to load.
    /// For local providers (mistral.rs, llama.cpp, candle) this triggers
    /// the download + load synchronously, so the next inference call
    /// does not pay the startup cost.
    ///
    /// Idempotent: calling ``load`` on an already-loaded model is a no-op
    /// that returns immediately.
    #[gen_stub(override_return_type(type_repr = "typing.Coroutine[typing.Any, typing.Any, None]", imports = ("typing",)))]
    fn load<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let local = self.local_model.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            match local {
                Some(lm) => lm.load().await.map_err(crate::error::blazen_error_to_pyerr),
                None => Err(pyo3::exceptions::PyNotImplementedError::new_err(
                    "load() is only supported for local in-process providers (mistral.rs, llama.cpp, candle)",
                )),
            }
        })
    }

    /// Drop the loaded model and free its memory / VRAM.
    ///
    /// For remote providers this raises ``NotImplementedError``.
    /// For local providers this frees GPU memory so the process can
    /// load a different model. Idempotent.
    #[gen_stub(override_return_type(type_repr = "typing.Coroutine[typing.Any, typing.Any, None]", imports = ("typing",)))]
    fn unload<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let local = self.local_model.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            match local {
                Some(lm) => lm
                    .unload()
                    .await
                    .map_err(crate::error::blazen_error_to_pyerr),
                None => Err(pyo3::exceptions::PyNotImplementedError::new_err(
                    "unload() is only supported for local in-process providers",
                )),
            }
        })
    }

    /// Whether the model is currently loaded in memory / VRAM.
    ///
    /// Always returns ``False`` for remote providers (they have no local
    /// model to load). Returns the real state for local providers.
    #[gen_stub(override_return_type(type_repr = "typing.Coroutine[typing.Any, typing.Any, builtins.bool]", imports = ("typing", "builtins")))]
    fn is_loaded<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let local = self.local_model.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            match local {
                Some(lm) => Ok(lm.is_loaded().await),
                None => Ok(false),
            }
        })
    }

    /// Approximate VRAM footprint in bytes, if the implementation can
    /// report it. Returns ``None`` for remote providers or for local
    /// providers that do not expose memory usage.
    #[gen_stub(override_return_type(type_repr = "typing.Coroutine[typing.Any, typing.Any, typing.Optional[builtins.int]]", imports = ("typing", "builtins")))]
    fn vram_bytes<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let local = self.local_model.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            match local {
                Some(lm) => Ok(lm.vram_bytes().await),
                None => Ok(None),
            }
        })
    }
}

// ---------------------------------------------------------------------------
// PyLazyCompletionStream -- async iterator over streamed chunks
// ---------------------------------------------------------------------------

/// Pending stream initialization data for [`LazyStreamState::NotStarted`].
///
/// Boxed to keep the enum variant sizes balanced (avoids
/// `clippy::large_enum_variant`).
pub(crate) struct PendingStream {
    pub(crate) model: Arc<dyn CompletionModel>,
    pub(crate) request: Option<CompletionRequest>,
}

/// Internal state for [`PyLazyCompletionStream`].
///
/// The stream is lazily initialized on the first `__anext__` call so that the
/// Python caller can use the natural `async for chunk in model.stream(...)`
/// form without having to `await` the method first.
pub(crate) enum LazyStreamState {
    /// The underlying stream has not yet been requested.
    NotStarted(Box<PendingStream>),
    /// The stream is active and yielding chunks.
    Active(PinnedChunkStream),
    /// The stream has been fully consumed or errored out.
    Exhausted,
}

/// Async iterator over streamed completion chunks.
///
/// Implements the Python `__aiter__` / `__anext__` protocol so it can be used
/// with `async for`. The underlying HTTP stream is lazily initialized on the
/// first `__anext__` call, allowing the natural one-liner form:
///
///     async for chunk in model.stream([ChatMessage.user("Hi!")]):
///         ...
#[gen_stub_pyclass]
#[pyclass(name = "CompletionStream")]
pub struct PyLazyCompletionStream {
    pub(crate) state: Arc<Mutex<LazyStreamState>>,
}

#[gen_stub_pymethods]
#[pymethods]
impl PyLazyCompletionStream {
    fn __aiter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    #[gen_stub(override_return_type(type_repr = "typing.Coroutine[typing.Any, typing.Any, CompletionResponse]", imports = ("typing",)))]
    fn __anext__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let state = self.state.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let mut guard = state.lock().await;

            // Lazily initialize the stream on the first call.
            if let LazyStreamState::NotStarted(pending) = &mut *guard {
                let request = pending
                    .request
                    .take()
                    .ok_or_else(|| BlazenPyError::Llm("stream request already consumed".into()))?;
                match pending.model.stream(request).await {
                    Ok(stream) => {
                        *guard = LazyStreamState::Active(stream);
                    }
                    Err(e) => {
                        *guard = LazyStreamState::Exhausted;
                        return Err(crate::error::blazen_error_to_pyerr(e));
                    }
                }
            }

            match &mut *guard {
                LazyStreamState::Active(stream) => match stream.next().await {
                    Some(Ok(chunk)) => Python::attach(|py| {
                        pythonize::pythonize(py, &chunk)
                            .map(Bound::unbind)
                            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
                    }),
                    Some(Err(e)) => {
                        *guard = LazyStreamState::Exhausted;
                        Err(crate::error::blazen_error_to_pyerr(e))
                    }
                    None => {
                        *guard = LazyStreamState::Exhausted;
                        Err(pyo3::exceptions::PyStopAsyncIteration::new_err(
                            "stream exhausted",
                        ))
                    }
                },
                LazyStreamState::Exhausted => Err(pyo3::exceptions::PyStopAsyncIteration::new_err(
                    "stream exhausted",
                )),
                // Unreachable: we just initialized above.
                LazyStreamState::NotStarted(_) => Err(pyo3::exceptions::PyRuntimeError::new_err(
                    "stream in inconsistent state",
                )),
            }
        })
    }

    fn __repr__(&self) -> String {
        "CompletionStream(...)".to_owned()
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Build a [`CompletionRequest`] from messages and optional [`PyCompletionOptions`].
pub(crate) fn build_request(
    py: Python<'_>,
    messages: Vec<ChatMessage>,
    options: Option<&PyCompletionOptions>,
) -> PyResult<CompletionRequest> {
    let mut request = CompletionRequest::new(messages);

    if let Some(opts) = options {
        if let Some(t) = opts.temperature {
            request = request.with_temperature(t);
        }
        if let Some(mt) = opts.max_tokens {
            request = request.with_max_tokens(mt);
        }
        if let Some(tp) = opts.top_p {
            request = request.with_top_p(tp);
        }
        if let Some(ref m) = opts.model {
            request = request.with_model(m.clone());
        }
        if let Some(ref tools) = opts.tools {
            let rust_tools: Vec<ToolDefinition> = tools
                .iter()
                .map(|t| {
                    let tool = t.borrow(py);
                    ToolDefinition {
                        name: tool.name.clone(),
                        description: tool.description.clone(),
                        parameters: tool.parameters.clone(),
                    }
                })
                .collect();
            request = request.with_tools(rust_tools);
        }
        if let Some(ref fmt) = opts.response_format {
            let schema = crate::convert::py_to_json(py, fmt.bind(py))?;
            request = request.with_response_format(schema);
        }
    }

    Ok(request)
}

// ---------------------------------------------------------------------------
// PySubclassCompletionModel -- Python-subclass adapter
// ---------------------------------------------------------------------------

/// Bridges a Python `CompletionModel` subclass into the Rust
/// [`CompletionModel`] trait. Methods on this adapter dispatch back into
/// the Python object's `complete()` override, so Rust-side helpers
/// (`run_agent`, `with_retry`, `with_cache`, `with_fallback`,
/// `complete_batch`) work uniformly for both built-in providers and
/// Python subclasses.
///
/// The adapter works in three phases for each `complete()` call:
///
/// 1. Under the GIL, translate the Rust [`CompletionRequest`] into
///    Python-side [`PyChatMessage`] / [`PyCompletionOptions`] values,
///    invoke the subclass's `complete()` method to obtain a coroutine,
///    capture the active asyncio task locals, and convert the coroutine
///    into a Rust future.
/// 2. Outside the GIL (inside `pyo3_async_runtimes::tokio::scope`), drive
///    the future to completion so the Python coroutine runs on the
///    correct event loop.
/// 3. Under the GIL again, extract the returned
///    [`PyCompletionResponse`] (preferred path) or fall back to
///    `depythonize` if the subclass returns a compatible dict.
///
/// Streaming through the adapter is not yet supported; `stream()`
/// returns [`BlazenError::Unsupported`]. Callers who want streaming from
/// a subclassed model must call `stream()` directly on the Python object.
pub(crate) struct PySubclassCompletionModel {
    py_obj: Py<PyAny>,
    model_id: String,
}

impl PySubclassCompletionModel {
    pub(crate) fn new(py_obj: Py<PyAny>, model_id: String) -> Self {
        Self { py_obj, model_id }
    }
}

#[async_trait::async_trait]
impl CompletionModel for PySubclassCompletionModel {
    fn model_id(&self) -> &str {
        &self.model_id
    }

    async fn complete(
        &self,
        request: blazen_llm::CompletionRequest,
    ) -> Result<blazen_llm::CompletionResponse, BlazenError> {
        // Phase 1: under GIL, translate the request into Python values,
        // call `complete(messages, options)`, capture asyncio task
        // locals, and convert the returned coroutine into a Rust future.
        let (fut, locals) = tokio::task::block_in_place(|| {
            Python::attach(|py| -> PyResult<_> {
                // Convert messages into `PyChatMessage` instances.
                let messages_py: Vec<Py<PyChatMessage>> = request
                    .messages
                    .iter()
                    .map(|m| Py::new(py, PyChatMessage { inner: m.clone() }))
                    .collect::<PyResult<_>>()?;

                // Build the options dict if any field is set.
                let options_py = build_py_options_from_request(py, &request)?;

                let host = self.py_obj.bind(py);
                let coro = match options_py {
                    Some(opts) => host.call_method1("complete", (messages_py, opts)),
                    None => host.call_method1("complete", (messages_py,)),
                }
                .map_err(|e| {
                    pyo3::exceptions::PyRuntimeError::new_err(format!(
                        "subclass complete() raised before yielding a coroutine: {e}"
                    ))
                })?;

                let locals = pyo3_async_runtimes::tokio::get_current_locals(py)?;
                let fut =
                    pyo3_async_runtimes::into_future_with_locals(&locals, coro).map_err(|e| {
                        pyo3::exceptions::PyTypeError::new_err(format!(
                            "subclass complete() must be an async def returning a coroutine: {e}"
                        ))
                    })?;
                Ok((fut, locals))
            })
        })
        .map_err(|e: PyErr| {
            BlazenError::provider("subclass", format!("dispatch setup failed: {e}"))
        })?;

        // Phase 2: drive the Python coroutine to completion.
        let py_result = pyo3_async_runtimes::tokio::scope(locals, fut)
            .await
            .map_err(|e: PyErr| {
                BlazenError::provider("subclass", format!("subclass complete() raised: {e}"))
            })?;

        // Phase 3: under the GIL, extract a `CompletionResponse` from
        // the Python result. Preferred path: the subclass returned a
        // `PyCompletionResponse`. Fallback: allow a compatible dict.
        tokio::task::block_in_place(|| {
            Python::attach(|py| -> PyResult<blazen_llm::CompletionResponse> {
                let bound = py_result.bind(py);
                if let Ok(resp) = bound.extract::<PyRef<'_, PyCompletionResponse>>() {
                    return Ok(resp.inner.clone());
                }
                let response: blazen_llm::CompletionResponse = pythonize::depythonize(bound)
                    .map_err(|e| {
                        pyo3::exceptions::PyValueError::new_err(format!(
                            "subclass complete() must return CompletionResponse or a compatible dict: {e}"
                        ))
                    })?;
                Ok(response)
            })
        })
        .map_err(|e: PyErr| {
            BlazenError::provider(
                "subclass",
                format!("failed to decode subclass complete() result: {e}"),
            )
        })
    }

    async fn stream(
        &self,
        _request: blazen_llm::CompletionRequest,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<StreamChunk, BlazenError>> + Send>>, BlazenError>
    {
        Err(BlazenError::unsupported(
            "stream() on subclassed CompletionModel is not yet supported from Rust-side callers; call stream() directly on the Python subclass instance",
        ))
    }

    fn provider_config(&self) -> Option<&blazen_llm::ProviderConfig> {
        None
    }
}

/// Public re-export of [`build_py_options_from_request`] for sibling
/// modules that need to reify a Rust `CompletionRequest` back into
/// Python [`PyCompletionOptions`] (e.g. the middleware adapter).
pub(crate) fn build_py_options_from_request_helper(
    py: Python<'_>,
    request: &blazen_llm::CompletionRequest,
) -> PyResult<Option<Py<PyCompletionOptions>>> {
    build_py_options_from_request(py, request)
}

/// Build a [`PyCompletionOptions`] instance from a [`CompletionRequest`].
///
/// Returns `None` if no request field is set that would populate
/// options, so the subclass's `complete()` method can be called with
/// only the `messages` positional argument (matching the Python
/// signature `async def complete(self, messages, options=None)`).
fn build_py_options_from_request(
    py: Python<'_>,
    request: &blazen_llm::CompletionRequest,
) -> PyResult<Option<Py<PyCompletionOptions>>> {
    if request.temperature.is_none()
        && request.max_tokens.is_none()
        && request.top_p.is_none()
        && request.model.is_none()
        && request.tools.is_empty()
        && request.response_format.is_none()
    {
        return Ok(None);
    }

    // Translate tool definitions back into `PyToolDef` instances.
    // Subclasses typically only inspect `name`/`description`/`parameters`
    // on the tool; the handler is not invoked from the subclass path
    // (tool dispatch is driven from the Rust `run_agent` loop), so we
    // pass `py.None()` as a placeholder.
    let tools_py = if request.tools.is_empty() {
        None
    } else {
        let tools: Vec<Py<PyToolDef>> = request
            .tools
            .iter()
            .map(|t| {
                Py::new(
                    py,
                    PyToolDef {
                        name: t.name.clone(),
                        description: t.description.clone(),
                        parameters: t.parameters.clone(),
                        handler: py.None(),
                    },
                )
            })
            .collect::<PyResult<_>>()?;
        Some(tools)
    };

    let response_format = match request.response_format.as_ref() {
        Some(v) => Some(crate::convert::json_to_py(py, v)?),
        None => None,
    };

    let opts = PyCompletionOptions {
        temperature: request.temperature,
        max_tokens: request.max_tokens,
        top_p: request.top_p,
        model: request.model.clone(),
        tools: tools_py,
        response_format,
    };

    Py::new(py, opts).map(Some)
}

/// Build an `Arc<dyn CompletionModel>` from a bound [`PyCompletionModel`].
///
/// If the model has a concrete `inner` (built-in provider), returns that
/// directly. Otherwise constructs a [`PySubclassCompletionModel`] that
/// dispatches back into the Python subclass object.
pub(crate) fn arc_from_bound(bound: &Bound<'_, PyCompletionModel>) -> Arc<dyn CompletionModel> {
    let borrow = bound.borrow();
    if let Some(ref inner) = borrow.inner {
        return inner.clone();
    }
    let model_id = borrow
        .config
        .as_ref()
        .and_then(|c| c.model_id.clone())
        .unwrap_or_else(|| "subclass".to_owned());
    drop(borrow);
    let py_obj: Py<PyAny> = bound.clone().into_any().unbind();
    Arc::new(PySubclassCompletionModel::new(py_obj, model_id))
}

// ---------------------------------------------------------------------------
// Feature-gated mistralrs factory (separate impl block so pyo3-stub-gen
// does not try to resolve the type when the feature is disabled)
// ---------------------------------------------------------------------------

#[cfg(feature = "mistralrs")]
#[gen_stub_pymethods]
#[pymethods]
impl PyCompletionModel {
    /// Create a local mistral.rs provider.
    ///
    /// Runs LLM inference entirely on-device using the mistral.rs engine.
    /// No API key is required.
    ///
    /// Args:
    ///     options: Typed ``MistralRsOptions`` with required ``model_id``
    ///         (HuggingFace model ID or local GGUF path).
    #[staticmethod]
    #[pyo3(signature = (*, options))]
    fn mistralrs(
        options: PyRef<'_, crate::providers::options::PyMistralRsOptions>,
    ) -> PyResult<Self> {
        let opts = options.inner.clone();
        let concrete = Arc::new(
            blazen_llm::MistralRsProvider::from_options(opts)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?,
        );
        Ok(Self {
            inner: Some(concrete.clone()),
            local_model: Some(concrete),
            config: None,
        })
    }
}
