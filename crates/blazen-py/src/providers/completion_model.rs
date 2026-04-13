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
    /// Tool definitions for function calling. Each tool is a dict with
    /// ``name``, ``description``, and ``parameters`` keys.
    #[pyo3(get, set)]
    pub tools: Option<Py<PyAny>>,
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
        tools: Option<Py<PyAny>>,
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
#[pyclass(name = "CompletionModel", from_py_object)]
#[derive(Clone)]
pub struct PyCompletionModel {
    pub(crate) inner: Arc<dyn CompletionModel>,
    /// Present iff the underlying provider is a local in-process model
    /// (mistral.rs, llama.cpp, candle) that implements
    /// [`blazen_llm::LocalModel`]. `None` for remote HTTP providers.
    /// Populated by the provider factory methods.
    pub(crate) local_model: Option<Arc<dyn blazen_llm::LocalModel>>,
}

#[gen_stub_pymethods]
#[pymethods]
impl PyCompletionModel {
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
            inner: Arc::new(
                blazen_llm::providers::openai::OpenAiProvider::from_options(opts)
                    .map_err(crate::error::blazen_error_to_pyerr)?,
            ),
            local_model: None,
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
            inner: Arc::new(
                blazen_llm::providers::anthropic::AnthropicProvider::from_options(opts)
                    .map_err(crate::error::blazen_error_to_pyerr)?,
            ),
            local_model: None,
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
            inner: Arc::new(
                blazen_llm::providers::gemini::GeminiProvider::from_options(opts)
                    .map_err(crate::error::blazen_error_to_pyerr)?,
            ),
            local_model: None,
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
            inner: Arc::new(
                blazen_llm::providers::azure::AzureOpenAiProvider::from_options(opts)
                    .map_err(crate::error::blazen_error_to_pyerr)?,
            ),
            local_model: None,
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
            inner: Arc::new(
                blazen_llm::providers::openrouter::OpenRouterProvider::from_options(opts)
                    .map_err(crate::error::blazen_error_to_pyerr)?,
            ),
            local_model: None,
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
            inner: Arc::new(
                blazen_llm::providers::groq::GroqProvider::from_options(opts)
                    .map_err(crate::error::blazen_error_to_pyerr)?,
            ),
            local_model: None,
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
            inner: Arc::new(
                blazen_llm::providers::together::TogetherProvider::from_options(opts)
                    .map_err(crate::error::blazen_error_to_pyerr)?,
            ),
            local_model: None,
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
            inner: Arc::new(
                blazen_llm::providers::mistral::MistralProvider::from_options(opts)
                    .map_err(crate::error::blazen_error_to_pyerr)?,
            ),
            local_model: None,
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
            inner: Arc::new(
                blazen_llm::providers::deepseek::DeepSeekProvider::from_options(opts)
                    .map_err(crate::error::blazen_error_to_pyerr)?,
            ),
            local_model: None,
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
            inner: Arc::new(
                blazen_llm::providers::fireworks::FireworksProvider::from_options(opts)
                    .map_err(crate::error::blazen_error_to_pyerr)?,
            ),
            local_model: None,
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
            inner: Arc::new(
                blazen_llm::providers::perplexity::PerplexityProvider::from_options(opts)
                    .map_err(crate::error::blazen_error_to_pyerr)?,
            ),
            local_model: None,
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
            inner: Arc::new(
                blazen_llm::providers::xai::XaiProvider::from_options(opts)
                    .map_err(crate::error::blazen_error_to_pyerr)?,
            ),
            local_model: None,
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
            inner: Arc::new(
                blazen_llm::providers::cohere::CohereProvider::from_options(opts)
                    .map_err(crate::error::blazen_error_to_pyerr)?,
            ),
            local_model: None,
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
            inner: Arc::new(
                blazen_llm::providers::bedrock::BedrockProvider::from_options(opts)
                    .map_err(crate::error::blazen_error_to_pyerr)?,
            ),
            local_model: None,
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
            inner: Arc::new(
                blazen_llm::providers::fal::FalProvider::from_options(opts)
                    .map_err(crate::error::blazen_error_to_pyerr)?,
            ),
            local_model: None,
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
    fn model_id(&self) -> &str {
        self.inner.model_id()
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
    fn with_retry(&self, config: Option<PyRef<'_, PyRetryConfig>>) -> Self {
        let retry_config = config.map(|c| c.inner.clone()).unwrap_or_default();
        let model = RetryCompletionModel::from_arc(self.inner.clone(), retry_config);
        Self {
            inner: Arc::new(model),
            local_model: self.local_model.clone(),
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
    fn with_fallback(models: Vec<PyRef<'_, PyCompletionModel>>) -> PyResult<Self> {
        if models.is_empty() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "with_fallback requires at least one model",
            ));
        }
        let providers: Vec<Arc<dyn CompletionModel>> =
            models.iter().map(|m| m.inner.clone()).collect();
        let model = FallbackModel::new(providers);
        // A fallback chain is a composition of heterogeneous providers, so
        // there is no single `LocalModel` to forward load/unload to. Callers
        // that need local-model control should apply it to the individual
        // component models before combining them via `with_fallback`.
        Ok(Self {
            inner: Arc::new(model),
            local_model: None,
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
    fn with_cache(&self, config: Option<PyRef<'_, PyCacheConfig>>) -> Self {
        let cache_config = config.map(|c| c.inner.clone()).unwrap_or_default();
        let model = CachedCompletionModel::from_arc(self.inner.clone(), cache_config);
        Self {
            inner: Arc::new(model),
            local_model: self.local_model.clone(),
        }
    }

    // -----------------------------------------------------------------
    // Completion
    // -----------------------------------------------------------------

    /// Perform a chat completion.
    ///
    /// Args:
    ///     messages: A list of ChatMessage objects.
    ///     temperature: Optional sampling temperature (0.0-2.0).
    ///     max_tokens: Optional maximum tokens to generate.
    ///     top_p: Optional nucleus sampling parameter (0.0-1.0).
    ///     model: Optional model override for this request.
    ///     tools: Optional list of dicts with ``name``, ``description``, and
    ///         ``parameters`` keys for function calling.
    ///     response_format: Optional JSON schema dict for structured output.
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
        let rust_messages: Vec<ChatMessage> = messages.iter().map(|m| m.inner.clone()).collect();
        let request = build_request(py, rust_messages, options.as_deref())?;
        let inner = self.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let response = inner.complete(request).await.map_err(BlazenPyError::from)?;
            Ok(PyCompletionResponse { inner: response })
        })
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
        let rust_messages: Vec<ChatMessage> = messages.iter().map(|m| m.inner.clone()).collect();
        let request = build_request(py, rust_messages, options.as_deref())?;

        let inner = self.inner.clone();

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
        format!("CompletionModel(model_id='{}')", self.inner.model_id())
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
struct PendingStream {
    model: Arc<dyn CompletionModel>,
    request: Option<CompletionRequest>,
}

/// Internal state for [`PyLazyCompletionStream`].
///
/// The stream is lazily initialized on the first `__anext__` call so that the
/// Python caller can use the natural `async for chunk in model.stream(...)`
/// form without having to `await` the method first.
enum LazyStreamState {
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
    state: Arc<Mutex<LazyStreamState>>,
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
        if let Some(ref tools_py) = opts.tools {
            let tools_bound = tools_py.bind(py);
            let tools_list: &Bound<'_, pyo3::types::PyList> = tools_bound.cast()?;
            let tool_vec: Vec<Bound<'_, PyAny>> = tools_list.iter().collect();
            let rust_tools = extract_tool_definitions(py, &tool_vec)?;
            request = request.with_tools(rust_tools);
        }
        if let Some(ref fmt) = opts.response_format {
            let schema = crate::convert::py_to_json(py, fmt.bind(py))?;
            request = request.with_response_format(schema);
        }
    }

    Ok(request)
}

/// Extract a list of [`ToolDefinition`] from Python dicts (or dict-like objects).
fn extract_tool_definitions(
    py: Python<'_>,
    tool_list: &[Bound<'_, PyAny>],
) -> PyResult<Vec<ToolDefinition>> {
    let mut rust_tools = Vec::with_capacity(tool_list.len());
    for tool in tool_list {
        let name: String = tool.get_item("name")?.extract()?;
        let description: String = tool.get_item("description")?.extract()?;
        let parameters = crate::convert::py_to_json(py, &tool.get_item("parameters")?)?;
        rust_tools.push(ToolDefinition {
            name,
            description,
            parameters,
        });
    }
    Ok(rust_tools)
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
            inner: concrete.clone(),
            local_model: Some(concrete),
        })
    }
}
