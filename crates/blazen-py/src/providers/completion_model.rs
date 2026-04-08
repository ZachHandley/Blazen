//! Python wrapper for the CompletionModel type with all provider constructors.

use std::pin::Pin;
use std::sync::Arc;
use std::time::Duration;

use pyo3::prelude::*;
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pymethods};
use tokio::sync::Mutex;
use tokio_stream::{Stream, StreamExt};

use blazen_llm::cache::{CacheConfig, CacheStrategy, CachedCompletionModel};
use blazen_llm::fallback::FallbackModel;
use blazen_llm::retry::{RetryCompletionModel, RetryConfig};
use blazen_llm::types::ToolDefinition;
use blazen_llm::{BlazenError, ChatMessage, CompletionModel, CompletionRequest, StreamChunk};

use crate::error::BlazenPyError;
use crate::types::{PyChatMessage, PyCompletionResponse};

/// Type alias for the pinned boxed stream returned by `CompletionModel::stream`.
type PinnedChunkStream = Pin<Box<dyn Stream<Item = Result<StreamChunk, BlazenError>> + Send>>;

/// Deserialize an optional Python options dict into [`ProviderOptions`].
/// Returns the default (all-`None`) options when no dict is provided.
fn depy_provider_options(
    options: Option<&Bound<'_, PyAny>>,
) -> PyResult<blazen_llm::types::provider_options::ProviderOptions> {
    match options {
        Some(o) => Ok(pythonize::depythonize(o)?),
        None => Ok(blazen_llm::types::provider_options::ProviderOptions::default()),
    }
}

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
///     >>> model = CompletionModel.openai("sk-...")
///     >>> model = CompletionModel.anthropic("sk-ant-...")
///     >>> model = CompletionModel.openrouter("sk-or-...").with_model("meta-llama/llama-3-70b")
///     >>>
///     >>> response = await model.complete([
///     ...     ChatMessage.user("What is 2+2?")
///     ... ])
#[gen_stub_pyclass]
#[pyclass(name = "CompletionModel", from_py_object)]
#[derive(Clone)]
pub struct PyCompletionModel {
    pub(crate) inner: Arc<dyn CompletionModel>,
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
    ///     api_key: Your OpenAI API key.
    ///     options: Optional dict with ``model`` and ``baseUrl`` overrides.
    #[staticmethod]
    #[pyo3(signature = (api_key, *, options=None))]
    fn openai(api_key: &str, options: Option<&Bound<'_, PyAny>>) -> PyResult<Self> {
        let opts = depy_provider_options(options)?;
        Ok(Self {
            inner: Arc::new(blazen_llm::providers::openai::OpenAiProvider::from_options(
                api_key, opts,
            )),
        })
    }

    /// Create an Anthropic provider.
    ///
    /// Args:
    ///     api_key: Your Anthropic API key.
    ///     options: Optional dict with ``model`` and ``baseUrl`` overrides.
    #[staticmethod]
    #[pyo3(signature = (api_key, *, options=None))]
    fn anthropic(api_key: &str, options: Option<&Bound<'_, PyAny>>) -> PyResult<Self> {
        let opts = depy_provider_options(options)?;
        Ok(Self {
            inner: Arc::new(
                blazen_llm::providers::anthropic::AnthropicProvider::from_options(api_key, opts),
            ),
        })
    }

    /// Create a Google Gemini provider.
    ///
    /// Args:
    ///     api_key: Your Google API key.
    ///     options: Optional dict with ``model`` and ``baseUrl`` overrides.
    #[staticmethod]
    #[pyo3(signature = (api_key, *, options=None))]
    fn gemini(api_key: &str, options: Option<&Bound<'_, PyAny>>) -> PyResult<Self> {
        let opts = depy_provider_options(options)?;
        Ok(Self {
            inner: Arc::new(blazen_llm::providers::gemini::GeminiProvider::from_options(
                api_key, opts,
            )),
        })
    }

    /// Create an Azure OpenAI provider.
    ///
    /// Args:
    ///     api_key: Your Azure API key.
    ///     options: Dict with required ``resourceName`` and ``deploymentName``,
    ///         plus optional ``apiVersion``.
    #[staticmethod]
    #[pyo3(signature = (api_key, *, options))]
    fn azure(api_key: &str, options: &Bound<'_, PyAny>) -> PyResult<Self> {
        let opts: blazen_llm::types::provider_options::AzureOptions =
            pythonize::depythonize(options)?;
        Ok(Self {
            inner: Arc::new(
                blazen_llm::providers::azure::AzureOpenAiProvider::from_options(api_key, opts),
            ),
        })
    }

    /// Create an OpenRouter provider.
    ///
    /// Args:
    ///     api_key: Your OpenRouter API key.
    ///     options: Optional dict with ``model`` override.
    #[staticmethod]
    #[pyo3(signature = (api_key, *, options=None))]
    fn openrouter(api_key: &str, options: Option<&Bound<'_, PyAny>>) -> PyResult<Self> {
        let opts = depy_provider_options(options)?;
        Ok(Self {
            inner: Arc::new(
                blazen_llm::providers::openrouter::OpenRouterProvider::from_options(api_key, opts),
            ),
        })
    }

    /// Create a Groq provider.
    ///
    /// Args:
    ///     api_key: Your Groq API key.
    ///     options: Optional dict with ``model`` override.
    #[staticmethod]
    #[pyo3(signature = (api_key, *, options=None))]
    fn groq(api_key: &str, options: Option<&Bound<'_, PyAny>>) -> PyResult<Self> {
        let opts = depy_provider_options(options)?;
        Ok(Self {
            inner: Arc::new(blazen_llm::providers::groq::GroqProvider::from_options(
                api_key, opts,
            )),
        })
    }

    /// Create a Together AI provider.
    ///
    /// Args:
    ///     api_key: Your Together API key.
    ///     options: Optional dict with ``model`` override.
    #[staticmethod]
    #[pyo3(signature = (api_key, *, options=None))]
    fn together(api_key: &str, options: Option<&Bound<'_, PyAny>>) -> PyResult<Self> {
        let opts = depy_provider_options(options)?;
        Ok(Self {
            inner: Arc::new(
                blazen_llm::providers::together::TogetherProvider::from_options(api_key, opts),
            ),
        })
    }

    /// Create a Mistral provider.
    ///
    /// Args:
    ///     api_key: Your Mistral API key.
    ///     options: Optional dict with ``model`` override.
    #[staticmethod]
    #[pyo3(signature = (api_key, *, options=None))]
    fn mistral(api_key: &str, options: Option<&Bound<'_, PyAny>>) -> PyResult<Self> {
        let opts = depy_provider_options(options)?;
        Ok(Self {
            inner: Arc::new(
                blazen_llm::providers::mistral::MistralProvider::from_options(api_key, opts),
            ),
        })
    }

    /// Create a DeepSeek provider.
    ///
    /// Args:
    ///     api_key: Your DeepSeek API key.
    ///     options: Optional dict with ``model`` override.
    #[staticmethod]
    #[pyo3(signature = (api_key, *, options=None))]
    fn deepseek(api_key: &str, options: Option<&Bound<'_, PyAny>>) -> PyResult<Self> {
        let opts = depy_provider_options(options)?;
        Ok(Self {
            inner: Arc::new(
                blazen_llm::providers::deepseek::DeepSeekProvider::from_options(api_key, opts),
            ),
        })
    }

    /// Create a Fireworks AI provider.
    ///
    /// Args:
    ///     api_key: Your Fireworks API key.
    ///     options: Optional dict with ``model`` override.
    #[staticmethod]
    #[pyo3(signature = (api_key, *, options=None))]
    fn fireworks(api_key: &str, options: Option<&Bound<'_, PyAny>>) -> PyResult<Self> {
        let opts = depy_provider_options(options)?;
        Ok(Self {
            inner: Arc::new(
                blazen_llm::providers::fireworks::FireworksProvider::from_options(api_key, opts),
            ),
        })
    }

    /// Create a Perplexity provider.
    ///
    /// Args:
    ///     api_key: Your Perplexity API key.
    ///     options: Optional dict with ``model`` override.
    #[staticmethod]
    #[pyo3(signature = (api_key, *, options=None))]
    fn perplexity(api_key: &str, options: Option<&Bound<'_, PyAny>>) -> PyResult<Self> {
        let opts = depy_provider_options(options)?;
        Ok(Self {
            inner: Arc::new(
                blazen_llm::providers::perplexity::PerplexityProvider::from_options(api_key, opts),
            ),
        })
    }

    /// Create an xAI (Grok) provider.
    ///
    /// Args:
    ///     api_key: Your xAI API key.
    ///     options: Optional dict with ``model`` override.
    #[staticmethod]
    #[pyo3(signature = (api_key, *, options=None))]
    fn xai(api_key: &str, options: Option<&Bound<'_, PyAny>>) -> PyResult<Self> {
        let opts = depy_provider_options(options)?;
        Ok(Self {
            inner: Arc::new(blazen_llm::providers::xai::XaiProvider::from_options(
                api_key, opts,
            )),
        })
    }

    /// Create a Cohere provider.
    ///
    /// Args:
    ///     api_key: Your Cohere API key.
    ///     options: Optional dict with ``model`` override.
    #[staticmethod]
    #[pyo3(signature = (api_key, *, options=None))]
    fn cohere(api_key: &str, options: Option<&Bound<'_, PyAny>>) -> PyResult<Self> {
        let opts = depy_provider_options(options)?;
        Ok(Self {
            inner: Arc::new(blazen_llm::providers::cohere::CohereProvider::from_options(
                api_key, opts,
            )),
        })
    }

    /// Create an AWS Bedrock provider.
    ///
    /// Args:
    ///     api_key: Your Bedrock API key.
    ///     options: Dict with required ``region`` and optional ``model``.
    #[staticmethod]
    #[pyo3(signature = (api_key, *, options))]
    fn bedrock(api_key: &str, options: &Bound<'_, PyAny>) -> PyResult<Self> {
        let opts: blazen_llm::types::provider_options::BedrockOptions =
            pythonize::depythonize(options)?;
        Ok(Self {
            inner: Arc::new(
                blazen_llm::providers::bedrock::BedrockProvider::from_options(api_key, opts),
            ),
        })
    }

    /// Create a fal.ai provider.
    ///
    /// Args:
    ///     api_key: Your fal.ai API key.
    ///     options: Optional dict for selecting the model,
    ///         endpoint, enterprise tier, and auto-routing. Defaults to
    ///         the OpenAI-chat endpoint
    ///         (``openrouter/router/openai/v1/chat/completions``).
    #[staticmethod]
    #[pyo3(signature = (api_key, *, options=None))]
    fn fal(api_key: &str, options: Option<&Bound<'_, PyAny>>) -> PyResult<Self> {
        let opts: blazen_llm::types::provider_options::FalOptions = match options {
            Some(o) => pythonize::depythonize(o)?,
            None => blazen_llm::types::provider_options::FalOptions::default(),
        };
        Ok(Self {
            inner: Arc::new(blazen_llm::providers::fal::FalProvider::from_options(
                api_key, opts,
            )),
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
    ///     max_retries: Maximum retry attempts (default: 3).
    ///     initial_delay_ms: Delay before first retry in ms (default: 1000).
    ///     max_delay_ms: Upper bound on backoff delay in ms (default: 30000).
    ///
    /// Returns:
    ///     A new CompletionModel with retry behaviour.
    ///
    /// Example:
    ///     >>> model = CompletionModel.openai("sk-...").with_retry(max_retries=5)
    #[pyo3(signature = (*, max_retries=None, initial_delay_ms=None, max_delay_ms=None))]
    fn with_retry(
        &self,
        max_retries: Option<u32>,
        initial_delay_ms: Option<u64>,
        max_delay_ms: Option<u64>,
    ) -> Self {
        let config = RetryConfig {
            max_retries: max_retries.unwrap_or(3),
            initial_delay: Duration::from_millis(initial_delay_ms.unwrap_or(1000)),
            max_delay: Duration::from_millis(max_delay_ms.unwrap_or(30_000)),
            honor_retry_after: true,
            jitter: true,
        };
        let model = RetryCompletionModel::from_arc(self.inner.clone(), config);
        Self {
            inner: Arc::new(model),
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
    ///     >>> primary = CompletionModel.openai("sk-...")
    ///     >>> backup = CompletionModel.anthropic("sk-ant-...")
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
        Ok(Self {
            inner: Arc::new(model),
        })
    }

    /// Wrap this model with response caching.
    ///
    /// Repeated identical requests are served from an in-memory cache
    /// without hitting the underlying provider. Streaming requests are
    /// never cached.
    ///
    /// Args:
    ///     ttl_seconds: Cache entry time-to-live in seconds (default: 300).
    ///     max_entries: Maximum cache entries before eviction (default: 1000).
    ///
    /// Returns:
    ///     A new CompletionModel with caching enabled.
    ///
    /// Example:
    ///     >>> model = CompletionModel.openai("sk-...").with_cache(ttl_seconds=600)
    #[pyo3(signature = (*, ttl_seconds=None, max_entries=None))]
    fn with_cache(&self, ttl_seconds: Option<u64>, max_entries: Option<usize>) -> Self {
        let config = CacheConfig {
            strategy: CacheStrategy::ContentHash,
            ttl: Duration::from_secs(ttl_seconds.unwrap_or(300)),
            max_entries: max_entries.unwrap_or(1000),
        };
        let model = CachedCompletionModel::from_arc(self.inner.clone(), config);
        Self {
            inner: Arc::new(model),
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
fn build_request(
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
