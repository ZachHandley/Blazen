//! Python wrapper for the CompletionModel type with all provider constructors.

use std::sync::Arc;
use std::time::Duration;

use pyo3::prelude::*;
use tokio_stream::StreamExt;

use blazen_llm::cache::{CacheConfig, CacheStrategy, CachedCompletionModel};
use blazen_llm::fallback::FallbackModel;
use blazen_llm::retry::{RetryCompletionModel, RetryConfig};
use blazen_llm::types::ToolDefinition;
use blazen_llm::{ChatMessage, CompletionModel, CompletionRequest};

use crate::error::BlazenPyError;
use crate::types::{PyChatMessage, PyCompletionResponse, PyStreamChunk};

// ---------------------------------------------------------------------------
// PyCompletionOptions
// ---------------------------------------------------------------------------

/// Options for a chat completion request.
///
/// Example:
///     >>> opts = CompletionOptions(temperature=0.7, max_tokens=1000)
///     >>> response = await model.complete(messages, opts)
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
#[pyclass(name = "CompletionModel", from_py_object)]
#[derive(Clone)]
pub struct PyCompletionModel {
    pub(crate) inner: Arc<dyn CompletionModel>,
}

#[pymethods]
impl PyCompletionModel {
    // -----------------------------------------------------------------
    // Provider constructors
    // -----------------------------------------------------------------

    /// Create an OpenAI provider.
    ///
    /// Args:
    ///     api_key: Your OpenAI API key.
    ///     model: Optional model name (default: "gpt-4o").
    #[staticmethod]
    #[pyo3(signature = (api_key, model=None))]
    fn openai(api_key: &str, model: Option<&str>) -> Self {
        let mut provider = blazen_llm::providers::openai::OpenAiProvider::new(api_key);
        if let Some(m) = model {
            provider = provider.with_model(m);
        }
        Self {
            inner: Arc::new(provider),
        }
    }

    /// Create an Anthropic provider.
    ///
    /// Args:
    ///     api_key: Your Anthropic API key.
    ///     model: Optional model name (default: "claude-sonnet-4-20250514").
    #[staticmethod]
    #[pyo3(signature = (api_key, model=None))]
    fn anthropic(api_key: &str, model: Option<&str>) -> Self {
        let mut provider = blazen_llm::providers::anthropic::AnthropicProvider::new(api_key);
        if let Some(m) = model {
            provider = provider.with_model(m);
        }
        Self {
            inner: Arc::new(provider),
        }
    }

    /// Create a Google Gemini provider.
    ///
    /// Args:
    ///     api_key: Your Google API key.
    ///     model: Optional model name (default: "gemini-2.0-flash").
    #[staticmethod]
    #[pyo3(signature = (api_key, model=None))]
    fn gemini(api_key: &str, model: Option<&str>) -> Self {
        let mut provider = blazen_llm::providers::gemini::GeminiProvider::new(api_key);
        if let Some(m) = model {
            provider = provider.with_model(m);
        }
        Self {
            inner: Arc::new(provider),
        }
    }

    /// Create an Azure OpenAI provider.
    ///
    /// Args:
    ///     api_key: Your Azure API key.
    ///     resource_name: The Azure resource name (subdomain).
    ///     deployment_name: The model deployment name.
    #[staticmethod]
    fn azure(api_key: &str, resource_name: &str, deployment_name: &str) -> Self {
        let provider = blazen_llm::providers::azure::AzureOpenAiProvider::new(
            api_key,
            resource_name,
            deployment_name,
        );
        Self {
            inner: Arc::new(provider),
        }
    }

    /// Create an OpenRouter provider.
    ///
    /// Args:
    ///     api_key: Your OpenRouter API key.
    ///     model: Optional model name.
    #[staticmethod]
    #[pyo3(signature = (api_key, model=None))]
    fn openrouter(api_key: &str, model: Option<&str>) -> Self {
        let mut provider = blazen_llm::providers::openrouter::OpenRouterProvider::new(api_key);
        if let Some(m) = model {
            provider = provider.with_model(m);
        }
        Self {
            inner: Arc::new(provider),
        }
    }

    /// Create a Groq provider.
    ///
    /// Args:
    ///     api_key: Your Groq API key.
    ///     model: Optional model name.
    #[staticmethod]
    #[pyo3(signature = (api_key, model=None))]
    fn groq(api_key: &str, model: Option<&str>) -> Self {
        let mut provider = blazen_llm::providers::groq::GroqProvider::new(api_key);
        if let Some(m) = model {
            provider = provider.with_model(m);
        }
        Self {
            inner: Arc::new(provider),
        }
    }

    /// Create a Together AI provider.
    ///
    /// Args:
    ///     api_key: Your Together API key.
    ///     model: Optional model name.
    #[staticmethod]
    #[pyo3(signature = (api_key, model=None))]
    fn together(api_key: &str, model: Option<&str>) -> Self {
        let mut provider = blazen_llm::providers::together::TogetherProvider::new(api_key);
        if let Some(m) = model {
            provider = provider.with_model(m);
        }
        Self {
            inner: Arc::new(provider),
        }
    }

    /// Create a Mistral provider.
    ///
    /// Args:
    ///     api_key: Your Mistral API key.
    ///     model: Optional model name.
    #[staticmethod]
    #[pyo3(signature = (api_key, model=None))]
    fn mistral(api_key: &str, model: Option<&str>) -> Self {
        let mut provider = blazen_llm::providers::mistral::MistralProvider::new(api_key);
        if let Some(m) = model {
            provider = provider.with_model(m);
        }
        Self {
            inner: Arc::new(provider),
        }
    }

    /// Create a DeepSeek provider.
    ///
    /// Args:
    ///     api_key: Your DeepSeek API key.
    ///     model: Optional model name.
    #[staticmethod]
    #[pyo3(signature = (api_key, model=None))]
    fn deepseek(api_key: &str, model: Option<&str>) -> Self {
        let mut provider = blazen_llm::providers::deepseek::DeepSeekProvider::new(api_key);
        if let Some(m) = model {
            provider = provider.with_model(m);
        }
        Self {
            inner: Arc::new(provider),
        }
    }

    /// Create a Fireworks AI provider.
    ///
    /// Args:
    ///     api_key: Your Fireworks API key.
    ///     model: Optional model name.
    #[staticmethod]
    #[pyo3(signature = (api_key, model=None))]
    fn fireworks(api_key: &str, model: Option<&str>) -> Self {
        let mut provider = blazen_llm::providers::fireworks::FireworksProvider::new(api_key);
        if let Some(m) = model {
            provider = provider.with_model(m);
        }
        Self {
            inner: Arc::new(provider),
        }
    }

    /// Create a Perplexity provider.
    ///
    /// Args:
    ///     api_key: Your Perplexity API key.
    ///     model: Optional model name.
    #[staticmethod]
    #[pyo3(signature = (api_key, model=None))]
    fn perplexity(api_key: &str, model: Option<&str>) -> Self {
        let mut provider = blazen_llm::providers::perplexity::PerplexityProvider::new(api_key);
        if let Some(m) = model {
            provider = provider.with_model(m);
        }
        Self {
            inner: Arc::new(provider),
        }
    }

    /// Create an xAI (Grok) provider.
    ///
    /// Args:
    ///     api_key: Your xAI API key.
    ///     model: Optional model name.
    #[staticmethod]
    #[pyo3(signature = (api_key, model=None))]
    fn xai(api_key: &str, model: Option<&str>) -> Self {
        let mut provider = blazen_llm::providers::xai::XaiProvider::new(api_key);
        if let Some(m) = model {
            provider = provider.with_model(m);
        }
        Self {
            inner: Arc::new(provider),
        }
    }

    /// Create a Cohere provider.
    ///
    /// Args:
    ///     api_key: Your Cohere API key.
    ///     model: Optional model name.
    #[staticmethod]
    #[pyo3(signature = (api_key, model=None))]
    fn cohere(api_key: &str, model: Option<&str>) -> Self {
        let mut provider = blazen_llm::providers::cohere::CohereProvider::new(api_key);
        if let Some(m) = model {
            provider = provider.with_model(m);
        }
        Self {
            inner: Arc::new(provider),
        }
    }

    /// Create an AWS Bedrock provider.
    ///
    /// Args:
    ///     api_key: Your Bedrock API key.
    ///     region: The AWS region.
    ///     model: Optional model name.
    #[staticmethod]
    #[pyo3(signature = (api_key, region, model=None))]
    fn bedrock(api_key: &str, region: &str, model: Option<&str>) -> Self {
        let mut provider = blazen_llm::providers::bedrock::BedrockProvider::new(api_key, region);
        if let Some(m) = model {
            provider = provider.with_model(m);
        }
        Self {
            inner: Arc::new(provider),
        }
    }

    /// Create a fal.ai provider.
    ///
    /// Args:
    ///     api_key: Your fal.ai API key.
    ///     model: Optional LLM model name (e.g. "anthropic/claude-sonnet-4.5").
    #[staticmethod]
    #[pyo3(signature = (api_key, model=None))]
    fn fal(api_key: &str, model: Option<&str>) -> Self {
        let mut provider = blazen_llm::providers::fal::FalProvider::new(api_key);
        if let Some(m) = model {
            provider = provider.with_llm_model(m);
        }
        Self {
            inner: Arc::new(provider),
        }
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

    /// Stream a chat completion, calling a callback for each chunk.
    ///
    /// Args:
    ///     messages: A list of ChatMessage objects.
    ///     on_chunk: Callback function receiving each chunk as a dict with
    ///         keys: ``delta``, ``finish_reason``, ``tool_calls``.
    ///     temperature: Optional sampling temperature (0.0-2.0).
    ///     max_tokens: Optional maximum tokens to generate.
    ///     top_p: Optional nucleus sampling parameter (0.0-1.0).
    ///     model: Optional model override for this request.
    ///     tools: Optional list of dicts with ``name``, ``description``, and
    ///         ``parameters`` keys for function calling.
    ///     response_format: Optional JSON schema dict for structured output.
    ///
    /// Example:
    ///     >>> async def handle_chunk(chunk):
    ///     ...     if chunk["delta"]:
    ///     ...         print(chunk["delta"], end="")
    ///     >>> await model.stream([ChatMessage.user("Hi!")], handle_chunk)
    #[pyo3(signature = (messages, on_chunk, options=None))]
    fn stream<'py>(
        &self,
        py: Python<'py>,
        messages: Vec<PyRef<'py, PyChatMessage>>,
        on_chunk: Py<PyAny>,
        options: Option<PyRef<'py, PyCompletionOptions>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let rust_messages: Vec<ChatMessage> = messages.iter().map(|m| m.inner.clone()).collect();
        let request = build_request(py, rust_messages, options.as_deref())?;

        let inner = self.inner.clone();

        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let stream = inner
                .stream(request)
                .await
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

            let mut stream = std::pin::pin!(stream);
            while let Some(result) = stream.next().await {
                match result {
                    Ok(chunk) => {
                        let py_chunk = PyStreamChunk { inner: chunk };

                        // Call the Python callback
                        tokio::task::block_in_place(|| {
                            Python::attach(|py| {
                                on_chunk.call1(py, (py_chunk,))?;
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
    }

    fn __repr__(&self) -> String {
        format!("CompletionModel(model_id='{}')", self.inner.model_id())
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
            let schema = crate::workflow::event::py_to_json(py, fmt.bind(py))?;
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
        let parameters = crate::workflow::event::py_to_json(py, &tool.get_item("parameters")?)?;
        rust_tools.push(ToolDefinition {
            name,
            description,
            parameters,
        });
    }
    Ok(rust_tools)
}
