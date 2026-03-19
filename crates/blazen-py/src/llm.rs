//! Python wrappers for LLM provider types.
//!
//! Exposes [`CompletionModel`](blazen_llm::CompletionModel) implementations
//! to Python with static constructor methods for each provider.

use std::sync::Arc;

use pyo3::prelude::*;
use tokio_stream::StreamExt;

use blazen_llm::{
    ChatMessage, CompletionModel, CompletionRequest, CompletionResponse, ContentPart, ImageContent,
    ImageSource, MessageContent, RequestTiming, Role, TokenUsage, ToolCall,
};

use crate::error::BlazenPyError;

// ---------------------------------------------------------------------------
// PyRole
// ---------------------------------------------------------------------------

/// Role constants for chat messages.
///
/// Example:
///     >>> ChatMessage(role=Role.USER, content="Hello!")
///     >>> ChatMessage(role=Role.SYSTEM, content="You are helpful.")
#[pyclass(name = "Role", frozen)]
pub struct PyRole;

#[pymethods]
impl PyRole {
    #[classattr]
    const SYSTEM: &'static str = "system";
    #[classattr]
    const USER: &'static str = "user";
    #[classattr]
    const ASSISTANT: &'static str = "assistant";
    #[classattr]
    const TOOL: &'static str = "tool";
}

// ---------------------------------------------------------------------------
// PyContentPart
// ---------------------------------------------------------------------------

#[pyclass(name = "ContentPart", from_py_object)]
#[derive(Clone)]
pub struct PyContentPart {
    pub(crate) inner: ContentPart,
}

#[pymethods]
impl PyContentPart {
    /// Create a text content part.
    #[staticmethod]
    #[pyo3(signature = (*, text))]
    fn text(text: &str) -> Self {
        Self {
            inner: ContentPart::Text {
                text: text.to_owned(),
            },
        }
    }

    /// Create an image content part from a URL.
    #[staticmethod]
    #[pyo3(signature = (*, url, media_type=None))]
    fn image_url(url: &str, media_type: Option<&str>) -> Self {
        Self {
            inner: ContentPart::Image(ImageContent {
                source: ImageSource::Url {
                    url: url.to_owned(),
                },
                media_type: media_type.map(String::from),
            }),
        }
    }

    /// Create an image content part from base64 data.
    #[staticmethod]
    #[pyo3(signature = (*, data, media_type))]
    fn image_base64(data: &str, media_type: &str) -> Self {
        Self {
            inner: ContentPart::Image(ImageContent {
                source: ImageSource::Base64 {
                    data: data.to_owned(),
                },
                media_type: Some(media_type.to_owned()),
            }),
        }
    }

    fn __repr__(&self) -> String {
        match &self.inner {
            ContentPart::Text { text } => format!("ContentPart.text(text='{text}')"),
            ContentPart::Image(_) => "ContentPart(image)".to_owned(),
            ContentPart::File(_) => "ContentPart(file)".to_owned(),
        }
    }
}

// ---------------------------------------------------------------------------
// PyChatMessage
// ---------------------------------------------------------------------------

/// A single message in a chat conversation.
///
/// Example:
///     >>> msg = ChatMessage(content="Hello, world!")
///     >>> msg = ChatMessage(role="system", content="You are a helpful assistant.")
///     >>> msg = ChatMessage.user("What is 2+2?")
#[pyclass(name = "ChatMessage", from_py_object)]
#[derive(Clone)]
pub struct PyChatMessage {
    pub(crate) inner: ChatMessage,
}

#[pymethods]
impl PyChatMessage {
    /// Create a new chat message.
    ///
    /// Args:
    ///     role: One of "system", "user", "assistant", "tool" (default: "user").
    ///     content: The message text.
    ///     parts: Optional list of ContentPart objects for multimodal messages.
    #[new]
    #[pyo3(signature = (role="user", content=None, parts=None))]
    fn new(
        role: &str,
        content: Option<&str>,
        parts: Option<Vec<PyRef<'_, PyContentPart>>>,
    ) -> PyResult<Self> {
        let role = match role {
            "system" => Role::System,
            "user" => Role::User,
            "assistant" => Role::Assistant,
            "tool" => Role::Tool,
            other => {
                return Err(BlazenPyError::InvalidArgument(format!(
                    "unknown role: '{other}' (expected system, user, assistant, or tool)"
                ))
                .into());
            }
        };

        let message_content = if let Some(parts) = parts {
            MessageContent::Parts(parts.iter().map(|p| p.inner.clone()).collect())
        } else {
            MessageContent::Text(content.unwrap_or("").to_owned())
        };

        Ok(Self {
            inner: ChatMessage {
                role,
                content: message_content,
            },
        })
    }

    /// Create a system message.
    #[staticmethod]
    fn system(content: &str) -> Self {
        Self {
            inner: ChatMessage::system(content),
        }
    }

    /// Create a user message.
    #[staticmethod]
    fn user(content: &str) -> Self {
        Self {
            inner: ChatMessage::user(content),
        }
    }

    /// Create an assistant message.
    #[staticmethod]
    fn assistant(content: &str) -> Self {
        Self {
            inner: ChatMessage::assistant(content),
        }
    }

    /// Create a tool result message.
    #[staticmethod]
    fn tool(content: &str) -> Self {
        Self {
            inner: ChatMessage::tool(content),
        }
    }

    /// Create a user message with text and an image URL.
    #[staticmethod]
    #[pyo3(signature = (*, text, url, media_type=None))]
    fn user_image_url(text: &str, url: &str, media_type: Option<&str>) -> Self {
        Self {
            inner: ChatMessage::user_image_url(text, url, media_type),
        }
    }

    /// Create a user message with text and a base64-encoded image.
    #[staticmethod]
    #[pyo3(signature = (*, text, data, media_type))]
    fn user_image_base64(text: &str, data: &str, media_type: &str) -> Self {
        Self {
            inner: ChatMessage::user_image_base64(text, data, media_type),
        }
    }

    /// Create a user message from a list of ContentPart objects.
    #[staticmethod]
    #[pyo3(signature = (*, parts))]
    fn user_parts(parts: Vec<PyRef<'_, PyContentPart>>) -> Self {
        let rust_parts: Vec<ContentPart> = parts.iter().map(|p| p.inner.clone()).collect();
        Self {
            inner: ChatMessage::user_parts(rust_parts),
        }
    }

    /// Get the role as a string.
    #[getter]
    fn role(&self) -> &str {
        match self.inner.role {
            Role::System => "system",
            Role::User => "user",
            Role::Assistant => "assistant",
            Role::Tool => "tool",
        }
    }

    /// Get the message content as a string.
    #[getter]
    fn content(&self) -> Option<&str> {
        self.inner.content.as_text()
    }

    fn __repr__(&self) -> String {
        format!(
            "ChatMessage(role='{}', content='{}')",
            self.role(),
            self.content().unwrap_or("")
        )
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
        let mut provider =
            blazen_llm::providers::openai_compat::OpenAiCompatProvider::openrouter(api_key);
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
        let mut provider =
            blazen_llm::providers::openai_compat::OpenAiCompatProvider::groq(api_key);
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
        let mut provider =
            blazen_llm::providers::openai_compat::OpenAiCompatProvider::together(api_key);
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
        let mut provider =
            blazen_llm::providers::openai_compat::OpenAiCompatProvider::mistral(api_key);
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
        let mut provider =
            blazen_llm::providers::openai_compat::OpenAiCompatProvider::deepseek(api_key);
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
        let mut provider =
            blazen_llm::providers::openai_compat::OpenAiCompatProvider::fireworks(api_key);
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
        let mut provider =
            blazen_llm::providers::openai_compat::OpenAiCompatProvider::perplexity(api_key);
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
        let mut provider = blazen_llm::providers::openai_compat::OpenAiCompatProvider::xai(api_key);
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
        let mut provider =
            blazen_llm::providers::openai_compat::OpenAiCompatProvider::cohere(api_key);
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
        let mut provider =
            blazen_llm::providers::openai_compat::OpenAiCompatProvider::bedrock(api_key, region);
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
    ///     model: Optional model name.
    #[staticmethod]
    #[pyo3(signature = (api_key, model=None))]
    fn fal(api_key: &str, model: Option<&str>) -> Self {
        let mut provider = blazen_llm::providers::fal::FalProvider::new(api_key);
        if let Some(m) = model {
            provider = provider.with_model(m);
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
    // Completion
    // -----------------------------------------------------------------

    /// Perform a chat completion.
    ///
    /// Args:
    ///     messages: A list of ChatMessage objects.
    ///     temperature: Optional sampling temperature (0.0-2.0).
    ///     max_tokens: Optional maximum tokens to generate.
    ///     model: Optional model override for this request.
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
    #[pyo3(signature = (messages, temperature=None, max_tokens=None, model=None, response_format=None))]
    fn complete<'py>(
        &self,
        py: Python<'py>,
        messages: Vec<PyRef<'py, PyChatMessage>>,
        temperature: Option<f32>,
        max_tokens: Option<u32>,
        model: Option<String>,
        response_format: Option<&Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let rust_messages: Vec<ChatMessage> = messages.iter().map(|m| m.inner.clone()).collect();

        let mut request = CompletionRequest::new(rust_messages);
        if let Some(t) = temperature {
            request = request.with_temperature(t);
        }
        if let Some(mt) = max_tokens {
            request = request.with_max_tokens(mt);
        }
        if let Some(m) = model {
            request = request.with_model(m);
        }
        if let Some(fmt) = response_format {
            let schema = crate::event::py_to_json(py, fmt)?;
            request = request.with_response_format(schema);
        }

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
    ///     model: Optional model override for this request.
    ///
    /// Example:
    ///     >>> async def handle_chunk(chunk):
    ///     ...     if chunk["delta"]:
    ///     ...         print(chunk["delta"], end="")
    ///     >>> await model.stream([ChatMessage.user("Hi!")], handle_chunk)
    #[pyo3(signature = (messages, on_chunk, *, temperature=None, max_tokens=None, model=None))]
    fn stream<'py>(
        &self,
        py: Python<'py>,
        messages: Vec<PyRef<'py, PyChatMessage>>,
        on_chunk: Py<PyAny>,
        temperature: Option<f32>,
        max_tokens: Option<u32>,
        model: Option<String>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let rust_messages: Vec<ChatMessage> = messages.iter().map(|m| m.inner.clone()).collect();

        let mut request = CompletionRequest::new(rust_messages);
        if let Some(t) = temperature {
            request = request.with_temperature(t);
        }
        if let Some(mt) = max_tokens {
            request = request.with_max_tokens(mt);
        }
        if let Some(m) = model {
            request = request.with_model(m);
        }

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
                        let chunk_json = serde_json::json!({
                            "delta": chunk.delta,
                            "finish_reason": chunk.finish_reason,
                            "tool_calls": chunk.tool_calls.iter().map(|tc| {
                                serde_json::json!({"id": tc.id, "name": tc.name, "arguments": tc.arguments})
                            }).collect::<Vec<_>>(),
                        });

                        // Call the Python callback
                        tokio::task::block_in_place(|| {
                            Python::attach(|py| {
                                let py_val = crate::event::json_to_py(py, &chunk_json)?;
                                on_chunk.call1(py, (py_val,))?;
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
// PyToolCall
// ---------------------------------------------------------------------------

/// A tool invocation requested by the model.
#[pyclass(name = "ToolCall", from_py_object)]
#[derive(Clone)]
pub struct PyToolCall {
    inner: ToolCall,
}

#[pymethods]
impl PyToolCall {
    #[getter]
    fn id(&self) -> &str {
        &self.inner.id
    }

    #[getter]
    fn name(&self) -> &str {
        &self.inner.name
    }

    #[getter]
    fn arguments(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        crate::event::json_to_py(py, &self.inner.arguments)
    }

    fn __getitem__(&self, py: Python<'_>, key: &str) -> PyResult<Py<PyAny>> {
        match key {
            "id" => Ok(self.inner.id.clone().into_pyobject(py)?.into_any().unbind()),
            "name" => Ok(self
                .inner
                .name
                .clone()
                .into_pyobject(py)?
                .into_any()
                .unbind()),
            "arguments" => crate::event::json_to_py(py, &self.inner.arguments),
            _ => Err(pyo3::exceptions::PyKeyError::new_err(key.to_owned())),
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "ToolCall(id='{}', name='{}')",
            self.inner.id, self.inner.name
        )
    }
}

// ---------------------------------------------------------------------------
// PyTokenUsage
// ---------------------------------------------------------------------------

/// Token usage statistics for a completion.
#[pyclass(name = "TokenUsage", from_py_object)]
#[derive(Clone)]
pub struct PyTokenUsage {
    inner: TokenUsage,
}

#[pymethods]
impl PyTokenUsage {
    #[getter]
    fn prompt_tokens(&self) -> u32 {
        self.inner.prompt_tokens
    }

    #[getter]
    fn completion_tokens(&self) -> u32 {
        self.inner.completion_tokens
    }

    #[getter]
    fn total_tokens(&self) -> u32 {
        self.inner.total_tokens
    }

    fn __getitem__(&self, py: Python<'_>, key: &str) -> PyResult<Py<PyAny>> {
        match key {
            "prompt_tokens" => Ok(self
                .inner
                .prompt_tokens
                .into_pyobject(py)?
                .into_any()
                .unbind()),
            "completion_tokens" => Ok(self
                .inner
                .completion_tokens
                .into_pyobject(py)?
                .into_any()
                .unbind()),
            "total_tokens" => Ok(self
                .inner
                .total_tokens
                .into_pyobject(py)?
                .into_any()
                .unbind()),
            _ => Err(pyo3::exceptions::PyKeyError::new_err(key.to_owned())),
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "TokenUsage(prompt_tokens={}, completion_tokens={}, total_tokens={})",
            self.inner.prompt_tokens, self.inner.completion_tokens, self.inner.total_tokens
        )
    }
}

// ---------------------------------------------------------------------------
// PyRequestTiming
// ---------------------------------------------------------------------------

/// Request timing metadata.
#[pyclass(name = "RequestTiming", frozen, from_py_object)]
#[derive(Clone)]
pub struct PyRequestTiming {
    inner: RequestTiming,
}

#[pymethods]
impl PyRequestTiming {
    #[getter]
    fn queue_ms(&self) -> Option<u64> {
        self.inner.queue_ms
    }

    #[getter]
    fn execution_ms(&self) -> Option<u64> {
        self.inner.execution_ms
    }

    #[getter]
    fn total_ms(&self) -> Option<u64> {
        self.inner.total_ms
    }

    fn __getitem__(&self, py: Python<'_>, key: &str) -> PyResult<Py<PyAny>> {
        match key {
            "queue_ms" => Ok(self.inner.queue_ms.into_pyobject(py)?.into_any().unbind()),
            "execution_ms" => Ok(self
                .inner
                .execution_ms
                .into_pyobject(py)?
                .into_any()
                .unbind()),
            "total_ms" => Ok(self.inner.total_ms.into_pyobject(py)?.into_any().unbind()),
            _ => Err(pyo3::exceptions::PyKeyError::new_err(key.to_owned())),
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "RequestTiming(queue_ms={:?}, execution_ms={:?}, total_ms={:?})",
            self.inner.queue_ms, self.inner.execution_ms, self.inner.total_ms
        )
    }
}

// ---------------------------------------------------------------------------
// PyCompletionResponse
// ---------------------------------------------------------------------------

/// The result of a chat completion.
///
/// Supports both attribute access and dict-style access for backwards
/// compatibility:
///     >>> response.content        # attribute
///     >>> response["content"]     # dict-style
#[pyclass(name = "CompletionResponse", from_py_object)]
#[derive(Clone)]
pub struct PyCompletionResponse {
    pub(crate) inner: CompletionResponse,
}

#[pymethods]
impl PyCompletionResponse {
    #[getter]
    fn content(&self) -> Option<&str> {
        self.inner.content.as_deref()
    }

    #[getter]
    fn model(&self) -> &str {
        &self.inner.model
    }

    #[getter]
    fn finish_reason(&self) -> Option<&str> {
        self.inner.finish_reason.as_deref()
    }

    #[getter]
    fn tool_calls(&self) -> Vec<PyToolCall> {
        self.inner
            .tool_calls
            .iter()
            .map(|tc| PyToolCall { inner: tc.clone() })
            .collect()
    }

    #[getter]
    fn usage(&self) -> Option<PyTokenUsage> {
        self.inner.usage.clone().map(|u| PyTokenUsage { inner: u })
    }

    #[getter]
    fn cost(&self) -> Option<f64> {
        self.inner.cost
    }

    #[getter]
    fn timing(&self) -> Option<PyRequestTiming> {
        self.inner
            .timing
            .clone()
            .map(|t| PyRequestTiming { inner: t })
    }

    #[getter]
    fn images(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let val = serde_json::to_value(&self.inner.images).unwrap_or_default();
        crate::event::json_to_py(py, &val)
    }

    #[getter]
    fn audio(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let val = serde_json::to_value(&self.inner.audio).unwrap_or_default();
        crate::event::json_to_py(py, &val)
    }

    #[getter]
    fn videos(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let val = serde_json::to_value(&self.inner.videos).unwrap_or_default();
        crate::event::json_to_py(py, &val)
    }

    #[getter]
    fn metadata_extra(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        crate::event::json_to_py(py, &self.inner.metadata)
    }

    fn __getitem__(&self, py: Python<'_>, key: &str) -> PyResult<Py<PyAny>> {
        match key {
            "content" => match &self.inner.content {
                Some(s) => Ok(s.clone().into_pyobject(py)?.into_any().unbind()),
                None => Ok(py.None()),
            },
            "model" => Ok(self
                .inner
                .model
                .clone()
                .into_pyobject(py)?
                .into_any()
                .unbind()),
            "finish_reason" => match &self.inner.finish_reason {
                Some(s) => Ok(s.clone().into_pyobject(py)?.into_any().unbind()),
                None => Ok(py.None()),
            },
            "tool_calls" => {
                // Return as list of dicts for backwards compat
                let tool_calls: Vec<serde_json::Value> = self
                    .inner
                    .tool_calls
                    .iter()
                    .map(|tc| {
                        serde_json::json!({
                            "id": tc.id,
                            "name": tc.name,
                            "arguments": tc.arguments,
                        })
                    })
                    .collect();
                crate::event::json_to_py(py, &serde_json::Value::Array(tool_calls))
            }
            "usage" => {
                if let Some(usage) = &self.inner.usage {
                    let val = serde_json::json!({
                        "prompt_tokens": usage.prompt_tokens,
                        "completion_tokens": usage.completion_tokens,
                        "total_tokens": usage.total_tokens,
                    });
                    crate::event::json_to_py(py, &val)
                } else {
                    Ok(py.None())
                }
            }
            "cost" => match self.inner.cost {
                Some(c) => Ok(c.into_pyobject(py)?.into_any().unbind()),
                None => Ok(py.None()),
            },
            "timing" => match &self.inner.timing {
                Some(t) => {
                    let val = serde_json::json!({
                        "queue_ms": t.queue_ms,
                        "execution_ms": t.execution_ms,
                        "total_ms": t.total_ms,
                    });
                    crate::event::json_to_py(py, &val)
                }
                None => Ok(py.None()),
            },
            "images" => {
                let val = serde_json::to_value(&self.inner.images).unwrap_or_default();
                crate::event::json_to_py(py, &val)
            }
            "audio" => {
                let val = serde_json::to_value(&self.inner.audio).unwrap_or_default();
                crate::event::json_to_py(py, &val)
            }
            "videos" => {
                let val = serde_json::to_value(&self.inner.videos).unwrap_or_default();
                crate::event::json_to_py(py, &val)
            }
            "metadata" => crate::event::json_to_py(py, &self.inner.metadata),
            _ => Err(pyo3::exceptions::PyKeyError::new_err(key.to_owned())),
        }
    }

    fn keys(&self) -> Vec<&str> {
        vec![
            "content",
            "model",
            "finish_reason",
            "tool_calls",
            "usage",
            "cost",
            "timing",
            "images",
            "audio",
            "videos",
            "metadata",
        ]
    }

    fn __repr__(&self) -> String {
        format!(
            "CompletionResponse(model='{}', content='{}')",
            self.inner.model,
            self.inner.content.as_deref().unwrap_or(""),
        )
    }
}
