//! Python wrappers for LLM provider types.
//!
//! Exposes [`CompletionModel`](zagents_llm::CompletionModel) implementations
//! to Python with static constructor methods for each provider.

use std::sync::Arc;

use pyo3::prelude::*;

use zagents_llm::{ChatMessage, CompletionModel, CompletionRequest, MessageContent, Role};

use crate::error::ZAgentsPyError;

// ---------------------------------------------------------------------------
// PyChatMessage
// ---------------------------------------------------------------------------

/// A single message in a chat conversation.
///
/// Example:
///     >>> msg = ChatMessage("user", "Hello, world!")
///     >>> msg = ChatMessage.system("You are a helpful assistant.")
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
    ///     role: One of "system", "user", "assistant", "tool".
    ///     content: The message text.
    #[new]
    fn new(role: &str, content: &str) -> PyResult<Self> {
        let role = match role {
            "system" => Role::System,
            "user" => Role::User,
            "assistant" => Role::Assistant,
            "tool" => Role::Tool,
            other => {
                return Err(ZAgentsPyError::InvalidArgument(format!(
                    "unknown role: '{other}' (expected system, user, assistant, or tool)"
                ))
                .into());
            }
        };
        Ok(Self {
            inner: ChatMessage {
                role,
                content: MessageContent::Text(content.to_owned()),
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
    inner: Arc<dyn CompletionModel>,
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
        let mut provider = zagents_llm::providers::openai::OpenAiProvider::new(api_key);
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
        let mut provider = zagents_llm::providers::anthropic::AnthropicProvider::new(api_key);
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
        let mut provider = zagents_llm::providers::gemini::GeminiProvider::new(api_key);
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
        let provider = zagents_llm::providers::azure::AzureOpenAiProvider::new(
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
            zagents_llm::providers::openai_compat::OpenAiCompatProvider::openrouter(api_key);
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
            zagents_llm::providers::openai_compat::OpenAiCompatProvider::groq(api_key);
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
            zagents_llm::providers::openai_compat::OpenAiCompatProvider::together(api_key);
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
            zagents_llm::providers::openai_compat::OpenAiCompatProvider::mistral(api_key);
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
            zagents_llm::providers::openai_compat::OpenAiCompatProvider::deepseek(api_key);
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
            zagents_llm::providers::openai_compat::OpenAiCompatProvider::fireworks(api_key);
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
            zagents_llm::providers::openai_compat::OpenAiCompatProvider::perplexity(api_key);
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
        let mut provider =
            zagents_llm::providers::openai_compat::OpenAiCompatProvider::xai(api_key);
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
            zagents_llm::providers::openai_compat::OpenAiCompatProvider::cohere(api_key);
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
            zagents_llm::providers::openai_compat::OpenAiCompatProvider::bedrock(api_key, region);
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
        let mut provider = zagents_llm::providers::fal::FalProvider::new(api_key);
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
    ///
    /// Returns:
    ///     A dict with keys: content, model, tool_calls, usage, finish_reason.
    ///
    /// Example:
    ///     >>> response = await model.complete([
    ///     ...     ChatMessage.system("You are helpful."),
    ///     ...     ChatMessage.user("What is 2+2?"),
    ///     ... ])
    ///     >>> print(response["content"])
    #[pyo3(signature = (messages, temperature=None, max_tokens=None, model=None))]
    fn complete<'py>(
        &self,
        py: Python<'py>,
        messages: Vec<PyRef<'py, PyChatMessage>>,
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
            let response = inner
                .complete(request)
                .await
                .map_err(ZAgentsPyError::from)?;

            // Serialize the response to a JSON value that can be converted to
            // a Python dict when the GIL is reacquired by the pyo3 runtime.
            let mut result_map = serde_json::Map::new();
            result_map.insert(
                "content".to_owned(),
                response
                    .content
                    .map_or(serde_json::Value::Null, serde_json::Value::String),
            );
            result_map.insert(
                "model".to_owned(),
                serde_json::Value::String(response.model),
            );
            result_map.insert(
                "finish_reason".to_owned(),
                response
                    .finish_reason
                    .map_or(serde_json::Value::Null, serde_json::Value::String),
            );

            // Tool calls
            let tool_calls: Vec<serde_json::Value> = response
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
            result_map.insert(
                "tool_calls".to_owned(),
                serde_json::Value::Array(tool_calls),
            );

            // Usage
            let usage_val = if let Some(usage) = &response.usage {
                serde_json::json!({
                    "prompt_tokens": usage.prompt_tokens,
                    "completion_tokens": usage.completion_tokens,
                    "total_tokens": usage.total_tokens,
                })
            } else {
                serde_json::Value::Null
            };
            result_map.insert("usage".to_owned(), usage_val);

            Ok(crate::event::JsonValue(serde_json::Value::Object(
                result_map,
            )))
        })
    }

    fn __repr__(&self) -> String {
        format!("CompletionModel(model_id='{}')", self.inner.model_id())
    }
}
