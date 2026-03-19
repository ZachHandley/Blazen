//! Python wrapper for the CompletionModel type with all provider constructors.

use std::sync::Arc;

use pyo3::prelude::*;
use tokio_stream::StreamExt;

use blazen_llm::{ChatMessage, CompletionModel, CompletionRequest};

use crate::error::BlazenPyError;
use crate::types::{PyChatMessage, PyCompletionResponse};

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
            let schema = crate::workflow::event::py_to_json(py, fmt)?;
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
                                let py_val = crate::workflow::event::json_to_py(py, &chunk_json)?;
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
