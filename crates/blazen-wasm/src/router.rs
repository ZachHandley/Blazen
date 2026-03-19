//! Request routing for the OpenAI-compatible API.
//!
//! Dispatches incoming HTTP requests to the appropriate blazen-llm handler
//! based on the request path and method.

use serde::{Deserialize, Serialize};

use blazen_llm::error::BlazenError;
use blazen_llm::http::HttpClient;
use blazen_llm::types::{ChatMessage, CompletionRequest, CompletionResponse, Role};

use crate::keys::KeyProvider;

// ---------------------------------------------------------------------------
// OpenAI-compatible request/response types
// ---------------------------------------------------------------------------

/// OpenAI-compatible chat completion request body.
#[derive(Debug, Deserialize)]
pub struct OaiChatRequest {
    /// Model identifier (e.g. "gpt-4.1", "claude-sonnet-4-20250514").
    pub model: String,
    /// Conversation messages.
    pub messages: Vec<OaiMessage>,
    /// Sampling temperature.
    #[serde(default)]
    pub temperature: Option<f32>,
    /// Maximum tokens to generate.
    #[serde(default)]
    pub max_tokens: Option<u32>,
    /// Nucleus sampling.
    #[serde(default)]
    pub top_p: Option<f32>,
    /// Whether to stream the response.
    #[serde(default)]
    pub stream: Option<bool>,
}

/// A message in the OpenAI chat format.
#[derive(Debug, Deserialize)]
pub struct OaiMessage {
    pub role: String,
    pub content: Option<String>,
}

/// OpenAI-compatible chat completion response.
#[derive(Debug, Serialize)]
pub struct OaiChatResponse {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<OaiChoice>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<OaiUsage>,
}

/// A choice in the OpenAI response.
#[derive(Debug, Serialize)]
pub struct OaiChoice {
    pub index: u32,
    pub message: OaiResponseMessage,
    pub finish_reason: Option<String>,
}

/// The assistant's message in the response.
#[derive(Debug, Serialize)]
pub struct OaiResponseMessage {
    pub role: String,
    pub content: Option<String>,
}

/// Token usage in the OpenAI response format.
#[derive(Debug, Serialize)]
pub struct OaiUsage {
    pub prompt_tokens: u64,
    pub completion_tokens: u64,
    pub total_tokens: u64,
}

/// An error response in the OpenAI format.
#[derive(Debug, Serialize)]
pub struct OaiErrorResponse {
    pub error: OaiErrorBody,
}

/// The error body within an OpenAI error response.
#[derive(Debug, Serialize)]
pub struct OaiErrorBody {
    pub message: String,
    #[serde(rename = "type")]
    pub error_type: String,
    pub code: Option<String>,
}

// ---------------------------------------------------------------------------
// Route result
// ---------------------------------------------------------------------------

/// The result of routing a request: status code + JSON body.
pub struct RouteResponse {
    pub status: u16,
    pub body: Vec<u8>,
    pub content_type: String,
}

impl RouteResponse {
    /// Create a JSON response with the given status.
    pub fn json(status: u16, value: &impl Serialize) -> Self {
        Self {
            status,
            body: serde_json::to_vec(value).unwrap_or_default(),
            content_type: "application/json".to_owned(),
        }
    }

    /// 200 OK with a JSON body.
    pub fn ok(value: &impl Serialize) -> Self {
        Self::json(200, value)
    }

    /// Error response in OpenAI format.
    pub fn error(status: u16, message: impl Into<String>, error_type: impl Into<String>) -> Self {
        let resp = OaiErrorResponse {
            error: OaiErrorBody {
                message: message.into(),
                error_type: error_type.into(),
                code: None,
            },
        };
        Self::json(status, &resp)
    }

    /// 404 Not Found.
    pub fn not_found() -> Self {
        Self::error(404, "Not found", "not_found")
    }

    /// 405 Method Not Allowed.
    pub fn method_not_allowed() -> Self {
        Self::error(405, "Method not allowed", "method_not_allowed")
    }

    /// Map a `BlazenError` to the appropriate HTTP status and OpenAI error body.
    pub fn from_blazen_error(err: &BlazenError) -> Self {
        let (status, error_type) = match err {
            BlazenError::Auth { .. } => (401, "authentication_error"),
            BlazenError::RateLimit { .. } => (429, "rate_limit_error"),
            BlazenError::Validation { .. } => (400, "invalid_request_error"),
            BlazenError::ContentPolicy { .. } => (400, "content_policy_violation"),
            BlazenError::Unsupported { .. } => (400, "unsupported"),
            BlazenError::Completion(_) => (500, "completion_error"),
            _ => (500, "server_error"),
        };
        Self::error(status, err.to_string(), error_type)
    }
}

// ---------------------------------------------------------------------------
// Router
// ---------------------------------------------------------------------------

/// Route an incoming HTTP request to the appropriate handler.
///
/// Returns a `RouteResponse` with the status code and serialized body.
pub fn route(
    method: &str,
    path: &str,
    body: &[u8],
    _keys: &KeyProvider,
    _http_client: &std::sync::Arc<dyn HttpClient>,
) -> RouteResponse {
    match (method, path) {
        ("GET", "/health") => handle_health(),
        ("POST", "/v1/chat/completions") => handle_chat_completions(body, _keys, _http_client),
        ("POST", "/v1/images/generations") => handle_stub("Image generation"),
        ("POST", "/v1/audio/speech") => handle_stub("Text-to-speech"),
        ("POST", "/v1/agent/run") => handle_stub("Agent execution"),
        (
            _,
            "/health"
            | "/v1/chat/completions"
            | "/v1/images/generations"
            | "/v1/audio/speech"
            | "/v1/agent/run",
        ) => RouteResponse::method_not_allowed(),
        _ => RouteResponse::not_found(),
    }
}

// ---------------------------------------------------------------------------
// Handlers
// ---------------------------------------------------------------------------

/// Health check -- always returns 200 with a simple JSON body.
fn handle_health() -> RouteResponse {
    RouteResponse::ok(&serde_json::json!({
        "status": "ok",
        "service": "blazen-wasm",
        "version": env!("CARGO_PKG_VERSION"),
    }))
}

/// Chat completions handler.
///
/// Parses the OpenAI-compatible request, resolves the provider from the model
/// string, dispatches to blazen-llm, and formats the response.
fn handle_chat_completions(
    body: &[u8],
    keys: &KeyProvider,
    _http_client: &std::sync::Arc<dyn HttpClient>,
) -> RouteResponse {
    // Parse the request body
    let oai_request: OaiChatRequest = match serde_json::from_slice(body) {
        Ok(req) => req,
        Err(e) => {
            return RouteResponse::error(
                400,
                format!("Invalid request body: {e}"),
                "invalid_request_error",
            );
        }
    };

    // Streaming is not yet supported in the WASM component
    if oai_request.stream == Some(true) {
        return RouteResponse::error(
            400,
            "Streaming is not yet supported in the WASM component",
            "unsupported",
        );
    }

    // Resolve provider from model name.
    // Convention: "provider/model" (e.g., "openai/gpt-4.1", "anthropic/claude-sonnet-4-20250514")
    // If no slash, default to openai.
    let (provider_name, model_id) = match oai_request.model.split_once('/') {
        Some((p, m)) => (p, m.to_owned()),
        None => ("openai", oai_request.model.clone()),
    };

    // Check we have a key for this provider
    let _api_key = match keys.get(provider_name) {
        Some(key) => key,
        None => {
            return RouteResponse::error(
                401,
                format!("No API key configured for provider: {provider_name}"),
                "authentication_error",
            );
        }
    };

    // Convert OAI messages to blazen-llm ChatMessages
    let messages: Vec<ChatMessage> = oai_request
        .messages
        .iter()
        .filter_map(|m| {
            let content = m.content.as_deref().unwrap_or("");
            match m.role.as_str() {
                "system" => Some(ChatMessage::system(content)),
                "user" => Some(ChatMessage::user(content)),
                "assistant" => Some(ChatMessage::assistant(content)),
                _ => None,
            }
        })
        .collect();

    if messages.is_empty() {
        return RouteResponse::error(400, "No valid messages provided", "invalid_request_error");
    }

    // Build the blazen-llm CompletionRequest
    let mut _request = CompletionRequest::new(messages).with_model(&model_id);

    if let Some(temp) = oai_request.temperature {
        _request = _request.with_temperature(temp);
    }
    if let Some(max) = oai_request.max_tokens {
        _request = _request.with_max_tokens(max);
    }
    if let Some(top_p) = oai_request.top_p {
        _request = _request.with_top_p(top_p);
    }

    // NOTE: Actually dispatching to the provider requires async execution.
    // In the WASI preview2 model, the incoming-handler is synchronous.
    // For the MVP we validate the request and return a placeholder indicating
    // the route is wired up. Real async dispatch will be added once we
    // integrate with wasi:http's async model or use a blocking executor.
    //
    // TODO: Execute the completion request against the resolved provider.
    // This requires:
    //   1. Constructing the appropriate provider (OpenAI, Anthropic, etc.)
    //      with the resolved API key and the WASI HTTP client
    //   2. Calling provider.complete(request).await
    //   3. Mapping the CompletionResponse back to OaiChatResponse

    RouteResponse::error(
        501,
        format!(
            "Provider dispatch not yet implemented for {provider_name}/{model_id}. \
             Request validated successfully."
        ),
        "not_implemented",
    )
}

/// Stub handler for endpoints that are not yet implemented.
fn handle_stub(feature: &str) -> RouteResponse {
    RouteResponse::error(
        501,
        format!("{feature} is not yet implemented in the WASM component"),
        "not_implemented",
    )
}
