//! Request routing for the OpenAI-compatible API.
//!
//! Dispatches incoming HTTP requests to the appropriate blazen-llm handler
//! based on the request path and method.

use std::collections::HashMap;
use std::sync::{LazyLock, RwLock};

use serde::{Deserialize, Serialize};

use blazen_llm::CompletionModel;
use blazen_llm::error::BlazenError;
use blazen_llm::http::HttpClient;
use blazen_llm::providers::anthropic::AnthropicProvider;
use blazen_llm::providers::azure::AzureOpenAiProvider;
use blazen_llm::providers::bedrock::BedrockProvider;
use blazen_llm::providers::cohere::CohereProvider;
use blazen_llm::providers::deepseek::DeepSeekProvider;
use blazen_llm::providers::fal::FalProvider;
use blazen_llm::providers::fireworks::FireworksProvider;
use blazen_llm::providers::gemini::GeminiProvider;
use blazen_llm::providers::groq::GroqProvider;
use blazen_llm::providers::mistral::MistralProvider;
use blazen_llm::providers::openai::OpenAiProvider;
use blazen_llm::providers::openai_compat::{AuthMethod, OpenAiCompatConfig, OpenAiCompatProvider};
use blazen_llm::providers::openrouter::OpenRouterProvider;
use blazen_llm::providers::perplexity::PerplexityProvider;
use blazen_llm::providers::together::TogetherProvider;
use blazen_llm::providers::xai::XaiProvider;
use blazen_llm::types::{ChatMessage, CompletionRequest, CompletionResponse};

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

/// A message in the `OpenAI` chat format.
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

/// A choice in the `OpenAI` response.
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

/// Token usage in the `OpenAI` response format.
#[derive(Debug, Serialize)]
pub struct OaiUsage {
    pub prompt_tokens: u64,
    pub completion_tokens: u64,
    pub total_tokens: u64,
}

/// An error response in the `OpenAI` format.
#[derive(Debug, Serialize)]
pub struct OaiErrorResponse {
    pub error: OaiErrorBody,
}

/// The error body within an `OpenAI` error response.
#[derive(Debug, Serialize)]
pub struct OaiErrorBody {
    pub message: String,
    #[serde(rename = "type")]
    pub error_type: String,
    pub code: Option<String>,
}

// ---------------------------------------------------------------------------
// Custom provider registration
// ---------------------------------------------------------------------------

/// Request body for registering a custom OpenAI-compatible provider.
#[derive(Debug, Deserialize)]
pub struct RegisterProviderRequest {
    /// Unique name for this provider (used as the provider prefix in model strings).
    pub name: String,
    /// Base URL for the API (e.g. "<https://my-llm.example.com/v1>").
    pub base_url: String,
    /// API key for this provider.
    pub api_key: String,
    /// Default model to use.
    #[serde(default = "default_custom_model")]
    pub default_model: String,
    /// Authentication method: "bearer", "`api_key_header`", "`azure_api_key`", "`key_prefix`".
    #[serde(default = "default_auth_method")]
    pub auth_method: String,
    /// Optional custom header name for API key auth (used with "`api_key_header`").
    pub api_key_header: Option<String>,
    /// Extra headers to include in every request.
    #[serde(default)]
    pub extra_headers: HashMap<String, String>,
    /// Whether this provider supports the /models listing endpoint.
    #[serde(default)]
    pub supports_model_listing: bool,
}

fn default_custom_model() -> String {
    "default".to_owned()
}

fn default_auth_method() -> String {
    "bearer".to_owned()
}

/// Stored configuration for a custom provider.
#[derive(Debug, Clone)]
struct CustomProvider {
    config: OpenAiCompatConfig,
}

/// Global registry for custom providers registered at runtime.
static CUSTOM_PROVIDERS: LazyLock<RwLock<HashMap<String, CustomProvider>>> =
    LazyLock::new(|| RwLock::new(HashMap::new()));

/// Provider info for the listing endpoint.
#[derive(Debug, Serialize)]
struct ProviderInfo {
    name: String,
    #[serde(rename = "type")]
    provider_type: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    base_url: Option<String>,
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

    /// Error response in `OpenAI` format.
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
    #[must_use] 
    pub fn not_found() -> Self {
        Self::error(404, "Not found", "not_found")
    }

    /// 405 Method Not Allowed.
    #[must_use] 
    pub fn method_not_allowed() -> Self {
        Self::error(405, "Method not allowed", "method_not_allowed")
    }

    /// Map a `BlazenError` to the appropriate HTTP status and `OpenAI` error body.
    #[must_use] 
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
    keys: &KeyProvider,
    http_client: &std::sync::Arc<dyn HttpClient>,
) -> RouteResponse {
    match (method, path) {
        ("GET", "/health") => handle_health(),
        ("GET", "/v1/providers") => handle_list_providers(keys),
        ("POST", "/v1/providers/register") => handle_register_provider(body),
        ("POST", "/v1/chat/completions") => handle_chat_completions(body, keys, http_client),
        ("POST", "/v1/images/generations") => handle_stub("Image generation"),
        ("POST", "/v1/audio/speech") => handle_stub("Text-to-speech"),
        ("POST", "/v1/agent/run") => handle_stub("Agent execution"),
        (
            _,
            "/health"
            | "/v1/providers"
            | "/v1/providers/register"
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

/// List all available providers (built-in + custom registered).
fn handle_list_providers(keys: &KeyProvider) -> RouteResponse {
    let built_in = [
        ("openai", "native"),
        ("anthropic", "native"),
        ("gemini", "native"),
        ("azure", "native"),
        ("fal", "native"),
        ("openrouter", "openai_compat"),
        ("groq", "openai_compat"),
        ("together", "openai_compat"),
        ("mistral", "openai_compat"),
        ("deepseek", "openai_compat"),
        ("fireworks", "openai_compat"),
        ("perplexity", "openai_compat"),
        ("xai", "openai_compat"),
        ("cohere", "openai_compat"),
        ("bedrock", "openai_compat"),
    ];

    let mut providers: Vec<ProviderInfo> = built_in
        .iter()
        .map(|(name, ptype)| ProviderInfo {
            name: (*name).to_owned(),
            provider_type: if keys.has_key(name) {
                (*ptype).to_owned()
            } else {
                format!("{ptype} (no key)")
            },
            base_url: None,
        })
        .collect();

    if let Ok(custom) = CUSTOM_PROVIDERS.read() {
        for (name, cp) in custom.iter() {
            providers.push(ProviderInfo {
                name: name.clone(),
                provider_type: "custom".to_owned(),
                base_url: Some(cp.config.base_url.clone()),
            });
        }
    }

    RouteResponse::ok(&serde_json::json!({ "providers": providers }))
}

/// Register a custom OpenAI-compatible provider at runtime.
fn handle_register_provider(body: &[u8]) -> RouteResponse {
    let req: RegisterProviderRequest = match serde_json::from_slice(body) {
        Ok(r) => r,
        Err(e) => {
            return RouteResponse::error(
                400,
                format!("Invalid registration body: {e}"),
                "invalid_request_error",
            );
        }
    };

    if req.name.is_empty() || req.base_url.is_empty() {
        return RouteResponse::error(
            400,
            "name and base_url are required",
            "invalid_request_error",
        );
    }

    let auth_method = match req.auth_method.as_str() {
        "bearer" => AuthMethod::Bearer,
        "api_key_header" => {
            let header = req
                .api_key_header
                .unwrap_or_else(|| "x-api-key".to_owned());
            AuthMethod::ApiKeyHeader(header)
        }
        "azure_api_key" => AuthMethod::AzureApiKey,
        "key_prefix" => AuthMethod::KeyPrefix,
        other => {
            return RouteResponse::error(
                400,
                format!("Unknown auth_method: {other}. Valid: bearer, api_key_header, azure_api_key, key_prefix"),
                "invalid_request_error",
            );
        }
    };

    let extra_headers: Vec<(String, String)> = req
        .extra_headers
        .into_iter()
        .collect();

    let config = OpenAiCompatConfig {
        provider_name: req.name.clone(),
        base_url: req.base_url,
        api_key: req.api_key,
        default_model: req.default_model,
        auth_method,
        extra_headers,
        query_params: Vec::new(),
        supports_model_listing: req.supports_model_listing,
    };

    let name = req.name.clone();
    match CUSTOM_PROVIDERS.write() {
        Ok(mut registry) => {
            registry.insert(req.name, CustomProvider { config });
        }
        Err(_) => {
            return RouteResponse::error(500, "Failed to write to registry", "server_error");
        }
    }

    RouteResponse::ok(&serde_json::json!({
        "status": "registered",
        "provider": name,
    }))
}

/// Chat completions handler.
///
/// Parses the OpenAI-compatible request, resolves the provider from the model
/// string, dispatches to blazen-llm, and formats the response.
fn handle_chat_completions(
    body: &[u8],
    keys: &KeyProvider,
    http_client: &std::sync::Arc<dyn HttpClient>,
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

    // Resolve API key -- check built-in keys first, then custom providers
    let api_key = keys
        .get(provider_name)
        .map(String::from)
        .or_else(|| {
            CUSTOM_PROVIDERS
                .read()
                .ok()
                .and_then(|reg| reg.get(provider_name).map(|cp| cp.config.api_key.clone()))
        });

    let Some(api_key) = api_key else {
        return RouteResponse::error(
            401,
            format!("No API key configured for provider: {provider_name}"),
            "authentication_error",
        );
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
    let mut request = CompletionRequest::new(messages).with_model(&model_id);

    if let Some(temp) = oai_request.temperature {
        request = request.with_temperature(temp);
    }
    if let Some(max) = oai_request.max_tokens {
        request = request.with_max_tokens(max);
    }
    if let Some(top_p) = oai_request.top_p {
        request = request.with_top_p(top_p);
    }

    // Resolve and construct the provider
    let provider: Box<dyn CompletionModel> = match resolve_provider(
        provider_name,
        &model_id,
        &api_key,
        std::sync::Arc::clone(http_client),
    ) {
        Ok(p) => p,
        Err(resp) => return resp,
    };

    // Execute the completion via the WASI blocking executor
    let response = match crate::executor::wasi_block_on(provider.complete(request)) {
        Ok(resp) => resp,
        Err(e) => return RouteResponse::from_blazen_error(&e),
    };

    // Map the response to OpenAI format
    RouteResponse::ok(&to_oai_response(&model_id, &response))
}

/// Stub handler for endpoints that are not yet implemented.
fn handle_stub(feature: &str) -> RouteResponse {
    RouteResponse::error(
        501,
        format!("{feature} is not yet implemented in the WASM component"),
        "not_implemented",
    )
}

// ---------------------------------------------------------------------------
// Provider resolution
// ---------------------------------------------------------------------------

/// Map a provider name to a concrete `CompletionModel` implementation.
fn resolve_provider(
    provider_name: &str,
    model_id: &str,
    api_key: &str,
    http_client: std::sync::Arc<dyn HttpClient>,
) -> Result<Box<dyn CompletionModel>, RouteResponse> {
    match provider_name {
        // Native providers with dedicated API formats
        "openai" => Ok(Box::new(
            OpenAiProvider::new_with_client(api_key, http_client).with_model(model_id),
        )),
        "anthropic" => Ok(Box::new(
            AnthropicProvider::new_with_client(api_key, http_client).with_model(model_id),
        )),
        "gemini" => Ok(Box::new(
            GeminiProvider::new_with_client(api_key, http_client).with_model(model_id),
        )),
        "azure" => {
            // Azure requires resource_name/deployment_name. We use the model_id
            // as deployment_name and expect AZURE_RESOURCE_NAME env var.
            let resource_name = std::env::var("AZURE_RESOURCE_NAME").unwrap_or_default();
            if resource_name.is_empty() {
                return Err(RouteResponse::error(
                    400,
                    "Azure requires AZURE_RESOURCE_NAME environment variable",
                    "invalid_request_error",
                ));
            }
            Ok(Box::new(AzureOpenAiProvider::new_with_client(
                api_key,
                resource_name,
                model_id,
                http_client,
            )))
        }
        // model_id sets the request body `model` field; the URL path is the
        // default OpenAiChat endpoint (openrouter/router/openai/v1/chat/completions).
        // To target a non-default fal endpoint family, use the FalProvider builder
        // directly with `with_llm_endpoint(...)` or `with_enterprise()`.
        "fal" => Ok(Box::new(
            FalProvider::new_with_client(api_key, http_client).with_llm_model(model_id),
        )),

        // OpenAI-compatible providers (dedicated types)
        "openrouter" => Ok(Box::new(
            OpenRouterProvider::new_with_client(api_key, http_client).with_model(model_id),
        )),
        "groq" => Ok(Box::new(
            GroqProvider::new_with_client(api_key, http_client).with_model(model_id),
        )),
        "together" => Ok(Box::new(
            TogetherProvider::new_with_client(api_key, http_client).with_model(model_id),
        )),
        "mistral" => Ok(Box::new(
            MistralProvider::new_with_client(api_key, http_client).with_model(model_id),
        )),
        "deepseek" => Ok(Box::new(
            DeepSeekProvider::new_with_client(api_key, http_client).with_model(model_id),
        )),
        "fireworks" => Ok(Box::new(
            FireworksProvider::new_with_client(api_key, http_client).with_model(model_id),
        )),
        "perplexity" => Ok(Box::new(
            PerplexityProvider::new_with_client(api_key, http_client).with_model(model_id),
        )),
        "xai" => Ok(Box::new(
            XaiProvider::new_with_client(api_key, http_client).with_model(model_id),
        )),
        "cohere" => Ok(Box::new(
            CohereProvider::new_with_client(api_key, http_client).with_model(model_id),
        )),
        "bedrock" => {
            let region =
                std::env::var("BEDROCK_REGION").unwrap_or_else(|_| "us-east-1".to_owned());
            Ok(Box::new(
                BedrockProvider::new_with_client(api_key, region, http_client)
                    .with_model(model_id),
            ))
        }

        // Check custom provider registry
        _ => {
            let custom_config = CUSTOM_PROVIDERS
                .read()
                .ok()
                .and_then(|reg| reg.get(provider_name).map(|cp| cp.config.clone()));

            match custom_config {
                Some(mut config) => {
                    // Use the api_key from the resolved key (env var takes precedence)
                    if !api_key.is_empty() {
                        api_key.clone_into(&mut config.api_key);
                    }
                    Ok(Box::new(
                        OpenAiCompatProvider::new_with_client(config, http_client)
                            .with_model(model_id),
                    ))
                }
                None => Err(RouteResponse::error(
                    400,
                    format!(
                        "Unknown provider: {provider_name}. Use POST /v1/providers/register to add custom providers."
                    ),
                    "invalid_request_error",
                )),
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Response mapping
// ---------------------------------------------------------------------------

/// Simple incrementing counter for response IDs.
fn next_response_id() -> String {
    use std::sync::atomic::{AtomicU64, Ordering};
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    format!("chatcmpl-wasm-{n}")
}

/// Map a `CompletionResponse` to the OpenAI-compatible response format.
fn to_oai_response(model_id: &str, response: &CompletionResponse) -> OaiChatResponse {
    OaiChatResponse {
        id: next_response_id(),
        object: "chat.completion".to_owned(),
        created: 0, // WASI has no easy wall-clock access; consumers can ignore this
        model: if response.model.is_empty() {
            model_id.to_owned()
        } else {
            response.model.clone()
        },
        choices: vec![OaiChoice {
            index: 0,
            message: OaiResponseMessage {
                role: "assistant".to_owned(),
                content: response.content.clone(),
            },
            finish_reason: response.finish_reason.clone(),
        }],
        usage: response.usage.as_ref().map(|u| OaiUsage {
            prompt_tokens: u64::from(u.prompt_tokens),
            completion_tokens: u64::from(u.completion_tokens),
            total_tokens: u64::from(u.total_tokens),
        }),
    }
}
