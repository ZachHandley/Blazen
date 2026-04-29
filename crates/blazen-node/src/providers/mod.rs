//! Provider implementations for the Node.js bindings.

pub mod anthropic;
pub mod azure;
pub mod bedrock;
pub mod capability_providers;
pub mod cohere;
pub mod completion_model;
pub mod custom;
pub mod deepseek;
pub mod embedding_models;
pub mod fal;
pub mod fireworks;
pub mod free_fns;
pub mod gemini;
pub mod groq;
pub mod middleware;
pub mod mistral;
pub mod openai;
pub mod openai_compat;
pub mod openrouter;
pub mod perplexity;
pub mod together;
pub mod transcription;
pub mod typed_tool;
pub mod wrappers;
pub mod xai;

// Local backend providers (feature-gated)
#[cfg(feature = "candle-embed")]
pub mod candle_embed;
#[cfg(feature = "candle-llm")]
pub mod candle_llm;
#[cfg(feature = "diffusion")]
pub mod diffusion;
#[cfg(feature = "embed")]
pub mod embed;
#[cfg(all(feature = "fastembed", not(target_env = "musl")))]
pub mod fastembed;
#[cfg(feature = "llamacpp")]
pub mod llamacpp;
#[cfg(feature = "mistralrs")]
pub mod mistralrs;
#[cfg(feature = "piper")]
pub mod piper;
#[cfg(feature = "tract")]
pub mod tract;
#[cfg(feature = "whispercpp")]
pub mod whispercpp;

// Re-export the main types.
pub use anthropic::JsAnthropicProvider;
pub use azure::JsAzureOpenAiProvider;
pub use bedrock::JsBedrockProvider;
pub use capability_providers::{
    CapabilityProviderConfig, JsBackgroundRemovalProvider, JsImageProvider, JsMusicProvider,
    JsTTSProvider, JsThreeDProvider, JsVideoProvider, JsVoiceProvider,
};
pub use cohere::JsCohereProvider;
pub use completion_model::JsCompletionModel;
pub use custom::{CustomProviderOptions, JsCustomProvider};
pub use deepseek::JsDeepSeekProvider;
pub use embedding_models::{JsOpenAiCompatEmbeddingModel, JsOpenAiEmbeddingModel};
pub use fal::{JsFalEmbeddingModel, JsFalProvider};
pub use fireworks::JsFireworksProvider;
pub use free_fns::{
    JsProviderEnvVar, env_var_for_provider, extract_inline_artifacts, format_provider_http_tail,
    get_context_window, provider_env_vars, register_from_model_info, resolve_api_key,
};
pub use gemini::JsGeminiProvider;
pub use groq::JsGroqProvider;
pub use middleware::{
    JsCacheMiddleware, JsMiddleware, JsMiddlewareConfig, JsMiddlewareStack, JsRetryMiddleware,
};
pub use mistral::JsMistralProvider;
pub use openai::JsOpenAiProvider;
pub use openai_compat::{JsAuthMethod, JsOpenAiCompatConfig, JsOpenAiCompatProvider};
pub use openrouter::JsOpenRouterProvider;
pub use perplexity::JsPerplexityProvider;
pub use together::JsTogetherProvider;
pub use transcription::JsTranscription;
pub use typed_tool::{JsTypedTool, typed_tool_simple};
pub use wrappers::{JsCachedCompletionModel, JsFallbackModel, JsRetryCompletionModel};
pub use xai::JsXaiProvider;

#[cfg(feature = "candle-embed")]
pub use candle_embed::{JsCandleEmbedOptions, JsCandleEmbedProvider};
#[cfg(feature = "candle-llm")]
pub use candle_llm::{JsCandleLlmOptions, JsCandleLlmProvider};
#[cfg(feature = "diffusion")]
pub use diffusion::{JsDiffusionOptions, JsDiffusionProvider, JsDiffusionScheduler};
#[cfg(feature = "embed")]
pub use embed::JsEmbedProvider;
#[cfg(all(feature = "fastembed", not(target_env = "musl")))]
pub use fastembed::{JsFastEmbedModel, JsFastEmbedOptions, JsFastEmbedResponse};
#[cfg(feature = "llamacpp")]
pub use llamacpp::{
    JsLlamaCppChatMessageInput, JsLlamaCppChatRole, JsLlamaCppInferenceChunk,
    JsLlamaCppInferenceChunkStream, JsLlamaCppInferenceResult, JsLlamaCppInferenceUsage,
    JsLlamaCppOptions, JsLlamaCppProvider,
};
#[cfg(feature = "mistralrs")]
pub use mistralrs::{
    JsChatMessageInput, JsChatRole, JsInferenceChunk, JsInferenceChunkStream, JsInferenceImage,
    JsInferenceImageSource, JsInferenceResult, JsInferenceToolCall, JsInferenceUsage,
    JsMistralRsProvider,
};
#[cfg(feature = "piper")]
pub use piper::{JsPiperOptions, JsPiperProvider};
#[cfg(feature = "tract")]
pub use tract::{JsTractEmbedModel, JsTractOptions, JsTractResponse};
#[cfg(feature = "whispercpp")]
pub use whispercpp::JsWhisperCppProvider;
