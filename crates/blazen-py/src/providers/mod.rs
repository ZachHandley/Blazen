//! Provider wrappers for LLM completion models and compute providers.

pub mod anthropic;
pub mod azure;
pub mod bedrock;
pub mod capability_providers;
pub mod cohere;
pub mod completion_model;
pub mod config;
pub mod custom;
pub mod decorators;
pub mod deepseek;
pub mod fal;
pub mod fireworks;
pub mod funcs;
pub mod gemini;
pub mod groq;
pub mod middleware;
pub mod mistral;
pub mod openai;
pub mod openai_compat;
pub mod openai_embedding;
pub mod openrouter;
pub mod options;
pub mod perplexity;
pub mod together;
pub mod xai;

#[cfg(feature = "llamacpp")]
pub mod llamacpp;

#[cfg(feature = "candle-llm")]
pub mod candle_llm;

#[cfg(feature = "candle-embed")]
pub mod candle_embed;

#[cfg(feature = "mistralrs")]
pub mod mistralrs;

#[cfg(feature = "whispercpp")]
pub mod whispercpp;

#[cfg(feature = "piper")]
pub mod piper;

#[cfg(feature = "diffusion")]
pub mod diffusion;

#[cfg(all(feature = "embed", not(target_env = "musl")))]
pub mod fastembed;

#[cfg(feature = "tract")]
pub mod tract;

pub use completion_model::PyCompletionModel;
pub use decorators::{PyCachedCompletionModel, PyFallbackModel, PyRetryCompletionModel};
pub use middleware::{PyCacheMiddleware, PyMiddleware, PyMiddlewareStack, PyRetryMiddleware};
