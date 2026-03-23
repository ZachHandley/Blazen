//! LLM provider implementations.
//!
//! All providers are always available. The only opt-in features are:
//! - `reqwest` — enables the native HTTP client (for non-WASM targets)
//! - `tiktoken` — enables exact BPE token counting
//!
//! ## Native providers (custom API formats)
//!
//! - [`openai`] — `OpenAI` Chat Completions API
//! - [`anthropic`] — Anthropic Messages API
//! - [`gemini`] — Google Gemini API
//! - [`azure`] — Azure `OpenAI` Service
//! - [`fal`] — fal.ai compute platform (LLM + media generation)
//!
//! ## OpenAI-compatible providers
//!
//! - [`openai_compat`] — Generic OpenAI-compatible base
//! - [`groq`] — Groq (ultra-fast LPU inference)
//! - [`openrouter`] — `OpenRouter` (400+ models)
//! - [`together`] — Together AI
//! - [`mistral`] — Mistral AI
//! - [`deepseek`] — `DeepSeek`
//! - [`fireworks`] — Fireworks AI
//! - [`perplexity`] — Perplexity
//! - [`xai`] — xAI (Grok)
//! - [`cohere`] — Cohere
//! - [`bedrock`] — AWS Bedrock (via Mantle)

// Shared SSE parser used by OpenAI-compatible and Azure providers.
pub(crate) mod sse;

// Shared multimodal content helpers and HTTP utilities.
pub(crate) mod openai_format;

// Native providers
pub mod anthropic;
pub mod azure;
pub mod fal;
pub mod gemini;
pub mod openai;
pub mod openai_compat;

// OpenAI-compatible dedicated providers
pub mod bedrock;
pub mod cohere;
pub mod deepseek;
pub mod fireworks;
pub mod groq;
pub mod mistral;
pub mod openrouter;
pub mod perplexity;
pub mod together;
pub mod xai;
