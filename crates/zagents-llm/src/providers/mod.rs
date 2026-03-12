//! LLM provider implementations.
//!
//! Each provider is behind a feature flag so that downstream crates only pull
//! in the HTTP dependencies they actually need.
//!
//! ## OpenAI-compatible providers
//!
//! The [`openai_compat`] module provides a single [`OpenAiCompatProvider`]
//! that works with any OpenAI-compatible endpoint. This covers OpenAI,
//! OpenRouter, Groq, Together AI, Mistral, DeepSeek, Fireworks, Perplexity,
//! xAI, Cohere, and AWS Bedrock (Mantle).
//!
//! The original [`openai`] module is retained for backwards compatibility but
//! is a simpler wrapper around the same underlying logic.
//!
//! [`OpenAiCompatProvider`]: openai_compat::OpenAiCompatProvider

// Shared SSE parser used by OpenAI-compatible, Azure, and Gemini providers.
// Enabled whenever any provider that needs it is active.
#[cfg(any(feature = "openai", feature = "azure"))]
pub(crate) mod sse;

#[cfg(feature = "openai")]
pub mod openai;

#[cfg(feature = "openai")]
pub mod openai_compat;

#[cfg(feature = "anthropic")]
pub mod anthropic;

#[cfg(feature = "gemini")]
pub mod gemini;

#[cfg(feature = "fal")]
pub mod fal;

#[cfg(feature = "azure")]
pub mod azure;
