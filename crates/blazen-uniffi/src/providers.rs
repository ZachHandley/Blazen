//! Provider factories (deleted in Wave D of the provider refactor).
//!
//! Historically this module hosted ~27 `#[uniffi::export]` `new_*_model` /
//! `new_*_embedding_model` free functions that constructed the now-deleted
//! central [`crate::llm::Model`] / [`crate::llm::EmbeddingModel`] opaques.
//! All of them have been replaced by the per-engine concrete provider
//! classes in [`crate::concrete::llm`] (`OpenAiProvider`, `AnthropicProvider`,
//! …), which expose richer per-provider builder APIs and align with the
//! polymorphic `LlmProvider` / `BaseProvider` capability surface.
//!
//! The module is intentionally left in place (rather than removed) so
//! external consumers tracking `pub mod providers;` don't see a hard
//! breakage in the module graph. It carries no public symbols today.
