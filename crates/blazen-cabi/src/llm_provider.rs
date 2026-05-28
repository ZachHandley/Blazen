//! Polymorphic LLM provider opaque + lifecycle C ABI.
//!
//! [`BlazenLlmProvider`] is the capability-erased counterpart of the
//! per-engine cabi classes in [`crate::llm_providers`]. It wraps an
//! `Arc<dyn blazen_uniffi::concrete::bases::LlmProvider>` so callers can
//! hand any concrete engine (`OpenAiProvider`, `AnthropicProvider`,
//! `GeminiProvider`, …) to surfaces that take a provider polymorphically
//! — currently the Agent constructor ([`crate::tool_handler::blazen_agent_new`])
//! and the batch helpers ([`crate::batch`]).
//!
//! ## Construction
//!
//! [`BlazenLlmProvider`] is **not** constructed directly. Each per-engine
//! cabi class exposes a `blazen_<engine>_provider_as_llm_provider` C
//! function that clones its inner `Arc<<Engine>Provider>`, coerces to
//! `Arc<dyn LlmProvider>`, and returns a freshly-boxed
//! `*mut BlazenLlmProvider`. The original per-engine handle remains
//! valid — `as_llm_provider` is non-consuming.
//!
//! ## Ownership
//!
//! - The handle returned by `..._as_llm_provider` is caller-owned.
//!   Free with [`blazen_llm_provider_free`]. Double-free is undefined
//!   behavior.
//! - The wrapped `Arc` is reference-counted, so freeing the
//!   `BlazenLlmProvider` does **not** invalidate the per-engine handle
//!   it was derived from; both clean up independently.

#![allow(dead_code)] // `into_ptr` is consumed by sibling modules; the public C fn keeps the symbol live.

use std::sync::Arc;

use blazen_uniffi::concrete::bases::LlmProvider as InnerLlmProvider;

/// Opaque wrapper around `Arc<dyn blazen_uniffi::concrete::bases::LlmProvider>`.
///
/// Free with [`blazen_llm_provider_free`].
pub struct BlazenLlmProvider(pub(crate) Arc<dyn InnerLlmProvider>);

impl BlazenLlmProvider {
    /// Heap-allocate the handle and return the raw pointer the caller owns.
    pub(crate) fn into_ptr(self) -> *mut BlazenLlmProvider {
        Box::into_raw(Box::new(self))
    }
}

/// Frees a [`BlazenLlmProvider`] handle. No-op on a null pointer.
///
/// # Safety
///
/// `handle` must be null OR a pointer previously produced by one of the
/// `blazen_<engine>_provider_as_llm_provider` C functions in
/// [`crate::llm_providers`]. Double-free is undefined behavior.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_llm_provider_free(handle: *mut BlazenLlmProvider) {
    if handle.is_null() {
        return;
    }
    // SAFETY: per the contract, `handle` came from `Box::into_raw`.
    drop(unsafe { Box::from_raw(handle) });
}
