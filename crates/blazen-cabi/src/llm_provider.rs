//! Polymorphic LLM provider opaque + lifecycle C ABI.
//!
//! [`BlazenLlmProvider`] is the polymorphic counterpart of the
//! per-engine cabi classes in [`crate::llm_providers`]. It carries both the
//! capability-erased `Arc<dyn blazen_uniffi::concrete::bases::LlmProvider>`
//! (completion only) and the concrete engine's real
//! `Arc<dyn blazen_llm::Model>` (which also streams), so callers can hand any
//! concrete engine (`OpenAiProvider`, `AnthropicProvider`, `GeminiProvider`,
//! …) to surfaces that take a provider polymorphically — currently the Agent
//! constructor ([`crate::tool_handler::blazen_agent_new`]), the batch helpers
//! ([`crate::batch`]), and by-name (streaming-capable) dispatch through the
//! C-ABI [`blazen_manager::ModelManager`]
//! ([`crate::manager::blazen_model_manager_register_remote`]).
//!
//! ## Construction
//!
//! [`BlazenLlmProvider`] is **not** constructed directly. Each per-engine
//! cabi class exposes a `blazen_<engine>_provider_as_llm_provider` C
//! function that clones its inner `Arc<<Engine>Provider>`, coerces it to both
//! `Arc<dyn LlmProvider>` and `Arc<dyn blazen_llm::Model>` (the latter via the
//! engine's `as_model()` accessor), and returns a freshly-boxed
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

/// Opaque wrapper carrying both the capability-erased completion provider and
/// the real `Arc<dyn blazen_llm::Model>` for the same concrete engine.
///
/// Free with [`blazen_llm_provider_free`].
///
/// Why two fields: `provider` is the uniffi completion capability trait
/// (`complete` only) used by the Agent / batch surfaces; `model` is the full
/// `blazen_llm::Model` (which also streams). The per-engine
/// `..._as_llm_provider` C functions populate both from the same concrete
/// `Arc<<Engine>Provider>` (via its now-`pub` `as_model()` accessor), so a
/// remote provider registered by name through the C-ABI `ModelManager` can
/// dispatch BOTH completion and streaming — the stream surface is no longer
/// lost at the capability-erasure boundary.
pub struct BlazenLlmProvider {
    pub(crate) provider: Arc<dyn InnerLlmProvider>,
    pub(crate) model: Arc<dyn blazen_llm::Model>,
}

impl BlazenLlmProvider {
    /// Build a handle from the capability provider and the real model erased
    /// from the same concrete engine.
    pub(crate) fn new(
        provider: Arc<dyn InnerLlmProvider>,
        model: Arc<dyn blazen_llm::Model>,
    ) -> Self {
        BlazenLlmProvider { provider, model }
    }

    /// Heap-allocate the handle and return the raw pointer the caller owns.
    pub(crate) fn into_ptr(self) -> *mut BlazenLlmProvider {
        Box::into_raw(Box::new(self))
    }

    /// Hand back the real `Arc<dyn blazen_llm::Model>` so it can be filed in a
    /// [`blazen_manager::ModelManager`] for by-name dispatch via
    /// [`crate::manager::blazen_model_manager_register_remote`]. Because this
    /// is the concrete engine's own model (not a completion-only adapter), the
    /// registered model streams as well as completes.
    pub(crate) fn as_model(&self) -> Arc<dyn blazen_llm::Model> {
        Arc::clone(&self.model)
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
