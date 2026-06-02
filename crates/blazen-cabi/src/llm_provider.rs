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

use std::pin::Pin;
use std::sync::Arc;

use blazen_uniffi::concrete::bases::LlmProvider as InnerLlmProvider;
use blazen_uniffi::llm::ModelRequest as WireModelRequest;
use futures_util::Stream;

/// Opaque wrapper around `Arc<dyn blazen_uniffi::concrete::bases::LlmProvider>`.
///
/// Free with [`blazen_llm_provider_free`].
pub struct BlazenLlmProvider(pub(crate) Arc<dyn InnerLlmProvider>);

impl BlazenLlmProvider {
    /// Heap-allocate the handle and return the raw pointer the caller owns.
    pub(crate) fn into_ptr(self) -> *mut BlazenLlmProvider {
        Box::into_raw(Box::new(self))
    }

    /// Erase the wrapped capability-typed provider to an
    /// `Arc<dyn blazen_llm::Model>` so it can be filed in a
    /// [`blazen_manager::ModelManager`] for by-name dispatch via
    /// [`crate::manager::blazen_model_manager_register_remote`].
    ///
    /// Why an adapter (and not a direct downcast): the polymorphic handle only
    /// retains the capability-erased `Arc<dyn LlmProvider>` (the uniffi
    /// completion trait). The concrete engine's `as_model()` accessor — which
    /// hands back the real `Arc<dyn blazen_llm::Model>` — is crate-private to
    /// `blazen-uniffi`, so the cabi can't reach it. [`LlmProviderModelAdapter`]
    /// bridges the two by converting `blazen_llm` request/response records to
    /// and from the uniffi wire records around the inner `complete` call.
    pub(crate) fn as_model(&self) -> Arc<dyn blazen_llm::Model> {
        Arc::new(LlmProviderModelAdapter(Arc::clone(&self.0)))
    }
}

/// Map a uniffi wire [`blazen_uniffi::errors::BlazenError`] back to a core
/// [`blazen_llm::BlazenError`]. `blazen-uniffi` only ships the forward
/// `From<blazen_llm::BlazenError>` impl (core → wire); this is the reverse
/// hop the adapter needs so a remote-provider failure surfaces as a typed
/// core error (which the manager surface then re-wraps to wire on the way out
/// via the existing `into_inner_error`). Categories that don't have a 1:1 core
/// variant fold into `Provider` / `Internal` carrying the Display string.
fn wire_err_to_core(err: blazen_uniffi::errors::BlazenError) -> blazen_llm::BlazenError {
    use blazen_llm::BlazenError as C;
    use blazen_uniffi::errors::BlazenError as W;

    let display = err.to_string();
    match err {
        W::Auth { message } => C::Auth { message },
        W::RateLimit { retry_after_ms, .. } => C::RateLimit { retry_after_ms },
        W::Timeout { elapsed_ms, .. } => C::Timeout { elapsed_ms },
        W::Validation { message } => C::validation(message),
        W::ContentPolicy { message } => C::ContentPolicy { message },
        W::Unsupported { message } => C::Unsupported { message },
        W::Tool { message } => C::Tool {
            name: None,
            message,
        },
        W::Provider {
            message,
            provider,
            status,
            ..
        } => C::Provider {
            provider: provider.unwrap_or_else(|| "remote".to_owned()),
            message,
            status_code: status.and_then(|s| u16::try_from(s).ok()),
        },
        _ => C::Provider {
            provider: "remote".to_owned(),
            message: display,
            status_code: None,
        },
    }
}

/// Adapts an `Arc<dyn blazen_uniffi::concrete::bases::LlmProvider>` to the
/// core [`blazen_llm::Model`] trait so a polymorphic remote provider can be
/// registered into a [`blazen_manager::ModelManager`] and dispatched by name.
///
/// Completion round-trips through the uniffi wire records
/// ([`blazen_uniffi::llm::ModelRequest`] / `ModelResponse`) using the existing
/// `From<CoreModelRequest>` / `TryFrom<ModelResponse>` impls in
/// `blazen-uniffi`. Streaming is unsupported: the uniffi `LlmProvider`
/// capability trait has no streaming surface, so `stream` returns
/// `BlazenError::Unsupported` — by-name streaming requires a concrete local
/// backend (registered via the lifecycle path), not a polymorphic remote
/// provider handle.
struct LlmProviderModelAdapter(Arc<dyn InnerLlmProvider>);

#[async_trait::async_trait]
impl blazen_llm::Model for LlmProviderModelAdapter {
    fn model_id(&self) -> &'static str {
        // The uniffi `LlmProvider` trait exposes a stable engine id rather than
        // a borrowed model string; `Model::model_id` returns `&str`, so fall
        // back to a static label. The dispatch id supplied at registration time
        // is what the manager keys on, so this borrow is informational only.
        "remote-provider"
    }

    async fn complete(
        &self,
        request: blazen_llm::ModelRequest,
    ) -> Result<blazen_llm::ModelResponse, blazen_llm::BlazenError> {
        let wire_req: WireModelRequest = request.into();
        let wire_resp = self.0.complete(wire_req).await.map_err(wire_err_to_core)?;
        blazen_llm::ModelResponse::try_from(wire_resp).map_err(wire_err_to_core)
    }

    async fn stream(
        &self,
        _request: blazen_llm::ModelRequest,
    ) -> Result<
        Pin<Box<dyn Stream<Item = Result<blazen_llm::StreamChunk, blazen_llm::BlazenError>> + Send>>,
        blazen_llm::BlazenError,
    > {
        Err(blazen_llm::BlazenError::unsupported(
            "streaming is not supported for a polymorphic remote provider registered through the \
             C-ABI; use the per-engine streaming entry points or register a local backend",
        ))
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
