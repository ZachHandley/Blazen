//! Polymorphic embedding provider opaque + lifecycle C ABI.
//!
//! [`BlazenEmbeddingProvider`] is the capability-erased counterpart of
//! the per-engine cabi embedding classes in [`crate::embed`]. It wraps
//! an `Arc<dyn blazen_uniffi::concrete::bases::EmbeddingProvider>` so
//! callers can hand any concrete engine
//! (`FastembedProvider`, `TractEmbedProvider`, `CandleEmbedProvider`,
//! `OpenAiEmbeddingProvider`, `FalEmbeddingProvider`) to surfaces that
//! take a provider polymorphically.
//!
//! ## Construction
//!
//! Each per-engine cabi embedding class exposes a
//! `blazen_<engine>_provider_as_embedding_provider` C function that
//! clones its inner `Arc<<Engine>Provider>`, coerces to
//! `Arc<dyn EmbeddingProvider>`, and returns a freshly-boxed
//! `*mut BlazenEmbeddingProvider`. The original per-engine handle
//! remains valid — `as_embedding_provider` is non-consuming.
//!
//! ## Ownership
//!
//! - The handle returned by `..._as_embedding_provider` is caller-owned.
//!   Free with [`blazen_embedding_provider_free`]. Double-free is
//!   undefined behavior.
//! - The wrapped `Arc` is reference-counted, so freeing the
//!   `BlazenEmbeddingProvider` does **not** invalidate the per-engine
//!   handle it was derived from; both clean up independently.

#![allow(dead_code)] // `into_ptr` is consumed by sibling modules; the public C fn keeps the symbol live.

use std::sync::Arc;

use blazen_uniffi::concrete::bases::EmbeddingProvider as InnerEmbeddingProvider;

/// Opaque wrapper around
/// `Arc<dyn blazen_uniffi::concrete::bases::EmbeddingProvider>`.
///
/// Free with [`blazen_embedding_provider_free`].
pub struct BlazenEmbeddingProvider(pub(crate) Arc<dyn InnerEmbeddingProvider>);

impl BlazenEmbeddingProvider {
    /// Heap-allocate the handle and return the raw pointer the caller owns.
    pub(crate) fn into_ptr(self) -> *mut BlazenEmbeddingProvider {
        Box::into_raw(Box::new(self))
    }
}

/// Frees a [`BlazenEmbeddingProvider`] handle. No-op on a null pointer.
///
/// # Safety
///
/// `handle` must be null OR a pointer previously produced by one of the
/// `blazen_<engine>_provider_as_embedding_provider` C functions in
/// [`crate::embed`]. Double-free is undefined behavior.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_embedding_provider_free(handle: *mut BlazenEmbeddingProvider) {
    if handle.is_null() {
        return;
    }
    // SAFETY: per the contract, `handle` came from `Box::into_raw`.
    drop(unsafe { Box::from_raw(handle) });
}
