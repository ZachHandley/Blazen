//! Typed facade response for the [`blazen_embed`] backend.
//!
//! [`blazen_embed::EmbedResponse`] is a target-conditional alias resolving
//! to either `FastEmbedResponse` or `TractResponse`. This module exposes a
//! single [`JsEmbedResponse`] napi mirror so JS callers see a unified
//! shape regardless of the platform's chosen backend.

#![cfg(feature = "embed")]

use napi_derive::napi;

use blazen_llm::EmbedResponse as RustEmbedResponse;

// ---------------------------------------------------------------------------
// JsEmbedResponse
// ---------------------------------------------------------------------------

/// Response from a local embedding operation.
///
/// Mirrors [`blazen_llm::EmbedResponse`] (a target-conditional alias for the
/// underlying backend's response type).
#[napi(object, js_name = "EmbedResponse")]
pub struct JsEmbedResponse {
    /// The embedding vectors. JS `Number` is `f64`, so vectors are widened
    /// from the backend's native `f32` for transport.
    pub embeddings: Vec<Vec<f64>>,
    /// The model identifier that produced these embeddings.
    pub model: String,
}

impl From<RustEmbedResponse> for JsEmbedResponse {
    fn from(response: RustEmbedResponse) -> Self {
        Self {
            embeddings: response
                .embeddings
                .into_iter()
                .map(|v| v.into_iter().map(f64::from).collect())
                .collect(),
            model: response.model,
        }
    }
}
