//! Bridge between [`blazen_embed_candle::CandleEmbedModel`] and
//! [`crate::EmbeddingModel`].
//!
//! When the `engine` feature is enabled on `blazen-embed-candle`, inference
//! runs locally via the candle framework. Without it, every call surfaces as
//! a [`BlazenError::Provider`] containing the `EngineNotAvailable` message.

use async_trait::async_trait;
use blazen_embed_candle::CandleEmbedModel;

use crate::error::BlazenError;
use crate::types::EmbeddingResponse;

#[async_trait]
impl crate::traits::EmbeddingModel for CandleEmbedModel {
    fn model_id(&self) -> &str {
        CandleEmbedModel::model_id(self)
    }

    fn dimensions(&self) -> usize {
        CandleEmbedModel::dimensions(self)
    }

    async fn embed(&self, texts: &[String]) -> Result<EmbeddingResponse, BlazenError> {
        let resp = CandleEmbedModel::embed(self, texts)
            .await
            .map_err(|e| BlazenError::provider("candle-embed", e.to_string()))?;

        Ok(EmbeddingResponse {
            embeddings: resp.embeddings,
            model: resp.model,
            usage: None,
            cost: None,
            timing: None,
            metadata: serde_json::Value::Null,
        })
    }
}

// ---------------------------------------------------------------------------
// LocalModel implementation
// ---------------------------------------------------------------------------

/// `LocalModel` bridge: gives callers explicit `load`/`unload` control over
/// the underlying candle embedding engine while preserving the existing
/// eager-load behavior of
/// [`CandleEmbedModel::from_options`](blazen_embed_candle::CandleEmbedModel::from_options)
/// and the lazy auto-reload behavior of
/// [`CandleEmbedModel::embed`](blazen_embed_candle::CandleEmbedModel::embed).
///
/// The impl forwards to the inherent methods on [`CandleEmbedModel`] and
/// wraps [`blazen_embed_candle::CandleEmbedError`] into
/// [`BlazenError::Provider`] via [`BlazenError::provider`]. The upstream
/// crate does not define a `From<CandleEmbedError> for BlazenError`
/// conversion (and cannot, because `blazen-embed-candle` does not depend
/// on `blazen-llm` -- the dependency edge runs the other way), so we do
/// the conversion inline here, matching the pattern used by the
/// `mistralrs` backend bridge.
///
/// Without the upstream `engine` feature, the inherent `load`, `unload`,
/// and `is_loaded` methods on [`CandleEmbedModel`] are stubs that return
/// `EngineNotAvailable` (for `load`), succeed as no-ops (for `unload`),
/// or return `false` (for `is_loaded`). This mirrors the behavior of
/// `embed` and lets downstream crates depend on `LocalModel` without
/// unconditionally pulling in the heavy candle runtime.
#[async_trait]
impl crate::traits::LocalModel for CandleEmbedModel {
    async fn load(&self) -> Result<(), BlazenError> {
        CandleEmbedModel::load(self)
            .await
            .map_err(|e| BlazenError::provider("candle-embed", e.to_string()))
    }

    async fn unload(&self) -> Result<(), BlazenError> {
        CandleEmbedModel::unload(self)
            .await
            .map_err(|e| BlazenError::provider("candle-embed", e.to_string()))
    }

    async fn is_loaded(&self) -> bool {
        CandleEmbedModel::is_loaded(self).await
    }

    fn device(&self) -> crate::device::Device {
        CandleEmbedModel::device_str(self)
            .and_then(|s| crate::device::Device::parse(s).ok())
            .unwrap_or(crate::device::Device::Cpu)
    }
}
