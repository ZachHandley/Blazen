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
