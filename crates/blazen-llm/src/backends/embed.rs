//! Bridge between [`blazen_embed::EmbedModel`] and
//! [`crate::EmbeddingModel`].

use async_trait::async_trait;
use blazen_embed::EmbedModel;

use crate::error::BlazenError;
use crate::types::EmbeddingResponse;

#[async_trait]
impl crate::traits::EmbeddingModel for EmbedModel {
    fn model_id(&self) -> &str {
        EmbedModel::model_id(self)
    }

    fn dimensions(&self) -> usize {
        EmbedModel::dimensions(self)
    }

    async fn embed(&self, texts: &[String]) -> Result<EmbeddingResponse, BlazenError> {
        let resp = EmbedModel::embed(self, texts)
            .await
            .map_err(|e| BlazenError::provider("embed", e.to_string()))?;

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
