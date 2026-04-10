//! Bridge between [`blazen_embed_fastembed::FastEmbedModel`] and
//! [`crate::EmbeddingModel`].

use async_trait::async_trait;
use blazen_embed_fastembed::FastEmbedModel;

use crate::error::BlazenError;
use crate::types::EmbeddingResponse;

#[async_trait]
impl crate::traits::EmbeddingModel for FastEmbedModel {
    fn model_id(&self) -> &str {
        FastEmbedModel::model_id(self)
    }

    fn dimensions(&self) -> usize {
        FastEmbedModel::dimensions(self)
    }

    async fn embed(&self, texts: &[String]) -> Result<EmbeddingResponse, BlazenError> {
        let resp = FastEmbedModel::embed(self, texts)
            .await
            .map_err(|e| BlazenError::provider("fastembed", e.to_string()))?;

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
