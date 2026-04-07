//! Node wrapper for `Citation`.

use napi_derive::napi;

use blazen_llm::Citation;

/// A web/document citation backing a model statement.
///
/// Populated by Perplexity (`citations` array), Gemini (`groundingMetadata`),
/// and any provider that returns retrieval-augmented citations.
#[napi(object)]
pub struct JsCitation {
    /// The cited URL.
    pub url: String,
    /// Document or web-page title, if available.
    pub title: Option<String>,
    /// Excerpt/snippet from the source.
    pub snippet: Option<String>,
    /// Byte offset in the response text where this citation starts.
    pub start: Option<i64>,
    /// Byte offset in the response text where this citation ends.
    pub end: Option<i64>,
    /// Optional document id (for retrieval-augmented citations).
    #[napi(js_name = "documentId")]
    pub document_id: Option<String>,
    /// Provider-specific extra fields preserved as JSON.
    pub metadata: serde_json::Value,
}

impl From<&Citation> for JsCitation {
    #[allow(clippy::cast_possible_wrap)]
    fn from(c: &Citation) -> Self {
        Self {
            url: c.url.clone(),
            title: c.title.clone(),
            snippet: c.snippet.clone(),
            start: c.start.map(|v| v as i64),
            end: c.end.map(|v| v as i64),
            document_id: c.document_id.clone(),
            metadata: c.metadata.clone(),
        }
    }
}

impl From<Citation> for JsCitation {
    #[allow(clippy::cast_possible_wrap)]
    fn from(c: Citation) -> Self {
        Self {
            url: c.url,
            title: c.title,
            snippet: c.snippet,
            start: c.start.map(|v| v as i64),
            end: c.end.map(|v| v as i64),
            document_id: c.document_id,
            metadata: c.metadata,
        }
    }
}
