//! Node wrapper for `Citation`.
//!
//! Exposes both a plain `#[napi(object)]` shape ([`JsCitation`]) used inside
//! `#[napi(object)]` response types, and a richer class wrapper
//! ([`JsCitationClass`], JS name `Citation`) with getters and a constructor
//! for direct construction from JavaScript.

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

// ---------------------------------------------------------------------------
// JsCitationClass — class wrapper exposed to JS as `Citation`
// ---------------------------------------------------------------------------

/// Options for constructing a [`JsCitationClass`] from JavaScript.
#[napi(object)]
pub struct CitationOptions {
    /// The cited URL.
    pub url: String,
    pub title: Option<String>,
    pub snippet: Option<String>,
    pub start: Option<i64>,
    pub end: Option<i64>,
    #[napi(js_name = "documentId")]
    pub document_id: Option<String>,
    pub metadata: Option<serde_json::Value>,
}

/// Class wrapper around [`Citation`].
///
/// Provides getters and a constructor for callers that prefer working with
/// class instances over plain object dicts.
///
/// ```typescript
/// import { Citation } from 'blazen';
///
/// const c = new Citation({ url: "https://example.com", title: "Example" });
/// console.log(c.url, c.title);
/// ```
#[napi(js_name = "Citation")]
pub struct JsCitationClass {
    pub(crate) inner: Citation,
}

#[napi]
#[allow(
    clippy::must_use_candidate,
    clippy::missing_errors_doc,
    clippy::needless_pass_by_value,
    clippy::cast_possible_wrap,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]
impl JsCitationClass {
    /// Construct a citation. Most callers receive these via
    /// `CompletionResponse.citations` rather than building them by hand.
    #[napi(constructor)]
    pub fn new(options: CitationOptions) -> Self {
        Self {
            inner: Citation {
                url: options.url,
                title: options.title,
                snippet: options.snippet,
                start: options.start.map(|v| v as usize),
                end: options.end.map(|v| v as usize),
                document_id: options.document_id,
                metadata: options.metadata.unwrap_or(serde_json::Value::Null),
            },
        }
    }

    /// The cited URL.
    #[napi(getter)]
    pub fn url(&self) -> String {
        self.inner.url.clone()
    }

    /// Optional document or page title.
    #[napi(getter)]
    pub fn title(&self) -> Option<String> {
        self.inner.title.clone()
    }

    /// Optional excerpt from the cited source.
    #[napi(getter)]
    pub fn snippet(&self) -> Option<String> {
        self.inner.snippet.clone()
    }

    /// Byte offset in the response text where this citation begins.
    #[napi(getter)]
    pub fn start(&self) -> Option<i64> {
        self.inner.start.map(|v| v as i64)
    }

    /// Byte offset in the response text where this citation ends.
    #[napi(getter)]
    pub fn end(&self) -> Option<i64> {
        self.inner.end.map(|v| v as i64)
    }

    /// Optional document identifier for retrieval-augmented citations.
    #[napi(getter, js_name = "documentId")]
    pub fn document_id(&self) -> Option<String> {
        self.inner.document_id.clone()
    }

    /// Provider-specific extra fields preserved as JSON.
    #[napi(getter)]
    pub fn metadata(&self) -> serde_json::Value {
        self.inner.metadata.clone()
    }
}

impl From<Citation> for JsCitationClass {
    fn from(inner: Citation) -> Self {
        Self { inner }
    }
}

impl From<&Citation> for JsCitationClass {
    fn from(inner: &Citation) -> Self {
        Self {
            inner: inner.clone(),
        }
    }
}
