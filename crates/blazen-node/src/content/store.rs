//! [`JsContentStore`] — napi class wrapping
//! [`Arc<dyn blazen_llm::content::ContentStore>`].
//!
//! Exposes the lifecycle methods (`put`, `resolve`, `fetchBytes`,
//! `metadata`, `delete`) plus static factories for every built-in
//! implementation:
//!
//! - `ContentStore.inMemory()` — ephemeral process-local map.
//! - `ContentStore.localFile(root)` — on-disk directory.
//! - `ContentStore.openaiFiles(apiKey, baseUrl?)` — `OpenAI` Files API.
//! - `ContentStore.anthropicFiles(apiKey, baseUrl?)` — Anthropic Files API.
//! - `ContentStore.geminiFiles(apiKey, baseUrl?)` — Gemini Files API.
//! - `ContentStore.falStorage(apiKey, baseUrl?)` — fal.ai storage.
//!
//! All async methods return real `Promise<T>` to JS via napi-rs's `async fn`
//! support, matching the pattern used by [`crate::agent::run_agent`].

use std::path::PathBuf;
use std::sync::Arc;

use blazen_llm::content::{
    AnthropicFilesStore, ContentBody, ContentHint, ContentStore, FalStorageStore, GeminiFilesStore,
    InMemoryContentStore, OpenAiFilesStore,
};
use blazen_llm::types::MediaSource;
use napi::Either;
use napi::bindgen_prelude::{Buffer, Result};
use napi_derive::napi;

use crate::error::llm_error_to_napi;

use super::handle::JsContentHandle;
use super::kind::JsContentKind;

// ---------------------------------------------------------------------------
// PutOptions — caller-supplied metadata for `put`
// ---------------------------------------------------------------------------

/// Optional hints attached to a `put` call.
///
/// Mirrors [`blazen_llm::content::ContentHint`] minus its builder API. Every
/// field is optional; the store may auto-detect from the bytes when a hint is
/// missing.
#[napi(object)]
pub struct PutOptions {
    /// MIME type, if known.
    #[napi(js_name = "mimeType")]
    pub mime_type: Option<String>,
    /// Caller's preferred classification — overrides any auto-detection.
    pub kind: Option<JsContentKind>,
    /// Human-readable display name (filename, caption).
    #[napi(js_name = "displayName")]
    pub display_name: Option<String>,
    /// Byte size, if known up-front. `i64` since napi has no `u64`.
    #[napi(js_name = "byteSize")]
    pub byte_size: Option<i64>,
}

impl PutOptions {
    fn into_hint(self) -> ContentHint {
        let mut hint = ContentHint::default();
        if let Some(mime) = self.mime_type {
            hint = hint.with_mime_type(mime);
        }
        if let Some(kind) = self.kind {
            hint = hint.with_kind(kind.into());
        }
        if let Some(name) = self.display_name {
            hint = hint.with_display_name(name);
        }
        if let Some(size) = self.byte_size {
            #[allow(clippy::cast_sign_loss)]
            {
                hint = hint.with_byte_size(size as u64);
            }
        }
        hint
    }
}

// ---------------------------------------------------------------------------
// JsContentMetadata
// ---------------------------------------------------------------------------

/// Cheap metadata summary returned by
/// [`JsContentStore::metadata`](JsContentStore::metadata).
#[napi(object)]
pub struct JsContentMetadata {
    pub kind: JsContentKind,
    #[napi(js_name = "mimeType")]
    pub mime_type: Option<String>,
    #[napi(js_name = "byteSize")]
    pub byte_size: Option<i64>,
    #[napi(js_name = "displayName")]
    pub display_name: Option<String>,
}

// ---------------------------------------------------------------------------
// Body parsing
// ---------------------------------------------------------------------------

/// Decide whether a string body is a URL or a local filesystem path.
///
/// A URL is anything containing `"://"`; everything else is treated as a
/// local path so callers can drop in a relative or absolute filename without
/// extra ceremony.
fn parse_string_body(s: String) -> ContentBody {
    if s.contains("://") {
        ContentBody::Url(s)
    } else {
        ContentBody::LocalPath(PathBuf::from(s))
    }
}

// ---------------------------------------------------------------------------
// JsContentStore
// ---------------------------------------------------------------------------

/// Pluggable registry for multimodal content. Wraps
/// [`Arc<dyn blazen_llm::content::ContentStore>`].
///
/// Construct via the static factories (e.g. `ContentStore.inMemory()`).
/// Stores are cheap to clone — internally an `Arc` — so passing the same
/// instance across multiple agents / requests is fine.
#[napi(js_name = "ContentStore")]
pub struct JsContentStore {
    inner: Arc<dyn ContentStore>,
}

impl JsContentStore {
    /// Build a JS-side wrapper from any `ContentStore` implementation.
    pub fn from_arc(inner: Arc<dyn ContentStore>) -> Self {
        Self { inner }
    }

    /// Borrow the underlying store as `Arc<dyn ContentStore>`.
    #[must_use]
    pub fn as_arc(&self) -> Arc<dyn ContentStore> {
        Arc::clone(&self.inner)
    }
}

#[napi]
#[allow(clippy::missing_errors_doc, clippy::needless_pass_by_value)]
impl JsContentStore {
    // -----------------------------------------------------------------
    // Static factories — one per built-in store
    // -----------------------------------------------------------------

    /// Build a default ephemeral in-memory store.
    #[napi(factory, js_name = "inMemory")]
    #[must_use]
    pub fn in_memory() -> Self {
        Self::from_arc(Arc::new(InMemoryContentStore::new()))
    }

    /// Build a filesystem-backed store rooted at `root`. The directory is
    /// created if it doesn't yet exist.
    #[napi(factory, js_name = "localFile")]
    pub fn local_file(root: String) -> Result<Self> {
        let store = blazen_llm::content::LocalFileContentStore::new(PathBuf::from(root))
            .map_err(llm_error_to_napi)?;
        Ok(Self::from_arc(Arc::new(store)))
    }

    /// Build a store backed by the `OpenAI` Files API.
    #[napi(factory, js_name = "openaiFiles")]
    #[must_use]
    pub fn openai_files(api_key: String, base_url: Option<String>) -> Self {
        let mut store = OpenAiFilesStore::new(api_key);
        if let Some(url) = base_url {
            store = store.with_base_url(url);
        }
        Self::from_arc(Arc::new(store))
    }

    /// Build a store backed by the Anthropic Files API.
    #[napi(factory, js_name = "anthropicFiles")]
    #[must_use]
    pub fn anthropic_files(api_key: String, base_url: Option<String>) -> Self {
        let mut store = AnthropicFilesStore::new(api_key);
        if let Some(url) = base_url {
            store = store.with_base_url(url);
        }
        Self::from_arc(Arc::new(store))
    }

    /// Build a store backed by the Gemini Files API.
    #[napi(factory, js_name = "geminiFiles")]
    #[must_use]
    pub fn gemini_files(api_key: String, base_url: Option<String>) -> Self {
        let mut store = GeminiFilesStore::new(api_key);
        if let Some(url) = base_url {
            store = store.with_base_url(url);
        }
        Self::from_arc(Arc::new(store))
    }

    /// Build a store backed by fal.ai's storage API.
    #[napi(factory, js_name = "falStorage")]
    #[must_use]
    pub fn fal_storage(api_key: String, base_url: Option<String>) -> Self {
        let mut store = FalStorageStore::new(api_key);
        if let Some(url) = base_url {
            store = store.with_base_url(url);
        }
        Self::from_arc(Arc::new(store))
    }

    // -----------------------------------------------------------------
    // Lifecycle methods
    // -----------------------------------------------------------------

    /// Persist content and return a freshly-issued handle.
    ///
    /// `body` is either:
    /// - a `Buffer` — inline bytes uploaded to the store, or
    /// - a `string` — interpreted as a URL when it contains `"://"` (the
    ///   store records the reference) and as a local filesystem path
    ///   otherwise (the store reads or copies the file as needed).
    #[napi]
    pub async fn put(
        &self,
        body: Either<Buffer, String>,
        options: PutOptions,
    ) -> Result<JsContentHandle> {
        let content_body = match body {
            Either::A(buf) => ContentBody::Bytes(buf.to_vec()),
            Either::B(s) => parse_string_body(s),
        };
        let hint = options.into_hint();
        let handle = self
            .inner
            .put(content_body, hint)
            .await
            .map_err(llm_error_to_napi)?;
        Ok(JsContentHandle::from_rust(&handle))
    }

    /// Resolve a handle to a wire-renderable [`MediaSource`] (returned as a
    /// JS object — the same JSON shape Blazen's request builders accept).
    #[napi]
    pub async fn resolve(&self, handle: JsContentHandle) -> Result<serde_json::Value> {
        let rust_handle = handle.to_rust();
        let source: MediaSource = self
            .inner
            .resolve(&rust_handle)
            .await
            .map_err(llm_error_to_napi)?;
        serde_json::to_value(&source).map_err(|e| {
            napi::Error::new(
                napi::Status::GenericFailure,
                format!("failed to serialize MediaSource: {e}"),
            )
        })
    }

    /// Fetch raw bytes for a handle. Tools that need to operate on the
    /// actual content (parse a PDF, transcribe audio) call this; most tools
    /// reason over the handle and let `resolve` produce the wire form.
    #[napi(js_name = "fetchBytes")]
    pub async fn fetch_bytes(&self, handle: JsContentHandle) -> Result<Buffer> {
        let rust_handle = handle.to_rust();
        let bytes = self
            .inner
            .fetch_bytes(&rust_handle)
            .await
            .map_err(llm_error_to_napi)?;
        Ok(Buffer::from(bytes))
    }

    /// Cheap metadata lookup without materializing the bytes.
    #[napi]
    pub async fn metadata(&self, handle: JsContentHandle) -> Result<JsContentMetadata> {
        let rust_handle = handle.to_rust();
        let meta = self
            .inner
            .metadata(&rust_handle)
            .await
            .map_err(llm_error_to_napi)?;
        #[allow(clippy::cast_possible_wrap)]
        Ok(JsContentMetadata {
            kind: meta.kind.into(),
            mime_type: meta.mime_type,
            byte_size: meta.byte_size.map(|n| n as i64),
            display_name: meta.display_name,
        })
    }

    /// Optional cleanup — remove the handle from the store. Default
    /// implementations on most stores are no-ops.
    #[napi]
    pub async fn delete(&self, handle: JsContentHandle) -> Result<()> {
        let rust_handle = handle.to_rust();
        self.inner
            .delete(&rust_handle)
            .await
            .map_err(llm_error_to_napi)?;
        Ok(())
    }
}
