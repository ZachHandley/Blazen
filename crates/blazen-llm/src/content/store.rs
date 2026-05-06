//! [`ContentStore`] trait and supporting types.
//!
//! A `ContentStore` is the per-conversation registry that holds the bytes
//! (or a reference to where the bytes live) for every multimodal
//! [`ContentPart`] that flows through a Blazen conversation. It hands out
//! [`ContentHandle`]s to callers and resolves them back to a wire-renderable
//! [`MediaSource`] when the request builder needs to serialize them.
//!
//! The store sits at the boundary between Blazen's typed content model and
//! whatever physical backend the user wants to use:
//!
//! - In-memory `HashMap<id, Bytes>` for ephemeral / test / small workloads.
//! - On-disk directory for persistence between turns or across restarts.
//! - Provider file APIs (`OpenAI` Files, Anthropic Files, Gemini Files, fal
//!   storage) for large content the model needs to reference natively.
//! - Custom user storage (S3, GCS, R2, Postgres LO, etc.) via the
//!   user-supplied callback variant.
//!
//! This module defines only the trait and shared scaffolding types.
//! Concrete implementations live under [`super::stores`].
//!
//! [`ContentPart`]: crate::types::ContentPart
//! [`MediaSource`]: crate::types::MediaSource
//! [`ContentHandle`]: super::handle::ContentHandle

use std::path::PathBuf;
use std::sync::Arc;

use async_trait::async_trait;
use bytes::Bytes;
use serde::{Deserialize, Serialize};

use super::handle::ContentHandle;
use super::kind::ContentKind;
use crate::error::BlazenError;
use crate::types::{MediaSource, ProviderId};

// ---------------------------------------------------------------------------
// ByteStream — pinned, boxed, fallible stream of byte chunks
// ---------------------------------------------------------------------------

/// A pinned, boxed, fallible stream of byte chunks.
///
/// Used by [`ContentBody::Stream`] for streaming uploads and by
/// [`ContentStore::fetch_stream`] for streaming downloads. Stores backed by
/// HTTP, S3, or the filesystem should produce / consume these incrementally;
/// memory-bound stores may buffer.
pub type ByteStream =
    std::pin::Pin<Box<dyn futures_core::Stream<Item = Result<Bytes, BlazenError>> + Send>>;

// ---------------------------------------------------------------------------
// ContentBody — what a caller provides to `put`
// ---------------------------------------------------------------------------

/// The payload supplied to [`ContentStore::put`].
///
/// Different stores support different inputs natively; the `Bytes` and
/// `LocalPath` variants are universally supported, while `Url` and
/// `ProviderFile` are intended for stores that can record a reference
/// without copying the bytes.
///
/// `Clone` is implemented manually because the [`Stream`](Self::Stream)
/// variant carries a non-clonable [`ByteStream`]; cloning a `Stream`
/// variant panics with `unreachable!` and is a programmer error — consume
/// streaming bodies by value.
#[derive(Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ContentBody {
    /// Inline byte payload. Universal — every store accepts this.
    Bytes {
        /// Raw bytes that make up the content.
        data: Vec<u8>,
    },
    /// Public URL. Stores that record references (e.g.
    /// [`InMemoryContentStore`](super::stores::InMemoryContentStore)) keep
    /// the URL and resolve directly to it; stores that materialize bytes
    /// (e.g. [`LocalFileContentStore`](super::stores::LocalFileContentStore))
    /// fetch and persist locally.
    Url {
        /// Public URL to the content.
        url: String,
    },
    /// Local filesystem path. Native-target stores treat this as a
    /// reference; stores that need to upload (provider file APIs) read the
    /// bytes during `put`.
    LocalPath {
        /// Path to the file on the local filesystem.
        path: PathBuf,
    },
    /// Reference to content already in a provider's file API. Stores
    /// generally record this verbatim and resolve back to a matching
    /// [`MediaSource::ProviderFile`].
    ProviderFile {
        /// Provider that owns the file ID.
        provider: ProviderId,
        /// Provider-issued file identifier.
        id: String,
    },
    /// Streaming byte source. The store consumes the stream during `put`
    /// and is free to spool to disk, forward as a chunked upload, or
    /// drain into bytes.
    ///
    /// Stores that have a true streaming upload path (filesystem, S3,
    /// HTTP multipart) should consume the stream incrementally;
    /// memory-bound stores may buffer.
    ///
    /// `#[serde(skip)]` excludes this variant from (de)serialization —
    /// [`ByteStream`] is neither `Serialize` nor `Deserialize`. Bindings
    /// that route `ContentBody` through `serde_json` must check for
    /// `Stream` first and handle it on a separate path.
    #[serde(skip)]
    Stream {
        /// The byte stream to be consumed by the store.
        stream: ByteStream,
        /// Hint for the total length when known up front (e.g. from a
        /// `Content-Length` header). Stores can use this to pre-allocate
        /// or to choose between simple and resumable upload paths.
        size_hint: Option<u64>,
    },
}

impl Clone for ContentBody {
    fn clone(&self) -> Self {
        match self {
            Self::Bytes { data } => Self::Bytes { data: data.clone() },
            Self::Url { url } => Self::Url { url: url.clone() },
            Self::LocalPath { path } => Self::LocalPath { path: path.clone() },
            Self::ProviderFile { provider, id } => Self::ProviderFile {
                provider: *provider,
                id: id.clone(),
            },
            Self::Stream { .. } => {
                unreachable!("ContentBody::Stream is not clonable; consume by value")
            }
        }
    }
}

impl std::fmt::Debug for ContentBody {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Bytes { data } => f
                .debug_struct("Bytes")
                .field("data_len", &data.len())
                .finish(),
            Self::Url { url } => f.debug_struct("Url").field("url", url).finish(),
            Self::LocalPath { path } => f.debug_struct("LocalPath").field("path", path).finish(),
            Self::ProviderFile { provider, id } => f
                .debug_struct("ProviderFile")
                .field("provider", provider)
                .field("id", id)
                .finish(),
            Self::Stream { size_hint, .. } => f
                .debug_struct("Stream")
                .field("size_hint", size_hint)
                .finish_non_exhaustive(),
        }
    }
}

// ---------------------------------------------------------------------------
// ContentHint — caller-supplied metadata for `put`
// ---------------------------------------------------------------------------

/// Caller-supplied hints that the store uses when assigning a
/// [`ContentHandle`] during [`ContentStore::put`].
///
/// All fields are optional; the store may auto-detect them from bytes
/// (via [`super::detect`]) when not provided.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ContentHint {
    /// MIME type, if known.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mime_type: Option<String>,
    /// Caller's preferred classification — overrides any automatic
    /// detection from bytes / extension / MIME.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub kind_hint: Option<ContentKind>,
    /// Human-readable display name (filename, caption, etc.).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub display_name: Option<String>,
    /// Byte size, if known up-front.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub byte_size: Option<u64>,
}

impl ContentHint {
    /// Builder: attach a MIME hint.
    #[must_use]
    pub fn with_mime_type(mut self, mime: impl Into<String>) -> Self {
        self.mime_type = Some(mime.into());
        self
    }

    /// Builder: attach a kind override.
    #[must_use]
    pub fn with_kind(mut self, kind: ContentKind) -> Self {
        self.kind_hint = Some(kind);
        self
    }

    /// Builder: attach a display name.
    #[must_use]
    pub fn with_display_name(mut self, name: impl Into<String>) -> Self {
        self.display_name = Some(name.into());
        self
    }

    /// Builder: attach a byte size.
    #[must_use]
    pub fn with_byte_size(mut self, size: u64) -> Self {
        self.byte_size = Some(size);
        self
    }
}

// ---------------------------------------------------------------------------
// ContentMetadata — cheap-to-fetch summary
// ---------------------------------------------------------------------------

/// Summary metadata about stored content, returned by
/// [`ContentStore::metadata`] without materializing the bytes.
#[derive(Debug, Clone)]
pub struct ContentMetadata {
    pub kind: ContentKind,
    pub mime_type: Option<String>,
    pub byte_size: Option<u64>,
    pub display_name: Option<String>,
}

// ---------------------------------------------------------------------------
// ContentStore trait
// ---------------------------------------------------------------------------

/// The pluggable content registry that backs handle resolution.
///
/// Implementations are expected to be cheap to clone (typically
/// `Arc<Inner>`) and `Send + Sync` so they can be shared across the
/// agent runner, the request builders, and any tool execution context.
///
/// Lifecycle:
/// 1. [`put`](Self::put) — caller hands bytes / a reference to the store
///    and gets a [`ContentHandle`] back. The handle carries kind +
///    metadata so it can be type-checked at the tool boundary without
///    calling back into the store.
/// 2. [`resolve`](Self::resolve) — request builder asks the store for the
///    cheapest wire-renderable form (URL, base64, local path, provider
///    file ID). Stores prefer references over inline bytes when possible
///    to keep the wire payload small.
/// 3. [`fetch_bytes`](Self::fetch_bytes) — tool implementations that
///    actually need the raw bytes call this. Most do not; the model
///    typically reasons over the handle and the tool dispatches using the
///    resolved [`MediaSource`].
/// 4. [`metadata`](Self::metadata) — cheap path for systems that need to
///    introspect a handle without dereferencing.
/// 5. [`delete`](Self::delete) — optional cleanup. Default is a no-op.
#[async_trait]
pub trait ContentStore: Send + Sync + std::fmt::Debug {
    /// Persist content and return a freshly-issued handle.
    async fn put(&self, body: ContentBody, hint: ContentHint)
    -> Result<ContentHandle, BlazenError>;

    /// Resolve a handle to a wire-renderable [`MediaSource`].
    ///
    /// Stores should return the cheapest representation they can produce:
    /// `Url` > `ProviderFile` > `Base64` > `File`. Callers serialize
    /// whatever comes back via the existing per-provider renderers.
    async fn resolve(&self, handle: &ContentHandle) -> Result<MediaSource, BlazenError>;

    /// Fetch raw bytes. Tools that need to operate on the actual content
    /// (e.g. transcribe audio, parse a PDF) call this.
    async fn fetch_bytes(&self, handle: &ContentHandle) -> Result<Vec<u8>, BlazenError>;

    /// Cheap metadata lookup without materializing bytes.
    ///
    /// Default impl resolves the handle and reports what the resolved
    /// source can tell us; stores that track richer metadata should
    /// override.
    async fn metadata(&self, handle: &ContentHandle) -> Result<ContentMetadata, BlazenError> {
        let _ = self.resolve(handle).await?;
        Ok(ContentMetadata {
            kind: handle.kind,
            mime_type: handle.mime_type.clone(),
            byte_size: handle.byte_size,
            display_name: handle.display_name.clone(),
        })
    }

    /// Stream raw bytes back. Default impl buffers `fetch_bytes` into a
    /// single chunk so existing impls keep working. Stores backed by
    /// HTTP / S3 / disk should override to actually stream.
    async fn fetch_stream(&self, handle: &ContentHandle) -> Result<ByteStream, BlazenError> {
        let bytes = self.fetch_bytes(handle).await?;
        Ok(Box::pin(futures_util::stream::once(async move {
            Ok(Bytes::from(bytes))
        })))
    }

    /// Optional cleanup hook. Default: no-op.
    async fn delete(&self, _handle: &ContentHandle) -> Result<(), BlazenError> {
        Ok(())
    }
}

/// Type-erased reference to a [`ContentStore`].
///
/// The agent runner and request builders accept `Arc<dyn ContentStore>` so
/// that the same store can be shared across many call sites.
pub type DynContentStore = Arc<dyn ContentStore>;
