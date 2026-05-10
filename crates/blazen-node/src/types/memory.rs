//! Memory store and backend bindings for the Node.js SDK.
//!
//! Wraps `blazen-memory` types and backends for use from JavaScript/TypeScript.

use std::sync::Arc;

use async_trait::async_trait;
use napi::bindgen_prelude::Result;
use napi_derive::napi;

use blazen_memory::Memory;
use blazen_memory::backends::inmemory::InMemoryBackend;
#[cfg(not(target_os = "wasi"))]
use blazen_memory::backends::jsonl::JsonlBackend;
use blazen_memory::search::{
    compute_elid_similarity as core_compute_elid_similarity,
    compute_embedding_simhash_similarity as core_compute_embedding_simhash_similarity,
    compute_text_simhash_similarity as core_compute_text_simhash_similarity,
    simhash_from_hex as core_simhash_from_hex, simhash_to_hex as core_simhash_to_hex,
};
use blazen_memory::store::{MemoryBackend, MemoryStore};
use blazen_memory::types::StoredEntry;
use blazen_memory_valkey::UpstashBackend;
#[cfg(not(target_os = "wasi"))]
use blazen_memory_valkey::ValkeyBackend;

use super::embedding::JsEmbeddingModel;
use crate::error::{memory_error_to_napi, to_napi_error};

// ---------------------------------------------------------------------------
// Backend wrapper enum -- dispatches MemoryBackend through the three variants
// ---------------------------------------------------------------------------

/// Internal enum that implements [`MemoryBackend`] by dispatching to one of
/// the concrete backend types. Required because napi constructors cannot accept
/// trait objects.
enum BackendKind {
    InMemory(Arc<InMemoryBackend>),
    #[cfg(not(target_os = "wasi"))]
    Jsonl(Arc<JsonlBackend>),
    #[cfg(not(target_os = "wasi"))]
    Valkey(Arc<ValkeyBackend>),
    Upstash(Arc<UpstashBackend>),
}

#[async_trait]
impl MemoryBackend for BackendKind {
    async fn put(&self, entry: StoredEntry) -> blazen_memory::error::Result<()> {
        match self {
            Self::InMemory(b) => b.put(entry).await,
            #[cfg(not(target_os = "wasi"))]
            Self::Jsonl(b) => b.put(entry).await,
            #[cfg(not(target_os = "wasi"))]
            Self::Valkey(b) => b.put(entry).await,
            Self::Upstash(b) => b.put(entry).await,
        }
    }

    async fn get(&self, id: &str) -> blazen_memory::error::Result<Option<StoredEntry>> {
        match self {
            Self::InMemory(b) => b.get(id).await,
            #[cfg(not(target_os = "wasi"))]
            Self::Jsonl(b) => b.get(id).await,
            #[cfg(not(target_os = "wasi"))]
            Self::Valkey(b) => b.get(id).await,
            Self::Upstash(b) => b.get(id).await,
        }
    }

    async fn delete(&self, id: &str) -> blazen_memory::error::Result<bool> {
        match self {
            Self::InMemory(b) => b.delete(id).await,
            #[cfg(not(target_os = "wasi"))]
            Self::Jsonl(b) => b.delete(id).await,
            #[cfg(not(target_os = "wasi"))]
            Self::Valkey(b) => b.delete(id).await,
            Self::Upstash(b) => b.delete(id).await,
        }
    }

    async fn list(&self) -> blazen_memory::error::Result<Vec<StoredEntry>> {
        match self {
            Self::InMemory(b) => b.list().await,
            #[cfg(not(target_os = "wasi"))]
            Self::Jsonl(b) => b.list().await,
            #[cfg(not(target_os = "wasi"))]
            Self::Valkey(b) => b.list().await,
            Self::Upstash(b) => b.list().await,
        }
    }

    async fn len(&self) -> blazen_memory::error::Result<usize> {
        match self {
            Self::InMemory(b) => b.len().await,
            #[cfg(not(target_os = "wasi"))]
            Self::Jsonl(b) => b.len().await,
            #[cfg(not(target_os = "wasi"))]
            Self::Valkey(b) => b.len().await,
            Self::Upstash(b) => b.len().await,
        }
    }

    async fn search_by_bands(
        &self,
        bands: &[String],
        limit: usize,
    ) -> blazen_memory::error::Result<Vec<StoredEntry>> {
        match self {
            Self::InMemory(b) => b.search_by_bands(bands, limit).await,
            #[cfg(not(target_os = "wasi"))]
            Self::Jsonl(b) => b.search_by_bands(bands, limit).await,
            #[cfg(not(target_os = "wasi"))]
            Self::Valkey(b) => b.search_by_bands(bands, limit).await,
            Self::Upstash(b) => b.search_by_bands(bands, limit).await,
        }
    }
}

// ---------------------------------------------------------------------------
// JsMemoryBackend (subclassable base)
// ---------------------------------------------------------------------------

/// Base class for custom memory storage backends.
///
/// Extend and override all methods to implement a custom backend
/// (e.g. `PostgreSQL`, `DynamoDB`, `SQLite`).
///
/// ```javascript
/// class PostgresBackend extends MemoryBackend {
///     async put(entry) { /* ... */ }
///     async get(id) { /* ... */ }
///     async delete(id) { /* ... */ }
///     async list() { /* ... */ }
///     async len() { /* ... */ }
///     async searchByBands(bands, limit) { /* ... */ }
/// }
/// ```
#[napi(js_name = "MemoryBackend")]
pub struct JsMemoryBackend {}

#[napi]
#[allow(
    clippy::new_without_default,
    clippy::must_use_candidate,
    clippy::missing_errors_doc,
    clippy::unused_async,
    clippy::needless_pass_by_value
)]
impl JsMemoryBackend {
    /// Create a new memory backend base instance.
    #[napi(constructor)]
    pub fn new() -> Self {
        Self {}
    }

    /// Store an entry. Subclasses **must** override this method.
    #[napi]
    pub async fn put(&self, _entry: serde_json::Value) -> Result<()> {
        Err(napi::Error::from_reason("subclass must override put()"))
    }

    /// Retrieve an entry by ID. Subclasses **must** override this method.
    #[napi]
    pub async fn get(&self, _id: String) -> Result<Option<serde_json::Value>> {
        Err(napi::Error::from_reason("subclass must override get()"))
    }

    /// Delete an entry by ID. Subclasses **must** override this method.
    #[napi]
    pub async fn delete(&self, _id: String) -> Result<bool> {
        Err(napi::Error::from_reason("subclass must override delete()"))
    }

    /// List all stored entries. Subclasses **must** override this method.
    #[napi]
    pub async fn list(&self) -> Result<Vec<serde_json::Value>> {
        Err(napi::Error::from_reason("subclass must override list()"))
    }

    /// Return the number of stored entries. Subclasses **must** override this method.
    #[napi]
    pub async fn len(&self) -> Result<u32> {
        Err(napi::Error::from_reason("subclass must override len()"))
    }

    /// Search for entries whose LSH bands overlap with the given bands.
    /// Subclasses **must** override this method.
    #[napi(js_name = "searchByBands")]
    pub async fn search_by_bands(
        &self,
        _bands: Vec<String>,
        _limit: u32,
    ) -> Result<Vec<serde_json::Value>> {
        Err(napi::Error::from_reason(
            "subclass must override searchByBands()",
        ))
    }
}

// ---------------------------------------------------------------------------
// JsMemoryStore (subclassable base)
// ---------------------------------------------------------------------------

/// Base class for custom high-level memory stores.
///
/// Mirrors [`blazen_memory::store::MemoryStore`]. The concrete
/// implementation that ships with Blazen is [`JsMemory`], which
/// handles embedding, ELID encoding, and search on top of a
/// [`JsMemoryBackend`]. Subclassing `MemoryStore` lets callers plug a
/// JS-side store (e.g. one that delegates to a managed cloud service
/// like `Pinecone` or `Weaviate`) into the same surface that
/// `Memory.local` and `Memory(embedder, backend)` expose.
///
/// Subclasses **must** override every method. Entries and results are
/// exchanged as plain JSON objects matching the
/// [`blazen_memory::MemoryEntry`] / [`blazen_memory::MemoryResult`]
/// shapes:
///
/// ```javascript
/// class CloudMemoryStore extends MemoryStore {
///     async add(entries) { /* ... */ }
///     async search(query, limit, metadataFilter) { /* ... */ }
///     async searchLocal(query, limit, metadataFilter) { /* ... */ }
///     async get(id) { /* ... */ }
///     async delete(id) { /* ... */ }
///     async len() { /* ... */ }
/// }
/// ```
#[napi(js_name = "MemoryStore")]
pub struct JsMemoryStore {}

#[napi]
#[allow(
    clippy::new_without_default,
    clippy::must_use_candidate,
    clippy::missing_errors_doc,
    clippy::unused_async,
    clippy::needless_pass_by_value
)]
impl JsMemoryStore {
    /// Create a new memory store base instance.
    #[napi(constructor)]
    pub fn new() -> Self {
        Self {}
    }

    /// Add one or more entries to the store. Subclasses **must**
    /// override this method.
    #[napi]
    pub async fn add(&self, _entries: Vec<serde_json::Value>) -> Result<Vec<String>> {
        Err(napi::Error::from_reason("subclass must override add()"))
    }

    /// Search using a query string with the configured embedding
    /// model. Subclasses **must** override this method.
    #[napi]
    pub async fn search(
        &self,
        _query: String,
        _limit: u32,
        _metadata_filter: Option<serde_json::Value>,
    ) -> Result<Vec<serde_json::Value>> {
        Err(napi::Error::from_reason("subclass must override search()"))
    }

    /// Search using only text-level `SimHash` (no embedding model
    /// required). Subclasses **must** override this method.
    #[napi(js_name = "searchLocal")]
    pub async fn search_local(
        &self,
        _query: String,
        _limit: u32,
        _metadata_filter: Option<serde_json::Value>,
    ) -> Result<Vec<serde_json::Value>> {
        Err(napi::Error::from_reason(
            "subclass must override searchLocal()",
        ))
    }

    /// Retrieve a single entry by id. Subclasses **must** override
    /// this method.
    #[napi]
    pub async fn get(&self, _id: String) -> Result<Option<serde_json::Value>> {
        Err(napi::Error::from_reason("subclass must override get()"))
    }

    /// Delete an entry by id. Subclasses **must** override this
    /// method.
    #[napi]
    pub async fn delete(&self, _id: String) -> Result<bool> {
        Err(napi::Error::from_reason("subclass must override delete()"))
    }

    /// Return the number of entries. Subclasses **must** override
    /// this method.
    #[napi]
    pub async fn len(&self) -> Result<u32> {
        Err(napi::Error::from_reason("subclass must override len()"))
    }

    /// Whether the store contains no entries. Default implementation
    /// returns `len() == 0`; subclasses may override for efficiency.
    #[napi(js_name = "isEmpty")]
    pub async fn is_empty(&self) -> Result<bool> {
        Err(napi::Error::from_reason(
            "subclass must override isEmpty() (or override len())",
        ))
    }
}

// ---------------------------------------------------------------------------
// JsInMemoryBackend
// ---------------------------------------------------------------------------

/// An in-memory backend for the memory store.
///
/// Data lives only as long as the process; nothing is persisted.
///
/// ```javascript
/// const backend = new InMemoryBackend();
/// const memory = new Memory(embedder, backend);
/// ```
#[napi(js_name = "InMemoryBackend")]
pub struct JsInMemoryBackend {
    inner: Arc<InMemoryBackend>,
}

#[napi]
#[allow(clippy::new_without_default, clippy::must_use_candidate)]
impl JsInMemoryBackend {
    /// Create a new, empty in-memory backend.
    #[napi(constructor)]
    pub fn new() -> Self {
        Self {
            inner: Arc::new(InMemoryBackend::new()),
        }
    }
}

// ---------------------------------------------------------------------------
// JsJsonlBackend
// ---------------------------------------------------------------------------

/// A JSONL file-backed backend for the memory store.
///
/// Loads entries from the file on creation, appends new entries, and rewrites
/// on updates/deletes.
///
/// ```javascript
/// const backend = await JsonlBackend.create("./memory.jsonl");
/// const memory = new Memory(embedder, backend);
/// ```
#[cfg(not(target_os = "wasi"))]
#[napi(js_name = "JsonlBackend")]
pub struct JsJsonlBackend {
    inner: Arc<JsonlBackend>,
}

#[cfg(not(target_os = "wasi"))]
#[napi]
#[allow(clippy::missing_errors_doc)]
impl JsJsonlBackend {
    /// Create a JSONL backend at the given file path.
    ///
    /// If the file exists, its contents are loaded. Otherwise an empty store
    /// is created and the file is written on the first insert.
    #[napi(factory)]
    pub async fn create(path: String) -> Result<Self> {
        let backend = JsonlBackend::new(&path).await.map_err(to_napi_error)?;
        Ok(Self {
            inner: Arc::new(backend),
        })
    }
}

// ---------------------------------------------------------------------------
// JsValkeyBackend
// ---------------------------------------------------------------------------

/// A Valkey/Redis-backed backend for the memory store.
///
/// ```javascript
/// const backend = ValkeyBackend.create("redis://localhost:6379");
/// const memory = new Memory(embedder, backend);
/// ```
#[cfg(not(target_os = "wasi"))]
#[napi(js_name = "ValkeyBackend")]
pub struct JsValkeyBackend {
    inner: Arc<ValkeyBackend>,
}

#[cfg(not(target_os = "wasi"))]
#[napi]
#[allow(clippy::missing_errors_doc, clippy::needless_pass_by_value)]
impl JsValkeyBackend {
    /// Create a Valkey backend connected to the given Redis/Valkey URL.
    ///
    /// The URL should be in standard Redis format, e.g. `redis://localhost:6379`.
    #[napi(factory)]
    pub fn create(url: String) -> Result<Self> {
        let backend = ValkeyBackend::new(&url).map_err(to_napi_error)?;
        Ok(Self {
            inner: Arc::new(backend),
        })
    }
}

// ---------------------------------------------------------------------------
// JsUpstashBackend
// ---------------------------------------------------------------------------

/// An Upstash Redis REST-backed backend for the memory store.
///
/// Wasi-compatible alternative to [`JsValkeyBackend`] for Cloudflare Workers,
/// Deno, and other wasi hosts that cannot use raw TCP. Talks to Upstash's
/// REST API over the host-registered HTTP client (set via
/// `setDefaultHttpClient`).
///
/// ```javascript
/// const backend = UpstashBackend.create("https://us1-merry-cat-32242.upstash.io", "AYAg...");
/// const memory = Memory.withUpstash(embedder, backend);
/// ```
#[napi(js_name = "UpstashBackend")]
pub struct JsUpstashBackend {
    inner: Arc<UpstashBackend>,
}

#[napi]
#[allow(clippy::must_use_candidate, clippy::needless_pass_by_value)]
impl JsUpstashBackend {
    /// Create an Upstash REST backend.
    ///
    /// `restUrl` is the Upstash REST endpoint (e.g.
    /// `https://us1-merry-cat-32242.upstash.io`). `restToken` is the REST
    /// token, sent as a `Bearer` token on every request. The HTTP client is
    /// resolved via `setDefaultHttpClient` — call that before issuing any
    /// memory operations.
    ///
    /// `prefix` overrides the default key prefix (`blazen:memory:`). Pass
    /// `null`/`undefined` for the default. Useful when running multiple
    /// logical stores against the same Upstash database.
    #[napi(factory)]
    pub fn create(rest_url: String, rest_token: String, prefix: Option<String>) -> Self {
        let http = default_http_client();
        let mut backend = UpstashBackend::new(rest_url, rest_token, http);
        if let Some(p) = prefix {
            backend = backend.with_prefix(p);
        }
        Self {
            inner: Arc::new(backend),
        }
    }
}

/// Resolve the platform-appropriate default HTTP client. Mirrors
/// `blazen_llm::default_http_client` (which is `pub(crate)`): on wasi we
/// build a [`blazen_llm::http_napi_wasi::LazyHttpClient`] (which defers to
/// `setDefaultHttpClient`), on native we build a stock
/// [`blazen_llm::ReqwestHttpClient`].
#[cfg(target_os = "wasi")]
fn default_http_client() -> Arc<dyn blazen_llm::http::HttpClient> {
    blazen_llm::http_napi_wasi::LazyHttpClient::new().into_arc()
}

#[cfg(not(target_os = "wasi"))]
fn default_http_client() -> Arc<dyn blazen_llm::http::HttpClient> {
    blazen_llm::ReqwestHttpClient::new().into_arc()
}

// ---------------------------------------------------------------------------
// JsRetryMemoryBackend
// ---------------------------------------------------------------------------

use crate::generated::JsRetryConfig;
use blazen_memory::RetryMemoryBackend;
use napi::Either;

/// Type alias for the napi-rs union accepted by `RetryMemoryBackend.wrap*`
/// constructors. napi-rs cannot accept trait objects across the FFI, so we
/// enumerate the concrete backend classes the binding ships. The wasi fork
/// substitutes `JsUpstashBackend` for the native `JsJsonlBackend` /
/// `JsValkeyBackend` pair (which both depend on local FS / raw TCP).
#[cfg(not(target_os = "wasi"))]
type AnyBackend<'a> = Either<
    &'a JsInMemoryBackend,
    Either<&'a JsJsonlBackend, Either<&'a JsValkeyBackend, &'a JsUpstashBackend>>,
>;

#[cfg(target_os = "wasi")]
type AnyBackend<'a> = Either<&'a JsInMemoryBackend, &'a JsUpstashBackend>;

/// A `MemoryBackend` decorator that retries transient errors with
/// exponential backoff.
///
/// Mirrors `RetryCompletionModel` for `MemoryBackend`. Use one of the
/// `wrapInMemory` / `wrapJsonl` / `wrapValkey` factories to wrap the
/// matching backend.
///
/// ```javascript
/// const inner = new InMemoryBackend();
/// const retried = RetryMemoryBackend.wrapInMemory(inner, { maxRetries: 5 });
/// ```
#[napi(js_name = "RetryMemoryBackend")]
pub struct JsRetryMemoryBackend {
    #[allow(dead_code)] // Held to keep the wrapped backend alive; future
    // memory ABI expansions will let callers pass this through to `Memory.local*`.
    inner: Arc<dyn MemoryBackend>,
}

#[napi]
#[allow(
    clippy::must_use_candidate,
    clippy::missing_errors_doc,
    clippy::needless_pass_by_value
)]
impl JsRetryMemoryBackend {
    /// Wrap an `InMemoryBackend` with retry-on-transient-error behaviour.
    #[napi(factory, js_name = "wrapInMemory")]
    pub fn wrap_in_memory(backend: &JsInMemoryBackend, config: Option<JsRetryConfig>) -> Self {
        let cfg = config.map(Into::into).unwrap_or_default();
        let wrapped =
            RetryMemoryBackend::new(Arc::clone(&backend.inner) as Arc<dyn MemoryBackend>, cfg);
        Self {
            inner: wrapped.into_arc(),
        }
    }
}

#[cfg(not(target_os = "wasi"))]
#[napi]
#[allow(
    clippy::must_use_candidate,
    clippy::missing_errors_doc,
    clippy::needless_pass_by_value
)]
impl JsRetryMemoryBackend {
    /// Wrap a `JsonlBackend` with retry-on-transient-error behaviour.
    #[napi(factory, js_name = "wrapJsonl")]
    pub fn wrap_jsonl(backend: &JsJsonlBackend, config: Option<JsRetryConfig>) -> Self {
        let cfg = config.map(Into::into).unwrap_or_default();
        let wrapped =
            RetryMemoryBackend::new(Arc::clone(&backend.inner) as Arc<dyn MemoryBackend>, cfg);
        Self {
            inner: wrapped.into_arc(),
        }
    }

    /// Wrap a `ValkeyBackend` with retry-on-transient-error behaviour.
    #[napi(factory, js_name = "wrapValkey")]
    pub fn wrap_valkey(backend: &JsValkeyBackend, config: Option<JsRetryConfig>) -> Self {
        let cfg = config.map(Into::into).unwrap_or_default();
        let wrapped =
            RetryMemoryBackend::new(Arc::clone(&backend.inner) as Arc<dyn MemoryBackend>, cfg);
        Self {
            inner: wrapped.into_arc(),
        }
    }

    /// Wrap an `UpstashBackend` with retry-on-transient-error behaviour.
    #[napi(factory, js_name = "wrapUpstash")]
    pub fn wrap_upstash(backend: &JsUpstashBackend, config: Option<JsRetryConfig>) -> Self {
        let cfg = config.map(Into::into).unwrap_or_default();
        let wrapped =
            RetryMemoryBackend::new(Arc::clone(&backend.inner) as Arc<dyn MemoryBackend>, cfg);
        Self {
            inner: wrapped.into_arc(),
        }
    }

    /// Generic factory accepting any of the four concrete backends. Useful
    /// when the caller doesn't statically know which backend is in hand.
    #[napi(factory, js_name = "wrap")]
    pub fn wrap(backend: AnyBackend<'_>, config: Option<JsRetryConfig>) -> Self {
        match backend {
            Either::A(b) => Self::wrap_in_memory(b, config),
            Either::B(Either::A(b)) => Self::wrap_jsonl(b, config),
            Either::B(Either::B(Either::A(b))) => Self::wrap_valkey(b, config),
            Either::B(Either::B(Either::B(b))) => Self::wrap_upstash(b, config),
        }
    }
}

#[cfg(target_os = "wasi")]
#[napi]
#[allow(
    clippy::must_use_candidate,
    clippy::missing_errors_doc,
    clippy::needless_pass_by_value
)]
impl JsRetryMemoryBackend {
    /// Wrap an `UpstashBackend` with retry-on-transient-error behaviour.
    #[napi(factory, js_name = "wrapUpstash")]
    pub fn wrap_upstash(backend: &JsUpstashBackend, config: Option<JsRetryConfig>) -> Self {
        let cfg = config.map(Into::into).unwrap_or_default();
        let wrapped =
            RetryMemoryBackend::new(Arc::clone(&backend.inner) as Arc<dyn MemoryBackend>, cfg);
        Self {
            inner: wrapped.into_arc(),
        }
    }

    /// Generic factory accepting either of the wasi-compatible backends.
    /// Useful when the caller doesn't statically know which backend is in hand.
    #[napi(factory, js_name = "wrap")]
    pub fn wrap(backend: AnyBackend<'_>, config: Option<JsRetryConfig>) -> Self {
        match backend {
            Either::A(b) => Self::wrap_in_memory(b, config),
            Either::B(b) => Self::wrap_upstash(b, config),
        }
    }
}

// ---------------------------------------------------------------------------
// Result / Entry objects
// ---------------------------------------------------------------------------

/// A search result from the memory store.
#[napi(object)]
pub struct JsMemoryResult {
    /// The entry id.
    pub id: String,
    /// The original text content.
    pub text: String,
    /// Similarity score in 0.0..=1.0 (higher = more similar).
    pub score: f64,
    /// Arbitrary user metadata.
    pub metadata: serde_json::Value,
}

/// A stored entry retrieved from the memory store.
#[napi(object)]
pub struct JsMemoryEntry {
    /// The entry id.
    pub id: String,
    /// The original text content.
    pub text: String,
    /// Arbitrary user metadata.
    pub metadata: serde_json::Value,
}

// ---------------------------------------------------------------------------
// JsMemory
// ---------------------------------------------------------------------------

/// A memory store that uses ELID for vector indexing and similarity search.
///
/// Supports two modes:
/// - **Full mode** (`new Memory(embedder, backend)`): embedding-based search
/// - **Local mode** (`Memory.local(backend)`): text `SimHash` only, no embedder
///
/// ```javascript
/// import { Memory, EmbeddingModel, InMemoryBackend } from 'blazen';
///
/// const embedder = EmbeddingModel.openai({ apiKey: key });
/// const memory = new Memory(embedder, new InMemoryBackend());
///
/// await memory.add("doc1", "Paris is the capital of France");
/// const results = await memory.search("What is France's capital?", 5);
/// console.log(results[0].text);
/// ```
#[napi(js_name = "Memory")]
pub struct JsMemory {
    inner: Memory,
}

#[napi]
#[allow(clippy::must_use_candidate, clippy::missing_errors_doc)]
impl JsMemory {
    /// Create a memory store with an embedding model for full ELID-based search.
    ///
    /// @param embedder - The embedding model to use for vectorisation.
    /// @param backend  - An `InMemoryBackend` instance.
    #[napi(constructor)]
    pub fn new(embedder: &JsEmbeddingModel, backend: &JsInMemoryBackend) -> Result<Self> {
        let arc = embedder.inner_arc().ok_or_else(|| {
            napi::Error::from_reason(
                "Memory requires a concrete EmbeddingModel, not a subclassed instance without a provider",
            )
        })?;
        let inner = Memory::new(arc, BackendKind::InMemory(Arc::clone(&backend.inner)));
        Ok(Self { inner })
    }

    /// Create a memory store in local-only mode (no embedding model) with an `InMemoryBackend`.
    ///
    /// Only `searchLocal()` is available; `search()` will throw.
    #[napi(factory)]
    pub fn local(backend: &JsInMemoryBackend) -> Self {
        let inner = Memory::local(BackendKind::InMemory(Arc::clone(&backend.inner)));
        Self { inner }
    }

    /// Add a text entry to the memory store.
    ///
    /// @param id       - A unique identifier for this entry.
    /// @param text     - The text content to store.
    /// @param metadata - Optional arbitrary metadata (will be stored as JSON).
    /// @returns The id of the stored entry.
    #[napi]
    pub async fn add(
        &self,
        id: String,
        text: String,
        metadata: Option<serde_json::Value>,
    ) -> Result<String> {
        let entry = blazen_memory::MemoryEntry {
            id,
            text,
            metadata: metadata.unwrap_or(serde_json::Value::Null),
        };

        let ids = self.inner.add(vec![entry]).await.map_err(to_napi_error)?;

        Ok(ids.into_iter().next().unwrap_or_default())
    }

    /// Add multiple text entries to the memory store in a single batch.
    ///
    /// @param entries - An array of `{ id, text, metadata? }` objects.
    /// @returns The ids of the stored entries.
    #[napi]
    pub async fn add_many(&self, entries: Vec<JsAddEntry>) -> Result<Vec<String>> {
        let entries: Vec<blazen_memory::MemoryEntry> = entries
            .into_iter()
            .map(|e| blazen_memory::MemoryEntry {
                id: e.id,
                text: e.text,
                metadata: e.metadata.unwrap_or(serde_json::Value::Null),
            })
            .collect();

        self.inner.add(entries).await.map_err(to_napi_error)
    }

    /// Search using a query string with the configured embedding model.
    ///
    /// Returns up to `limit` results sorted by descending similarity.
    /// Requires an embedding model (throws if created with `Memory.local()`).
    ///
    /// @param query - The search query text.
    /// @param limit - Maximum number of results (default: 10).
    /// @param `metadata_filter` - Optional JSON object to filter results. Only
    ///   entries whose metadata is a superset of the filter are returned.
    #[napi]
    pub async fn search(
        &self,
        query: String,
        limit: Option<u32>,
        metadata_filter: Option<serde_json::Value>,
    ) -> Result<Vec<JsMemoryResult>> {
        let limit = limit.unwrap_or(10) as usize;
        let results = self
            .inner
            .search(&query, limit, metadata_filter.as_ref())
            .await
            .map_err(to_napi_error)?;

        Ok(results
            .into_iter()
            .map(|r| JsMemoryResult {
                id: r.id,
                text: r.text,
                score: r.score,
                metadata: r.metadata,
            })
            .collect())
    }

    /// Search using only text-level `SimHash` (no embedding model required).
    ///
    /// This is a cheaper, lower-quality search that works in local-only mode.
    ///
    /// @param query - The search query text.
    /// @param limit - Maximum number of results (default: 10).
    /// @param `metadata_filter` - Optional JSON object to filter results. Only
    ///   entries whose metadata is a superset of the filter are returned.
    #[napi]
    pub async fn search_local(
        &self,
        query: String,
        limit: Option<u32>,
        metadata_filter: Option<serde_json::Value>,
    ) -> Result<Vec<JsMemoryResult>> {
        let limit = limit.unwrap_or(10) as usize;
        let results = self
            .inner
            .search_local(&query, limit, metadata_filter.as_ref())
            .await
            .map_err(to_napi_error)?;

        Ok(results
            .into_iter()
            .map(|r| JsMemoryResult {
                id: r.id,
                text: r.text,
                score: r.score,
                metadata: r.metadata,
            })
            .collect())
    }

    /// Retrieve a single entry by id.
    ///
    /// @param id - The entry identifier.
    /// @returns The entry, or `null` if not found.
    #[napi]
    pub async fn get(&self, id: String) -> Result<Option<JsMemoryEntry>> {
        let entry = self.inner.get(&id).await.map_err(to_napi_error)?;

        Ok(entry.map(|e| JsMemoryEntry {
            id: e.id,
            text: e.text,
            metadata: e.metadata,
        }))
    }

    /// Delete an entry by id.
    ///
    /// @param id - The entry identifier.
    /// @returns `true` if the entry existed and was deleted.
    #[napi]
    pub async fn delete(&self, id: String) -> Result<bool> {
        self.inner.delete(&id).await.map_err(to_napi_error)
    }

    /// Return the number of entries in the store.
    #[napi]
    pub async fn count(&self) -> Result<u32> {
        let len = self.inner.len().await.map_err(to_napi_error)?;
        #[allow(clippy::cast_possible_truncation)]
        Ok(len as u32)
    }
}

#[cfg(not(target_os = "wasi"))]
#[napi]
#[allow(clippy::must_use_candidate, clippy::missing_errors_doc)]
impl JsMemory {
    /// Create a memory store with an embedding model and a `JsonlBackend`.
    #[napi(factory)]
    pub fn with_jsonl(embedder: &JsEmbeddingModel, backend: &JsJsonlBackend) -> Result<Self> {
        let arc = embedder.inner_arc().ok_or_else(|| {
            napi::Error::from_reason(
                "Memory requires a concrete EmbeddingModel, not a subclassed instance without a provider",
            )
        })?;
        let inner = Memory::new(arc, BackendKind::Jsonl(Arc::clone(&backend.inner)));
        Ok(Self { inner })
    }

    /// Create a memory store with an embedding model and a `ValkeyBackend`.
    #[napi(factory)]
    pub fn with_valkey(embedder: &JsEmbeddingModel, backend: &JsValkeyBackend) -> Result<Self> {
        let arc = embedder.inner_arc().ok_or_else(|| {
            napi::Error::from_reason(
                "Memory requires a concrete EmbeddingModel, not a subclassed instance without a provider",
            )
        })?;
        let inner = Memory::new(arc, BackendKind::Valkey(Arc::clone(&backend.inner)));
        Ok(Self { inner })
    }

    /// Create a memory store in local-only mode with a `JsonlBackend`.
    #[napi(factory)]
    pub fn local_jsonl(backend: &JsJsonlBackend) -> Self {
        let inner = Memory::local(BackendKind::Jsonl(Arc::clone(&backend.inner)));
        Self { inner }
    }

    /// Create a memory store in local-only mode with a `ValkeyBackend`.
    #[napi(factory)]
    pub fn local_valkey(backend: &JsValkeyBackend) -> Self {
        let inner = Memory::local(BackendKind::Valkey(Arc::clone(&backend.inner)));
        Self { inner }
    }
}

#[napi]
#[allow(clippy::must_use_candidate, clippy::missing_errors_doc)]
impl JsMemory {
    /// Create a memory store with an embedding model and an `UpstashBackend`.
    #[napi(factory, js_name = "withUpstash")]
    pub fn with_upstash(embedder: &JsEmbeddingModel, backend: &JsUpstashBackend) -> Result<Self> {
        let arc = embedder.inner_arc().ok_or_else(|| {
            napi::Error::from_reason(
                "Memory requires a concrete EmbeddingModel, not a subclassed instance without a provider",
            )
        })?;
        let inner = Memory::new(arc, BackendKind::Upstash(Arc::clone(&backend.inner)));
        Ok(Self { inner })
    }

    /// Create a memory store in local-only mode with an `UpstashBackend`.
    #[napi(factory, js_name = "localUpstash")]
    pub fn local_upstash(backend: &JsUpstashBackend) -> Self {
        let inner = Memory::local(BackendKind::Upstash(Arc::clone(&backend.inner)));
        Self { inner }
    }
}

// ---------------------------------------------------------------------------
// Input type for add_many
// ---------------------------------------------------------------------------

/// An entry to add to the memory store (used by `addMany`).
#[napi(object)]
pub struct JsAddEntry {
    /// Unique identifier. If empty, one will be generated.
    pub id: String,
    /// The text content to store.
    pub text: String,
    /// Optional arbitrary metadata.
    pub metadata: Option<serde_json::Value>,
}

// ---------------------------------------------------------------------------
// JsStoredEntry
// ---------------------------------------------------------------------------

/// A persisted memory entry as returned from the underlying backend.
///
/// Mirrors [`blazen_memory::types::StoredEntry`] for use from JavaScript.
/// `simhash` is the 64-bit text-level `SimHash` clamped into `i64` (saturating
/// at `i64::MAX` if the underlying `u64` exceeds the positive range — this is
/// a one-way representation suitable for display, not lossless round-trip).
#[napi(object)]
pub struct JsStoredEntry {
    /// Unique identifier for this entry.
    pub id: String,
    /// The original text content.
    pub text: String,
    /// Raw embedding vector (empty when running in local-only mode or when
    /// the embedding has not been retained alongside the entry).
    pub embedding: Vec<f64>,
    /// Arbitrary user metadata.
    pub metadata: serde_json::Value,
    /// Text-level `SimHash` (64-bit), clamped from `u64` into a signed `i64`
    /// for representation. Saturates at `i64::MAX`.
    pub simhash: i64,
    /// LSH bands derived from the embedding `SimHash`. Each band is a 32-bit
    /// integer chunk (clamped into `i64` for JS interop).
    #[napi(js_name = "simhashBands")]
    pub simhash_bands: Vec<i64>,
}

// ---------------------------------------------------------------------------
// Free functions
// ---------------------------------------------------------------------------

/// Default seed for embedding `SimHash`, matching `Memory`'s default
/// (`"ELIDSIMH"` interpreted as a `u64`).
const DEFAULT_SIMHASH_SEED: u64 = 0x454c_4944_5349_4d48;

/// Compute similarity between two text strings using 64-bit `SimHash`.
///
/// Hashes each input with `elid::simhash` and compares using normalized
/// Hamming distance over 64 bits. Returns a value in `[0.0, 1.0]`.
#[napi(js_name = "computeTextSimhashSimilarity")]
#[allow(clippy::needless_pass_by_value)]
#[must_use]
pub fn compute_text_simhash_similarity(a: String, b: String) -> f64 {
    let ha = elid::simhash(&a);
    let hb = elid::simhash(&b);
    core_compute_text_simhash_similarity(ha, hb)
}

/// Compute similarity between two embedding vectors using 128-bit `SimHash`.
///
/// Hashes each input vector with `elid::embeddings::vector_simhash::simhash_128`
/// (using the default `Memory` seed) and compares using normalized Hamming
/// distance over 128 bits. Returns a value in `[0.0, 1.0]`.
#[napi(js_name = "computeEmbeddingSimhashSimilarity")]
#[must_use]
pub fn compute_embedding_simhash_similarity(a: Vec<f64>, b: Vec<f64>) -> f64 {
    #[allow(clippy::cast_possible_truncation)]
    let af: Vec<f32> = a.into_iter().map(|x| x as f32).collect();
    #[allow(clippy::cast_possible_truncation)]
    let bf: Vec<f32> = b.into_iter().map(|x| x as f32).collect();
    let ha = elid::embeddings::vector_simhash::simhash_128(&af, DEFAULT_SIMHASH_SEED);
    let hb = elid::embeddings::vector_simhash::simhash_128(&bf, DEFAULT_SIMHASH_SEED);
    core_compute_embedding_simhash_similarity(ha, hb)
}

/// Compute ELID-based similarity between two embedding vectors.
///
/// Encodes each vector with the default Mini128 ELID profile, then computes
/// Hamming distance between the ELID payloads, normalized to `[0.0, 1.0]`.
#[napi(js_name = "computeElidSimilarity")]
#[allow(clippy::missing_errors_doc)]
pub fn compute_elid_similarity(a: Vec<f64>, b: Vec<f64>) -> Result<f64> {
    #[allow(clippy::cast_possible_truncation)]
    let af: Vec<f32> = a.into_iter().map(|x| x as f32).collect();
    #[allow(clippy::cast_possible_truncation)]
    let bf: Vec<f32> = b.into_iter().map(|x| x as f32).collect();
    let profile = elid::embeddings::Profile::default();
    let elid_a = elid::embeddings::encode(&af, &profile)
        .map_err(|e| memory_error_to_napi(blazen_memory::MemoryError::Elid(e)))?;
    let elid_b = elid::embeddings::encode(&bf, &profile)
        .map_err(|e| memory_error_to_napi(blazen_memory::MemoryError::Elid(e)))?;
    core_compute_elid_similarity(elid_a.as_str(), elid_b.as_str()).map_err(memory_error_to_napi)
}

/// Parse a hex-encoded 128-bit `SimHash` and return its value as a decimal string.
///
/// The Node binding does not enable napi-rs's `napi6` feature, so JavaScript
/// `BigInt` is not available. We return the parsed value as a base-10 string,
/// which JS can convert with `BigInt(s)` if needed.
///
/// Returns an error if the input is not a valid 32-character hex string
/// representing a 128-bit value.
#[napi(js_name = "simhashFromHex")]
#[allow(clippy::needless_pass_by_value, clippy::missing_errors_doc)]
pub fn simhash_from_hex(hex: String) -> Result<String> {
    let value = core_simhash_from_hex(&hex)
        .ok_or_else(|| napi::Error::from_reason(format!("invalid 128-bit hex SimHash: {hex:?}")))?;
    Ok(value.to_string())
}

/// Encode a 128-bit `SimHash` (passed as a decimal string) as a zero-padded
/// 32-character hex string.
///
/// The value is accepted as a decimal string (e.g. produced by
/// `BigInt(...).toString()` in JS) because the Node binding does not enable
/// napi-rs's `napi6` feature and therefore cannot accept a `BigInt` directly.
///
/// Returns an error if the input is not a valid base-10 representation of a
/// non-negative 128-bit integer.
#[napi(js_name = "simhashToHex")]
#[allow(clippy::needless_pass_by_value, clippy::missing_errors_doc)]
pub fn simhash_to_hex(value: String) -> Result<String> {
    let parsed: u128 = value.parse().map_err(|_| {
        napi::Error::from_reason(format!(
            "simhashToHex: expected a non-negative 128-bit decimal integer, got {value:?}"
        ))
    })?;
    Ok(core_simhash_to_hex(parsed))
}
