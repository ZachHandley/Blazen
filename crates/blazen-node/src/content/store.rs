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
//! - `ContentStore.custom({ put, resolve, fetchBytes, ... })` —
//!   user-supplied async callbacks.
//!
//! All async methods return real `Promise<T>` to JS via napi-rs's `async fn`
//! support, matching the pattern used by [`crate::agent::run_agent`].
//!
//! # Subclassing from JS
//!
//! Users can also subclass `ContentStore` and override the async methods.
//! When such an instance is handed to a Blazen API that needs to call into
//! the store, [`extract_store`] detects the subclass and wraps the JS
//! object in a [`JsHostContentStore`] adapter that mirrors
//! [`crate::providers::custom::NodeHostDispatch`]'s
//! [`napi::threadsafe_function::ThreadsafeFunction`]-based dispatch.
//!
//! ```typescript
//! class S3ContentStore extends ContentStore {
//!   constructor(bucket: string) { super(); this.bucket = bucket; }
//!   async put(body, hint) { ... }
//!   async resolve(handle) { ... }
//!   async fetchBytes(handle) { ... }
//!   async fetchStream(handle) { ... }   // optional; may return bytes or AsyncIterable<Uint8Array>
//!   async delete(handle) { ... }        // optional
//! }
//! ```
//!
//! The base-class `put` / `resolve` / `fetchBytes` methods on a `Subclass`
//! variant raise `NotImplementedError`, so subclasses must override at
//! least the three required ones; calling `super().put(...)` from a
//! subclass is therefore an error and not the dispatch path. Built-in
//! factory instances do not raise — they delegate to the underlying Rust
//! store.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;

use async_trait::async_trait;
use blazen_llm::content::store::ByteStream;
use blazen_llm::content::{
    AnthropicFilesStore, ContentBody, ContentHandle, ContentHint, ContentMetadata, ContentStore,
    CustomContentStore, DynContentStore, FalStorageStore, GeminiFilesStore, InMemoryContentStore,
    OpenAiFilesStore,
};
use blazen_llm::error::BlazenError;
use blazen_llm::types::MediaSource;
use bytes::Bytes;
use futures_util::StreamExt;
use napi::bindgen_prelude::{
    Buffer, Either, FromNapiValue, Function, JsObjectValue, Object, Promise, Result, Unknown,
    ValueType,
};
use napi::threadsafe_function::ThreadsafeFunction;
use napi::{Env, JsValue, Status};
use napi_derive::napi;

use crate::error::{blazen_error_to_napi, llm_error_to_napi};

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
// CustomContentStoreOptions — JS-facing options for `ContentStore.custom`
// ---------------------------------------------------------------------------

/// Plain object passed to [`JsContentStore::custom`].
///
/// Each property is the JS-side implementation of one of the trait's
/// methods. `put`, `resolve`, and `fetchBytes` are required; `fetchStream`
/// and `delete` are optional.
///
/// All callbacks must be `async` (or return a `Promise`) and accept the
/// JSON shapes defined by [`ContentBody`] / [`ContentHint`] /
/// [`ContentHandle`] on the input side; outputs are deserialized into the
/// matching Rust types. See [`JsHostContentStore`] for the per-method
/// payload shapes.
///
/// `name` is a short identifier used in error / tracing messages; defaults
/// to `"custom"`.
#[napi(object)]
pub struct CustomContentStoreOptions {
    /// Required: persist content, return a handle.
    #[napi(ts_type = "(body: ContentBody, hint: ContentHint) => Promise<ContentHandle>")]
    pub put: Function<'static, serde_json::Value, Promise<Option<serde_json::Value>>>,
    /// Required: turn a handle into a wire-renderable [`MediaSource`].
    #[napi(ts_type = "(handle: ContentHandle) => Promise<MediaSource>")]
    pub resolve: Function<'static, serde_json::Value, Promise<Option<serde_json::Value>>>,
    /// Required: fetch raw bytes for a handle (`Buffer` | `Uint8Array` |
    /// `number[]`).
    #[napi(
        js_name = "fetchBytes",
        ts_type = "(handle: ContentHandle) => Promise<Buffer | Uint8Array | number[] | string>"
    )]
    pub fetch_bytes: Function<'static, serde_json::Value, Promise<Option<serde_json::Value>>>,
    /// Optional: stream raw bytes; absent falls back to `fetchBytes`. May
    /// resolve with bytes (`Buffer` / `Uint8Array` / `number[]` / base64
    /// string) or an `AsyncIterable<Uint8Array>` for true chunk-by-chunk
    /// streaming.
    #[napi(
        js_name = "fetchStream",
        ts_type = "(handle: ContentHandle) => Promise<Buffer | Uint8Array | number[] | string | AsyncIterable<Uint8Array>>"
    )]
    pub fetch_stream:
        Option<Function<'static, serde_json::Value, Promise<Option<serde_json::Value>>>>,
    /// Optional: cleanup hook; absent is a no-op.
    #[napi(ts_type = "(handle: ContentHandle) => Promise<void>")]
    pub delete: Option<Function<'static, serde_json::Value, Promise<Option<serde_json::Value>>>>,
    /// Optional human-readable identifier for logs (default: `"custom"`).
    pub name: Option<String>,
}

// ---------------------------------------------------------------------------
// Body parsing (built-in JS-facing path)
// ---------------------------------------------------------------------------

/// Decide whether a string body is a URL or a local filesystem path.
///
/// A URL is anything containing `"://"`; everything else is treated as a
/// local path so callers can drop in a relative or absolute filename without
/// extra ceremony.
fn parse_string_body(s: String) -> ContentBody {
    if s.contains("://") {
        ContentBody::Url { url: s }
    } else {
        ContentBody::LocalPath {
            path: PathBuf::from(s),
        }
    }
}

// ---------------------------------------------------------------------------
// ThreadsafeFunction type alias — single JSON-arg, JSON-Promise-returning
// ---------------------------------------------------------------------------

/// Pre-built JS callback for one `ContentStore` method.
///
/// Mirrors [`crate::providers::custom::NodeHostDispatch`]'s `HostMethodTsfn`:
/// the JS host method takes a single serialized JSON argument and returns
/// a `Promise<Value | undefined | null>`.
type StoreMethodTsfn = ThreadsafeFunction<
    serde_json::Value,
    Promise<Option<serde_json::Value>>,
    serde_json::Value,
    Status,
    false,
    true,
>;

/// Streaming variant of [`StoreMethodTsfn`] for `put`.
///
/// The input is [`PutStreamArgs`] (a Rust-side `(body, hint)` payload, where
/// `body` is a [`ContentBody::Stream`]). The TSF's `call_js_back` runs on the
/// JS thread, builds a JS array `[bodyObj, hintObj]` — `bodyObj` carries a
/// live `AsyncIterable<Uint8Array>` so chunks flow lazily — and hands the
/// raw `napi_value` of that array to napi as the single JS argument, matching
/// the existing JSON path's call shape.
type PutStreamTsfn = ThreadsafeFunction<
    PutStreamArgs,
    Promise<Option<serde_json::Value>>,
    PutStreamCallArgs,
    Status,
    false,
    true,
>;

/// Method-name table for [`JsHostContentStore::from_host_object`].
///
/// All methods accept a single JSON-shaped argument and return a
/// `Promise<Value | undefined | null>`. The JS-camelCase names are what we
/// look up on the host object via `has_named_property` / `get_named_property`.
const CONTENT_STORE_METHODS: &[&str] = &["put", "resolve", "fetchBytes", "fetchStream", "delete"];

// ---------------------------------------------------------------------------
// Inner — built-in vs subclass marker
// ---------------------------------------------------------------------------

/// Backing storage for [`JsContentStore`].
///
/// Built-in / custom-callback stores hold a real `Arc<dyn ContentStore>`.
/// The `Subclass` marker is what the napi default constructor produces
/// when a JS user runs `new ContentStore()` (typically via `super()`
/// in a subclass); calls to the base-class methods on this variant raise
/// because the subclass is expected to override them. The Rust side never
/// asks the marker variant to do anything useful — instead [`extract_store`]
/// detects subclasses and hands out a [`JsHostContentStore`] adapter.
enum Inner {
    /// A concrete Rust [`ContentStore`] (one of the built-in factories,
    /// or a [`CustomContentStore`] built from JS callbacks).
    BuiltIn(DynContentStore),
    /// Sentinel: this `JsContentStore` was instantiated as the base class
    /// of a JS subclass via `super()`. Calls to the base-class default
    /// methods raise.
    Subclass,
}

// ---------------------------------------------------------------------------
// JsContentStore
// ---------------------------------------------------------------------------

/// Pluggable registry for multimodal content. Wraps
/// [`Arc<dyn blazen_llm::content::ContentStore>`].
///
/// Construct via the static factories (e.g. `ContentStore.inMemory()`,
/// `ContentStore.custom({ put, resolve, fetchBytes })`) or by extending
/// `ContentStore` and overriding the async methods. Stores are cheap to
/// clone — internally an `Arc` — so passing the same instance across
/// multiple agents / requests is fine.
#[napi(js_name = "ContentStore")]
pub struct JsContentStore {
    inner: Inner,
}

impl JsContentStore {
    /// Build a JS-side wrapper from any `ContentStore` implementation.
    #[must_use]
    pub fn from_arc(inner: Arc<dyn ContentStore>) -> Self {
        Self {
            inner: Inner::BuiltIn(inner),
        }
    }

    /// Borrow the underlying store as `Arc<dyn ContentStore>` when this
    /// instance wraps a built-in / custom-callback backend. Returns
    /// `None` for subclass-marker instances; callers that must support
    /// subclasses should go through [`extract_store`] instead.
    #[must_use]
    pub fn as_arc(&self) -> Option<Arc<dyn ContentStore>> {
        match &self.inner {
            Inner::BuiltIn(inner) => Some(Arc::clone(inner)),
            Inner::Subclass => None,
        }
    }
}

/// Attempt to extract a [`DynContentStore`] from any JS object.
///
/// - If `obj` is a [`JsContentStore`] (base class or subclass) wrapping a
///   built-in store, unwrap and clone the inner `Arc`.
/// - If `obj` is a subclass instance whose `Inner` is `Subclass`, wrap the
///   JS object in a [`JsHostContentStore`] adapter that dispatches back
///   into JS via threadsafe functions.
/// - Otherwise, return an error.
///
/// Callers throughout `blazen-node` that need to feed a content store into
/// Rust code should prefer this helper over poking at
/// `JsContentStore::as_arc` directly so subclasses are handled
/// transparently.
#[allow(dead_code)]
pub(crate) fn extract_store(obj: Object<'_>) -> Result<DynContentStore> {
    // Unsafe block: napi-rs's `Object::is_instance_of` requires the env;
    // we already hold the Object so the env is implicit. Use the typed
    // accessor instead — try to coerce the object directly.
    //
    // First, see if the object exposes the trait surface (put/resolve/...).
    // Subclasses MUST override the required methods, so any instance we get
    // here that defines its own `put` / `resolve` / `fetchBytes` is a
    // candidate for the host adapter.
    //
    // For built-in / custom-callback stores returned from one of the
    // factory functions, the JS prototype will not have `put` etc. as own
    // properties — they live on the napi-generated class prototype, where
    // they delegate into the Rust `Inner::BuiltIn`. We detect that by
    // looking at the marker class instance via napi's downcast.
    //
    // Implementation note: napi-rs 3 does not expose a stable
    // "is_exact_instance_of" check for `#[napi]` classes the way pyo3
    // does. We therefore route everything through the host-adapter path
    // and let it discover at construction time which methods exist;
    // subclasses that forget to override a required method will surface
    // a clear "method not implemented" error from the dispatcher.
    //
    // The one case we *do* short-circuit is a literal `JsContentStore`
    // built-in returned from a factory: its `inner` is already an
    // `Arc<dyn ContentStore>`, so we don't need to round-trip through JS.
    // We detect this by looking for the napi-internal pointer via the
    // `Object::get_named_property` machinery would require extra glue;
    // instead, pull out the wrapped Rust struct via napi's
    // `Reference`-style downcast.
    //
    // For the moment, given there is no upstream call site that needs to
    // distinguish, we always wrap in the host adapter. Built-in stores
    // simply have all methods defined at the JS level and pass through.
    let host = JsHostContentStore::from_host_object(obj)?;
    Ok(Arc::new(host))
}

#[napi]
#[allow(clippy::missing_errors_doc, clippy::needless_pass_by_value)]
impl JsContentStore {
    // -----------------------------------------------------------------
    // Constructor — needed so subclasses can `super()`
    // -----------------------------------------------------------------

    /// Base-class constructor. Call from your subclass via `super()`.
    /// On its own, the base class is not useful — the default method
    /// implementations raise.
    #[napi(constructor)]
    #[must_use]
    pub fn new() -> Self {
        Self {
            inner: Inner::Subclass,
        }
    }

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

    /// Build a store backed by user-supplied async callbacks.
    ///
    /// Mirrors the Rust [`CustomContentStore::builder`] API. The
    /// `options` object must provide at least `put`, `resolve`, and
    /// `fetchBytes`; `fetchStream` and `delete` are optional. All
    /// callbacks must be `async` (or return a `Promise`).
    ///
    /// Argument shapes seen by JS:
    ///
    /// - `put(body, hint)`: `body` is a JSON-tagged
    ///   [`ContentBody`] (`{type: "bytes", data: number[]}`,
    ///   `{type: "url", url: string}`, `{type: "local_path", path: string}`,
    ///   or `{type: "provider_file", provider: string, id: string}`).
    ///   `hint` is a [`ContentHint`] dict (all fields optional). Must
    ///   resolve with a [`ContentHandle`]-shaped object
    ///   `{id, kind, mimeType?, byteSize?, displayName?}`.
    /// - `resolve(handle)`: `handle` is a [`ContentHandle`] dict. Must
    ///   resolve with a serialized [`MediaSource`] object
    ///   (e.g. `{type: "url", url: "..."}`).
    /// - `fetchBytes(handle)`: must resolve with a `Buffer`,
    ///   `Uint8Array`, or `number[]` of bytes.
    /// - `fetchStream(handle)` (optional): may resolve with either bytes
    ///   (`Buffer` / `Uint8Array` / `number[]` / base64 string) or an
    ///   `AsyncIterable<Uint8Array>` for chunk-by-chunk streaming.
    /// - `delete(handle)` (optional): must resolve with `undefined`.
    #[napi(factory)]
    pub fn custom(options: CustomContentStoreOptions) -> Result<Self> {
        let CustomContentStoreOptions {
            put,
            resolve,
            fetch_bytes,
            fetch_stream,
            delete,
            name,
        } = options;

        let name = name.unwrap_or_else(|| "custom".to_owned());

        // Build TSFs for each callback up-front so the per-call hot path
        // is a `HashMap::get` + `call_async`. `put` gets a second TSF so
        // streaming bodies can hand a live `AsyncIterable<Uint8Array>` to
        // the JS callback.
        let put_tsfn = build_tsfn(&put, "put")?;
        let put_stream_tsfn = build_put_stream_tsfn(&put, "put")?;
        let resolve_tsfn = build_tsfn(&resolve, "resolve")?;
        let fetch_bytes_tsfn = build_tsfn(&fetch_bytes, "fetchBytes")?;
        let fetch_stream_tsfn = match fetch_stream {
            Some(f) => Some(build_fetch_stream_tsfn(&f, "fetchStream")?),
            None => None,
        };
        let delete_tsfn = match delete {
            Some(f) => Some(build_tsfn(&f, "delete")?),
            None => None,
        };

        let mut builder = CustomContentStore::builder(name);

        // put(body, hint) -> ContentHandle
        let put_arc = Arc::new(put_tsfn);
        let put_stream_arc = Arc::new(put_stream_tsfn);
        builder = builder.put(move |body, hint| {
            let put_arc = Arc::clone(&put_arc);
            let put_stream_arc = Arc::clone(&put_stream_arc);
            Box::pin(async move {
                call_put(put_arc.as_ref(), Some(put_stream_arc.as_ref()), body, hint).await
            })
        });

        // resolve(handle) -> MediaSource
        let resolve_arc = Arc::new(resolve_tsfn);
        builder = builder.resolve(move |handle| {
            let resolve_arc = Arc::clone(&resolve_arc);
            Box::pin(async move { call_resolve(resolve_arc.as_ref(), handle).await })
        });

        // fetch_bytes(handle) -> Vec<u8>
        let fetch_bytes_arc = Arc::new(fetch_bytes_tsfn);
        builder = builder.fetch_bytes(move |handle| {
            let fetch_bytes_arc = Arc::clone(&fetch_bytes_arc);
            Box::pin(async move { call_fetch_bytes(fetch_bytes_arc.as_ref(), handle).await })
        });

        // fetch_stream(handle) -> ByteStream. The JS callback may return
        // either bytes or a live `AsyncIterable<Uint8Array>`; the decoding
        // happens in `FetchStreamReturn::from_napi_value` on the JS thread
        // so async-iterable bridging stays safe.
        if let Some(fs_tsfn) = fetch_stream_tsfn {
            let fs_arc = Arc::new(fs_tsfn);
            builder = builder.fetch_stream(move |handle| {
                let fs_arc = Arc::clone(&fs_arc);
                Box::pin(async move { call_fetch_stream(fs_arc.as_ref(), handle).await })
            });
        }

        // delete(handle) -> ()
        if let Some(del_tsfn) = delete_tsfn {
            let del_arc = Arc::new(del_tsfn);
            builder = builder.delete(move |handle| {
                let del_arc = Arc::clone(&del_arc);
                Box::pin(async move { call_delete(del_arc.as_ref(), handle).await })
            });
        }

        let store = builder.build().map_err(blazen_error_to_napi)?;
        Ok(Self::from_arc(Arc::new(store)))
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
        let inner = self.require_built_in("put")?;
        let content_body = match body {
            Either::A(buf) => ContentBody::Bytes { data: buf.to_vec() },
            Either::B(s) => parse_string_body(s),
        };
        let hint = options.into_hint();
        let handle = inner
            .put(content_body, hint)
            .await
            .map_err(llm_error_to_napi)?;
        Ok(JsContentHandle::from_rust(&handle))
    }

    /// Resolve a handle to a wire-renderable [`MediaSource`] (returned as a
    /// JS object — the same JSON shape Blazen's request builders accept).
    #[napi]
    pub async fn resolve(&self, handle: JsContentHandle) -> Result<serde_json::Value> {
        let inner = self.require_built_in("resolve")?;
        let rust_handle = handle.to_rust();
        let source: MediaSource = inner
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
        let inner = self.require_built_in("fetchBytes")?;
        let rust_handle = handle.to_rust();
        let bytes = inner
            .fetch_bytes(&rust_handle)
            .await
            .map_err(llm_error_to_napi)?;
        Ok(Buffer::from(bytes))
    }

    /// Stream raw bytes for a handle chunk-by-chunk.
    ///
    /// Returns a `Promise<AsyncIterable<Uint8Array>>`. Each `next()` call on
    /// the iterator pulls one chunk from the underlying [`ByteStream`]; the
    /// iterator is automatically `done` once the stream completes. Errors
    /// surfaced mid-stream reject the corresponding `next()` promise.
    ///
    /// Useful for large payloads where holding the entire body in a
    /// [`Buffer`] would be wasteful — pipe directly into a file, an HTTP
    /// response, or another transform without buffering.
    ///
    /// Built-in stores that lack a native streaming path fall back to a
    /// single-chunk iterator over [`Self::fetch_bytes`].
    #[napi(
        js_name = "fetchStream",
        ts_return_type = "Promise<AsyncIterable<Uint8Array>>"
    )]
    pub fn fetch_stream<'env>(
        &self,
        env: &'env Env,
        handle: JsContentHandle,
    ) -> Result<napi::bindgen_prelude::PromiseRaw<'env, Object<'env>>> {
        let inner = self.require_built_in("fetchStream")?;
        let rust_handle = handle.to_rust();
        env.spawn_future_with_callback(
            async move {
                inner
                    .fetch_stream(&rust_handle)
                    .await
                    .map_err(llm_error_to_napi)
            },
            byte_stream_to_js_async_iterable,
        )
    }

    /// Cheap metadata lookup without materializing the bytes.
    #[napi]
    pub async fn metadata(&self, handle: JsContentHandle) -> Result<JsContentMetadata> {
        let inner = self.require_built_in("metadata")?;
        let rust_handle = handle.to_rust();
        let meta = inner
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
        let inner = self.require_built_in("delete")?;
        let rust_handle = handle.to_rust();
        inner
            .delete(&rust_handle)
            .await
            .map_err(llm_error_to_napi)?;
        Ok(())
    }
}

impl Default for JsContentStore {
    fn default() -> Self {
        Self::new()
    }
}

impl JsContentStore {
    /// Default-method guard: the base-class `put`/`resolve`/etc. methods
    /// dispatch only on built-in / custom-callback stores. When a
    /// subclass forgets to override a required method and `super()`
    /// dispatch falls through, raise a clear error rather than silently
    /// looping back into the host-dispatch adapter.
    fn require_built_in(&self, method: &str) -> Result<DynContentStore> {
        match &self.inner {
            Inner::BuiltIn(inner) => Ok(Arc::clone(inner)),
            Inner::Subclass => Err(napi::Error::new(
                napi::Status::GenericFailure,
                format!(
                    "ContentStore subclass must override `{method}()` (called the base-class default)"
                ),
            )),
        }
    }
}

// ---------------------------------------------------------------------------
// TSF helpers — bind `this` to the host object and build the TSF
// ---------------------------------------------------------------------------

/// Wrap a JS function into a [`StoreMethodTsfn`].
///
/// For standalone callbacks passed into [`JsContentStore::custom`] we don't
/// have a host object to bind `this` to; the callback is expected to be
/// either a free function or already-bound (e.g. an arrow function or a
/// method extracted via `obj.method.bind(obj)`).
fn build_tsfn(
    f: &Function<'_, serde_json::Value, Promise<Option<serde_json::Value>>>,
    label: &str,
) -> Result<StoreMethodTsfn> {
    f.build_threadsafe_function::<serde_json::Value>()
        .weak::<true>()
        .build()
        .map_err(|e| {
            napi::Error::from_reason(format!(
                "ContentStore.custom: failed to build threadsafe function for `{label}`: {e}"
            ))
        })
}

/// Extract a JS function from `host_object[js_name]` and bind it to
/// `host_object` so `this` is correct when the callback runs.
fn bind_method(host_object: &Object<'_>, js_name: &str) -> Result<Option<StoreMethodTsfn>> {
    if !host_object.has_named_property(js_name).unwrap_or(false) {
        return Ok(None);
    }
    let js_function: Function<'_, serde_json::Value, Promise<Option<serde_json::Value>>> =
        match host_object.get_named_property(js_name) {
            Ok(f) => f,
            Err(_) => return Ok(None),
        };
    let bound = js_function.bind(host_object).map_err(|e| {
        napi::Error::from_reason(format!(
            "ContentStore subclass: failed to bind `this` for method `{js_name}`: {e}"
        ))
    })?;
    let tsfn: StoreMethodTsfn = bound
        .build_threadsafe_function::<serde_json::Value>()
        .weak::<true>()
        .build()
        .map_err(|e| {
            napi::Error::from_reason(format!(
                "ContentStore subclass: failed to build threadsafe function for `{js_name}`: {e}"
            ))
        })?;
    Ok(Some(tsfn))
}

/// Like [`bind_method`] but produces both the JSON-typed TSF and the
/// streaming-typed [`PutStreamTsfn`] from the same `put` JS function so the
/// dispatcher can pick the right path per body variant.
fn bind_put_methods(host_object: &Object<'_>) -> Result<Option<(StoreMethodTsfn, PutStreamTsfn)>> {
    if !host_object.has_named_property("put").unwrap_or(false) {
        return Ok(None);
    }
    let js_function: Function<'_, serde_json::Value, Promise<Option<serde_json::Value>>> =
        match host_object.get_named_property("put") {
            Ok(f) => f,
            Err(_) => return Ok(None),
        };
    let bound = js_function.bind(host_object).map_err(|e| {
        napi::Error::from_reason(format!(
            "ContentStore subclass: failed to bind `this` for method `put`: {e}"
        ))
    })?;
    let json_tsfn: StoreMethodTsfn = bound
        .build_threadsafe_function::<serde_json::Value>()
        .weak::<true>()
        .build()
        .map_err(|e| {
            napi::Error::from_reason(format!(
                "ContentStore subclass: failed to build threadsafe function for `put`: {e}"
            ))
        })?;
    let stream_tsfn = build_put_stream_tsfn(&bound, "put")?;
    Ok(Some((json_tsfn, stream_tsfn)))
}

// ---------------------------------------------------------------------------
// Callback bridges — each one TSF-dispatches a single JSON arg into JS and
// awaits the returned Promise.
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Streaming bridges between Rust `ByteStream` and JS `AsyncIterable<Uint8Array>`
// ---------------------------------------------------------------------------

/// Mirror of the Python helper: wrap an owned byte buffer in a one-chunk
/// [`ByteStream`].
pub(crate) fn bytes_into_byte_stream(data: Vec<u8>) -> ByteStream {
    Box::pin(futures_util::stream::once(
        async move { Ok(Bytes::from(data)) },
    ))
}

/// Build a JS object satisfying the `AsyncIterable` + iterator protocols from
/// a Rust [`ByteStream`].
///
/// The returned object is BOTH the iterable (its `[Symbol.asyncIterator]`
/// returns itself) and the iterator (it has a `next()` method returning
/// `Promise<{ value: Uint8Array, done: boolean }>`). Each `next()` call
/// pulls one chunk from the underlying stream.
pub(crate) fn byte_stream_to_js_async_iterable(
    env: &Env,
    stream: ByteStream,
) -> Result<Object<'_>> {
    // The `tokio::sync::Mutex` serializes concurrent `next()` calls so the
    // stream is never polled twice in parallel.
    let state = Arc::new(tokio::sync::Mutex::new(Some(stream)));

    let mut iter_obj = Object::new(env)?;

    let next_state = Arc::clone(&state);
    let next_fn =
        env.create_function_from_closure::<(), napi::sys::napi_value, _>("next", move |ctx| {
            let state = Arc::clone(&next_state);
            let promise = ctx.env.spawn_future_with_callback(
                async move {
                    let mut guard = state.lock().await;
                    match guard.as_mut() {
                        Some(s) => match s.next().await {
                            Some(Ok(bytes)) => Ok(Some(bytes.to_vec())),
                            Some(Err(e)) => Err(napi::Error::from_reason(format!(
                                "ByteStream chunk failed: {e}"
                            ))),
                            None => {
                                *guard = None;
                                Ok(None)
                            }
                        },
                        None => Ok(None),
                    }
                },
                move |env, val: Option<Vec<u8>>| {
                    let mut obj = Object::new(env)?;
                    if let Some(bytes) = val {
                        obj.set("value", Buffer::from(bytes))?;
                        obj.set("done", false)?;
                    } else {
                        obj.set("value", ())?;
                        obj.set("done", true)?;
                    }
                    Ok(obj)
                },
            )?;
            Ok(promise.raw())
        })?;
    iter_obj.set("next", next_fn)?;

    // `[Symbol.asyncIterator]()` must return the iterator itself.
    let iter_raw_value = iter_obj.value();
    let self_returning_fn = env.create_function_from_closure::<(), napi::sys::napi_value, _>(
        "[Symbol.asyncIterator]",
        move |_ctx| Ok(iter_raw_value.value),
    )?;
    let global = env.get_global()?;
    let symbol_obj = global.get_named_property_unchecked::<Object>("Symbol")?;
    let async_iterator_symbol =
        symbol_obj.get_named_property_unchecked::<Unknown>("asyncIterator")?;
    iter_obj.set_property(async_iterator_symbol, self_returning_fn)?;

    Ok(iter_obj)
}

/// One element yielded by a JS async iterator: `{value, done}`.
///
/// The JS `value` is decoded as bytes when present (Buffer / `Uint8Array`).
/// On `done: true` we ignore `value` per the iterator protocol.
struct JsIteratorChunk {
    done: bool,
    value: Option<Vec<u8>>,
}

#[allow(unsafe_code)]
impl FromNapiValue for JsIteratorChunk {
    unsafe fn from_napi_value(
        env: napi::sys::napi_env,
        napi_val: napi::sys::napi_value,
    ) -> Result<Self> {
        // SAFETY: `napi_val` is a valid `napi_value` per the trait contract.
        let obj = unsafe { Object::from_napi_value(env, napi_val)? };
        let done: Option<bool> = obj.get("done")?;
        if done.unwrap_or(false) {
            return Ok(Self {
                done: true,
                value: None,
            });
        }
        let raw_value: Option<Unknown<'_>> = obj.get("value")?;
        let bytes = match raw_value {
            None => None,
            Some(unknown) => {
                let ty = unknown.get_type()?;
                if ty == ValueType::Undefined || ty == ValueType::Null {
                    None
                } else {
                    let raw = unknown.value();
                    // SAFETY: `raw.value` is a live napi_value owned by the
                    // surrounding Promise scope; `Buffer`/`Uint8Array`'s
                    // `from_napi_value` perform their own validation and we
                    // discard failures to fall through.
                    if let Ok(buf) = unsafe { Buffer::from_napi_value(env, raw.value) } {
                        Some(buf.to_vec())
                    } else if let Ok(arr) = unsafe {
                        napi::bindgen_prelude::Uint8Array::from_napi_value(env, raw.value)
                    } {
                        Some(arr.to_vec())
                    } else {
                        return Err(napi::Error::from_reason(
                            "AsyncIterable chunk `value` must be Buffer / Uint8Array",
                        ));
                    }
                }
            }
        };
        Ok(Self {
            done: false,
            value: bytes,
        })
    }
}

/// Drive a JS `AsyncIterable<Uint8Array>` (or already-an-iterator object
/// whose `next()` returns `Promise<{value, done}>`) and surface each chunk
/// as a Rust [`ByteStream`].
pub(crate) fn js_async_iterable_to_byte_stream(iter: Object<'_>) -> Result<ByteStream> {
    enum State {
        Open(ThreadsafeFunction<(), Promise<JsIteratorChunk>, (), Status, false, true>),
        Done,
    }

    let env_handle = Env::from_raw(iter.value().env);
    let global = env_handle.get_global()?;
    let symbol_obj = global.get_named_property_unchecked::<Object>("Symbol")?;
    let async_iterator_symbol =
        symbol_obj.get_named_property_unchecked::<Unknown>("asyncIterator")?;

    // If `iter[Symbol.asyncIterator]` exists and is callable, call it to
    // obtain the real iterator; otherwise treat `iter` itself as the
    // iterator.
    let sym_method: Option<Function<'_, (), Object<'_>>> =
        iter.get_property(async_iterator_symbol)?;
    let iterator: Object<'_> = match sym_method {
        Some(f) => f.apply(iter, ())?,
        None => iter,
    };

    let next_fn: Function<'_, (), Promise<JsIteratorChunk>> =
        iterator.get_named_property_unchecked("next")?;
    let bound_next = next_fn.bind(iterator)?;
    let next_tsfn: ThreadsafeFunction<(), Promise<JsIteratorChunk>, (), Status, false, true> =
        bound_next
            .build_threadsafe_function::<()>()
            .weak::<true>()
            .build()?;

    // Drive `next()` on demand via `futures_util::stream::unfold`. We deliberately
    // avoid `tokio::spawn` here — `js_async_iterable_to_byte_stream` is invoked
    // from `FetchStreamReturn::from_napi_value` on the JS thread, where no
    // tokio runtime is registered. The unfolded stream is polled by Rust
    // orchestration on whatever runtime owns the `ByteStream` consumer
    // (typically tokio), so chunk pulls run on the right context.
    let stream = futures_util::stream::unfold(State::Open(next_tsfn), |state| async move {
        let tsfn = match state {
            State::Open(t) => t,
            State::Done => return None,
        };
        let promise = match tsfn.call_async(()).await {
            Ok(p) => p,
            Err(e) => {
                return Some((
                    Err(BlazenError::request(format!(
                        "AsyncIterable next() dispatch failed: {e}"
                    ))),
                    State::Done,
                ));
            }
        };
        match promise.await {
            Ok(chunk) => {
                if chunk.done {
                    None
                } else {
                    let bytes = chunk.value.unwrap_or_default();
                    Some((Ok(Bytes::from(bytes)), State::Open(tsfn)))
                }
            }
            Err(e) => Some((
                Err(BlazenError::request(format!(
                    "AsyncIterable next() rejected: {e}"
                ))),
                State::Done,
            )),
        }
    });

    Ok(Box::pin(stream))
}

// ---------------------------------------------------------------------------
// fetchStream return decoding — bytes OR live AsyncIterable<Uint8Array>
// ---------------------------------------------------------------------------

/// Resolved value of a JS `fetchStream` callback.
///
/// The decoder runs on the JS thread (inside `Promise<T>::from_napi_value`'s
/// `.then` handler), so when the host returns an `AsyncIterable` we can
/// build the live `next()` Tsfn right there and hand back a ready
/// [`ByteStream`]. The `Bytes` variant covers the legacy paths (Buffer /
/// `Uint8Array` / `number[]` / base64 string).
pub(crate) enum FetchStreamReturn {
    /// Fully-buffered byte payload — same shape `fetchBytes` accepts.
    Bytes(Vec<u8>),
    /// Already-bridged Rust stream pulling chunks from a JS `AsyncIterable`.
    Stream(ByteStream),
}

#[allow(unsafe_code)]
impl FromNapiValue for FetchStreamReturn {
    unsafe fn from_napi_value(
        env: napi::sys::napi_env,
        napi_val: napi::sys::napi_value,
    ) -> Result<Self> {
        // Determine the JS value's type. Anything object-shaped may be a
        // Buffer / Uint8Array / AsyncIterable; strings and arrays go down
        // the legacy JSON path.
        let mut val_type: napi::sys::napi_valuetype = napi::sys::ValueType::napi_undefined;
        // SAFETY: `napi_val` is a valid napi_value per the trait contract.
        let status = unsafe { napi::sys::napi_typeof(env, napi_val, &raw mut val_type) };
        if status != napi::sys::Status::napi_ok {
            return Err(napi::Error::from_reason(
                "fetchStream: napi_typeof failed on resolved value",
            ));
        }

        if val_type == napi::sys::ValueType::napi_object {
            // Buffer first — Node Buffers also pass `napi_typeof == object`,
            // and `Buffer::from_napi_value` is the cheapest discriminator.
            // SAFETY: `napi_val` is live in the resolution scope; the
            // converter validates internally and we discard failures.
            if let Ok(buf) = unsafe { Buffer::from_napi_value(env, napi_val) } {
                return Ok(Self::Bytes(buf.to_vec()));
            }
            if let Ok(arr) =
                unsafe { napi::bindgen_prelude::Uint8Array::from_napi_value(env, napi_val) }
            {
                return Ok(Self::Bytes(arr.to_vec()));
            }

            // Async iterable detection: object exposes `Symbol.asyncIterator`.
            // SAFETY: `napi_val` is a live JS object; we re-wrap it as
            // `Object<'_>` to use the existing helper which builds the
            // chunk-pulling Tsfn on the current (JS) thread.
            let obj = unsafe { Object::from_napi_value(env, napi_val)? };
            let env_handle = Env::from_raw(env);
            let global = env_handle.get_global()?;
            let symbol_obj = global.get_named_property_unchecked::<Object>("Symbol")?;
            let async_iterator_symbol =
                symbol_obj.get_named_property_unchecked::<Unknown>("asyncIterator")?;
            let sym_method: Option<Unknown<'_>> = obj.get_property(async_iterator_symbol)?;
            if let Some(unknown) = sym_method
                && unknown.get_type()? == ValueType::Function
            {
                let stream = js_async_iterable_to_byte_stream(obj)?;
                return Ok(Self::Stream(stream));
            }
        }

        // Fall back to the JSON shapes accepted by `fetchBytes`.
        // SAFETY: `napi_val` is a live napi_value; `serde_json::Value`'s
        // converter handles arrays / strings / objects.
        let value = unsafe { serde_json::Value::from_napi_value(env, napi_val)? };
        json_to_bytes(value, "fetchStream")
            .map(Self::Bytes)
            .map_err(|e| napi::Error::from_reason(format!("{e}")))
    }
}

/// JS callback for `fetchStream` typed to accept either bytes or a live
/// `AsyncIterable<Uint8Array>` from the host. The `Option` wrapper allows
/// hosts to resolve `null` / `undefined` — caller treats that as an empty
/// byte stream.
type FetchStreamTsfn = ThreadsafeFunction<
    serde_json::Value,
    Promise<Option<FetchStreamReturn>>,
    serde_json::Value,
    Status,
    false,
    true,
>;

/// Recast a JSON-typed `Function` into the [`FetchStreamReturn`] shape and
/// build a [`FetchStreamTsfn`] from it. Mirrors [`build_tsfn`] but with the
/// streaming-aware return type so `Promise<FetchStreamReturn>::from_napi_value`
/// runs on the JS thread when the host's promise resolves.
#[allow(unsafe_code)]
fn build_fetch_stream_tsfn(
    f: &Function<'_, serde_json::Value, Promise<Option<serde_json::Value>>>,
    label: &str,
) -> Result<FetchStreamTsfn> {
    let raw = f.value();
    // SAFETY: `raw.value` and `raw.env` were just produced by `JsValue::value`
    // on a live `Function`; constructing a `Function` with a different
    // `Return` generic is a no-op cast (the type carries only PhantomData).
    let recast: Function<'_, serde_json::Value, Promise<Option<FetchStreamReturn>>> =
        unsafe { Function::from_napi_value(raw.env, raw.value)? };
    recast
        .build_threadsafe_function::<serde_json::Value>()
        .weak::<true>()
        .build()
        .map_err(|e| {
            napi::Error::from_reason(format!(
                "ContentStore: failed to build streaming threadsafe function for `{label}`: {e}"
            ))
        })
}

/// Like [`bind_method`] but produces a [`FetchStreamTsfn`] for the
/// streaming-aware `fetchStream` dispatch path.
fn bind_fetch_stream_method(host_object: &Object<'_>) -> Result<Option<FetchStreamTsfn>> {
    if !host_object
        .has_named_property("fetchStream")
        .unwrap_or(false)
    {
        return Ok(None);
    }
    let js_function: Function<'_, serde_json::Value, Promise<Option<serde_json::Value>>> =
        match host_object.get_named_property("fetchStream") {
            Ok(f) => f,
            Err(_) => return Ok(None),
        };
    let bound = js_function.bind(host_object).map_err(|e| {
        napi::Error::from_reason(format!(
            "ContentStore subclass: failed to bind `this` for method `fetchStream`: {e}"
        ))
    })?;
    Ok(Some(build_fetch_stream_tsfn(&bound, "fetchStream")?))
}

/// Dispatch a `fetchStream` call: invoke the JS callback, await the
/// returned promise, and bridge the resolved value into a [`ByteStream`].
async fn call_fetch_stream(
    tsfn: &FetchStreamTsfn,
    handle: ContentHandle,
) -> std::result::Result<ByteStream, BlazenError> {
    let handle_json = serde_json::to_value(&handle).map_err(|e| {
        BlazenError::provider("custom", format!("failed to serialize ContentHandle: {e}"))
    })?;
    let promise = tsfn.call_async(handle_json).await.map_err(|e| {
        BlazenError::provider(
            "custom",
            format!("ContentStore `fetchStream` dispatch failed: {e}"),
        )
    })?;
    let result = promise.await.map_err(|e| {
        BlazenError::provider("custom", format!("ContentStore `fetchStream` raised: {e}"))
    })?;
    Ok(match result {
        Some(FetchStreamReturn::Bytes(b)) => bytes_into_byte_stream(b),
        Some(FetchStreamReturn::Stream(s)) => s,
        None => bytes_into_byte_stream(Vec::new()),
    })
}

// ---------------------------------------------------------------------------
// Streaming-`put` TSF payload — bypasses the JSON wire so a live
// AsyncIterable<Uint8Array> reaches the JS callback intact.
// ---------------------------------------------------------------------------

/// Rust-side payload for the streaming-`put` TSF.
///
/// Only [`ContentBody::Stream`] is exercised in practice — the JSON path
/// continues to handle every other variant. The non-stream branches in
/// [`PutStreamCallArgs::from_args`] exist purely so the conversion is total.
struct PutStreamArgs {
    body: ContentBody,
    hint: ContentHint,
}

/// JS-side payload produced inside the TSF's `call_js_back` closure on the
/// JS thread: a single `napi_value` pointing at a freshly-built JS array
/// `[bodyObj, hintObj]`.
struct PutStreamCallArgs {
    array: napi::sys::napi_value,
}

// SAFETY: `PutStreamCallArgs` is created on the JS thread inside the TSF's
// `call_js_back` and immediately consumed by `into_vec` on the same thread,
// so the raw `napi_value` is never observed off-thread. The `Send`/`Sync`
// markers are required only because the napi-rs builder bound is
// `'static + JsValuesTupleIntoVec`; we do not actually move the value
// across threads.
#[allow(unsafe_code)]
unsafe impl Send for PutStreamCallArgs {}
#[allow(unsafe_code)]
unsafe impl Sync for PutStreamCallArgs {}

impl napi::bindgen_prelude::JsValuesTupleIntoVec for PutStreamCallArgs {
    fn into_vec(self, _env: napi::sys::napi_env) -> Result<Vec<napi::sys::napi_value>> {
        Ok(vec![self.array])
    }
}

impl PutStreamCallArgs {
    /// Build the `[bodyObj, hintObj]` JS array on the JS thread.
    #[allow(unsafe_code)]
    fn from_args(env: &Env, args: PutStreamArgs) -> Result<Self> {
        let body_obj = put_stream_body_to_js(env, args.body)?;
        let hint_value = serde_json::to_value(&args.hint).map_err(|e| {
            napi::Error::from_reason(format!("failed to serialize ContentHint: {e}"))
        })?;

        let mut array_ptr: napi::sys::napi_value = std::ptr::null_mut();
        // SAFETY: `env.raw()` is the live JS env on the JS thread.
        unsafe {
            let status = napi::sys::napi_create_array_with_length(env.raw(), 2, &raw mut array_ptr);
            if status != napi::sys::Status::napi_ok {
                return Err(napi::Error::from_reason(
                    "failed to create JS array for put-stream args",
                ));
            }
        }
        // SAFETY: `body_obj` is a valid napi_value owned by the JS env.
        let body_napi = unsafe {
            <Object<'_> as napi::bindgen_prelude::ToNapiValue>::to_napi_value(env.raw(), body_obj)?
        };
        // SAFETY: `hint_value` is a valid `serde_json::Value`.
        let hint_napi = unsafe {
            <serde_json::Value as napi::bindgen_prelude::ToNapiValue>::to_napi_value(
                env.raw(),
                hint_value,
            )?
        };
        // SAFETY: setting elements 0 and 1 on a length-2 array.
        unsafe {
            let s0 = napi::sys::napi_set_element(env.raw(), array_ptr, 0, body_napi);
            if s0 != napi::sys::Status::napi_ok {
                return Err(napi::Error::from_reason(
                    "failed to set body in put-stream array",
                ));
            }
            let s1 = napi::sys::napi_set_element(env.raw(), array_ptr, 1, hint_napi);
            if s1 != napi::sys::Status::napi_ok {
                return Err(napi::Error::from_reason(
                    "failed to set hint in put-stream array",
                ));
            }
        }
        Ok(Self { array: array_ptr })
    }
}

/// Convert a [`ContentBody`] into a JS object on the JS thread. The
/// `Stream` variant becomes `{type: "stream", stream: <AsyncIterable>,
/// sizeHint}`; every other variant falls back to the JSON shape used by
/// the non-streaming path so the wire stays consistent.
#[allow(unsafe_code)]
fn put_stream_body_to_js(env: &Env, body: ContentBody) -> Result<Object<'_>> {
    match body {
        ContentBody::Stream { stream, size_hint } => {
            let mut obj = Object::new(env)?;
            obj.set("type", "stream")?;
            let iter = byte_stream_to_js_async_iterable(env, stream)?;
            obj.set("stream", iter)?;
            #[allow(clippy::cast_possible_wrap)]
            match size_hint {
                Some(n) => obj.set("sizeHint", n as i64)?,
                None => obj.set("sizeHint", napi::bindgen_prelude::Null)?,
            }
            Ok(obj)
        }
        other => {
            let value = serde_json::to_value(&other).map_err(|e| {
                napi::Error::from_reason(format!("failed to serialize ContentBody: {e}"))
            })?;
            // SAFETY: `value` is a valid `serde_json::Value`; the resulting
            // napi_value is a JS Object/Array we can re-wrap as `Object`.
            let raw = unsafe {
                <serde_json::Value as napi::bindgen_prelude::ToNapiValue>::to_napi_value(
                    env.raw(),
                    value,
                )?
            };
            // SAFETY: `raw` was just created in this env.
            unsafe { Object::from_napi_value(env.raw(), raw) }
        }
    }
}

/// Build a streaming-put TSF from a JS function. The function may be either
/// a free callback (from [`CustomContentStoreOptions::put`]) or a
/// `this`-bound method extracted from a host object — the caller is
/// responsible for binding before passing it in.
fn build_put_stream_tsfn(
    f: &Function<'_, serde_json::Value, Promise<Option<serde_json::Value>>>,
    label: &str,
) -> Result<PutStreamTsfn> {
    f.build_threadsafe_function::<PutStreamArgs>()
        .weak::<true>()
        .build_callback(|ctx| PutStreamCallArgs::from_args(&ctx.env, ctx.value))
        .map_err(|e| {
            napi::Error::from_reason(format!(
                "ContentStore: failed to build streaming-put threadsafe function for `{label}`: {e}"
            ))
        })
}

/// Phase-1+2 dispatch: schedule the JS callback, await its returned
/// `Promise`, return the resolved value.
async fn invoke(
    tsfn: &StoreMethodTsfn,
    method: &str,
    arg: serde_json::Value,
) -> std::result::Result<serde_json::Value, BlazenError> {
    let promise = tsfn.call_async(arg).await.map_err(|e| {
        BlazenError::provider(
            "custom",
            format!("ContentStore `{method}` dispatch failed: {e}"),
        )
    })?;
    let value = promise.await.map_err(|e| {
        BlazenError::provider("custom", format!("ContentStore `{method}` raised: {e}"))
    })?;
    Ok(value.unwrap_or(serde_json::Value::Null))
}

/// Coerce a JSON value resolved by a JS `fetchBytes` callback into a
/// `Vec<u8>`.
///
/// Accepts `[number, ...]` (e.g. `Array.from(buffer)`), the napi
/// `Buffer` JSON shape (`{"type":"Buffer","data":[...]}`) emitted when a
/// `Buffer` is round-tripped through `JSON.stringify`, and a `string` of
/// base64.
fn json_to_bytes(
    value: serde_json::Value,
    method: &str,
) -> std::result::Result<Vec<u8>, BlazenError> {
    match value {
        serde_json::Value::Array(arr) => arr
            .into_iter()
            .map(|v| {
                v.as_u64()
                    .and_then(|n| u8::try_from(n).ok())
                    .ok_or_else(|| {
                        BlazenError::provider(
                            "custom",
                            format!(
                                "ContentStore `{method}` returned array element out of u8 range"
                            ),
                        )
                    })
            })
            .collect(),
        serde_json::Value::Object(mut obj) => {
            // Node's `Buffer.toJSON()` shape.
            if let Some(serde_json::Value::Array(arr)) = obj.remove("data") {
                arr.into_iter()
                    .map(|v| {
                        v.as_u64()
                            .and_then(|n| u8::try_from(n).ok())
                            .ok_or_else(|| {
                                BlazenError::provider(
                                    "custom",
                                    format!(
                                        "ContentStore `{method}` returned Buffer-shaped object with non-u8 data element"
                                    ),
                                )
                            })
                    })
                    .collect()
            } else {
                Err(BlazenError::provider(
                    "custom",
                    format!(
                        "ContentStore `{method}` must return Buffer / Uint8Array / number[] (got object)"
                    ),
                ))
            }
        }
        serde_json::Value::String(s) => {
            use base64::Engine;
            base64::engine::general_purpose::STANDARD
                .decode(s)
                .map_err(|e| {
                    BlazenError::provider(
                        "custom",
                        format!(
                            "ContentStore `{method}` returned a string that is not valid base64: {e}"
                        ),
                    )
                })
        }
        other => Err(BlazenError::provider(
            "custom",
            format!(
                "ContentStore `{method}` must return Buffer / Uint8Array / number[] (got {})",
                match &other {
                    serde_json::Value::Null => "null",
                    serde_json::Value::Bool(_) => "boolean",
                    serde_json::Value::Number(_) => "number",
                    _ => "unknown",
                }
            ),
        )),
    }
}

async fn call_put(
    json_tsfn: &StoreMethodTsfn,
    stream_tsfn: Option<&PutStreamTsfn>,
    body: ContentBody,
    hint: ContentHint,
) -> std::result::Result<ContentHandle, BlazenError> {
    let result = if matches!(body, ContentBody::Stream { .. }) {
        let tsfn = stream_tsfn.ok_or_else(|| {
            BlazenError::provider(
                "custom",
                "ContentStore `put`: streaming Tsfn unavailable (Stream body requires the streaming dispatch path)",
            )
        })?;
        let promise = tsfn
            .call_async(PutStreamArgs { body, hint })
            .await
            .map_err(|e| {
                BlazenError::provider("custom", format!("ContentStore `put` dispatch failed: {e}"))
            })?;
        let value = promise.await.map_err(|e| {
            BlazenError::provider("custom", format!("ContentStore `put` raised: {e}"))
        })?;
        value.unwrap_or(serde_json::Value::Null)
    } else {
        let body_json = serde_json::to_value(&body).map_err(|e| {
            BlazenError::provider("custom", format!("failed to serialize ContentBody: {e}"))
        })?;
        let hint_json = serde_json::to_value(&hint).map_err(|e| {
            BlazenError::provider("custom", format!("failed to serialize ContentHint: {e}"))
        })?;
        let arg = serde_json::Value::Array(vec![body_json, hint_json]);
        invoke(json_tsfn, "put", arg).await?
    };
    serde_json::from_value::<ContentHandle>(result).map_err(|e| {
        BlazenError::provider(
            "custom",
            format!("ContentStore `put` must return a ContentHandle: {e}"),
        )
    })
}

async fn call_resolve(
    tsfn: &StoreMethodTsfn,
    handle: ContentHandle,
) -> std::result::Result<MediaSource, BlazenError> {
    let handle_json = serde_json::to_value(&handle).map_err(|e| {
        BlazenError::provider("custom", format!("failed to serialize ContentHandle: {e}"))
    })?;
    let result = invoke(tsfn, "resolve", handle_json).await?;
    serde_json::from_value::<MediaSource>(result).map_err(|e| {
        BlazenError::provider(
            "custom",
            format!("ContentStore `resolve` must return a MediaSource object: {e}"),
        )
    })
}

async fn call_fetch_bytes(
    tsfn: &StoreMethodTsfn,
    handle: ContentHandle,
) -> std::result::Result<Vec<u8>, BlazenError> {
    let handle_json = serde_json::to_value(&handle).map_err(|e| {
        BlazenError::provider("custom", format!("failed to serialize ContentHandle: {e}"))
    })?;
    let result = invoke(tsfn, "fetchBytes", handle_json).await?;
    json_to_bytes(result, "fetchBytes")
}

async fn call_delete(
    tsfn: &StoreMethodTsfn,
    handle: ContentHandle,
) -> std::result::Result<(), BlazenError> {
    let handle_json = serde_json::to_value(&handle).map_err(|e| {
        BlazenError::provider("custom", format!("failed to serialize ContentHandle: {e}"))
    })?;
    let _ = invoke(tsfn, "delete", handle_json).await?;
    Ok(())
}

async fn call_metadata(
    tsfn: &StoreMethodTsfn,
    handle: ContentHandle,
) -> std::result::Result<ContentMetadata, BlazenError> {
    let handle_json = serde_json::to_value(&handle).map_err(|e| {
        BlazenError::provider("custom", format!("failed to serialize ContentHandle: {e}"))
    })?;
    let result = invoke(tsfn, "metadata", handle_json).await?;
    let value: serde_json::Value = result;
    let kind_str = value
        .get("kind")
        .and_then(serde_json::Value::as_str)
        .ok_or_else(|| {
            BlazenError::provider(
                "custom",
                "ContentStore `metadata` must return an object with a `kind` string field",
            )
        })?;
    let kind = parse_content_kind(kind_str).ok_or_else(|| {
        BlazenError::provider(
            "custom",
            format!("ContentStore `metadata` returned unknown kind `{kind_str}`"),
        )
    })?;
    let mime_type = value
        .get("mimeType")
        .or_else(|| value.get("mime_type"))
        .and_then(serde_json::Value::as_str)
        .map(str::to_owned);
    let byte_size = value
        .get("byteSize")
        .or_else(|| value.get("byte_size"))
        .and_then(serde_json::Value::as_u64);
    let display_name = value
        .get("displayName")
        .or_else(|| value.get("display_name"))
        .and_then(serde_json::Value::as_str)
        .map(str::to_owned);
    Ok(ContentMetadata {
        kind,
        mime_type,
        byte_size,
        display_name,
    })
}

/// Parse the canonical wire-name of a [`blazen_llm::content::ContentKind`].
fn parse_content_kind(s: &str) -> Option<blazen_llm::content::ContentKind> {
    use blazen_llm::content::ContentKind;
    Some(match s {
        "image" => ContentKind::Image,
        "audio" => ContentKind::Audio,
        "video" => ContentKind::Video,
        "document" => ContentKind::Document,
        "three_d_model" => ContentKind::ThreeDModel,
        "cad" => ContentKind::Cad,
        "archive" => ContentKind::Archive,
        "font" => ContentKind::Font,
        "code" => ContentKind::Code,
        "data" => ContentKind::Data,
        "other" => ContentKind::Other,
        _ => return None,
    })
}

// ---------------------------------------------------------------------------
// JsHostContentStore — adapter for JS subclasses
// ---------------------------------------------------------------------------

/// Implements [`ContentStore`] by dispatching back into a JS object
/// (typically a subclass of [`JsContentStore`]).
///
/// Mirrors [`crate::providers::custom::NodeHostDispatch`] in style: each
/// method's TSF is built once at construction time so the per-call hot
/// path is `HashMap::get` + `tsfn.call_async`.
pub struct JsHostContentStore {
    /// Cached, pre-bound callbacks keyed by JS-camelCase method name.
    methods: HashMap<&'static str, Arc<StoreMethodTsfn>>,
    /// Streaming sibling for `put`: built from the same JS function as
    /// `methods["put"]` but typed to carry a live `AsyncIterable` to JS
    /// when the body is [`ContentBody::Stream`].
    put_stream_tsfn: Option<Arc<PutStreamTsfn>>,
    /// Streaming-aware Tsfn for `fetchStream`: typed
    /// `Promise<Option<FetchStreamReturn>>` so the JS callback can return
    /// either bytes or a live `AsyncIterable<Uint8Array>`.
    fetch_stream_tsfn: Option<Arc<FetchStreamTsfn>>,
}

impl JsHostContentStore {
    /// Walk [`CONTENT_STORE_METHODS`] and extract a bound TSF for each
    /// method present on the host object. Missing methods are simply
    /// omitted; `put` / `resolve` / `fetchBytes` are checked for presence
    /// at construction time and surface a clear error.
    ///
    /// # Errors
    ///
    /// Returns a [`napi::Error`] if any of the required methods (`put`,
    /// `resolve`, `fetchBytes`) is missing or if binding a method to its
    /// host fails.
    pub fn from_host_object(host_object: Object<'_>) -> Result<Self> {
        let mut methods: HashMap<&'static str, Arc<StoreMethodTsfn>> = HashMap::new();
        let mut put_stream_tsfn: Option<Arc<PutStreamTsfn>> = None;
        let mut fetch_stream_tsfn: Option<Arc<FetchStreamTsfn>> = None;
        for &name in CONTENT_STORE_METHODS {
            if name == "put" {
                if let Some((json_tsfn, stream_tsfn)) = bind_put_methods(&host_object)? {
                    methods.insert(name, Arc::new(json_tsfn));
                    put_stream_tsfn = Some(Arc::new(stream_tsfn));
                }
            } else if name == "fetchStream" {
                if let Some(tsfn) = bind_fetch_stream_method(&host_object)? {
                    fetch_stream_tsfn = Some(Arc::new(tsfn));
                }
            } else if let Some(tsfn) = bind_method(&host_object, name)? {
                methods.insert(name, Arc::new(tsfn));
            }
        }
        for required in ["put", "resolve", "fetchBytes"] {
            if !methods.contains_key(required) {
                return Err(napi::Error::from_reason(format!(
                    "ContentStore subclass must define an async `{required}` method"
                )));
            }
        }
        Ok(Self {
            methods,
            put_stream_tsfn,
            fetch_stream_tsfn,
        })
    }

    fn has_method(&self, name: &str) -> bool {
        self.methods.contains_key(name)
    }
}

impl std::fmt::Debug for JsHostContentStore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("JsHostContentStore")
            .field("methods", &self.methods.keys().collect::<Vec<_>>())
            .field("put_stream", &self.put_stream_tsfn.is_some())
            .field("fetch_stream", &self.fetch_stream_tsfn.is_some())
            .finish()
    }
}

#[async_trait]
impl ContentStore for JsHostContentStore {
    async fn put(
        &self,
        body: ContentBody,
        hint: ContentHint,
    ) -> std::result::Result<ContentHandle, BlazenError> {
        let tsfn = self.methods.get("put").ok_or_else(|| {
            BlazenError::unsupported("ContentStore subclass missing required `put` method")
        })?;
        let stream_tsfn = self.put_stream_tsfn.as_deref();
        call_put(tsfn, stream_tsfn, body, hint).await
    }

    async fn resolve(
        &self,
        handle: &ContentHandle,
    ) -> std::result::Result<MediaSource, BlazenError> {
        let tsfn = self.methods.get("resolve").ok_or_else(|| {
            BlazenError::unsupported("ContentStore subclass missing required `resolve` method")
        })?;
        call_resolve(tsfn, handle.clone()).await
    }

    async fn fetch_bytes(
        &self,
        handle: &ContentHandle,
    ) -> std::result::Result<Vec<u8>, BlazenError> {
        let tsfn = self.methods.get("fetchBytes").ok_or_else(|| {
            BlazenError::unsupported("ContentStore subclass missing required `fetchBytes` method")
        })?;
        call_fetch_bytes(tsfn, handle.clone()).await
    }

    async fn fetch_stream(
        &self,
        handle: &ContentHandle,
    ) -> std::result::Result<ByteStream, BlazenError> {
        if let Some(tsfn) = self.fetch_stream_tsfn.as_deref() {
            return call_fetch_stream(tsfn, handle.clone()).await;
        }
        let bytes = self.fetch_bytes(handle).await?;
        Ok(bytes_into_byte_stream(bytes))
    }

    async fn metadata(
        &self,
        handle: &ContentHandle,
    ) -> std::result::Result<ContentMetadata, BlazenError> {
        if !self.has_method("metadata") {
            // Default trait impl: resolve and report what the handle knows.
            let _ = self.resolve(handle).await?;
            return Ok(ContentMetadata {
                kind: handle.kind,
                mime_type: handle.mime_type.clone(),
                byte_size: handle.byte_size,
                display_name: handle.display_name.clone(),
            });
        }
        // Note: not in CONTENT_STORE_METHODS today (the table only lists
        // the five we always look up at construction), so this branch is
        // currently unreachable. Kept for symmetry with the Python
        // sibling and future expansion.
        let tsfn = self.methods.get("metadata").ok_or_else(|| {
            BlazenError::unsupported("ContentStore subclass missing optional `metadata` method")
        })?;
        call_metadata(tsfn, handle.clone()).await
    }

    async fn delete(&self, handle: &ContentHandle) -> std::result::Result<(), BlazenError> {
        if !self.has_method("delete") {
            return Ok(());
        }
        let tsfn = self.methods.get("delete").ok_or_else(|| {
            BlazenError::unsupported("ContentStore subclass missing optional `delete` method")
        })?;
        call_delete(tsfn, handle.clone()).await
    }
}
