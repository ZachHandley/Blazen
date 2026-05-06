//! `wasm-bindgen` wrapper for [`blazen_llm::content`].
//!
//! Exposes a `ContentStore` class to TypeScript that wraps any
//! [`blazen_llm::content::ContentStore`] implementation. Static factories
//! cover the in-process [`InMemoryContentStore`] plus the four provider-file
//! stores that work over HTTP (`OpenAI` Files, Anthropic Files, Gemini
//! Files, fal.ai Storage). The local-filesystem store is intentionally not
//! exposed — `wasm32` targets have no filesystem access.
//!
//! Module-level functions [`image_input`], [`audio_input`], [`video_input`],
//! [`file_input`], [`three_d_input`], and [`cad_input`] mirror the
//! [`blazen_llm::content::tool_input`] schema-builder helpers so JS callers
//! can declare typed tool inputs that accept content handles.
//!
//! ## Example
//!
//! ```js
//! import init, { ContentStore, imageInput } from '@blazen/sdk';
//!
//! await init();
//!
//! const store = ContentStore.inMemory();
//! const handle = await store.put(
//!   new Uint8Array(await (await fetch('/cat.png')).arrayBuffer()),
//!   'image',
//!   'image/png',
//!   'cat.png',
//! );
//!
//! const schema = imageInput('photo', 'the photo to describe');
//! ```
//!
//! ## Plugging in a custom backend
//!
//! WASM users can supply their own [`blazen_llm::content::ContentStore`]
//! implementation in two equivalent ways. Both end up wrapped behind the
//! same [`JsHostContentStore`] adapter that mirrors the Node
//! `JsHostContentStore` (and Python `PyHostContentStore`) sibling.
//!
//! ### Option A — callbacks
//!
//! ```js
//! const store = ContentStore.custom({
//!   put: async (body, hint) => ({ id, kind, mimeType, byteSize, displayName }),
//!   resolve: async (handle) => ({ type: 'url', url: '...' }),
//!   fetchBytes: async (handle) => new Uint8Array([...]),
//!   fetchStream: async (handle) => readableStreamOrBytes,  // optional
//!   delete: async (handle) => {},                          // optional
//! });
//! ```
//!
//! ### Option B — subclass
//!
//! ```js
//! class S3ContentStore extends ContentStore {
//!   constructor(bucket) { super(); this.bucket = bucket; }
//!   async put(body, hint) { ... }
//!   async resolve(handle) { ... }
//!   async fetchBytes(handle) { ... }
//!   async fetchStream(handle) { ... }   // optional; falls back to fetchBytes
//!   async delete(handle) { ... }        // optional
//! }
//! ```
//!
//! The base-class methods (`put` / `resolve` / `fetchBytes` / `delete`) on a
//! subclass-marker instance raise — subclasses MUST override at least the
//! three required methods. `super()` in the JS subclass constructor is the
//! supported entry point for marking the instance as a subclass.

use std::cell::RefCell;
use std::pin::Pin;
use std::rc::Rc;
use std::sync::Arc;

use async_trait::async_trait;
use bytes::Bytes;
use futures_util::{StreamExt, stream};
use js_sys::{Object, Promise, Reflect, Uint8Array};
use wasm_bindgen::prelude::*;
use wasm_bindgen_futures::{JsFuture, future_to_promise};
use web_sys::{ReadableStream, ReadableStreamDefaultController, ReadableStreamDefaultReader};

use blazen_llm::BlazenError;
use blazen_llm::content::store::ByteStream;
use blazen_llm::content::stores::{
    AnthropicFilesStore, FalStorageStore, GeminiFilesStore, InMemoryContentStore, OpenAiFilesStore,
};
use blazen_llm::content::tool_input::{
    audio_input as rust_audio_input, cad_input as rust_cad_input, file_input as rust_file_input,
    image_input as rust_image_input, three_d_input as rust_three_d_input,
    video_input as rust_video_input,
};
use blazen_llm::content::{
    ContentBody, ContentHandle, ContentHint, ContentKind, ContentStore, DynContentStore,
};
use blazen_llm::types::MediaSource;

// ---------------------------------------------------------------------------
// TypeScript type declarations for the ContentStore surface
// ---------------------------------------------------------------------------

#[wasm_bindgen(typescript_custom_section)]
const TS_CONTENT_STORE_TYPES: &str = r#"
/**
 * Caller-supplied hints surfaced to a custom `ContentStore.put` callback.
 * Mirrors `blazen_llm::content::ContentHint` — every field is optional;
 * the store may auto-detect them from bytes when not provided.
 */
export interface ContentHint {
    /** MIME type, if known. */
    mime_type?: string | null;
    /** Caller's preferred classification — overrides any automatic detection. */
    kind_hint?: ContentKind | null;
    /** Human-readable display name (filename, caption, etc.). */
    display_name?: string | null;
    /** Byte size, if known up-front. */
    byte_size?: number | null;
}

/**
 * The body argument to `ContentStore.put` (and the value handed to the
 * `put` callback on a custom store). One of:
 *
 * - `bytes` — buffered bytes;
 * - `url` — a public URL the store may record by reference;
 * - `local_path` — a filesystem path (rejected on `wasm32` — the local
 *   store is not exposed but the variant is reachable via the JS callback
 *   surface for parity with native bindings);
 * - `provider_file` — an existing handle on a provider (e.g. `OpenAI` /
 *   Anthropic / Gemini / fal Files);
 * - `stream` — a live `ReadableStream<Uint8Array>` for chunked upload,
 *   plus an optional `sizeHint` if the total byte size is known.
 */
export type ContentBody =
    | { type: "bytes"; data: Uint8Array | number[] }
    | { type: "url"; url: string }
    | { type: "local_path"; path: string }
    | { type: "provider_file"; provider: string; id: string }
    | { type: "stream"; stream: ReadableStream<Uint8Array>; sizeHint: number | null };

/**
 * Body argument accepted by the host-side `ContentStore.put(body, ...)`
 * method. Bytes / URL / `ReadableStream<Uint8Array>` are accepted directly;
 * pass the tagged-stream object form (`{type:"stream", ...}`) when you want
 * to provide a `sizeHint` alongside the stream.
 */
export type ContentPutBody =
    | Uint8Array
    | number[]
    | string
    | ReadableStream<Uint8Array>
    | { type: "stream"; stream: ReadableStream<Uint8Array>; sizeHint?: number | null };

/**
 * Return type of a custom `ContentStore.fetchStream` callback. May yield a
 * live `ReadableStream<Uint8Array>` for chunk-by-chunk delivery, or a
 * single buffered chunk as `Uint8Array` / `number[]`.
 */
export type FetchStreamResult = ReadableStream<Uint8Array> | Uint8Array | number[];

/**
 * The options bag accepted by `ContentStore.custom({...})`. Mirrors the
 * surface of `blazen_llm::content::ContentStore`. `put`, `resolve`, and
 * `fetchBytes` are required; `fetchStream` and `delete` are optional.
 *
 * - `fetchStream` may resolve with a `ReadableStream<Uint8Array>` for
 *   real chunk-by-chunk streaming, or with `Uint8Array` / `number[]`
 *   for a single buffered chunk.
 * - `put` receives the same `ContentBody` shape the Rust side serializes
 *   via `serde_wasm_bindgen` — including the `{type: "stream", stream,
 *   sizeHint}` variant for streaming uploads.
 */
export interface CustomContentStoreOptions {
    put: (body: ContentBody, hint: ContentHint) => Promise<ContentHandle> | ContentHandle;
    resolve: (handle: ContentHandle) => Promise<MediaSource> | MediaSource;
    fetchBytes: (handle: ContentHandle) => Promise<Uint8Array | number[]> | Uint8Array | number[];
    fetchStream?: (handle: ContentHandle) => Promise<FetchStreamResult> | FetchStreamResult;
    delete?: (handle: ContentHandle) => Promise<void> | void;
    name?: string;
}
"#;

// Typed `JsValue` newtypes — these surface in the generated `.d.ts` as the
// named TypeScript types declared in `TS_CONTENT_STORE_TYPES` above, while
// remaining `JsValue` on the Rust side. Without these aliases, wasm-bindgen
// emits `any` for every `JsValue` parameter / return.
#[wasm_bindgen]
extern "C" {
    /// Strongly-typed alias of `JsValue` that surfaces in the `.d.ts` as
    /// `CustomContentStoreOptions`.
    #[wasm_bindgen(typescript_type = "CustomContentStoreOptions")]
    pub type CustomContentStoreOptionsTs;

    /// Strongly-typed alias of `JsValue` that surfaces in the `.d.ts` as
    /// `ContentPutBody`.
    #[wasm_bindgen(typescript_type = "ContentPutBody")]
    pub type ContentPutBodyTs;

    /// Strongly-typed alias of `JsValue` that surfaces in the `.d.ts` as
    /// `ContentHandle`.
    #[wasm_bindgen(typescript_type = "ContentHandle")]
    pub type ContentHandleTs;

    /// Strongly-typed alias of `Promise` that surfaces in the `.d.ts` as
    /// `Promise<ContentHandle>`.
    #[wasm_bindgen(typescript_type = "Promise<ContentHandle>")]
    pub type PromiseContentHandle;

    /// Strongly-typed alias of `Promise` that surfaces in the `.d.ts` as
    /// `Promise<MediaSource>`.
    #[wasm_bindgen(typescript_type = "Promise<MediaSource>")]
    pub type PromiseMediaSource;

    /// Strongly-typed alias of `Promise` that surfaces in the `.d.ts` as
    /// `Promise<Uint8Array>`.
    #[wasm_bindgen(typescript_type = "Promise<Uint8Array>")]
    pub type PromiseUint8Array;

    /// Strongly-typed alias of `Promise` that surfaces in the `.d.ts` as
    /// `Promise<ReadableStream<Uint8Array>>`.
    #[wasm_bindgen(typescript_type = "Promise<ReadableStream<Uint8Array>>")]
    pub type PromiseReadableStream;

    /// Strongly-typed alias of `Promise` that surfaces in the `.d.ts` as
    /// `Promise<void>`.
    #[wasm_bindgen(typescript_type = "Promise<void>")]
    pub type PromiseVoid;
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Map a string kind hint (`"image"`, `"audio"`, etc.) to a [`ContentKind`].
///
/// Unknown / unrecognised strings collapse to [`ContentKind::Other`] rather
/// than erroring — this matches the behaviour of the Rust schema-builder
/// helpers.
fn parse_kind(s: &str) -> ContentKind {
    match s {
        "image" => ContentKind::Image,
        "audio" => ContentKind::Audio,
        "video" => ContentKind::Video,
        "document" => ContentKind::Document,
        "three_d_model" | "three-d-model" | "3d" | "3d_model" => ContentKind::ThreeDModel,
        "cad" => ContentKind::Cad,
        "archive" => ContentKind::Archive,
        "font" => ContentKind::Font,
        "code" => ContentKind::Code,
        "data" => ContentKind::Data,
        _ => ContentKind::Other,
    }
}

/// 2^53 — the largest integer exactly representable in `f64`
/// (`Number.MAX_SAFE_INTEGER` on the JS side). We accept any
/// non-negative finite size hint up to this bound and treat anything
/// else as "no hint" rather than overflowing.
const MAX_SAFE_INT_F64: f64 = 9_007_199_254_740_992.0;

/// Convert a JS body into a [`ContentBody`].
///
/// Accepts:
/// - `Uint8Array` (or `number[]`) → [`ContentBody::Bytes`]
/// - `string` → [`ContentBody::Url`]
/// - `ReadableStream<Uint8Array>` → [`ContentBody::Stream`] (size hint
///   defaults to `None`; callers wanting a hint should construct the
///   tagged stream object themselves — see below).
/// - Tagged stream object `{ type: "stream", stream: ReadableStream,
///   sizeHint?: number | null }` → [`ContentBody::Stream`] with optional
///   size hint.
///
/// Returns a [`JsValue`] error for any other value shape.
fn body_from_js(value: &JsValue) -> Result<ContentBody, JsValue> {
    if let Some(s) = value.as_string() {
        return Ok(ContentBody::Url { url: s });
    }
    if value.is_instance_of::<Uint8Array>() {
        let arr = Uint8Array::from(value.clone());
        return Ok(ContentBody::Bytes { data: arr.to_vec() });
    }
    if value.is_instance_of::<ReadableStream>() {
        let rs: ReadableStream = value.clone().unchecked_into();
        let stream = readable_stream_to_byte_stream(rs).map_err(|e| {
            JsValue::from_str(&format!(
                "ContentStore.put: failed to bridge ReadableStream<Uint8Array>: {e:?}"
            ))
        })?;
        return Ok(ContentBody::Stream {
            stream,
            size_hint: None,
        });
    }
    // Tagged stream object: {type: "stream", stream: ReadableStream, sizeHint?: number | null}
    if value.is_object()
        && let Ok(ty) = Reflect::get(value, &JsValue::from_str("type"))
        && ty.as_string().as_deref() == Some("stream")
        && let Ok(stream_val) = Reflect::get(value, &JsValue::from_str("stream"))
        && stream_val.is_instance_of::<ReadableStream>()
    {
        let rs: ReadableStream = stream_val.unchecked_into();
        let stream = readable_stream_to_byte_stream(rs).map_err(|e| {
            JsValue::from_str(&format!(
                "ContentStore.put: failed to bridge ReadableStream<Uint8Array>: {e:?}"
            ))
        })?;
        let size_hint = Reflect::get(value, &JsValue::from_str("sizeHint"))
            .ok()
            .and_then(|v| {
                if v.is_undefined() || v.is_null() {
                    None
                } else {
                    v.as_f64().and_then(|n| {
                        if n.is_finite() && (0.0..=MAX_SAFE_INT_F64).contains(&n) {
                            #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
                            Some(n as u64)
                        } else {
                            None
                        }
                    })
                }
            });
        return Ok(ContentBody::Stream { stream, size_hint });
    }
    // Fall back: arrays of numbers serialize as a Vec<u8>
    if let Ok(bytes) = serde_wasm_bindgen::from_value::<Vec<u8>>(value.clone()) {
        return Ok(ContentBody::Bytes { data: bytes });
    }
    Err(JsValue::from_str(
        "ContentStore.put: body must be a Uint8Array, string (URL), ReadableStream<Uint8Array>, or {type: \"stream\", stream, sizeHint?}",
    ))
}

/// Serialize a [`ContentBody`] for hand-off to a JS callback.
///
/// Non-streaming variants go through `serde_wasm_bindgen::to_value` — same
/// shape they've always had on the JS side. The `Stream` variant is built
/// manually because [`ByteStream`] is `!Serialize`: we wrap it as a live
/// `web_sys::ReadableStream<Uint8Array>` and emit
/// `{type: "stream", stream: <ReadableStream>, sizeHint: number | null}`.
fn body_to_js_value(body: ContentBody) -> Result<JsValue, BlazenError> {
    match body {
        ContentBody::Stream { stream, size_hint } => {
            let rs = byte_stream_to_readable_stream(stream).map_err(|e| {
                BlazenError::provider(
                    "custom",
                    format!("failed to wrap ByteStream as ReadableStream: {e:?}"),
                )
            })?;
            let size_hint_js = serde_wasm_bindgen::to_value(&size_hint).map_err(|e| {
                BlazenError::provider("custom", format!("failed to serialize sizeHint: {e}"))
            })?;
            let obj = Object::new();
            let set = |key: &str, value: &JsValue| -> Result<(), BlazenError> {
                Reflect::set(&obj, &JsValue::from_str(key), value).map(|_| ()).map_err(|e| {
                    BlazenError::provider(
                        "custom",
                        format!("failed to set `{key}` on ContentBody object: {e:?}"),
                    )
                })
            };
            set("type", &JsValue::from_str("stream"))?;
            set("stream", rs.as_ref())?;
            set("sizeHint", &size_hint_js)?;
            Ok(obj.into())
        }
        other => serde_wasm_bindgen::to_value(&other).map_err(|e| {
            BlazenError::provider("custom", format!("failed to serialize ContentBody: {e}"))
        }),
    }
}

/// Build a [`ContentHint`] from optional caller hints.
fn build_hint(
    kind_hint: Option<&str>,
    mime_type: Option<String>,
    display_name: Option<String>,
) -> ContentHint {
    let mut hint = ContentHint::default();
    if let Some(k) = kind_hint {
        hint = hint.with_kind(parse_kind(k));
    }
    if let Some(m) = mime_type {
        hint = hint.with_mime_type(m);
    }
    if let Some(d) = display_name {
        hint = hint.with_display_name(d);
    }
    hint
}

/// Deserialize a JS-side [`ContentHandle`] (plain object) into the Rust
/// type. Returns a `JsValue` error suitable for `future_to_promise`.
fn handle_from_js(value: JsValue) -> Result<ContentHandle, JsValue> {
    serde_wasm_bindgen::from_value(value)
        .map_err(|e| JsValue::from_str(&format!("invalid ContentHandle: {e}")))
}

// ---------------------------------------------------------------------------
// SendFuture wrapper (same pattern as js_completion.rs)
// ---------------------------------------------------------------------------

/// Wrapper that unsafely implements [`Send`] for a non-Send future.
///
/// SAFETY: WASM is single-threaded; there is no other thread to race with.
/// `JsValue`, `js_sys::Function`, and the `JsFuture` returned by
/// `wasm_bindgen_futures` are all `!Send` on the wasm32 target, but the
/// `async_trait`-generated [`ContentStore`] futures still need to satisfy a
/// `+ Send` bound. We project through this wrapper for the hot-path.
struct SendFuture<F>(F);

unsafe impl<F> Send for SendFuture<F> {}

impl<F: std::future::Future> std::future::Future for SendFuture<F> {
    type Output = F::Output;

    fn poll(
        self: Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Self::Output> {
        // SAFETY: We are not moving F, just projecting through the wrapper.
        let inner = unsafe { self.map_unchecked_mut(|s| &mut s.0) };
        inner.poll(cx)
    }
}

// ---------------------------------------------------------------------------
// JsHostContentStore — adapter for JS subclasses / callback option objects
// ---------------------------------------------------------------------------

/// Implements [`ContentStore`] by dispatching back into a JS object — either
/// a subclass instance of [`WasmContentStore`] or an options object passed to
/// [`WasmContentStore::custom`].
///
/// Methods are introspected once at construction time via [`js_sys::Reflect`]
/// and cached as [`js_sys::Function`]. The per-call hot path is then a
/// direct `Function::call1` followed by `JsFuture::from(promise).await`,
/// mirroring the dispatch pattern in
/// [`crate::js_completion::JsCompletionHandler`].
pub struct JsHostContentStore {
    /// `this`-binding for method dispatch — either the subclass instance or
    /// the JS options object.
    js_object: JsValue,
    put_fn: js_sys::Function,
    resolve_fn: js_sys::Function,
    fetch_bytes_fn: js_sys::Function,
    fetch_stream_fn: Option<js_sys::Function>,
    delete_fn: Option<js_sys::Function>,
}

// SAFETY: WASM is single-threaded. `JsValue` / `js_sys::Function` are
// `!Send + !Sync` on wasm32 because they reference JS-managed handles, but
// there is no other thread to race with on this target. Mirrors the
// equivalent assertions on `JsCompletionHandler` (see `js_completion.rs`).
unsafe impl Send for JsHostContentStore {}
unsafe impl Sync for JsHostContentStore {}

impl std::fmt::Debug for JsHostContentStore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("JsHostContentStore")
            .field("has_fetch_stream", &self.fetch_stream_fn.is_some())
            .field("has_delete", &self.delete_fn.is_some())
            .finish_non_exhaustive()
    }
}

impl JsHostContentStore {
    /// Introspect `js_object` for `put` / `resolve` / `fetchBytes` (required)
    /// and `fetchStream` / `delete` (optional). Returns a [`JsValue`] error
    /// suitable for surfacing through `future_to_promise` if a required
    /// method is missing or not callable.
    ///
    /// # Errors
    ///
    /// Returns a `JsValue` if any of the required methods (`put`, `resolve`,
    /// `fetchBytes`) is missing or is not a function.
    pub fn from_js_object(js_object: JsValue) -> Result<Self, JsValue> {
        let put_fn = require_method(&js_object, "put")?;
        let resolve_fn = require_method(&js_object, "resolve")?;
        let fetch_bytes_fn = require_method(&js_object, "fetchBytes")?;
        let fetch_stream_fn = optional_method(&js_object, "fetchStream")?;
        let delete_fn = optional_method(&js_object, "delete")?;
        Ok(Self {
            js_object,
            put_fn,
            resolve_fn,
            fetch_bytes_fn,
            fetch_stream_fn,
            delete_fn,
        })
    }

    /// Internal non-Send dispatch: serialize a single argument, call the JS
    /// function with `this == js_object`, await the returned promise (if
    /// any), and return the resolved [`JsValue`].
    async fn call1(&self, func: &js_sys::Function, arg: JsValue) -> Result<JsValue, BlazenError> {
        let result = func
            .call1(&self.js_object, &arg)
            .map_err(|e| BlazenError::provider("custom", format!("{e:?}")))?;
        if result.has_type::<Promise>() {
            let promise: Promise = result.unchecked_into();
            wasm_bindgen_futures::JsFuture::from(promise)
                .await
                .map_err(|e| BlazenError::provider("custom", format!("{e:?}")))
        } else {
            Ok(result)
        }
    }

    /// Internal non-Send dispatch with two arguments. Used for `put(body,
    /// hint)`.
    async fn call2(
        &self,
        func: &js_sys::Function,
        arg1: JsValue,
        arg2: JsValue,
    ) -> Result<JsValue, BlazenError> {
        let result = func
            .call2(&self.js_object, &arg1, &arg2)
            .map_err(|e| BlazenError::provider("custom", format!("{e:?}")))?;
        if result.has_type::<Promise>() {
            let promise: Promise = result.unchecked_into();
            wasm_bindgen_futures::JsFuture::from(promise)
                .await
                .map_err(|e| BlazenError::provider("custom", format!("{e:?}")))
        } else {
            Ok(result)
        }
    }

    async fn put_impl(
        &self,
        body: ContentBody,
        hint: ContentHint,
    ) -> Result<ContentHandle, BlazenError> {
        let body_js = body_to_js_value(body)?;
        let hint_js = serde_wasm_bindgen::to_value(&hint).map_err(|e| {
            BlazenError::provider("custom", format!("failed to serialize ContentHint: {e}"))
        })?;
        let result = self.call2(&self.put_fn, body_js, hint_js).await?;
        serde_wasm_bindgen::from_value::<ContentHandle>(result).map_err(|e| {
            BlazenError::provider(
                "custom",
                format!("ContentStore `put` must return a ContentHandle: {e}"),
            )
        })
    }

    async fn resolve_impl(&self, handle: ContentHandle) -> Result<MediaSource, BlazenError> {
        let handle_js = serde_wasm_bindgen::to_value(&handle).map_err(|e| {
            BlazenError::provider("custom", format!("failed to serialize ContentHandle: {e}"))
        })?;
        let result = self.call1(&self.resolve_fn, handle_js).await?;
        serde_wasm_bindgen::from_value::<MediaSource>(result).map_err(|e| {
            BlazenError::provider(
                "custom",
                format!("ContentStore `resolve` must return a MediaSource object: {e}"),
            )
        })
    }

    async fn fetch_bytes_impl(&self, handle: ContentHandle) -> Result<Vec<u8>, BlazenError> {
        let handle_js = serde_wasm_bindgen::to_value(&handle).map_err(|e| {
            BlazenError::provider("custom", format!("failed to serialize ContentHandle: {e}"))
        })?;
        let result = self.call1(&self.fetch_bytes_fn, handle_js).await?;
        js_value_to_bytes(&result, "fetchBytes")
    }

    async fn fetch_stream_impl(&self, handle: ContentHandle) -> Result<ByteStream, BlazenError> {
        let func = self
            .fetch_stream_fn
            .as_ref()
            .ok_or_else(|| BlazenError::unsupported("fetchStream not provided"))?;
        let handle_js = serde_wasm_bindgen::to_value(&handle).map_err(|e| {
            BlazenError::provider("custom", format!("failed to serialize ContentHandle: {e}"))
        })?;
        let result = self.call1(func, handle_js).await?;
        js_value_to_byte_stream(result, "fetchStream")
    }

    async fn delete_impl(&self, handle: ContentHandle) -> Result<(), BlazenError> {
        let Some(func) = self.delete_fn.as_ref() else {
            return Ok(());
        };
        let handle_js = serde_wasm_bindgen::to_value(&handle).map_err(|e| {
            BlazenError::provider("custom", format!("failed to serialize ContentHandle: {e}"))
        })?;
        let _ = self.call1(func, handle_js).await?;
        Ok(())
    }
}

#[async_trait]
impl ContentStore for JsHostContentStore {
    async fn put(
        &self,
        body: ContentBody,
        hint: ContentHint,
    ) -> Result<ContentHandle, BlazenError> {
        // SAFETY: WASM is single-threaded.
        SendFuture(self.put_impl(body, hint)).await
    }

    async fn resolve(&self, handle: &ContentHandle) -> Result<MediaSource, BlazenError> {
        SendFuture(self.resolve_impl(handle.clone())).await
    }

    async fn fetch_bytes(&self, handle: &ContentHandle) -> Result<Vec<u8>, BlazenError> {
        SendFuture(self.fetch_bytes_impl(handle.clone())).await
    }

    async fn fetch_stream(&self, handle: &ContentHandle) -> Result<ByteStream, BlazenError> {
        // If host implements fetchStream, dispatch to it; otherwise fall back to fetchBytes.
        if self.fetch_stream_fn.is_some() {
            SendFuture(self.fetch_stream_impl(handle.clone())).await
        } else {
            let bytes = SendFuture(self.fetch_bytes_impl(handle.clone())).await?;
            Ok(bytes_into_byte_stream(bytes))
        }
    }

    async fn delete(&self, handle: &ContentHandle) -> Result<(), BlazenError> {
        SendFuture(self.delete_impl(handle.clone())).await
    }
}

// ---------------------------------------------------------------------------
// JS-side method introspection helpers
// ---------------------------------------------------------------------------

/// Look up `name` on `obj`, requiring it to be a function.
fn require_method(obj: &JsValue, name: &str) -> Result<js_sys::Function, JsValue> {
    let val = js_sys::Reflect::get(obj, &JsValue::from_str(name)).map_err(|_| {
        JsValue::from_str(&format!(
            "ContentStore custom/subclass: missing required `{name}` method"
        ))
    })?;
    if !val.is_function() {
        return Err(JsValue::from_str(&format!(
            "ContentStore custom/subclass: `{name}` must be a function"
        )));
    }
    Ok(val.unchecked_into())
}

/// Look up `name` on `obj`. Returns `Ok(None)` when absent / `undefined`,
/// `Ok(Some(func))` when present and callable, or an error when present but
/// not a function.
fn optional_method(obj: &JsValue, name: &str) -> Result<Option<js_sys::Function>, JsValue> {
    let Ok(val) = js_sys::Reflect::get(obj, &JsValue::from_str(name)) else {
        return Ok(None);
    };
    if val.is_undefined() || val.is_null() {
        return Ok(None);
    }
    if !val.is_function() {
        return Err(JsValue::from_str(&format!(
            "ContentStore custom/subclass: `{name}` must be a function or omitted"
        )));
    }
    Ok(Some(val.unchecked_into()))
}

/// Coerce a `JsValue` resolved by a JS `fetchBytes` / `fetchStream` callback
/// into a `Vec<u8>`.
///
/// Accepts a `Uint8Array` (the canonical case) and any value
/// `serde_wasm_bindgen` can decode as `Vec<u8>` (e.g. a plain JS array of
/// numbers).
fn js_value_to_bytes(value: &JsValue, method: &str) -> Result<Vec<u8>, BlazenError> {
    if value.is_instance_of::<Uint8Array>() {
        let arr = Uint8Array::from(value.clone());
        return Ok(arr.to_vec());
    }
    serde_wasm_bindgen::from_value::<Vec<u8>>(value.clone()).map_err(|e| {
        BlazenError::provider(
            "custom",
            format!("ContentStore `{method}` must return a Uint8Array or number[]: {e}"),
        )
    })
}

/// Coerce a `JsValue` resolved by a JS `fetchStream` callback into a
/// [`ByteStream`].
///
/// A `ReadableStream<Uint8Array>` is bridged chunk-by-chunk via
/// [`readable_stream_to_byte_stream`]. A `Uint8Array` / `number[]` is
/// wrapped as a single-chunk stream via [`bytes_into_byte_stream`]. Anything
/// else returns a `BlazenError`.
fn js_value_to_byte_stream(value: JsValue, method: &str) -> Result<ByteStream, BlazenError> {
    if value.is_instance_of::<ReadableStream>() {
        let rs: ReadableStream = value.unchecked_into();
        return readable_stream_to_byte_stream(rs).map_err(|e| {
            BlazenError::provider(
                "custom",
                format!("ContentStore `{method}` ReadableStream bridge failed: {e:?}"),
            )
        });
    }
    if let Ok(bytes) = js_value_to_bytes(&value, method) {
        return Ok(bytes_into_byte_stream(bytes));
    }
    Err(BlazenError::provider(
        "custom",
        format!(
            "ContentStore `{method}` must return ReadableStream<Uint8Array> or bytes (Uint8Array / number[])"
        ),
    ))
}

// ---------------------------------------------------------------------------
// ByteStream <-> web_sys::ReadableStream bridges
// ---------------------------------------------------------------------------

/// Wrap a `Vec<u8>` as a single-chunk [`ByteStream`].
///
/// Mirror of the Python / Node `bytes_into_byte_stream` helpers.
pub(crate) fn bytes_into_byte_stream(data: Vec<u8>) -> ByteStream {
    Box::pin(stream::once(async move { Ok(Bytes::from(data)) }))
}

/// Build a `web_sys::ReadableStream<Uint8Array>` whose underlying source pulls
/// chunks from a Rust [`ByteStream`].
///
/// Each `pull(controller)` invocation polls one chunk from the stream and
/// either enqueues it as a `Uint8Array`, errors the controller on failure, or
/// closes the controller on end-of-stream.
pub(crate) fn byte_stream_to_readable_stream(
    stream: ByteStream,
) -> Result<ReadableStream, JsValue> {
    let stream_cell: Rc<RefCell<Option<ByteStream>>> = Rc::new(RefCell::new(Some(stream)));

    let pull_closure = Closure::wrap(Box::new(move |controller: JsValue| -> Promise {
        let stream_cell = Rc::clone(&stream_cell);
        future_to_promise(SendFuture(async move {
            let controller: ReadableStreamDefaultController = controller.unchecked_into();
            // Take the stream out of the cell across the await so we never
            // hold a RefCell borrow over a yield point.
            let mut taken = stream_cell.borrow_mut().take();
            let next = match taken.as_mut() {
                Some(s) => s.next().await,
                None => None,
            };
            *stream_cell.borrow_mut() = taken;
            match next {
                Some(Ok(bytes)) => {
                    let arr =
                        Uint8Array::new_with_length(u32::try_from(bytes.len()).unwrap_or(u32::MAX));
                    arr.copy_from(&bytes);
                    controller.enqueue_with_chunk(&arr)?;
                    Ok(JsValue::UNDEFINED)
                }
                Some(Err(e)) => {
                    *stream_cell.borrow_mut() = None;
                    controller.error_with_e(&JsValue::from_str(&format!("{e}")));
                    Ok(JsValue::UNDEFINED)
                }
                None => {
                    *stream_cell.borrow_mut() = None;
                    controller.close()?;
                    Ok(JsValue::UNDEFINED)
                }
            }
        }))
    }) as Box<dyn FnMut(JsValue) -> Promise>);

    let source = Object::new();
    Reflect::set(
        &source,
        &JsValue::from_str("pull"),
        pull_closure.as_ref().unchecked_ref(),
    )?;
    pull_closure.forget();

    ReadableStream::new_with_underlying_source(&source)
}

/// Adapt a JS `ReadableStream<Uint8Array>` into a Rust [`ByteStream`].
///
/// Acquires a default reader, then drives `reader.read()` once per chunk via
/// `futures_util::stream::unfold`. Rejected reads and unexpected chunk shapes
/// terminate the Rust stream with a [`BlazenError`].
#[allow(clippy::needless_pass_by_value, clippy::unnecessary_wraps)]
pub(crate) fn readable_stream_to_byte_stream(rs: ReadableStream) -> Result<ByteStream, JsValue> {
    let reader: ReadableStreamDefaultReader = rs.get_reader().unchecked_into();

    let s = stream::unfold(Some(reader), |state| async move {
        let reader = state?;
        let promise = reader.read();
        let result = match SendFuture(JsFuture::from(promise)).await {
            Ok(v) => v,
            Err(e) => {
                return Some((
                    Err(BlazenError::request(format!(
                        "ReadableStream read rejected: {e:?}"
                    ))),
                    None,
                ));
            }
        };
        let done = Reflect::get(&result, &JsValue::from_str("done"))
            .ok()
            .and_then(|v| v.as_bool())
            .unwrap_or(false);
        if done {
            return None;
        }
        let value = match Reflect::get(&result, &JsValue::from_str("value")) {
            Ok(v) => v,
            Err(e) => {
                return Some((
                    Err(BlazenError::request(format!(
                        "ReadableStream read result missing `value`: {e:?}"
                    ))),
                    None,
                ));
            }
        };
        if !value.is_instance_of::<Uint8Array>() {
            return Some((
                Err(BlazenError::request(
                    "ReadableStream chunk is not a Uint8Array".to_string(),
                )),
                None,
            ));
        }
        let arr: Uint8Array = value.unchecked_into();
        let bytes = Bytes::from(arr.to_vec());
        Some((Ok(bytes), Some(reader)))
    });

    Ok(Box::pin(SendStream(s)))
}

/// `Send` shim for a non-Send Rust [`futures_core::Stream`] backed by JS
/// handles. See [`SendFuture`] for the same justification — WASM is
/// single-threaded so there is no other thread to race with.
struct SendStream<S>(S);

unsafe impl<S> Send for SendStream<S> {}

impl<S: futures_util::Stream> futures_util::Stream for SendStream<S> {
    type Item = S::Item;

    fn poll_next(
        self: Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Option<<S as futures_util::Stream>::Item>> {
        // SAFETY: We are not moving S, just projecting through the wrapper.
        let inner = unsafe { self.map_unchecked_mut(|s| &mut s.0) };
        inner.poll_next(cx)
    }
}

// ---------------------------------------------------------------------------
// Inner — built-in vs subclass marker
// ---------------------------------------------------------------------------

/// Backing storage for [`WasmContentStore`].
///
/// Built-in factory instances (and `ContentStore.custom({...})`) hold a real
/// `Arc<dyn ContentStore>`. The `Subclass` marker is what the
/// `wasm-bindgen`-generated default constructor produces when a JS user runs
/// `new ContentStore()` (typically via `super()` in a subclass); calls to
/// the base-class methods on this variant raise because the subclass is
/// expected to override them. The Rust side never asks the marker variant
/// to do anything useful — instead [`extract_store`] detects subclasses and
/// hands out a [`JsHostContentStore`] adapter.
enum Inner {
    /// A concrete Rust [`ContentStore`] (one of the built-in factories or a
    /// [`JsHostContentStore`] built from `ContentStore.custom({...})`).
    BuiltIn(DynContentStore),
    /// Sentinel: this `WasmContentStore` was instantiated as the base class
    /// of a JS subclass via `super()`. Calls to the base-class default
    /// methods raise.
    Subclass,
}

// ---------------------------------------------------------------------------
// WasmContentStore
// ---------------------------------------------------------------------------

/// A type-erased reference to any [`blazen_llm::content::ContentStore`].
///
/// Construct one of the built-in implementations via the static factory
/// methods (`inMemory`, `openaiFiles`, `anthropicFiles`, `geminiFiles`,
/// `falStorage`, `custom`); from JS the class is named `ContentStore`.
///
/// You can also `extends ContentStore` from JS and override
/// `put` / `resolve` / `fetchBytes` (required) and `fetchStream` / `delete`
/// (optional). `super()` produces a base-class marker whose default methods
/// raise; subclasses must override the required methods.
#[wasm_bindgen(js_name = "ContentStore")]
pub struct WasmContentStore {
    inner: Inner,
}

// SAFETY: WASM is single-threaded; there is no other thread to race with.
unsafe impl Send for WasmContentStore {}
unsafe impl Sync for WasmContentStore {}

#[wasm_bindgen(js_class = "ContentStore")]
#[allow(clippy::must_use_candidate, clippy::missing_errors_doc)]
impl WasmContentStore {
    // -------- Constructor (for subclass `super()`) ------------------------

    /// Base-class constructor. Call from your JS subclass via `super()`.
    /// On its own, the base class is not useful — the default method
    /// implementations raise. Subclasses must override at least `put`,
    /// `resolve`, and `fetchBytes`.
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self {
            inner: Inner::Subclass,
        }
    }

    // -------- Static factories --------------------------------------------

    /// Build an in-process [`InMemoryContentStore`].
    ///
    /// Suitable for ephemeral / small content. Bytes are kept in WASM
    /// memory; URLs and provider-file references are recorded by
    /// reference.
    #[wasm_bindgen(js_name = "inMemory")]
    pub fn in_memory() -> Self {
        Self {
            inner: Inner::BuiltIn(Arc::new(InMemoryContentStore::new()) as DynContentStore),
        }
    }

    /// Build an [`OpenAiFilesStore`] backed by the platform `fetch`
    /// HTTP client.
    ///
    /// @param apiKey - The `OpenAI` API key. Used as `Bearer <apiKey>`
    ///                 on every request.
    #[wasm_bindgen(js_name = "openaiFiles")]
    pub fn openai_files(api_key: String) -> Self {
        Self {
            inner: Inner::BuiltIn(Arc::new(OpenAiFilesStore::new(api_key)) as DynContentStore),
        }
    }

    /// Build an [`AnthropicFilesStore`] backed by the platform `fetch`
    /// HTTP client.
    ///
    /// @param apiKey - The Anthropic API key, sent as `x-api-key` on every
    ///                 request.
    #[wasm_bindgen(js_name = "anthropicFiles")]
    pub fn anthropic_files(api_key: String) -> Self {
        Self {
            inner: Inner::BuiltIn(Arc::new(AnthropicFilesStore::new(api_key)) as DynContentStore),
        }
    }

    /// Build a [`GeminiFilesStore`] backed by the platform `fetch`
    /// HTTP client.
    ///
    /// @param apiKey - The Google AI / Gemini API key.
    #[wasm_bindgen(js_name = "geminiFiles")]
    pub fn gemini_files(api_key: String) -> Self {
        Self {
            inner: Inner::BuiltIn(Arc::new(GeminiFilesStore::new(api_key)) as DynContentStore),
        }
    }

    /// Build a [`FalStorageStore`] backed by the platform `fetch`
    /// HTTP client.
    ///
    /// @param apiKey - The fal.ai API key.
    #[wasm_bindgen(js_name = "falStorage")]
    pub fn fal_storage(api_key: String) -> Self {
        Self {
            inner: Inner::BuiltIn(Arc::new(FalStorageStore::new(api_key)) as DynContentStore),
        }
    }

    /// Build a store backed by user-supplied async callbacks.
    ///
    /// Mirrors the Rust `CustomContentStore::builder` API and the
    /// Node / Python `ContentStore.custom(...)` factories. The `options`
    /// object must provide at least `put`, `resolve`, and `fetchBytes`;
    /// `fetchStream` and `delete` are optional. All callbacks must be
    /// `async` (or return a `Promise`).
    ///
    /// Argument shapes seen by JS — see the `CustomContentStoreOptions`
    /// TypeScript interface in the generated `.d.ts` for the canonical
    /// types:
    ///
    /// - `put(body, hint)`: `body` is a JSON-tagged
    ///   [`ContentBody`] — one of `{type: "bytes", data}`,
    ///   `{type: "url", url}`, `{type: "local_path", path}`,
    ///   `{type: "provider_file", provider, id}`, or
    ///   `{type: "stream", stream: ReadableStream<Uint8Array>, sizeHint}`
    ///   for streaming uploads. `hint` is a [`ContentHint`] dict (all
    ///   fields optional). Must resolve with a [`ContentHandle`]-shaped
    ///   object `{id, kind, mimeType?, byteSize?, displayName?}`.
    /// - `resolve(handle)`: `handle` is a [`ContentHandle`] dict. Must
    ///   resolve with a serialized [`MediaSource`] object
    ///   (e.g. `{type: "url", url: "..."}`).
    /// - `fetchBytes(handle)`: must resolve with a `Uint8Array` or
    ///   `number[]` of bytes.
    /// - `fetchStream(handle)` (optional): may resolve with a
    ///   `ReadableStream<Uint8Array>` for chunk-by-chunk delivery or with
    ///   `Uint8Array` / `number[]` for a single buffered chunk.
    /// - `delete(handle)` (optional): must resolve with `undefined`.
    ///
    /// # Errors
    ///
    /// Returns a `JsValue` error if any of the required methods is missing
    /// or is not a function.
    pub fn custom(options: CustomContentStoreOptionsTs) -> Result<WasmContentStore, JsValue> {
        let host = JsHostContentStore::from_js_object(options.into())?;
        Ok(Self {
            inner: Inner::BuiltIn(Arc::new(host) as DynContentStore),
        })
    }

    // -------- Async methods -----------------------------------------------

    /// Persist content and return a [`ContentHandle`].
    ///
    /// @param body         - One of:
    ///                       - `Uint8Array` (or `number[]`) of bytes,
    ///                       - `string` containing a public URL (depends on
    ///                         store support),
    ///                       - `ReadableStream<Uint8Array>` for streaming
    ///                         uploads (size hint inferred as `null`),
    ///                       - `{type: "stream", stream: ReadableStream<Uint8Array>, sizeHint?: number | null}`
    ///                         to attach a size hint alongside the stream.
    /// @param kindHint     - Optional kind override (`"image"`, `"audio"`,
    ///                       `"video"`, `"document"`, `"three_d_model"`,
    ///                       `"cad"`, `"archive"`, `"font"`, `"code"`,
    ///                       `"data"`). When omitted, the store auto-detects.
    /// @param mimeType     - Optional MIME type hint.
    /// @param displayName  - Optional display name (e.g. original filename).
    /// @returns `Promise<ContentHandle>` — the issued handle as a plain
    ///          JS object.
    pub fn put(
        &self,
        body: &ContentPutBodyTs,
        kind_hint: Option<String>,
        mime_type: Option<String>,
        display_name: Option<String>,
    ) -> PromiseContentHandle {
        let inner_res = self.require_built_in("put");
        let body_res = body_from_js(body.as_ref());
        let promise = future_to_promise(async move {
            let inner = inner_res?;
            let body = body_res?;
            let hint = build_hint(kind_hint.as_deref(), mime_type, display_name);
            let handle = inner
                .put(body, hint)
                .await
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
            serde_wasm_bindgen::to_value(&handle).map_err(|e| JsValue::from_str(&e.to_string()))
        });
        promise.unchecked_into()
    }

    /// Resolve a handle to a wire-renderable [`MediaSource`].
    ///
    /// @param handle - A [`ContentHandle`]-shaped object as produced by
    ///                 `put` (or constructed manually with at least an
    ///                 `id` and `kind`).
    /// @returns `Promise<MediaSource>` serialized as a plain JS object
    ///          (e.g. `{ type: "url", url: "..." }`).
    pub fn resolve(&self, handle: ContentHandleTs) -> PromiseMediaSource {
        let inner_res = self.require_built_in("resolve");
        let promise = future_to_promise(async move {
            let inner = inner_res?;
            let handle = handle_from_js(handle.into())?;
            let resolved = inner
                .resolve(&handle)
                .await
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
            serde_wasm_bindgen::to_value(&resolved).map_err(|e| JsValue::from_str(&e.to_string()))
        });
        promise.unchecked_into()
    }

    /// Fetch the raw bytes for a stored handle.
    ///
    /// @param handle - A [`ContentHandle`]-shaped object.
    /// @returns `Promise<Uint8Array>` — the raw bytes. Stores that record
    ///          references rather than bytes (e.g. URL inputs to the
    ///          in-memory store) reject the promise.
    #[wasm_bindgen(js_name = "fetchBytes")]
    pub fn fetch_bytes(&self, handle: ContentHandleTs) -> PromiseUint8Array {
        let inner_res = self.require_built_in("fetchBytes");
        let promise = future_to_promise(async move {
            let inner = inner_res?;
            let handle = handle_from_js(handle.into())?;
            let bytes = inner
                .fetch_bytes(&handle)
                .await
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
            // Copy the bytes into a JS-owned Uint8Array. `Uint8Array::from`
            // takes an `&[u8]` slice and allocates on the JS heap.
            let arr = Uint8Array::from(bytes.as_slice());
            Ok(arr.into())
        });
        promise.unchecked_into()
    }

    /// Fetch the bytes for a stored handle as a `ReadableStream<Uint8Array>`.
    ///
    /// Lets host code consume the content chunk-by-chunk via the standard
    /// WHATWG Streams API (`stream.getReader().read()`), avoiding the need
    /// to buffer the entire payload in memory the way [`Self::fetch_bytes`]
    /// does. Stores that don't natively stream fall back to a single-chunk
    /// stream wrapping the buffered bytes (the
    /// [`blazen_llm::content::ContentStore::fetch_stream`] default).
    ///
    /// @param handle - A [`ContentHandle`]-shaped object.
    /// @returns `Promise<ReadableStream<Uint8Array>>` — a live stream of
    ///          byte chunks. The promise rejects on dispatch errors (e.g.
    ///          subclass marker without an override); chunk-level errors
    ///          surface as a `ReadableStream` `error` once the consumer
    ///          starts reading.
    #[wasm_bindgen(js_name = "fetchStream")]
    pub fn fetch_stream(&self, handle: ContentHandleTs) -> PromiseReadableStream {
        let inner_res = self.require_built_in("fetchStream");
        let promise = future_to_promise(async move {
            let inner = inner_res?;
            let handle = handle_from_js(handle.into())?;
            let stream = inner
                .fetch_stream(&handle)
                .await
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
            let rs = byte_stream_to_readable_stream(stream).map_err(|e| {
                JsValue::from_str(&format!(
                    "ContentStore.fetchStream: failed to wrap ByteStream as ReadableStream: {e:?}"
                ))
            })?;
            Ok(rs.into())
        });
        promise.unchecked_into()
    }

    /// Delete a handle from the store. Idempotent for stores that
    /// implement deletion; a no-op for stores that don't track lifetime.
    ///
    /// @param handle - A [`ContentHandle`]-shaped object.
    /// @returns `Promise<void>`.
    pub fn delete(&self, handle: ContentHandleTs) -> PromiseVoid {
        let inner_res = self.require_built_in("delete");
        let promise = future_to_promise(async move {
            let inner = inner_res?;
            let handle = handle_from_js(handle.into())?;
            inner
                .delete(&handle)
                .await
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
            Ok(JsValue::UNDEFINED)
        });
        promise.unchecked_into()
    }
}

impl Default for WasmContentStore {
    fn default() -> Self {
        Self::new()
    }
}

impl WasmContentStore {
    /// Borrow the inner `Arc<dyn ContentStore>` for use by other crate
    /// modules (e.g. the agent runner) that need to wire a JS-supplied store
    /// into the Rust runtime. Returns `None` for subclass-marker instances;
    /// callers that must support subclasses should go through
    /// [`extract_store`] instead.
    #[must_use]
    pub fn inner_arc(&self) -> Option<DynContentStore> {
        match &self.inner {
            Inner::BuiltIn(inner) => Some(Arc::clone(inner)),
            Inner::Subclass => None,
        }
    }

    /// Default-method guard: the base-class `put` / `resolve` / `fetchBytes`
    /// / `delete` methods dispatch only on built-in / custom-callback
    /// stores. When a subclass forgets to override a required method and
    /// `super()` dispatch falls through, raise a clear JS error rather than
    /// silently looping back.
    fn require_built_in(&self, method: &str) -> Result<DynContentStore, JsValue> {
        match &self.inner {
            Inner::BuiltIn(inner) => Ok(Arc::clone(inner)),
            Inner::Subclass => Err(JsValue::from_str(&format!(
                "ContentStore subclass must override `{method}()` (called the base-class default)"
            ))),
        }
    }
}

/// Attempt to extract a [`DynContentStore`] from any JS value.
///
/// - If `value` is a [`WasmContentStore`] base-class / factory instance
///   wrapping a built-in store, unwrap and clone the inner `Arc`. (Detected
///   by the lack of an own `put` method on the JS object — built-in
///   instances inherit `put` from the wasm-bindgen-generated class
///   prototype, but we route everything through the host adapter for
///   simplicity and correctness; built-in stores work fine through that
///   path because their JS prototype has all required methods.)
/// - If `value` is a subclass instance overriding the trait surface, wrap
///   it in a [`JsHostContentStore`] adapter.
/// - If `value` is a plain options object with the right method shape, wrap
///   it as well — same path as `ContentStore.custom({...})`.
///
/// Returns a [`JsValue`] error if `value` does not present the required
/// methods (`put`, `resolve`, `fetchBytes`).
///
/// # Errors
///
/// Surfaces the underlying [`JsHostContentStore::from_js_object`] error if
/// any required method is missing.
#[allow(dead_code)]
pub(crate) fn extract_store(value: JsValue) -> Result<DynContentStore, JsValue> {
    let host = JsHostContentStore::from_js_object(value)?;
    Ok(Arc::new(host) as DynContentStore)
}

// ---------------------------------------------------------------------------
// Free schema-builder functions
// ---------------------------------------------------------------------------

/// Build a JSON Schema fragment declaring an image content-reference input.
///
/// Mirrors [`blazen_llm::content::tool_input::image_input`].
///
/// @param name        - Property name the model passes the handle id as.
/// @param description - Human-readable description for the model.
/// @returns The schema as a plain JS object.
#[wasm_bindgen(js_name = "imageInput")]
#[allow(clippy::missing_errors_doc)]
pub fn image_input(name: String, description: String) -> Result<JsValue, JsValue> {
    let schema = rust_image_input(name, description);
    serde_wasm_bindgen::to_value(&schema).map_err(|e| JsValue::from_str(&e.to_string()))
}

/// Build a JSON Schema fragment declaring an audio content-reference input.
///
/// Mirrors [`blazen_llm::content::tool_input::audio_input`].
#[wasm_bindgen(js_name = "audioInput")]
#[allow(clippy::missing_errors_doc)]
pub fn audio_input(name: String, description: String) -> Result<JsValue, JsValue> {
    let schema = rust_audio_input(name, description);
    serde_wasm_bindgen::to_value(&schema).map_err(|e| JsValue::from_str(&e.to_string()))
}

/// Build a JSON Schema fragment declaring a video content-reference input.
///
/// Mirrors [`blazen_llm::content::tool_input::video_input`].
#[wasm_bindgen(js_name = "videoInput")]
#[allow(clippy::missing_errors_doc)]
pub fn video_input(name: String, description: String) -> Result<JsValue, JsValue> {
    let schema = rust_video_input(name, description);
    serde_wasm_bindgen::to_value(&schema).map_err(|e| JsValue::from_str(&e.to_string()))
}

/// Build a JSON Schema fragment declaring a generic file/document
/// content-reference input.
///
/// Mirrors [`blazen_llm::content::tool_input::file_input`].
#[wasm_bindgen(js_name = "fileInput")]
#[allow(clippy::missing_errors_doc)]
pub fn file_input(name: String, description: String) -> Result<JsValue, JsValue> {
    let schema = rust_file_input(name, description);
    serde_wasm_bindgen::to_value(&schema).map_err(|e| JsValue::from_str(&e.to_string()))
}

/// Build a JSON Schema fragment declaring a 3D-model content-reference
/// input.
///
/// Mirrors [`blazen_llm::content::tool_input::three_d_input`].
#[wasm_bindgen(js_name = "threeDInput")]
#[allow(clippy::missing_errors_doc)]
pub fn three_d_input(name: String, description: String) -> Result<JsValue, JsValue> {
    let schema = rust_three_d_input(name, description);
    serde_wasm_bindgen::to_value(&schema).map_err(|e| JsValue::from_str(&e.to_string()))
}

/// Build a JSON Schema fragment declaring a CAD-file content-reference
/// input.
///
/// Mirrors [`blazen_llm::content::tool_input::cad_input`].
#[wasm_bindgen(js_name = "cadInput")]
#[allow(clippy::missing_errors_doc)]
pub fn cad_input(name: String, description: String) -> Result<JsValue, JsValue> {
    let schema = rust_cad_input(name, description);
    serde_wasm_bindgen::to_value(&schema).map_err(|e| JsValue::from_str(&e.to_string()))
}
