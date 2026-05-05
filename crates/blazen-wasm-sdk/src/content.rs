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

use std::sync::Arc;

use js_sys::{Promise, Uint8Array};
use wasm_bindgen::prelude::*;
use wasm_bindgen_futures::future_to_promise;

use blazen_llm::content::{
    ContentBody, ContentHandle, ContentHint, ContentKind, DynContentStore,
};
use blazen_llm::content::stores::{
    AnthropicFilesStore, FalStorageStore, GeminiFilesStore, InMemoryContentStore, OpenAiFilesStore,
};
use blazen_llm::content::tool_input::{
    audio_input as rust_audio_input, cad_input as rust_cad_input, file_input as rust_file_input,
    image_input as rust_image_input, three_d_input as rust_three_d_input,
    video_input as rust_video_input,
};

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

/// Convert a JS body (`Uint8Array` -> `Bytes`, `string` -> `Url`) into a
/// [`ContentBody`].
///
/// Returns a [`JsValue`] error if the value is neither a `Uint8Array` nor a
/// string.
fn body_from_js(value: &JsValue) -> Result<ContentBody, JsValue> {
    if let Some(s) = value.as_string() {
        return Ok(ContentBody::Url(s));
    }
    if value.is_instance_of::<Uint8Array>() {
        let arr = Uint8Array::from(value.clone());
        return Ok(ContentBody::Bytes(arr.to_vec()));
    }
    // Fall back: arrays of numbers serialize as a Vec<u8>
    if let Ok(bytes) = serde_wasm_bindgen::from_value::<Vec<u8>>(value.clone()) {
        return Ok(ContentBody::Bytes(bytes));
    }
    Err(JsValue::from_str(
        "ContentStore.put: body must be a Uint8Array (bytes) or a string (URL)",
    ))
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
// WasmContentStore
// ---------------------------------------------------------------------------

/// A type-erased reference to any [`blazen_llm::content::ContentStore`].
///
/// Construct one of the built-in implementations via the static factory
/// methods (`inMemory`, `openaiFiles`, `anthropicFiles`, `geminiFiles`,
/// `falStorage`); from JS the class is named `ContentStore`.
#[wasm_bindgen(js_name = "ContentStore")]
pub struct WasmContentStore {
    inner: DynContentStore,
}

// SAFETY: WASM is single-threaded; there is no other thread to race with.
unsafe impl Send for WasmContentStore {}
unsafe impl Sync for WasmContentStore {}

#[wasm_bindgen(js_class = "ContentStore")]
#[allow(clippy::must_use_candidate, clippy::missing_errors_doc)]
impl WasmContentStore {
    // -------- Static factories --------------------------------------------

    /// Build an in-process [`InMemoryContentStore`].
    ///
    /// Suitable for ephemeral / small content. Bytes are kept in WASM
    /// memory; URLs and provider-file references are recorded by
    /// reference.
    #[wasm_bindgen(js_name = "inMemory")]
    pub fn in_memory() -> Self {
        Self {
            inner: Arc::new(InMemoryContentStore::new()) as DynContentStore,
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
            inner: Arc::new(OpenAiFilesStore::new(api_key)) as DynContentStore,
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
            inner: Arc::new(AnthropicFilesStore::new(api_key)) as DynContentStore,
        }
    }

    /// Build a [`GeminiFilesStore`] backed by the platform `fetch`
    /// HTTP client.
    ///
    /// @param apiKey - The Google AI / Gemini API key.
    #[wasm_bindgen(js_name = "geminiFiles")]
    pub fn gemini_files(api_key: String) -> Self {
        Self {
            inner: Arc::new(GeminiFilesStore::new(api_key)) as DynContentStore,
        }
    }

    /// Build a [`FalStorageStore`] backed by the platform `fetch`
    /// HTTP client.
    ///
    /// @param apiKey - The fal.ai API key.
    #[wasm_bindgen(js_name = "falStorage")]
    pub fn fal_storage(api_key: String) -> Self {
        Self {
            inner: Arc::new(FalStorageStore::new(api_key)) as DynContentStore,
        }
    }

    // -------- Async methods -----------------------------------------------

    /// Persist content and return a [`ContentHandle`].
    ///
    /// @param body         - `Uint8Array` of bytes, or a `string` containing
    ///                       a public URL (depends on store support).
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
        body: &JsValue,
        kind_hint: Option<String>,
        mime_type: Option<String>,
        display_name: Option<String>,
    ) -> Promise {
        let inner = Arc::clone(&self.inner);
        let body_res = body_from_js(body);
        future_to_promise(async move {
            let body = body_res?;
            let hint = build_hint(kind_hint.as_deref(), mime_type, display_name);
            let handle = inner
                .put(body, hint)
                .await
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
            serde_wasm_bindgen::to_value(&handle)
                .map_err(|e| JsValue::from_str(&e.to_string()))
        })
    }

    /// Resolve a handle to a wire-renderable [`MediaSource`].
    ///
    /// @param handle - A [`ContentHandle`]-shaped object as produced by
    ///                 `put` (or constructed manually with at least an
    ///                 `id` and `kind`).
    /// @returns `Promise<MediaSource>` serialized as a plain JS object
    ///          (e.g. `{ type: "url", url: "..." }`).
    pub fn resolve(&self, handle: JsValue) -> Promise {
        let inner = Arc::clone(&self.inner);
        future_to_promise(async move {
            let handle = handle_from_js(handle)?;
            let resolved = inner
                .resolve(&handle)
                .await
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
            serde_wasm_bindgen::to_value(&resolved)
                .map_err(|e| JsValue::from_str(&e.to_string()))
        })
    }

    /// Fetch the raw bytes for a stored handle.
    ///
    /// @param handle - A [`ContentHandle`]-shaped object.
    /// @returns `Promise<Uint8Array>` — the raw bytes. Stores that record
    ///          references rather than bytes (e.g. URL inputs to the
    ///          in-memory store) reject the promise.
    #[wasm_bindgen(js_name = "fetchBytes")]
    pub fn fetch_bytes(&self, handle: JsValue) -> Promise {
        let inner = Arc::clone(&self.inner);
        future_to_promise(async move {
            let handle = handle_from_js(handle)?;
            let bytes = inner
                .fetch_bytes(&handle)
                .await
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
            // Copy the bytes into a JS-owned Uint8Array. `Uint8Array::from`
            // takes an `&[u8]` slice and allocates on the JS heap.
            let arr = Uint8Array::from(bytes.as_slice());
            Ok(arr.into())
        })
    }

    /// Delete a handle from the store. Idempotent for stores that
    /// implement deletion; a no-op for stores that don't track lifetime.
    ///
    /// @param handle - A [`ContentHandle`]-shaped object.
    /// @returns `Promise<void>`.
    pub fn delete(&self, handle: JsValue) -> Promise {
        let inner = Arc::clone(&self.inner);
        future_to_promise(async move {
            let handle = handle_from_js(handle)?;
            inner
                .delete(&handle)
                .await
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
            Ok(JsValue::UNDEFINED)
        })
    }
}

impl WasmContentStore {
    /// Borrow the inner `Arc<dyn ContentStore>` for use by other
    /// crate modules (e.g. the agent runner) that need to wire a
    /// JS-supplied store into the Rust runtime.
    #[must_use]
    pub fn inner_arc(&self) -> DynContentStore {
        Arc::clone(&self.inner)
    }
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
