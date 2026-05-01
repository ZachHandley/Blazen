//! `wasm-bindgen` wrapper for [`blazen_memory::Memory`].
//!
//! Exposes an in-memory vector store to JavaScript with `add`, `search`,
//! `searchLocal`, `get`, `delete`, and `count` methods. The store uses ELID
//! for vector indexing when an embedding model is provided, or falls back to
//! text-level `SimHash` in local-only mode.

use std::pin::Pin;
use std::sync::Arc;

use js_sys::Promise;
use serde::{Deserialize, Serialize};
use tsify_next::Tsify;
use wasm_bindgen::prelude::*;
use wasm_bindgen_futures::future_to_promise;

use blazen_memory::search as mem_search;
use blazen_memory::store::MemoryStore;
use blazen_memory::{InMemoryBackend, Memory, MemoryEntry, StoredEntry};

use crate::embedding::WasmEmbeddingModel;

/// Default seed used by `blazen-memory` when computing 128-bit embedding `SimHash`.
/// Matches `blazen_memory::memory::DEFAULT_SEED` ("ELIDSIMH").
const DEFAULT_EMBEDDING_SEED: u64 = 0x454c_4944_5349_4d48;

// ---------------------------------------------------------------------------
// TypeScript type declarations
// ---------------------------------------------------------------------------

#[wasm_bindgen(typescript_custom_section)]
const TS_MEMORY_TYPES: &str = r#"
/** A single search result returned by `Memory.search` / `Memory.searchLocal`. */
export interface MemorySearchResult {
    /** The entry identifier. */
    id: string;
    /** The original text content. */
    text: string;
    /** Similarity score in [0, 1] where higher means more similar. */
    score: number;
    /** Arbitrary metadata attached to the entry, or `null`. */
    metadata: any;
}

/** A stored entry returned by `Memory.get`. */
export interface MemoryEntry {
    /** The entry identifier. */
    id: string;
    /** The original text content. */
    text: string;
    /** Arbitrary metadata attached to the entry, or `null`. */
    metadata: any;
}
"#;

// ---------------------------------------------------------------------------
// WasmMemory
// ---------------------------------------------------------------------------

/// An in-memory vector store with ELID-based similarity search.
///
/// Supports two modes:
/// - **Full mode** (`new Memory(embedder)`): embedding-based search via `search()`
/// - **Local mode** (`Memory.local()`): text `SimHash` only via `searchLocal()`
///
/// ```js
/// import { Memory, EmbeddingModel } from '@blazen/sdk';
///
/// const embedder = EmbeddingModel.openai();
/// const memory = new Memory(embedder);
///
/// await memory.add("doc1", "Paris is the capital of France");
/// const results = await memory.search("What is France's capital?", 5);
/// console.log(results[0].text);
/// ```
#[wasm_bindgen(js_name = "Memory")]
pub struct WasmMemory {
    inner: Arc<Memory>,
}

// SAFETY: WASM is single-threaded.
unsafe impl Send for WasmMemory {}
unsafe impl Sync for WasmMemory {}

#[wasm_bindgen(js_class = "Memory")]
#[allow(
    clippy::must_use_candidate,
    clippy::cast_possible_truncation,
    clippy::missing_errors_doc
)]
impl WasmMemory {
    // -----------------------------------------------------------------------
    // Constructors
    // -----------------------------------------------------------------------

    /// Create a memory store with an embedding model for full ELID-based search.
    ///
    /// Uses an in-memory backend (data is lost on page reload).
    #[wasm_bindgen(constructor)]
    pub fn new(embedder: &WasmEmbeddingModel) -> Self {
        let inner = Memory::new(embedder.inner_arc(), InMemoryBackend::new());
        Self {
            inner: Arc::new(inner),
        }
    }

    /// Create a memory store in local-only mode (no embedding model).
    ///
    /// Only `searchLocal()` is available; `search()` will reject.
    /// Uses text-level `SimHash` for similarity.
    #[wasm_bindgen(js_name = "local")]
    pub fn local() -> Self {
        let inner = Memory::local(InMemoryBackend::new());
        Self {
            inner: Arc::new(inner),
        }
    }

    /// Create a memory store backed by a JavaScript object implementing
    /// the storage backend methods.
    ///
    /// The `backend` object must implement the following async methods:
    /// - `put(entry)` -- store or update an entry
    /// - `get(id)` -- retrieve an entry by ID, return `null` if not found
    /// - `delete(id)` -- delete an entry, return `true` if it existed
    /// - `list()` -- return all stored entries as an array
    /// - `len()` -- return the number of stored entries
    /// - `searchByBands(bands, limit)` -- return candidate entries matching
    ///   any of the given LSH band strings
    ///
    /// Each entry object has the shape:
    /// ```ts
    /// interface StoredEntry {
    ///   id: string;
    ///   text: string;
    ///   elid: string | null;
    ///   simhashHex: string | null;
    ///   textSimhash: number;
    ///   bands: string[];
    ///   metadata: any;
    /// }
    /// ```
    ///
    /// This enables plugging in custom storage backends (IndexedDB,
    /// localStorage, remote APIs) from JavaScript.
    ///
    /// ```js
    /// const backend = {
    ///   async put(entry) { localStorage.setItem(entry.id, JSON.stringify(entry)); },
    ///   async get(id) { const e = localStorage.getItem(id); return e ? JSON.parse(e) : null; },
    ///   async delete(id) { const had = !!localStorage.getItem(id); localStorage.removeItem(id); return had; },
    ///   async list() { return []; },
    ///   async len() { return localStorage.length; },
    ///   async searchByBands(bands, limit) { return []; },
    /// };
    /// const memory = Memory.fromJsBackend(embedder, backend);
    /// ```
    #[wasm_bindgen(js_name = "fromJsBackend")]
    pub fn from_js_backend(
        embedder: &WasmEmbeddingModel,
        backend: JsValue,
    ) -> Result<WasmMemory, JsValue> {
        // Validate that the backend has the required methods.
        let required_methods = ["put", "get", "delete", "list", "len", "searchByBands"];
        for method in &required_methods {
            let val = js_sys::Reflect::get(&backend, &JsValue::from_str(method))
                .map_err(|_| JsValue::from_str(&format!("backend missing method '{method}'")))?;
            if !val.is_function() {
                return Err(JsValue::from_str(&format!(
                    "backend.{method} must be a function"
                )));
            }
        }

        let js_backend = JsMemoryBackend::new(backend);
        let inner = Memory::new(embedder.inner_arc(), js_backend);
        Ok(Self {
            inner: Arc::new(inner),
        })
    }

    /// Create a memory store in local-only mode backed by a JavaScript
    /// storage backend (no embedding model).
    ///
    /// See [`fromJsBackend`] for the backend object interface.
    #[wasm_bindgen(js_name = "localFromJsBackend")]
    pub fn local_from_js_backend(backend: JsValue) -> Result<WasmMemory, JsValue> {
        let required_methods = ["put", "get", "delete", "list", "len", "searchByBands"];
        for method in &required_methods {
            let val = js_sys::Reflect::get(&backend, &JsValue::from_str(method))
                .map_err(|_| JsValue::from_str(&format!("backend missing method '{method}'")))?;
            if !val.is_function() {
                return Err(JsValue::from_str(&format!(
                    "backend.{method} must be a function"
                )));
            }
        }

        let js_backend = JsMemoryBackend::new(backend);
        let inner = Memory::local(js_backend);
        Ok(Self {
            inner: Arc::new(inner),
        })
    }

    /// Create a memory store backed by a typed [`WasmInMemoryBackend`].
    ///
    /// Unlike [`fromJsBackend`], this consumes a Rust-native `InMemoryBackend`
    /// wrapper (no JS round-trips per call), so reads and writes stay inside
    /// the WASM linear memory.
    ///
    /// ```js
    /// import { Memory, InMemoryBackend, EmbeddingModel } from '@blazen/sdk';
    ///
    /// const embedder = EmbeddingModel.openai();
    /// const backend = new InMemoryBackend();
    /// const memory = Memory.fromBackend(embedder, backend);
    /// ```
    #[wasm_bindgen(js_name = "fromBackend")]
    #[must_use]
    pub fn from_backend(
        embedder: &WasmEmbeddingModel,
        backend: &WasmInMemoryBackend,
    ) -> WasmMemory {
        let arc: Arc<dyn blazen_memory::store::MemoryBackend> = backend.inner.clone();
        let inner = Memory::new_arc(embedder.inner_arc(), arc);
        Self {
            inner: Arc::new(inner),
        }
    }

    /// Create a memory store in local-only mode backed by a typed
    /// [`WasmInMemoryBackend`] (no embedding model).
    ///
    /// See [`fromBackend`] for the typed-backend rationale.
    #[wasm_bindgen(js_name = "localFromBackend")]
    #[must_use]
    pub fn local_from_backend(backend: &WasmInMemoryBackend) -> WasmMemory {
        let arc: Arc<dyn blazen_memory::store::MemoryBackend> = backend.inner.clone();
        let inner = Memory::local_arc(arc);
        Self {
            inner: Arc::new(inner),
        }
    }

    // -----------------------------------------------------------------------
    // Mutation
    // -----------------------------------------------------------------------

    /// Add a text entry to the memory store.
    ///
    /// @param id       - A unique identifier for this entry (pass `""` to auto-generate).
    /// @param text     - The text content to store.
    /// @param metadata - Optional arbitrary metadata (JSON-serializable value).
    /// @returns A `Promise<string>` resolving to the stored entry's id.
    pub fn add(&self, id: String, text: String, metadata: JsValue) -> Promise {
        let inner = Arc::clone(&self.inner);
        future_to_promise(async move {
            let meta: serde_json::Value = if metadata.is_undefined() || metadata.is_null() {
                serde_json::Value::Null
            } else {
                serde_wasm_bindgen::from_value(metadata)
                    .map_err(|e| JsValue::from_str(&e.to_string()))?
            };

            let entry = MemoryEntry {
                id,
                text,
                metadata: meta,
            };

            let ids = inner
                .add(vec![entry])
                .await
                .map_err(|e| JsValue::from_str(&e.to_string()))?;

            Ok(JsValue::from_str(ids.first().map_or("", String::as_str)))
        })
    }

    /// Add multiple text entries in a single batch.
    ///
    /// @param entries - A JS array of `{ id: string, text: string, metadata?: any }` objects.
    /// @returns A `Promise<string[]>` resolving to the stored entry ids.
    #[wasm_bindgen(js_name = "addMany")]
    pub fn add_many(&self, entries: JsValue) -> Promise {
        let inner = Arc::clone(&self.inner);
        future_to_promise(async move {
            let raw_entries: Vec<RawAddEntry> = serde_wasm_bindgen::from_value(entries)
                .map_err(|e| JsValue::from_str(&e.to_string()))?;

            let entries: Vec<MemoryEntry> = raw_entries
                .into_iter()
                .map(|e| MemoryEntry {
                    id: e.id,
                    text: e.text,
                    metadata: e.metadata.unwrap_or(serde_json::Value::Null),
                })
                .collect();

            let ids = inner
                .add(entries)
                .await
                .map_err(|e| JsValue::from_str(&e.to_string()))?;

            let arr = js_sys::Array::new_with_length(ids.len() as u32);
            for (i, id) in ids.iter().enumerate() {
                arr.set(i as u32, JsValue::from_str(id));
            }
            Ok(arr.into())
        })
    }

    // -----------------------------------------------------------------------
    // Search
    // -----------------------------------------------------------------------

    /// Search using a query string with the configured embedding model.
    ///
    /// Returns up to `limit` results sorted by descending similarity.
    /// Requires an embedding model (rejects if created with `Memory.local()`).
    ///
    /// @param query           - The search query text.
    /// @param limit           - Maximum number of results (default: 10).
    /// @param `metadata_filter` - Optional JSON object; only entries whose metadata
    ///   is a superset of the filter are returned.
    /// @returns A `Promise<MemorySearchResult[]>`.
    pub fn search(&self, query: String, limit: Option<u32>, metadata_filter: JsValue) -> Promise {
        let inner = Arc::clone(&self.inner);
        future_to_promise(async move {
            let limit = limit.unwrap_or(10) as usize;
            let filter: Option<serde_json::Value> =
                if metadata_filter.is_undefined() || metadata_filter.is_null() {
                    None
                } else {
                    Some(
                        serde_wasm_bindgen::from_value(metadata_filter)
                            .map_err(|e| JsValue::from_str(&e.to_string()))?,
                    )
                };

            let results = inner
                .search(&query, limit, filter.as_ref())
                .await
                .map_err(|e| JsValue::from_str(&e.to_string()))?;

            results_to_js_array(results)
        })
    }

    /// Search using only text-level `SimHash` (no embedding model required).
    ///
    /// This is a cheaper, lower-quality search that works in local-only mode.
    ///
    /// @param query           - The search query text.
    /// @param limit           - Maximum number of results (default: 10).
    /// @param `metadata_filter` - Optional JSON object to filter results.
    /// @returns A `Promise<MemorySearchResult[]>`.
    #[wasm_bindgen(js_name = "searchLocal")]
    pub fn search_local(
        &self,
        query: String,
        limit: Option<u32>,
        metadata_filter: JsValue,
    ) -> Promise {
        let inner = Arc::clone(&self.inner);
        future_to_promise(async move {
            let limit = limit.unwrap_or(10) as usize;
            let filter: Option<serde_json::Value> =
                if metadata_filter.is_undefined() || metadata_filter.is_null() {
                    None
                } else {
                    Some(
                        serde_wasm_bindgen::from_value(metadata_filter)
                            .map_err(|e| JsValue::from_str(&e.to_string()))?,
                    )
                };

            let results = inner
                .search_local(&query, limit, filter.as_ref())
                .await
                .map_err(|e| JsValue::from_str(&e.to_string()))?;

            results_to_js_array(results)
        })
    }

    // -----------------------------------------------------------------------
    // Single-entry operations
    // -----------------------------------------------------------------------

    /// Retrieve a single entry by id.
    ///
    /// @param id - The entry identifier.
    /// @returns A `Promise<MemoryEntry | null>`.
    pub fn get(&self, id: String) -> Promise {
        let inner = Arc::clone(&self.inner);
        future_to_promise(async move {
            let entry = inner
                .get(&id)
                .await
                .map_err(|e| JsValue::from_str(&e.to_string()))?;

            match entry {
                Some(e) => {
                    let obj = js_sys::Object::new();
                    js_sys::Reflect::set(&obj, &"id".into(), &e.id.into())?;
                    js_sys::Reflect::set(&obj, &"text".into(), &e.text.into())?;
                    let meta = serde_wasm_bindgen::to_value(&e.metadata).unwrap_or(JsValue::NULL);
                    js_sys::Reflect::set(&obj, &"metadata".into(), &meta)?;
                    Ok(obj.into())
                }
                None => Ok(JsValue::NULL),
            }
        })
    }

    /// Delete an entry by id.
    ///
    /// @param id - The entry identifier.
    /// @returns A `Promise<boolean>` — `true` if the entry existed and was deleted.
    pub fn delete(&self, id: String) -> Promise {
        let inner = Arc::clone(&self.inner);
        future_to_promise(async move {
            let deleted = inner
                .delete(&id)
                .await
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
            Ok(JsValue::from_bool(deleted))
        })
    }

    /// Return the number of entries in the store.
    ///
    /// @returns A `Promise<number>`.
    #[allow(clippy::cast_precision_loss)]
    pub fn count(&self) -> Promise {
        let inner = Arc::clone(&self.inner);
        future_to_promise(async move {
            let len = inner
                .len()
                .await
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
            Ok(JsValue::from_f64(len as f64))
        })
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Convert a `Vec<MemoryResult>` into a JS `Array` of plain objects.
#[allow(clippy::cast_possible_truncation)]
fn results_to_js_array(results: Vec<blazen_memory::MemoryResult>) -> Result<JsValue, JsValue> {
    let arr = js_sys::Array::new_with_length(results.len() as u32);
    for (i, r) in results.into_iter().enumerate() {
        let obj = js_sys::Object::new();
        js_sys::Reflect::set(&obj, &"id".into(), &r.id.into())?;
        js_sys::Reflect::set(&obj, &"text".into(), &r.text.into())?;
        js_sys::Reflect::set(&obj, &"score".into(), &JsValue::from_f64(r.score))?;
        let meta = serde_wasm_bindgen::to_value(&r.metadata).unwrap_or(JsValue::NULL);
        js_sys::Reflect::set(&obj, &"metadata".into(), &meta)?;
        arr.set(i as u32, obj.into());
    }
    Ok(arr.into())
}

// ---------------------------------------------------------------------------
// Serde helper for addMany input
// ---------------------------------------------------------------------------

#[derive(serde::Deserialize)]
struct RawAddEntry {
    id: String,
    text: String,
    metadata: Option<serde_json::Value>,
}

// ---------------------------------------------------------------------------
// SendFuture wrapper (same pattern as agent.rs / js_completion.rs)
// ---------------------------------------------------------------------------

/// Wrapper that unsafely implements `Send` for a non-Send future.
/// SAFETY: WASM is single-threaded.
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
// JsMemoryBackend — MemoryBackend backed by a JS object
// ---------------------------------------------------------------------------

/// A [`MemoryBackend`] implementation that delegates to a JavaScript object
/// with `put`, `get`, `delete`, `list`, `len`, and `searchByBands` methods.
struct JsMemoryBackend {
    backend: JsValue,
}

// SAFETY: WASM is single-threaded.
unsafe impl Send for JsMemoryBackend {}
unsafe impl Sync for JsMemoryBackend {}

impl JsMemoryBackend {
    fn new(backend: JsValue) -> Self {
        Self { backend }
    }

    /// Call a method on the JS backend object with the given arguments and
    /// await the result if it's a Promise.
    async fn call_method(
        &self,
        method: &str,
        args: &[JsValue],
    ) -> std::result::Result<JsValue, blazen_memory::MemoryError> {
        let func =
            js_sys::Reflect::get(&self.backend, &JsValue::from_str(method)).map_err(|e| {
                blazen_memory::MemoryError::Backend(format!("backend.{method} not found: {e:?}"))
            })?;

        let func: &js_sys::Function = func.unchecked_ref();
        let result = match args.len() {
            0 => func.call0(&self.backend),
            1 => func.call1(&self.backend, &args[0]),
            2 => func.call2(&self.backend, &args[0], &args[1]),
            _ => {
                let js_args = js_sys::Array::new();
                for arg in args {
                    js_args.push(arg);
                }
                func.apply(&self.backend, &js_args)
            }
        }
        .map_err(|e| {
            blazen_memory::MemoryError::Backend(format!("backend.{method}() threw: {e:?}"))
        })?;

        // Await if the result is a Promise.
        if result.has_type::<js_sys::Promise>() {
            let promise: js_sys::Promise = result.unchecked_into();
            wasm_bindgen_futures::JsFuture::from(promise)
                .await
                .map_err(|e| {
                    blazen_memory::MemoryError::Backend(format!(
                        "backend.{method}() rejected: {e:?}"
                    ))
                })
        } else {
            Ok(result)
        }
    }

    /// Convert a `StoredEntry` to a JS object for the backend.
    fn entry_to_js(entry: &StoredEntry) -> JsValue {
        let obj = js_sys::Object::new();
        let _ = js_sys::Reflect::set(&obj, &"id".into(), &entry.id.clone().into());
        let _ = js_sys::Reflect::set(&obj, &"text".into(), &entry.text.clone().into());
        let _ = js_sys::Reflect::set(
            &obj,
            &"elid".into(),
            &entry
                .elid
                .as_deref()
                .map_or(JsValue::NULL, |s| JsValue::from_str(s)),
        );
        let _ = js_sys::Reflect::set(
            &obj,
            &"simhashHex".into(),
            &entry
                .simhash_hex
                .as_deref()
                .map_or(JsValue::NULL, |s| JsValue::from_str(s)),
        );
        let _ = js_sys::Reflect::set(
            &obj,
            &"textSimhash".into(),
            &JsValue::from_f64(entry.text_simhash as f64),
        );
        let bands = js_sys::Array::new();
        for band in &entry.bands {
            bands.push(&JsValue::from_str(band));
        }
        let _ = js_sys::Reflect::set(&obj, &"bands".into(), &bands);
        let meta = serde_wasm_bindgen::to_value(&entry.metadata).unwrap_or(JsValue::NULL);
        let _ = js_sys::Reflect::set(&obj, &"metadata".into(), &meta);
        obj.into()
    }

    /// Convert a JS object back to a `StoredEntry`.
    fn js_to_entry(val: &JsValue) -> std::result::Result<StoredEntry, blazen_memory::MemoryError> {
        let id = js_sys::Reflect::get(val, &"id".into())
            .ok()
            .and_then(|v| v.as_string())
            .unwrap_or_default();
        let text = js_sys::Reflect::get(val, &"text".into())
            .ok()
            .and_then(|v| v.as_string())
            .unwrap_or_default();
        let elid = js_sys::Reflect::get(val, &"elid".into())
            .ok()
            .and_then(|v| v.as_string());
        let simhash_hex = js_sys::Reflect::get(val, &"simhashHex".into())
            .ok()
            .and_then(|v| v.as_string());
        #[allow(clippy::cast_sign_loss, clippy::cast_possible_truncation)]
        let text_simhash = js_sys::Reflect::get(val, &"textSimhash".into())
            .ok()
            .and_then(|v| v.as_f64())
            .map_or(0, |v| v as u64);
        let bands_val = js_sys::Reflect::get(val, &"bands".into()).unwrap_or(JsValue::UNDEFINED);
        let bands: Vec<String> = if bands_val.is_instance_of::<js_sys::Array>() {
            let arr: &js_sys::Array = bands_val.unchecked_ref();
            arr.iter().filter_map(|v| v.as_string()).collect()
        } else {
            Vec::new()
        };
        let metadata_val = js_sys::Reflect::get(val, &"metadata".into()).unwrap_or(JsValue::NULL);
        let metadata: serde_json::Value = if metadata_val.is_null() || metadata_val.is_undefined() {
            serde_json::Value::Null
        } else {
            serde_wasm_bindgen::from_value(metadata_val).unwrap_or(serde_json::Value::Null)
        };

        Ok(StoredEntry {
            id,
            text,
            elid,
            simhash_hex,
            text_simhash,
            bands,
            metadata,
        })
    }
}

#[async_trait::async_trait]
impl blazen_memory::store::MemoryBackend for JsMemoryBackend {
    async fn put(&self, entry: StoredEntry) -> blazen_memory::error::Result<()> {
        let js_entry = Self::entry_to_js(&entry);
        SendFuture(self.call_method("put", &[js_entry])).await?;
        Ok(())
    }

    async fn get(&self, id: &str) -> blazen_memory::error::Result<Option<StoredEntry>> {
        let result = SendFuture(self.call_method("get", &[JsValue::from_str(id)])).await?;
        if result.is_null() || result.is_undefined() {
            Ok(None)
        } else {
            Ok(Some(Self::js_to_entry(&result)?))
        }
    }

    async fn delete(&self, id: &str) -> blazen_memory::error::Result<bool> {
        let result = SendFuture(self.call_method("delete", &[JsValue::from_str(id)])).await?;
        Ok(result.is_truthy())
    }

    async fn list(&self) -> blazen_memory::error::Result<Vec<StoredEntry>> {
        let result = SendFuture(self.call_method("list", &[])).await?;
        let arr: &js_sys::Array = result.dyn_ref::<js_sys::Array>().ok_or_else(|| {
            blazen_memory::MemoryError::Backend("backend.list() must return an array".into())
        })?;

        let mut entries = Vec::with_capacity(arr.length() as usize);
        for i in 0..arr.length() {
            entries.push(Self::js_to_entry(&arr.get(i))?);
        }
        Ok(entries)
    }

    #[allow(clippy::cast_sign_loss, clippy::cast_possible_truncation)]
    async fn len(&self) -> blazen_memory::error::Result<usize> {
        let result = SendFuture(self.call_method("len", &[])).await?;
        Ok(result.as_f64().unwrap_or(0.0) as usize)
    }

    async fn search_by_bands(
        &self,
        bands: &[String],
        limit: usize,
    ) -> blazen_memory::error::Result<Vec<StoredEntry>> {
        let js_bands = js_sys::Array::new();
        for band in bands {
            js_bands.push(&JsValue::from_str(band));
        }
        let result = SendFuture(self.call_method(
            "searchByBands",
            &[js_bands.into(), JsValue::from_f64(limit as f64)],
        ))
        .await?;

        let arr: &js_sys::Array = result.dyn_ref::<js_sys::Array>().ok_or_else(|| {
            blazen_memory::MemoryError::Backend(
                "backend.searchByBands() must return an array".into(),
            )
        })?;

        let mut entries = Vec::with_capacity(arr.length() as usize);
        for i in 0..arr.length() {
            entries.push(Self::js_to_entry(&arr.get(i))?);
        }
        Ok(entries)
    }
}

// ---------------------------------------------------------------------------
// WasmStoredEntry — tsify mirror of `blazen_memory::StoredEntry`
// ---------------------------------------------------------------------------

/// Plain-data mirror of [`blazen_memory::StoredEntry`] for JS interop.
///
/// Field names are camelCased to match the JS backend convention used by
/// [`WasmMemory::from_js_backend`].
#[derive(Debug, Clone, Serialize, Deserialize, Tsify)]
#[tsify(into_wasm_abi, from_wasm_abi)]
#[serde(rename_all = "camelCase")]
pub struct WasmStoredEntry {
    /// Unique identifier for this entry.
    pub id: String,
    /// The original text content.
    pub text: String,
    /// ELID-encoded embedding string (`null` in local-only mode).
    pub elid: Option<String>,
    /// 128-bit embedding `SimHash` as a 32-character hex string (`null` in local mode).
    pub simhash_hex: Option<String>,
    /// String-level `SimHash` of the raw text (always available).
    /// Serialized as a JS number; values fit safely within 2^53 only for the low
    /// 53 bits, so for round-trip fidelity prefer [`WasmStoredEntry::simhash_hex`].
    pub text_simhash: u64,
    /// LSH band strings derived from the embedding `SimHash`.
    pub bands: Vec<String>,
    /// Arbitrary user metadata.
    pub metadata: serde_json::Value,
}

impl From<StoredEntry> for WasmStoredEntry {
    fn from(value: StoredEntry) -> Self {
        Self {
            id: value.id,
            text: value.text,
            elid: value.elid,
            simhash_hex: value.simhash_hex,
            text_simhash: value.text_simhash,
            bands: value.bands,
            metadata: value.metadata,
        }
    }
}

impl From<WasmStoredEntry> for StoredEntry {
    fn from(value: WasmStoredEntry) -> Self {
        Self {
            id: value.id,
            text: value.text,
            elid: value.elid,
            simhash_hex: value.simhash_hex,
            text_simhash: value.text_simhash,
            bands: value.bands,
            metadata: value.metadata,
        }
    }
}

// ---------------------------------------------------------------------------
// Free similarity helpers
// ---------------------------------------------------------------------------

/// Compute text-level `SimHash` similarity between two strings.
///
/// Hashes both inputs with `elid::simhash` and returns
/// `1.0 - hammingDistance / 64.0` in `[0, 1]`.
///
/// @param a - First string.
/// @param b - Second string.
/// @returns Similarity score where `1.0` means identical text hashes.
#[wasm_bindgen(js_name = "computeTextSimhashSimilarity")]
#[must_use]
pub fn compute_text_simhash_similarity(a: String, b: String) -> f64 {
    let ha = elid::simhash(&a);
    let hb = elid::simhash(&b);
    mem_search::compute_text_simhash_similarity(ha, hb)
}

/// Compute 128-bit `SimHash` similarity between two embedding vectors.
///
/// Computes `simhash_128` over each embedding using the default Blazen seed
/// (`"ELIDSIMH"`) and returns `1.0 - hammingDistance / 128.0` in `[0, 1]`.
///
/// @param a - First embedding (any dimensionality, must match `b`).
/// @param b - Second embedding.
/// @returns Similarity score where `1.0` means identical hashes.
#[wasm_bindgen(js_name = "computeEmbeddingSimhashSimilarity")]
#[must_use]
#[allow(clippy::cast_possible_truncation)]
pub fn compute_embedding_simhash_similarity(a: Vec<f64>, b: Vec<f64>) -> f64 {
    let a_f32: Vec<f32> = a.iter().map(|&v| v as f32).collect();
    let b_f32: Vec<f32> = b.iter().map(|&v| v as f32).collect();
    let ha = elid::embeddings::vector_simhash::simhash_128(&a_f32, DEFAULT_EMBEDDING_SEED);
    let hb = elid::embeddings::vector_simhash::simhash_128(&b_f32, DEFAULT_EMBEDDING_SEED);
    mem_search::compute_embedding_simhash_similarity(ha, hb)
}

/// Compute ELID similarity between two embedding vectors.
///
/// Encodes both embeddings with the default Mini128 ELID profile, then
/// returns `1.0 - hammingDistance / 128.0` in `[0, 1]`.
///
/// @param a - First embedding (any dimensionality, must match `b`).
/// @param b - Second embedding.
/// @returns Similarity score where `1.0` means identical ELIDs.
/// @throws If either embedding is invalid (e.g. empty or contains NaN).
#[wasm_bindgen(js_name = "computeElidSimilarity")]
#[allow(clippy::cast_possible_truncation, clippy::missing_errors_doc)]
pub fn compute_elid_similarity(a: Vec<f64>, b: Vec<f64>) -> Result<f64, JsError> {
    let a_f32: Vec<f32> = a.iter().map(|&v| v as f32).collect();
    let b_f32: Vec<f32> = b.iter().map(|&v| v as f32).collect();
    let profile = elid::embeddings::Profile::default();
    let elid_a = elid::embeddings::encode(&a_f32, &profile)
        .map_err(|e| JsError::new(&format!("ELID encode error (a): {e}")))?;
    let elid_b = elid::embeddings::encode(&b_f32, &profile)
        .map_err(|e| JsError::new(&format!("ELID encode error (b): {e}")))?;
    mem_search::compute_elid_similarity(&elid_a.to_string(), &elid_b.to_string())
        .map_err(|e| JsError::new(&e.to_string()))
}

/// Parse a hex-encoded text-level `SimHash` (u64) into a JS `bigint`.
///
/// Accepts up to 16 hex characters (case-insensitive, no `0x` prefix).
///
/// @param hex - Hex-encoded u64.
/// @returns The parsed value as a `bigint`.
/// @throws If the input is not a valid u64 hex string.
#[wasm_bindgen(js_name = "simhashFromHex")]
#[allow(clippy::missing_errors_doc)]
pub fn simhash_from_hex(hex: String) -> Result<u64, JsError> {
    u64::from_str_radix(hex.trim_start_matches("0x"), 16)
        .map_err(|e| JsError::new(&format!("invalid simhash hex: {e}")))
}

/// Encode a u64 `SimHash` as a zero-padded 16-character hex string.
///
/// @param value - The `SimHash` value as a `bigint`.
/// @returns Lowercase hex string of length 16.
#[wasm_bindgen(js_name = "simhashToHex")]
#[must_use]
pub fn simhash_to_hex(value: u64) -> String {
    format!("{value:016x}")
}

// ---------------------------------------------------------------------------
// WasmMemoryStore — JS-backed `MemoryStore` ABC
// ---------------------------------------------------------------------------

#[wasm_bindgen(typescript_custom_section)]
const TS_MEMORY_STORE: &str = r#"
/**
 * High-level memory store interface that JavaScript callers can implement to
 * back `MemoryStore`. Each method may return a value or a Promise that
 * resolves to that value.
 *
 * - `add(entries)` -- store one or more `{ id, text, metadata }` entries and
 *   return their final ids.
 * - `search(query, limit, metadataFilter)` -- embedding-based search.
 * - `searchLocal(query, limit, metadataFilter)` -- text-`SimHash` search.
 * - `get(id)` -- fetch a stored entry, or `null` if not found.
 * - `delete(id)` -- delete an entry, returning `true` if it existed.
 * - `len()` -- entry count.
 */
export interface MemoryStoreImpl {
    add(entries: Array<{ id: string; text: string; metadata: any }>): Promise<string[]> | string[];
    search(query: string, limit: number, metadataFilter: any): Promise<Array<{ id: string; text: string; score: number; metadata: any }>> | Array<{ id: string; text: string; score: number; metadata: any }>;
    searchLocal(query: string, limit: number, metadataFilter: any): Promise<Array<{ id: string; text: string; score: number; metadata: any }>> | Array<{ id: string; text: string; score: number; metadata: any }>;
    get(id: string): Promise<{ id: string; text: string; metadata: any } | null> | { id: string; text: string; metadata: any } | null;
    delete(id: string): Promise<boolean> | boolean;
    len(): Promise<number> | number;
}
"#;

/// JS-backed [`MemoryStore`](blazen_memory::store::MemoryStore) ABC.
///
/// Wraps a JavaScript object whose methods implement the
/// [`MemoryStore`](blazen_memory::store::MemoryStore) surface, so callers can
/// plug entire memory stores (`mem0`, vector DBs, etc.) into Blazen without
/// going through the `MemoryBackend` layer.
///
/// ```js
/// import { MemoryStore } from '@blazen/sdk';
///
/// const store = new MemoryStore({
///     async add(entries) { return entries.map(e => e.id || crypto.randomUUID()); },
///     async search(query, limit, filter) { return []; },
///     async searchLocal(query, limit, filter) { return []; },
///     async get(id) { return null; },
///     async delete(id) { return false; },
///     async len() { return 0; },
/// });
/// ```
#[wasm_bindgen(js_name = "MemoryStore")]
pub struct WasmMemoryStore {
    inner: Arc<JsMemoryStore>,
}

// SAFETY: WASM is single-threaded.
unsafe impl Send for WasmMemoryStore {}
unsafe impl Sync for WasmMemoryStore {}

#[wasm_bindgen(js_class = "MemoryStore")]
#[allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]
impl WasmMemoryStore {
    /// Wrap a JS object implementing the `MemoryStoreImpl` interface.
    ///
    /// # Errors
    ///
    /// Returns a `JsValue` error if `impl_obj` is missing one of the required
    /// methods (`add`, `search`, `searchLocal`, `get`, `delete`, `len`).
    #[wasm_bindgen(constructor)]
    pub fn new(impl_obj: JsValue) -> Result<WasmMemoryStore, JsValue> {
        let required = ["add", "search", "searchLocal", "get", "delete", "len"];
        for method in &required {
            let val = js_sys::Reflect::get(&impl_obj, &JsValue::from_str(method))
                .map_err(|_| JsValue::from_str(&format!("MemoryStore impl missing '{method}'")))?;
            if !val.is_function() {
                return Err(JsValue::from_str(&format!(
                    "MemoryStore.{method} must be a function"
                )));
            }
        }
        Ok(Self {
            inner: Arc::new(JsMemoryStore::new(impl_obj)),
        })
    }

    /// Add entries via the JS-backed store. Returns a `Promise<string[]>`.
    pub fn add(&self, entries: JsValue) -> Promise {
        let inner = Arc::clone(&self.inner);
        future_to_promise(async move {
            let raw_entries: Vec<RawAddEntry> = serde_wasm_bindgen::from_value(entries)
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
            let entries: Vec<MemoryEntry> = raw_entries
                .into_iter()
                .map(|e| MemoryEntry {
                    id: e.id,
                    text: e.text,
                    metadata: e.metadata.unwrap_or(serde_json::Value::Null),
                })
                .collect();
            let ids = blazen_memory::store::MemoryStore::add(&*inner, entries)
                .await
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
            let arr = js_sys::Array::new_with_length(ids.len() as u32);
            for (i, id) in ids.iter().enumerate() {
                arr.set(i as u32, JsValue::from_str(id));
            }
            Ok(arr.into())
        })
    }

    /// Embedding-based search. Returns a `Promise<MemorySearchResult[]>`.
    pub fn search(&self, query: String, limit: Option<u32>, metadata_filter: JsValue) -> Promise {
        let inner = Arc::clone(&self.inner);
        future_to_promise(async move {
            let limit = limit.unwrap_or(10) as usize;
            let filter: Option<serde_json::Value> =
                if metadata_filter.is_undefined() || metadata_filter.is_null() {
                    None
                } else {
                    Some(
                        serde_wasm_bindgen::from_value(metadata_filter)
                            .map_err(|e| JsValue::from_str(&e.to_string()))?,
                    )
                };
            let results =
                blazen_memory::store::MemoryStore::search(&*inner, &query, limit, filter.as_ref())
                    .await
                    .map_err(|e| JsValue::from_str(&e.to_string()))?;
            results_to_js_array(results)
        })
    }

    /// `SimHash`-only search. Returns a `Promise<MemorySearchResult[]>`.
    #[wasm_bindgen(js_name = "searchLocal")]
    pub fn search_local(
        &self,
        query: String,
        limit: Option<u32>,
        metadata_filter: JsValue,
    ) -> Promise {
        let inner = Arc::clone(&self.inner);
        future_to_promise(async move {
            let limit = limit.unwrap_or(10) as usize;
            let filter: Option<serde_json::Value> =
                if metadata_filter.is_undefined() || metadata_filter.is_null() {
                    None
                } else {
                    Some(
                        serde_wasm_bindgen::from_value(metadata_filter)
                            .map_err(|e| JsValue::from_str(&e.to_string()))?,
                    )
                };
            let results = blazen_memory::store::MemoryStore::search_local(
                &*inner,
                &query,
                limit,
                filter.as_ref(),
            )
            .await
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
            results_to_js_array(results)
        })
    }

    /// Retrieve a single stored entry. Returns a `Promise<MemoryEntry | null>`.
    pub fn get(&self, id: String) -> Promise {
        let inner = Arc::clone(&self.inner);
        future_to_promise(async move {
            let entry = blazen_memory::store::MemoryStore::get(&*inner, &id)
                .await
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
            match entry {
                Some(e) => {
                    let obj = js_sys::Object::new();
                    js_sys::Reflect::set(&obj, &"id".into(), &e.id.into())?;
                    js_sys::Reflect::set(&obj, &"text".into(), &e.text.into())?;
                    let meta = serde_wasm_bindgen::to_value(&e.metadata).unwrap_or(JsValue::NULL);
                    js_sys::Reflect::set(&obj, &"metadata".into(), &meta)?;
                    Ok(obj.into())
                }
                None => Ok(JsValue::NULL),
            }
        })
    }

    /// Delete an entry. Returns a `Promise<boolean>`.
    pub fn delete(&self, id: String) -> Promise {
        let inner = Arc::clone(&self.inner);
        future_to_promise(async move {
            let deleted = blazen_memory::store::MemoryStore::delete(&*inner, &id)
                .await
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
            Ok(JsValue::from_bool(deleted))
        })
    }

    /// Return the number of entries. Returns a `Promise<number>`.
    pub fn count(&self) -> Promise {
        let inner = Arc::clone(&self.inner);
        future_to_promise(async move {
            let len = blazen_memory::store::MemoryStore::len(&*inner)
                .await
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
            Ok(JsValue::from_f64(len as f64))
        })
    }
}

// ---------------------------------------------------------------------------
// JsMemoryStore — adapter from a JS object to `MemoryStore`
// ---------------------------------------------------------------------------

/// Adapter that implements [`MemoryStore`](blazen_memory::store::MemoryStore)
/// by delegating each method to the corresponding JS function.
struct JsMemoryStore {
    impl_obj: JsValue,
}

// SAFETY: WASM is single-threaded.
unsafe impl Send for JsMemoryStore {}
unsafe impl Sync for JsMemoryStore {}

impl JsMemoryStore {
    fn new(impl_obj: JsValue) -> Self {
        Self { impl_obj }
    }

    /// Call a method on the wrapped JS object and await any returned Promise.
    async fn call(
        &self,
        method: &str,
        args: &[JsValue],
    ) -> std::result::Result<JsValue, blazen_memory::MemoryError> {
        let func =
            js_sys::Reflect::get(&self.impl_obj, &JsValue::from_str(method)).map_err(|e| {
                blazen_memory::MemoryError::Backend(format!(
                    "MemoryStore.{method} not found: {e:?}"
                ))
            })?;
        let func: &js_sys::Function = func.unchecked_ref();
        let result = match args.len() {
            0 => func.call0(&self.impl_obj),
            1 => func.call1(&self.impl_obj, &args[0]),
            2 => func.call2(&self.impl_obj, &args[0], &args[1]),
            3 => func.call3(&self.impl_obj, &args[0], &args[1], &args[2]),
            _ => {
                let js_args = js_sys::Array::new();
                for arg in args {
                    js_args.push(arg);
                }
                func.apply(&self.impl_obj, &js_args)
            }
        }
        .map_err(|e| {
            blazen_memory::MemoryError::Backend(format!("MemoryStore.{method}() threw: {e:?}"))
        })?;

        if result.has_type::<js_sys::Promise>() {
            let promise: js_sys::Promise = result.unchecked_into();
            wasm_bindgen_futures::JsFuture::from(promise)
                .await
                .map_err(|e| {
                    blazen_memory::MemoryError::Backend(format!(
                        "MemoryStore.{method}() rejected: {e:?}"
                    ))
                })
        } else {
            Ok(result)
        }
    }

    fn js_to_result(
        val: &JsValue,
    ) -> std::result::Result<blazen_memory::MemoryResult, blazen_memory::MemoryError> {
        let id = js_sys::Reflect::get(val, &"id".into())
            .ok()
            .and_then(|v| v.as_string())
            .unwrap_or_default();
        let text = js_sys::Reflect::get(val, &"text".into())
            .ok()
            .and_then(|v| v.as_string())
            .unwrap_or_default();
        let score = js_sys::Reflect::get(val, &"score".into())
            .ok()
            .and_then(|v| v.as_f64())
            .unwrap_or(0.0);
        let meta_val = js_sys::Reflect::get(val, &"metadata".into()).unwrap_or(JsValue::NULL);
        let metadata: serde_json::Value = if meta_val.is_null() || meta_val.is_undefined() {
            serde_json::Value::Null
        } else {
            serde_wasm_bindgen::from_value(meta_val).unwrap_or(serde_json::Value::Null)
        };
        Ok(blazen_memory::MemoryResult {
            id,
            text,
            score,
            metadata,
        })
    }

    fn js_to_stored(val: &JsValue) -> std::result::Result<StoredEntry, blazen_memory::MemoryError> {
        let id = js_sys::Reflect::get(val, &"id".into())
            .ok()
            .and_then(|v| v.as_string())
            .unwrap_or_default();
        let text = js_sys::Reflect::get(val, &"text".into())
            .ok()
            .and_then(|v| v.as_string())
            .unwrap_or_default();
        let meta_val = js_sys::Reflect::get(val, &"metadata".into()).unwrap_or(JsValue::NULL);
        let metadata: serde_json::Value = if meta_val.is_null() || meta_val.is_undefined() {
            serde_json::Value::Null
        } else {
            serde_wasm_bindgen::from_value(meta_val).unwrap_or(serde_json::Value::Null)
        };
        Ok(StoredEntry {
            id,
            text,
            elid: None,
            simhash_hex: None,
            text_simhash: 0,
            bands: Vec::new(),
            metadata,
        })
    }
}

#[async_trait::async_trait]
impl blazen_memory::store::MemoryStore for JsMemoryStore {
    async fn add(&self, entries: Vec<MemoryEntry>) -> blazen_memory::error::Result<Vec<String>> {
        let arr = js_sys::Array::new();
        for e in &entries {
            let obj = js_sys::Object::new();
            let _ = js_sys::Reflect::set(&obj, &"id".into(), &e.id.clone().into());
            let _ = js_sys::Reflect::set(&obj, &"text".into(), &e.text.clone().into());
            let meta = serde_wasm_bindgen::to_value(&e.metadata).unwrap_or(JsValue::NULL);
            let _ = js_sys::Reflect::set(&obj, &"metadata".into(), &meta);
            arr.push(&obj);
        }
        let result = SendFuture(self.call("add", &[arr.into()])).await?;
        let arr: &js_sys::Array = result.dyn_ref::<js_sys::Array>().ok_or_else(|| {
            blazen_memory::MemoryError::Backend("MemoryStore.add() must return an array".into())
        })?;
        let mut ids = Vec::with_capacity(arr.length() as usize);
        for i in 0..arr.length() {
            ids.push(arr.get(i).as_string().unwrap_or_default());
        }
        Ok(ids)
    }

    async fn search(
        &self,
        query: &str,
        limit: usize,
        metadata_filter: Option<&serde_json::Value>,
    ) -> blazen_memory::error::Result<Vec<blazen_memory::MemoryResult>> {
        let filter_js = metadata_filter
            .and_then(|f| serde_wasm_bindgen::to_value(f).ok())
            .unwrap_or(JsValue::NULL);
        let result = SendFuture(self.call(
            "search",
            &[
                JsValue::from_str(query),
                JsValue::from_f64(limit as f64),
                filter_js,
            ],
        ))
        .await?;
        let arr: &js_sys::Array = result.dyn_ref::<js_sys::Array>().ok_or_else(|| {
            blazen_memory::MemoryError::Backend("MemoryStore.search() must return an array".into())
        })?;
        let mut out = Vec::with_capacity(arr.length() as usize);
        for i in 0..arr.length() {
            out.push(Self::js_to_result(&arr.get(i))?);
        }
        Ok(out)
    }

    async fn search_local(
        &self,
        query: &str,
        limit: usize,
        metadata_filter: Option<&serde_json::Value>,
    ) -> blazen_memory::error::Result<Vec<blazen_memory::MemoryResult>> {
        let filter_js = metadata_filter
            .and_then(|f| serde_wasm_bindgen::to_value(f).ok())
            .unwrap_or(JsValue::NULL);
        let result = SendFuture(self.call(
            "searchLocal",
            &[
                JsValue::from_str(query),
                JsValue::from_f64(limit as f64),
                filter_js,
            ],
        ))
        .await?;
        let arr: &js_sys::Array = result.dyn_ref::<js_sys::Array>().ok_or_else(|| {
            blazen_memory::MemoryError::Backend(
                "MemoryStore.searchLocal() must return an array".into(),
            )
        })?;
        let mut out = Vec::with_capacity(arr.length() as usize);
        for i in 0..arr.length() {
            out.push(Self::js_to_result(&arr.get(i))?);
        }
        Ok(out)
    }

    async fn get(&self, id: &str) -> blazen_memory::error::Result<Option<StoredEntry>> {
        let result = SendFuture(self.call("get", &[JsValue::from_str(id)])).await?;
        if result.is_null() || result.is_undefined() {
            Ok(None)
        } else {
            Ok(Some(Self::js_to_stored(&result)?))
        }
    }

    async fn delete(&self, id: &str) -> blazen_memory::error::Result<bool> {
        let result = SendFuture(self.call("delete", &[JsValue::from_str(id)])).await?;
        Ok(result.is_truthy())
    }

    async fn len(&self) -> blazen_memory::error::Result<usize> {
        let result = SendFuture(self.call("len", &[])).await?;
        #[allow(clippy::cast_sign_loss, clippy::cast_possible_truncation)]
        Ok(result.as_f64().unwrap_or(0.0) as usize)
    }
}

// ---------------------------------------------------------------------------
// WasmInMemoryBackend — standalone wrapper around `blazen_memory::InMemoryBackend`
// ---------------------------------------------------------------------------

/// In-process [`MemoryBackend`](blazen_memory::store::MemoryBackend) backed by
/// a `HashMap` behind a `RwLock`.
///
/// Intended for tests and short-lived sessions: data lives only inside the
/// current WASM module instance and is discarded on reload. Pair with
/// [`Memory.fromBackend`](WasmMemory::from_backend) to plug it into a
/// [`Memory`](WasmMemory) without going through the JS-backed bridge.
///
/// ```js
/// import { Memory, InMemoryBackend, EmbeddingModel } from '@blazen/sdk';
///
/// const embedder = EmbeddingModel.openai();
/// const backend = new InMemoryBackend();
/// const memory = Memory.fromBackend(embedder, backend);
/// await memory.add("doc1", "hello world", null);
/// ```
#[wasm_bindgen(js_name = "InMemoryBackend")]
pub struct WasmInMemoryBackend {
    inner: Arc<InMemoryBackend>,
}

// SAFETY: WASM is single-threaded.
unsafe impl Send for WasmInMemoryBackend {}
unsafe impl Sync for WasmInMemoryBackend {}

impl Default for WasmInMemoryBackend {
    fn default() -> Self {
        Self::new()
    }
}

#[wasm_bindgen(js_class = "InMemoryBackend")]
#[allow(
    clippy::must_use_candidate,
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::missing_errors_doc
)]
impl WasmInMemoryBackend {
    /// Create a new, empty in-memory backend.
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self {
            inner: Arc::new(InMemoryBackend::new()),
        }
    }

    /// Insert or update a stored entry.
    ///
    /// @param entry - A [`WasmStoredEntry`]-shaped object.
    /// @returns `Promise<void>`.
    pub fn put(&self, entry: WasmStoredEntry) -> Promise {
        let inner = Arc::clone(&self.inner);
        future_to_promise(async move {
            let stored: StoredEntry = entry.into();
            blazen_memory::store::MemoryBackend::put(&*inner, stored)
                .await
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
            Ok(JsValue::UNDEFINED)
        })
    }

    /// Retrieve a stored entry by id.
    ///
    /// @param id - The entry identifier.
    /// @returns `Promise<WasmStoredEntry | null>`.
    pub fn get(&self, id: String) -> Promise {
        let inner = Arc::clone(&self.inner);
        future_to_promise(async move {
            let entry = blazen_memory::store::MemoryBackend::get(&*inner, &id)
                .await
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
            match entry {
                Some(e) => {
                    let wrapper: WasmStoredEntry = e.into();
                    serde_wasm_bindgen::to_value(&wrapper)
                        .map_err(|err| JsValue::from_str(&err.to_string()))
                }
                None => Ok(JsValue::NULL),
            }
        })
    }

    /// Delete a stored entry by id.
    ///
    /// @param id - The entry identifier.
    /// @returns `Promise<boolean>` — `true` if the entry existed and was removed.
    pub fn delete(&self, id: String) -> Promise {
        let inner = Arc::clone(&self.inner);
        future_to_promise(async move {
            let deleted = blazen_memory::store::MemoryBackend::delete(&*inner, &id)
                .await
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
            Ok(JsValue::from_bool(deleted))
        })
    }

    /// Return all stored entries.
    ///
    /// @returns `Promise<WasmStoredEntry[]>`.
    pub fn list(&self) -> Promise {
        let inner = Arc::clone(&self.inner);
        future_to_promise(async move {
            let entries = blazen_memory::store::MemoryBackend::list(&*inner)
                .await
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
            let arr = js_sys::Array::new_with_length(entries.len() as u32);
            for (i, e) in entries.into_iter().enumerate() {
                let wrapper: WasmStoredEntry = e.into();
                let val = serde_wasm_bindgen::to_value(&wrapper)
                    .map_err(|err| JsValue::from_str(&err.to_string()))?;
                arr.set(i as u32, val);
            }
            Ok(arr.into())
        })
    }

    /// Return the number of stored entries.
    ///
    /// @returns `Promise<number>`.
    #[wasm_bindgen(js_name = "len")]
    pub fn len(&self) -> Promise {
        let inner = Arc::clone(&self.inner);
        future_to_promise(async move {
            let n = blazen_memory::store::MemoryBackend::len(&*inner)
                .await
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
            Ok(JsValue::from_f64(n as f64))
        })
    }

    /// Return `true` if the backend contains no entries.
    ///
    /// @returns `Promise<boolean>`.
    #[wasm_bindgen(js_name = "isEmpty")]
    pub fn is_empty(&self) -> Promise {
        let inner = Arc::clone(&self.inner);
        future_to_promise(async move {
            let n = blazen_memory::store::MemoryBackend::len(&*inner)
                .await
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
            Ok(JsValue::from_bool(n == 0))
        })
    }

    /// Return candidate entries that share at least one LSH band with the
    /// query bands.
    ///
    /// @param bands - Query band strings, one per band slot.
    /// @param limit - Soft cap on the number of returned entries.
    /// @returns `Promise<WasmStoredEntry[]>`.
    #[wasm_bindgen(js_name = "searchByBands")]
    pub fn search_by_bands(&self, bands: Vec<String>, limit: u32) -> Promise {
        let inner = Arc::clone(&self.inner);
        future_to_promise(async move {
            let entries = blazen_memory::store::MemoryBackend::search_by_bands(
                &*inner,
                &bands,
                limit as usize,
            )
            .await
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
            let arr = js_sys::Array::new_with_length(entries.len() as u32);
            for (i, e) in entries.into_iter().enumerate() {
                let wrapper: WasmStoredEntry = e.into();
                let val = serde_wasm_bindgen::to_value(&wrapper)
                    .map_err(|err| JsValue::from_str(&err.to_string()))?;
                arr.set(i as u32, val);
            }
            Ok(arr.into())
        })
    }
}

// ---------------------------------------------------------------------------
// WasmMemoryResult — standalone wrapper around `blazen_memory::MemoryResult`
// ---------------------------------------------------------------------------

/// A single search result returned by [`Memory`](WasmMemory) queries.
///
/// Exposed primarily as a typed return value for downstream code that wants
/// to construct `MemoryResult`s from JS (e.g. when implementing a custom
/// [`MemoryStore`](WasmMemoryStore)).
///
/// Field semantics mirror [`blazen_memory::MemoryResult`]:
/// - `id`       — the entry identifier
/// - `text`     — the stored text content
/// - `score`    — similarity score in `[0, 1]`, higher means more similar
/// - `metadata` — arbitrary JSON-serializable metadata, or `null`
#[wasm_bindgen(js_name = "MemoryResult")]
pub struct WasmMemoryResult {
    inner: blazen_memory::MemoryResult,
}

// SAFETY: WASM is single-threaded.
unsafe impl Send for WasmMemoryResult {}
unsafe impl Sync for WasmMemoryResult {}

#[wasm_bindgen(js_class = "MemoryResult")]
#[allow(clippy::must_use_candidate, clippy::missing_errors_doc)]
impl WasmMemoryResult {
    /// Construct a new [`MemoryResult`](blazen_memory::MemoryResult).
    ///
    /// @param id       - The entry identifier.
    /// @param text     - The stored text content.
    /// @param score    - Similarity score in `[0, 1]`.
    /// @param metadata - Arbitrary JSON-serializable metadata (`null` allowed).
    #[wasm_bindgen(constructor)]
    pub fn new(
        id: String,
        text: String,
        score: f64,
        metadata: JsValue,
    ) -> Result<WasmMemoryResult, JsValue> {
        let metadata: serde_json::Value = if metadata.is_undefined() || metadata.is_null() {
            serde_json::Value::Null
        } else {
            serde_wasm_bindgen::from_value(metadata)
                .map_err(|e| JsValue::from_str(&e.to_string()))?
        };
        Ok(Self {
            inner: blazen_memory::MemoryResult {
                id,
                text,
                score,
                metadata,
            },
        })
    }

    /// The entry identifier.
    #[wasm_bindgen(getter)]
    pub fn id(&self) -> String {
        self.inner.id.clone()
    }

    /// The stored text content.
    #[wasm_bindgen(getter)]
    pub fn text(&self) -> String {
        self.inner.text.clone()
    }

    /// Similarity score in `[0, 1]`.
    #[wasm_bindgen(getter)]
    pub fn score(&self) -> f64 {
        self.inner.score
    }

    /// Arbitrary metadata, decoded from `serde_json::Value` to a JS value.
    #[wasm_bindgen(getter)]
    pub fn metadata(&self) -> JsValue {
        serde_wasm_bindgen::to_value(&self.inner.metadata).unwrap_or(JsValue::NULL)
    }
}

impl From<blazen_memory::MemoryResult> for WasmMemoryResult {
    fn from(inner: blazen_memory::MemoryResult) -> Self {
        Self { inner }
    }
}

impl From<WasmMemoryResult> for blazen_memory::MemoryResult {
    fn from(value: WasmMemoryResult) -> Self {
        value.inner
    }
}
