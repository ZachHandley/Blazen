//! `wasm-bindgen` wrapper for [`blazen_memory::Memory`].
//!
//! Exposes an in-memory vector store to JavaScript with `add`, `search`,
//! `searchLocal`, `get`, `delete`, and `count` methods. The store uses ELID
//! for vector indexing when an embedding model is provided, or falls back to
//! text-level `SimHash` in local-only mode.

use std::pin::Pin;
use std::sync::Arc;

use js_sys::Promise;
use wasm_bindgen::prelude::*;
use wasm_bindgen_futures::future_to_promise;

use blazen_memory::store::MemoryStore;
use blazen_memory::{InMemoryBackend, Memory, MemoryEntry, StoredEntry};

use crate::embedding::WasmEmbeddingModel;

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
    ///   async list() { /* ... */ },
    ///   async len() { return localStorage.length; },
    ///   async searchByBands(bands, limit) { /* ... */ },
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
        let func = js_sys::Reflect::get(&self.backend, &JsValue::from_str(method))
            .map_err(|e| blazen_memory::MemoryError::Backend(format!(
                "backend.{method} not found: {e:?}"
            )))?;

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
        .map_err(|e| blazen_memory::MemoryError::Backend(format!(
            "backend.{method}() threw: {e:?}"
        )))?;

        // Await if the result is a Promise.
        if result.has_type::<js_sys::Promise>() {
            let promise: js_sys::Promise = result.unchecked_into();
            wasm_bindgen_futures::JsFuture::from(promise)
                .await
                .map_err(|e| blazen_memory::MemoryError::Backend(format!(
                    "backend.{method}() rejected: {e:?}"
                )))
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
            &entry.elid.as_deref().map_or(JsValue::NULL, |s| JsValue::from_str(s)),
        );
        let _ = js_sys::Reflect::set(
            &obj,
            &"simhashHex".into(),
            &entry.simhash_hex.as_deref().map_or(JsValue::NULL, |s| JsValue::from_str(s)),
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
        let bands_val = js_sys::Reflect::get(val, &"bands".into())
            .unwrap_or(JsValue::UNDEFINED);
        let bands: Vec<String> = if bands_val.is_instance_of::<js_sys::Array>() {
            let arr: &js_sys::Array = bands_val.unchecked_ref();
            arr.iter().filter_map(|v| v.as_string()).collect()
        } else {
            Vec::new()
        };
        let metadata_val = js_sys::Reflect::get(val, &"metadata".into())
            .unwrap_or(JsValue::NULL);
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
