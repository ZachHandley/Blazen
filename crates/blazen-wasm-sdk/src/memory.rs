//! `wasm-bindgen` wrapper for [`blazen_memory::Memory`].
//!
//! Exposes an in-memory vector store to JavaScript with `add`, `search`,
//! `searchLocal`, `get`, `delete`, and `count` methods. The store uses ELID
//! for vector indexing when an embedding model is provided, or falls back to
//! text-level `SimHash` in local-only mode.

use std::sync::Arc;

use js_sys::Promise;
use wasm_bindgen::prelude::*;
use wasm_bindgen_futures::future_to_promise;

use blazen_memory::store::MemoryStore;
use blazen_memory::{InMemoryBackend, Memory, MemoryEntry};

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
