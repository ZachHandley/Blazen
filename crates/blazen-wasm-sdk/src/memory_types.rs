//! TS-facing copy of [`blazen_memory::MemoryEntry`].
//!
//! [`crate::memory`] already declares a `MemoryEntry` interface via a
//! `#[wasm_bindgen(typescript_custom_section)]` block, but that only
//! produces a TypeScript declaration — it does not give Rust callers a
//! [`tsify_next::Tsify`]-derived bridge they can serialize through
//! `into_wasm_abi` / `from_wasm_abi`. This wrapper closes that gap by
//! mirroring the native [`blazen_memory::MemoryEntry`] shape exactly so
//! Rust code on the WASM side can return / accept the type with full
//! `serde-wasm-bindgen` support.

use serde::{Deserialize, Serialize};
use tsify_next::Tsify;

use blazen_memory::MemoryEntry;

/// TS-facing copy of [`blazen_memory::MemoryEntry`].
///
/// Matches the field set of the native struct (`id`, `text`, `metadata`)
/// so the JSON shape is identical to what
/// [`crate::memory::WasmMemory::get`] already returns through
/// `serde-wasm-bindgen`.
#[derive(Debug, Clone, Serialize, Deserialize, Tsify)]
#[tsify(into_wasm_abi, from_wasm_abi)]
pub struct WasmMemoryEntry {
    /// Unique identifier. If empty, one will be generated on add.
    pub id: String,
    /// The text content to store.
    pub text: String,
    /// Arbitrary user metadata.
    pub metadata: serde_json::Value,
}

impl From<MemoryEntry> for WasmMemoryEntry {
    fn from(value: MemoryEntry) -> Self {
        Self {
            id: value.id,
            text: value.text,
            metadata: value.metadata,
        }
    }
}

impl From<WasmMemoryEntry> for MemoryEntry {
    fn from(value: WasmMemoryEntry) -> Self {
        Self {
            id: value.id,
            text: value.text,
            metadata: value.metadata,
        }
    }
}
