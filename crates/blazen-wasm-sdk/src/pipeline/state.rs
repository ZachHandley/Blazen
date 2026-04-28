//! TS-facing snapshot of [`blazen_pipeline::PipelineState`].
//!
//! v1 of the WASM pipeline binding does not expose `input_mapper` /
//! `condition` callbacks (see [`mod.rs`](super) for the rationale), so JS
//! code never receives a *live* `PipelineState` reference. This struct
//! exists so the generated `.d.ts` carries the same shape used by the
//! Python and Node bindings, and so future versions that add async-callback
//! support have a pre-existing TS interface to plug into.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};
use tsify_next::Tsify;

/// Read-only view of pipeline state, mirroring
/// [`blazen_pipeline::PipelineState`] over the WASM ABI.
///
/// Captures the original pipeline input, the shared key/value store, and
/// the per-stage result map at a single point in time.
#[derive(Debug, Clone, Default, Serialize, Deserialize, Tsify)]
#[tsify(into_wasm_abi, from_wasm_abi)]
pub struct WasmPipelineState {
    /// The original pipeline input.
    pub input: serde_json::Value,
    /// Key/value entries written by stages via the shared store.
    pub shared: HashMap<String, serde_json::Value>,
    /// Stage outputs in completion order.
    pub stage_results: Vec<WasmStateStageEntry>,
}

/// A single `(stage_name, output)` pair surfaced through
/// [`WasmPipelineState::stage_results`].
///
/// Mirrors how the upstream `PipelineState` keeps stage results in an
/// `IndexMap` so insertion order is preserved when handed to JS.
#[derive(Debug, Clone, Serialize, Deserialize, Tsify)]
#[tsify(into_wasm_abi, from_wasm_abi)]
pub struct WasmStateStageEntry {
    /// Name of the completed stage.
    pub name: String,
    /// JSON output produced by the stage's workflow.
    pub output: serde_json::Value,
}
