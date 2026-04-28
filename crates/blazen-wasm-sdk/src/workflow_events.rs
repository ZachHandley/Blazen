//! TS-facing copies of the built-in workflow events `StartEvent` and
//! `StopEvent` from [`blazen_events`].
//!
//! The native [`blazen_events::StartEvent`] / [`blazen_events::StopEvent`]
//! cross the WASM ABI as `Box<dyn AnyEvent>`, which can't be exposed to JS
//! directly. These [`tsify_next::Tsify`]-derived wrappers flatten each event
//! to its data shape so callers can produce or inspect the canonical
//! built-in events from TypeScript with proper typing instead of an opaque
//! `JsValue`.
//!
//! `WasmInputRequestEvent` and `WasmInputResponseEvent` already live in
//! [`crate::events`]; this module rounds out the built-in event surface
//! with the workflow start/stop pair.

use blazen_events::{StartEvent, StopEvent};
use serde::{Deserialize, Serialize};
use tsify_next::Tsify;

// ---------------------------------------------------------------------------
// WasmStartEvent
// ---------------------------------------------------------------------------

/// TS-facing copy of [`blazen_events::StartEvent`].
///
/// Emitted to kick off a workflow with arbitrary JSON data.
#[derive(Debug, Clone, Serialize, Deserialize, Tsify)]
#[tsify(into_wasm_abi, from_wasm_abi)]
pub struct WasmStartEvent {
    /// Arbitrary payload passed into the workflow at start.
    pub data: serde_json::Value,
}

impl From<StartEvent> for WasmStartEvent {
    fn from(value: StartEvent) -> Self {
        Self { data: value.data }
    }
}

impl From<WasmStartEvent> for StartEvent {
    fn from(value: WasmStartEvent) -> Self {
        Self { data: value.data }
    }
}

// ---------------------------------------------------------------------------
// WasmStopEvent
// ---------------------------------------------------------------------------

/// TS-facing copy of [`blazen_events::StopEvent`].
///
/// Emitted to signal that a workflow has completed with a result.
#[derive(Debug, Clone, Serialize, Deserialize, Tsify)]
#[tsify(into_wasm_abi, from_wasm_abi)]
pub struct WasmStopEvent {
    /// The final result of the workflow.
    pub result: serde_json::Value,
}

impl From<StopEvent> for WasmStopEvent {
    fn from(value: StopEvent) -> Self {
        Self {
            result: value.result,
        }
    }
}

impl From<WasmStopEvent> for StopEvent {
    fn from(value: WasmStopEvent) -> Self {
        Self {
            result: value.result,
        }
    }
}
