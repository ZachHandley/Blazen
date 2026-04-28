//! TS-facing data shape for [`blazen_core::WorkflowResult`].
//!
//! The native [`WorkflowResult`](blazen_core::WorkflowResult) owns both the
//! terminal event (`Box<dyn AnyEvent>`) and an `Arc<SessionRefRegistry>`
//! that backs any session-ref markers carried by the payload. Neither field
//! crosses the WASM ABI directly, so this mirror flattens the result to the
//! information JS callers actually need:
//!
//! - the terminal event's type identifier,
//! - a JSON-marshalled copy of the event's payload, and
//! - the run id of the workflow that produced the result (recoverable from
//!   the carried session-ref registry).
//!
//! The full live result handle is still produced by
//! [`crate::handler::WasmWorkflowHandler::result`]; this type exists so the
//! same data can be inspected by any consumer that already speaks
//! `serde-wasm-bindgen` JSON shapes.

use serde::{Deserialize, Serialize};
use tsify_next::Tsify;

// ---------------------------------------------------------------------------
// WasmWorkflowResult
// ---------------------------------------------------------------------------

/// TS-facing copy of [`blazen_core::WorkflowResult`].
///
/// JS callers receive this shape from
/// [`crate::handler::WasmWorkflowHandler`] when they ask for the terminal
/// event in plain-object form (rather than the typed handle).
#[derive(Debug, Clone, Serialize, Deserialize, Tsify)]
#[tsify(into_wasm_abi, from_wasm_abi)]
#[serde(rename_all = "camelCase")]
pub struct WasmWorkflowResult {
    /// Event type identifier of the terminal event (matches
    /// [`blazen_events::Event::event_type`]). For a normal completion this
    /// is `"StopEvent"`; custom workflows may publish their own terminal
    /// event types.
    pub event_type: String,
    /// JSON-marshalled payload of the terminal event.
    pub data: serde_json::Value,
    /// Run id (UUID string) of the workflow that produced this result.
    pub run_id: String,
}
