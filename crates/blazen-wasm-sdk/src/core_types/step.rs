//! TS-facing data shapes for [`blazen_core::StepOutput`] and
//! [`blazen_core::StepRegistration`].
//!
//! These mirrors are deliberately decoupled from the live engine types:
//!
//! - [`blazen_core::StepOutput`] holds `Box<dyn AnyEvent>` payloads which
//!   cannot cross the WASM ABI directly. The JS-side equivalent flattens
//!   each variant to its event-type name plus a JSON payload, which is the
//!   shape every JS step handler already returns from
//!   [`crate::workflow::WasmWorkflowBuilder`].
//! - [`blazen_core::StepRegistration`] carries a non-cloneable
//!   `StepFn = Arc<dyn Fn(...) -> Pin<Box<...>>>` and an internal
//!   [`Semaphore`](tokio::sync::Semaphore). The JS-facing copy keeps just
//!   the metadata fields (`name`, `accepts`, `emits`, `max_concurrency`) so
//!   callers can introspect a registered step's wiring.
//!
//! Both types use [`tsify_next::Tsify`] so they round-trip through plain JS
//! objects with full TypeScript types.

use blazen_core::StepRegistration;
use serde::{Deserialize, Serialize};
use tsify_next::Tsify;

// ---------------------------------------------------------------------------
// WasmStepOutputKind
// ---------------------------------------------------------------------------

/// Discriminant for the variant carried by a [`WasmStepOutput`].
///
/// Mirrors the variants of [`blazen_core::StepOutput`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Tsify)]
#[tsify(into_wasm_abi, from_wasm_abi)]
#[serde(rename_all = "lowercase")]
pub enum WasmStepOutputKind {
    /// A single event will be routed downstream.
    Single,
    /// Multiple events will be routed (fan-out).
    Multiple,
    /// No event -- the step performed a side-effect only.
    None,
}

// ---------------------------------------------------------------------------
// WasmStepOutputEvent
// ---------------------------------------------------------------------------

/// A single event entry in a [`WasmStepOutput`].
///
/// Carries the event-type identifier plus an arbitrary JSON payload, matching
/// the shape JS step handlers already return from
/// [`crate::workflow::WasmWorkflowBuilder::add_step`].
#[derive(Debug, Clone, Serialize, Deserialize, Tsify)]
#[tsify(into_wasm_abi, from_wasm_abi)]
#[serde(rename_all = "camelCase")]
pub struct WasmStepOutputEvent {
    /// Event type identifier (matches
    /// [`blazen_events::Event::event_type`]).
    pub event_type: String,
    /// Arbitrary JSON payload carried by the event.
    pub data: serde_json::Value,
}

// ---------------------------------------------------------------------------
// WasmStepOutput
// ---------------------------------------------------------------------------

/// TS-facing copy of [`blazen_core::StepOutput`].
///
/// The shape is `{ kind, events }` where `kind` is the variant discriminant
/// and `events` is the list of payload events:
///
/// - `kind = "single"` -- exactly one entry in `events`.
/// - `kind = "multiple"` -- zero or more entries (fan-out).
/// - `kind = "none"` -- empty `events`.
///
/// JS step handlers can construct this directly to drive the workflow engine
/// without going through the implicit `{ eventType, data }` convention.
#[derive(Debug, Clone, Serialize, Deserialize, Tsify)]
#[tsify(into_wasm_abi, from_wasm_abi)]
#[serde(rename_all = "camelCase")]
pub struct WasmStepOutput {
    /// Variant discriminant.
    pub kind: WasmStepOutputKind,
    /// Events carried by this output. Always empty for
    /// [`WasmStepOutputKind::None`], length 1 for
    /// [`WasmStepOutputKind::Single`], and any non-negative length for
    /// [`WasmStepOutputKind::Multiple`].
    pub events: Vec<WasmStepOutputEvent>,
}

// ---------------------------------------------------------------------------
// WasmStepRegistration
// ---------------------------------------------------------------------------

/// TS-facing metadata mirror of [`blazen_core::StepRegistration`].
///
/// Drops the `handler` (a non-cloneable `Arc<dyn Fn(...)>`) and the internal
/// [`Semaphore`](tokio::sync::Semaphore); keeps the introspectable shape so
/// callers can list registered steps with their accepted/emitted event
/// types and concurrency limits.
#[derive(Debug, Clone, Serialize, Deserialize, Tsify)]
#[tsify(into_wasm_abi, from_wasm_abi)]
#[serde(rename_all = "camelCase")]
pub struct WasmStepRegistration {
    /// Human-readable name used in logging and error messages.
    pub name: String,
    /// Event type identifiers this step accepts.
    pub accepts: Vec<String>,
    /// Event type identifiers this step may emit (informational).
    pub emits: Vec<String>,
    /// Maximum number of concurrent invocations of this step
    /// (`0` = unlimited).
    pub max_concurrency: u32,
}

impl From<&StepRegistration> for WasmStepRegistration {
    fn from(value: &StepRegistration) -> Self {
        Self {
            name: value.name.clone(),
            accepts: value.accepts.iter().map(|s| (*s).to_owned()).collect(),
            emits: value.emits.iter().map(|s| (*s).to_owned()).collect(),
            #[allow(clippy::cast_possible_truncation)]
            max_concurrency: value.max_concurrency as u32,
        }
    }
}
