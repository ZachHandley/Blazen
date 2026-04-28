//! TS-facing copy of [`blazen_core::SessionPausePolicy`].
//!
//! The native enum decides what to do with live session references when a
//! workflow is paused or snapshotted. Exposed to TypeScript as a string
//! union so callers can configure pause behaviour through the workflow
//! builder without crossing through opaque `JsValue` payloads.

use serde::{Deserialize, Serialize};
use tsify_next::Tsify;

use blazen_core::SessionPausePolicy;

/// TS-facing copy of [`blazen_core::SessionPausePolicy`].
///
/// Variants match the native `serde(rename_all = "snake_case")` form so the
/// JS side sees `"pickle_or_error"`, `"pickle_or_serialize"`, `"warn_drop"`,
/// or `"hard_error"`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize, Tsify)]
#[serde(rename_all = "snake_case")]
#[tsify(into_wasm_abi, from_wasm_abi)]
pub enum WasmSessionPausePolicy {
    /// Try to pickle each live ref into the snapshot. **Default — recommended.**
    #[default]
    PickleOrError,
    /// First try each ref's `blazen_serialize` implementation; otherwise
    /// fall through to [`WasmSessionPausePolicy::PickleOrError`] semantics.
    PickleOrSerialize,
    /// Drop live refs from the snapshot and emit a warning per drop.
    WarnDrop,
    /// Refuse to pause if any live refs are in flight.
    HardError,
}

impl From<SessionPausePolicy> for WasmSessionPausePolicy {
    fn from(value: SessionPausePolicy) -> Self {
        match value {
            SessionPausePolicy::PickleOrError => Self::PickleOrError,
            SessionPausePolicy::PickleOrSerialize => Self::PickleOrSerialize,
            SessionPausePolicy::WarnDrop => Self::WarnDrop,
            SessionPausePolicy::HardError => Self::HardError,
        }
    }
}

impl From<WasmSessionPausePolicy> for SessionPausePolicy {
    fn from(value: WasmSessionPausePolicy) -> Self {
        match value {
            WasmSessionPausePolicy::PickleOrError => Self::PickleOrError,
            WasmSessionPausePolicy::PickleOrSerialize => Self::PickleOrSerialize,
            WasmSessionPausePolicy::WarnDrop => Self::WarnDrop,
            WasmSessionPausePolicy::HardError => Self::HardError,
        }
    }
}
