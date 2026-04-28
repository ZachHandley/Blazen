//! Typed Node bindings for [`blazen_core::step::StepOutput`] and
//! [`blazen_core::step::StepRegistration`].
//!
//! These types live in the workflow runtime and are normally consumed
//! purely on the Rust side (the JS step handler builds them implicitly
//! by returning an event, an array of events, or `null`). The bindings
//! here exist so callers writing custom orchestrators can introspect
//! the registered step graph -- e.g. read step names, accepted/emitted
//! event types, and concurrency limits -- without falling out into
//! Rust-only territory.
//!
//! [`JsStepOutput`] mirrors the three-variant Rust enum via factory
//! constructors (`StepOutput.single(event)`,
//! `StepOutput.multiple([...])`, `StepOutput.none()`) and a `kind`
//! discriminator string. JS step handlers authored via the standard
//! [`crate::workflow::JsWorkflow`] API never need to construct one
//! explicitly -- they just `return event` -- but custom transports
//! that drive the Rust event loop manually do.

use napi_derive::napi;

use blazen_core::step::StepRegistration;

// ---------------------------------------------------------------------------
// JsStepOutputKind
// ---------------------------------------------------------------------------

/// Variant tag for [`JsStepOutput`].
#[napi(string_enum, js_name = "StepOutputKind")]
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum JsStepOutputKind {
    /// A single outbound event, routed normally.
    Single,
    /// A fan-out of events. Every entry is dispatched.
    Multiple,
    /// Side-effect only -- the workflow loop produces no downstream
    /// events from this step.
    None,
}

// ---------------------------------------------------------------------------
// JsStepOutput
// ---------------------------------------------------------------------------

/// One of three terminal results from a step body.
///
/// Mirrors [`blazen_core::step::StepOutput`]:
///
/// * `Single` -- one outbound event, routed normally.
/// * `Multiple` -- a fan-out of events. Every entry is dispatched.
/// * `None` -- side-effect only, the workflow loop produces no
///   downstream events from this step.
///
/// Construct via the static factories
/// (`StepOutput.single(event)`, `StepOutput.multiple([...])`,
/// `StepOutput.none()`); the `kind` and `events` getters let callers
/// introspect after the fact.
///
/// Events are exchanged as plain JSON objects of the same shape that
/// [`crate::workflow::JsWorkflow`] handlers return -- `{ type: string,
/// ... }`.
#[napi(js_name = "StepOutput")]
pub struct JsStepOutput {
    kind: JsStepOutputKind,
    events: Vec<serde_json::Value>,
}

#[napi]
#[allow(clippy::must_use_candidate, clippy::missing_errors_doc)]
impl JsStepOutput {
    /// Construct a single-event output.
    #[napi(factory)]
    pub fn single(event: serde_json::Value) -> Self {
        Self {
            kind: JsStepOutputKind::Single,
            events: vec![event],
        }
    }

    /// Construct a fan-out output from an array of events.
    #[napi(factory)]
    pub fn multiple(events: Vec<serde_json::Value>) -> Self {
        Self {
            kind: JsStepOutputKind::Multiple,
            events,
        }
    }

    /// Construct a no-output result (side-effect only).
    #[napi(factory)]
    pub fn none() -> Self {
        Self {
            kind: JsStepOutputKind::None,
            events: Vec::new(),
        }
    }

    /// Active variant tag.
    #[napi(getter)]
    pub fn kind(&self) -> JsStepOutputKind {
        self.kind
    }

    /// Whether this output has no events (the `None` variant).
    #[napi(js_name = "isNone")]
    pub fn is_none(&self) -> bool {
        self.kind == JsStepOutputKind::None
    }

    /// Whether this output carries exactly one event.
    #[napi(js_name = "isSingle")]
    pub fn is_single(&self) -> bool {
        self.kind == JsStepOutputKind::Single
    }

    /// Whether this output carries multiple events.
    #[napi(js_name = "isMultiple")]
    pub fn is_multiple(&self) -> bool {
        self.kind == JsStepOutputKind::Multiple
    }

    /// All events carried by this output.
    ///
    /// Empty for `None`; length 1 for `Single`; length N for
    /// `Multiple`.
    #[napi]
    pub fn events(&self) -> Vec<serde_json::Value> {
        self.events.clone()
    }
}

// ---------------------------------------------------------------------------
// JsStepRegistration
// ---------------------------------------------------------------------------

/// Read-only metadata for a step registered on a `Workflow`.
///
/// Wraps [`blazen_core::step::StepRegistration`]. Returned by workflow
/// introspection helpers so callers can list the steps that will run,
/// their accepted/emitted event types, and the configured
/// max-concurrency.
///
/// The handler closure itself is not exposed -- it is a Rust closure
/// type that cannot cross the FFI boundary -- but every other field
/// is available as a getter.
#[napi(js_name = "StepRegistration")]
pub struct JsStepRegistration {
    pub(crate) inner: StepRegistration,
}

#[napi]
#[allow(clippy::must_use_candidate, clippy::missing_errors_doc)]
impl JsStepRegistration {
    /// Human-readable name for this step.
    #[napi(getter)]
    pub fn name(&self) -> String {
        self.inner.name.clone()
    }

    /// Event type identifiers this step accepts (matches
    /// `event.type`).
    #[napi(getter)]
    pub fn accepts(&self) -> Vec<String> {
        self.inner.accepts.iter().map(|s| (*s).to_owned()).collect()
    }

    /// Event type identifiers this step may emit (informational only;
    /// the runtime does not enforce this).
    #[napi(getter)]
    pub fn emits(&self) -> Vec<String> {
        self.inner.emits.iter().map(|s| (*s).to_owned()).collect()
    }

    /// Maximum number of concurrent invocations of this step. `0`
    /// means unlimited.
    #[napi(getter, js_name = "maxConcurrency")]
    pub fn max_concurrency(&self) -> u32 {
        u32::try_from(self.inner.max_concurrency).unwrap_or(u32::MAX)
    }
}

impl From<StepRegistration> for JsStepRegistration {
    fn from(inner: StepRegistration) -> Self {
        Self { inner }
    }
}
