//! Typed Node bindings for the global step deserializer registry.
//!
//! [`JsStepDeserializerRegistry`] is a thin wrapper around
//! [`blazen_core::StepDeserializerRegistry`] for read-only introspection.
//! Registration of *new* step builders cannot be done from JS — a
//! `StepBuilderFn` is a Rust function pointer that produces a
//! `StepRegistration`, which carries an `Arc<dyn Fn ... -> Future>`. JS
//! handlers are TSFNs and are registered through the workflow builder
//! instead of the step deserializer registry.
//!
//! The free functions [`register_step_builder_id`],
//! [`lookup_step_builder_ids`], and [`registered_step_builder_ids`]
//! mirror the free functions in [`blazen_core::step_registry`] but
//! restrict their JS surface to ID introspection (no JS callable can be
//! coerced into a `StepBuilderFn`).

use blazen_core::step_registry as core_step_registry;
use napi_derive::napi;

// ---------------------------------------------------------------------------
// StepDeserializerRegistry
// ---------------------------------------------------------------------------

/// Read-only handle to the process-global step deserializer registry.
///
/// Construct via [`JsStepDeserializerRegistry::global`]. The registry is
/// populated by Rust-side `#[step]`-annotated functions (or by manual
/// calls to `blazen_core::register_step_builder` from Rust crates that
/// link into this binary). JS code can list the IDs that are currently
/// registered but cannot insert new ones.
#[napi(js_name = "StepDeserializerRegistry")]
pub struct JsStepDeserializerRegistry {}

#[napi]
#[allow(
    clippy::must_use_candidate,
    clippy::missing_errors_doc,
    clippy::unused_self,
    clippy::new_without_default
)]
impl JsStepDeserializerRegistry {
    /// Obtain a handle to the process-global registry.
    #[napi(factory)]
    pub fn global() -> Self {
        Self {}
    }

    /// Backwards-compatible no-arg constructor that returns the same
    /// global handle as [`Self::global`]. Kept so JS callers can do
    /// `new StepDeserializerRegistry()` without thinking about the
    /// factory naming.
    #[napi(constructor)]
    pub fn new() -> Self {
        Self {}
    }

    /// Returns `true` when a builder is registered under `stepId`.
    #[allow(clippy::needless_pass_by_value)]
    #[napi]
    pub fn has(&self, step_id: String) -> bool {
        core_step_registry::lookup_step_builder(&step_id).is_some()
    }

    /// All registered step IDs. The order is unspecified.
    #[napi(js_name = "stepIds")]
    pub fn step_ids(&self) -> Vec<String> {
        core_step_registry::registered_step_ids()
            .into_iter()
            .map(str::to_owned)
            .collect()
    }

    /// Number of registered step builders.
    #[napi]
    pub fn len(&self) -> u32 {
        u32::try_from(core_step_registry::registered_step_ids().len()).unwrap_or(u32::MAX)
    }

    /// `true` when the registry is empty.
    #[napi(js_name = "isEmpty")]
    pub fn is_empty(&self) -> bool {
        core_step_registry::registered_step_ids().is_empty()
    }
}

// ---------------------------------------------------------------------------
// Free functions
// ---------------------------------------------------------------------------

/// JS-facing parity hook for [`blazen_core::register_step_builder`].
///
/// Returns `false` unconditionally because a `StepBuilderFn` is a Rust
/// `fn() -> StepRegistration` (a function pointer, not a closure) and
/// JS callables cannot be coerced into one. The hook is exposed so
/// build tools can detect the registry surface; actual registration
/// happens through Rust-side `#[step]`-annotated functions.
#[napi(js_name = "registerStepBuilder")]
#[must_use]
pub fn register_step_builder_id(_step_id: String) -> bool {
    false
}

/// `true` when a step builder is registered under `stepId`.
#[allow(clippy::needless_pass_by_value)]
#[napi(js_name = "lookupStepBuilder")]
#[must_use]
pub fn lookup_step_builder_ids(step_id: String) -> bool {
    core_step_registry::lookup_step_builder(&step_id).is_some()
}

/// All step IDs registered in the process-global registry. Order is
/// unspecified.
#[napi(js_name = "registeredStepIds")]
#[must_use]
pub fn registered_step_builder_ids() -> Vec<String> {
    core_step_registry::registered_step_ids()
        .into_iter()
        .map(str::to_owned)
        .collect()
}
