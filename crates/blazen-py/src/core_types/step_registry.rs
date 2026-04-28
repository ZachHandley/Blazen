//! Python wrappers for the global step-deserializer registry described
//! in [`blazen_core::step_registry`].
//!
//! Both the `StepDeserializerRegistry` class and the free-function entry
//! points (`register_step_builder`, `lookup_step_builder`,
//! `registered_step_ids`) are exposed.
//!
//! Registration from Python is intentionally limited: the core registry
//! stores bare `fn() -> StepRegistration` pointers, which cannot be
//! constructed from a Python callable without a stable global trampoline.
//! Until that machinery exists, the binding only exposes registry
//! *introspection* (listing IDs, looking up presence) plus a hook for
//! the `#[step]` proc-macro path that registers Rust-side builders at
//! import time. This matches the comment in the core module that user
//! code "must call this explicitly" -- from Python that means via Rust
//! code linked into the extension module, not from `.py` files.

use pyo3::prelude::*;
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pyfunction, gen_stub_pymethods};

use blazen_core::step_registry::{
    StepBuilderFn, lookup_step_builder as core_lookup, register_step_builder as core_register,
    registered_step_ids as core_registered,
};

// ---------------------------------------------------------------------------
// PyStepDeserializerRegistry
// ---------------------------------------------------------------------------

/// Python-visible handle to the *process-global*
/// [`blazen_core::step_registry::StepDeserializerRegistry`].
///
/// The core crate exposes the registry as a singleton accessible only
/// through the `register_step_builder` / `lookup_step_builder` /
/// `registered_step_ids` free functions, so this class is a thin
/// introspection facade -- every method delegates to those functions.
/// Construction is allowed for ergonomic reasons but every instance is
/// equivalent and shares the same global state.
#[gen_stub_pyclass]
#[pyclass(name = "StepDeserializerRegistry", frozen)]
#[derive(Default)]
pub struct PyStepDeserializerRegistry {}

#[gen_stub_pymethods]
#[pymethods]
impl PyStepDeserializerRegistry {
    /// Construct a handle. All instances reference the same
    /// process-global registry.
    #[new]
    fn new_py() -> Self {
        Self {}
    }

    /// Number of registered step builders in the process-global
    /// registry.
    fn len(&self) -> usize {
        core_registered().len()
    }

    /// Whether the registry has any registered builders.
    fn is_empty(&self) -> bool {
        core_registered().is_empty()
    }

    /// Return every registered step ID. Order is unspecified.
    fn step_ids(&self) -> Vec<String> {
        core_registered().into_iter().map(str::to_owned).collect()
    }

    /// Return `True` if a builder is registered under `step_id`.
    fn contains(&self, step_id: &str) -> bool {
        core_lookup(step_id).is_some()
    }

    fn __repr__(&self) -> String {
        format!("StepDeserializerRegistry(len={})", core_registered().len())
    }
}

// ---------------------------------------------------------------------------
// Free-function bindings (process-global registry)
// ---------------------------------------------------------------------------

/// Internal helper that the `#[step]` proc-macro can call from Rust at
/// extension-load time to register a builder for the process-global
/// registry. Kept on this module so the wiring is co-located with the
/// other registry exports.
pub fn register_builder_internal(step_id: &'static str, builder: StepBuilderFn) {
    core_register(step_id, builder);
}

/// Register a step builder in the process-global step registry.
///
/// **Note:** This binding currently rejects Python callables -- the core
/// registry stores bare `fn` pointers, which cannot be synthesised from a
/// Python function without a per-step trampoline. The `#[step]`
/// proc-macro registers Rust-side builders directly; from Python you can
/// inspect the registry via `lookup_step_builder` and
/// `registered_step_ids`, but registration itself must happen on the
/// Rust side (or via a future trampoline plumbed through this binding).
#[gen_stub_pyfunction]
#[pyfunction]
#[pyo3(name = "register_step_builder")]
#[allow(clippy::needless_pass_by_value)]
pub fn register_step_builder(_step_id: String, _builder: Bound<'_, PyAny>) -> PyResult<()> {
    Err(pyo3::exceptions::PyNotImplementedError::new_err(
        "register_step_builder from Python is not yet supported -- the core registry \
         requires a `fn() -> StepRegistration` pointer, which cannot be constructed \
         from a Python callable. Register builders on the Rust side via the \
         `#[step]` proc-macro or `blazen_core::register_step_builder`.",
    ))
}

/// Return `True` if `step_id` is registered in the process-global
/// registry.
///
/// The core API returns a fresh `StepRegistration` (an `Arc<Fn ...>`),
/// which is not exposed to Python. This binding therefore reports
/// presence only -- callers that need to *use* the registration must do
/// so on the Rust side.
#[gen_stub_pyfunction]
#[pyfunction]
#[pyo3(name = "lookup_step_builder")]
pub fn lookup_step_builder(step_id: &str) -> bool {
    core_lookup(step_id).is_some()
}

/// Return every step ID currently registered in the process-global
/// registry.
#[gen_stub_pyfunction]
#[pyfunction]
#[pyo3(name = "registered_step_ids")]
pub fn registered_step_ids() -> Vec<String> {
    core_registered().into_iter().map(str::to_owned).collect()
}
