//! [`BlazenState`] base class for typed, per-field workflow state.
//!
//! Subclass this to create state objects with mixed serializable and
//! non-serializable fields. Each field is stored individually in the
//! context with its optimal storage tier.

use pyo3::prelude::*;
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pymethods};

/// Base class for typed workflow state objects.
///
/// Subclass with `@dataclass` to get typed, per-field context storage.
/// Fields listed in `Meta.transient` are excluded from serialization
/// and recreated via the `restore()` hook.
///
/// ```python
/// @dataclass
/// class MyState(BlazenState):
///     input_path: str = ""
///     conn: sqlite3.Connection | None = None
///
///     class Meta:
///         transient = {"conn"}
///         store_by = {}
///
///     def restore(self):
///         if self.input_path:
///             self.conn = sqlite3.connect(self.input_path)
/// ```
#[gen_stub_pyclass]
#[pyclass(name = "BlazenState", subclass, dict)]
pub struct PyBlazenState {}

#[gen_stub_pymethods]
#[pymethods]
impl PyBlazenState {
    #[new]
    #[pyo3(signature = (**_kwargs))]
    fn new(_kwargs: Option<&Bound<'_, pyo3::types::PyDict>>) -> Self {
        Self {}
    }
}

/// Convenience implementation of the `FieldStore` structural protocol that
/// delegates to plain Python callables. Use as a value in
/// `BlazenState.Meta.store_by` to route a specific field through custom
/// storage (e.g. S3, a database, Redis) without having to subclass
/// anything.
#[gen_stub_pyclass]
#[pyclass(name = "CallbackFieldStore")]
pub struct PyCallbackFieldStore {
    save_fn: Py<PyAny>,
    load_fn: Py<PyAny>,
}

#[gen_stub_pymethods]
#[pymethods]
impl PyCallbackFieldStore {
    #[new]
    fn new(save_fn: Py<PyAny>, load_fn: Py<PyAny>) -> Self {
        Self { save_fn, load_fn }
    }

    /// Persist `value` under `key` by invoking the user-supplied `save_fn`.
    ///
    /// The `_ctx` argument is intentionally unused — the convenience wrapper
    /// presents the simpler `(key, value)` signature to user callbacks. If
    /// you need access to the context, implement the structural protocol
    /// directly with your own class.
    fn save(&self, py: Python<'_>, key: String, value: Py<PyAny>, _ctx: Py<PyAny>) -> PyResult<()> {
        self.save_fn.call1(py, (key, value))?;
        Ok(())
    }

    /// Load and return the value stored under `key` by invoking the
    /// user-supplied `load_fn`.
    ///
    /// The `_ctx` argument is intentionally unused — see `save` for details.
    fn load(&self, py: Python<'_>, key: String, _ctx: Py<PyAny>) -> PyResult<Py<PyAny>> {
        self.load_fn.call1(py, (key,))
    }

    fn __repr__(&self) -> String {
        "CallbackFieldStore(save_fn=..., load_fn=...)".to_string()
    }
}
