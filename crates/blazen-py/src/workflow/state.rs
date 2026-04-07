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
