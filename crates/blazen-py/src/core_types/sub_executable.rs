//! Python binding for the [`blazen_core::SubExecutable`] trait.
//!
//! `SubExecutable` is the uniform trait for anything embeddable as a child
//! runner inside a parent `Workflow` -- both `Workflow` and `Pipeline`
//! implement it natively. This module exposes it as a subclassable ABC so a
//! Python user can write a custom child runner (`async def execute(self,
//! input, ctx)`) and embed it in a parent workflow via a
//! [`SubPipelineStep`](crate::pipeline::subpipeline::PySubPipelineStep).
//!
//! ## Bridging Python async -> Rust async
//!
//! The [`PySubExecutableAdapter`] wraps the Python instance and implements the
//! Rust trait by:
//!
//! 1. Acquiring the GIL, converting the JSON input + a [`PyContext`] wrapper
//!    into Python objects, and calling `instance.execute(input, ctx)` to get a
//!    coroutine.
//! 2. Capturing the active asyncio task locals and converting the coroutine
//!    into a Rust future.
//! 3. Driving the future to completion outside the GIL.
//! 4. Re-acquiring the GIL and converting the returned Python value back into
//!    `serde_json::Value`.

use async_trait::async_trait;
use pyo3::prelude::*;
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pymethods};

use blazen_core::{Context, SubExecutable, WorkflowError};

use crate::convert::{json_to_py, py_to_json};
use crate::workflow::context::PyContext;

// ---------------------------------------------------------------------------
// PySubExecutable -- ABC subclassable from Python
// ---------------------------------------------------------------------------

/// Abstract base class mirroring the [`SubExecutable`] trait.
///
/// Subclass this from Python and override
/// ``async def execute(self, input, ctx)`` to define a custom child runner
/// that can be embedded inside a parent ``Workflow`` via a
/// :class:`SubPipelineStep`. The method receives the opaque JSON input (as a
/// native Python value) plus the parent :class:`Context`, and must return the
/// terminal result (any JSON-serializable value).
///
/// Concrete runners (``Workflow`` and ``Pipeline``) implement
/// [`SubExecutable`] on the Rust side and do not go through this shim.
///
/// Example:
///     >>> class DoubleRunner(SubExecutable):
///     ...     async def execute(self, input, ctx):
///     ...         return {"value": input["value"] * 2}
#[gen_stub_pyclass]
#[pyclass(name = "SubExecutable", subclass)]
pub struct PySubExecutable {
    /// Marker -- subclasses store their own state on the Python side.
    _private: (),
}

#[gen_stub_pymethods]
#[pymethods]
impl PySubExecutable {
    #[new]
    fn new() -> Self {
        Self { _private: () }
    }

    /// Run this child executable with the given input and parent context.
    ///
    /// The default implementation raises ``NotImplementedError``. Override
    /// this in a subclass with an ``async def``.
    ///
    /// Args:
    ///     input: The opaque input payload (a native Python value).
    ///     ctx: The parent :class:`Context`.
    #[pyo3(signature = (input, ctx))]
    fn execute(&self, input: &Bound<'_, PyAny>, ctx: &PyContext) -> PyResult<Py<PyAny>> {
        let _ = (input, ctx);
        Err(pyo3::exceptions::PyNotImplementedError::new_err(
            "SubExecutable.execute must be overridden by an `async def` in a subclass",
        ))
    }
}

// ---------------------------------------------------------------------------
// PySubExecutableAdapter
// ---------------------------------------------------------------------------

/// Adapter that lets a Python-side [`PySubExecutable`] subclass satisfy the
/// Rust [`SubExecutable`] trait. Used when a Python caller passes a
/// `SubExecutable` subclass into a `SubPipelineStep`.
#[derive(Debug)]
pub struct PySubExecutableAdapter {
    instance: Py<PyAny>,
}

impl PySubExecutableAdapter {
    /// Wrap a Python object that implements `execute`.
    #[must_use]
    pub fn new(instance: Py<PyAny>) -> Self {
        Self { instance }
    }
}

#[async_trait]
impl SubExecutable for PySubExecutableAdapter {
    async fn execute(
        &self,
        input: serde_json::Value,
        ctx: Context,
    ) -> Result<serde_json::Value, WorkflowError> {
        // Phase 1: under the GIL, build the args, call `execute` to get a
        // coroutine, capture asyncio task locals, and convert it into a Rust
        // future.
        let setup = tokio::task::block_in_place(|| {
            Python::attach(|py| -> PyResult<_> {
                let bound = self.instance.bind(py);
                let py_input = json_to_py(py, &input)?;
                let py_ctx = Py::new(py, PyContext { inner: ctx.clone() })?;
                let coro = bound.call_method1("execute", (py_input, py_ctx))?;
                let locals = pyo3_async_runtimes::tokio::get_current_locals(py)?;
                let fut =
                    pyo3_async_runtimes::into_future_with_locals(&locals, coro).map_err(|e| {
                        pyo3::exceptions::PyTypeError::new_err(format!(
                            "SubExecutable.execute did not return an awaitable coroutine \
                             (override it with `async def`): {e}"
                        ))
                    })?;
                Ok((fut, locals))
            })
        });

        let (fut, locals) = setup.map_err(|e| {
            WorkflowError::ValidationFailed(format!("SubExecutable.execute setup failed: {e}"))
        })?;

        // Phase 2: drive the Python coroutine to completion outside the GIL.
        let py_result = pyo3_async_runtimes::tokio::scope(locals, fut)
            .await
            .map_err(|e: PyErr| {
                WorkflowError::ValidationFailed(format!("SubExecutable.execute raised: {e}"))
            })?;

        // Phase 3: re-acquire the GIL and convert the result back to JSON.
        tokio::task::block_in_place(|| {
            Python::attach(|py| -> Result<serde_json::Value, WorkflowError> {
                let bound = py_result.bind(py);
                if bound.is_none() {
                    return Ok(serde_json::Value::Null);
                }
                py_to_json(py, bound).map_err(|e| {
                    WorkflowError::ValidationFailed(format!(
                        "SubExecutable.execute returned a non-JSON-serializable value: {e}"
                    ))
                })
            })
        })
    }
}
