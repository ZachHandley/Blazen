//! Python wrapper for typed-tool authoring.
//!
//! Mirrors [`blazen_llm::TypedTool`] / [`blazen_llm::typed_tool_simple`] in
//! Python ergonomics. Accepts a Pydantic model class (or any object with a
//! ``model_json_schema()`` method) as the parameter schema and dispatches
//! decoded arguments into a handler callable.

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyType};
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pyfunction, gen_stub_pymethods};

use crate::agent::PyToolDef;

/// A typed Python tool whose parameter schema comes from a Pydantic model.
///
/// Holds an internal [`ToolDef`] you can hand to the agent loop via
/// :meth:`as_tool_def`. The class also exposes the same ``name`` /
/// ``description`` / ``parameters`` getters so it interoperates with
/// any code that introspects a tool surface.
///
/// Example:
///     >>> from pydantic import BaseModel
///     >>> class AddArgs(BaseModel):
///     ...     a: int
///     ...     b: int
///     >>> async def add(args: AddArgs) -> dict:
///     ...     return {"sum": args.a + args.b}
///     >>> tool = TypedTool(
///     ...     name="add",
///     ...     description="Add two numbers",
///     ...     args_model=AddArgs,
///     ...     handler=add,
///     ... )
///     >>> result = await run_agent(model, msgs, tools=[tool.as_tool_def()])
#[gen_stub_pyclass]
#[pyclass(name = "TypedTool")]
pub struct PyTypedTool {
    name: String,
    description: String,
    parameters: serde_json::Value,
    handler: Py<PyAny>,
}

#[gen_stub_pymethods]
#[pymethods]
impl PyTypedTool {
    /// Build a TypedTool.
    ///
    /// Args:
    ///     name: Tool name advertised to the model.
    ///     description: Human-readable description.
    ///     args_model: A class exposing ``model_json_schema()`` (any
    ///         Pydantic v2 model qualifies).
    ///     handler: Callable invoked with a parsed model instance.
    #[new]
    #[pyo3(signature = (*, name, description, args_model, handler))]
    fn new(
        py: Python<'_>,
        name: &str,
        description: &str,
        args_model: Bound<'_, PyType>,
        handler: Py<PyAny>,
    ) -> PyResult<Self> {
        let schema_obj = args_model.call_method0("model_json_schema")?;
        let parameters = crate::convert::py_to_json(py, &schema_obj)?;
        let wrapped = wrap_handler(py, args_model.unbind(), handler)?;
        Ok(Self {
            name: name.to_owned(),
            description: description.to_owned(),
            parameters,
            handler: wrapped,
        })
    }

    #[getter]
    fn name(&self) -> &str {
        &self.name
    }

    #[getter]
    fn description(&self) -> &str {
        &self.description
    }

    #[getter]
    #[gen_stub(override_return_type(type_repr = "dict[str, typing.Any]", imports = ("typing",)))]
    fn parameters(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        crate::convert::json_to_py(py, &self.parameters)
    }

    #[getter]
    fn handler(&self, py: Python<'_>) -> Py<PyAny> {
        self.handler.clone_ref(py)
    }

    /// Return a [`ToolDef`] suitable for [`run_agent`] / agent loops.
    fn as_tool_def(&self, py: Python<'_>) -> PyResult<Py<PyToolDef>> {
        Py::new(
            py,
            PyToolDef {
                name: self.name.clone(),
                description: self.description.clone(),
                parameters: self.parameters.clone(),
                handler: self.handler.clone_ref(py),
            },
        )
    }

    fn __repr__(&self) -> String {
        format!("TypedTool(name={:?})", self.name)
    }
}

// ---------------------------------------------------------------------------
// typed_tool_simple
// ---------------------------------------------------------------------------

/// Convenience constructor mirroring [`blazen_llm::typed_tool_simple`].
///
/// Returns a [`ToolDef`] directly so it can be passed straight to
/// [`run_agent`].
#[gen_stub_pyfunction]
#[pyfunction]
#[pyo3(signature = (*, name, description, args_model, handler))]
pub fn typed_tool_simple(
    py: Python<'_>,
    name: &str,
    description: &str,
    args_model: Bound<'_, PyType>,
    handler: Py<PyAny>,
) -> PyResult<Py<PyToolDef>> {
    let schema_obj = args_model.call_method0("model_json_schema")?;
    let parameters = crate::convert::py_to_json(py, &schema_obj)?;
    let wrapped = wrap_handler(py, args_model.unbind(), handler)?;
    Py::new(
        py,
        PyToolDef {
            name: name.to_owned(),
            description: description.to_owned(),
            parameters,
            handler: wrapped,
        },
    )
}

// ---------------------------------------------------------------------------
// Handler wrapping helper
// ---------------------------------------------------------------------------

/// Build a Python wrapper that constructs the Pydantic model from the
/// raw args dict, calls the user handler, and dumps a Pydantic result
/// via ``model_dump()`` so the agent loop sees plain JSON-able values.
fn wrap_handler(py: Python<'_>, args_model: Py<PyType>, handler: Py<PyAny>) -> PyResult<Py<PyAny>> {
    // Stash the model + handler as globals so the inner def-blocks can
    // resolve them at call time (they would not be captured as closure
    // cells from a `locals` dict). Wrappers reference them by name.
    let globals = PyDict::new(py);
    globals.set_item("_args_model", args_model.clone_ref(py))?;
    globals.set_item("_handler", handler.clone_ref(py))?;

    let code = r#"
import inspect

if inspect.iscoroutinefunction(_handler):
    async def _wrapper(raw_args):
        parsed = _args_model(**(raw_args or {}))
        result = await _handler(parsed)
        if hasattr(result, "model_dump"):
            return result.model_dump()
        return result
else:
    def _wrapper(raw_args):
        parsed = _args_model(**(raw_args or {}))
        result = _handler(parsed)
        if hasattr(result, "model_dump"):
            return result.model_dump()
        return result
"#;

    py.run(
        &std::ffi::CString::new(code).unwrap(),
        Some(&globals),
        Some(&globals),
    )?;
    let wrapper = globals
        .get_item("_wrapper")?
        .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("typed_tool: wrapper missing"))?;
    Ok(wrapper.unbind())
}
