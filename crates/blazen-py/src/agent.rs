//! Python wrappers for the agentic tool execution loop.
//!
//! Exposes [`run_agent`](blazen_llm::agent::run_agent) to Python with
//! [`PyToolDef`] for defining tools from Python callables.
//!
//! Tool handlers may be either synchronous functions or async coroutine
//! functions. Async handlers are detected at registration time and awaited
//! using the `pyo3-async-runtimes` bridge.

use std::sync::Arc;

use pyo3::prelude::*;
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pyfunction, gen_stub_pymethods};

use blazen_llm::agent::{AgentConfig, AgentResult as RustAgentResult, run_agent as rust_run_agent};
use blazen_llm::error::BlazenError;
use blazen_llm::traits::Tool;
use blazen_llm::types::{ChatMessage, ToolDefinition};

use crate::providers::PyCompletionModel;
use crate::types::{PyChatMessage, PyCompletionResponse};

// ---------------------------------------------------------------------------
// PyToolWrapper -- bridges a Python callable into the Rust Tool trait
// ---------------------------------------------------------------------------

/// A Python-implemented tool that satisfies the Rust [`Tool`] trait.
///
/// Supports both synchronous and async Python callables. When `is_async` is
/// true, the result of calling the handler is a coroutine that gets converted
/// to a Rust future via `pyo3_async_runtimes` and awaited outside the GIL.
struct PyToolWrapper {
    name: String,
    description: String,
    parameters: serde_json::Value,
    callable: Py<PyAny>,
    is_async: bool,
    locals: pyo3_async_runtimes::TaskLocals,
}

#[async_trait::async_trait]
impl Tool for PyToolWrapper {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: self.name.clone(),
            description: self.description.clone(),
            parameters: self.parameters.clone(),
        }
    }

    async fn execute(
        &self,
        arguments: serde_json::Value,
    ) -> Result<serde_json::Value, BlazenError> {
        let args_value = arguments;
        let callable = Python::attach(|py| self.callable.clone_ref(py));

        if self.is_async {
            // Async path: call to get a coroutine, convert to Rust future, await
            let locals = self.locals.clone();

            let coroutine: Py<PyAny> = tokio::task::block_in_place(|| {
                Python::attach(|py| {
                    let args_py = crate::convert::json_to_py(py, &args_value)?;
                    callable.call1(py, (args_py,))
                })
            })
            .map_err(|e: PyErr| BlazenError::tool_error(e.to_string()))?;

            let future = Python::attach(|py| {
                pyo3_async_runtimes::into_future_with_locals(&locals, coroutine.into_bound(py))
            })
            .map_err(|e: PyErr| BlazenError::tool_error(e.to_string()))?;

            let py_result = pyo3_async_runtimes::tokio::scope(locals, future)
                .await
                .map_err(|e: PyErr| BlazenError::tool_error(e.to_string()))?;

            let result = tokio::task::block_in_place(|| {
                Python::attach(|py| crate::convert::py_to_json(py, py_result.bind(py)))
            })
            .map_err(|e: PyErr| BlazenError::tool_error(e.to_string()))?;

            Ok(result)
        } else {
            // Sync path: call directly and convert the result
            let result = tokio::task::block_in_place(|| {
                Python::attach(|py| {
                    let args_py = crate::convert::json_to_py(py, &args_value)?;
                    let result = callable.call1(py, (args_py,))?;
                    crate::convert::py_to_json(py, result.bind(py))
                })
            })
            .map_err(|e: PyErr| BlazenError::tool_error(e.to_string()))?;

            Ok(result)
        }
    }
}

// ---------------------------------------------------------------------------
// PyToolDef -- user-facing tool definition
// ---------------------------------------------------------------------------

/// A tool definition for the agent.
///
/// Example:
///     >>> tool = ToolDef(
///     ...     name="search",
///     ...     description="Search the web",
///     ...     parameters={"type": "object", "properties": {"query": {"type": "string"}}},
///     ...     handler=lambda args: {"results": []}
///     ... )
#[gen_stub_pyclass]
#[pyclass(name = "ToolDef")]
pub struct PyToolDef {
    pub(crate) name: String,
    pub(crate) description: String,
    pub(crate) parameters: serde_json::Value,
    pub(crate) handler: Py<PyAny>,
}

#[gen_stub_pymethods]
#[pymethods]
impl PyToolDef {
    #[new]
    #[pyo3(signature = (*, name, description, parameters, handler))]
    fn new(
        py: Python<'_>,
        name: &str,
        description: &str,
        parameters: &Bound<'_, PyAny>,
        handler: Py<PyAny>,
    ) -> PyResult<Self> {
        let params = crate::convert::py_to_json(py, parameters)?;
        Ok(Self {
            name: name.to_owned(),
            description: description.to_owned(),
            parameters: params,
            handler,
        })
    }

    /// The tool name (as advertised to the model).
    #[getter]
    fn name(&self) -> &str {
        &self.name
    }

    /// Human-readable description of what the tool does.
    #[getter]
    fn description(&self) -> &str {
        &self.description
    }

    /// JSON-schema parameters dict describing the tool's arguments.
    #[getter]
    fn parameters(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        crate::convert::json_to_py(py, &self.parameters)
    }

    /// The Python callable that implements the tool.
    #[getter]
    fn handler(&self, py: Python<'_>) -> Py<PyAny> {
        self.handler.clone_ref(py)
    }

    fn __repr__(&self) -> String {
        format!("ToolDef(name='{}')", self.name)
    }
}

// ---------------------------------------------------------------------------
// PyAgentResult
// ---------------------------------------------------------------------------

/// Result of an agent run.
///
/// Example:
///     >>> result = await run_agent(model, messages, tools=[tool])
///     >>> print(result.response.content)
///     >>> print(result.iterations)
///     >>> print(result.total_cost)
#[gen_stub_pyclass]
#[pyclass(name = "AgentResult")]
pub struct PyAgentResult {
    inner: RustAgentResult,
}

#[gen_stub_pymethods]
#[pymethods]
impl PyAgentResult {
    /// The final completion response from the model.
    #[getter]
    fn response(&self) -> PyCompletionResponse {
        PyCompletionResponse {
            inner: self.inner.response.clone(),
        }
    }

    /// Full message history including all tool calls and results.
    #[getter]
    fn messages(&self) -> Vec<PyChatMessage> {
        self.inner
            .messages
            .iter()
            .map(|m| PyChatMessage { inner: m.clone() })
            .collect()
    }

    /// Number of tool call rounds that occurred.
    #[getter]
    fn iterations(&self) -> u32 {
        self.inner.iterations
    }

    /// Aggregated cost across all rounds, if available.
    #[getter]
    fn total_cost(&self) -> Option<f64> {
        self.inner.total_cost
    }

    fn __repr__(&self) -> String {
        format!(
            "AgentResult(iterations={}, cost={:?})",
            self.inner.iterations, self.inner.total_cost
        )
    }
}

// ---------------------------------------------------------------------------
// run_agent function
// ---------------------------------------------------------------------------

/// Run an agentic tool execution loop.
///
/// Sends messages to the model with tool definitions, executes tool calls,
/// feeds results back, and repeats until the model stops calling tools
/// or max_iterations is reached.
///
/// Args:
///     model: The completion model to use.
///     messages: Initial conversation messages.
///     tools: List of ToolDef objects.
///     max_iterations: Maximum tool call rounds (default: 10).
///     system_prompt: Optional system prompt.
///     temperature: Optional sampling temperature.
///     max_tokens: Optional max tokens per call.
///     add_finish_tool: Whether to add a built-in "finish" tool (default: False).
///     tool_concurrency: Maximum concurrent tool executions per round (default: 0 = unlimited).
///
/// Returns:
///     AgentResult with the final response and full conversation history.
#[gen_stub_pyfunction]
#[gen_stub(override_return_type(type_repr = "typing.Coroutine[typing.Any, typing.Any, AgentResult]", imports = ("typing",)))]
#[pyfunction]
#[pyo3(signature = (model, messages, *, tools, max_iterations=10, system_prompt=None, temperature=None, max_tokens=None, add_finish_tool=false, tool_concurrency=0))]
#[allow(clippy::too_many_arguments)]
pub fn run_agent<'py>(
    py: Python<'py>,
    model: Bound<'py, PyCompletionModel>,
    messages: Vec<PyRef<'py, PyChatMessage>>,
    tools: Vec<PyRef<'py, PyToolDef>>,
    max_iterations: u32,
    system_prompt: Option<String>,
    temperature: Option<f32>,
    max_tokens: Option<u32>,
    add_finish_tool: bool,
    tool_concurrency: usize,
) -> PyResult<Bound<'py, PyAny>> {
    let rust_messages: Vec<ChatMessage> = messages.iter().map(|m| m.inner.clone()).collect();

    // Capture task locals for async tool handler support
    let locals = pyo3_async_runtimes::tokio::get_current_locals(py)?;
    let inspect = py.import("inspect")?;

    let rust_tools: Vec<Arc<dyn Tool>> = tools
        .iter()
        .map(|t| {
            let is_async: bool = inspect
                .call_method1("iscoroutinefunction", (&t.handler,))
                .and_then(|r| r.extract())
                .unwrap_or(false);
            Arc::new(PyToolWrapper {
                name: t.name.clone(),
                description: t.description.clone(),
                parameters: t.parameters.clone(),
                callable: t.handler.clone_ref(py),
                is_async,
                locals: locals.clone(),
            }) as Arc<dyn Tool>
        })
        .collect();

    let mut config = AgentConfig::new(rust_tools).with_max_iterations(max_iterations);
    if let Some(prompt) = system_prompt {
        config = config.with_system_prompt(prompt);
    }
    if let Some(temp) = temperature {
        config = config.with_temperature(temp);
    }
    if let Some(max) = max_tokens {
        config = config.with_max_tokens(max);
    }
    if add_finish_tool {
        config = config.with_finish_tool();
    }
    if tool_concurrency > 0 {
        config = config.with_tool_concurrency(tool_concurrency);
    }

    let inner_model = crate::providers::completion_model::arc_from_bound(&model);

    pyo3_async_runtimes::tokio::future_into_py_with_locals(py, locals, async move {
        let result = rust_run_agent(inner_model.as_ref(), rust_messages, config)
            .await
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        Ok(PyAgentResult { inner: result })
    })
}
