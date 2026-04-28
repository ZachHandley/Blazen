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

use blazen_llm::agent::{
    AgentConfig, AgentEvent as RustAgentEvent, AgentResult as RustAgentResult,
    run_agent as rust_run_agent, run_agent_with_callback as rust_run_agent_with_callback,
};
use blazen_llm::error::BlazenError;
use blazen_llm::traits::Tool;
use blazen_llm::types::{ChatMessage, ToolDefinition};

use crate::providers::PyCompletionModel;
use crate::types::{PyChatMessage, PyCompletionResponse, PyToolCall, PyToolOutput};

/// Convert a Python tool-handler return value into a Rust [`ToolOutput`].
///
/// If the Python object is an instance of [`PyToolOutput`] (i.e. the user
/// explicitly returned `ToolOutput(data=..., llm_override=...)`), unpack
/// it without re-serialising the data. Otherwise convert the value via
/// [`crate::convert::py_to_json`] and wrap it as a default `ToolOutput`
/// with no override (each provider applies its own conversion from `data`).
fn py_result_to_tool_output(
    py: Python<'_>,
    result: &Bound<'_, PyAny>,
) -> PyResult<blazen_llm::types::ToolOutput<serde_json::Value>> {
    if let Ok(typed) = result.extract::<PyToolOutput>() {
        return Ok(typed.into_inner());
    }
    let value = crate::convert::py_to_json(py, result)?;
    Ok(blazen_llm::types::ToolOutput::new(value))
}

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
    ) -> Result<blazen_llm::types::ToolOutput<serde_json::Value>, BlazenError> {
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

            let output = tokio::task::block_in_place(|| {
                Python::attach(|py| py_result_to_tool_output(py, py_result.bind(py)))
            })
            .map_err(|e: PyErr| BlazenError::tool_error(e.to_string()))?;

            Ok(output)
        } else {
            // Sync path: call directly and convert the result
            let output = tokio::task::block_in_place(|| {
                Python::attach(|py| {
                    let args_py = crate::convert::json_to_py(py, &args_value)?;
                    let result = callable.call1(py, (args_py,))?;
                    py_result_to_tool_output(py, result.bind(py))
                })
            })
            .map_err(|e: PyErr| BlazenError::tool_error(e.to_string()))?;

            Ok(output)
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
// PyAgentConfig
// ---------------------------------------------------------------------------

/// Typed configuration for the agentic tool execution loop.
///
/// Mirrors [`blazen_llm::AgentConfig`] minus the `tools` slot (tools are
/// passed explicitly to the run function). Use this when calling
/// [`run_agent`] with a fully-typed config rather than positional kwargs.
///
/// Example:
///     >>> config = AgentConfig(max_iterations=20, system_prompt="be terse",
///     ...                      temperature=0.0, tool_concurrency=4)
#[gen_stub_pyclass]
#[pyclass(name = "AgentConfig", from_py_object)]
#[derive(Clone, Default)]
pub struct PyAgentConfig {
    pub(crate) max_iterations: u32,
    pub(crate) add_finish_tool: bool,
    pub(crate) system_prompt: Option<String>,
    pub(crate) temperature: Option<f32>,
    pub(crate) max_tokens: Option<u32>,
    pub(crate) tool_concurrency: usize,
}

#[gen_stub_pymethods]
#[pymethods]
impl PyAgentConfig {
    /// Construct an AgentConfig.
    #[new]
    #[pyo3(signature = (
        *,
        max_iterations=10,
        add_finish_tool=false,
        system_prompt=None,
        temperature=None,
        max_tokens=None,
        tool_concurrency=0,
    ))]
    fn new(
        max_iterations: u32,
        add_finish_tool: bool,
        system_prompt: Option<String>,
        temperature: Option<f32>,
        max_tokens: Option<u32>,
        tool_concurrency: usize,
    ) -> Self {
        Self {
            max_iterations,
            add_finish_tool,
            system_prompt,
            temperature,
            max_tokens,
            tool_concurrency,
        }
    }

    /// Maximum number of tool-call rounds before forcing a stop.
    #[getter]
    fn max_iterations(&self) -> u32 {
        self.max_iterations
    }

    /// Whether the implicit "finish" tool is added.
    #[getter]
    fn add_finish_tool(&self) -> bool {
        self.add_finish_tool
    }

    /// Optional system prompt prepended to messages.
    #[getter]
    fn system_prompt(&self) -> Option<&str> {
        self.system_prompt.as_deref()
    }

    /// Sampling temperature.
    #[getter]
    fn temperature(&self) -> Option<f32> {
        self.temperature
    }

    /// Maximum tokens per completion call.
    #[getter]
    fn max_tokens(&self) -> Option<u32> {
        self.max_tokens
    }

    /// Maximum concurrent tool executions per round (0 = unlimited).
    #[getter]
    fn tool_concurrency(&self) -> usize {
        self.tool_concurrency
    }

    fn __repr__(&self) -> String {
        format!(
            "AgentConfig(max_iterations={}, tool_concurrency={}, add_finish_tool={})",
            self.max_iterations, self.tool_concurrency, self.add_finish_tool
        )
    }
}

impl PyAgentConfig {
    /// Build the underlying Rust [`AgentConfig`] from a typed config and an
    /// already-resolved tool list.
    pub(crate) fn to_rust(&self, tools: Vec<Arc<dyn Tool>>) -> AgentConfig {
        let mut config = AgentConfig::new(tools).with_max_iterations(self.max_iterations);
        if let Some(prompt) = self.system_prompt.as_ref() {
            config = config.with_system_prompt(prompt.clone());
        }
        if let Some(t) = self.temperature {
            config = config.with_temperature(t);
        }
        if let Some(mt) = self.max_tokens {
            config = config.with_max_tokens(mt);
        }
        if self.add_finish_tool {
            config = config.with_finish_tool();
        }
        if self.tool_concurrency > 0 {
            config = config.with_tool_concurrency(self.tool_concurrency);
        }
        config
    }
}

// ---------------------------------------------------------------------------
// PyAgentEvent
// ---------------------------------------------------------------------------

/// Events emitted during agent execution.
///
/// Mirrors [`blazen_llm::AgentEvent`]. Yielded to the user-supplied callback
/// passed to [`run_agent_with_callback`] -- one of three variants:
/// - ``"tool_called"`` (carries iteration + tool_call),
/// - ``"tool_result"`` (carries iteration + tool_name + result),
/// - ``"iteration_complete"`` (carries iteration + had_tool_calls).
///
/// Inspect via the ``kind`` getter and the per-variant fields.
#[gen_stub_pyclass]
#[pyclass(name = "AgentEvent", frozen, from_py_object)]
#[derive(Clone)]
pub struct PyAgentEvent {
    pub(crate) inner: RustAgentEvent,
}

#[gen_stub_pymethods]
#[pymethods]
impl PyAgentEvent {
    /// Variant tag: ``"tool_called"``, ``"tool_result"``, or
    /// ``"iteration_complete"``.
    #[getter]
    fn kind(&self) -> &'static str {
        match &self.inner {
            RustAgentEvent::ToolCalled { .. } => "tool_called",
            RustAgentEvent::ToolResult { .. } => "tool_result",
            RustAgentEvent::IterationComplete { .. } => "iteration_complete",
        }
    }

    /// The 0-based iteration index at which this event fired.
    #[getter]
    fn iteration(&self) -> u32 {
        match &self.inner {
            RustAgentEvent::ToolCalled { iteration, .. }
            | RustAgentEvent::ToolResult { iteration, .. }
            | RustAgentEvent::IterationComplete { iteration, .. } => *iteration,
        }
    }

    /// The tool call carried by ``ToolCalled`` events. ``None`` otherwise.
    #[getter]
    fn tool_call(&self) -> Option<PyToolCall> {
        if let RustAgentEvent::ToolCalled { tool_call, .. } = &self.inner {
            Some(PyToolCall::from(tool_call))
        } else {
            None
        }
    }

    /// The tool name carried by ``ToolResult`` events. ``None`` otherwise.
    #[getter]
    fn tool_name(&self) -> Option<&str> {
        if let RustAgentEvent::ToolResult { tool_name, .. } = &self.inner {
            Some(tool_name)
        } else {
            None
        }
    }

    /// The tool result value for ``ToolResult`` events. ``None`` otherwise.
    #[getter]
    #[gen_stub(override_return_type(type_repr = "typing.Optional[typing.Any]", imports = ("typing",)))]
    fn result(&self, py: Python<'_>) -> PyResult<Option<Py<PyAny>>> {
        if let RustAgentEvent::ToolResult { result, .. } = &self.inner {
            Ok(Some(crate::convert::json_to_py(py, result)?))
        } else {
            Ok(None)
        }
    }

    /// Whether the model emitted any tool calls in this iteration. Set on
    /// ``IterationComplete`` events; ``None`` otherwise.
    #[getter]
    fn had_tool_calls(&self) -> Option<bool> {
        if let RustAgentEvent::IterationComplete { had_tool_calls, .. } = &self.inner {
            Some(*had_tool_calls)
        } else {
            None
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "AgentEvent(kind={:?}, iteration={})",
            self.kind(),
            self.iteration()
        )
    }
}

impl From<RustAgentEvent> for PyAgentEvent {
    fn from(inner: RustAgentEvent) -> Self {
        Self { inner }
    }
}

// ---------------------------------------------------------------------------
// run_agent_with_callback function
// ---------------------------------------------------------------------------

/// Run the agent loop while invoking a Python callback for every
/// [`AgentEvent`].
///
/// The callback is dispatched synchronously from the agent's Tokio task; if
/// it raises, the error is logged and the loop continues. For long-running
/// observers, schedule work onto an executor inside the callback.
///
/// Args:
///     model: The completion model to use.
///     messages: Initial conversation messages.
///     tools: List of ToolDef objects.
///     config: Typed [`AgentConfig`].
///     callback: Python callable receiving one [`AgentEvent`] per call.
#[gen_stub_pyfunction]
#[gen_stub(override_return_type(type_repr = "typing.Coroutine[typing.Any, typing.Any, AgentResult]", imports = ("typing",)))]
#[pyfunction]
#[pyo3(signature = (model, messages, *, tools, config, callback))]
pub fn run_agent_with_callback<'py>(
    py: Python<'py>,
    model: Bound<'py, PyCompletionModel>,
    messages: Vec<PyRef<'py, PyChatMessage>>,
    tools: Vec<PyRef<'py, PyToolDef>>,
    config: PyRef<'py, PyAgentConfig>,
    callback: Py<PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    let rust_messages: Vec<ChatMessage> = messages.iter().map(|m| m.inner.clone()).collect();

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

    let rust_config = config.to_rust(rust_tools);
    let inner_model = crate::providers::completion_model::arc_from_bound(&model);
    let cb = Arc::new(callback);

    pyo3_async_runtimes::tokio::future_into_py_with_locals(py, locals, async move {
        let cb_for_closure = Arc::clone(&cb);
        let on_event = move |event: RustAgentEvent| {
            let cb_ref = &cb_for_closure;
            let _ = Python::attach(|py| -> PyResult<()> {
                let py_event = Py::new(py, PyAgentEvent { inner: event })?;
                cb_ref.call1(py, (py_event,))?;
                Ok(())
            });
        };

        let result = rust_run_agent_with_callback(
            inner_model.as_ref(),
            rust_messages,
            rust_config,
            on_event,
        )
        .await
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        Ok(PyAgentResult { inner: result })
    })
}

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
