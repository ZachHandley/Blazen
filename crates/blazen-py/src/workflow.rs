//! Python wrapper for [`Workflow`](blazen_core::Workflow) and
//! [`WorkflowBuilder`](blazen_core::WorkflowBuilder).

use std::time::Duration;

use pyo3::prelude::*;
use pyo3::types::PyDict;

use blazen_core::snapshot::WorkflowSnapshot;

use crate::error::{BlazenPyError, to_py_result};
use crate::event::dict_to_json;
use crate::handler::PyWorkflowHandler;
use crate::step::PyStepWrapper;

/// A validated, ready-to-run workflow.
///
/// Construct a `Workflow` by providing a name and a list of step wrappers
/// (created with the `@step` decorator).
///
/// Example:
///     >>> @step
///     ... async def echo(ctx, ev):
///     ...     return StopEvent(result=ev.to_dict())
///     >>>
///     >>> wf = Workflow("echo-wf", [echo])
///     >>> handler = await wf.run({"message": "hello"})
///     >>> result = await handler.result()
#[pyclass(name = "Workflow")]
pub struct PyWorkflow {
    name: String,
    steps: Vec<Py<PyStepWrapper>>,
    timeout: Option<f64>,
}

#[pymethods]
impl PyWorkflow {
    /// Create a new workflow.
    ///
    /// Args:
    ///     name: A human-readable name for the workflow.
    ///     steps: A list of `_StepWrapper` objects created by `@step`.
    ///     timeout: Optional timeout in seconds (default: 300).
    #[new]
    #[pyo3(signature = (name, steps, timeout=None))]
    fn new(name: &str, steps: Vec<PyRef<'_, PyStepWrapper>>, timeout: Option<f64>) -> Self {
        let step_refs: Vec<Py<PyStepWrapper>> = steps.into_iter().map(Into::into).collect();

        Self {
            name: name.to_string(),
            steps: step_refs,
            timeout,
        }
    }

    /// Execute the workflow with a JSON payload.
    ///
    /// The payload is wrapped in a `StartEvent` and dispatched to the
    /// workflow engine.
    ///
    /// Args:
    ///     input: A dict of data to pass as the start event payload.
    ///
    /// Returns:
    ///     A `WorkflowHandler` for awaiting results or streaming events.
    ///
    /// Example:
    ///     >>> handler = await wf.run({"prompt": "Hello!"})
    ///     >>> result = await handler.result()
    #[pyo3(signature = (**kwargs))]
    fn run<'py>(
        &self,
        py: Python<'py>,
        kwargs: Option<&Bound<'py, PyDict>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let data = if let Some(kw) = kwargs {
            dict_to_json(kw)?
        } else {
            serde_json::Value::Object(serde_json::Map::new())
        };

        let locals = pyo3_async_runtimes::tokio::get_current_locals(py)?;

        // Build the workflow here with task locals
        let mut builder = blazen_core::WorkflowBuilder::new(&self.name);
        for step in &self.steps {
            let step_ref = step.borrow(py);
            let registration = step_ref.to_registration_with_locals(locals.clone())?;
            builder = builder.step(registration);
        }
        if let Some(t) = self.timeout {
            builder = builder.timeout(Duration::from_secs_f64(t));
        }
        let workflow = builder.build().map_err(BlazenPyError::from)?;

        pyo3_async_runtimes::tokio::future_into_py_with_locals(py, locals, async move {
            let handler = workflow.run(data).await.map_err(BlazenPyError::from)?;
            to_py_result(Ok(PyWorkflowHandler::new(handler)))
        })
    }

    /// Resume a workflow from a previously captured snapshot.
    ///
    /// This is a static method -- call it as `Workflow.resume(...)`.
    ///
    /// The steps must be the same set (or a compatible superset) of steps
    /// that were registered when the workflow was originally created.
    ///
    /// Args:
    ///     snapshot_json: A JSON string produced by `WorkflowHandler.pause()`.
    ///     steps: A list of `_StepWrapper` objects created by `@step`.
    ///     timeout: Optional timeout in seconds (default: 300).
    ///
    /// Returns:
    ///     A new `WorkflowHandler` for the resumed workflow.
    ///
    /// Example:
    ///     >>> handler = await Workflow.resume(snapshot_json, [step1, step2])
    ///     >>> result = await handler.result()
    #[staticmethod]
    #[pyo3(signature = (snapshot_json, steps, timeout=None))]
    fn resume<'py>(
        py: Python<'py>,
        snapshot_json: &str,
        steps: Vec<PyRef<'_, PyStepWrapper>>,
        timeout: Option<f64>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let snapshot = WorkflowSnapshot::from_json(snapshot_json).map_err(BlazenPyError::from)?;

        let locals = pyo3_async_runtimes::tokio::get_current_locals(py)?;

        let mut registrations = Vec::with_capacity(steps.len());
        for step in &steps {
            registrations.push(step.to_registration_with_locals(locals.clone())?);
        }

        let timeout_dur = timeout.map(Duration::from_secs_f64);

        pyo3_async_runtimes::tokio::future_into_py_with_locals(py, locals, async move {
            let handler = blazen_core::Workflow::resume(snapshot, registrations, timeout_dur)
                .await
                .map_err(BlazenPyError::from)?;
            to_py_result(Ok(PyWorkflowHandler::new(handler)))
        })
    }

    fn __repr__(&self) -> String {
        "Workflow(...)".to_owned()
    }
}
