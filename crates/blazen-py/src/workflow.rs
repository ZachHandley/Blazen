//! Python wrapper for [`Workflow`](blazen_core::Workflow) and
//! [`WorkflowBuilder`](blazen_core::WorkflowBuilder).

use std::sync::Arc;
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
    inner: Arc<blazen_core::Workflow>,
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
    fn new(
        name: &str,
        steps: Vec<PyRef<'_, PyStepWrapper>>,
        timeout: Option<f64>,
    ) -> PyResult<Self> {
        let mut builder = blazen_core::WorkflowBuilder::new(name);

        for step in &steps {
            let registration = step.to_registration()?;
            builder = builder.step(registration);
        }

        if let Some(t) = timeout {
            builder = builder.timeout(Duration::from_secs_f64(t));
        }

        let workflow = builder.build().map_err(BlazenPyError::from)?;

        Ok(Self {
            inner: Arc::new(workflow),
        })
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

        let workflow = self.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
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

        let mut registrations = Vec::with_capacity(steps.len());
        for step in &steps {
            registrations.push(step.to_registration()?);
        }

        let timeout_dur = timeout.map(Duration::from_secs_f64);

        pyo3_async_runtimes::tokio::future_into_py(py, async move {
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
