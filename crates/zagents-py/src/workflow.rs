//! Python wrapper for [`Workflow`](zagents_core::Workflow) and
//! [`WorkflowBuilder`](zagents_core::WorkflowBuilder).

use std::sync::Arc;
use std::time::Duration;

use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::error::{ZAgentsPyError, to_py_result};
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
    inner: Arc<zagents_core::Workflow>,
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
        let mut builder = zagents_core::WorkflowBuilder::new(name);

        for step in &steps {
            let registration = step.to_registration()?;
            builder = builder.step(registration);
        }

        if let Some(t) = timeout {
            builder = builder.timeout(Duration::from_secs_f64(t));
        }

        let workflow = builder.build().map_err(ZAgentsPyError::from)?;

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
            let handler = workflow.run(data).await.map_err(ZAgentsPyError::from)?;
            to_py_result(Ok(PyWorkflowHandler::new(handler)))
        })
    }

    fn __repr__(&self) -> String {
        "Workflow(...)".to_owned()
    }
}
