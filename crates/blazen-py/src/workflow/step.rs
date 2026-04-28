//! Python `@step` decorator and step registration.
//!
//! The `@step` decorator wraps a Python async function so it can be registered
//! with a [`Workflow`](super::workflow::PyWorkflow). The wrapper inspects the
//! function name and optional metadata attributes to determine routing.

use std::sync::Arc;

use pyo3::prelude::*;
use pyo3::types::{PyCFunction, PyDict, PyTuple};
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pyfunction, gen_stub_pymethods};

use blazen_events::AnyEvent;

use super::context::PyContext;
use super::session_ref::{
    PySessionRegistryHandle, step_runner, with_python_session_registry, with_session_registry,
};
use crate::error::BlazenPyError;
use blazen_events::intern_event_type;

use super::event::{PyEvent, any_event_to_py_event, py_event_to_any_event};

/// Internal wrapper created by the `@step` decorator.
///
/// Holds the Python async function and metadata for registration with
/// the workflow engine.
///
/// Example:
///     >>> @step
///     ... async def analyze(ctx: Context, ev: Event) -> Event:
///     ...     return Event("ResultEvent", answer=42)
#[gen_stub_pyclass]
#[pyclass(name = "_StepWrapper")]
pub struct PyStepWrapper {
    /// The wrapped Python async function.
    pub(crate) func: Py<PyAny>,
    /// The step name (derived from the function name).
    #[pyo3(get)]
    pub(crate) name: String,
    /// Event type identifiers this step accepts.
    #[pyo3(get, set)]
    pub(crate) accepts: Vec<String>,
    /// Event type identifiers this step may emit.
    #[pyo3(get, set)]
    pub(crate) emits: Vec<String>,
    /// Maximum concurrency (0 = unlimited).
    #[pyo3(get, set)]
    pub(crate) max_concurrency: usize,
    /// Whether the wrapped function is an async coroutine function.
    pub(crate) is_async: bool,
}

impl PyStepWrapper {
    /// Clone the inner `Py<PyAny>` function handle while holding the GIL.
    fn clone_func(&self, py: Python<'_>) -> Py<PyAny> {
        self.func.clone_ref(py)
    }
}

#[gen_stub_pymethods]
#[pymethods]
impl PyStepWrapper {
    /// Call the underlying function (makes the wrapper callable like the
    /// original function for testing convenience).
    #[pyo3(signature = (ctx, event))]
    fn __call__<'py>(
        &self,
        py: Python<'py>,
        ctx: &Bound<'py, PyAny>,
        event: &Bound<'py, PyAny>,
    ) -> PyResult<Py<PyAny>> {
        self.func.call1(py, (ctx, event))
    }

    fn __repr__(&self) -> String {
        format!(
            "_StepWrapper(name='{}', accepts={:?}, emits={:?})",
            self.name, self.accepts, self.emits
        )
    }
}

impl PyStepWrapper {
    /// Convert this wrapper into a [`StepRegistration`](blazen_core::StepRegistration)
    /// that can be added to a [`WorkflowBuilder`](blazen_core::WorkflowBuilder),
    /// using the provided task locals for async Python function calls.
    pub fn to_registration_with_locals(
        &self,
        locals: pyo3_async_runtimes::TaskLocals,
    ) -> PyResult<blazen_core::StepRegistration> {
        let accepts: Vec<&'static str> =
            self.accepts.iter().map(|s| intern_event_type(s)).collect();

        let emits: Vec<&'static str> = self.emits.iter().map(|s| intern_event_type(s)).collect();

        // Clone the function handle while we have the GIL
        let func = Python::attach(|py| self.clone_func(py));
        let step_name = self.name.clone();
        let is_async = self.is_async;

        let handler: blazen_core::StepFn = Arc::new(
            move |event: Box<dyn AnyEvent>,
                  ctx: blazen_core::Context|
                  -> std::pin::Pin<
                Box<
                    dyn std::future::Future<
                            Output = std::result::Result<
                                blazen_core::StepOutput,
                                blazen_core::WorkflowError,
                            >,
                        > + Send,
                >,
            > {
                let func = Python::attach(|py| func.clone_ref(py));
                let step_name = step_name.clone();
                let locals = locals.clone();

                Box::pin(async move {
                    // Pull the session-ref registry off the context. We need
                    // to install it in two places: as a Tokio `task_local!`
                    // (for the synchronous step path and the result/stream
                    // futures) and as a Python `ContextVar` (for the async
                    // step path, since `pyo3-async-runtimes` runs the user
                    // coroutine on Python's asyncio loop thread).
                    let registry = ctx.session_refs_arc().await;
                    let registry_for_py = Arc::clone(&registry);
                    let py_ctx = PyContext::new(ctx);

                    with_session_registry(registry, async move {
                        // Convert Rust event to PyEvent (inside the Tokio
                        // scope so input markers from prior steps resolve).
                        let py_event = any_event_to_py_event(&*event);

                        // Call the Python function
                        let py_result: Py<PyAny> = if is_async {
                            // Async path: build the user coroutine and wrap it
                            // in `_blazen_run_step(handle, user_coro)` so the
                            // session-registry contextvar is set inside the
                            // asyncio Task before the user body runs.
                            let coroutine: Py<PyAny> =
                                Python::attach(|py| -> PyResult<Py<PyAny>> {
                                    let py_event_obj = Py::new(py, py_event)?;
                                    let py_ctx_obj = Py::new(py, py_ctx)?;
                                    let user_coro = func.call1(py, (py_ctx_obj, py_event_obj))?;
                                    let handle = Py::new(
                                        py,
                                        PySessionRegistryHandle::new(Arc::clone(&registry_for_py)),
                                    )?;
                                    let runner = step_runner(py)?;
                                    let wrapped = runner.call1((handle, user_coro))?;
                                    Ok(wrapped.unbind())
                                })
                                .map_err(|e: PyErr| {
                                    blazen_core::WorkflowError::StepFailed {
                                        step_name: step_name.clone(),
                                        source: Box::new(BlazenPyError::Workflow(e.to_string())),
                                    }
                                })?;

                            // Convert the wrapped coroutine to a Rust future and await it.
                            let future = Python::attach(|py| {
                                pyo3_async_runtimes::into_future_with_locals(
                                    &locals,
                                    coroutine.into_bound(py),
                                )
                            })
                            .map_err(|e: PyErr| {
                                blazen_core::WorkflowError::StepFailed {
                                    step_name: step_name.clone(),
                                    source: Box::new(BlazenPyError::Workflow(e.to_string())),
                                }
                            })?;

                            pyo3_async_runtimes::tokio::scope(locals.clone(), future)
                                .await
                                .map_err(|e: PyErr| blazen_core::WorkflowError::StepFailed {
                                    step_name: step_name.clone(),
                                    source: Box::new(BlazenPyError::Workflow(e.to_string())),
                                })?
                        } else {
                            // Sync path: call the user function with the
                            // contextvar installed so any event constructors
                            // it builds see the session registry. The Tokio
                            // task_local! is also live, but the contextvar
                            // covers the case where the sync step itself
                            // calls into more Python code that constructs
                            // events.
                            Python::attach(|py| -> PyResult<Py<PyAny>> {
                                let py_event_obj = Py::new(py, py_event)?;
                                let py_ctx_obj = Py::new(py, py_ctx)?;
                                with_python_session_registry(
                                    py,
                                    Arc::clone(&registry_for_py),
                                    |py| func.call1(py, (py_ctx_obj, py_event_obj)),
                                )
                            })
                            .map_err(|e: PyErr| {
                                blazen_core::WorkflowError::StepFailed {
                                    step_name: step_name.clone(),
                                    source: Box::new(BlazenPyError::Workflow(e.to_string())),
                                }
                            })?
                        };

                        // Convert the Python return value back to a Rust event
                        Python::attach(|py| py_result_to_step_output(py, &py_result, &step_name))
                    })
                    .await
                })
            },
        );

        Ok(blazen_core::StepRegistration::new(
            self.name.clone(),
            accepts,
            emits,
            handler,
            self.max_concurrency,
        ))
    }
}

/// Convert a Python step return value into a [`StepOutput`](blazen_core::StepOutput).
///
/// Handles `None` (no output), a list of `PyEvent`s (fan-out), or a single
/// `PyEvent`.
fn py_result_to_step_output(
    py: Python<'_>,
    py_result: &Py<PyAny>,
    step_name: &str,
) -> std::result::Result<blazen_core::StepOutput, blazen_core::WorkflowError> {
    let bound = py_result.bind(py);

    if bound.is_none() {
        return Ok(blazen_core::StepOutput::None);
    }

    if let Ok(list) = bound.cast::<pyo3::types::PyList>() {
        let mut events: Vec<Box<dyn AnyEvent>> = Vec::with_capacity(list.len());
        for item in list.iter() {
            let ev: Bound<'_, PyEvent> = item
                .cast::<PyEvent>()
                .map_err(|e| blazen_core::WorkflowError::StepFailed {
                    step_name: step_name.to_owned(),
                    source: Box::new(BlazenPyError::Workflow(e.to_string())),
                })?
                .clone();
            events.push(py_event_to_any_event(&ev.borrow()));
        }
        return Ok(blazen_core::StepOutput::Multiple(events));
    }

    let ev_bound: Bound<'_, PyEvent> = bound
        .cast::<PyEvent>()
        .map_err(|e| blazen_core::WorkflowError::StepFailed {
            step_name: step_name.to_owned(),
            source: Box::new(BlazenPyError::Workflow(format!(
                "step must return an Event, list of Events, or None: {e}"
            ))),
        })?
        .clone();
    let ev = ev_bound.borrow();
    Ok(blazen_core::StepOutput::Single(py_event_to_any_event(&ev)))
}

// ---------------------------------------------------------------------------
// @step decorator function
// ---------------------------------------------------------------------------

/// Decorator that wraps a Python async function as a workflow step.
///
/// The decorated function should have the signature:
///
///     async def my_step(ctx: Context, ev: Event) -> Event | list[Event] | None
///
/// or:
///
///     def my_step(ctx: Context, ev: Event) -> Event | list[Event] | None
///
/// By default the step accepts `StartEvent` and emits any event type.
/// Override with `.accepts` and `.emits` attributes on the returned wrapper.
///
/// Example:
///     >>> @step
///     ... async def analyze(ctx: Context, ev: Event) -> Event:
///     ...     return StopEvent(result={"done": True})
///
///     # Customize accepted event types:
///     >>> analyze.accepts = ["AnalyzeEvent"]
///     >>> analyze.emits = ["StopEvent"]
/// Infer the `accepts` list from the type annotation of the `ev` parameter.
///
/// If the second parameter (after `ctx`) is annotated with an `Event` subclass,
/// use that class name as the accepted event type. Falls back to
/// `["blazen::StartEvent"]` when:
/// - No annotation is present
/// - The annotation is the base `Event` class
/// - The annotation cannot be resolved
fn infer_accepts_from_hints(py: Python<'_>, func: &Py<PyAny>) -> Vec<String> {
    let default = vec!["blazen::StartEvent".to_owned()];

    // Use typing.get_type_hints() to resolve annotations
    let Ok(typing) = py.import("typing") else {
        return default;
    };
    let Ok(hints) = typing.call_method1("get_type_hints", (func,)) else {
        return default;
    };
    let Ok(hints_dict) = hints.cast::<PyDict>() else {
        return default;
    };

    // Look for the "ev" or "event" parameter annotation
    let ev_hint = hints_dict
        .get_item("ev")
        .ok()
        .flatten()
        .or_else(|| hints_dict.get_item("event").ok().flatten());

    let Some(ev_type) = ev_hint else {
        return default;
    };

    // Check if it's a subclass of our Event class
    let event_cls_bound = py.get_type::<super::event::PyEvent>().into_any();
    let Ok(event_cls) = event_cls_bound.cast::<pyo3::types::PyType>() else {
        return default;
    };

    // Check if the annotated type is a subclass of Event using Python's issubclass()
    let Ok(builtins) = py.import("builtins") else {
        return default;
    };
    let is_subclass: bool = match builtins.call_method1("issubclass", (&ev_type, &event_cls)) {
        Ok(r) => match r.extract() {
            Ok(b) => b,
            Err(_) => return default,
        },
        Err(_) => return default,
    };

    if !is_subclass {
        return default;
    }

    // Get the class name
    let class_name: String = match ev_type.getattr("__name__") {
        Ok(n) => match n.extract() {
            Ok(s) => s,
            Err(_) => return default,
        },
        Err(_) => return default,
    };

    // Map known classes to their internal event type strings
    match class_name.as_str() {
        "Event" => default, // Bare Event = default StartEvent
        "StartEvent" => vec!["blazen::StartEvent".to_owned()],
        "StopEvent" => vec!["blazen::StopEvent".to_owned()],
        name => vec![name.to_owned()],
    }
}

#[gen_stub_pyfunction]
#[pyfunction]
#[pyo3(signature = (func=None, *, accepts=None, emits=None, max_concurrency=0))]
pub fn step(
    py: Python<'_>,
    func: Option<Py<PyAny>>,
    accepts: Option<Vec<String>>,
    emits: Option<Vec<String>>,
    max_concurrency: usize,
) -> PyResult<Py<PyAny>> {
    let emits = emits.unwrap_or_default();

    if let Some(func) = func {
        // Used as @step (without arguments)
        let name: String = func.getattr(py, "__name__")?.extract(py)?;

        // Detect whether the function is async
        let inspect = py.import("inspect")?;
        let is_async: bool = inspect
            .call_method1("iscoroutinefunction", (&func,))?
            .extract()?;

        // Infer accepts from type hints if not explicitly provided
        let accepts = accepts.unwrap_or_else(|| infer_accepts_from_hints(py, &func));

        let wrapper = PyStepWrapper {
            func,
            name,
            accepts,
            emits,
            max_concurrency,
            is_async,
        };
        Ok(Py::new(py, wrapper)?.into_any())
    } else {
        // Used as @step(accepts=..., emits=...) -- return a decorator
        let decorator = PyCFunction::new_closure(
            py,
            None,
            None,
            move |args: &Bound<'_, PyTuple>,
                  _kwargs: Option<&Bound<'_, PyDict>>|
                  -> PyResult<Py<PyAny>> {
                let py = args.py();
                let func: Py<PyAny> = args.get_item(0)?.extract()?;
                let name: String = func.getattr(py, "__name__")?.extract(py)?;

                // Detect whether the function is async
                let inspect = py.import("inspect")?;
                let is_async: bool = inspect
                    .call_method1("iscoroutinefunction", (&func,))?
                    .extract()?;

                // Infer accepts from type hints if not explicitly provided
                let resolved_accepts = match &accepts {
                    Some(a) => a.clone(),
                    None => infer_accepts_from_hints(py, &func),
                };

                let wrapper = PyStepWrapper {
                    func,
                    name,
                    accepts: resolved_accepts,
                    emits: emits.clone(),
                    max_concurrency,
                    is_async,
                };
                Ok(Py::new(py, wrapper)?.into_any())
            },
        )?;
        Ok(decorator.unbind().into())
    }
}
