//! Python wrappers for [`Stage`], [`ParallelStage`], [`JoinStrategy`],
//! and [`StageKind`].
//!
//! `Stage` and `ParallelStage` are constructed from a Python `Workflow`
//! reference plus optional `input_mapper` / `condition` Python callables.
//! The actual `blazen_core::Workflow` is built lazily inside
//! [`PyPipeline::start`](super::pipeline::PyPipeline) so that the Python
//! task locals available at run-time are captured into each stage's
//! workflow handler.

use std::sync::Arc;

use pyo3::prelude::*;
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pyclass_enum, gen_stub_pymethods};

use blazen_pipeline::{
    ConditionFn, InputMapperFn, JoinStrategy, LoopDecision, LoopStage, LoopUntilFn, ParallelStage,
    Stage, StageKind,
};

use crate::convert::{json_to_py, py_to_json};
use crate::pipeline::state::PyPipelineState;
use crate::workflow::workflow::PyWorkflow;

/// How to join the results of parallel branches.
#[gen_stub_pyclass_enum]
#[pyclass(name = "JoinStrategy", eq, eq_int, from_py_object)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PyJoinStrategy {
    /// Wait for all branches to complete and collect all results.
    WaitAll,
    /// Return as soon as the first branch completes; cancel the rest.
    FirstCompletes,
}

impl From<PyJoinStrategy> for JoinStrategy {
    fn from(p: PyJoinStrategy) -> Self {
        match p {
            PyJoinStrategy::WaitAll => Self::WaitAll,
            PyJoinStrategy::FirstCompletes => Self::FirstCompletes,
        }
    }
}

/// The decision returned by a [`LoopStage`]'s ``until`` predicate after each
/// round.
///
/// Construct one of the three variants via the classmethod factories:
///
/// ```text
/// >>> LoopDecision.cont()           # run the inner stage again
/// >>> LoopDecision.done()           # stop cleanly; the loop succeeds
/// >>> LoopDecision.abort("reason")  # stop with an error
/// ```
///
/// (`cont` is named without a trailing underscore so it reads as
/// ``LoopDecision.cont()``; ``continue`` is a Python keyword.)
#[gen_stub_pyclass]
#[pyclass(name = "LoopDecision", from_py_object)]
#[derive(Clone)]
pub struct PyLoopDecision {
    pub(crate) inner: LoopDecision,
}

#[gen_stub_pymethods]
#[pymethods]
impl PyLoopDecision {
    /// Run the inner stage again (subject to the ``max_iterations`` cap).
    #[classmethod]
    #[pyo3(name = "cont")]
    fn cont(_cls: &Bound<'_, pyo3::types::PyType>) -> Self {
        Self {
            inner: LoopDecision::Continue,
        }
    }

    /// Stop looping cleanly; the loop stage succeeds.
    #[classmethod]
    fn done(_cls: &Bound<'_, pyo3::types::PyType>) -> Self {
        Self {
            inner: LoopDecision::Done,
        }
    }

    /// Stop looping with an error carrying the given reason.
    #[classmethod]
    fn abort(_cls: &Bound<'_, pyo3::types::PyType>, reason: String) -> Self {
        Self {
            inner: LoopDecision::Abort(reason),
        }
    }

    /// ``True`` if this decision is [`LoopDecision.cont`].
    #[getter]
    fn is_continue(&self) -> bool {
        matches!(self.inner, LoopDecision::Continue)
    }

    /// ``True`` if this decision is [`LoopDecision.done`].
    #[getter]
    fn is_done(&self) -> bool {
        matches!(self.inner, LoopDecision::Done)
    }

    /// The abort reason if this decision is [`LoopDecision.abort`], else
    /// ``None``.
    #[getter]
    fn abort_reason(&self) -> Option<String> {
        match &self.inner {
            LoopDecision::Abort(reason) => Some(reason.clone()),
            _ => None,
        }
    }

    fn __repr__(&self) -> String {
        match &self.inner {
            LoopDecision::Continue => "LoopDecision.cont()".to_owned(),
            LoopDecision::Done => "LoopDecision.done()".to_owned(),
            LoopDecision::Abort(reason) => format!("LoopDecision.abort({reason:?})"),
        }
    }
}

/// A single sequential stage in a pipeline.
///
/// Wraps a `Workflow` with optional input mapping and conditional execution.
///
/// Example:
/// ```text
///  >>> ingest = Workflow("ingest", [extract_step])
///  >>> stage = Stage(name="ingest", workflow=ingest)
/// ```
#[gen_stub_pyclass]
#[pyclass(name = "Stage")]
pub struct PyStage {
    pub(crate) name: String,
    pub(crate) workflow: Py<PyWorkflow>,
    pub(crate) input_mapper: Option<Py<PyAny>>,
    pub(crate) condition: Option<Py<PyAny>>,
}

#[gen_stub_pymethods]
#[pymethods]
impl PyStage {
    /// Create a new sequential stage.
    ///
    /// Args:
    ///     name: Human-readable name for the stage (used in results
    ///         and event provenance).
    ///     workflow: The `Workflow` to execute for this stage.
    ///     input_mapper: Optional callable `(state: PipelineState) -> Any`
    ///         that transforms pipeline state into the workflow input.
    ///         When `None`, the previous stage's output (or pipeline input
    ///         for the first stage) is passed through directly.
    ///     condition: Optional callable `(state: PipelineState) -> bool`
    ///         that decides whether the stage runs. When `None` the stage
    ///         always runs. When the callable returns `False` the stage is
    ///         skipped.
    #[new]
    #[pyo3(signature = (name, workflow, input_mapper=None, condition=None))]
    fn new(
        name: &str,
        workflow: Py<PyWorkflow>,
        input_mapper: Option<Py<PyAny>>,
        condition: Option<Py<PyAny>>,
    ) -> Self {
        Self {
            name: name.to_owned(),
            workflow,
            input_mapper,
            condition,
        }
    }

    /// The stage's name.
    #[getter]
    fn name(&self) -> &str {
        &self.name
    }

    fn __repr__(&self) -> String {
        format!(
            "Stage(name={:?}, has_input_mapper={}, has_condition={})",
            self.name,
            self.input_mapper.is_some(),
            self.condition.is_some()
        )
    }
}

/// A parallel stage running multiple branches concurrently.
///
/// Each branch is a [`Stage`]. All branches start simultaneously and the
/// `join_strategy` determines how results are collected.
///
/// Example:
/// ```text
///  >>> parallel = ParallelStage(
///  ...     name="fanout",
///  ...     branches=[stage_a, stage_b, stage_c],
///  ...     join_strategy=JoinStrategy.WaitAll,
///  ... )
/// ```
#[gen_stub_pyclass]
#[pyclass(name = "ParallelStage")]
pub struct PyParallelStage {
    pub(crate) name: String,
    pub(crate) branches: Vec<Py<PyStage>>,
    pub(crate) join_strategy: PyJoinStrategy,
}

#[gen_stub_pymethods]
#[pymethods]
impl PyParallelStage {
    /// Create a new parallel stage.
    ///
    /// Args:
    ///     name: Human-readable name for the parallel group.
    ///     branches: A list of `Stage` objects to run concurrently.
    ///     join_strategy: How to join branch results. Defaults to
    ///         `JoinStrategy.WaitAll`.
    #[new]
    #[pyo3(signature = (name, branches, join_strategy=None))]
    fn new(
        name: &str,
        branches: Vec<PyRef<'_, PyStage>>,
        join_strategy: Option<PyJoinStrategy>,
    ) -> Self {
        let branch_refs: Vec<Py<PyStage>> = branches.into_iter().map(Into::into).collect();
        Self {
            name: name.to_owned(),
            branches: branch_refs,
            join_strategy: join_strategy.unwrap_or(PyJoinStrategy::WaitAll),
        }
    }

    /// The stage's name.
    #[getter]
    fn name(&self) -> &str {
        &self.name
    }

    /// The configured join strategy.
    #[getter]
    fn join_strategy(&self) -> PyJoinStrategy {
        self.join_strategy
    }

    fn __repr__(&self) -> String {
        format!(
            "ParallelStage(name={:?}, branches={}, join_strategy={:?})",
            self.name,
            self.branches.len(),
            self.join_strategy
        )
    }
}

/// The inner stage wrapped by a [`LoopStage`] -- either sequential or
/// parallel.
pub(crate) enum PyLoopInner {
    Sequential(Py<PyStage>),
    Parallel(Py<PyParallelStage>),
}

/// A loop stage that re-runs an inner stage until a predicate signals
/// completion (or a maximum iteration count is reached).
///
/// The inner stage may be a [`Stage`] or a [`ParallelStage`]; the ``until``
/// predicate is evaluated after each round with the current
/// :class:`PipelineState` and the number of rounds completed so far (1-based),
/// returning a :class:`LoopDecision`.
///
/// Example:
/// ```text
///  >>> def keep_going(state, rounds):
///  ...     if rounds >= 5:
///  ...         return LoopDecision.done()
///  ...     return LoopDecision.cont()
///  >>>
///  >>> refine = LoopStage(
///  ...     name="refine",
///  ...     max_iterations=10,
///  ...     inner=refine_stage,
///  ...     until=keep_going,
///  ... )
/// ```
#[gen_stub_pyclass]
#[pyclass(name = "LoopStage")]
pub struct PyLoopStage {
    pub(crate) name: String,
    pub(crate) max_iterations: u32,
    pub(crate) inner: PyLoopInner,
    pub(crate) until: Py<PyAny>,
}

#[gen_stub_pymethods]
#[pymethods]
impl PyLoopStage {
    /// Create a new loop stage.
    ///
    /// Args:
    ///     name: Human-readable name for the loop stage (used in results
    ///         and event provenance).
    ///     max_iterations: Hard cap on the number of rounds. The loop stops
    ///         once this many rounds have run even if ``until`` never
    ///         returns ``LoopDecision.done()``.
    ///     inner: The inner ``Stage`` or ``ParallelStage`` to run each round.
    ///     until: Callable ``(state: PipelineState, rounds: int) ->
    ///         LoopDecision`` evaluated after each round. ``rounds`` is the
    ///         1-based count of rounds completed so far.
    #[new]
    #[pyo3(signature = (name, max_iterations, inner, until))]
    fn new(
        py: Python<'_>,
        name: &str,
        max_iterations: u32,
        inner: Py<PyAny>,
        until: Py<PyAny>,
    ) -> PyResult<Self> {
        let bound = inner.bind(py);
        let inner = if let Ok(stage) = bound.cast::<PyStage>() {
            PyLoopInner::Sequential(stage.clone().unbind())
        } else if let Ok(parallel) = bound.cast::<PyParallelStage>() {
            PyLoopInner::Parallel(parallel.clone().unbind())
        } else {
            return Err(pyo3::exceptions::PyTypeError::new_err(
                "LoopStage `inner` must be a Stage or ParallelStage",
            ));
        };
        Ok(Self {
            name: name.to_owned(),
            max_iterations,
            inner,
            until,
        })
    }

    /// The stage's name.
    #[getter]
    fn name(&self) -> &str {
        &self.name
    }

    /// The hard cap on the number of rounds.
    #[getter]
    fn max_iterations(&self) -> u32 {
        self.max_iterations
    }

    fn __repr__(&self) -> String {
        format!(
            "LoopStage(name={:?}, max_iterations={})",
            self.name, self.max_iterations
        )
    }
}

/// A stage in a pipeline -- sequential, parallel, or a loop.
#[gen_stub_pyclass_enum]
#[pyclass(name = "StageKind", eq, eq_int, from_py_object)]
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum PyStageKind {
    Sequential,
    Parallel,
    Loop,
}

// ---------------------------------------------------------------------------
// Materialization helpers used by `PyPipeline::start`/`resume`.
// ---------------------------------------------------------------------------

/// Build a Rust `Stage` from a `PyStage`, materializing the inner workflow
/// using the current Python task locals.
///
/// Returns the materialized `Stage` plus an optional per-stage timeout.
pub(crate) fn build_rust_stage(
    py: Python<'_>,
    stage: &PyStage,
    locals: &pyo3_async_runtimes::TaskLocals,
) -> PyResult<Stage> {
    let workflow_ref = stage.workflow.borrow(py);
    let workflow = workflow_ref.build_workflow_with_locals(py, locals.clone())?;

    let input_mapper = stage
        .input_mapper
        .as_ref()
        .map(|cb| build_input_mapper(cb.clone_ref(py)));
    let condition = stage
        .condition
        .as_ref()
        .map(|cb| build_condition(cb.clone_ref(py)));

    Ok(Stage {
        name: stage.name.clone(),
        workflow,
        input_mapper,
        condition,
        output_mapper: None,
    })
}

pub(crate) fn build_rust_parallel_stage(
    py: Python<'_>,
    parallel: &PyParallelStage,
    locals: &pyo3_async_runtimes::TaskLocals,
) -> PyResult<ParallelStage> {
    let mut branches = Vec::with_capacity(parallel.branches.len());
    for branch in &parallel.branches {
        let branch_ref = branch.borrow(py);
        let rust_stage = build_rust_stage(py, &branch_ref, locals)?;
        branches.push(rust_stage);
    }
    Ok(ParallelStage {
        name: parallel.name.clone(),
        branches,
        join_strategy: parallel.join_strategy.into(),
    })
}

pub(crate) enum PendingStage {
    Sequential(Py<PyStage>),
    Parallel(Py<PyParallelStage>),
    Loop(Py<PyLoopStage>),
}

impl PendingStage {
    pub(crate) fn materialize(
        &self,
        py: Python<'_>,
        locals: &pyo3_async_runtimes::TaskLocals,
    ) -> PyResult<StageKind> {
        match self {
            PendingStage::Sequential(s) => {
                let stage = s.borrow(py);
                Ok(StageKind::Sequential(build_rust_stage(py, &stage, locals)?))
            }
            PendingStage::Parallel(p) => {
                let parallel = p.borrow(py);
                Ok(StageKind::Parallel(build_rust_parallel_stage(
                    py, &parallel, locals,
                )?))
            }
            PendingStage::Loop(l) => {
                let loop_stage = l.borrow(py);
                Ok(StageKind::Loop(build_rust_loop_stage(
                    py,
                    &loop_stage,
                    locals,
                )?))
            }
        }
    }
}

/// Build a Rust `LoopStage` from a `PyLoopStage`, materializing the inner
/// stage and bridging the Python `until` callback into a [`LoopUntilFn`].
pub(crate) fn build_rust_loop_stage(
    py: Python<'_>,
    loop_stage: &PyLoopStage,
    locals: &pyo3_async_runtimes::TaskLocals,
) -> PyResult<LoopStage> {
    let inner = match &loop_stage.inner {
        PyLoopInner::Sequential(s) => {
            let stage = s.borrow(py);
            StageKind::Sequential(build_rust_stage(py, &stage, locals)?)
        }
        PyLoopInner::Parallel(p) => {
            let parallel = p.borrow(py);
            StageKind::Parallel(build_rust_parallel_stage(py, &parallel, locals)?)
        }
    };
    let until = build_loop_until(loop_stage.until.clone_ref(py));
    Ok(LoopStage {
        name: loop_stage.name.clone(),
        max_iterations: loop_stage.max_iterations,
        inner: Box::new(inner),
        until,
        on_round_complete: None,
    })
}

#[allow(unsafe_code)]
fn build_input_mapper(callback: Py<PyAny>) -> InputMapperFn {
    let callback = Arc::new(callback);
    Arc::new(move |state| -> serde_json::Value {
        let cb = Arc::clone(&callback);
        Python::attach(|py| -> serde_json::Value {
            // SAFETY: `state` is a borrowed reference valid for the duration
            // of this closure call. `PyPipelineState` does not escape the
            // closure call.
            let state_view = unsafe { PyPipelineState::from_ref(state) };
            let Ok(py_state) = Py::new(py, state_view) else {
                return serde_json::Value::Null;
            };
            match cb.call1(py, (py_state,)) {
                Ok(result) => py_to_json(py, result.bind(py)).unwrap_or(serde_json::Value::Null),
                Err(e) => {
                    e.print(py);
                    serde_json::Value::Null
                }
            }
        })
    })
}

#[allow(unsafe_code)]
fn build_condition(callback: Py<PyAny>) -> ConditionFn {
    let callback = Arc::new(callback);
    Arc::new(move |state| -> bool {
        let cb = Arc::clone(&callback);
        Python::attach(|py| -> bool {
            let state_view = unsafe { PyPipelineState::from_ref(state) };
            let Ok(py_state) = Py::new(py, state_view) else {
                return false;
            };
            match cb.call1(py, (py_state,)) {
                Ok(result) => result.extract::<bool>(py).unwrap_or(false),
                Err(e) => {
                    e.print(py);
                    false
                }
            }
        })
    })
}

#[allow(unsafe_code)]
fn build_loop_until(callback: Py<PyAny>) -> LoopUntilFn {
    let callback = Arc::new(callback);
    Arc::new(move |state, rounds| -> LoopDecision {
        let cb = Arc::clone(&callback);
        Python::attach(|py| -> LoopDecision {
            // SAFETY: `state` is a borrowed reference valid for the duration
            // of this closure call. `PyPipelineState` does not escape the
            // closure call.
            let state_view = unsafe { PyPipelineState::from_ref(state) };
            let Ok(py_state) = Py::new(py, state_view) else {
                return LoopDecision::Abort(
                    "failed to construct PipelineState view for loop predicate".to_owned(),
                );
            };
            match cb.call1(py, (py_state, rounds)) {
                Ok(result) => match result.bind(py).cast::<PyLoopDecision>() {
                    Ok(decision) => decision.borrow().inner.clone(),
                    Err(_) => LoopDecision::Abort(
                        "loop `until` callback did not return a LoopDecision".to_owned(),
                    ),
                },
                Err(e) => {
                    e.print(py);
                    LoopDecision::Abort("loop `until` callback raised an exception".to_owned())
                }
            }
        })
    })
}

// Suppress unused-import warning on `json_to_py` (kept for symmetry with
// future getters that may need it).
const _: fn() = || {
    let _ = json_to_py;
};
