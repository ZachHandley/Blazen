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

use blazen_pipeline::{ConditionFn, InputMapperFn, JoinStrategy, ParallelStage, Stage, StageKind};

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

/// A single sequential stage in a pipeline.
///
/// Wraps a `Workflow` with optional input mapping and conditional execution.
///
/// Example:
///     >>> ingest = Workflow("ingest", [extract_step])
///     >>> stage = Stage(name="ingest", workflow=ingest)
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
///     >>> parallel = ParallelStage(
///     ...     name="fanout",
///     ...     branches=[stage_a, stage_b, stage_c],
///     ...     join_strategy=JoinStrategy.WaitAll,
///     ... )
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

/// A stage in a pipeline -- either sequential or parallel.
#[gen_stub_pyclass_enum]
#[pyclass(name = "StageKind", eq, eq_int, from_py_object)]
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum PyStageKind {
    Sequential,
    Parallel,
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
        }
    }
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

// Suppress unused-import warning on `json_to_py` (kept for symmetry with
// future getters that may need it).
const _: fn() = || {
    let _ = json_to_py;
};
