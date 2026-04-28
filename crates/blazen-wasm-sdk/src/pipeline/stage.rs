//! WASM bindings for pipeline stages.
//!
//! Exposes [`WasmStage`] and [`WasmParallelStage`] as `wasm_bindgen` classes
//! wrapping [`blazen_pipeline::Stage`] and [`blazen_pipeline::ParallelStage`].

use std::sync::{Arc, Mutex};

use wasm_bindgen::prelude::*;

use blazen_pipeline::{ConditionFn, InputMapperFn, PipelineState};

use crate::pipeline::state::WasmPipelineState;
use crate::workflow::WasmWorkflow;

// ---------------------------------------------------------------------------
// JsClosure — Send+Sync newtype around a JS callback
// ---------------------------------------------------------------------------

/// Newtype around a [`js_sys::Function`] that vacuously implements
/// `Send + Sync` so JS callables can satisfy the `Send + Sync` bounds on
/// [`InputMapperFn`] / [`ConditionFn`].
///
/// SAFETY: wasm32 is single-threaded, so every closure access happens on
/// the same JS thread that constructed the function reference.
struct JsClosure(js_sys::Function);

#[allow(unsafe_code)]
unsafe impl Send for JsClosure {}
#[allow(unsafe_code)]
unsafe impl Sync for JsClosure {}

/// Build a [`WasmPipelineState`] snapshot from a [`PipelineState`] reference
/// using only the public accessors on `PipelineState`. The `shared` map and
/// per-stage `stage_results` are not iterable through the public API, so the
/// snapshot exposes the original pipeline `input` and the last stage's
/// output (synthesized via [`PipelineState::last_result`]).
fn snapshot_state(state: &PipelineState) -> WasmPipelineState {
    WasmPipelineState {
        input: state.input().clone(),
        shared: std::collections::HashMap::new(),
        stage_results: Vec::new(),
    }
}

/// Wrap a JS function as an [`InputMapperFn`].
///
/// The returned closure invokes the JS callback synchronously via
/// [`js_sys::Function::call1`] with a serialized [`WasmPipelineState`]
/// argument and converts the result back into a [`serde_json::Value`].
/// Any error (call failure, deserialization failure) collapses to
/// [`serde_json::Value::Null`].
fn build_input_mapper(callback: js_sys::Function) -> InputMapperFn {
    let wrapper = Arc::new(JsClosure(callback));
    Arc::new(move |state: &PipelineState| -> serde_json::Value {
        let snapshot = snapshot_state(state);
        let state_js = match serde_wasm_bindgen::to_value(&snapshot) {
            Ok(v) => v,
            Err(_) => JsValue::NULL,
        };
        match wrapper.0.call1(&JsValue::NULL, &state_js) {
            Ok(result) => {
                serde_wasm_bindgen::from_value(result).unwrap_or(serde_json::Value::Null)
            }
            Err(_) => serde_json::Value::Null,
        }
    })
}

/// Wrap a JS function as a [`ConditionFn`].
///
/// Mirrors [`build_input_mapper`] but coerces the result to `bool`,
/// defaulting to `false` on any error.
fn build_condition(callback: js_sys::Function) -> ConditionFn {
    let wrapper = Arc::new(JsClosure(callback));
    Arc::new(move |state: &PipelineState| -> bool {
        let snapshot = snapshot_state(state);
        let state_js = match serde_wasm_bindgen::to_value(&snapshot) {
            Ok(v) => v,
            Err(_) => JsValue::NULL,
        };
        match wrapper.0.call1(&JsValue::NULL, &state_js) {
            Ok(result) => result.as_bool().unwrap_or(false),
            Err(_) => false,
        }
    })
}

// ---------------------------------------------------------------------------
// JoinStrategy
// ---------------------------------------------------------------------------

/// Strategy used by a [`WasmParallelStage`] to join its concurrent branches.
///
/// - `WaitAll`: wait for every branch to complete and aggregate the
///   results.
/// - `FirstCompletes`: return as soon as the first branch finishes; the
///   remaining branches are cancelled.
#[wasm_bindgen(js_name = "JoinStrategy")]
#[derive(Debug, Clone, Copy)]
pub enum WasmJoinStrategy {
    /// Wait for every branch to complete and aggregate the results.
    WaitAll,
    /// Return as soon as the first branch finishes; cancel the rest.
    FirstCompletes,
}

impl From<WasmJoinStrategy> for blazen_pipeline::JoinStrategy {
    fn from(value: WasmJoinStrategy) -> Self {
        match value {
            WasmJoinStrategy::WaitAll => Self::WaitAll,
            WasmJoinStrategy::FirstCompletes => Self::FirstCompletes,
        }
    }
}

// ---------------------------------------------------------------------------
// StageKind
// ---------------------------------------------------------------------------

/// Tag enum mirroring [`blazen_pipeline::StageKind`].
///
/// Surfaced for symmetry with the Python and Node bindings; the WASM
/// pipeline binding doesn't use this discriminant directly because the
/// builder takes typed [`WasmStage`] / [`WasmParallelStage`] values, but
/// JS callers may want to switch on stage kind when iterating completed
/// snapshots.
#[wasm_bindgen(js_name = "StageKind")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WasmStageKind {
    /// A single sequential stage.
    Sequential,
    /// A parallel stage with multiple branches.
    Parallel,
}

// ---------------------------------------------------------------------------
// WasmStage
// ---------------------------------------------------------------------------

/// A single sequential stage in a pipeline.
///
/// Wraps a [`Workflow`](crate::workflow::WasmWorkflow) under a human-readable
/// name. Stages are added to a [`PipelineBuilder`](super::WasmPipelineBuilder)
/// which consumes them at build time.
///
/// ```typescript
/// const wf = new Workflow("my-wf");
/// // ... addStep ...
/// const stage = new Stage("preprocess", wf);
/// ```
///
/// # Consumption semantics
///
/// A `Stage` instance can only be added to a single `Pipeline`. Once
/// consumed, subsequent attempts to use the same instance will throw a
/// `Stage already consumed by a Pipeline` error. The internal storage uses
/// [`Mutex<Option<...>>`] because `wasm-bindgen` only hands us
/// `&self`/`&mut self`, but we need to *move* the underlying
/// [`blazen_pipeline::Stage`] out when the pipeline is built.
///
/// Building a `Stage` also consumes the underlying `Workflow`'s buffered
/// step list and builder (the same single-shot rule that applies to
/// `workflow.run()` / `workflow.runHandler()`), so a `Workflow` can only
/// be wrapped in one `Stage`.
#[wasm_bindgen(js_name = "Stage")]
pub struct WasmStage {
    inner: Mutex<Option<blazen_pipeline::Stage>>,
}

#[wasm_bindgen(js_class = "Stage")]
impl WasmStage {
    /// Create a new sequential stage from a [`Workflow`](crate::workflow::WasmWorkflow).
    ///
    /// The workflow is materialized into its core
    /// [`blazen_core::Workflow`] form at construction time, so any
    /// subsequent modifications to the JS `Workflow` instance will not
    /// affect this stage.
    ///
    /// `input_mapper` is an optional `(state: PipelineState) => unknown`
    /// JS callable that transforms the pipeline state into the workflow
    /// input. When `null`/`undefined` the previous stage's output (or the
    /// pipeline input for the first stage) is passed through directly.
    ///
    /// `condition` is an optional `(state: PipelineState) => boolean` JS
    /// callable that decides whether the stage runs. When `null`/`undefined`
    /// the stage always runs; when the callable returns `false` the stage
    /// is skipped.
    ///
    /// # Errors
    ///
    /// Returns a `JsValue` error if the workflow has already been
    /// consumed (e.g. `run()` was called on it earlier) or if the
    /// underlying `WorkflowBuilder::build()` fails.
    #[wasm_bindgen(constructor)]
    pub fn new(
        name: String,
        workflow: &mut WasmWorkflow,
        input_mapper: Option<js_sys::Function>,
        condition: Option<js_sys::Function>,
    ) -> Result<WasmStage, JsValue> {
        let core_workflow = workflow.build_workflow()?;
        let input_mapper_fn = input_mapper.map(build_input_mapper);
        let condition_fn = condition.map(build_condition);
        Ok(Self {
            inner: Mutex::new(Some(blazen_pipeline::Stage {
                name,
                workflow: core_workflow,
                input_mapper: input_mapper_fn,
                condition: condition_fn,
            })),
        })
    }

    /// The stage's human-readable name.
    ///
    /// Returns an empty string if the stage has already been consumed by
    /// a `Pipeline`.
    #[wasm_bindgen(getter)]
    #[must_use]
    pub fn name(&self) -> String {
        let guard = self.inner.lock().expect("poisoned");
        guard.as_ref().map(|s| s.name.clone()).unwrap_or_default()
    }
}

impl WasmStage {
    /// Take the underlying [`blazen_pipeline::Stage`], consuming this
    /// `WasmStage` instance.
    ///
    /// Returns an error if the stage has already been consumed.
    pub(crate) fn take(&self) -> Result<blazen_pipeline::Stage, JsValue> {
        self.inner
            .lock()
            .expect("poisoned")
            .take()
            .ok_or_else(|| JsValue::from_str("Stage already consumed by a Pipeline"))
    }
}

// ---------------------------------------------------------------------------
// WasmParallelStage
// ---------------------------------------------------------------------------

/// A parallel pipeline stage that fans out across multiple branches.
///
/// Each branch is a [`WasmStage`]; branches execute concurrently and are
/// joined according to a [`WasmJoinStrategy`]. As with [`WasmStage`], the
/// branches are consumed when the parallel stage is constructed, so each
/// branch `Stage` instance can only be used once.
///
/// ```typescript
/// const a = new Stage("a", wfA);
/// const b = new Stage("b", wfB);
/// const fanOut = new ParallelStage("fan-out", [a, b], JoinStrategy.WaitAll);
/// ```
#[wasm_bindgen(js_name = "ParallelStage")]
pub struct WasmParallelStage {
    inner: Mutex<Option<blazen_pipeline::ParallelStage>>,
}

#[wasm_bindgen(js_class = "ParallelStage")]
impl WasmParallelStage {
    /// Create a new parallel stage from a list of branch [`WasmStage`]s.
    ///
    /// `branches` is a JS array of [`WasmStage`] instances. The Stage
    /// instances are moved into the parallel stage (wasm-bindgen
    /// transfers ownership at the ABI boundary), so the JS-side `Stage`
    /// objects become unusable after this call.
    ///
    /// `join_strategy` is optional and defaults to
    /// [`WasmJoinStrategy::WaitAll`].
    ///
    /// # Errors
    ///
    /// Returns a `JsValue` error if any of the supplied branches has
    /// already been consumed by a previous `Pipeline` or `ParallelStage`.
    #[wasm_bindgen(constructor)]
    pub fn new(
        name: String,
        branches: Vec<WasmStage>,
        join_strategy: Option<WasmJoinStrategy>,
    ) -> Result<WasmParallelStage, JsValue> {
        let mut rust_branches = Vec::with_capacity(branches.len());
        for branch in &branches {
            rust_branches.push(branch.take()?);
        }

        Ok(Self {
            inner: Mutex::new(Some(blazen_pipeline::ParallelStage {
                name,
                branches: rust_branches,
                join_strategy: join_strategy.unwrap_or(WasmJoinStrategy::WaitAll).into(),
            })),
        })
    }

    /// The parallel stage's human-readable name.
    ///
    /// Returns an empty string if the stage has already been consumed by
    /// a `Pipeline`.
    #[wasm_bindgen(getter)]
    #[must_use]
    pub fn name(&self) -> String {
        let guard = self.inner.lock().expect("poisoned");
        guard.as_ref().map(|s| s.name.clone()).unwrap_or_default()
    }
}

impl WasmParallelStage {
    /// Take the underlying [`blazen_pipeline::ParallelStage`], consuming
    /// this `WasmParallelStage` instance.
    ///
    /// Returns an error if the stage has already been consumed.
    pub(crate) fn take(&self) -> Result<blazen_pipeline::ParallelStage, JsValue> {
        self.inner
            .lock()
            .expect("poisoned")
            .take()
            .ok_or_else(|| JsValue::from_str("ParallelStage already consumed by a Pipeline"))
    }
}
