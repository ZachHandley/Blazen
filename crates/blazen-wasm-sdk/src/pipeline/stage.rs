//! WASM bindings for pipeline stages.
//!
//! Exposes [`WasmStage`] and [`WasmParallelStage`] as `wasm_bindgen` classes
//! wrapping [`blazen_pipeline::Stage`] and [`blazen_pipeline::ParallelStage`].
//!
//! # v1 Limitations
//!
//! `input_mapper` and `condition` are **not exposed** in this version. The
//! upstream [`InputMapperFn`](blazen_pipeline::InputMapperFn) and
//! [`ConditionFn`](blazen_pipeline::ConditionFn) are synchronous Rust
//! closures, and bridging a JS callback into a sync closure on wasm32
//! would require either blocking on a JS Promise (impossible on wasm32 —
//! the runtime is single-threaded) or extending `blazen_pipeline` with
//! async-fn support upstream. Until that lands, every stage runs
//! unconditionally and its workflow receives the previous stage's output
//! verbatim.

use std::sync::Mutex;

use wasm_bindgen::prelude::*;

use crate::workflow::WasmWorkflow;

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
    /// # Errors
    ///
    /// Returns a `JsValue` error if the workflow has already been
    /// consumed (e.g. `run()` was called on it earlier) or if the
    /// underlying `WorkflowBuilder::build()` fails.
    #[wasm_bindgen(constructor)]
    pub fn new(name: String, workflow: &mut WasmWorkflow) -> Result<WasmStage, JsValue> {
        let core_workflow = workflow.build_workflow()?;
        Ok(Self {
            inner: Mutex::new(Some(blazen_pipeline::Stage {
                name,
                workflow: core_workflow,
                input_mapper: None,
                condition: None,
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
