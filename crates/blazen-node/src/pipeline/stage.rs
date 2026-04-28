//! JavaScript bindings for pipeline stages.
//!
//! Exposes [`JsStage`] and [`JsParallelStage`] as NAPI classes wrapping
//! [`blazen_pipeline::Stage`] and [`blazen_pipeline::ParallelStage`].
//!
//! # Limitations (v1)
//!
//! `input_mapper` and `condition` support is **out of scope** for the
//! initial Node binding because [`blazen_pipeline`]'s
//! [`InputMapperFn`](blazen_pipeline::InputMapperFn) and
//! [`ConditionFn`](blazen_pipeline::ConditionFn) are synchronous Rust
//! closures. Bridging a JS callable into a sync Rust closure would
//! require blocking on a `ThreadsafeFunction` call inside a Tokio
//! worker, which is unsafe. Tracked in Phase 14: requires async-fn
//! support upstream in `blazen_pipeline`.

use std::sync::Mutex;

use napi::bindgen_prelude::*;
use napi_derive::napi;

// ---------------------------------------------------------------------------
// JoinStrategy
// ---------------------------------------------------------------------------

/// Strategy used by a [`JsParallelStage`] to join its concurrent
/// branches.
///
/// - `WaitAll`: wait for every branch to complete and aggregate the
///   results.
/// - `FirstCompletes`: return as soon as the first branch finishes;
///   the remaining branches are cancelled.
#[napi(string_enum, js_name = "JoinStrategy")]
#[derive(Debug, Clone, Copy)]
pub enum JsJoinStrategy {
    WaitAll,
    FirstCompletes,
}

impl From<JsJoinStrategy> for blazen_pipeline::JoinStrategy {
    fn from(j: JsJoinStrategy) -> Self {
        match j {
            JsJoinStrategy::WaitAll => Self::WaitAll,
            JsJoinStrategy::FirstCompletes => Self::FirstCompletes,
        }
    }
}

// ---------------------------------------------------------------------------
// JsStage
// ---------------------------------------------------------------------------

/// A single sequential pipeline stage.
///
/// Wraps a [`blazen_core::Workflow`] (built from a [`Workflow`] JS
/// instance) under a human-readable name. Stages are added to a
/// `Pipeline` builder which consumes them at build time.
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
/// `Stage already consumed by a Pipeline` error. The internal storage
/// uses [`Mutex<Option<...>>`](std::sync::Mutex) because napi-rs only
/// hands us `&self`/`&mut self`, but we need to *move* the underlying
/// [`blazen_pipeline::Stage`] out when the pipeline is built.
#[napi(js_name = "Stage")]
pub struct JsStage {
    inner: Mutex<Option<blazen_pipeline::Stage>>,
}

#[napi]
#[allow(
    clippy::must_use_candidate,
    clippy::missing_errors_doc,
    clippy::missing_panics_doc,
    clippy::needless_pass_by_value
)]
impl JsStage {
    /// Create a new sequential stage from a [`Workflow`].
    ///
    /// The workflow is materialized into its core
    /// [`blazen_core::Workflow`] form via
    /// [`JsWorkflow::build_workflow`](crate::workflow::workflow::JsWorkflow::build_workflow)
    /// at construction time, so any subsequent modifications to the JS
    /// `Workflow` instance will not affect this stage.
    #[napi(constructor)]
    pub fn new(name: String, workflow: &crate::workflow::workflow::JsWorkflow) -> Result<Self> {
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
    /// Returns an empty string if the stage has already been consumed
    /// by a `Pipeline`.
    #[napi(getter, js_name = "name")]
    pub fn js_name(&self) -> String {
        let guard = self.inner.lock().expect("poisoned");
        guard.as_ref().map(|s| s.name.clone()).unwrap_or_default()
    }
}

impl JsStage {
    /// Take the underlying [`blazen_pipeline::Stage`], consuming this
    /// `JsStage` instance.
    ///
    /// Returns an error if the stage has already been consumed.
    pub(crate) fn take(&self) -> napi::Result<blazen_pipeline::Stage> {
        self.inner.lock().expect("poisoned").take().ok_or_else(|| {
            napi::Error::new(
                napi::Status::GenericFailure,
                "Stage already consumed by a Pipeline",
            )
        })
    }
}

// ---------------------------------------------------------------------------
// JsParallelStage
// ---------------------------------------------------------------------------

/// A parallel pipeline stage that fans out across multiple branches.
///
/// Each branch is a [`JsStage`]; branches execute concurrently and are
/// joined according to a [`JsJoinStrategy`]. As with [`JsStage`], the
/// branches are consumed when the parallel stage is constructed, so
/// each branch `Stage` instance can only be used once.
///
/// ```typescript
/// const a = new Stage("a", wfA);
/// const b = new Stage("b", wfB);
/// const fanOut = new ParallelStage("fan-out", [a, b], JoinStrategy.WaitAll);
/// ```
#[napi(js_name = "ParallelStage")]
pub struct JsParallelStage {
    inner: Mutex<Option<blazen_pipeline::ParallelStage>>,
}

#[napi]
#[allow(
    clippy::must_use_candidate,
    clippy::missing_errors_doc,
    clippy::missing_panics_doc,
    clippy::needless_pass_by_value
)]
impl JsParallelStage {
    /// Create a new parallel stage from a list of branch [`JsStage`]s.
    ///
    /// Each branch stage is consumed (its underlying
    /// [`blazen_pipeline::Stage`] is moved out) at construction time,
    /// so the same `Stage` instance cannot be used in two
    /// `ParallelStage`s.
    ///
    /// `joinStrategy` defaults to [`JsJoinStrategy::WaitAll`].
    #[napi(constructor)]
    pub fn new(
        name: String,
        branches: Vec<&JsStage>,
        join_strategy: Option<JsJoinStrategy>,
    ) -> Result<Self> {
        let mut rust_branches = Vec::with_capacity(branches.len());
        for b in branches {
            rust_branches.push(b.take()?);
        }
        Ok(Self {
            inner: Mutex::new(Some(blazen_pipeline::ParallelStage {
                name,
                branches: rust_branches,
                join_strategy: join_strategy.unwrap_or(JsJoinStrategy::WaitAll).into(),
            })),
        })
    }

    /// The parallel stage's human-readable name.
    ///
    /// Returns an empty string if the stage has already been consumed
    /// by a `Pipeline`.
    #[napi(getter, js_name = "name")]
    pub fn js_name(&self) -> String {
        let guard = self.inner.lock().expect("poisoned");
        guard.as_ref().map(|s| s.name.clone()).unwrap_or_default()
    }
}

impl JsParallelStage {
    /// Take the underlying [`blazen_pipeline::ParallelStage`],
    /// consuming this `JsParallelStage` instance.
    ///
    /// Returns an error if the stage has already been consumed.
    pub(crate) fn take(&self) -> napi::Result<blazen_pipeline::ParallelStage> {
        self.inner.lock().expect("poisoned").take().ok_or_else(|| {
            napi::Error::new(
                napi::Status::GenericFailure,
                "ParallelStage already consumed by a Pipeline",
            )
        })
    }
}
