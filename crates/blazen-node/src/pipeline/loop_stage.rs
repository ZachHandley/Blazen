//! JavaScript bindings for pipeline loop stages.
//!
//! Exposes [`JsLoopStage`] and [`JsLoopDecision`] as NAPI surfaces wrapping
//! [`blazen_pipeline::LoopStage`] and [`blazen_pipeline::LoopDecision`].
//!
//! # Limitations (v1)
//!
//! `blazen_pipeline`'s loop `until` predicate
//! ([`LoopUntilFn`](blazen_pipeline::LoopUntilFn)) and per-round hook
//! ([`RoundCompleteFn`](blazen_pipeline::RoundCompleteFn)) are *synchronous*
//! Rust closures. Bridging a JS callable into a sync Rust closure would
//! require blocking on a `ThreadsafeFunction` call inside a Tokio worker,
//! which is unsafe — the same constraint that keeps `input_mapper` /
//! `condition` off [`JsStage`](crate::pipeline::stage::JsStage). For v1 the
//! loop therefore runs its inner stage exactly `maxIterations` times: the
//! installed `until` predicate returns [`JsLoopDecision::Continue`] until the
//! hard cap is hit, at which point the engine stops the loop regardless.
//! Exposing the [`JsLoopDecision`] enum lets callers reason about the loop
//! outcome and keeps the binding at parity with the typed Rust surface.

use std::sync::Arc;
use std::sync::Mutex;

use napi::bindgen_prelude::*;
use napi_derive::napi;

use blazen_pipeline::{LoopDecision, LoopStage, StageKind};

use crate::pipeline::stage::{JsParallelStage, JsStage};

// ---------------------------------------------------------------------------
// JsLoopDecision
// ---------------------------------------------------------------------------

/// The decision returned by a loop stage's `until` predicate after each round.
///
/// - `Continue`: run the inner stage again (subject to the `maxIterations`
///   cap).
/// - `Done`: stop looping cleanly; the loop stage succeeds.
/// - `Abort`: stop looping with an error.
#[napi(string_enum, js_name = "LoopDecision")]
#[derive(Debug, Clone, Copy)]
pub enum JsLoopDecision {
    Continue,
    Done,
    Abort,
}

impl From<JsLoopDecision> for LoopDecision {
    fn from(d: JsLoopDecision) -> Self {
        match d {
            JsLoopDecision::Continue => LoopDecision::Continue,
            JsLoopDecision::Done => LoopDecision::Done,
            JsLoopDecision::Abort => LoopDecision::Abort(String::new()),
        }
    }
}

// ---------------------------------------------------------------------------
// JsLoopStage
// ---------------------------------------------------------------------------

/// A pipeline stage that re-runs an inner stage until a hard iteration cap is
/// reached.
///
/// The inner stage is a [`JsStage`] (sequential) or [`JsParallelStage`]
/// (parallel); it is consumed at construction time, so the same `Stage` /
/// `ParallelStage` instance cannot be reused. As noted in the module docs,
/// the v1 loop runs the inner stage exactly `maxIterations` times.
///
/// ```typescript
/// const inner = new Stage("refine", wf);
/// const loop = new LoopStage("refine-loop", 3, inner);
/// ```
#[napi(js_name = "LoopStage")]
pub struct JsLoopStage {
    inner: Mutex<Option<LoopStage>>,
}

#[napi]
#[allow(
    clippy::must_use_candidate,
    clippy::missing_errors_doc,
    clippy::missing_panics_doc,
    clippy::needless_pass_by_value
)]
impl JsLoopStage {
    /// Create a loop stage from a sequential [`JsStage`].
    ///
    /// `maxIterations` is the hard cap on the number of rounds. The inner
    /// stage is consumed at construction time.
    #[napi(constructor)]
    pub fn new(name: String, max_iterations: u32, inner: &JsStage) -> Result<Self> {
        let core_inner = StageKind::Sequential(inner.take()?);
        Ok(Self {
            inner: Mutex::new(Some(Self::build_loop(name, max_iterations, core_inner))),
        })
    }

    /// Create a loop stage whose inner body is a parallel fan-out stage.
    ///
    /// `maxIterations` is the hard cap on the number of rounds. The inner
    /// parallel stage is consumed at construction time.
    #[napi(factory, js_name = "fromParallel")]
    pub fn from_parallel(
        name: String,
        max_iterations: u32,
        inner: &JsParallelStage,
    ) -> Result<Self> {
        let core_inner = StageKind::Parallel(inner.take()?);
        Ok(Self {
            inner: Mutex::new(Some(Self::build_loop(name, max_iterations, core_inner))),
        })
    }

    /// The loop stage's human-readable name.
    ///
    /// Returns an empty string if the stage has already been consumed by a
    /// `Pipeline`.
    #[napi(getter, js_name = "name")]
    pub fn js_name(&self) -> String {
        let guard = self.inner.lock().expect("poisoned");
        guard.as_ref().map(|s| s.name.clone()).unwrap_or_default()
    }

    /// The hard iteration cap.
    ///
    /// Returns `0` if the stage has already been consumed by a `Pipeline`.
    #[napi(getter, js_name = "maxIterations")]
    pub fn js_max_iterations(&self) -> u32 {
        let guard = self.inner.lock().expect("poisoned");
        guard.as_ref().map_or(0, |s| s.max_iterations)
    }
}

impl JsLoopStage {
    /// Build a [`LoopStage`] whose `until` predicate runs the inner stage up
    /// to `max_iterations` times (returning [`LoopDecision::Continue`] until
    /// the engine's hard cap stops it).
    fn build_loop(name: String, max_iterations: u32, inner: StageKind) -> LoopStage {
        LoopStage {
            name,
            max_iterations,
            inner: Box::new(inner),
            until: Arc::new(|_state, _round| LoopDecision::Continue),
            on_round_complete: None,
        }
    }

    /// Take the underlying [`blazen_pipeline::LoopStage`], consuming this
    /// `JsLoopStage` instance.
    ///
    /// Returns an error if the stage has already been consumed.
    pub(crate) fn take(&self) -> napi::Result<LoopStage> {
        self.inner.lock().expect("poisoned").take().ok_or_else(|| {
            napi::Error::new(
                napi::Status::GenericFailure,
                "LoopStage already consumed by a Pipeline",
            )
        })
    }
}
