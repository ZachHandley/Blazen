//! Standalone JS-visible wrappers for [`blazen_core::SubWorkflowStep`] and
//! [`blazen_core::ParallelSubWorkflowsStep`].
//!
//! Each wrapper builds its inner child [`blazen_core::Workflow`] eagerly at
//! construction time (the same constraint the napi binding has — the child
//! workflow's JS step closures are baked in before the parent dispatches).
//! Once constructed, an instance can be reused across multiple parent
//! workflows: pass it to
//! [`WasmWorkflow::add_subworkflow_step`](crate::workflow::WasmWorkflow::add_subworkflow_step)
//! or
//! [`WasmWorkflow::add_parallel_subworkflows`](crate::workflow::WasmWorkflow::add_parallel_subworkflows).

use std::sync::Arc;
use std::time::Duration;

use wasm_bindgen::prelude::*;

use blazen_core::{
    JoinStrategy as CoreJoinStrategy, ParallelSubWorkflowsStep, SubWorkflowStep, Workflow,
};
use blazen_events::intern_event_type;

/// JS-visible wrapper around [`blazen_core::SubWorkflowStep`].
///
/// Mirrors the Node binding's `SubWorkflowStep` class. Construct from a
/// pre-built child [`crate::workflow::WasmWorkflow`] plus routing metadata,
/// then pass to a parent workflow via
/// [`WasmWorkflow::add_subworkflow_step`](crate::workflow::WasmWorkflow::add_subworkflow_step).
///
/// ```js
/// const child = new Workflow("enrich");
/// child.addStep("enrich", ["blazen::StartEvent"], async (ev) => ({
///   type: "blazen::StopEvent",
///   result: { ok: true },
/// }));
/// const step = new SubWorkflowStep(
///   "enrich",
///   ["blazen::StartEvent"],
///   ["enrich::output"],
///   child,
/// );
/// ```
#[wasm_bindgen(js_name = "SubWorkflowStep")]
pub struct WasmSubWorkflowStep {
    pub(crate) name: String,
    pub(crate) accepts: Vec<String>,
    pub(crate) emits: Vec<String>,
    pub(crate) inner_workflow: Arc<Workflow>,
    pub(crate) timeout: Option<Duration>,
}

// SAFETY: WASM is single-threaded.
unsafe impl Send for WasmSubWorkflowStep {}
unsafe impl Sync for WasmSubWorkflowStep {}

#[wasm_bindgen(js_class = "SubWorkflowStep")]
#[allow(clippy::must_use_candidate)]
impl WasmSubWorkflowStep {
    /// Create a sub-workflow step.
    ///
    /// `inner` is consumed at construction: its builder is finalised so the
    /// child workflow can be reused across multiple parent dispatches. Pass
    /// an optional `timeoutSecs` (positive number) to cap the wall-clock
    /// time for each child run.
    #[wasm_bindgen(constructor)]
    pub fn new(
        name: String,
        accepts: Vec<String>,
        emits: Vec<String>,
        inner: &mut crate::workflow::WasmWorkflow,
        timeout_secs: Option<f64>,
    ) -> Result<WasmSubWorkflowStep, JsValue> {
        let inner_workflow = inner.build_workflow()?;
        Ok(Self {
            name,
            accepts,
            emits,
            inner_workflow: Arc::new(inner_workflow),
            timeout: timeout_secs.and_then(|s| {
                if s > 0.0 {
                    Some(Duration::from_secs_f64(s))
                } else {
                    None
                }
            }),
        })
    }

    /// The step name.
    #[wasm_bindgen(getter)]
    pub fn name(&self) -> String {
        self.name.clone()
    }

    /// Event type identifiers this step accepts.
    #[wasm_bindgen(getter)]
    pub fn accepts(&self) -> Vec<String> {
        self.accepts.clone()
    }

    /// Event type identifiers this step may emit.
    #[wasm_bindgen(getter)]
    pub fn emits(&self) -> Vec<String> {
        self.emits.clone()
    }
}

impl WasmSubWorkflowStep {
    /// Materialize a [`SubWorkflowStep`] for handing to a `WorkflowBuilder`.
    pub(crate) fn to_core(&self) -> SubWorkflowStep {
        let accepts: Vec<&'static str> = self
            .accepts
            .iter()
            .map(|s| intern_event_type(s))
            .collect();
        let emits: Vec<&'static str> = self
            .emits
            .iter()
            .map(|s| intern_event_type(s))
            .collect();
        let mut step = SubWorkflowStep::with_json_mappers(
            self.name.clone(),
            accepts,
            emits,
            Arc::clone(&self.inner_workflow),
        );
        if let Some(t) = self.timeout {
            step = step.with_timeout(t);
        }
        step
    }
}

/// JS-visible wrapper around [`blazen_core::ParallelSubWorkflowsStep`].
///
/// Mirrors the Node binding's `ParallelSubWorkflowsStep` class. Pass an
/// array of pre-built [`WasmSubWorkflowStep`] branches plus the join
/// strategy.
///
/// ```js
/// const fanout = new ParallelSubWorkflowsStep(
///   "enrich-fanout",
///   ["blazen::StartEvent"],
///   ["enrich-fanout::output"],
///   [stepA, stepB, stepC],
///   "WaitAll",
/// );
/// ```
#[wasm_bindgen(js_name = "ParallelSubWorkflowsStep")]
pub struct WasmParallelSubWorkflowsStep {
    pub(crate) name: String,
    pub(crate) accepts: Vec<String>,
    pub(crate) emits: Vec<String>,
    pub(crate) branches: Vec<BranchSpec>,
    pub(crate) join_strategy: CoreJoinStrategy,
}

/// Internal-only branch snapshot. We don't keep `&WasmSubWorkflowStep`
/// references across the JS↔Rust boundary — wasm-bindgen would refuse the
/// `Vec<&...>` argument type — so we eagerly clone the bits we need.
#[derive(Clone)]
pub(crate) struct BranchSpec {
    pub name: String,
    pub accepts: Vec<String>,
    pub emits: Vec<String>,
    pub inner_workflow: Arc<Workflow>,
    pub timeout: Option<Duration>,
}

// SAFETY: WASM is single-threaded.
unsafe impl Send for WasmParallelSubWorkflowsStep {}
unsafe impl Sync for WasmParallelSubWorkflowsStep {}

#[wasm_bindgen(js_class = "ParallelSubWorkflowsStep")]
#[allow(clippy::must_use_candidate)]
impl WasmParallelSubWorkflowsStep {
    /// Create a parallel sub-workflow fan-out step.
    ///
    /// `branches` is an array of pre-constructed [`WasmSubWorkflowStep`]
    /// instances. `joinStrategy` accepts the case-insensitive strings
    /// `"WaitAll"` (default) or `"FirstCompletes"`.
    #[wasm_bindgen(constructor)]
    pub fn new(
        name: String,
        accepts: Vec<String>,
        emits: Vec<String>,
        branches: Vec<WasmSubWorkflowStep>,
        join_strategy: Option<String>,
    ) -> Result<WasmParallelSubWorkflowsStep, JsValue> {
        let branch_specs: Vec<BranchSpec> = branches
            .into_iter()
            .map(|b| BranchSpec {
                name: b.name,
                accepts: b.accepts,
                emits: b.emits,
                inner_workflow: b.inner_workflow,
                timeout: b.timeout,
            })
            .collect();
        let join = match join_strategy.as_deref() {
            Some("FirstCompletes" | "firstCompletes" | "first_completes") => {
                CoreJoinStrategy::FirstCompletes
            }
            Some("WaitAll" | "waitAll" | "wait_all") | None => CoreJoinStrategy::WaitAll,
            Some(other) => {
                return Err(JsValue::from_str(&format!(
                    "ParallelSubWorkflowsStep: unknown joinStrategy '{other}' \
                     (expected 'WaitAll' or 'FirstCompletes')"
                )));
            }
        };
        Ok(Self {
            name,
            accepts,
            emits,
            branches: branch_specs,
            join_strategy: join,
        })
    }

    /// The step name.
    #[wasm_bindgen(getter)]
    pub fn name(&self) -> String {
        self.name.clone()
    }

    /// Event type identifiers this step accepts.
    #[wasm_bindgen(getter)]
    pub fn accepts(&self) -> Vec<String> {
        self.accepts.clone()
    }

    /// Event type identifiers this step may emit.
    #[wasm_bindgen(getter)]
    pub fn emits(&self) -> Vec<String> {
        self.emits.clone()
    }
}

impl WasmParallelSubWorkflowsStep {
    /// Materialize a [`ParallelSubWorkflowsStep`] for handing to a builder.
    pub(crate) fn to_core(&self) -> ParallelSubWorkflowsStep {
        let accepts: Vec<&'static str> = self
            .accepts
            .iter()
            .map(|s| intern_event_type(s))
            .collect();
        let emits: Vec<&'static str> = self
            .emits
            .iter()
            .map(|s| intern_event_type(s))
            .collect();
        let branches: Vec<SubWorkflowStep> = self
            .branches
            .iter()
            .map(|b| {
                let b_accepts: Vec<&'static str> =
                    b.accepts.iter().map(|s| intern_event_type(s)).collect();
                let b_emits: Vec<&'static str> =
                    b.emits.iter().map(|s| intern_event_type(s)).collect();
                let mut step = SubWorkflowStep::with_json_mappers(
                    b.name.clone(),
                    b_accepts,
                    b_emits,
                    Arc::clone(&b.inner_workflow),
                );
                if let Some(t) = b.timeout {
                    step = step.with_timeout(t);
                }
                step
            })
            .collect();
        ParallelSubWorkflowsStep {
            name: self.name.clone(),
            accepts,
            emits,
            branches,
            join_strategy: self.join_strategy,
        }
    }
}
