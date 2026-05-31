//! [`Pipeline`] -- a validated, runnable pipeline over the real
//! [`blazen_pipeline::Pipeline`] engine.

use std::sync::Arc;

use crate::errors::{BlazenError, BlazenResult};
use crate::pipeline::handler::PipelineHandler;
use crate::runtime::runtime;
use crate::workflow::{Event, WorkflowResult};

use blazen_pipeline::{Pipeline as CorePipeline, PipelineResult};

/// A validated, runnable pipeline.
///
/// Wraps the real [`blazen_pipeline::Pipeline`]. Multiple runs are allowed —
/// invoking [`run`](Self::run) twice in a row is safe and produces
/// independent runs — but the implementation rejects **overlapping** runs on
/// the same handle to avoid surprising aliasing of inner workflow state across
/// concurrent foreign callers.
#[derive(uniffi::Object)]
pub struct Pipeline {
    inner: CorePipeline,
    /// Stage names in registration order, mirrored from the builder.
    stage_names: Vec<String>,
    running: parking_lot::Mutex<bool>,
}

impl Pipeline {
    /// Wrap a built core pipeline plus its mirrored stage-name list.
    pub(crate) fn new(inner: CorePipeline, stage_names: Vec<String>) -> Arc<Self> {
        Arc::new(Self {
            inner,
            stage_names,
            running: parking_lot::Mutex::new(false),
        })
    }

    fn lock_running(self: &Arc<Self>) -> BlazenResult<RunGuard> {
        let mut guard = self.running.lock();
        if *guard {
            return Err(BlazenError::Validation {
                message: "Pipeline.run is already in flight on this handle; await it before starting another".into(),
            });
        }
        *guard = true;
        Ok(RunGuard {
            pipeline: Arc::clone(self),
        })
    }
}

/// Drop-guard that clears the `running` flag when a pipeline run ends
/// (including via panic or cancellation), so the same `Pipeline` handle can be
/// re-used for a fresh run afterwards.
struct RunGuard {
    pipeline: Arc<Pipeline>,
}

impl Drop for RunGuard {
    fn drop(&mut self) {
        *self.pipeline.running.lock() = false;
    }
}

#[uniffi::export(async_runtime = "tokio")]
impl Pipeline {
    /// Execute the pipeline to completion. `input_json` is parsed as JSON and
    /// passed as the first stage's `StartEvent` payload; each subsequent stage
    /// receives the previous stage's output.
    ///
    /// Returns a [`WorkflowResult`] whose `event` field is a synthetic
    /// `StopEvent` carrying the final stage output, and whose `total_*_tokens`
    /// / `total_cost_usd` fields are the engine's aggregated totals across
    /// every stage.
    ///
    /// This is the result-only shorthand (parity with `Workflow::run`). For
    /// streaming intermediate events, pausing, snapshotting, or human-in-the-
    /// loop input, use [`start`](Self::start) and drive the returned
    /// [`PipelineHandler`].
    pub async fn run(self: Arc<Self>, input_json: String) -> BlazenResult<WorkflowResult> {
        let _guard = self.lock_running()?;
        let input: serde_json::Value = serde_json::from_str(&input_json)?;
        let result = self
            .inner
            .clone()
            .run(input)
            .await
            .map_err(BlazenError::from)?;
        Ok(pipeline_result_to_wire(&result))
    }

    /// Run the pipeline and return a live [`PipelineHandler`] instead of
    /// blocking for the final result.
    ///
    /// The returned handler exposes the full control surface — stream
    /// intermediate events to a foreign
    /// [`PipelineEventSink`](crate::pipeline::PipelineEventSink), `pause` /
    /// `resume_in_place`, `snapshot`, `respond_to_input` for human-in-the-loop,
    /// `abort`, `progress`, and running `usage_total` / `cost_total_usd` —
    /// plus `result()` to await the terminal [`WorkflowResult`]. Mirrors
    /// `Workflow::run_with_handler`.
    pub async fn start(self: Arc<Self>, input_json: String) -> BlazenResult<Arc<PipelineHandler>> {
        let input: serde_json::Value = serde_json::from_str(&input_json)?;
        let handler = self.inner.clone().start(input);
        Ok(PipelineHandler::new(handler))
    }
}

#[uniffi::export]
impl Pipeline {
    /// Synchronous variant of [`run`](Self::run) — blocks the current thread
    /// on the shared Tokio runtime. Provided for callers that want
    /// fire-and-forget usage without engaging their host language's async
    /// machinery (Ruby scripts, simple Go `main` functions).
    pub fn run_blocking(self: Arc<Self>, input_json: String) -> BlazenResult<WorkflowResult> {
        let this = Arc::clone(&self);
        runtime().block_on(async move { this.run(input_json).await })
    }

    /// Synchronous variant of [`start`](Self::start) — blocks the current
    /// thread on the shared Tokio runtime while the pipeline is launched, then
    /// returns the live handler. The pipeline keeps running on the shared
    /// runtime after this returns.
    pub fn start_blocking(
        self: Arc<Self>,
        input_json: String,
    ) -> BlazenResult<Arc<PipelineHandler>> {
        let this = Arc::clone(&self);
        runtime().block_on(async move { this.start(input_json).await })
    }

    /// Stage names in registration order — useful for foreign-side
    /// introspection / debug logging without re-running the pipeline.
    #[must_use]
    pub fn stage_names(self: Arc<Self>) -> Vec<String> {
        self.stage_names.clone()
    }
}

/// Convert a [`PipelineResult`] into the wire-format [`WorkflowResult`] the
/// uniffi surface uses for both pipelines and workflows. The terminal event is
/// a synthetic `StopEvent` carrying the pipeline's final output.
pub(crate) fn pipeline_result_to_wire(result: &PipelineResult) -> WorkflowResult {
    WorkflowResult {
        event: Event {
            event_type: "StopEvent".into(),
            data_json: result.final_output.to_string(),
        },
        total_input_tokens: u64::from(result.usage_total.prompt_tokens),
        total_output_tokens: u64::from(result.usage_total.completion_tokens),
        total_cost_usd: result.cost_total_usd,
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use crate::errors::BlazenResult;
    use crate::pipeline::PipelineBuilder;
    use crate::workflow::{Event, StepHandler, StepOutput, Workflow, WorkflowBuilder};

    /// A trivial Rust-side [`StepHandler`] that increments a `value` field on
    /// the incoming JSON payload and emits it as a `StopEvent`. The
    /// foreign-export trait can still be implemented directly in Rust for
    /// tests, exercising the same dispatch path foreign handlers use.
    struct IncrementHandler;

    #[async_trait::async_trait]
    impl StepHandler for IncrementHandler {
        async fn invoke(&self, event: Event) -> BlazenResult<StepOutput> {
            // A `StartEvent` serializes as `{"data": <payload>}`, so the
            // step's payload lives under `.data`. Fall back to the top level
            // for already-unwrapped shapes.
            let parsed: serde_json::Value =
                serde_json::from_str(&event.data_json).unwrap_or(serde_json::Value::Null);
            let payload = parsed.get("data").cloned().unwrap_or(parsed);
            let cur = payload
                .get("value")
                .and_then(serde_json::Value::as_i64)
                .unwrap_or(0);
            Ok(StepOutput::Single {
                event: Event {
                    event_type: "StopEvent".into(),
                    data_json: serde_json::json!({ "value": cur + 1 }).to_string(),
                },
            })
        }
    }

    fn increment_workflow(name: &str) -> Arc<Workflow> {
        WorkflowBuilder::new(name.to_string())
            .step(
                "inc".into(),
                vec!["blazen::StartEvent".into()],
                vec!["StopEvent".into()],
                Arc::new(IncrementHandler),
            )
            .expect("step")
            .build()
            .expect("build workflow")
    }

    /// The migrated uniffi pipeline runs on the real `blazen_pipeline` engine:
    /// two sequential increment stages turn `value: 0` into `value: 2`.
    #[tokio::test]
    async fn uniffi_pipeline_uses_real_engine_run() {
        let pipeline = PipelineBuilder::new("inc-pipe".into())
            .add_workflow(increment_workflow("a"))
            .expect("add a")
            .add_workflow(increment_workflow("b"))
            .expect("add b")
            .build()
            .expect("build pipeline");

        assert_eq!(pipeline.clone().stage_names(), vec!["stage-0", "stage-1"]);

        let result = pipeline
            .run("{\"value\": 0}".into())
            .await
            .expect("run pipeline");
        let out: serde_json::Value =
            serde_json::from_str(&result.event.data_json).expect("final output json");
        assert_eq!(out["value"], serde_json::json!(2));
    }

    /// `start()` returns a live handler exposing progress / snapshot / result.
    #[tokio::test]
    async fn uniffi_pipeline_start_returns_handler_with_progress_and_result() {
        let pipeline = PipelineBuilder::new("inc-pipe".into())
            .stage("first".into(), increment_workflow("a"))
            .expect("add first")
            .stage("second".into(), increment_workflow("b"))
            .expect("add second")
            .build()
            .expect("build pipeline");

        assert_eq!(pipeline.clone().stage_names(), vec!["first", "second"]);

        let handler = pipeline
            .start("{\"value\": 10}".into())
            .await
            .expect("start");

        // progress() is callable on the live handle and reports the stage cap.
        let progress = handler.clone().progress().await.expect("progress present");
        assert_eq!(progress.total_stages, 2);

        let result = handler.result().await.expect("result");
        let out: serde_json::Value =
            serde_json::from_str(&result.event.data_json).expect("final output json");
        assert_eq!(out["value"], serde_json::json!(12));
    }

    /// A parallel stage fans two branches out under `wait_all`, collecting both
    /// branch outputs keyed by branch name into the final output.
    #[tokio::test]
    async fn uniffi_pipeline_parallel_wait_all() {
        let pipeline = PipelineBuilder::new("fan".into())
            .parallel(
                "fan-out".into(),
                vec!["left".into(), "right".into()],
                vec![increment_workflow("l"), increment_workflow("r")],
                true,
            )
            .expect("add parallel")
            .build()
            .expect("build pipeline");

        let result = pipeline.run("{\"value\": 5}".into()).await.expect("run");
        let out: serde_json::Value =
            serde_json::from_str(&result.event.data_json).expect("final output json");
        assert_eq!(out["left"]["value"], serde_json::json!(6));
        assert_eq!(out["right"]["value"], serde_json::json!(6));
    }

    /// Mismatched branch/workflow counts are rejected at builder time.
    #[test]
    fn uniffi_pipeline_parallel_length_mismatch_errors() {
        let res = PipelineBuilder::new("bad".into()).parallel(
            "p".into(),
            vec!["only-one".into()],
            vec![increment_workflow("a"), increment_workflow("b")],
            true,
        );
        assert!(matches!(
            res,
            Err(crate::errors::BlazenError::Validation { .. })
        ));
    }

    /// An empty pipeline fails validation at build time (engine-enforced).
    #[test]
    fn uniffi_pipeline_empty_errors() {
        let res = PipelineBuilder::new("empty".into()).build();
        assert!(matches!(
            res,
            Err(crate::errors::BlazenError::Validation { .. })
        ));
    }
}
