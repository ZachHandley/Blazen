//! JavaScript workflow builder and runner.
//!
//! Provides [`JsWorkflow`] which lets TypeScript/JavaScript users define
//! workflows with step handlers as async functions.

use std::sync::Arc;
use std::time::Duration;

use napi::bindgen_prelude::*;
use napi::threadsafe_function::{ThreadsafeFunction, ThreadsafeFunctionCallMode};
use napi_derive::napi;
use tokio_stream::StreamExt;
use zagents_events::{AnyEvent, intern_event_type};

use crate::context::JsContext;
use crate::error::workflow_error_to_napi;
use crate::event::{any_event_to_js_value, js_value_to_any_event};

// ---------------------------------------------------------------------------
// Type aliases for ThreadsafeFunction variants
// ---------------------------------------------------------------------------

/// Step handler: takes (event, ctx) and returns a `serde_json::Value`.
type StepHandlerTsfn = ThreadsafeFunction<(serde_json::Value, JsContext), serde_json::Value>;

/// Stream callback: takes a `serde_json::Value`, returns nothing meaningful.
/// We use the default Unknown return type and fire-and-forget via `call`.
type StreamCallbackTsfn = ThreadsafeFunction<serde_json::Value>;

// ---------------------------------------------------------------------------
// Step registration data
// ---------------------------------------------------------------------------

/// Internal representation of a step before building the workflow.
///
/// The handler is wrapped in `Arc` because `ThreadsafeFunction` does not
/// implement `Clone`, but the workflow engine needs to clone step handlers
/// for concurrent dispatch.
struct JsStepRegistration {
    name: String,
    event_types: Vec<String>,
    handler: Arc<StepHandlerTsfn>,
}

// ---------------------------------------------------------------------------
// JsWorkflowResult
// ---------------------------------------------------------------------------

/// The result of a workflow run.
#[napi(object)]
pub struct JsWorkflowResult {
    /// The event type of the final result (typically "`zagents::StopEvent`").
    #[napi(js_name = "type")]
    pub event_type: String,
    /// The result data as a JSON object.
    pub data: serde_json::Value,
}

// ---------------------------------------------------------------------------
// JsWorkflow
// ---------------------------------------------------------------------------

/// A workflow builder and runner.
///
/// Create a workflow, add steps with async handler functions, then run it.
///
/// ```javascript
/// const workflow = new Workflow("my-workflow");
///
/// workflow.addStep("analyze", ["zagents::StartEvent"], async (event, ctx) => {
///   const text = event.message;
///   return { type: "zagents::StopEvent", result: { analyzed: text } };
/// });
///
/// const result = await workflow.run({ message: "hello" });
/// ```
#[napi(js_name = "Workflow")]
pub struct JsWorkflow {
    name: String,
    steps: Vec<JsStepRegistration>,
    timeout_secs: Option<f64>,
}

#[napi]
#[allow(clippy::must_use_candidate, clippy::missing_errors_doc)]
impl JsWorkflow {
    /// Create a new workflow with the given name.
    #[napi(constructor)]
    pub fn new(name: String) -> Self {
        Self {
            name,
            steps: Vec::new(),
            timeout_secs: Some(300.0), // 5 min default
        }
    }

    /// Add a step to the workflow.
    ///
    /// - `name`: Human-readable step name.
    /// - `eventTypes`: Array of event type strings this step handles.
    /// - `handler`: Async function `(event, ctx) => Event` that processes
    ///   events and returns the next event.
    #[napi(js_name = "addStep")]
    pub fn add_step(
        &mut self,
        name: String,
        event_types: Vec<String>,
        handler: StepHandlerTsfn,
    ) -> Result<()> {
        self.steps.push(JsStepRegistration {
            name,
            event_types,
            handler: Arc::new(handler),
        });
        Ok(())
    }

    /// Set the workflow timeout in seconds.
    ///
    /// Set to 0 or negative to disable the timeout.
    #[napi(js_name = "setTimeout")]
    pub fn set_timeout(&mut self, seconds: f64) {
        if seconds <= 0.0 {
            self.timeout_secs = None;
        } else {
            self.timeout_secs = Some(seconds);
        }
    }

    /// Run the workflow with the given input data.
    ///
    /// The input is wrapped in a `StartEvent` automatically.
    /// Returns the final result when the workflow completes via a `StopEvent`.
    #[napi]
    pub async fn run(&self, input: serde_json::Value) -> Result<JsWorkflowResult> {
        let workflow = self.build_workflow()?;

        let handler = workflow.run(input).await.map_err(workflow_error_to_napi)?;

        let result = handler.result().await.map_err(workflow_error_to_napi)?;

        Ok(make_result(&*result))
    }

    /// Run the workflow with streaming.
    ///
    /// The `onEvent` callback receives intermediate events published via
    /// `ctx.writeEventToStream()` from within step handlers.
    ///
    /// Returns the final result when the workflow completes.
    #[napi(js_name = "runStreaming")]
    pub async fn run_streaming(
        &self,
        input: serde_json::Value,
        on_event: StreamCallbackTsfn,
    ) -> Result<JsWorkflowResult> {
        let workflow = self.build_workflow()?;

        let handler = workflow.run(input).await.map_err(workflow_error_to_napi)?;

        // Subscribe to the stream before awaiting the result.
        let mut stream = handler.stream_events();

        // Spawn a task to forward stream events to the JS callback.
        // We use `call` with `NonBlocking` mode (fire-and-forget) because:
        // 1. We don't need the return value from the stream callback.
        // 2. `call_async` returns a future that is not Send-safe.
        let on_event = Arc::new(on_event);
        let on_event_clone = Arc::clone(&on_event);
        let stream_handle = tokio::spawn(async move {
            while let Some(event) = stream.next().await {
                let js_event = any_event_to_js_value(&*event);
                // Fire-and-forget: call the JS callback without awaiting.
                let _ = on_event_clone.call(Ok(js_event), ThreadsafeFunctionCallMode::NonBlocking);
            }
        });

        let result = handler.result().await.map_err(workflow_error_to_napi)?;

        // Wait for the stream consumer to finish.
        let _ = stream_handle.await;

        Ok(make_result(&*result))
    }
}

impl JsWorkflow {
    /// Build the internal `Workflow` from the registered steps.
    fn build_workflow(&self) -> Result<zagents_core::Workflow> {
        let mut builder = zagents_core::WorkflowBuilder::new(self.name.clone());

        if let Some(secs) = self.timeout_secs {
            builder = builder.timeout(Duration::from_secs_f64(secs));
        } else {
            builder = builder.no_timeout();
        }

        for step in &self.steps {
            let registration = make_step_registration(step);
            builder = builder.step(registration);
        }

        builder.build().map_err(workflow_error_to_napi)
    }
}

/// Create a [`StepRegistration`](zagents_core::StepRegistration) from a JS step.
fn make_step_registration(step: &JsStepRegistration) -> zagents_core::StepRegistration {
    let accepts: Vec<&'static str> = step
        .event_types
        .iter()
        .map(|s| intern_event_type(s))
        .collect();

    // Arc clone is cheap -- the ThreadsafeFunction itself is shared.
    let handler_tsfn = Arc::clone(&step.handler);

    let handler: zagents_core::StepFn =
        Arc::new(
            move |event: Box<dyn AnyEvent>,
                  ctx: zagents_core::Context|
                  -> std::pin::Pin<
                Box<
                    dyn std::future::Future<
                            Output = std::result::Result<
                                zagents_core::StepOutput,
                                zagents_core::WorkflowError,
                            >,
                        > + Send,
                >,
            > {
                let tsfn = Arc::clone(&handler_tsfn);

                Box::pin(async move {
                    // Convert the Rust event to a JS-friendly JSON value.
                    let js_event = any_event_to_js_value(&*event);
                    let js_ctx = JsContext::new(ctx);

                    // Call the JavaScript handler function.
                    // ThreadsafeFunction::call_async returns a Future that resolves
                    // to the JS function's return value (serde_json::Value).
                    let result_value: serde_json::Value =
                        tsfn.call_async(Ok((js_event, js_ctx))).await.map_err(
                            |e: napi::Error| zagents_core::WorkflowError::Context(e.to_string()),
                        )?;

                    // Convert the JS return value back to a Rust event.
                    if result_value.is_null() {
                        return Ok(zagents_core::StepOutput::None);
                    }

                    // Check if it's an array (multiple events).
                    if let serde_json::Value::Array(arr) = &result_value {
                        let events: Vec<Box<dyn AnyEvent>> =
                            arr.iter().map(js_value_to_any_event).collect();
                        return Ok(zagents_core::StepOutput::Multiple(events));
                    }

                    // Single event.
                    let event = js_value_to_any_event(&result_value);
                    Ok(zagents_core::StepOutput::Single(event))
                })
            },
        );

    zagents_core::StepRegistration {
        name: step.name.clone(),
        accepts,
        emits: vec![], // JS steps don't declare emits statically.
        handler,
        max_concurrency: 0,
    }
}

/// Convert a result event to a [`JsWorkflowResult`].
fn make_result(event: &dyn AnyEvent) -> JsWorkflowResult {
    let event_type = event.event_type_id().to_owned();
    let json = event.to_json();

    // For StopEvent, extract the result field.
    let data = if event_type == "zagents::StopEvent" {
        json.get("result")
            .cloned()
            .unwrap_or(serde_json::Value::Null)
    } else {
        json
    };

    JsWorkflowResult { event_type, data }
}
