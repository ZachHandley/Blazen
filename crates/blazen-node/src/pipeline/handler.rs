use std::sync::Arc;

use napi::Status;
use napi::bindgen_prelude::*;
use napi::threadsafe_function::{ThreadsafeFunction, ThreadsafeFunctionCallMode};
use napi_derive::napi;
use tokio::sync::Mutex;
use tokio_stream::StreamExt;

use blazen_core::runtime;

use crate::error::pipeline_error_to_napi;
use crate::pipeline::progress::JsProgressSnapshot;
use crate::pipeline::snapshot::{JsPipelineResult, JsPipelineSnapshot};
use crate::workflow::event::any_event_to_js_value;
use crate::workflow::events_typed::JsInputResponseEvent;

type StreamCallbackTsfn =
    ThreadsafeFunction<serde_json::Value, Unknown<'static>, serde_json::Value, Status, false, true>;

#[napi(js_name = "PipelineHandler")]
pub struct JsPipelineHandler {
    inner: Arc<Mutex<Option<blazen_pipeline::PipelineHandler>>>,
}

impl JsPipelineHandler {
    pub(crate) fn new(handler: blazen_pipeline::PipelineHandler) -> Self {
        Self {
            inner: Arc::new(Mutex::new(Some(handler))),
        }
    }
}

#[napi]
#[allow(
    clippy::must_use_candidate,
    clippy::missing_errors_doc,
    clippy::needless_pass_by_value
)]
impl JsPipelineHandler {
    /// Await the final pipeline result. Consumes the handler.
    #[napi]
    pub async fn result(&self) -> Result<JsPipelineResult> {
        let handler = {
            let mut guard = self.inner.lock().await;
            guard.take().ok_or_else(|| {
                napi::Error::new(
                    napi::Status::GenericFailure,
                    "PipelineHandler already consumed",
                )
            })?
        };
        let result = handler.result().await.map_err(pipeline_error_to_napi)?;
        Ok(JsPipelineResult::from_inner(result))
    }

    /// Pause the pipeline and return a snapshot. Consumes the handler.
    #[napi]
    pub async fn pause(&self) -> Result<JsPipelineSnapshot> {
        let handler = {
            let mut guard = self.inner.lock().await;
            guard.take().ok_or_else(|| {
                napi::Error::new(
                    napi::Status::GenericFailure,
                    "PipelineHandler already consumed",
                )
            })?
        };
        let snapshot = handler.pause().await.map_err(pipeline_error_to_napi)?;
        Ok(JsPipelineSnapshot::from_inner(snapshot))
    }

    /// Resume a paused pipeline in place.
    #[napi(js_name = "resumeInPlace")]
    pub async fn resume_in_place(&self) -> Result<()> {
        let guard = self.inner.lock().await;
        let handler = guard.as_ref().ok_or_else(|| {
            napi::Error::new(
                napi::Status::GenericFailure,
                "PipelineHandler already consumed",
            )
        })?;
        handler.resume_in_place().map_err(pipeline_error_to_napi)
    }

    /// Capture a snapshot without stopping the pipeline.
    #[napi]
    pub async fn snapshot(&self) -> Result<JsPipelineSnapshot> {
        let guard = self.inner.lock().await;
        let handler = guard.as_ref().ok_or_else(|| {
            napi::Error::new(
                napi::Status::GenericFailure,
                "PipelineHandler already consumed",
            )
        })?;
        let snapshot = handler.snapshot().await.map_err(pipeline_error_to_napi)?;
        Ok(JsPipelineSnapshot::from_inner(snapshot))
    }

    /// Best-effort polled view of the pipeline's stage cursor. Mirrors
    /// [`blazen_pipeline::PipelineHandler::progress`].
    ///
    /// Returns `null` after [`Self::result`] has consumed the handler.
    #[napi]
    pub async fn progress(&self) -> Result<Option<JsProgressSnapshot>> {
        let guard = self.inner.lock().await;
        let Some(handler) = guard.as_ref() else {
            return Ok(None);
        };
        Ok(Some(handler.progress().into()))
    }

    /// Respond to an input request from a stage that is paused on an
    /// `InputRequestEvent`. The response is fanned out to the active
    /// stage's inner workflow(s). Mirrors
    /// [`blazen_pipeline::PipelineHandler::respond_to_input`].
    ///
    /// The `request_id` must match the `InputRequestEvent.request_id` that
    /// was published by a step inside the running stage.
    #[napi(js_name = "respondToInput")]
    pub async fn respond_to_input(
        &self,
        request_id: String,
        response: serde_json::Value,
    ) -> Result<()> {
        let guard = self.inner.lock().await;
        let handler = guard.as_ref().ok_or_else(|| {
            napi::Error::new(
                napi::Status::GenericFailure,
                "PipelineHandler already consumed",
            )
        })?;
        let input_response = blazen_events::InputResponseEvent {
            request_id,
            response,
        };
        handler
            .respond_to_input(input_response)
            .map_err(pipeline_error_to_napi)
    }

    /// Respond to an input request using a typed [`JsInputResponseEvent`].
    ///
    /// Equivalent to [`Self::respond_to_input`] but accepts the typed
    /// event object so JS callers can pass a single value already shaped
    /// like the input-response event they may have built earlier.
    #[napi(js_name = "respondToInputTyped")]
    pub async fn respond_to_input_typed(&self, event: JsInputResponseEvent) -> Result<()> {
        let guard = self.inner.lock().await;
        let handler = guard.as_ref().ok_or_else(|| {
            napi::Error::new(
                napi::Status::GenericFailure,
                "PipelineHandler already consumed",
            )
        })?;
        handler
            .respond_to_input(event.into_event())
            .map_err(pipeline_error_to_napi)
    }

    /// Aggregated token usage across the pipeline run so far. Mirrors
    /// [`blazen_pipeline::PipelineHandler::usage_total`]. Returns `null`
    /// after the handler has been consumed by [`Self::result`].
    #[napi(js_name = "usageTotal")]
    pub async fn usage_total(&self) -> Result<Option<crate::types::JsTokenUsageClass>> {
        let guard = self.inner.lock().await;
        let Some(handler) = guard.as_ref() else {
            return Ok(None);
        };
        let usage = handler.usage_total().await;
        Ok(Some(crate::types::JsTokenUsageClass::from(&usage)))
    }

    /// Aggregated cost in USD across the pipeline run so far. Mirrors
    /// [`blazen_pipeline::PipelineHandler::cost_total_usd`]. Returns `null`
    /// after the handler has been consumed by [`Self::result`].
    #[napi(js_name = "costTotalUsd")]
    pub async fn cost_total_usd(&self) -> Result<Option<f64>> {
        let guard = self.inner.lock().await;
        let Some(handler) = guard.as_ref() else {
            return Ok(None);
        };
        Ok(Some(handler.cost_total_usd().await))
    }

    /// Abort the pipeline.
    #[napi]
    pub async fn abort(&self) -> Result<()> {
        let guard = self.inner.lock().await;
        let handler = guard.as_ref().ok_or_else(|| {
            napi::Error::new(
                napi::Status::GenericFailure,
                "PipelineHandler already consumed",
            )
        })?;
        handler.abort().map_err(pipeline_error_to_napi)
    }

    /// Subscribe to intermediate events from pipeline stages.
    /// The callback `(eventJson) => void` is invoked for each `PipelineEvent`;
    /// `eventJson` is a JS object with shape `{ stageName, branchName, workflowRunId, event }`.
    #[napi(
        js_name = "streamEvents",
        ts_args_type = "onEvent: (event: { stageName: string; branchName: string; workflowRunId: string; event: Event }) => void"
    )]
    pub async fn stream_events(&self, on_event: StreamCallbackTsfn) -> Result<()> {
        let guard = self.inner.lock().await;
        let handler = guard.as_ref().ok_or_else(|| {
            napi::Error::new(
                napi::Status::GenericFailure,
                "PipelineHandler already consumed",
            )
        })?;
        let mut stream = handler.stream_events();
        let on_event = Arc::new(on_event);

        runtime::spawn(async move {
            while let Some(event) = stream.next().await {
                let js_inner = any_event_to_js_value(&*event.event);
                let js_payload = serde_json::json!({
                    "stageName": event.stage_name,
                    "branchName": event.branch_name,
                    "workflowRunId": event.workflow_run_id.to_string(),
                    "event": js_inner,
                });
                let _ = on_event.call(js_payload, ThreadsafeFunctionCallMode::NonBlocking);
            }
        });

        Ok(())
    }
}
