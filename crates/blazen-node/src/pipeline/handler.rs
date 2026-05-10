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
    #[napi(js_name = "streamEvents")]
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
