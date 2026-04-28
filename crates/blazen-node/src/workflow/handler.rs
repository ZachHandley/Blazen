//! JavaScript wrapper for [`WorkflowHandler`](blazen_core::WorkflowHandler).
//!
//! Exposes the handler as a napi class so TypeScript users can control a
//! running workflow: await the final result, stream intermediate events,
//! or pause the workflow to obtain a serializable snapshot.

use std::sync::Arc;

use napi::Status;
use napi::bindgen_prelude::*;
use napi::threadsafe_function::{ThreadsafeFunction, ThreadsafeFunctionCallMode};
use napi_derive::napi;
use tokio::sync::Mutex;
use tokio_stream::StreamExt;

use super::event::any_event_to_js_value;
use super::events_typed::JsInputResponseEvent;
use super::workflow::JsWorkflowResult;
use crate::error::workflow_error_to_napi;

/// Stream callback: takes a `serde_json::Value`, returns nothing meaningful.
/// `CalleeHandled = false` to avoid the error-first callback convention.
/// `Weak = true` so it does not prevent Node.js from exiting.
type StreamCallbackTsfn =
    ThreadsafeFunction<serde_json::Value, Unknown<'static>, serde_json::Value, Status, false, true>;

/// A handle to a running workflow.
///
/// Returned by `Workflow.runWithHandler()`. Provides methods to:
///
/// - **`result()`** -- await the final workflow result (consumes the handler).
/// - **`pause()`** -- signal the workflow to pause.
/// - **`snapshot()`** -- get a serializable snapshot as a JSON string.
/// - **`resumeInPlace()`** -- resume a paused workflow without creating a new one.
/// - **`respondToInput(requestId, response)`** -- respond to an input request.
/// - **`abort()`** -- abort the running workflow.
/// - **`streamEvents(callback)`** -- subscribe to intermediate events
///   published via `ctx.writeEventToStream()`.
///
/// **Important:** `result()` consumes the handler internally. You can only
/// call it once. The other control methods (`pause`, `resumeInPlace`,
/// `abort`, `respondToInput`, `snapshot`) borrow the handler and can be
/// called multiple times.
///
/// ```javascript
/// const handler = await workflow.runWithHandler({ message: "hello" });
///
/// // Option A: just get the result
/// const result = await handler.result();
///
/// // Option B: pause, snapshot, then resume
/// await handler.pause();
/// const snap = await handler.snapshot();
/// fs.writeFileSync("snapshot.json", snap);
/// await handler.resumeInPlace();
/// const result = await handler.result();
///
/// // Option C: stream events, then get the result
/// handler.streamEvents((event) => console.log(event));
/// const result = await handler.result();
/// ```
#[napi(js_name = "WorkflowHandler")]
pub struct JsWorkflowHandler {
    /// The inner handler is wrapped in `Arc<Mutex<Option<...>>>` because:
    ///
    /// - `Arc` makes it `Clone` + `Send` + `Sync` for napi.
    /// - `Mutex` provides interior mutability for `&self` methods.
    /// - `Option` allows `take()` since `result()` consumes the Rust handler.
    inner: Arc<Mutex<Option<blazen_core::WorkflowHandler>>>,
}

#[napi]
#[allow(clippy::missing_errors_doc)]
impl JsWorkflowHandler {
    /// Await the final workflow result.
    ///
    /// Returns the result when the workflow completes via a `StopEvent`.
    ///
    /// This method consumes the handler internally -- it can only be called
    /// once.
    #[napi]
    pub async fn result(&self) -> Result<JsWorkflowResult> {
        let handler = {
            let mut guard = self.inner.lock().await;
            guard.take().ok_or_else(|| {
                napi::Error::new(
                    napi::Status::GenericFailure,
                    "WorkflowHandler already consumed (result() or pause() was already called)"
                        .to_string(),
                )
            })?
        };

        let result = handler.result().await.map_err(workflow_error_to_napi)?;
        Ok(make_result(&*result.event))
    }

    /// Signal the running workflow to pause.
    ///
    /// After pausing, use `snapshot()` to get a serializable snapshot, or
    /// `resumeInPlace()` to continue execution.
    #[napi]
    pub async fn pause(&self) -> Result<()> {
        let guard = self.inner.lock().await;
        let handler = guard.as_ref().ok_or_else(|| {
            napi::Error::new(
                napi::Status::GenericFailure,
                "WorkflowHandler already consumed".to_string(),
            )
        })?;
        handler.pause().map_err(workflow_error_to_napi)?;
        Ok(())
    }

    /// Resume a paused workflow in place without creating a new handler.
    #[napi(js_name = "resumeInPlace")]
    pub async fn resume_in_place(&self) -> Result<()> {
        let guard = self.inner.lock().await;
        let handler = guard.as_ref().ok_or_else(|| {
            napi::Error::new(
                napi::Status::GenericFailure,
                "WorkflowHandler already consumed".to_string(),
            )
        })?;
        handler.resume_in_place().map_err(workflow_error_to_napi)?;
        Ok(())
    }

    /// Get a serializable snapshot of the workflow as a JSON string.
    ///
    /// The snapshot contains all workflow state and can be saved to a file
    /// or database. Use `Workflow.resume(snapshotJson)` to resume later.
    #[napi]
    pub async fn snapshot(&self) -> Result<String> {
        let guard = self.inner.lock().await;
        let handler = guard.as_ref().ok_or_else(|| {
            napi::Error::new(
                napi::Status::GenericFailure,
                "WorkflowHandler already consumed".to_string(),
            )
        })?;
        let snap = handler.snapshot().await.map_err(workflow_error_to_napi)?;
        snap.to_json().map_err(workflow_error_to_napi)
    }

    /// Respond to an input request from a paused workflow.
    ///
    /// The `request_id` must match the `InputRequestEvent.request_id` that
    /// was published by the workflow step.
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
                "WorkflowHandler already consumed".to_string(),
            )
        })?;
        let input_response = blazen_events::InputResponseEvent {
            request_id,
            response,
        };
        handler
            .respond_to_input(input_response)
            .map_err(workflow_error_to_napi)?;
        Ok(())
    }

    /// Respond to an input request using a typed [`JsInputResponseEvent`].
    ///
    /// Equivalent to [`Self::respond_to_input`] but accepts the typed
    /// event object so JS callers can pass a single value already shaped
    /// like the [`crate::workflow::events_typed::JsInputResponseEvent`]
    /// they may have built earlier.
    #[napi(js_name = "respondToInputTyped")]
    pub async fn respond_to_input_typed(&self, event: JsInputResponseEvent) -> Result<()> {
        let guard = self.inner.lock().await;
        let handler = guard.as_ref().ok_or_else(|| {
            napi::Error::new(
                napi::Status::GenericFailure,
                "WorkflowHandler already consumed".to_string(),
            )
        })?;
        handler
            .respond_to_input(event.into_event())
            .map_err(workflow_error_to_napi)?;
        Ok(())
    }

    /// Abort the running workflow.
    #[napi]
    pub async fn abort(&self) -> Result<()> {
        let guard = self.inner.lock().await;
        let handler = guard.as_ref().ok_or_else(|| {
            napi::Error::new(
                napi::Status::GenericFailure,
                "WorkflowHandler already consumed".to_string(),
            )
        })?;
        handler.abort().map_err(workflow_error_to_napi)?;
        Ok(())
    }

    /// Subscribe to intermediate events published by steps via
    /// `ctx.writeEventToStream()`.
    ///
    /// The `onEvent` callback receives each event as a plain object.
    /// This must be called **before** `result()` or `pause()`.
    ///
    /// Events published before this call are not replayed.
    #[napi(js_name = "streamEvents")]
    pub async fn stream_events(&self, on_event: StreamCallbackTsfn) -> Result<()> {
        let guard = self.inner.lock().await;
        let handler = guard.as_ref().ok_or_else(|| {
            napi::Error::new(
                napi::Status::GenericFailure,
                "WorkflowHandler already consumed -- streamEvents() must be called before result() or pause()"
                    .to_string(),
            )
        })?;

        let mut stream = handler.stream_events();
        let on_event = Arc::new(on_event);

        // Spawn a forwarding task. The stream will end when the workflow
        // completes or is paused (signaled by the StreamEnd sentinel).
        tokio::spawn(async move {
            while let Some(event) = stream.next().await {
                // Stop on the stream-end sentinel (same as Python bindings).
                if event.event_type_id() == "blazen::StreamEnd" {
                    break;
                }
                let js_event = any_event_to_js_value(&*event);
                let _ = on_event.call(js_event, ThreadsafeFunctionCallMode::NonBlocking);
            }
        });

        Ok(())
    }
}

impl JsWorkflowHandler {
    /// Create a new `JsWorkflowHandler` wrapping a Rust `WorkflowHandler`.
    pub(crate) fn new(handler: blazen_core::WorkflowHandler) -> Self {
        Self {
            inner: Arc::new(Mutex::new(Some(handler))),
        }
    }
}

/// Convert a result event to a [`JsWorkflowResult`].
///
/// This is a copy of the helper in `workflow.rs` -- kept here to avoid
/// circular dependencies. Both produce the same output format.
fn make_result(event: &dyn blazen_events::AnyEvent) -> JsWorkflowResult {
    let event_type = event.event_type_id().to_owned();
    let json = event.to_json();

    let data = if event_type == "blazen::StopEvent" {
        json.get("result")
            .cloned()
            .unwrap_or(serde_json::Value::Null)
    } else {
        json
    };

    JsWorkflowResult { event_type, data }
}
