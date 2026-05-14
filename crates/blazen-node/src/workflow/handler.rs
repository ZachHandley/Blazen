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

use blazen_core::runtime;

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
type PinnedEventStream = std::pin::Pin<
    Box<dyn tokio_stream::Stream<Item = Box<dyn blazen_events::AnyEvent>> + Send + Unpin>,
>;

#[napi(js_name = "WorkflowHandler")]
pub struct JsWorkflowHandler {
    /// The inner handler is wrapped in `Arc<Mutex<Option<...>>>` because:
    ///
    /// - `Arc` makes it `Clone` + `Send` + `Sync` for napi.
    /// - `Mutex` provides interior mutability for `&self` methods.
    /// - `Option` allows `take()` since `result()` consumes the Rust handler.
    inner: Arc<Mutex<Option<blazen_core::WorkflowHandler>>>,
    /// Pre-subscribed event stream captured at handler-wrap time so the
    /// first `streamEvents()` call doesn't race against the event loop and
    /// miss events published by the first step. `take()`d on first use;
    /// subsequent `streamEvents()` calls fall back to a fresh subscription.
    pre_stream: Arc<Mutex<Option<PinnedEventStream>>>,
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
        Ok(make_result(&result))
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

    /// Aggregated token usage across the workflow run so far.
    ///
    /// Mirrors [`blazen_core::WorkflowHandler::usage_total`]. Returns
    /// `null` after the handler has been consumed by [`Self::result`].
    #[napi(js_name = "usageTotal")]
    pub async fn usage_total(&self) -> Result<Option<crate::types::JsTokenUsageClass>> {
        let guard = self.inner.lock().await;
        let Some(handler) = guard.as_ref() else {
            return Ok(None);
        };
        Ok(Some(handler.usage_total().await.into()))
    }

    /// Aggregated cost in USD across the workflow run so far. Mirrors
    /// [`blazen_core::WorkflowHandler::cost_total_usd`]. Returns `null`
    /// after the handler has been consumed by [`Self::result`].
    #[napi(js_name = "costTotalUsd")]
    pub async fn cost_total_usd(&self) -> Result<Option<f64>> {
        let guard = self.inner.lock().await;
        let Some(handler) = guard.as_ref() else {
            return Ok(None);
        };
        Ok(Some(handler.cost_total_usd().await))
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
    /// The first call drains the pre-subscribed stream that was set up
    /// before the event loop spawned, so the very first step's events
    /// are captured. Subsequent calls subscribe a fresh stream that
    /// starts from the current point in time.
    #[napi(js_name = "streamEvents")]
    pub async fn stream_events(&self, on_event: StreamCallbackTsfn) -> Result<()> {
        // Prefer the pre-subscribed stream (captured in `new()` before the
        // event loop spawn) for the first call; that's the only way events
        // from the first step survive the wrap-and-subscribe race window.
        let pre = self.pre_stream.lock().await.take();
        let on_event = Arc::new(on_event);

        if let Some(mut stream) = pre {
            runtime::spawn(async move {
                while let Some(event) = stream.next().await {
                    if event.event_type_id() == "blazen::StreamEnd" {
                        break;
                    }
                    let js_event = any_event_to_js_value(&*event);
                    let _ = on_event.call(js_event, ThreadsafeFunctionCallMode::NonBlocking);
                }
            });
            return Ok(());
        }

        // No pre-stream available (already consumed); subscribe fresh.
        let guard = self.inner.lock().await;
        let handler = guard.as_ref().ok_or_else(|| {
            napi::Error::new(
                napi::Status::GenericFailure,
                "WorkflowHandler already consumed -- streamEvents() must be called before result() or pause()"
                    .to_string(),
            )
        })?;

        let mut stream = handler.stream_events();
        runtime::spawn(async move {
            while let Some(event) = stream.next().await {
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
    ///
    /// Pulls the pre-subscribed initial stream out of the handler immediately
    /// so the first `streamEvents()` callback sees events from the very first
    /// step. The pre-subscription was set up in
    /// `Workflow::run_with_event_and_session_refs` BEFORE the event loop was
    /// spawned.
    ///
    /// # Panics
    ///
    /// Panics if the handler does not expose a pre-subscribed initial stream,
    /// which should be impossible on a handler returned directly from
    /// `Workflow::run` and indicates a bug in the wiring otherwise.
    pub(crate) fn new(mut handler: blazen_core::WorkflowHandler) -> Self {
        let stream = handler
            .take_initial_stream()
            .expect("WorkflowHandler must expose a pre-subscribed initial stream");
        Self {
            inner: Arc::new(Mutex::new(Some(handler))),
            pre_stream: Arc::new(Mutex::new(Some(Box::pin(stream)))),
        }
    }
}

/// Convert a [`blazen_core::WorkflowResult`] to a [`JsWorkflowResult`].
///
/// Carries the usage / cost rollups from Wave 3 alongside the terminal
/// event payload. Kept local to avoid a circular dep with `workflow.rs`.
fn make_result(result: &blazen_core::WorkflowResult) -> JsWorkflowResult {
    let event = &*result.event;
    let event_type = event.event_type_id().to_owned();
    let json = event.to_json();

    let data = if event_type == "blazen::StopEvent" {
        json.get("result")
            .cloned()
            .unwrap_or(serde_json::Value::Null)
    } else {
        json
    };

    JsWorkflowResult {
        event_type,
        data,
        usage_total: result.usage_total.clone().into(),
        cost_total_usd: result.cost_total_usd,
    }
}
