//! Node bindings for [`blazen_controlplane::Client`].
//!
//! Exposes a [`JsControlPlaneClient`] class and a
//! [`JsControlPlaneRunEventStream`] class. The stream class implements
//! the JS iterator protocol (`next()` returning
//! `Promise<{ done, value? }>`) and a `[Symbol.asyncIterator]` method so
//! callers can drive it with `for await`.

use std::path::Path;
use std::pin::Pin;
use std::sync::Arc;

use futures_util::StreamExt;
use napi::Status;
use napi::bindgen_prelude::*;
use napi_derive::napi;
use tokio::sync::Mutex;
use uuid::Uuid;

use blazen_controlplane::Client;
use blazen_controlplane::error::ControlPlaneError;
use blazen_core::distributed::{OrchestratorClient, RunEventStream, SubmitWorkflowRequest};
use blazen_core::error::WorkflowError;

use crate::controlplane::types::{
    JsClientConnectOptions, JsRunEvent, JsRunStateSnapshot, JsSubmitWorkflowOptions,
    JsSubscribeAllOptions, JsWorkerInfo,
};
use crate::error::workflow_error_to_napi;

// ---------------------------------------------------------------------------
// Error mapping
// ---------------------------------------------------------------------------

#[allow(clippy::needless_pass_by_value)]
fn controlplane_error_to_napi(err: ControlPlaneError) -> napi::Error {
    let prefix = match &err {
        ControlPlaneError::Encode(_) => "ControlPlaneEncodeError",
        ControlPlaneError::Json(_) => "ControlPlaneJsonError",
        ControlPlaneError::Transport(_) => "ControlPlaneTransportError",
        ControlPlaneError::EnvelopeVersion { .. } => "ControlPlaneEnvelopeVersionError",
        ControlPlaneError::Tls(_) => "ControlPlaneTlsError",
        ControlPlaneError::Unauthenticated(_) => "ControlPlaneUnauthenticatedError",
        ControlPlaneError::NoMatchingWorker { .. } => "ControlPlaneNoMatchingWorkerError",
        ControlPlaneError::MissingVramHint => "ControlPlaneMissingVramHintError",
        ControlPlaneError::UnknownRun(_) => "ControlPlaneUnknownRunError",
        ControlPlaneError::UnknownWorker(_) => "ControlPlaneUnknownWorkerError",
        ControlPlaneError::Workflow(_) => "ControlPlaneWorkflowError",
        ControlPlaneError::Rpc(_) => "ControlPlaneRpcError",
    };
    napi::Error::new(Status::GenericFailure, format!("[{prefix}] {err}"))
}

// ---------------------------------------------------------------------------
// JsControlPlaneClient
// ---------------------------------------------------------------------------

/// Orchestrator-side client for the control plane. Submit / cancel /
/// describe runs, list workers, drain workers, subscribe to events.
///
/// ```typescript
/// const client = await ControlPlaneClient.connect("http://cp:7445");
/// const snap = await client.submitWorkflow({
///   workflowName: "summarize",
///   input: { url: "https://example.com" },
///   waitForWorker: true,
/// });
/// for await (const event of client.subscribeRunEvents(snap.runId)) {
///   console.log(event.eventType, event.data);
/// }
/// ```
#[napi(js_name = "ControlPlaneClient")]
pub struct JsControlPlaneClient {
    /// Wrapped in `Arc` so each `subscribeRunEvents` call can pin the
    /// client for the lifetime of the returned stream without forcing
    /// JS callers to keep the original client alive.
    inner: Arc<Client>,
}

#[napi]
#[allow(
    clippy::must_use_candidate,
    clippy::missing_errors_doc,
    clippy::missing_panics_doc,
    clippy::needless_pass_by_value
)]
impl JsControlPlaneClient {
    /// Open a connection to the control plane at `endpoint`. Pass
    /// `{ mtls: { cert, key, ca } }` to use mTLS.
    #[napi(factory)]
    pub async fn connect(endpoint: String, opts: Option<JsClientConnectOptions>) -> Result<Self> {
        let client = if let Some(mtls) = opts.and_then(|o| o.mtls) {
            Client::with_mtls(
                endpoint,
                Path::new(&mtls.cert),
                Path::new(&mtls.key),
                Path::new(&mtls.ca),
            )
            .await
        } else {
            Client::connect(endpoint, None).await
        }
        .map_err(controlplane_error_to_napi)?;
        Ok(Self {
            inner: Arc::new(client),
        })
    }

    /// Submit a workflow run.
    #[napi(js_name = "submitWorkflow")]
    pub async fn submit_workflow(
        &self,
        opts: JsSubmitWorkflowOptions,
    ) -> Result<JsRunStateSnapshot> {
        let request = SubmitWorkflowRequest {
            workflow_name: opts.workflow_name,
            workflow_version: opts.workflow_version,
            input: opts.input,
            required_tags: opts.required_tags.unwrap_or_default(),
            idempotency_key: opts.idempotency_key,
            deadline_ms: opts.deadline_ms.map(|b| b.get_u64().1),
            wait_for_worker: opts.wait_for_worker.unwrap_or(false),
            resource_hint: None,
        };
        let snap = self
            .inner
            .submit_workflow(request)
            .await
            .map_err(workflow_error_to_napi)?;
        Ok(snap.into())
    }

    /// Cancel an in-flight workflow run.
    #[napi(js_name = "cancelWorkflow")]
    pub async fn cancel_workflow(&self, run_id: String) -> Result<JsRunStateSnapshot> {
        let uuid = parse_run_id(&run_id)?;
        let snap = self
            .inner
            .cancel_workflow(uuid)
            .await
            .map_err(workflow_error_to_napi)?;
        Ok(snap.into())
    }

    /// Describe the current state of a workflow run.
    #[napi(js_name = "describeWorkflow")]
    pub async fn describe_workflow(&self, run_id: String) -> Result<JsRunStateSnapshot> {
        let uuid = parse_run_id(&run_id)?;
        let snap = self
            .inner
            .describe_workflow(uuid)
            .await
            .map_err(workflow_error_to_napi)?;
        Ok(snap.into())
    }

    /// List currently-connected workers.
    #[napi(js_name = "listWorkers")]
    pub async fn list_workers(&self) -> Result<Vec<JsWorkerInfo>> {
        let workers = self
            .inner
            .list_workers()
            .await
            .map_err(workflow_error_to_napi)?;
        Ok(workers.into_iter().map(Into::into).collect())
    }

    /// Drain a worker. Pass `immediate = true` to refuse new work right
    /// away; otherwise let in-flight runs finish first.
    #[napi(js_name = "drainWorker")]
    pub async fn drain_worker(&self, node_id: String, immediate: bool) -> Result<()> {
        self.inner
            .drain_worker(node_id, immediate)
            .await
            .map_err(controlplane_error_to_napi)
    }

    /// Subscribe to events for a specific run. The returned stream is
    /// a JS `AsyncIterableIterator<RunEvent>` (object with both `next`
    /// and `[Symbol.asyncIterator]`).
    #[napi(
        js_name = "subscribeRunEvents",
        ts_return_type = "AsyncIterableIterator<RunEvent>"
    )]
    pub fn subscribe_run_events<'env>(
        &self,
        env: &'env Env,
        run_id: String,
    ) -> Result<Object<'env>> {
        let uuid = parse_run_id(&run_id)?;
        let client = Arc::clone(&self.inner);
        let fut: OpenStreamFuture = Box::pin(async move {
            let stream = subscribe_run_events_static(&client, uuid)
                .await
                .map_err(workflow_error_to_napi)?;
            Ok((client, stream))
        });
        let pending: PendingStream = Arc::new(Mutex::new(StreamState::Pending(Some(fut))));
        build_run_event_iterable(env, &pending)
    }

    /// Subscribe to events across all runs, optionally filtered by tag
    /// predicates. Returns an `AsyncIterableIterator<RunEvent>`.
    #[napi(
        js_name = "subscribeAll",
        ts_return_type = "AsyncIterableIterator<RunEvent>"
    )]
    pub fn subscribe_all<'env>(
        &self,
        env: &'env Env,
        opts: Option<JsSubscribeAllOptions>,
    ) -> Result<Object<'env>> {
        let required_tags = opts.and_then(|o| o.required_tags).unwrap_or_default();
        let client = Arc::clone(&self.inner);
        let fut: OpenStreamFuture = Box::pin(async move {
            let stream = subscribe_all_static(&client, required_tags)
                .await
                .map_err(controlplane_error_to_napi)?;
            Ok((client, stream))
        });
        let pending: PendingStream = Arc::new(Mutex::new(StreamState::Pending(Some(fut))));
        build_run_event_iterable(env, &pending)
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn parse_run_id(run_id: &str) -> Result<Uuid> {
    Uuid::parse_str(run_id)
        .map_err(|e| napi::Error::from_reason(format!("invalid run id `{run_id}`: {e}")))
}

/// Open a `subscribe_run_events` stream and detach its borrow from the
/// `Client`, returning a `'static`-bound stream that the caller pins
/// alongside an `Arc<Client>` for the duration of the iteration.
///
/// # Safety / soundness
///
/// `Client::subscribe_run_events` produces a [`RunEventStream<'a>`]
/// whose `'a` lifetime ties the stream to `&self`. Inside the
/// implementation, the borrow exists only while the gRPC `Streaming`
/// is being constructed; the resulting stream owns the underlying
/// `tonic::Streaming` outright (a self-contained `hyper` body) and no
/// longer touches `&Client` after creation. The `'a` bound is a
/// defensive trait lifetime, not a real outstanding borrow. Carrying
/// the matching `Arc<Client>` alongside the cast stream pins the
/// client for at least as long as the stream lives, so the lifetime
/// extension is observationally sound.
#[allow(unsafe_code)]
async fn subscribe_run_events_static(
    client: &Arc<Client>,
    run_id: Uuid,
) -> std::result::Result<RunEventStream<'static>, WorkflowError> {
    let stream = client.subscribe_run_events(run_id).await?;
    // SAFETY: see the function-level doc comment above. The stream is
    // backed by a `tonic::Streaming` that holds its own state; the
    // accompanying `Arc<Client>` keeps the gRPC channel alive for as
    // long as the stream is consumed.
    let static_stream: RunEventStream<'static> = unsafe { std::mem::transmute(stream) };
    Ok(static_stream)
}

/// Mirror of [`subscribe_run_events_static`] for the [`Client::subscribe_all`]
/// inherent method.
#[allow(unsafe_code)]
async fn subscribe_all_static(
    client: &Arc<Client>,
    required_tags: Vec<String>,
) -> std::result::Result<RunEventStream<'static>, ControlPlaneError> {
    let stream = client.subscribe_all(required_tags).await?;
    // SAFETY: see [`subscribe_run_events_static`]. The same reasoning
    // applies — the stream is self-contained after construction.
    let static_stream: RunEventStream<'static> = unsafe { std::mem::transmute(stream) };
    Ok(static_stream)
}

// ---------------------------------------------------------------------------
// Async iterable bridge
// ---------------------------------------------------------------------------

/// Future that lazily opens the stream on first `next()` call. We hold
/// it in `Pending(Some(_))` until first poll, then transition to
/// `Active(stream, client)` (keeping the client `Arc` alive so the
/// `'static`-cast borrow remains valid).
type OpenStreamFuture = Pin<
    Box<dyn std::future::Future<Output = Result<(Arc<Client>, RunEventStream<'static>)>> + Send>,
>;

enum StreamState {
    Pending(Option<OpenStreamFuture>),
    /// `Arc<Client>` is held purely to pin the underlying gRPC channel
    /// for the lifetime of the stream (see [`subscribe_run_events_static`]
    /// for the soundness argument). The field is never read directly.
    Active(#[allow(dead_code)] Arc<Client>, RunEventStream<'static>),
    Closed,
}

type PendingStream = Arc<Mutex<StreamState>>;

/// Build the JS-side `AsyncIterableIterator<RunEvent>` object. Mirrors
/// the pattern in [`crate::content::store::byte_stream_to_js_async_iterable`].
fn build_run_event_iterable<'env>(env: &'env Env, state: &PendingStream) -> Result<Object<'env>> {
    let mut iter_obj = Object::new(env)?;

    let next_state = Arc::clone(state);
    let next_fn =
        env.create_function_from_closure::<(), napi::sys::napi_value, _>("next", move |ctx| {
            let state = Arc::clone(&next_state);
            let promise = ctx.env.spawn_future_with_callback(
                async move { pull_next_event(state).await },
                |env, val: Option<JsRunEvent>| {
                    let mut obj = Object::new(env)?;
                    if let Some(event) = val {
                        obj.set("value", event)?;
                        obj.set("done", false)?;
                    } else {
                        obj.set("value", ())?;
                        obj.set("done", true)?;
                    }
                    Ok(obj)
                },
            )?;
            Ok(promise.raw())
        })?;
    iter_obj.set("next", next_fn)?;

    // `[Symbol.asyncIterator]()` returns the iterator itself.
    let iter_raw_value = iter_obj.value();
    let self_returning_fn = env.create_function_from_closure::<(), napi::sys::napi_value, _>(
        "[Symbol.asyncIterator]",
        move |_ctx| Ok(iter_raw_value.value),
    )?;
    let global = env.get_global()?;
    let symbol_obj = global.get_named_property_unchecked::<Object>("Symbol")?;
    let async_iterator_symbol =
        symbol_obj.get_named_property_unchecked::<Unknown>("asyncIterator")?;
    iter_obj.set_property(async_iterator_symbol, self_returning_fn)?;

    Ok(iter_obj)
}

/// Pull the next [`RunEvent`] from the wrapped stream, opening it
/// lazily if this is the first call.
async fn pull_next_event(state: PendingStream) -> Result<Option<JsRunEvent>> {
    // Stage 1: open the stream if still pending. We do this outside the
    // long-lived lock so concurrent `next()` callers serialize cleanly.
    let pending_future = {
        let mut guard = state.lock().await;
        match &mut *guard {
            StreamState::Pending(slot) => slot.take(),
            StreamState::Active(_, _) | StreamState::Closed => None,
        }
    };
    if let Some(fut) = pending_future {
        match fut.await {
            Ok((client, stream)) => {
                let mut guard = state.lock().await;
                // Only transition if we're still pending — another
                // concurrent caller may have raced us, but only one
                // future was stored so this is the only writer.
                *guard = StreamState::Active(client, stream);
            }
            Err(e) => {
                let mut guard = state.lock().await;
                *guard = StreamState::Closed;
                return Err(e);
            }
        }
    }

    // Stage 2: pull a single item from the active stream.
    let mut guard = state.lock().await;
    match &mut *guard {
        StreamState::Active(_, stream) => match stream.next().await {
            Some(Ok(event)) => Ok(Some(JsRunEvent::from(event))),
            Some(Err(e)) => {
                *guard = StreamState::Closed;
                Err(workflow_error_to_napi(e))
            }
            None => {
                *guard = StreamState::Closed;
                Ok(None)
            }
        },
        StreamState::Closed => Ok(None),
        StreamState::Pending(_) => {
            // Should be unreachable — Stage 1 either transitioned to
            // Active or returned an error.
            *guard = StreamState::Closed;
            Err(napi::Error::from_reason(
                "stream state inconsistent: still pending after open future",
            ))
        }
    }
}
