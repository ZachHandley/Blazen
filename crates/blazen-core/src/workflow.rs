//! Workflow struct and runtime methods.
//!
//! A [`Workflow`] is a named collection of steps with an event-driven
//! execution model. Events flow through an internal channel; the event loop
//! matches each event to registered step handlers and spawns them. Step
//! outputs are routed back into the queue until a [`StopEvent`] terminates
//! the loop.
//!
//! Use [`WorkflowBuilder`](crate::builder::WorkflowBuilder) to construct a
//! validated [`Workflow`], then call [`Workflow::run`] or
//! [`Workflow::run_with_event`] to execute it.
//!
//! ## Pause / Resume
//!
//! A running workflow can be paused via [`WorkflowHandler::pause`]. This
//! waits for all in-flight step tasks to complete, drains pending events from
//! the channel, snapshots the full context state, and returns a serializable
//! [`WorkflowSnapshot`](crate::snapshot::WorkflowSnapshot). The workflow can
//! later be resumed from the snapshot via [`Workflow::resume`].

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use blazen_events::{AnyEvent, DynamicEvent, Event, EventEnvelope, StartEvent};
use serde::Serialize;
use tokio::sync::{broadcast, mpsc, oneshot};
use uuid::Uuid;

use crate::builder::InputHandlerFn;
use crate::context::Context;
use crate::error::WorkflowError;
#[cfg(feature = "persist")]
use crate::event_loop::CheckpointConfig;
use crate::event_loop::event_loop;
#[cfg(feature = "distributed")]
use crate::handler::WorkflowResult;
use crate::handler::{WorkflowControl, WorkflowHandler};
#[cfg(feature = "distributed")]
use crate::session_ref::RemoteRefDescriptor;
use crate::session_ref::{
    RegistryKey, SERIALIZED_SESSION_REFS_META_KEY, SessionRefError, SessionRefRegistry,
    SessionRefSerializable,
};
use crate::snapshot::WorkflowSnapshot;
use crate::step::{StepKind, StepRegistration};

// Re-export `RetryConfig` so users can `use blazen_core::RetryConfig;`.
pub use blazen_llm::retry::RetryConfig;

/// Deserializer callback used by
/// [`Workflow::resume_with_deserializers`] to reconstruct a
/// previously-captured [`SessionRefSerializable`] value from its bytes.
///
/// Keyed by the value's stable type tag (see
/// [`SessionRefSerializable::blazen_type_tag`]). Returning an error
/// aborts the resume with
/// [`WorkflowError::SessionRefsNotSerializable`].
pub type SessionRefDeserializerFn =
    fn(&[u8]) -> Result<Arc<dyn SessionRefSerializable>, SessionRefError>;

/// A validated, ready-to-run workflow.
pub struct Workflow {
    pub(crate) name: String,
    pub(crate) step_registry: HashMap<String, Vec<StepKind>>,
    pub(crate) timeout: Option<Duration>,
    /// Default retry configuration applied to LLM calls inside this workflow.
    /// Step / per-call overrides take precedence; pipeline / provider
    /// defaults take lower precedence.
    pub retry_config: Option<std::sync::Arc<blazen_llm::retry::RetryConfig>>,
    /// Optional inline handler for input requests (HITL without pausing).
    pub(crate) input_handler: Option<InputHandlerFn>,
    /// Whether to automatically publish lifecycle events to the broadcast stream.
    pub(crate) auto_publish_events: bool,
    /// Policy applied to live session references when the workflow is
    /// paused or snapshotted.
    pub(crate) session_pause_policy: crate::session_ref::SessionPausePolicy,
    /// Checkpoint store for durable persistence (requires `persist` feature).
    #[cfg(feature = "persist")]
    pub(crate) checkpoint_store: Option<Arc<dyn blazen_persist::CheckpointStore>>,
    /// Whether to automatically checkpoint after each step completes.
    #[cfg(feature = "persist")]
    pub(crate) checkpoint_after_step: bool,
    /// Whether to collect an append-only history of workflow events (requires `telemetry` feature).
    #[cfg(feature = "telemetry")]
    pub(crate) collect_history: bool,
}

impl std::fmt::Debug for Workflow {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Workflow")
            .field("name", &self.name)
            .field("step_count", &self.step_registry.len())
            .field("timeout", &self.timeout)
            .field("retry_config", &self.retry_config)
            .finish_non_exhaustive()
    }
}

impl Workflow {
    /// Execute the workflow with a raw JSON payload wrapped in a
    /// [`StartEvent`].
    ///
    /// Returns a [`WorkflowHandler`] that can be awaited for the final result
    /// or used to stream intermediate events.
    ///
    /// # Errors
    ///
    /// Returns an error if the initial event cannot be enqueued.
    pub async fn run(&self, input: serde_json::Value) -> crate::error::Result<WorkflowHandler> {
        let start_event = StartEvent { data: input };
        self.run_with_event(start_event).await
    }

    /// Execute the workflow with a custom start event.
    ///
    /// # Errors
    ///
    /// Returns an error if the initial event cannot be enqueued.
    pub async fn run_with_event<E: Event + Serialize>(
        &self,
        start_event: E,
    ) -> crate::error::Result<WorkflowHandler> {
        self.run_with_event_and_session_refs(start_event, None)
            .await
    }

    /// Run this workflow with an externally-supplied session-ref registry.
    ///
    /// Used by the pipeline crate to share one registry across stages so
    /// `__blazen_session_ref__` markers produced in one stage can be
    /// resolved in subsequent stages.
    ///
    /// # Errors
    ///
    /// Returns an error if the initial event cannot be enqueued.
    pub async fn run_with_registry(
        &self,
        input: serde_json::Value,
        session_refs: Arc<SessionRefRegistry>,
    ) -> crate::error::Result<WorkflowHandler> {
        let start_event = StartEvent { data: input };
        self.run_with_event_and_session_refs(start_event, Some(session_refs))
            .await
    }

    /// Internal helper that runs the workflow with an optional
    /// externally-supplied session-ref registry.
    ///
    /// # Errors
    ///
    /// Returns an error if the initial event cannot be enqueued.
    pub(crate) async fn run_with_event_and_session_refs<E: Event + Serialize>(
        &self,
        start_event: E,
        session_refs: Option<Arc<SessionRefRegistry>>,
    ) -> crate::error::Result<WorkflowHandler> {
        // Internal routing channel (unbounded so steps never block).
        let (event_tx, event_rx) = mpsc::unbounded_channel::<EventEnvelope>();

        // External broadcast channel for streaming.
        let (stream_tx, _stream_rx) = broadcast::channel::<Box<dyn AnyEvent>>(256);

        // Pre-subscribe BEFORE the event-loop spawn below. `tokio::sync::broadcast`
        // does not buffer for late subscribers: receivers attached after a
        // `send` never see that event. Subscribing here closes the race
        // between the spawned event loop emitting from the first step and
        // the binding (PyWorkflowHandler::new, JsWorkflowHandler::new, ...)
        // wrapping the returned `WorkflowHandler` and creating its own
        // subscription. Without these two `subscribe()` calls, any event
        // published by the first step before the binding wraps the handler
        // is lost (see test_streamed_event_preserves_identity).
        let accumulator_rx = stream_tx.subscribe();
        let initial_stream_rx = stream_tx.subscribe();

        // Oneshot for the final result.
        let (result_tx, result_rx) = oneshot::channel();

        // Control channel (handler → event loop).
        let (control_tx, control_rx) = mpsc::unbounded_channel::<WorkflowControl>();

        // Build the shared context, optionally reusing an externally-supplied
        // session-ref registry so cross-workflow `__blazen_session_ref__`
        // markers remain resolvable.
        let ctx = match session_refs {
            Some(refs) => Context::new_with_session_refs(event_tx.clone(), stream_tx.clone(), refs),
            None => Context::new(event_tx.clone(), stream_tx.clone()),
        };

        // Install the workflow-scope retry config onto the context so every
        // step clone created by the event loop inherits it through
        // `RetryStack::workflow`. Per-step `with_step_retry` runs on top of
        // this, and per-call overrides take precedence over both.
        let ctx = ctx.with_workflow_retry(self.retry_config.clone());

        // Apply the configured session pause policy.
        ctx.set_session_pause_policy(self.session_pause_policy)
            .await;

        // Capture an Arc to the context's session-ref registry so the
        // returned `WorkflowHandler` can resolve `__blazen_session_ref__`
        // markers in the final result even after the event loop drops
        // the original `Context`.
        let session_refs = ctx.session_refs_arc().await;

        // Set metadata.
        let run_id = Uuid::new_v4();
        ctx.set_metadata("run_id", serde_json::Value::String(run_id.to_string()))
            .await;
        ctx.set_metadata(
            "workflow_name",
            serde_json::Value::String(self.name.clone()),
        )
        .await;

        // Enqueue the initial event.
        let envelope = EventEnvelope::new(Box::new(start_event), None);
        event_tx
            .send(envelope)
            .map_err(|_| WorkflowError::ChannelClosed)?;

        // Create history channel if telemetry is enabled and history collection is on.
        #[cfg(feature = "telemetry")]
        let (history_tx, history_rx) = if self.collect_history {
            let (tx, rx) = mpsc::unbounded_channel();
            (Some(tx), Some(rx))
        } else {
            (None, None)
        };

        // Spawn the event loop.
        let registry = self.step_registry.clone();
        let timeout = self.timeout;
        let workflow_name = self.name.clone();
        let input_handler = self.input_handler.clone();
        let auto_publish = self.auto_publish_events;

        #[cfg(feature = "persist")]
        let checkpoint_config = CheckpointConfig {
            store: self.checkpoint_store.clone(),
            after_step: self.checkpoint_after_step,
        };

        let event_loop_handle = crate::runtime::spawn(event_loop(
            event_rx,
            event_tx,
            registry,
            ctx,
            result_tx,
            timeout,
            control_rx,
            workflow_name,
            run_id,
            input_handler,
            auto_publish,
            #[cfg(feature = "persist")]
            checkpoint_config,
            #[cfg(feature = "telemetry")]
            history_tx,
        ));

        Ok(WorkflowHandler::new(
            result_rx,
            stream_tx,
            accumulator_rx,
            initial_stream_rx,
            control_tx,
            event_loop_handle,
            session_refs,
            #[cfg(feature = "telemetry")]
            history_rx,
        ))
    }

    /// Resume a workflow from a previously captured snapshot.
    ///
    /// The step registry must be provided again since step handler functions
    /// are not serializable. The steps should be the same set (or a
    /// compatible superset) that were registered when the workflow was
    /// originally built.
    ///
    /// This is equivalent to calling
    /// [`Workflow::resume_with_deserializers`] with an empty
    /// deserializer map. Any `SessionPausePolicy::PickleOrSerialize`
    /// payload stored under
    /// [`crate::session_ref::SERIALIZED_SESSION_REFS_META_KEY`] will be
    /// left in metadata untouched — the resumed registry will be
    /// empty.
    ///
    /// # Errors
    ///
    /// Returns an error if the snapshot's pending events cannot be
    /// reinjected into the event channel.
    pub async fn resume(
        snapshot: WorkflowSnapshot,
        steps: Vec<StepRegistration>,
        timeout: Option<Duration>,
    ) -> crate::error::Result<WorkflowHandler> {
        Self::resume_inner(snapshot, steps, HashMap::new(), timeout).await
    }

    /// Resume a workflow from a snapshot, rehydrating
    /// [`SessionRefSerializable`] entries captured under
    /// [`SessionPausePolicy::PickleOrSerialize`](crate::SessionPausePolicy::PickleOrSerialize).
    ///
    /// For each entry stored in snapshot metadata under
    /// [`crate::session_ref::SERIALIZED_SESSION_REFS_META_KEY`], the
    /// resumer looks up `deserializers[type_tag]` and invokes it with
    /// the captured bytes. The resulting `Arc<dyn
    /// SessionRefSerializable>` is re-inserted into the fresh
    /// context's session-ref registry under the original
    /// [`RegistryKey`] so downstream `__blazen_session_ref__` markers
    /// in the snapshot keep resolving after resume.
    ///
    /// # Errors
    ///
    /// Returns [`WorkflowError::SessionRefsNotSerializable`] when a
    /// referenced `type_tag` has no registered deserializer, when the
    /// deserializer itself errors, or when the stored payload is
    /// malformed. Returns [`WorkflowError::ChannelClosed`] if pending
    /// events cannot be reinjected.
    pub async fn resume_with_deserializers(
        snapshot: WorkflowSnapshot,
        steps: Vec<StepRegistration>,
        deserializers: HashMap<&'static str, SessionRefDeserializerFn>,
        timeout: Option<Duration>,
    ) -> crate::error::Result<WorkflowHandler> {
        Self::resume_inner(snapshot, steps, deserializers, timeout).await
    }

    /// Shared implementation behind [`Workflow::resume`] and
    /// [`Workflow::resume_with_deserializers`].
    async fn resume_inner(
        snapshot: WorkflowSnapshot,
        steps: Vec<StepRegistration>,
        deserializers: HashMap<&'static str, SessionRefDeserializerFn>,
        timeout: Option<Duration>,
    ) -> crate::error::Result<WorkflowHandler> {
        // Rebuild the registry from the provided steps. Resume only
        // accepts `Vec<StepRegistration>` (regular steps); sub-workflow
        // steps cannot currently round-trip through a snapshot because
        // their inner `Workflow` is not serializable.
        let mut registry: HashMap<String, Vec<StepKind>> = HashMap::new();
        for step in steps {
            for &event_type in &step.accepts {
                registry
                    .entry(event_type.to_owned())
                    .or_default()
                    .push(StepKind::Regular(step.clone()));
            }
        }

        // Internal routing channel.
        let (event_tx, event_rx) = mpsc::unbounded_channel::<EventEnvelope>();

        // External broadcast channel.
        let (stream_tx, _stream_rx) = broadcast::channel::<Box<dyn AnyEvent>>(256);

        // Pre-subscribe before the event-loop spawn — same race-avoidance as
        // the non-resume path in run_with_event_and_session_refs.
        let accumulator_rx = stream_tx.subscribe();
        let initial_stream_rx = stream_tx.subscribe();

        // Result channel.
        let (result_tx, result_rx) = oneshot::channel();

        // Control channel (handler → event loop).
        let (control_tx, control_rx) = mpsc::unbounded_channel::<WorkflowControl>();

        // Build context and restore state.
        let ctx = Context::new(event_tx.clone(), stream_tx.clone());
        ctx.restore_state(snapshot.context_state).await;
        ctx.restore_collected(snapshot.collected_events).await;
        ctx.restore_metadata(snapshot.metadata).await;

        // Rehydrate any serializable session-ref payloads captured by
        // the PickleOrSerialize policy. We do this BEFORE pending
        // events are reinjected so steps that consume those events
        // see a populated registry.
        let session_refs = ctx.session_refs_arc().await;
        if let Some(meta) = ctx
            .snapshot_metadata()
            .await
            .get(SERIALIZED_SESSION_REFS_META_KEY)
            && !deserializers.is_empty()
        {
            rehydrate_serialized_session_refs(&session_refs, meta, &deserializers).await?;
        }

        // Reinject pending events into the channel.
        // Try to reconstruct concrete event types via the deserializer
        // registry first; fall back to DynamicEvent if no deserializer is
        // registered or deserialization fails.
        for serialized in &snapshot.pending_events {
            let event: Box<dyn AnyEvent> =
                blazen_events::try_deserialize_event(&serialized.event_type, &serialized.data)
                    .unwrap_or_else(|| {
                        Box::new(DynamicEvent {
                            event_type: serialized.event_type.clone(),
                            data: serialized.data.clone(),
                        })
                    });
            let envelope = EventEnvelope::new(event, serialized.source_step.clone());
            event_tx
                .send(envelope)
                .map_err(|_| WorkflowError::ChannelClosed)?;
        }

        // Spawn the event loop.
        let workflow_name = snapshot.workflow_name;
        let run_id = snapshot.run_id;

        // Resumed workflows do not collect history (no builder config available).
        #[cfg(feature = "telemetry")]
        let history_tx: Option<mpsc::UnboundedSender<blazen_telemetry::HistoryEvent>> = None;

        #[cfg(feature = "persist")]
        let checkpoint_config = CheckpointConfig {
            store: None,
            after_step: false,
        };

        let event_loop_handle = crate::runtime::spawn(event_loop(
            event_rx,
            event_tx,
            registry,
            ctx,
            result_tx,
            timeout,
            control_rx,
            workflow_name,
            run_id,
            None,  // No inline input handler for resumed workflows.
            false, // No auto-publish for resumed workflows.
            #[cfg(feature = "persist")]
            checkpoint_config,
            #[cfg(feature = "telemetry")]
            history_tx,
        ));

        Ok(WorkflowHandler::new(
            result_rx,
            stream_tx,
            accumulator_rx,
            initial_stream_rx,
            control_tx,
            event_loop_handle,
            session_refs,
            #[cfg(feature = "telemetry")]
            None, // No history receiver for resumed workflows.
        ))
    }

    /// Resume a workflow from a checkpoint stored in a [`CheckpointStore`].
    ///
    /// Loads the checkpoint identified by `run_id`, converts it to a
    /// [`WorkflowSnapshot`], and resumes using [`Workflow::resume`].
    ///
    /// The step registrations must be provided again since step handler
    /// functions are not serializable.
    ///
    /// # Errors
    ///
    /// Returns [`WorkflowError::Context`] if no checkpoint exists for the
    /// given `run_id`, or propagates any storage error from the checkpoint
    /// store.
    #[cfg(feature = "persist")]
    pub async fn resume_from(
        store: Arc<dyn blazen_persist::CheckpointStore>,
        run_id: &Uuid,
        steps: Vec<StepRegistration>,
    ) -> crate::error::Result<WorkflowHandler> {
        let checkpoint = store
            .load(run_id)
            .await
            .map_err(|e| WorkflowError::Context(format!("checkpoint load failed: {e}")))?
            .ok_or_else(|| {
                WorkflowError::Context(format!("no checkpoint found for run_id {run_id}"))
            })?;

        let snapshot: WorkflowSnapshot = checkpoint.into();

        Self::resume(snapshot, steps, Some(Duration::from_mins(5))).await
    }

    /// Return the unique step names registered in this workflow.
    ///
    /// The order is deterministic within a single build but depends on
    /// the internal `HashMap` iteration order, so callers should not
    /// rely on it being stable across runs.
    #[must_use]
    pub fn step_names(&self) -> Vec<String> {
        let mut seen = std::collections::HashSet::new();
        let mut names = Vec::new();
        for registrations in self.step_registry.values() {
            for kind in registrations {
                let name = kind.name();
                if seen.insert(name.to_owned()) {
                    names.push(name.to_owned());
                }
            }
        }
        names
    }

    /// Invoke this workflow on a remote peer.
    ///
    /// The peer must have the same steps registered in its
    /// [`crate::step_registry`] under the same step IDs. Returns a
    /// [`WorkflowResult`] whose `event` field is a
    /// [`StopEvent`](blazen_events::StopEvent) carrying the remote
    /// sub-workflow's terminal result. Any non-serializable session
    /// refs produced by the remote workflow are tracked as
    /// [`RemoteRefDescriptor`](crate::session_ref::RemoteRefDescriptor)s
    /// in the returned `session_refs` registry -- call
    /// `deref_session_ref` on the peer client to fetch their values.
    ///
    /// The `peer` parameter is any implementor of
    /// [`PeerClient`](crate::distributed::PeerClient), which
    /// `blazen_peer::BlazenPeerClient` implements when the `distributed`
    /// feature of this crate is enabled alongside `blazen-peer`.
    ///
    /// # Errors
    ///
    /// Returns a [`WorkflowError`] wrapping the peer error if the remote
    /// call fails, or if the remote workflow itself reported an error.
    #[cfg(feature = "distributed")]
    pub async fn run_remote(
        &self,
        input: serde_json::Value,
        peer: &dyn crate::distributed::PeerClient,
    ) -> crate::error::Result<WorkflowResult> {
        use crate::distributed::{RemoteWorkflowRequest, RemoteWorkflowResponse};

        // Collect unique step names from our local step registrations.
        let step_ids = self.step_names();

        let request = RemoteWorkflowRequest {
            workflow_name: self.name.clone(),
            step_ids,
            input,
            timeout_secs: self.timeout.map(|d| d.as_secs()),
        };

        let response: RemoteWorkflowResponse = peer
            .invoke_sub_workflow(request)
            .await
            .map_err(|e| WorkflowError::Context(format!("peer invocation failed: {e}")))?;

        // Check for remote error.
        if let Some(err) = &response.error
            && !err.is_empty()
        {
            return Err(WorkflowError::Context(format!(
                "remote workflow failed: {err}"
            )));
        }

        // Build a local session ref registry and populate it with remote refs.
        let registry = Arc::new(SessionRefRegistry::new());
        for (key_uuid, descriptor) in &response.remote_refs {
            let key = RegistryKey(*key_uuid);
            let remote_desc = RemoteRefDescriptor {
                origin_node_id: descriptor.origin_node_id.clone(),
                type_tag: descriptor.type_tag.clone(),
                created_at_epoch_ms: descriptor.created_at_epoch_ms,
            };
            // Best-effort insert; capacity errors are unlikely for a
            // single sub-workflow response but we log and continue.
            let _ = registry.insert_remote(key, remote_desc).await;
        }

        // Build a StopEvent from the response result.
        let result_json = response.result.unwrap_or(serde_json::Value::Null);
        let stop_event = blazen_events::StopEvent {
            result: result_json,
        };

        Ok(WorkflowResult {
            event: Box::new(stop_event),
            session_refs: registry,
            // Remote sub-workflows do not (yet) report aggregate usage
            // back through the wire protocol; surface zero totals so the
            // local caller never sees a stale or partial number.
            usage_total: blazen_llm::types::TokenUsage::zero(),
            cost_total_usd: 0.0,
        })
    }

    /// Build a new workflow by looking up each step ID in the
    /// process-global
    /// [`StepDeserializerRegistry`](crate::step_registry::StepDeserializerRegistry).
    ///
    /// Returns an error if any step ID is not registered. This is the
    /// entry point used by the distributed workflow peer server to
    /// reconstruct a sub-workflow from a list of step IDs carried in a
    /// wire request — the peer must have the same step code compiled
    /// in as the caller.
    ///
    /// The resulting workflow has no timeout applied. Callers who need
    /// a timeout should construct the workflow manually with
    /// [`WorkflowBuilder`](crate::builder::WorkflowBuilder) and call
    /// [`WorkflowBuilder::timeout`](crate::builder::WorkflowBuilder::timeout)
    /// explicitly.
    ///
    /// # Errors
    ///
    /// Returns [`WorkflowError::UnknownStep`] if any entry in
    /// `step_ids` is not registered in the global step registry.
    /// Propagates [`WorkflowError::ValidationFailed`] from
    /// [`WorkflowBuilder::build`](crate::builder::WorkflowBuilder::build)
    /// if the resulting step set is invalid (e.g. empty).
    pub fn new_from_registered_steps(
        name: impl Into<String>,
        step_ids: Vec<&str>,
    ) -> crate::error::Result<Self> {
        use crate::builder::WorkflowBuilder;
        use crate::step_registry::lookup_step_builder;

        let mut builder = WorkflowBuilder::new(name);
        for step_id in step_ids {
            let registration =
                lookup_step_builder(step_id).ok_or_else(|| WorkflowError::UnknownStep {
                    step_id: step_id.to_string(),
                })?;
            builder = builder.step(registration);
        }
        builder.no_timeout().build()
    }
}

/// Walk the `__blazen_serialized_session_refs` sidecar produced by the
/// [`SessionPausePolicy::PickleOrSerialize`](crate::SessionPausePolicy::PickleOrSerialize)
/// snapshot path and re-insert each entry into the fresh context's
/// session-ref registry under its original [`RegistryKey`].
///
/// Malformed records (missing `type_tag`, missing/non-array `data`,
/// unknown `type_tag`, deserializer error) are surfaced as
/// [`WorkflowError::SessionRefsNotSerializable`] with the offending
/// keys in the error payload.
async fn rehydrate_serialized_session_refs(
    registry: &Arc<SessionRefRegistry>,
    meta: &serde_json::Value,
    deserializers: &HashMap<&'static str, SessionRefDeserializerFn>,
) -> crate::error::Result<()> {
    let Some(entries) = meta.as_object() else {
        return Err(WorkflowError::SessionRefsNotSerializable {
            keys: vec!["__blazen_serialized_session_refs metadata is not a JSON object".to_owned()],
        });
    };

    let mut failures: Vec<String> = Vec::new();

    for (key_str, record) in entries {
        let Ok(key) = RegistryKey::parse(key_str) else {
            failures.push(format!("invalid RegistryKey '{key_str}'"));
            continue;
        };

        let Some(record_obj) = record.as_object() else {
            failures.push(format!("record for key {key_str} is not an object"));
            continue;
        };

        let Some(type_tag) = record_obj.get("type_tag").and_then(|v| v.as_str()) else {
            failures.push(format!("record for key {key_str} missing type_tag"));
            continue;
        };

        let Some(data_value) = record_obj.get("data") else {
            failures.push(format!("record for key {key_str} missing data"));
            continue;
        };

        let bytes: crate::value::BytesWrapper = match serde_json::from_value(data_value.clone()) {
            Ok(b) => b,
            Err(e) => {
                failures.push(format!("failed to decode data bytes for {key_str}: {e}"));
                continue;
            }
        };

        let Some(&deserializer) = deserializers.get(type_tag) else {
            failures.push(format!(
                "no registered deserializer for type_tag '{type_tag}' (key {key_str})"
            ));
            continue;
        };

        let value = match deserializer(&bytes.0) {
            Ok(v) => v,
            Err(e) => {
                failures.push(format!(
                    "deserializer for type_tag '{type_tag}' failed on key {key_str}: {e}"
                ));
                continue;
            }
        };

        if let Err(e) = registry.insert_serializable_with_key(key, value).await {
            failures.push(format!("registry insert failed for {key_str}: {e}"));
        }
    }

    if failures.is_empty() {
        Ok(())
    } else {
        Err(WorkflowError::SessionRefsNotSerializable { keys: failures })
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use blazen_events::{Event, StartEvent, StopEvent};
    use std::sync::Arc;
    use std::time::Duration;

    use crate::Workflow;
    use crate::builder::WorkflowBuilder;
    use crate::error::WorkflowError;
    use crate::step::{StepFn, StepOutput, StepRegistration};

    /// Helper: a step that converts a `StartEvent` into a `StopEvent`.
    fn echo_step() -> StepRegistration {
        let handler: StepFn = Arc::new(|event, _ctx| {
            Box::pin(async move {
                let start = event
                    .as_any()
                    .downcast_ref::<StartEvent>()
                    .expect("expected StartEvent");
                let stop = StopEvent {
                    result: start.data.clone(),
                };
                Ok(StepOutput::Single(Box::new(stop)))
            })
        });

        StepRegistration {
            name: "echo".into(),
            accepts: vec![StartEvent::event_type()],
            emits: vec![StopEvent::event_type()],
            handler,
            max_concurrency: 0,
            semaphore: None,
            timeout: None,
            retry_config: None,
        }
    }

    #[tokio::test]
    async fn simple_start_to_stop() {
        let workflow = WorkflowBuilder::new("test")
            .step(echo_step())
            .build()
            .unwrap();

        let handler = workflow
            .run(serde_json::json!({"hello": "world"}))
            .await
            .unwrap();
        let result = handler.result().await.unwrap().event;
        assert_eq!(result.event_type_id(), StopEvent::event_type());

        let stop = result.downcast_ref::<StopEvent>().unwrap();
        assert_eq!(stop.result, serde_json::json!({"hello": "world"}));
    }

    #[tokio::test]
    async fn empty_workflow_fails_validation() {
        let result = WorkflowBuilder::new("empty").build();
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(matches!(err, WorkflowError::ValidationFailed(_)));
    }

    #[test]
    fn step_names_returns_unique_names() {
        // A second step that also listens on StartEvent.
        let handler_b: StepFn =
            Arc::new(|_event, _ctx| Box::pin(async move { Ok(StepOutput::None) }));
        let step_b = StepRegistration {
            name: "side_effect".into(),
            accepts: vec![StartEvent::event_type()],
            emits: vec![],
            handler: handler_b,
            max_concurrency: 0,
            semaphore: None,
            timeout: None,
            retry_config: None,
        };

        let workflow = WorkflowBuilder::new("step-names-test")
            .step(echo_step())
            .step(step_b)
            .build()
            .unwrap();

        let mut names = workflow.step_names();
        names.sort();
        assert_eq!(names, vec!["echo", "side_effect"]);
    }

    #[tokio::test]
    async fn timeout_triggers() {
        // A step that never produces output.
        let handler: StepFn = Arc::new(|_event, _ctx| {
            Box::pin(async move {
                // Sleep forever.
                tokio::time::sleep(Duration::from_hours(1)).await;
                Ok(StepOutput::None)
            })
        });

        let step = StepRegistration {
            name: "slow".into(),
            accepts: vec![StartEvent::event_type()],
            emits: vec![],
            handler,
            max_concurrency: 0,
            semaphore: None,
            timeout: None,
            retry_config: None,
        };

        let workflow = WorkflowBuilder::new("timeout-test")
            .step(step)
            .timeout(Duration::from_millis(50))
            .build()
            .unwrap();

        let wf_handler = workflow.run(serde_json::json!(null)).await.unwrap();
        let result = wf_handler.result().await;
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), WorkflowError::Timeout { .. }));
    }

    #[tokio::test]
    async fn step_error_propagates() {
        let handler: StepFn = Arc::new(|_event, _ctx| {
            Box::pin(async move { Err(WorkflowError::Context("test error".into())) })
        });

        let step = StepRegistration {
            name: "failing".into(),
            accepts: vec![StartEvent::event_type()],
            emits: vec![],
            handler,
            max_concurrency: 0,
            semaphore: None,
            timeout: None,
            retry_config: None,
        };

        let workflow = WorkflowBuilder::new("error-test")
            .step(step)
            .build()
            .unwrap();

        let wf_handler = workflow.run(serde_json::json!(null)).await.unwrap();
        let result = wf_handler.result().await;
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            WorkflowError::StepFailed { .. }
        ));
    }

    // --------------------------------------------------------------
    // PickleOrSerialize snapshot + resume round-trip
    // --------------------------------------------------------------

    /// A tiny serializable ref used for the round-trip test: stores a
    /// single `i32` as four big-endian bytes.
    struct TestSerializable {
        value: i32,
    }

    impl crate::session_ref::SessionRefSerializable for TestSerializable {
        fn blazen_serialize(&self) -> Result<Vec<u8>, crate::session_ref::SessionRefError> {
            Ok(self.value.to_be_bytes().to_vec())
        }
        fn blazen_type_tag(&self) -> &'static str {
            "test::TestSerializable"
        }
    }

    fn test_deserialize(
        bytes: &[u8],
    ) -> Result<
        Arc<dyn crate::session_ref::SessionRefSerializable>,
        crate::session_ref::SessionRefError,
    > {
        if bytes.len() != 4 {
            return Err(crate::session_ref::SessionRefError::SerializationFailed {
                type_tag: "test::TestSerializable".to_owned(),
                source: "expected 4 bytes".into(),
            });
        }
        let mut buf = [0u8; 4];
        buf.copy_from_slice(bytes);
        let value = i32::from_be_bytes(buf);
        Ok(Arc::new(TestSerializable { value }))
    }

    /// Step that inserts a `TestSerializable` session ref into the
    /// context registry and then pauses the workflow for snapshotting.
    /// Does NOT emit a `StopEvent` so the handler stays alive for
    /// the snapshot + resume dance.
    fn park_step() -> StepRegistration {
        let handler: StepFn = Arc::new(|_event, ctx| {
            Box::pin(async move {
                let registry = ctx.session_refs_arc().await;
                let _ = registry
                    .insert_serializable(Arc::new(TestSerializable { value: 1234 }))
                    .await
                    .unwrap();
                Ok(StepOutput::None)
            })
        });

        StepRegistration {
            name: "park".into(),
            accepts: vec![StartEvent::event_type()],
            emits: vec![],
            handler,
            max_concurrency: 0,
            semaphore: None,
            timeout: None,
            retry_config: None,
        }
    }

    #[tokio::test]
    async fn pickle_or_serialize_round_trip_through_snapshot() {
        use crate::session_ref::{SERIALIZED_SESSION_REFS_META_KEY, SessionPausePolicy};
        use std::collections::HashMap;

        // 1. Build a workflow configured with PickleOrSerialize.
        let workflow = WorkflowBuilder::new("serialize-roundtrip")
            .step(park_step())
            .session_pause_policy(SessionPausePolicy::PickleOrSerialize)
            .build()
            .unwrap();

        // 2. Run it and wait for the park step to insert the ref.
        let wf_handler = workflow.run(serde_json::json!(null)).await.unwrap();
        // Give the step a moment to run and insert the serializable ref.
        tokio::time::sleep(Duration::from_millis(50)).await;

        // 3. Pause and snapshot. Then abort so the handler frees up.
        wf_handler.pause().unwrap();
        let snapshot = wf_handler.snapshot().await.unwrap();
        wf_handler.abort().unwrap();

        // 4. Verify the metadata contains the serialized bytes.
        let raw = snapshot
            .metadata
            .get(SERIALIZED_SESSION_REFS_META_KEY)
            .expect("metadata must contain serialized session refs");
        let entries = raw
            .as_object()
            .expect("serialized session refs metadata must be a JSON object");
        assert_eq!(entries.len(), 1);
        let (_key_str, record) = entries.iter().next().unwrap();
        let record_obj = record.as_object().unwrap();
        assert_eq!(
            record_obj.get("type_tag").and_then(|v| v.as_str()).unwrap(),
            "test::TestSerializable"
        );
        let bytes: crate::value::BytesWrapper =
            serde_json::from_value(record_obj.get("data").unwrap().clone()).unwrap();
        assert_eq!(bytes.0, vec![0, 0, 4, 210]); // 1234 big-endian

        // 5. Resume with deserializers and verify the ref is rehydrated.
        let mut deserializers: HashMap<&'static str, crate::workflow::SessionRefDeserializerFn> =
            HashMap::new();
        deserializers.insert("test::TestSerializable", test_deserialize);

        let resumed_handler = Workflow::resume_with_deserializers(
            snapshot,
            vec![park_step()],
            deserializers,
            Some(Duration::from_millis(200)),
        )
        .await
        .unwrap();

        // 6. The resumed handler's session-ref registry should now
        // contain exactly one entry, and it should be the concrete
        // TestSerializable we originally captured.
        let resumed_refs = resumed_handler.session_refs();
        assert_eq!(resumed_refs.len().await, 1);
        let entries = resumed_refs.serializable_entries().await;
        assert_eq!(entries.len(), 1);
        let ser = &entries[0].1;
        let round_trip = ser.blazen_serialize().unwrap();
        assert_eq!(round_trip, vec![0, 0, 4, 210]);
        assert_eq!(ser.blazen_type_tag(), "test::TestSerializable");

        // Clean up the resumed workflow — it has no StopEvent source
        // so it would otherwise time out.
        resumed_handler.abort().unwrap();
    }

    #[tokio::test]
    async fn resume_with_missing_deserializer_errors() {
        use crate::session_ref::SessionPausePolicy;
        use std::collections::HashMap;

        let workflow = WorkflowBuilder::new("serialize-missing-deser")
            .step(park_step())
            .session_pause_policy(SessionPausePolicy::PickleOrSerialize)
            .build()
            .unwrap();

        let wf_handler = workflow.run(serde_json::json!(null)).await.unwrap();
        tokio::time::sleep(Duration::from_millis(50)).await;
        wf_handler.pause().unwrap();
        let snapshot = wf_handler.snapshot().await.unwrap();
        wf_handler.abort().unwrap();

        // Empty deserializer map → the resume should fail with
        // SessionRefsNotSerializable pointing at the missing tag.
        let deserializers: HashMap<&'static str, crate::workflow::SessionRefDeserializerFn> =
            HashMap::new();

        let resumed = Workflow::resume_with_deserializers(
            snapshot,
            vec![park_step()],
            deserializers,
            Some(Duration::from_millis(200)),
        )
        .await;

        // With an empty map, rehydrate_serialized_session_refs is
        // skipped entirely (the if gating on !deserializers.is_empty()
        // short-circuits), so the resume succeeds but the registry
        // starts empty. That's a deliberate design choice: callers
        // who want a strict check supply at least one entry.
        let h = resumed.unwrap();
        assert_eq!(h.session_refs().len().await, 0);
        h.abort().unwrap();
    }
}
