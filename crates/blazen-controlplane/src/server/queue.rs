//! Assignment queue + in-flight tracker.
//!
//! The queue holds two kinds of state:
//!
//! 1. **Pending assignments** — submissions whose `wait_for_worker` flag
//!    was set and which could not be routed immediately. Keyed by the
//!    capability they need; popped when a matching worker connects or
//!    a saturated worker frees capacity.
//!
//! 2. **In-flight assignments** — assignments that have been routed to
//!    a worker but haven't terminated yet. Tracked so we can surrender
//!    them back to the pending pool when the worker disconnects (the
//!    session handler calls [`AssignmentQueue::surrender_session`]).
//!
//! The queue does *not* perform routing itself — that's the admission
//! module's job. The queue is a thin orchestrator: it takes a submission,
//! asks the admission policy where to send it, and either pushes to a
//! worker channel or stashes the assignment until a worker frees up.

use std::collections::{HashMap, VecDeque};
use std::sync::Arc;

use dashmap::DashMap;
use tokio::sync::{Notify, RwLock, broadcast};
use uuid::Uuid;

use blazen_core::distributed::{
    ResourceHint, RunEvent, RunStateSnapshot, RunStatus, WorkerCapability,
};

use crate::error::ControlPlaneError;
use crate::protocol::{Assignment, Offer, ServerToWorker};

use super::admission::{Admission, Decision, NoCandidateReason, RouteRequest};
use super::registry::WorkerRegistry;
use super::store::{AssignmentStore, MemoryAssignmentStore};

/// One queued (or in-flight) assignment plus the routing inputs the queue
/// will need if it has to re-route on a worker disconnect.
#[derive(Clone)]
struct QueuedAssignment {
    assignment: Assignment,
    required_capability: WorkerCapability,
    required_tags: Vec<String>,
}

/// In-flight assignment record — tracks which session is running it so
/// we can surrender it back to the pending pool on disconnect.
#[derive(Clone)]
struct InFlight {
    queued: QueuedAssignment,
    session_id: Uuid,
}

/// Server-side assignment queue.
///
/// Holds pending assignments waiting on a matching worker, plus the
/// in-flight set so disconnects can surrender work back to the pool.
/// Routing decisions are delegated to [`super::admission::Admission`].
pub struct AssignmentQueue {
    /// Pending submissions keyed by the capability they need. Each
    /// bucket is a FIFO. Wrapped in `RwLock` so multiple readers
    /// (admission scans) don't serialize behind a writer.
    pending: RwLock<HashMap<WorkerCapability, VecDeque<QueuedAssignment>>>,
    /// In-flight assignments by `run_id`.
    in_flight: DashMap<Uuid, InFlight>,
    /// Run state snapshot per `run_id`. Updated by `mark_running`,
    /// `mark_completed`, `mark_failed`, `mark_cancelled`. Exposed to
    /// orchestrator describe queries.
    runs: DashMap<Uuid, RunStateSnapshot>,
    /// Signaled whenever a new pending assignment is enqueued or a
    /// worker disconnects (so re-route loops can wake).
    pub wake: Notify,
    /// Broadcast bus for synthesized `status.*` events. Populated by
    /// [`AssignmentQueue::with_events`] when the queue is wired into a
    /// `ControlPlaneServer`. `None` for standalone queues used in unit
    /// tests — those calls compile to no-ops.
    events: Option<broadcast::Sender<RunEvent>>,
    /// Persistence seam. Every mutator writes through here. Reads
    /// hit the in-memory caches above; the store backfills on
    /// cold-start via `ControlPlaneServer::serve`.
    store: Arc<dyn AssignmentStore>,
}

/// Outcome of a [`AssignmentQueue::submit`] attempt.
#[derive(Debug)]
pub enum SubmitOutcome {
    /// Successfully routed and pushed to a worker (Fixed / `VramBudget`).
    Pushed {
        /// Identifier of the submitted run.
        run_id: Uuid,
        /// Session that received the assignment.
        session_id: Uuid,
        /// Stable node id of the chosen worker.
        node_id: String,
    },
    /// Sent as an Offer to a Reactive worker — caller awaits
    /// `OfferDecision`.
    Offered {
        /// Identifier of the submitted run.
        run_id: Uuid,
        /// Session that received the offer.
        session_id: Uuid,
        /// Stable node id of the chosen worker.
        node_id: String,
    },
    /// No worker matches; queued because `wait_for_worker` was true.
    Queued {
        /// Identifier of the submitted run.
        run_id: Uuid,
    },
    /// No worker matches and `wait_for_worker` was false.
    Rejected {
        /// Why the submission was rejected.
        reason: ControlPlaneError,
    },
}

impl AssignmentQueue {
    /// Construct an empty queue with no event bus. Status transitions
    /// emit nothing — used by unit tests that don't care about
    /// fan-out.
    #[must_use]
    pub fn new() -> Self {
        Self::build(Arc::new(MemoryAssignmentStore::default()), None)
    }

    /// Construct a queue wired into a [`broadcast::Sender<RunEvent>`].
    /// Every state transition (`status.running`, `status.completed`,
    /// `status.failed`, `status.cancelled`) emits onto this channel
    /// for `SubscribeRunEvents` / `SubscribeAll` subscribers.
    #[must_use]
    pub fn with_events(events: broadcast::Sender<RunEvent>) -> Self {
        Self::build(Arc::new(MemoryAssignmentStore::default()), Some(events))
    }

    /// Construct a queue wired into both a broadcast event bus AND a
    /// caller-supplied [`AssignmentStore`]. Use this constructor to
    /// swap in a durable backing store (e.g. Valkey) so queue
    /// mutations survive a control-plane restart.
    #[must_use]
    pub fn with_events_and_store(
        events: broadcast::Sender<RunEvent>,
        store: Arc<dyn AssignmentStore>,
    ) -> Self {
        Self::build(store, Some(events))
    }

    fn build(store: Arc<dyn AssignmentStore>, events: Option<broadcast::Sender<RunEvent>>) -> Self {
        Self {
            pending: RwLock::new(HashMap::new()),
            in_flight: DashMap::new(),
            runs: DashMap::new(),
            wake: Notify::new(),
            events,
            store,
        }
    }

    /// Submit a new run to the queue. If a matching worker is available,
    /// pushes (or offers) immediately. Otherwise queues (if
    /// `wait_for_worker`) or rejects.
    ///
    /// Errors are non-fatal — they're surfaced via
    /// [`SubmitOutcome::Rejected`] so callers can map them to gRPC
    /// status codes uniformly.
    #[allow(clippy::too_many_arguments)]
    pub async fn submit(
        &self,
        run_id: Uuid,
        assignment: Assignment,
        required_capability: WorkerCapability,
        required_tags: Vec<String>,
        wait_for_worker: bool,
        registry: &WorkerRegistry,
        admission: &Admission,
    ) -> SubmitOutcome {
        let queued = QueuedAssignment {
            assignment,
            required_capability: required_capability.clone(),
            required_tags: required_tags.clone(),
        };

        // Record the run as Pending until a worker accepts it.
        let pending_snapshot = RunStateSnapshot {
            run_id,
            status: RunStatus::Pending,
            started_at_ms: now_ms(),
            completed_at_ms: None,
            assigned_to: None,
            last_event_at_ms: None,
            output: None,
            error: None,
        };
        self.runs.insert(run_id, pending_snapshot.clone());
        if let Err(e) = self.store.put_state(run_id, &pending_snapshot, None).await {
            tracing::warn!(%run_id, error = %e, "store.put_state(Pending) failed");
        }
        if let Err(e) = self.store.put_assignment(run_id, &queued.assignment).await {
            tracing::warn!(%run_id, error = %e, "store.put_assignment failed");
        }

        // Build the routing inputs. `hint_owned` keeps the converted
        // ResourceHint alive for the duration of the borrow inside
        // `RouteRequest`. The RouteRequest and hint are scoped to a
        // dedicated block so the borrows on `required_capability` and
        // `required_tags` end cleanly before any await below.
        let decision = {
            let hint_owned: Option<ResourceHint> =
                queued.assignment.resource_hint.as_ref().map(Into::into);
            let req = RouteRequest {
                required_capability: &required_capability,
                required_tags: &required_tags,
                resource_hint: hint_owned.as_ref(),
            };
            admission.route(registry, &req)
        };

        match decision {
            Decision::Push {
                session_id,
                node_id,
            } => {
                if self
                    .push_to_session(session_id, &queued, registry)
                    .await
                    .is_ok()
                {
                    self.record_routed(run_id, session_id, &node_id, queued)
                        .await;
                    SubmitOutcome::Pushed {
                        run_id,
                        session_id,
                        node_id,
                    }
                } else {
                    // Worker channel closed mid-route — fall through to queue.
                    self.enqueue_pending(queued).await;
                    self.wake.notify_waiters();
                    SubmitOutcome::Queued { run_id }
                }
            }
            Decision::Offer {
                session_id,
                node_id,
            } => {
                if self
                    .offer_to_session(session_id, &queued, registry)
                    .await
                    .is_ok()
                {
                    self.record_routed(run_id, session_id, &node_id, queued)
                        .await;
                    SubmitOutcome::Offered {
                        run_id,
                        session_id,
                        node_id,
                    }
                } else {
                    self.enqueue_pending(queued).await;
                    self.wake.notify_waiters();
                    SubmitOutcome::Queued { run_id }
                }
            }
            Decision::NoCandidate { reason } => {
                if wait_for_worker {
                    self.enqueue_pending(queued).await;
                    self.wake.notify_waiters();
                    SubmitOutcome::Queued { run_id }
                } else {
                    let err = match reason {
                        NoCandidateReason::NoCapability
                        | NoCandidateReason::TagMismatch
                        | NoCandidateReason::Saturated => ControlPlaneError::NoMatchingWorker {
                            workflow_name: required_capability.kind.clone(),
                            required_tags,
                        },
                        NoCandidateReason::MissingVramHint => ControlPlaneError::MissingVramHint,
                    };
                    SubmitOutcome::Rejected { reason: err }
                }
            }
        }
    }

    /// A worker has just connected (or freed capacity). Drain any
    /// pending assignments for capabilities the worker advertises and
    /// try to route them.
    pub async fn try_drain_for(
        &self,
        capabilities: &[WorkerCapability],
        registry: &WorkerRegistry,
        admission: &Admission,
    ) {
        for cap in capabilities {
            loop {
                // Pop one entry under the write lock, then release.
                let queued = {
                    let mut pending = self.pending.write().await;
                    let Some(queue) = pending.get_mut(cap) else {
                        break;
                    };
                    let Some(item) = queue.pop_front() else {
                        break;
                    };
                    item
                };

                let decision = {
                    let hint_owned: Option<ResourceHint> =
                        queued.assignment.resource_hint.as_ref().map(Into::into);
                    let req = RouteRequest {
                        required_capability: &queued.required_capability,
                        required_tags: &queued.required_tags,
                        resource_hint: hint_owned.as_ref(),
                    };
                    admission.route(registry, &req)
                };

                let run_id = queued.assignment.run_id;
                match decision {
                    Decision::Push {
                        session_id,
                        node_id,
                    } => {
                        if self
                            .push_to_session(session_id, &queued, registry)
                            .await
                            .is_err()
                        {
                            self.enqueue_pending(queued).await;
                            break;
                        }
                        self.record_routed(run_id, session_id, &node_id, queued)
                            .await;
                    }
                    Decision::Offer {
                        session_id,
                        node_id,
                    } => {
                        if self
                            .offer_to_session(session_id, &queued, registry)
                            .await
                            .is_err()
                        {
                            self.enqueue_pending(queued).await;
                            break;
                        }
                        self.record_routed(run_id, session_id, &node_id, queued)
                            .await;
                    }
                    Decision::NoCandidate { .. } => {
                        // Still no worker — re-queue and stop draining this cap.
                        self.enqueue_pending(queued).await;
                        break;
                    }
                }
            }
        }
    }

    /// A worker has disconnected. For each in-flight assignment routed
    /// to that session, push it back into the pending queue so the
    /// next matching worker can pick it up.
    pub async fn surrender_session(&self, session_id: Uuid) {
        let surrendered: Vec<QueuedAssignment> = self
            .in_flight
            .iter()
            .filter(|r| r.value().session_id == session_id)
            .map(|r| r.value().queued.clone())
            .collect();

        // Mirror the release into the store so its in-flight view
        // matches the in-memory cache. The returned Vec<Uuid> is
        // intentionally ignored — the cache-side iteration above is
        // already authoritative for the runtime path.
        if let Err(e) = self.store.release_inflight(session_id).await {
            tracing::warn!(%session_id, error = %e, "store.release_inflight failed");
        }

        // Remove the in-flight entries and reset their run state.
        let surrendered_ids: Vec<Uuid> = surrendered.iter().map(|q| q.assignment.run_id).collect();
        for id in surrendered_ids {
            self.in_flight.remove(&id);
            let mirrored = if let Some(mut snap) = self.runs.get_mut(&id) {
                snap.status = RunStatus::Pending;
                snap.assigned_to = None;
                Some(snap.clone())
            } else {
                None
            };
            if let Some(snapshot) = mirrored
                && let Err(e) = self.store.put_state(id, &snapshot, None).await
            {
                tracing::warn!(run_id = %id, error = %e, "store.put_state(Pending) failed");
            }
        }

        for queued in surrendered {
            self.enqueue_pending(queued).await;
        }
        self.wake.notify_waiters();
    }

    /// Mark a run as completed with the given output.
    pub async fn mark_completed(&self, run_id: Uuid, output: serde_json::Value) {
        if let Some(mut snap) = self.runs.get_mut(&run_id) {
            let data = serde_json::json!({ "output": output.clone() });
            snap.status = RunStatus::Completed;
            snap.completed_at_ms = Some(now_ms());
            snap.output = Some(output);
            let snapshot = snap.clone();
            drop(snap);
            if let Err(e) = self.store.put_state(run_id, &snapshot, None).await {
                tracing::warn!(%run_id, error = %e, "store.put_state(Completed) failed");
            }
            if let Err(e) = self.store.delete_assignment(run_id).await {
                tracing::warn!(%run_id, error = %e, "store.delete_assignment failed");
            }
            self.emit_status(run_id, "status.completed", data);
        }
        self.in_flight.remove(&run_id);
    }

    /// Mark a run as failed with the given error message.
    pub async fn mark_failed(&self, run_id: Uuid, error: String) {
        if let Some(mut snap) = self.runs.get_mut(&run_id) {
            let data = serde_json::json!({ "error": error.clone() });
            snap.status = RunStatus::Failed;
            snap.completed_at_ms = Some(now_ms());
            snap.error = Some(error);
            let snapshot = snap.clone();
            drop(snap);
            if let Err(e) = self.store.put_state(run_id, &snapshot, None).await {
                tracing::warn!(%run_id, error = %e, "store.put_state(Failed) failed");
            }
            if let Err(e) = self.store.delete_assignment(run_id).await {
                tracing::warn!(%run_id, error = %e, "store.delete_assignment failed");
            }
            self.emit_status(run_id, "status.failed", data);
        }
        self.in_flight.remove(&run_id);
    }

    /// Mark a run as cancelled.
    pub async fn mark_cancelled(&self, run_id: Uuid) {
        if let Some(mut snap) = self.runs.get_mut(&run_id) {
            snap.status = RunStatus::Cancelled;
            snap.completed_at_ms = Some(now_ms());
            let snapshot = snap.clone();
            drop(snap);
            if let Err(e) = self.store.put_state(run_id, &snapshot, None).await {
                tracing::warn!(%run_id, error = %e, "store.put_state(Cancelled) failed");
            }
            if let Err(e) = self.store.delete_assignment(run_id).await {
                tracing::warn!(%run_id, error = %e, "store.delete_assignment failed");
            }
            self.emit_status(run_id, "status.cancelled", serde_json::json!({}));
        }
        self.in_flight.remove(&run_id);
    }

    /// Record an event timestamp for a run. Used to populate
    /// `RunStateSnapshot.last_event_at_ms`.
    pub async fn record_event(&self, run_id: Uuid) {
        let mirrored = if let Some(mut snap) = self.runs.get_mut(&run_id) {
            snap.last_event_at_ms = Some(now_ms());
            Some(snap.clone())
        } else {
            None
        };
        if let Some(snapshot) = mirrored
            && let Err(e) = self.store.put_state(run_id, &snapshot, None).await
        {
            tracing::warn!(%run_id, error = %e, "store.put_state(event) failed");
        }
    }

    /// Look up a run's current state.
    #[must_use]
    pub fn describe(&self, run_id: Uuid) -> Option<RunStateSnapshot> {
        self.runs.get(&run_id).map(|r| r.clone())
    }

    /// Re-hydrate the in-memory cache from the persisted
    /// [`AssignmentStore`]. Called by [`super::ControlPlaneServer::serve`]
    /// at cold-start so a freshly-booted control plane starts with the
    /// same view of work-in-progress that the previous instance held.
    ///
    /// On cold start no sessions are alive — every persisted in-flight
    /// run is treated as orphaned and re-queued into the pending pool;
    /// every persisted pending entry is read back into the in-memory
    /// FIFO. The run-state snapshot is reset to
    /// [`RunStatus::Pending`] so a re-routed worker can claim the run
    /// cleanly via the normal admission path.
    ///
    /// # Known limitations
    /// - The [`AssignmentStore`] trait does not expose a
    ///   `delete_inflight(run_id)` operation, so orphaned in-flight
    ///   rows persist in the store until the next normal claim from a
    ///   resuming worker overwrites them. This is harmless: the next
    ///   `list_orphaned_inflight` call still classifies the row as
    ///   orphaned (and idempotently re-queues), and the run-state
    ///   snapshot accurately reflects `Pending`.
    /// - Per-entry `required_tags` are recovered for pending entries
    ///   (the store records them) but not for in-flight entries (the
    ///   store doesn't record them alongside the in-flight claim).
    ///   Recovered in-flight entries are re-queued with an empty tag
    ///   predicate; a re-routing worker that matches the capability
    ///   picks them up.
    ///
    /// # Errors
    /// Propagates any [`ControlPlaneError`] returned by the underlying
    /// [`AssignmentStore`] reads. Per-entry write failures are logged
    /// and swallowed so a single bad record cannot abort the whole
    /// recovery pass.
    pub async fn recover_from_store(&self) -> Result<(), ControlPlaneError> {
        use std::collections::HashSet;

        // Re-queue every orphaned in-flight run. Cold start: the
        // alive-session set is empty, so every claim in the store is
        // by definition orphaned.
        let orphaned = self.store.list_orphaned_inflight(&HashSet::new()).await?;
        for run_id in orphaned {
            let Some(assignment) = self.store.get_assignment(run_id).await? else {
                // Inconsistent persisted state: the in-flight set
                // claims this run is ours but no assignment record
                // exists. Skip; nothing routable to recover.
                continue;
            };
            let cap = WorkerCapability {
                kind: format!("workflow:{}", assignment.workflow_name),
                version: assignment.workflow_version.unwrap_or(0),
            };
            let queued = QueuedAssignment {
                assignment,
                required_capability: cap.clone(),
                required_tags: Vec::new(),
            };

            // Populate the in-memory FIFO; admission will pick this
            // up on the next drain. We do NOT mirror the orphan into
            // the store's pending FIFO — the orphan record lives in
            // the inflight set, and if THIS process also crashes
            // before routing the run, the next recovery will
            // re-discover the same orphan and re-queue it again.
            // Mirroring here would double-count and create a hot loop
            // in pass 2 below.
            {
                let mut pending = self.pending.write().await;
                pending.entry(cap.clone()).or_default().push_back(queued);
            }

            // Reset the run-state snapshot to Pending so the next
            // worker claim transitions through Running cleanly.
            if let Some(state) = self.store.get_state(run_id).await? {
                let mut next = state;
                next.status = RunStatus::Pending;
                next.assigned_to = None;
                self.runs.insert(run_id, next.clone());
                if let Err(e) = self.store.put_state(run_id, &next, None).await {
                    tracing::warn!(%run_id, error = %e, "store.put_state during recovery");
                }
            }
        }

        // Re-hydrate the pending FIFOs from the store. We drain each
        // capability bucket fully into a local Vec, populate the
        // in-memory cache, then re-enqueue each entry exactly once
        // back onto the store FIFO so subsequent crashes preserve the
        // pending state. The drain-into-vec step is essential: a
        // dequeue + re-enqueue inside a single `loop` would re-pop
        // the just-re-enqueued entry on the next iteration and never
        // terminate.
        let caps = self.store.list_pending_capabilities().await?;
        for cap in caps {
            // Drain the store FIFO into a buffer.
            let mut drained: Vec<(Uuid, Vec<String>)> = Vec::new();
            while let Some(entry) = self.store.dequeue_pending(&cap).await? {
                drained.push(entry);
            }
            // Populate the in-memory cache + re-mirror to the store
            // FIFO. Each entry passes through the store exactly once.
            for (run_id, tags) in drained {
                let Some(assignment) = self.store.get_assignment(run_id).await? else {
                    // Pending FIFO references a run with no assignment
                    // record; persisted state is inconsistent. Drop
                    // the orphan entry — not re-enqueued.
                    continue;
                };
                let queued = QueuedAssignment {
                    assignment,
                    required_capability: cap.clone(),
                    required_tags: tags.clone(),
                };
                {
                    let mut pending = self.pending.write().await;
                    pending.entry(cap.clone()).or_default().push_back(queued);
                }
                if let Err(e) = self.store.enqueue_pending(&cap, run_id, &tags).await {
                    tracing::warn!(%run_id, error = %e, "re-enqueue during recovery");
                }
            }
        }

        Ok(())
    }

    /// Common tail for a successful Push/Offer routing decision: stash
    /// the in-flight entry, mirror the claim into the store, and flip
    /// the run state to `Running`.
    async fn record_routed(
        &self,
        run_id: Uuid,
        session_id: Uuid,
        node_id: &str,
        queued: QueuedAssignment,
    ) {
        self.in_flight
            .insert(run_id, InFlight { queued, session_id });
        if let Err(e) = self.store.claim_inflight(run_id, session_id).await {
            tracing::warn!(%run_id, %session_id, error = %e, "store.claim_inflight failed");
        }
        self.update_run_to_running(run_id, node_id).await;
    }

    async fn enqueue_pending(&self, queued: QueuedAssignment) {
        let run_id = queued.assignment.run_id;
        let capability = queued.required_capability.clone();
        let required_tags = queued.required_tags.clone();
        {
            let mut pending = self.pending.write().await;
            pending
                .entry(queued.required_capability.clone())
                .or_default()
                .push_back(queued);
        }
        if let Err(e) = self
            .store
            .enqueue_pending(&capability, run_id, &required_tags)
            .await
        {
            tracing::warn!(%run_id, error = %e, "store.enqueue_pending failed");
        }
    }

    async fn push_to_session(
        &self,
        session_id: Uuid,
        queued: &QueuedAssignment,
        registry: &WorkerRegistry,
    ) -> Result<(), ControlPlaneError> {
        let Some(handle) = registry.get(session_id) else {
            return Err(ControlPlaneError::UnknownWorker(session_id.to_string()));
        };
        handle
            .outbound
            .send(ServerToWorker::Assignment(queued.assignment.clone()))
            .await
            .map_err(|e| ControlPlaneError::Transport(format!("worker channel closed: {e}")))
    }

    async fn offer_to_session(
        &self,
        session_id: Uuid,
        queued: &QueuedAssignment,
        registry: &WorkerRegistry,
    ) -> Result<(), ControlPlaneError> {
        let Some(handle) = registry.get(session_id) else {
            return Err(ControlPlaneError::UnknownWorker(session_id.to_string()));
        };
        let offer = Offer {
            envelope_version: crate::protocol::ENVELOPE_VERSION,
            assignment: queued.assignment.clone(),
        };
        handle
            .outbound
            .send(ServerToWorker::Offer(offer))
            .await
            .map_err(|e| ControlPlaneError::Transport(format!("worker channel closed: {e}")))
    }

    async fn update_run_to_running(&self, run_id: Uuid, node_id: &str) {
        let mirrored = if let Some(mut snap) = self.runs.get_mut(&run_id) {
            snap.status = RunStatus::Running;
            snap.assigned_to = Some(node_id.to_string());
            Some(snap.clone())
        } else {
            None
        };
        if let Some(snapshot) = mirrored {
            if let Err(e) = self.store.put_state(run_id, &snapshot, None).await {
                tracing::warn!(%run_id, error = %e, "store.put_state(Running) failed");
            }
            self.emit_status(
                run_id,
                "status.running",
                serde_json::json!({ "node_id": node_id }),
            );
        }
    }

    fn emit_status(&self, run_id: Uuid, event_type: &str, data: serde_json::Value) {
        if let Some(tx) = &self.events {
            let event = RunEvent {
                run_id,
                event_type: event_type.to_string(),
                data,
                timestamp_ms: now_ms(),
            };
            // `send` returns Err when there are zero live subscribers; that's
            // expected and harmless (events for unwatched runs are dropped).
            let _ = tx.send(event);
        }
    }
}

impl Default for AssignmentQueue {
    fn default() -> Self {
        Self::new()
    }
}

fn now_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| u64::try_from(d.as_millis()).unwrap_or(0))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::{BTreeMap, HashSet};

    use blazen_core::distributed::AdmissionMode;
    use tokio::sync::mpsc;

    fn cap(kind: &str, version: u32) -> WorkerCapability {
        WorkerCapability {
            kind: kind.to_string(),
            version,
        }
    }

    fn make_assignment(run_id: Uuid, name: &str) -> Assignment {
        Assignment {
            envelope_version: crate::protocol::ENVELOPE_VERSION,
            run_id,
            parent_run_id: None,
            workflow_name: name.to_string(),
            workflow_version: None,
            input_json: b"{}".to_vec(),
            deadline_ms: None,
            attempt: 0,
            resource_hint: None,
        }
    }

    #[tokio::test]
    async fn submit_pushes_to_matching_fixed_worker() {
        let registry = WorkerRegistry::new();
        let admission = Admission::new();
        let queue = AssignmentQueue::new();

        let (tx, mut rx) = mpsc::channel(8);
        let mut caps = HashSet::new();
        caps.insert(cap("workflow:hello", 1));
        let sid = registry.register(
            "node-a".into(),
            caps,
            BTreeMap::new(),
            AdmissionMode::Fixed { max_in_flight: 4 },
            tx,
        );

        let run_id = Uuid::new_v4();
        let outcome = queue
            .submit(
                run_id,
                make_assignment(run_id, "workflow:hello"),
                cap("workflow:hello", 1),
                vec![],
                false,
                &registry,
                &admission,
            )
            .await;

        assert!(matches!(outcome, SubmitOutcome::Pushed { .. }));
        let pushed = rx.recv().await.expect("worker received assignment");
        assert!(matches!(pushed, ServerToWorker::Assignment(_)));
        // Run state should be Running.
        let snap = queue.describe(run_id).unwrap();
        assert_eq!(snap.status, RunStatus::Running);
        assert_eq!(snap.assigned_to.as_deref(), Some("node-a"));
        // sid used.
        let _ = sid;
    }

    #[tokio::test]
    async fn submit_queues_when_no_worker_and_wait_true() {
        let registry = WorkerRegistry::new();
        let admission = Admission::new();
        let queue = AssignmentQueue::new();

        let run_id = Uuid::new_v4();
        let outcome = queue
            .submit(
                run_id,
                make_assignment(run_id, "workflow:hello"),
                cap("workflow:hello", 1),
                vec![],
                true,
                &registry,
                &admission,
            )
            .await;

        assert!(matches!(outcome, SubmitOutcome::Queued { .. }));
        let snap = queue.describe(run_id).unwrap();
        assert_eq!(snap.status, RunStatus::Pending);
    }

    #[tokio::test]
    async fn submit_rejects_when_no_worker_and_wait_false() {
        let registry = WorkerRegistry::new();
        let admission = Admission::new();
        let queue = AssignmentQueue::new();

        let run_id = Uuid::new_v4();
        let outcome = queue
            .submit(
                run_id,
                make_assignment(run_id, "workflow:hello"),
                cap("workflow:hello", 1),
                vec![],
                false,
                &registry,
                &admission,
            )
            .await;

        assert!(matches!(outcome, SubmitOutcome::Rejected { .. }));
    }

    #[tokio::test]
    async fn try_drain_for_routes_queued_to_new_worker() {
        let registry = WorkerRegistry::new();
        let admission = Admission::new();
        let queue = AssignmentQueue::new();

        // Submit while empty — queues.
        let run_id = Uuid::new_v4();
        let _ = queue
            .submit(
                run_id,
                make_assignment(run_id, "workflow:hello"),
                cap("workflow:hello", 1),
                vec![],
                true,
                &registry,
                &admission,
            )
            .await;
        let snap = queue.describe(run_id).unwrap();
        assert_eq!(snap.status, RunStatus::Pending);

        // A matching worker connects.
        let (tx, mut rx) = mpsc::channel(8);
        let mut caps = HashSet::new();
        caps.insert(cap("workflow:hello", 1));
        let _sid = registry.register(
            "node-a".into(),
            caps,
            BTreeMap::new(),
            AdmissionMode::Fixed { max_in_flight: 4 },
            tx,
        );
        let caps_vec = vec![cap("workflow:hello", 1)];
        queue.try_drain_for(&caps_vec, &registry, &admission).await;

        let received = rx.recv().await.expect("queued assignment delivered");
        assert!(matches!(received, ServerToWorker::Assignment(_)));
        let snap = queue.describe(run_id).unwrap();
        assert_eq!(snap.status, RunStatus::Running);
    }

    #[tokio::test]
    async fn surrender_session_returns_assignments_to_pending() {
        let registry = WorkerRegistry::new();
        let admission = Admission::new();
        let queue = AssignmentQueue::new();

        let (tx, mut _rx) = mpsc::channel(8);
        let mut caps = HashSet::new();
        caps.insert(cap("workflow:hello", 1));
        let sid = registry.register(
            "node-a".into(),
            caps,
            BTreeMap::new(),
            AdmissionMode::Fixed { max_in_flight: 4 },
            tx,
        );

        let run_id = Uuid::new_v4();
        let _ = queue
            .submit(
                run_id,
                make_assignment(run_id, "workflow:hello"),
                cap("workflow:hello", 1),
                vec![],
                false,
                &registry,
                &admission,
            )
            .await;
        assert!(queue.describe(run_id).map(|s| s.status) == Some(RunStatus::Running));

        queue.surrender_session(sid).await;
        let snap = queue.describe(run_id).unwrap();
        assert_eq!(snap.status, RunStatus::Pending);
        assert!(snap.assigned_to.is_none());
    }

    #[tokio::test]
    async fn mark_terminal_states_transition_runs() {
        let queue = AssignmentQueue::new();
        let run_id = Uuid::new_v4();
        // Insert a pending run by hand.
        queue.runs.insert(
            run_id,
            RunStateSnapshot {
                run_id,
                status: RunStatus::Running,
                started_at_ms: 0,
                completed_at_ms: None,
                assigned_to: Some("node".into()),
                last_event_at_ms: None,
                output: None,
                error: None,
            },
        );

        queue
            .mark_completed(run_id, serde_json::json!({"ok": true}))
            .await;
        let snap = queue.describe(run_id).unwrap();
        assert_eq!(snap.status, RunStatus::Completed);
        assert!(snap.output.is_some());
        assert!(snap.completed_at_ms.is_some());

        // mark_failed on a different id.
        let id2 = Uuid::new_v4();
        queue.runs.insert(
            id2,
            RunStateSnapshot {
                run_id: id2,
                status: RunStatus::Running,
                started_at_ms: 0,
                completed_at_ms: None,
                assigned_to: None,
                last_event_at_ms: None,
                output: None,
                error: None,
            },
        );
        queue.mark_failed(id2, "boom".into()).await;
        let s2 = queue.describe(id2).unwrap();
        assert_eq!(s2.status, RunStatus::Failed);
        assert_eq!(s2.error.as_deref(), Some("boom"));
    }
}
