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

use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;
use std::time::Instant;

use dashmap::DashMap;
use tokio::sync::{Notify, RwLock, broadcast};
use uuid::Uuid;

use blazen_core::distributed::{
    NodeSelector, ResourceHint, RunEvent, RunStateSnapshot, RunStatus, TolerationSpec,
    WorkerCapability,
};

use crate::error::ControlPlaneError;
use crate::protocol::{Assignment, Offer, ServerToWorker};

use super::admission::{Admission, Decision, NoCandidateReason, RouteRequest};
use super::registry::WorkerRegistry;
use super::store::{AssignmentStore, MemoryAssignmentStore};

/// One queued (or in-flight) assignment plus the routing inputs the queue
/// will need if it has to re-route on a worker disconnect.
///
/// `enqueued_at` is captured once at the first time the assignment lands
/// in the pending pool (or, for a Push that goes straight to a worker,
/// the moment routing was attempted). It is preserved across
/// surrender-on-disconnect so a previously-in-flight job does not lose
/// its FIFO position within its priority band when it is re-queued.
#[derive(Clone)]
struct QueuedAssignment {
    assignment: Assignment,
    /// Tenant/place this assignment is scoped to. Buckets are keyed by
    /// `(place, capability)` so a submission in one place only ever
    /// routes to a worker in the same place.
    place: String,
    required_capability: WorkerCapability,
    required_tags: Vec<String>,
    enqueued_at: Instant,
}

/// In-flight assignment record — tracks which session is running it so
/// we can surrender it back to the pending pool on disconnect.
#[derive(Clone)]
struct InFlight {
    queued: QueuedAssignment,
    session_id: Uuid,
}

/// Owned routing inputs derived from an `Assignment`. Holds the converted
/// `ResourceHint`, `NodeSelector`, `Vec<TolerationSpec>`, and decoded
/// input `Value` so the `RouteRequest` can borrow them without lifetime
/// gymnastics inside the routing helper.
struct RouteInputs {
    hint: Option<ResourceHint>,
    selector: NodeSelector,
    tolerations: Vec<TolerationSpec>,
    input: Option<serde_json::Value>,
}

impl RouteInputs {
    fn from_assignment(assignment: &Assignment) -> Self {
        Self {
            hint: assignment.resource_hint.as_ref().map(Into::into),
            selector: assignment.selector.clone().into(),
            tolerations: assignment
                .tolerations
                .iter()
                .cloned()
                .map(Into::into)
                .collect(),
            input: assignment.input_value().ok(),
        }
    }
}

/// Composite key for the per-capability pending pool:
/// `(priority, enqueued_at, run_id)`. `BTreeMap` orders entries
/// lexicographically by this tuple — priority first (lowest =
/// highest-priority = first served), then `enqueued_at` (FIFO within
/// a priority), then `run_id` (deterministic tiebreak).
type PendingKey = (u8, Instant, Uuid);

/// One per-capability pending pool: priority-ordered map of pending
/// assignments.
type PendingBucket = BTreeMap<PendingKey, QueuedAssignment>;

/// Bucket key for the pending / deficit maps: `(place, capability)`.
/// Keying by place keeps each tenant's queue isolated — a worker that
/// connects for place A only drains place-A buckets.
type BucketKey = (String, WorkerCapability);

/// Per-capability state for the deficit-round-robin WFQ scheduler.
///
/// - `deficit[band]` is the accumulated service credit for that band.
/// - `served_this_round[band]` is `true` once that band has been
///   serviced in the current round; cleared at end-of-round.
/// - A "round" ends and a new top-up happens when every band is
///   either already served-this-round OR has insufficient deficit AND
///   no servable entries (i.e. we've made one full pass without being
///   able to make further progress without a top-up).
struct WfqState {
    deficit: BTreeMap<u8, i32>,
    served_this_round: BTreeMap<u8, bool>,
}

impl Default for WfqState {
    fn default() -> Self {
        let mut deficit = BTreeMap::new();
        let mut served = BTreeMap::new();
        for band in 0u8..PRIORITY_BANDS {
            deficit.insert(band, 0);
            served.insert(band, false);
        }
        Self {
            deficit,
            served_this_round: served,
        }
    }
}

impl WfqState {
    fn top_up(&mut self) {
        for band in 0u8..PRIORITY_BANDS {
            let weight = band_weight(band);
            let d = self.deficit.entry(band).or_insert(0);
            *d = d.saturating_add(weight);
            self.served_this_round.insert(band, false);
        }
    }
}

/// Server-side assignment queue.
///
/// Holds pending assignments waiting on a matching worker, plus the
/// in-flight set so disconnects can surrender work back to the pool.
/// Routing decisions are delegated to [`super::admission::Admission`].
pub struct AssignmentQueue {
    /// Pending submissions keyed by the capability they need.
    ///
    /// Each capability bucket is a `BTreeMap` keyed by
    /// `(priority, enqueued_at, run_id)`. `BTreeMap` orders entries
    /// lexicographically by key tuple: priority first (lowest numeric
    /// value = highest priority = first served), then `enqueued_at`
    /// (FIFO within a priority), then `run_id` (deterministic
    /// tiebreak). Wrapped in `RwLock` so multiple readers (admission
    /// scans) don't serialize behind a writer.
    pending: RwLock<HashMap<BucketKey, PendingBucket>>,
    /// Per-`(place, capability)` deficit counters for the
    /// weighted-fair-queueing (WFQ) scheduler. See [`Self::pop_pending`],
    /// [`band_weight`], and [`band_lo`] for the band math + weight
    /// guarantees.
    deficits: RwLock<HashMap<BucketKey, WfqState>>,
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
            deficits: RwLock::new(HashMap::new()),
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
        place: String,
        assignment: Assignment,
        required_capability: WorkerCapability,
        required_tags: Vec<String>,
        wait_for_worker: bool,
        registry: &WorkerRegistry,
        admission: &Admission,
    ) -> SubmitOutcome {
        let queued = QueuedAssignment {
            assignment,
            place: place.clone(),
            required_capability: required_capability.clone(),
            required_tags: required_tags.clone(),
            enqueued_at: Instant::now(),
        };

        // Record the run as Pending until a worker accepts it.
        self.record_pending(run_id, &queued.assignment).await;

        // Build the routing inputs. The owned ResourceHint / NodeSelector /
        // Vec<TolerationSpec> / Value live for the duration of the borrow
        // inside `RouteRequest`. The block scope ensures those borrows end
        // before any await below.
        let decision = {
            let inputs = RouteInputs::from_assignment(&queued.assignment);
            let req = RouteRequest {
                place: &place,
                required_capability: &required_capability,
                required_tags: &required_tags,
                resource_hint: inputs.hint.as_ref(),
                selector: &inputs.selector,
                tolerations: &inputs.tolerations,
                input: inputs.input.as_ref(),
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
                        | NoCandidateReason::SelectorMismatch
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

    /// A worker has just connected (or freed capacity) in `place`. Drain
    /// any pending assignments in THAT place for capabilities the worker
    /// advertises and try to route them. Other places' buckets are left
    /// untouched — a worker only ever drains its own tenant's queue.
    pub async fn try_drain_for(
        &self,
        place: &str,
        capabilities: &[WorkerCapability],
        registry: &WorkerRegistry,
        admission: &Admission,
    ) {
        for cap in capabilities {
            loop {
                // Pop one entry via the WFQ scheduler; releases all
                // locks before we await on routing below.
                let Some(queued) = self.pop_pending(place, cap).await else {
                    break;
                };

                let decision = {
                    let inputs = RouteInputs::from_assignment(&queued.assignment);
                    let req = RouteRequest {
                        place: &queued.place,
                        required_capability: &queued.required_capability,
                        required_tags: &queued.required_tags,
                        resource_hint: inputs.hint.as_ref(),
                        selector: &inputs.selector,
                        tolerations: &inputs.tolerations,
                        input: inputs.input.as_ref(),
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
            let priority = assignment.priority;
            let enqueued_at = Instant::now();
            let queued = QueuedAssignment {
                assignment,
                // The durable store carries no place; recovered runs land
                // in the default place.
                place: crate::protocol::DEFAULT_PLACE.to_string(),
                required_capability: cap.clone(),
                required_tags: Vec::new(),
                enqueued_at,
            };

            // Populate the in-memory pending pool; admission will pick
            // this up on the next drain. We do NOT mirror the orphan
            // into the store's pending FIFO — the orphan record lives
            // in the inflight set, and if THIS process also crashes
            // before routing the run, the next recovery will
            // re-discover the same orphan and re-queue it again.
            // Mirroring here would double-count and create a hot loop
            // in pass 2 below.
            {
                let mut pending = self.pending.write().await;
                pending
                    .entry((crate::protocol::DEFAULT_PLACE.to_string(), cap.clone()))
                    .or_default()
                    .insert((priority, enqueued_at, run_id), queued);
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
                let priority = assignment.priority;
                let enqueued_at = Instant::now();
                let queued = QueuedAssignment {
                    assignment,
                    // Place-agnostic store → recovered runs land in the
                    // default place.
                    place: crate::protocol::DEFAULT_PLACE.to_string(),
                    required_capability: cap.clone(),
                    required_tags: tags.clone(),
                    enqueued_at,
                };
                {
                    let mut pending = self.pending.write().await;
                    pending
                        .entry((crate::protocol::DEFAULT_PLACE.to_string(), cap.clone()))
                        .or_default()
                        .insert((priority, enqueued_at, run_id), queued);
                }
                if let Err(e) = self.store.enqueue_pending(&cap, run_id, &tags).await {
                    tracing::warn!(%run_id, error = %e, "re-enqueue during recovery");
                }
            }
        }

        Ok(())
    }

    /// Record a freshly-submitted run as `Pending` in the in-memory map
    /// and mirror the state + the wire assignment into the store so a
    /// cold restart can recover both.
    async fn record_pending(&self, run_id: Uuid, assignment: &Assignment) {
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
        if let Err(e) = self.store.put_assignment(run_id, assignment).await {
            tracing::warn!(%run_id, error = %e, "store.put_assignment failed");
        }
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
        let priority = queued.assignment.priority;
        let enqueued_at = queued.enqueued_at;
        let place = queued.place.clone();
        let capability = queued.required_capability.clone();
        let required_tags = queued.required_tags.clone();
        {
            let mut pending = self.pending.write().await;
            pending
                .entry((place, capability.clone()))
                .or_default()
                .insert((priority, enqueued_at, run_id), queued);
        }
        // The durable store is place-agnostic (keyed by capability only);
        // on recovery, persisted pending runs re-hydrate into the default
        // place. Mirror the FIFO position here as before.
        if let Err(e) = self
            .store
            .enqueue_pending(&capability, run_id, &required_tags)
            .await
        {
            tracing::warn!(%run_id, error = %e, "store.enqueue_pending failed");
        }
    }

    /// Pop the next pending assignment for `cap` according to the
    /// weighted-fair-queueing (WFQ) deficit-round-robin (DRR)
    /// scheduler.
    ///
    /// Returns `None` when the capability bucket is empty.
    ///
    /// ### Scheduler semantics
    ///
    /// Each `pop_pending` call returns at most one entry. State per
    /// capability is held in [`WfqState`]: a per-band `deficit` map, a
    /// `cursor` pointing at the next band to consider, and a
    /// `topped_up` flag marking whether the current round has had its
    /// per-band weight credited yet.
    ///
    /// One "round" = exactly one top-up of every band's deficit by its
    /// [`band_weight`]. Within a round we walk bands starting at
    /// `cursor` in priority order. For each band:
    ///
    /// - If the band has at least one entry AND
    ///   `deficit[band] >= SERVICE_COST`, serve the lex-smallest
    ///   `(priority, enqueued_at, run_id)` entry inside that band,
    ///   subtract `SERVICE_COST` from `deficit[band]`, advance the
    ///   cursor, and return. **We advance after serving so a single
    ///   band cannot monopolize even when its deficit would allow
    ///   more services in this round.** This is the anti-starvation
    ///   invariant: every band gets at most floor(deficit/cost)
    ///   services per round, and lower bands accumulate deficit
    ///   across rounds.
    /// - Otherwise advance the cursor and continue.
    ///
    /// When the cursor wraps back to 0 we end the current round and
    /// start a new one (top up again). If a full new round produces
    /// nothing — only possible when the bucket is empty — return
    /// `None`.
    ///
    /// ### Anti-starvation guarantee
    ///
    /// `SERVICE_COST = 256` (the maximum band weight). Band 7 (weight
    /// 32) accumulates 32 deficit per round, so it crosses
    /// `SERVICE_COST` after at most `ceil(SERVICE_COST / 32) = 8`
    /// rounds. Since every other band serves at most one entry per
    /// round, a band-7 entry waits at most ~8 services from other
    /// bands before it wins. This holds regardless of how full the
    /// higher bands are.
    async fn pop_pending(&self, place: &str, cap: &WorkerCapability) -> Option<QueuedAssignment> {
        let bucket_key: BucketKey = (place.to_string(), cap.clone());
        let mut pending = self.pending.write().await;
        let bucket = pending.get_mut(&bucket_key)?;
        if bucket.is_empty() {
            pending.remove(&bucket_key);
            let mut deficits_guard = self.deficits.write().await;
            deficits_guard.remove(&bucket_key);
            return None;
        }

        let mut deficits_guard = self.deficits.write().await;
        let state = deficits_guard.entry(bucket_key.clone()).or_default();

        // Bounded retry: at most PRIORITY_BANDS rounds. Each round
        // either serves (success), or marks at least one band as
        // exhausted (no servable entries) or unservable (deficit too
        // low — promoted on next top-up). Since the bucket is
        // non-empty and band 7 reaches SERVICE_COST after at most 8
        // rounds, we always serve within PRIORITY_BANDS top-ups
        // (PRIORITY_BANDS = 8 = ceil(SERVICE_COST / weight(7))).
        for _round_attempt in 0..PRIORITY_BANDS {
            // Walk bands in priority order. Serve the first band
            // that (a) hasn't been served this round, (b) has
            // deficit >= SERVICE_COST, and (c) has at least one
            // entry.
            let mut served = None;
            for band in 0u8..PRIORITY_BANDS {
                if *state.served_this_round.get(&band).unwrap_or(&false) {
                    continue;
                }
                let d = *state.deficit.get(&band).unwrap_or(&0);
                if d < SERVICE_COST {
                    continue;
                }
                let lo = band_lo(band);
                let hi = band_hi(band);
                let key_opt = bucket
                    .iter()
                    .find(|((p, _, _), _)| *p >= lo && *p <= hi)
                    .map(|(k, _)| *k);
                if let Some(key) = key_opt {
                    served = Some((band, key));
                    break;
                }
                // Band has deficit but no entries — mark served so
                // we don't keep re-checking it in this round.
                state.served_this_round.insert(band, true);
            }

            if let Some((band, key)) = served {
                let queued = bucket.remove(&key)?;
                let d_ref = state.deficit.entry(band).or_insert(0);
                *d_ref = d_ref.saturating_sub(SERVICE_COST);
                state.served_this_round.insert(band, true);
                if bucket.is_empty() {
                    pending.remove(&bucket_key);
                    deficits_guard.remove(&bucket_key);
                }
                return Some(queued);
            }

            // Nothing servable this round: top up and continue. This
            // either credits more deficit to bands that were below
            // threshold OR resets served_this_round so a band that
            // was "spent" this round becomes eligible next round.
            state.top_up();
        }

        None
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

/// Number of priority bands. The `u8` priority space (0..=255) is
/// partitioned into 8 contiguous bands of width [`PRIORITY_BAND_WIDTH`]
/// (32). Band 0 covers priorities 0..=31 (highest), band 7 covers
/// 224..=255 (lowest).
///
/// Bands are 32 wide so a user picking a priority in the natural 0..=255
/// range crosses a band boundary every 32 steps — coarse enough that
/// "I picked priority 100, my friend picked 110" lands in the same band
/// (same weight class), but fine enough that distinct priority *tiers*
/// are honored (priority 0 vs 100 vs 200 are visibly different bands).
const PRIORITY_BANDS: u8 = 8;
const PRIORITY_BAND_WIDTH: u8 = 32;

/// Per-serve deficit decrement, equal to the maximum band weight. With
/// `SERVICE_COST = MAX_WEIGHT`, band 0 (weight 256) services exactly
/// one entry per round, while band 7 (weight 32) services one entry
/// per 8 rounds. This guarantees:
///
/// 1. **Anti-starvation.** Even a steady stream of band-0 work cannot
///    keep band-7 work waiting forever — band 7 wins after at most
///    `ceil(MAX_WEIGHT / weight(7)) = 8` rounds.
/// 2. **Priority is honored.** Across any sustained mixed workload,
///    band 0 receives 8x the service rate of band 7.
const SERVICE_COST: i32 = 256;

/// Weight of a priority band: band 0 = 256, band 1 = 224, …, band 7 = 32.
///
/// Strictly decreasing so higher-priority bands are serviced more often
/// per round. The minimum weight (32) is non-zero, which is what
/// prevents the lowest band from starving.
fn band_weight(band: u8) -> i32 {
    256_i32 - i32::from(band) * i32::from(PRIORITY_BAND_WIDTH)
}

/// Inclusive lower bound of priorities in `band`.
fn band_lo(band: u8) -> u8 {
    band.saturating_mul(PRIORITY_BAND_WIDTH)
}

/// Inclusive upper bound of priorities in `band`.
fn band_hi(band: u8) -> u8 {
    band.saturating_mul(PRIORITY_BAND_WIDTH)
        .saturating_add(PRIORITY_BAND_WIDTH - 1)
}

/// Map a raw `u8` priority to its band index in `0..PRIORITY_BANDS`.
///
/// Exposed for the `priority_band_math_buckets_correctly` test; the
/// scheduler itself queries band membership via `band_lo` / `band_hi`
/// directly to avoid an extra division on the hot path.
#[cfg(test)]
fn priority_band(priority: u8) -> u8 {
    priority / PRIORITY_BAND_WIDTH
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
            priority: blazen_core::distributed::DEFAULT_PRIORITY,
            selector: crate::protocol::NodeSelectorWire::default(),
            tolerations: Vec::new(),
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
            "__default__".into(),
            caps,
            BTreeMap::new(),
            AdmissionMode::Fixed { max_in_flight: 4 },
            tx,
            BTreeMap::new(),
            Vec::new(),
            Vec::new(),
        );

        let run_id = Uuid::new_v4();
        let outcome = queue
            .submit(
                run_id,
                "__default__".into(),
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
                "__default__".into(),
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
                "__default__".into(),
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
                "__default__".into(),
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
            "__default__".into(),
            caps,
            BTreeMap::new(),
            AdmissionMode::Fixed { max_in_flight: 4 },
            tx,
            BTreeMap::new(),
            Vec::new(),
            Vec::new(),
        );
        let caps_vec = vec![cap("workflow:hello", 1)];
        queue
            .try_drain_for("__default__", &caps_vec, &registry, &admission)
            .await;

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
            "__default__".into(),
            caps,
            BTreeMap::new(),
            AdmissionMode::Fixed { max_in_flight: 4 },
            tx,
            BTreeMap::new(),
            Vec::new(),
            Vec::new(),
        );

        let run_id = Uuid::new_v4();
        let _ = queue
            .submit(
                run_id,
                "__default__".into(),
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

    fn make_assignment_p(run_id: Uuid, name: &str, priority: u8) -> Assignment {
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
            priority,
            selector: crate::protocol::NodeSelectorWire::default(),
            tolerations: Vec::new(),
        }
    }

    /// Enqueue a fresh assignment directly into the pending pool with
    /// the given priority and capability. Returns the `run_id`.
    async fn enqueue_for_test(
        queue: &AssignmentQueue,
        capability: WorkerCapability,
        priority: u8,
    ) -> Uuid {
        let run_id = Uuid::new_v4();
        let queued = QueuedAssignment {
            assignment: make_assignment_p(run_id, &capability.kind, priority),
            place: "__default__".to_string(),
            required_capability: capability.clone(),
            required_tags: Vec::new(),
            enqueued_at: Instant::now(),
        };
        queue.enqueue_pending(queued).await;
        run_id
    }

    #[test]
    fn priority_band_math_buckets_correctly() {
        // Boundaries: each band is 32 wide.
        assert_eq!(priority_band(0), 0);
        assert_eq!(priority_band(31), 0);
        assert_eq!(priority_band(32), 1);
        assert_eq!(priority_band(63), 1);
        assert_eq!(priority_band(64), 2);
        assert_eq!(priority_band(128), 4); // DEFAULT_PRIORITY
        assert_eq!(priority_band(223), 6);
        assert_eq!(priority_band(224), 7);
        assert_eq!(priority_band(255), 7);

        // Band lo/hi cover the full priority space without gaps or overlap.
        for band in 0u8..PRIORITY_BANDS {
            let lo = band_lo(band);
            let hi = band_hi(band);
            assert_eq!(lo, band * 32);
            assert_eq!(hi, band * 32 + 31);
            for p in lo..=hi {
                assert_eq!(priority_band(p), band);
            }
        }

        // Weights: 256, 224, 192, 160, 128, 96, 64, 32.
        let weights: Vec<i32> = (0u8..PRIORITY_BANDS).map(band_weight).collect();
        assert_eq!(weights, vec![256, 224, 192, 160, 128, 96, 64, 32]);
        // Lowest band weight is strictly positive (anti-starvation invariant).
        assert!(band_weight(PRIORITY_BANDS - 1) > 0);
    }

    #[tokio::test]
    async fn queue_orders_by_priority_then_fifo() {
        let queue = AssignmentQueue::new();
        let capability = cap("workflow:hello", 1);

        // Submit three: pri=100 (A), pri=50 (B), pri=100 (C), in that
        // order. Expected pop order: B (pri 50), then A, then C
        // (FIFO within priority 100).
        let id_a = enqueue_for_test(&queue, capability.clone(), 100).await;
        // Force a perceptible gap so enqueued_at differs.
        tokio::time::sleep(std::time::Duration::from_millis(2)).await;
        let id_b = enqueue_for_test(&queue, capability.clone(), 50).await;
        tokio::time::sleep(std::time::Duration::from_millis(2)).await;
        let id_c = enqueue_for_test(&queue, capability.clone(), 100).await;

        let first = queue.pop_pending("__default__", &capability).await.unwrap();
        assert_eq!(first.assignment.run_id, id_b, "pri-50 must come first");

        let second = queue.pop_pending("__default__", &capability).await.unwrap();
        assert_eq!(
            second.assignment.run_id, id_a,
            "earliest pri-100 next (FIFO within priority)"
        );

        let third = queue.pop_pending("__default__", &capability).await.unwrap();
        assert_eq!(third.assignment.run_id, id_c, "later pri-100 last");

        assert!(
            queue
                .pop_pending("__default__", &capability)
                .await
                .is_none()
        );
    }

    #[tokio::test]
    async fn wfq_does_not_starve_low_priority_under_steady_high() {
        // Bound rationale: weights are 256, 224, …, 32. The lowest
        // band (band 7, weight 32) accrues 32 deficit per round. With
        // `SERVICE_COST = 256`, band 7 wins after at most
        // `ceil(SERVICE_COST / 32) = 8` rounds. Each preceding round
        // services at most one band-0 entry (deficit math; band 0 also
        // costs SERVICE_COST per serve). So the lowest-priority entry
        // is returned within roughly 8 pops. We pad to 16 for safety
        // against off-by-one accumulation effects from the priming
        // round. The spec's "within first 50 pops" is a much looser
        // bound; we choose 16 to assert the algorithm actually works.
        const STARVATION_BOUND: usize = 16;

        let queue = AssignmentQueue::new();
        let capability = cap("workflow:hello", 1);

        // 100 priority-0 jobs.
        for _ in 0..100 {
            enqueue_for_test(&queue, capability.clone(), 0).await;
        }
        // Then 1 priority-255 job.
        let low_id = enqueue_for_test(&queue, capability.clone(), 255).await;

        let mut low_position = None;
        for i in 0..110 {
            let item = queue.pop_pending("__default__", &capability).await.unwrap();
            if item.assignment.run_id == low_id {
                low_position = Some(i);
                break;
            }
        }

        let pos = low_position.expect("priority-255 entry was never served");
        assert!(
            pos < STARVATION_BOUND,
            "priority-255 entry returned at pop {pos}, expected < {STARVATION_BOUND}"
        );
    }

    #[tokio::test]
    async fn surrender_preserves_priority_and_enqueued_at() {
        let registry = WorkerRegistry::new();
        let admission = Admission::new();
        let queue = AssignmentQueue::new();

        let (tx, mut _rx) = mpsc::channel(8);
        let mut caps = HashSet::new();
        caps.insert(cap("workflow:hello", 1));
        let sid = registry.register(
            "node-a".into(),
            "__default__".into(),
            caps,
            BTreeMap::new(),
            AdmissionMode::Fixed { max_in_flight: 4 },
            tx,
            BTreeMap::new(),
            Vec::new(),
            Vec::new(),
        );

        let run_id = Uuid::new_v4();
        let outcome = queue
            .submit(
                run_id,
                "__default__".into(),
                make_assignment_p(run_id, "workflow:hello", 50),
                cap("workflow:hello", 1),
                vec![],
                false,
                &registry,
                &admission,
            )
            .await;
        assert!(matches!(outcome, SubmitOutcome::Pushed { .. }));

        // Capture the original enqueued_at from the in-flight record.
        let original_enqueued_at = queue
            .in_flight
            .get(&run_id)
            .map(|r| r.value().queued.enqueued_at)
            .expect("in-flight record present after Push");

        // Surrender, then verify the re-queued entry is the very next
        // pop (the queue is otherwise empty) AND that its enqueued_at
        // is unchanged.
        queue.surrender_session(sid).await;

        let capability = cap("workflow:hello", 1);
        let popped = queue
            .pop_pending("__default__", &capability)
            .await
            .expect("surrendered entry is back in pending");
        assert_eq!(popped.assignment.run_id, run_id);
        assert_eq!(popped.assignment.priority, 50);
        assert_eq!(
            popped.enqueued_at, original_enqueued_at,
            "enqueued_at must be preserved across surrender"
        );
    }
}
