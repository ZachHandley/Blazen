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

use dashmap::DashMap;
use tokio::sync::{Notify, RwLock};
use uuid::Uuid;

use blazen_core::distributed::{ResourceHint, RunStateSnapshot, RunStatus, WorkerCapability};

use crate::error::ControlPlaneError;
use crate::protocol::{Assignment, Offer, ServerToWorker};

use super::admission::{Admission, Decision, NoCandidateReason, RouteRequest};
use super::registry::WorkerRegistry;

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
    /// Construct an empty queue.
    #[must_use]
    pub fn new() -> Self {
        Self {
            pending: RwLock::new(HashMap::new()),
            in_flight: DashMap::new(),
            runs: DashMap::new(),
            wake: Notify::new(),
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
        self.runs.insert(
            run_id,
            RunStateSnapshot {
                run_id,
                status: RunStatus::Pending,
                started_at_ms: now_ms(),
                completed_at_ms: None,
                assigned_to: None,
                last_event_at_ms: None,
                output: None,
                error: None,
            },
        );

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
                    self.in_flight
                        .insert(run_id, InFlight { queued, session_id });
                    self.update_run_to_running(run_id, &node_id);
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
                    self.in_flight
                        .insert(run_id, InFlight { queued, session_id });
                    self.update_run_to_running(run_id, &node_id);
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
                        self.in_flight
                            .insert(run_id, InFlight { queued, session_id });
                        self.update_run_to_running(run_id, &node_id);
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
                        self.in_flight
                            .insert(run_id, InFlight { queued, session_id });
                        self.update_run_to_running(run_id, &node_id);
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

        // Remove the in-flight entries and reset their run state.
        let surrendered_ids: Vec<Uuid> = surrendered.iter().map(|q| q.assignment.run_id).collect();
        for id in surrendered_ids {
            self.in_flight.remove(&id);
            if let Some(mut snap) = self.runs.get_mut(&id) {
                snap.status = RunStatus::Pending;
                snap.assigned_to = None;
            }
        }

        for queued in surrendered {
            self.enqueue_pending(queued).await;
        }
        self.wake.notify_waiters();
    }

    /// Mark a run as completed with the given output.
    pub fn mark_completed(&self, run_id: Uuid, output: serde_json::Value) {
        if let Some(mut snap) = self.runs.get_mut(&run_id) {
            snap.status = RunStatus::Completed;
            snap.completed_at_ms = Some(now_ms());
            snap.output = Some(output);
        }
        self.in_flight.remove(&run_id);
    }

    /// Mark a run as failed with the given error message.
    pub fn mark_failed(&self, run_id: Uuid, error: String) {
        if let Some(mut snap) = self.runs.get_mut(&run_id) {
            snap.status = RunStatus::Failed;
            snap.completed_at_ms = Some(now_ms());
            snap.error = Some(error);
        }
        self.in_flight.remove(&run_id);
    }

    /// Mark a run as cancelled.
    pub fn mark_cancelled(&self, run_id: Uuid) {
        if let Some(mut snap) = self.runs.get_mut(&run_id) {
            snap.status = RunStatus::Cancelled;
            snap.completed_at_ms = Some(now_ms());
        }
        self.in_flight.remove(&run_id);
    }

    /// Record an event timestamp for a run. Used to populate
    /// `RunStateSnapshot.last_event_at_ms`.
    pub fn record_event(&self, run_id: Uuid) {
        if let Some(mut snap) = self.runs.get_mut(&run_id) {
            snap.last_event_at_ms = Some(now_ms());
        }
    }

    /// Look up a run's current state.
    #[must_use]
    pub fn describe(&self, run_id: Uuid) -> Option<RunStateSnapshot> {
        self.runs.get(&run_id).map(|r| r.clone())
    }

    async fn enqueue_pending(&self, queued: QueuedAssignment) {
        let mut pending = self.pending.write().await;
        pending
            .entry(queued.required_capability.clone())
            .or_default()
            .push_back(queued);
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

    fn update_run_to_running(&self, run_id: Uuid, node_id: &str) {
        if let Some(mut snap) = self.runs.get_mut(&run_id) {
            snap.status = RunStatus::Running;
            snap.assigned_to = Some(node_id.to_string());
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

        queue.mark_completed(run_id, serde_json::json!({"ok": true}));
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
        queue.mark_failed(id2, "boom".into());
        let s2 = queue.describe(id2).unwrap();
        assert_eq!(s2.status, RunStatus::Failed);
        assert_eq!(s2.error.as_deref(), Some("boom"));
    }
}
