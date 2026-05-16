//! Persistence seam for [`super::queue::AssignmentQueue`].
//!
//! Phase 6 of the distributed control plane introduces a swappable
//! backing store for the three pieces of queue state previously held in
//! process-local `DashMap`s:
//!
//! 1. **Assignments** — the canonical wire form of every routed run,
//!    keyed by `run_id`. Needed so a freshly-restarted server can hand
//!    a re-discovered run back to a worker without the orchestrator
//!    re-submitting it.
//! 2. **Run-state snapshots** — the live `RunStateSnapshot` exposed via
//!    `describe` queries. Updated through every status transition.
//! 3. **Per-capability pending FIFOs** — submissions that arrived with
//!    `wait_for_worker = true` while no matching worker was connected.
//! 4. **In-flight ownership** — which `session_id` currently owns each
//!    `run_id`. Used on disconnect / cold start to surrender orphaned
//!    work back to the pending pool.
//!
//! The [`AssignmentStore`] trait captures exactly the operations the
//! queue needs. [`MemoryAssignmentStore`] is the default — a pure
//! in-memory implementation with the same semantics the direct-DashMap
//! queue had before this phase. A Valkey-backed implementation lives in
//! a sibling file behind the `valkey-store` feature gate and is added
//! in Phase 6C; this module is shape-only for the trait + memory impl.

use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use dashmap::DashMap;
use tokio::sync::RwLock;
use uuid::Uuid;

use blazen_core::distributed::{RunStateSnapshot, WorkerCapability};

use crate::error::ControlPlaneError;
use crate::protocol::Assignment;

/// Persistence seam for [`super::queue::AssignmentQueue`]. Every queue
/// mutator writes through this trait so the in-memory cache stays in
/// sync with whatever durable backing store (or no backing store, for
/// [`MemoryAssignmentStore`]) is wired in.
///
/// Implementations are `Send + Sync + std::fmt::Debug` so the queue can
/// hold an `Arc<dyn AssignmentStore>`.
#[async_trait]
pub trait AssignmentStore: Send + Sync + std::fmt::Debug {
    /// Persist (or overwrite) the canonical wire form of an
    /// [`Assignment`] keyed by `run_id`.
    ///
    /// # Errors
    /// Returns a [`ControlPlaneError`] when the underlying store cannot
    /// be reached or rejects the write.
    async fn put_assignment(
        &self,
        run_id: Uuid,
        assignment: &Assignment,
    ) -> Result<(), ControlPlaneError>;

    /// Look up a previously-stored assignment by `run_id`.
    ///
    /// # Errors
    /// Returns a [`ControlPlaneError`] when the underlying store cannot
    /// be reached or returns a malformed payload.
    async fn get_assignment(&self, run_id: Uuid) -> Result<Option<Assignment>, ControlPlaneError>;

    /// Remove an assignment record. Safe to call on missing keys.
    ///
    /// # Errors
    /// Returns a [`ControlPlaneError`] when the underlying store cannot
    /// be reached.
    async fn delete_assignment(&self, run_id: Uuid) -> Result<(), ControlPlaneError>;

    /// Persist (or overwrite) the run-state snapshot. When
    /// `ttl_after_terminal` is `Some` and the snapshot is in a terminal
    /// state, implementations MAY apply that TTL so completed runs are
    /// eventually evicted; in-memory implementations may ignore it.
    ///
    /// # Errors
    /// Returns a [`ControlPlaneError`] when the underlying store cannot
    /// be reached or rejects the write.
    async fn put_state(
        &self,
        run_id: Uuid,
        state: &RunStateSnapshot,
        ttl_after_terminal: Option<Duration>,
    ) -> Result<(), ControlPlaneError>;

    /// Look up a run-state snapshot.
    ///
    /// # Errors
    /// Returns a [`ControlPlaneError`] when the underlying store cannot
    /// be reached or returns a malformed payload.
    async fn get_state(&self, run_id: Uuid) -> Result<Option<RunStateSnapshot>, ControlPlaneError>;

    /// Push `run_id` (with its `required_tags`) onto the FIFO for
    /// `capability`. Used when a submit can't be routed immediately
    /// because no worker advertises the requested capability.
    ///
    /// # Errors
    /// Returns a [`ControlPlaneError`] when the underlying store cannot
    /// be reached or rejects the write.
    async fn enqueue_pending(
        &self,
        capability: &WorkerCapability,
        run_id: Uuid,
        required_tags: &[String],
    ) -> Result<(), ControlPlaneError>;

    /// Pop the head of `capability`'s pending FIFO. Returns `None` if
    /// the queue is empty.
    ///
    /// # Errors
    /// Returns a [`ControlPlaneError`] when the underlying store cannot
    /// be reached or returns a malformed payload.
    async fn dequeue_pending(
        &self,
        capability: &WorkerCapability,
    ) -> Result<Option<(Uuid, Vec<String>)>, ControlPlaneError>;

    /// Enumerate every capability whose pending FIFO is non-empty.
    /// Used by `ControlPlaneServer::serve` startup recovery and by
    /// drain loops.
    ///
    /// # Errors
    /// Returns a [`ControlPlaneError`] when the underlying store cannot
    /// be reached or returns a malformed payload.
    async fn list_pending_capabilities(&self) -> Result<Vec<WorkerCapability>, ControlPlaneError>;

    /// Claim that `session_id` is currently running `run_id`.
    ///
    /// # Errors
    /// Returns a [`ControlPlaneError`] when the underlying store cannot
    /// be reached or rejects the write.
    async fn claim_inflight(&self, run_id: Uuid, session_id: Uuid)
    -> Result<(), ControlPlaneError>;

    /// Drop every claim held by `session_id` and return the affected
    /// `run_id`s. Called when a worker disconnects.
    ///
    /// # Errors
    /// Returns a [`ControlPlaneError`] when the underlying store cannot
    /// be reached or rejects the write.
    async fn release_inflight(&self, session_id: Uuid) -> Result<Vec<Uuid>, ControlPlaneError>;

    /// Return every in-flight `run_id` whose claiming session is NOT in
    /// `alive_sessions`. Used by `ControlPlaneServer::serve` startup
    /// recovery to re-queue work owned by sessions that died with the
    /// previous server process (alive set is empty on cold start).
    ///
    /// # Errors
    /// Returns a [`ControlPlaneError`] when the underlying store cannot
    /// be reached or returns a malformed payload.
    async fn list_orphaned_inflight(
        &self,
        alive_sessions: &HashSet<Uuid>,
    ) -> Result<Vec<Uuid>, ControlPlaneError>;
}

/// A single entry in a per-capability pending FIFO: the queued
/// `run_id` plus the tag predicate it required at submit time.
type PendingEntry = (Uuid, Vec<String>);

/// Per-capability pending FIFO map used by [`MemoryAssignmentStore`].
type PendingMap = HashMap<WorkerCapability, VecDeque<PendingEntry>>;

/// In-memory, zero-overhead [`AssignmentStore`]. Used by default; no
/// persistence — all state vanishes on process exit. The TTL on
/// [`AssignmentStore::put_state`] is intentionally ignored: the memory
/// store has no out-of-band eviction loop, so completed runs accumulate
/// until [`AssignmentStore::delete_assignment`] / process restart.
#[derive(Default, Debug)]
pub struct MemoryAssignmentStore {
    assignments: DashMap<Uuid, Assignment>,
    states: DashMap<Uuid, RunStateSnapshot>,
    pending: RwLock<PendingMap>,
    /// `run_id` -> `session_id`.
    inflight: DashMap<Uuid, Uuid>,
}

impl MemoryAssignmentStore {
    /// Construct a fresh, empty in-memory store.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }
}

#[async_trait]
impl AssignmentStore for MemoryAssignmentStore {
    async fn put_assignment(
        &self,
        run_id: Uuid,
        assignment: &Assignment,
    ) -> Result<(), ControlPlaneError> {
        self.assignments.insert(run_id, assignment.clone());
        Ok(())
    }

    async fn get_assignment(&self, run_id: Uuid) -> Result<Option<Assignment>, ControlPlaneError> {
        Ok(self.assignments.get(&run_id).map(|r| r.clone()))
    }

    async fn delete_assignment(&self, run_id: Uuid) -> Result<(), ControlPlaneError> {
        self.assignments.remove(&run_id);
        Ok(())
    }

    async fn put_state(
        &self,
        run_id: Uuid,
        state: &RunStateSnapshot,
        _ttl_after_terminal: Option<Duration>,
    ) -> Result<(), ControlPlaneError> {
        self.states.insert(run_id, state.clone());
        Ok(())
    }

    async fn get_state(&self, run_id: Uuid) -> Result<Option<RunStateSnapshot>, ControlPlaneError> {
        Ok(self.states.get(&run_id).map(|r| r.clone()))
    }

    async fn enqueue_pending(
        &self,
        capability: &WorkerCapability,
        run_id: Uuid,
        required_tags: &[String],
    ) -> Result<(), ControlPlaneError> {
        let mut pending = self.pending.write().await;
        pending
            .entry(capability.clone())
            .or_default()
            .push_back((run_id, required_tags.to_vec()));
        Ok(())
    }

    async fn dequeue_pending(
        &self,
        capability: &WorkerCapability,
    ) -> Result<Option<(Uuid, Vec<String>)>, ControlPlaneError> {
        let mut pending = self.pending.write().await;
        let entry = pending.get_mut(capability).and_then(VecDeque::pop_front);
        Ok(entry)
    }

    async fn list_pending_capabilities(&self) -> Result<Vec<WorkerCapability>, ControlPlaneError> {
        let pending = self.pending.read().await;
        Ok(pending
            .iter()
            .filter(|(_, q)| !q.is_empty())
            .map(|(k, _)| k.clone())
            .collect())
    }

    async fn claim_inflight(
        &self,
        run_id: Uuid,
        session_id: Uuid,
    ) -> Result<(), ControlPlaneError> {
        self.inflight.insert(run_id, session_id);
        Ok(())
    }

    async fn release_inflight(&self, session_id: Uuid) -> Result<Vec<Uuid>, ControlPlaneError> {
        let affected: Vec<Uuid> = self
            .inflight
            .iter()
            .filter(|r| *r.value() == session_id)
            .map(|r| *r.key())
            .collect();
        for id in &affected {
            self.inflight.remove(id);
        }
        Ok(affected)
    }

    async fn list_orphaned_inflight(
        &self,
        alive_sessions: &HashSet<Uuid>,
    ) -> Result<Vec<Uuid>, ControlPlaneError> {
        Ok(self
            .inflight
            .iter()
            .filter(|r| !alive_sessions.contains(r.value()))
            .map(|r| *r.key())
            .collect())
    }
}

/// Construct a default in-memory store wrapped in `Arc<dyn AssignmentStore>`
/// for callers that want the trait object directly.
#[must_use]
pub fn default_store() -> Arc<dyn AssignmentStore> {
    Arc::new(MemoryAssignmentStore::default())
}

#[cfg(test)]
mod tests {
    use super::*;

    use blazen_core::distributed::RunStatus;

    use crate::protocol::ENVELOPE_VERSION;

    fn cap(kind: &str, version: u32) -> WorkerCapability {
        WorkerCapability {
            kind: kind.to_string(),
            version,
        }
    }

    fn make_assignment(run_id: Uuid, workflow_name: &str) -> Assignment {
        Assignment {
            envelope_version: ENVELOPE_VERSION,
            run_id,
            parent_run_id: None,
            workflow_name: workflow_name.to_string(),
            workflow_version: None,
            input_json: b"{}".to_vec(),
            deadline_ms: None,
            attempt: 0,
            resource_hint: None,
        }
    }

    fn make_state(run_id: Uuid, status: RunStatus) -> RunStateSnapshot {
        RunStateSnapshot {
            run_id,
            status,
            started_at_ms: 0,
            completed_at_ms: None,
            assigned_to: None,
            last_event_at_ms: None,
            output: None,
            error: None,
        }
    }

    #[tokio::test]
    async fn assignment_put_get_delete() {
        let store = MemoryAssignmentStore::new();
        let run_id = Uuid::new_v4();
        let assignment = make_assignment(run_id, "workflow:hello");

        store
            .put_assignment(run_id, &assignment)
            .await
            .expect("put");
        let fetched = store.get_assignment(run_id).await.expect("get");
        let fetched = fetched.expect("Some(assignment)");
        assert_eq!(fetched.run_id, run_id);
        assert_eq!(fetched.workflow_name, "workflow:hello");

        store.delete_assignment(run_id).await.expect("delete");
        let missing = store
            .get_assignment(run_id)
            .await
            .expect("get after delete");
        assert!(missing.is_none());
    }

    #[tokio::test]
    async fn state_put_get() {
        let store = MemoryAssignmentStore::new();
        let run_id = Uuid::new_v4();
        let snap = make_state(run_id, RunStatus::Running);

        store.put_state(run_id, &snap, None).await.expect("put");
        let fetched = store
            .get_state(run_id)
            .await
            .expect("get")
            .expect("Some(snapshot)");
        assert_eq!(fetched.run_id, run_id);
        assert_eq!(fetched.status, RunStatus::Running);
    }

    #[tokio::test]
    async fn pending_fifo_order() {
        let store = MemoryAssignmentStore::new();
        let capability = cap("workflow:hello", 1);
        let id1 = Uuid::new_v4();
        let id2 = Uuid::new_v4();
        let id3 = Uuid::new_v4();

        store
            .enqueue_pending(&capability, id1, &["tag=a".to_string()])
            .await
            .expect("enqueue 1");
        store
            .enqueue_pending(&capability, id2, &["tag=b".to_string()])
            .await
            .expect("enqueue 2");
        store
            .enqueue_pending(&capability, id3, &["tag=c".to_string()])
            .await
            .expect("enqueue 3");

        let first = store
            .dequeue_pending(&capability)
            .await
            .expect("dequeue 1")
            .expect("Some");
        let second = store
            .dequeue_pending(&capability)
            .await
            .expect("dequeue 2")
            .expect("Some");
        let third = store
            .dequeue_pending(&capability)
            .await
            .expect("dequeue 3")
            .expect("Some");
        let fourth = store.dequeue_pending(&capability).await.expect("dequeue 4");

        assert_eq!(first, (id1, vec!["tag=a".to_string()]));
        assert_eq!(second, (id2, vec!["tag=b".to_string()]));
        assert_eq!(third, (id3, vec!["tag=c".to_string()]));
        assert!(fourth.is_none());
    }

    #[tokio::test]
    async fn pending_per_capability() {
        let store = MemoryAssignmentStore::new();
        let cap_a = cap("workflow:a", 1);
        let cap_b = cap("workflow:b", 1);
        let id_a = Uuid::new_v4();
        let id_b = Uuid::new_v4();

        store
            .enqueue_pending(&cap_a, id_a, &[])
            .await
            .expect("enqueue a");
        store
            .enqueue_pending(&cap_b, id_b, &[])
            .await
            .expect("enqueue b");

        let mut caps = store
            .list_pending_capabilities()
            .await
            .expect("list capabilities");
        caps.sort_by(|l, r| l.kind.cmp(&r.kind));
        assert_eq!(caps, vec![cap_a.clone(), cap_b.clone()]);

        // Draining cap_a must not affect cap_b.
        let drained = store
            .dequeue_pending(&cap_a)
            .await
            .expect("dequeue a")
            .expect("Some");
        assert_eq!(drained.0, id_a);

        let mut still = store
            .list_pending_capabilities()
            .await
            .expect("list after drain");
        still.sort_by(|l, r| l.kind.cmp(&r.kind));
        assert_eq!(still, vec![cap_b.clone()]);

        let remaining = store
            .dequeue_pending(&cap_b)
            .await
            .expect("dequeue b")
            .expect("Some");
        assert_eq!(remaining.0, id_b);
    }

    #[tokio::test]
    async fn inflight_claim_release_round_trip() {
        let store = MemoryAssignmentStore::new();
        let session = Uuid::new_v4();
        let r1 = Uuid::new_v4();
        let r2 = Uuid::new_v4();
        let r3 = Uuid::new_v4();

        store.claim_inflight(r1, session).await.expect("claim 1");
        store.claim_inflight(r2, session).await.expect("claim 2");
        store.claim_inflight(r3, session).await.expect("claim 3");

        let mut released = store.release_inflight(session).await.expect("release");
        released.sort();
        let mut expected = vec![r1, r2, r3];
        expected.sort();
        assert_eq!(released, expected);

        let again = store
            .release_inflight(session)
            .await
            .expect("release again");
        assert!(again.is_empty());
    }

    #[tokio::test]
    async fn orphaned_inflight_filters_by_alive_set() {
        let store = MemoryAssignmentStore::new();
        let live_session = Uuid::new_v4();
        let dead_session = Uuid::new_v4();
        let live_run_a = Uuid::new_v4();
        let live_run_b = Uuid::new_v4();
        let orphan_run = Uuid::new_v4();

        store
            .claim_inflight(live_run_a, live_session)
            .await
            .expect("claim live a");
        store
            .claim_inflight(live_run_b, live_session)
            .await
            .expect("claim live b");
        store
            .claim_inflight(orphan_run, dead_session)
            .await
            .expect("claim orphan");

        let mut alive = HashSet::new();
        alive.insert(live_session);

        let orphans = store
            .list_orphaned_inflight(&alive)
            .await
            .expect("list orphans");
        assert_eq!(orphans, vec![orphan_run]);
    }
}
