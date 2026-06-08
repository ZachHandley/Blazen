//! Connected-worker registry with capability and tag indexes.
//!
//! The registry is the source of truth for which workers are currently
//! connected, what they can do, and how to reach them. The bidi-session
//! handler ([`super::session`]) writes to it on connect / disconnect;
//! the queue ([`super::queue`]) and admission logic ([`super::admission`])
//! read from it to make routing decisions.

use std::collections::{BTreeMap, HashSet};

use dashmap::DashMap;
use tokio::sync::mpsc;
use uuid::Uuid;

use blazen_core::distributed::{
    AdmissionMode, AdmissionSnapshot, WorkerCapability, WorkerInfo, WorkerTaint,
};

use crate::protocol::{NodeDescriptorWire, ServerToWorker};

/// One connected worker.
#[derive(Clone)]
pub struct WorkerHandle {
    pub session_id: Uuid,
    pub node_id: String,
    /// Tenant/place this worker serves. Determined server-side (the
    /// bearer-derived identity wins over the worker's self-reported
    /// [`crate::protocol::WorkerHello::place`]). `"__default__"` for
    /// single-tenant / legacy deployments.
    pub place: String,
    pub capabilities: HashSet<WorkerCapability>,
    pub tags: BTreeMap<String, String>,
    pub admission: AdmissionMode,
    /// Outbound channel — the bidi session task forwards anything sent
    /// here into the gRPC stream towards the worker.
    pub outbound: mpsc::Sender<ServerToWorker>,
    /// Latest heartbeat snapshot (Reactive/VramBudget only).
    pub admission_snapshot: Option<AdmissionSnapshot>,
    /// Last reported in-flight count.
    pub in_flight: u32,
    pub connected_at_ms: u64,
    /// Worker-side scheduling labels declared in [`WorkerHello::labels`].
    /// Filtered against [`Assignment::selector`] in admission. Empty for
    /// legacy v1 workers.
    pub labels: BTreeMap<String, String>,
    /// Worker-side taints. Jobs must carry a matching toleration to land
    /// here. Empty for legacy v1 workers.
    pub taints: Vec<WorkerTaint>,
    /// Capability-descriptor manifest the worker published in
    /// [`crate::protocol::WorkerHello::descriptors`]. Empty for legacy
    /// callers that don't publish a catalogue.
    pub descriptors: Vec<NodeDescriptorWire>,
}

/// Registry of currently-connected workers.
///
/// All operations are O(1) average via dashmap; the capability index is
/// rebuilt incrementally on register / unregister.
pub struct WorkerRegistry {
    /// Primary store: `session_id` → handle.
    sessions: DashMap<Uuid, WorkerHandle>,
    /// `node_id` → `session_id`, for `DrainWorker` and admin lookups.
    by_node_id: DashMap<String, Uuid>,
    /// `(place, capability)` → set of `session_ids` advertising it within
    /// that place. Keyed by place so a submission in one tenant never
    /// routes to a worker in another. Used by the queue to find candidate
    /// workers for a submitted workflow.
    capability_index: DashMap<(String, WorkerCapability), HashSet<Uuid>>,
    /// `place` → set of `session_ids` serving it. Powers
    /// [`Self::list_by_place`] and place-scoped cleanup on unregister.
    by_place: DashMap<String, HashSet<Uuid>>,
}

impl WorkerRegistry {
    #[must_use]
    pub fn new() -> Self {
        Self {
            sessions: DashMap::new(),
            by_node_id: DashMap::new(),
            capability_index: DashMap::new(),
            by_place: DashMap::new(),
        }
    }

    /// Register a freshly-connected worker. Returns the assigned
    /// `session_id`. If a worker with the same `node_id` is already
    /// registered, its session is *replaced* (the old session's
    /// outbound channel is dropped, which the old session task uses as
    /// a shutdown signal).
    #[must_use = "the returned session_id is the only handle to the registered worker"]
    #[allow(clippy::too_many_arguments)]
    pub fn register(
        &self,
        node_id: String,
        place: String,
        capabilities: HashSet<WorkerCapability>,
        tags: BTreeMap<String, String>,
        admission: AdmissionMode,
        outbound: mpsc::Sender<ServerToWorker>,
        labels: BTreeMap<String, String>,
        taints: Vec<WorkerTaint>,
        descriptors: Vec<NodeDescriptorWire>,
    ) -> Uuid {
        let session_id = Uuid::new_v4();
        let connected_at_ms = now_ms();

        // Evict any prior session for this node_id.
        if let Some(prior) = self.by_node_id.get(&node_id).map(|r| *r) {
            self.unregister(prior);
        }

        // Insert into the place-scoped capability index first using clones,
        // then move the originals into the handle — saves us cloning the
        // whole set just to satisfy ownership.
        for cap in &capabilities {
            self.capability_index
                .entry((place.clone(), cap.clone()))
                .or_default()
                .insert(session_id);
        }
        self.by_place
            .entry(place.clone())
            .or_default()
            .insert(session_id);

        let handle = WorkerHandle {
            session_id,
            node_id: node_id.clone(),
            place,
            capabilities,
            tags,
            admission,
            outbound,
            admission_snapshot: None,
            in_flight: 0,
            connected_at_ms,
            labels,
            taints,
            descriptors,
        };

        self.sessions.insert(session_id, handle);
        self.by_node_id.insert(node_id, session_id);
        session_id
    }

    /// Remove a session from the registry. Idempotent — no-op if
    /// `session_id` is unknown.
    pub fn unregister(&self, session_id: Uuid) {
        if let Some((_, handle)) = self.sessions.remove(&session_id) {
            self.by_node_id.remove(&handle.node_id);
            for cap in &handle.capabilities {
                if let Some(mut set) = self
                    .capability_index
                    .get_mut(&(handle.place.clone(), cap.clone()))
                {
                    set.remove(&session_id);
                }
                // Optional: drop the entry entirely if the set is empty.
                // Leave it for now — harmless and avoids extra contention.
            }
            if let Some(mut set) = self.by_place.get_mut(&handle.place) {
                set.remove(&session_id);
            }
        }
    }

    /// Look up a session by id.
    #[must_use]
    pub fn get(&self, session_id: Uuid) -> Option<WorkerHandle> {
        self.sessions.get(&session_id).map(|r| r.clone())
    }

    /// Find every session in `place` advertising a given capability.
    /// Workers in other places are never returned — tenancy isolation is
    /// enforced at the index level.
    #[must_use]
    pub fn workers_with_capability(
        &self,
        place: &str,
        cap: &WorkerCapability,
    ) -> Vec<WorkerHandle> {
        self.capability_index
            .get(&(place.to_string(), cap.clone()))
            .map(|set| {
                set.iter()
                    .filter_map(|sid| self.sessions.get(sid).map(|r| r.clone()))
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Snapshot of every connected worker — used by `ListWorkers`.
    #[must_use]
    pub fn list(&self) -> Vec<WorkerInfo> {
        self.sessions
            .iter()
            .map(|r| worker_info(r.value()))
            .collect()
    }

    /// Like [`Self::list`] but pairs each worker with its place. The
    /// `place` has no `core::WorkerInfo` home, so callers that need it on
    /// the wire read it from here.
    #[must_use]
    pub fn list_with_place(&self) -> Vec<(WorkerInfo, String)> {
        self.sessions
            .iter()
            .map(|r| {
                let h = r.value();
                (worker_info(h), h.place.clone())
            })
            .collect()
    }

    /// Snapshot of every connected worker serving `place`. Returns an
    /// empty `Vec` when no worker serves that place.
    #[must_use]
    pub fn list_by_place(&self, place: &str) -> Vec<WorkerInfo> {
        self.by_place
            .get(place)
            .map(|set| {
                set.iter()
                    .filter_map(|sid| self.sessions.get(sid).map(|r| worker_info(r.value())))
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Look up a session by node id (used by `DrainWorker`).
    #[must_use]
    pub fn session_for_node(&self, node_id: &str) -> Option<Uuid> {
        self.by_node_id.get(node_id).map(|r| *r)
    }

    /// Apply a heartbeat update.
    pub fn record_heartbeat(
        &self,
        session_id: Uuid,
        in_flight: u32,
        snapshot: Option<AdmissionSnapshot>,
    ) {
        if let Some(mut entry) = self.sessions.get_mut(&session_id) {
            entry.in_flight = in_flight;
            entry.admission_snapshot = snapshot;
        }
    }

    /// Union of every connected worker's capability-descriptor manifest.
    /// De-duplicates entries by `descriptor.id` — when more than one
    /// worker publishes the same descriptor (the common case for
    /// horizontally-scaled capability pools), the first one observed
    /// wins. Iteration order matches `dashmap`'s shard scan and is
    /// therefore not stable across calls.
    #[must_use]
    pub fn all_descriptors(&self) -> Vec<NodeDescriptorWire> {
        let mut seen: HashSet<String> = HashSet::new();
        let mut out: Vec<NodeDescriptorWire> = Vec::new();
        for entry in &self.sessions {
            for desc in &entry.value().descriptors {
                if seen.insert(desc.id.clone()) {
                    out.push(desc.clone());
                }
            }
        }
        out
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.sessions.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.sessions.is_empty()
    }
}

impl Default for WorkerRegistry {
    fn default() -> Self {
        Self::new()
    }
}

/// Project a [`WorkerHandle`] into the core [`WorkerInfo`] snapshot.
///
/// `core::WorkerInfo` carries no place field; callers that need the place
/// on the wire read it from the handle directly (see
/// [`super::super::protocol::WorkerInfoWire`]).
fn worker_info(h: &WorkerHandle) -> WorkerInfo {
    WorkerInfo {
        node_id: h.node_id.clone(),
        capabilities: h.capabilities.iter().cloned().collect(),
        tags: h.tags.clone(),
        admission: h.admission.clone(),
        in_flight: h.in_flight,
        admission_snapshot: h.admission_snapshot.clone(),
        connected_at_ms: h.connected_at_ms,
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

    fn cap(kind: &str, version: u32) -> WorkerCapability {
        WorkerCapability {
            kind: kind.to_string(),
            version,
        }
    }

    #[test]
    fn register_and_lookup() {
        let reg = WorkerRegistry::new();
        let (tx, _rx) = mpsc::channel(8);
        let mut caps = HashSet::new();
        caps.insert(cap("workflow:hello", 1));
        let sid = reg.register(
            "node-a".into(),
            "__default__".into(),
            caps.clone(),
            BTreeMap::new(),
            AdmissionMode::Fixed { max_in_flight: 4 },
            tx,
            BTreeMap::new(),
            Vec::new(),
            Vec::new(),
        );

        assert_eq!(reg.len(), 1);
        assert!(reg.get(sid).is_some());
        assert_eq!(reg.session_for_node("node-a"), Some(sid));
        let candidates = reg.workers_with_capability("__default__", &cap("workflow:hello", 1));
        assert_eq!(candidates.len(), 1);
        assert_eq!(candidates[0].session_id, sid);
    }

    #[test]
    fn unregister_clears_indexes() {
        let reg = WorkerRegistry::new();
        let (tx, _rx) = mpsc::channel(8);
        let mut caps = HashSet::new();
        caps.insert(cap("workflow:hello", 1));
        let sid = reg.register(
            "node-a".into(),
            "__default__".into(),
            caps,
            BTreeMap::new(),
            AdmissionMode::Reactive,
            tx,
            BTreeMap::new(),
            Vec::new(),
            Vec::new(),
        );

        reg.unregister(sid);
        assert!(reg.is_empty());
        assert!(reg.get(sid).is_none());
        assert!(reg.session_for_node("node-a").is_none());
        assert!(
            reg.workers_with_capability("__default__", &cap("workflow:hello", 1))
                .is_empty()
        );
        assert!(reg.list_by_place("__default__").is_empty());
    }

    #[test]
    fn re_register_same_node_evicts_old_session() {
        let reg = WorkerRegistry::new();
        let (tx1, _rx1) = mpsc::channel(8);
        let (tx2, _rx2) = mpsc::channel(8);
        let mut caps = HashSet::new();
        caps.insert(cap("workflow:hello", 1));

        let sid1 = reg.register(
            "node-a".into(),
            "__default__".into(),
            caps.clone(),
            BTreeMap::new(),
            AdmissionMode::Fixed { max_in_flight: 4 },
            tx1,
            BTreeMap::new(),
            Vec::new(),
            Vec::new(),
        );
        let sid2 = reg.register(
            "node-a".into(),
            "__default__".into(),
            caps,
            BTreeMap::new(),
            AdmissionMode::Fixed { max_in_flight: 4 },
            tx2,
            BTreeMap::new(),
            Vec::new(),
            Vec::new(),
        );

        assert_ne!(sid1, sid2);
        assert!(reg.get(sid1).is_none(), "old session evicted");
        assert!(reg.get(sid2).is_some(), "new session active");
        assert_eq!(reg.len(), 1);
    }

    #[test]
    fn heartbeat_updates_in_flight() {
        let reg = WorkerRegistry::new();
        let (tx, _rx) = mpsc::channel(8);
        let sid = reg.register(
            "node-a".into(),
            "__default__".into(),
            HashSet::new(),
            BTreeMap::new(),
            AdmissionMode::Fixed { max_in_flight: 4 },
            tx,
            BTreeMap::new(),
            Vec::new(),
            Vec::new(),
        );
        reg.record_heartbeat(sid, 3, None);
        assert_eq!(reg.get(sid).unwrap().in_flight, 3);
    }

    #[test]
    fn capability_lookup_is_place_scoped() {
        let reg = WorkerRegistry::new();
        let mut caps = HashSet::new();
        caps.insert(cap("workflow:hello", 1));

        let (tx_a, _rx_a) = mpsc::channel(8);
        let sid_a = reg.register(
            "node-a".into(),
            "place-a".into(),
            caps.clone(),
            BTreeMap::new(),
            AdmissionMode::Reactive,
            tx_a,
            BTreeMap::new(),
            Vec::new(),
            Vec::new(),
        );
        let (tx_b, _rx_b) = mpsc::channel(8);
        let _sid_b = reg.register(
            "node-b".into(),
            "place-b".into(),
            caps,
            BTreeMap::new(),
            AdmissionMode::Reactive,
            tx_b,
            BTreeMap::new(),
            Vec::new(),
            Vec::new(),
        );

        // Each place sees only its own worker.
        let in_a = reg.workers_with_capability("place-a", &cap("workflow:hello", 1));
        assert_eq!(in_a.len(), 1);
        assert_eq!(in_a[0].session_id, sid_a);
        assert_eq!(in_a[0].place, "place-a");

        let in_b = reg.workers_with_capability("place-b", &cap("workflow:hello", 1));
        assert_eq!(in_b.len(), 1);
        assert_eq!(in_b[0].node_id, "node-b");

        // A third place has no candidates.
        assert!(
            reg.workers_with_capability("place-c", &cap("workflow:hello", 1))
                .is_empty()
        );

        // list_by_place mirrors the isolation.
        assert_eq!(reg.list_by_place("place-a").len(), 1);
        assert_eq!(reg.list_by_place("place-b").len(), 1);
        assert!(reg.list_by_place("place-c").is_empty());
    }
}
