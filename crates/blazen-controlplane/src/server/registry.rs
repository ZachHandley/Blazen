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

use blazen_core::distributed::{AdmissionMode, AdmissionSnapshot, WorkerCapability, WorkerInfo};

use crate::protocol::ServerToWorker;

/// One connected worker.
#[derive(Clone)]
pub struct WorkerHandle {
    pub session_id: Uuid,
    pub node_id: String,
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
    /// capability → set of `session_ids` advertising it. Used by the
    /// queue to find candidate workers for a submitted workflow.
    capability_index: DashMap<WorkerCapability, HashSet<Uuid>>,
}

impl WorkerRegistry {
    #[must_use]
    pub fn new() -> Self {
        Self {
            sessions: DashMap::new(),
            by_node_id: DashMap::new(),
            capability_index: DashMap::new(),
        }
    }

    /// Register a freshly-connected worker. Returns the assigned
    /// `session_id`. If a worker with the same `node_id` is already
    /// registered, its session is *replaced* (the old session's
    /// outbound channel is dropped, which the old session task uses as
    /// a shutdown signal).
    #[must_use = "the returned session_id is the only handle to the registered worker"]
    pub fn register(
        &self,
        node_id: String,
        capabilities: HashSet<WorkerCapability>,
        tags: BTreeMap<String, String>,
        admission: AdmissionMode,
        outbound: mpsc::Sender<ServerToWorker>,
    ) -> Uuid {
        let session_id = Uuid::new_v4();
        let connected_at_ms = now_ms();

        // Evict any prior session for this node_id.
        if let Some(prior) = self.by_node_id.get(&node_id).map(|r| *r) {
            self.unregister(prior);
        }

        // Insert into the capability index first using clones, then move
        // the original set into the handle — saves us cloning the whole
        // set just to satisfy ownership.
        for cap in &capabilities {
            self.capability_index
                .entry(cap.clone())
                .or_default()
                .insert(session_id);
        }

        let handle = WorkerHandle {
            session_id,
            node_id: node_id.clone(),
            capabilities,
            tags,
            admission,
            outbound,
            admission_snapshot: None,
            in_flight: 0,
            connected_at_ms,
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
                if let Some(mut set) = self.capability_index.get_mut(cap) {
                    set.remove(&session_id);
                }
                // Optional: drop the entry entirely if the set is empty.
                // Leave it for now — harmless and avoids extra contention.
            }
        }
    }

    /// Look up a session by id.
    #[must_use]
    pub fn get(&self, session_id: Uuid) -> Option<WorkerHandle> {
        self.sessions.get(&session_id).map(|r| r.clone())
    }

    /// Find every session advertising a given capability.
    #[must_use]
    pub fn workers_with_capability(&self, cap: &WorkerCapability) -> Vec<WorkerHandle> {
        self.capability_index
            .get(cap)
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
            .map(|r| {
                let h = r.value();
                WorkerInfo {
                    node_id: h.node_id.clone(),
                    capabilities: h.capabilities.iter().cloned().collect(),
                    tags: h.tags.clone(),
                    admission: h.admission.clone(),
                    in_flight: h.in_flight,
                    admission_snapshot: h.admission_snapshot.clone(),
                    connected_at_ms: h.connected_at_ms,
                }
            })
            .collect()
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
            caps.clone(),
            BTreeMap::new(),
            AdmissionMode::Fixed { max_in_flight: 4 },
            tx,
        );

        assert_eq!(reg.len(), 1);
        assert!(reg.get(sid).is_some());
        assert_eq!(reg.session_for_node("node-a"), Some(sid));
        let candidates = reg.workers_with_capability(&cap("workflow:hello", 1));
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
            caps,
            BTreeMap::new(),
            AdmissionMode::Reactive,
            tx,
        );

        reg.unregister(sid);
        assert!(reg.is_empty());
        assert!(reg.get(sid).is_none());
        assert!(reg.session_for_node("node-a").is_none());
        assert!(
            reg.workers_with_capability(&cap("workflow:hello", 1))
                .is_empty()
        );
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
            caps.clone(),
            BTreeMap::new(),
            AdmissionMode::Fixed { max_in_flight: 4 },
            tx1,
        );
        let sid2 = reg.register(
            "node-a".into(),
            caps,
            BTreeMap::new(),
            AdmissionMode::Fixed { max_in_flight: 4 },
            tx2,
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
            HashSet::new(),
            BTreeMap::new(),
            AdmissionMode::Fixed { max_in_flight: 4 },
            tx,
        );
        reg.record_heartbeat(sid, 3, None);
        assert_eq!(reg.get(sid).unwrap().in_flight, 3);
    }
}
