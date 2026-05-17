//! Valkey-backed [`AssignmentStore`].
//!
//! Each [`AssignmentStore`] method maps onto one or two Redis operations
//! issued through a shared [`redis::aio::ConnectionManager`] (cheaply
//! cloneable, multiplexed over a single connection — same convention
//! used by [`blazen_persist::valkey::ValkeyCheckpointStore`]).
//!
//! All keys live under the `blazen:cp:` prefix:
//!
//! | Key | Type | Operations |
//! |-----|------|------------|
//! | `blazen:cp:assign:{run_id}` | string | `SET` / `GET` / `DEL` postcard-encoded [`Assignment`] |
//! | `blazen:cp:state:{run_id}` | string | `SET` (or `SETEX` for terminal states) / `GET` postcard-encoded [`RunStateSnapshotWire`] |
//! | `blazen:cp:pending:{kind}:{version}` | list | `RPUSH` (tail enqueue) / `LPOP` (head dequeue) of postcard-encoded `(Uuid, Vec<String>)` |
//! | `blazen:cp:pending_caps` | set | `SADD` / `SREM` / `SMEMBERS` tracking non-empty capability keys |
//! | `blazen:cp:inflight:{run_id}` | string | `SET` / `GET` / `DEL` of `session_id` (decimal UUID, human-readable) |
//! | `blazen:cp:inflight_by_session:{session_id}` | set | `SADD` / `SREM` / `SMEMBERS` of `run_id`s claimed by that session |
//! | `blazen:cp:inflight_runs` | set | `SADD` / `SREM` / `SMEMBERS` tracking every in-flight `run_id` for orphan scans |
//!
//! Binary payloads (`Assignment`, `RunStateSnapshotWire`, pending-FIFO
//! entries) are encoded with [`postcard`]. UUIDs that flow through Redis
//! sets are stored as their decimal-hyphenated string form so
//! `redis-cli SMEMBERS …` is human-legible and matches the convention
//! used elsewhere in the workspace.
//!
//! Errors from the `redis` crate are surfaced as
//! [`ControlPlaneError::Transport`]; postcard / JSON decode failures use
//! the auto-derived `From` impls on [`ControlPlaneError`].

use std::collections::HashSet;
use std::time::Duration;

use async_trait::async_trait;
use redis::AsyncCommands;
use uuid::Uuid;

use blazen_core::distributed::{RunStateSnapshot, RunStatus, WorkerCapability};

use crate::error::ControlPlaneError;
use crate::protocol::{Assignment, RunStateSnapshotWire};

use super::store::AssignmentStore;

/// Prefix shared by every key this store writes. Mirrors the
/// `blazen:checkpoint:` convention used by
/// [`blazen_persist::valkey::ValkeyCheckpointStore`].
const PREFIX: &str = "blazen:cp:";

/// Tracking set listing every capability whose pending FIFO currently
/// has at least one entry. Avoids `SCAN` for
/// [`AssignmentStore::list_pending_capabilities`].
const KEY_PENDING_CAPS: &str = "blazen:cp:pending_caps";

/// Tracking set listing every `run_id` with an outstanding in-flight
/// claim. Lets [`AssignmentStore::list_orphaned_inflight`] enumerate
/// candidates in a single `SMEMBERS` without scanning the keyspace.
const KEY_INFLIGHT_RUNS: &str = "blazen:cp:inflight_runs";

/// Default TTL applied to terminal run-state snapshots when neither the
/// caller of [`ValkeyAssignmentStore::new`] nor the per-call
/// `ttl_after_terminal` argument overrides it. One hour.
const DEFAULT_TERMINAL_TTL: Duration = Duration::from_hours(1);

/// Map a [`redis::RedisError`] into the project's transport variant.
/// The trait already has `From` impls for postcard / json, but no
/// `From<RedisError>` (we don't want to leak the `redis` type into the
/// public error surface), so all redis call sites funnel through this
/// helper for consistency.
fn redis_err(stage: &'static str, err: &redis::RedisError) -> ControlPlaneError {
    ControlPlaneError::Transport(format!("valkey {stage}: {err}"))
}

/// [`AssignmentStore`] implementation backed by Valkey / Redis.
///
/// Cloning is cheap: the underlying
/// [`redis::aio::ConnectionManager`] multiplexes a single connection
/// and tolerates connection drops via automatic reconnection.
#[derive(Clone, Debug)]
pub struct ValkeyAssignmentStore {
    conn: redis::aio::ConnectionManager,
    /// Optional fallback TTL applied to terminal run-state snapshots
    /// when [`AssignmentStore::put_state`] is called with
    /// `ttl_after_terminal = None`. `None` means snapshots live until
    /// explicitly deleted.
    terminal_state_ttl: Option<Duration>,
}

impl ValkeyAssignmentStore {
    /// Open a fresh [`redis::aio::ConnectionManager`] against `url`
    /// (form `redis://host:port/db` or `rediss://...`). The default
    /// terminal-state TTL is one hour; use
    /// [`ValkeyAssignmentStore::with_terminal_ttl`] to override.
    ///
    /// # Errors
    /// Returns [`ControlPlaneError::Transport`] on any connect or
    /// handshake failure.
    pub async fn new(url: &str) -> Result<Self, ControlPlaneError> {
        Self::with_terminal_ttl(url, Some(DEFAULT_TERMINAL_TTL)).await
    }

    /// Open with an explicit terminal-state TTL fallback. Pass `None`
    /// to disable TTL-based eviction of completed runs.
    ///
    /// # Errors
    /// Returns [`ControlPlaneError::Transport`] on any connect or
    /// handshake failure.
    pub async fn with_terminal_ttl(
        url: &str,
        terminal_state_ttl: Option<Duration>,
    ) -> Result<Self, ControlPlaneError> {
        let client = redis::Client::open(url).map_err(|e| redis_err("open", &e))?;
        let conn = redis::aio::ConnectionManager::new(client)
            .await
            .map_err(|e| redis_err("connect", &e))?;
        Ok(Self {
            conn,
            terminal_state_ttl,
        })
    }

    fn key_assign(run_id: Uuid) -> String {
        format!("{PREFIX}assign:{run_id}")
    }

    fn key_state(run_id: Uuid) -> String {
        format!("{PREFIX}state:{run_id}")
    }

    fn key_pending(cap: &WorkerCapability) -> String {
        format!("{PREFIX}pending:{}:{}", cap.kind, cap.version)
    }

    fn key_inflight(run_id: Uuid) -> String {
        format!("{PREFIX}inflight:{run_id}")
    }

    fn key_inflight_by_session(session_id: Uuid) -> String {
        format!("{PREFIX}inflight_by_session:{session_id}")
    }

    /// Compose the tracking-set member used in
    /// [`KEY_PENDING_CAPS`]. `capability.kind` may itself contain `:`
    /// (e.g. `workflow:hello`); the version is always a decimal `u32`
    /// at the tail, so we split on the LAST `:` when decoding.
    fn pending_caps_member(cap: &WorkerCapability) -> String {
        format!("{}:{}", cap.kind, cap.version)
    }

    fn parse_pending_caps_member(raw: &str) -> Option<WorkerCapability> {
        let (kind, version_str) = raw.rsplit_once(':')?;
        let version = version_str.parse::<u32>().ok()?;
        Some(WorkerCapability {
            kind: kind.to_string(),
            version,
        })
    }
}

#[async_trait]
impl AssignmentStore for ValkeyAssignmentStore {
    async fn put_assignment(
        &self,
        run_id: Uuid,
        assignment: &Assignment,
    ) -> Result<(), ControlPlaneError> {
        let bytes = postcard::to_allocvec(assignment)?;
        let mut conn = self.conn.clone();
        let () = conn
            .set(Self::key_assign(run_id), bytes)
            .await
            .map_err(|e| redis_err("set assign", &e))?;
        Ok(())
    }

    async fn get_assignment(&self, run_id: Uuid) -> Result<Option<Assignment>, ControlPlaneError> {
        let mut conn = self.conn.clone();
        let bytes: Option<Vec<u8>> = conn
            .get(Self::key_assign(run_id))
            .await
            .map_err(|e| redis_err("get assign", &e))?;
        match bytes {
            Some(b) => Ok(Some(postcard::from_bytes(&b)?)),
            None => Ok(None),
        }
    }

    async fn delete_assignment(&self, run_id: Uuid) -> Result<(), ControlPlaneError> {
        let mut conn = self.conn.clone();
        // `DEL` on a missing key returns 0; we don't care about the count.
        let _removed: usize = conn
            .del(Self::key_assign(run_id))
            .await
            .map_err(|e| redis_err("del assign", &e))?;
        Ok(())
    }

    async fn put_state(
        &self,
        run_id: Uuid,
        state: &RunStateSnapshot,
        ttl_after_terminal: Option<Duration>,
    ) -> Result<(), ControlPlaneError> {
        let wire = RunStateSnapshotWire::from_core(state)?;
        let bytes = postcard::to_allocvec(&wire)?;
        let mut conn = self.conn.clone();
        let key = Self::key_state(run_id);

        let is_terminal = matches!(
            state.status,
            RunStatus::Completed | RunStatus::Failed | RunStatus::Cancelled
        );
        let effective_ttl = if is_terminal {
            ttl_after_terminal.or(self.terminal_state_ttl)
        } else {
            None
        };

        if let Some(ttl) = effective_ttl {
            let secs = ttl.as_secs().max(1);
            let () = conn
                .set_ex(&key, bytes, secs)
                .await
                .map_err(|e| redis_err("set_ex state", &e))?;
        } else {
            let () = conn
                .set(&key, bytes)
                .await
                .map_err(|e| redis_err("set state", &e))?;
        }
        Ok(())
    }

    async fn get_state(&self, run_id: Uuid) -> Result<Option<RunStateSnapshot>, ControlPlaneError> {
        let mut conn = self.conn.clone();
        let bytes: Option<Vec<u8>> = conn
            .get(Self::key_state(run_id))
            .await
            .map_err(|e| redis_err("get state", &e))?;
        match bytes {
            Some(b) => {
                let wire: RunStateSnapshotWire = postcard::from_bytes(&b)?;
                Ok(Some(wire.to_core()?))
            }
            None => Ok(None),
        }
    }

    async fn enqueue_pending(
        &self,
        capability: &WorkerCapability,
        run_id: Uuid,
        required_tags: &[String],
    ) -> Result<(), ControlPlaneError> {
        let entry: (Uuid, Vec<String>) = (run_id, required_tags.to_vec());
        let bytes = postcard::to_allocvec(&entry)?;
        let mut conn = self.conn.clone();
        let _pushed: usize = conn
            .rpush(Self::key_pending(capability), bytes)
            .await
            .map_err(|e| redis_err("rpush pending", &e))?;
        let _added: usize = conn
            .sadd(KEY_PENDING_CAPS, Self::pending_caps_member(capability))
            .await
            .map_err(|e| redis_err("sadd pending_caps", &e))?;
        Ok(())
    }

    async fn dequeue_pending(
        &self,
        capability: &WorkerCapability,
    ) -> Result<Option<(Uuid, Vec<String>)>, ControlPlaneError> {
        let mut conn = self.conn.clone();
        let key = Self::key_pending(capability);
        let bytes: Option<Vec<u8>> = conn
            .lpop(&key, None)
            .await
            .map_err(|e| redis_err("lpop pending", &e))?;
        match bytes {
            Some(b) => {
                let entry: (Uuid, Vec<String>) = postcard::from_bytes(&b)?;
                let remaining: usize = conn
                    .llen(&key)
                    .await
                    .map_err(|e| redis_err("llen pending", &e))?;
                if remaining == 0 {
                    let _removed: usize = conn
                        .srem(KEY_PENDING_CAPS, Self::pending_caps_member(capability))
                        .await
                        .map_err(|e| redis_err("srem pending_caps", &e))?;
                }
                Ok(Some(entry))
            }
            None => Ok(None),
        }
    }

    async fn list_pending_capabilities(&self) -> Result<Vec<WorkerCapability>, ControlPlaneError> {
        let mut conn = self.conn.clone();
        let members: Vec<String> = conn
            .smembers(KEY_PENDING_CAPS)
            .await
            .map_err(|e| redis_err("smembers pending_caps", &e))?;
        Ok(members
            .iter()
            .filter_map(|m| Self::parse_pending_caps_member(m))
            .collect())
    }

    async fn claim_inflight(
        &self,
        run_id: Uuid,
        session_id: Uuid,
    ) -> Result<(), ControlPlaneError> {
        let mut conn = self.conn.clone();
        let () = conn
            .set(Self::key_inflight(run_id), session_id.to_string())
            .await
            .map_err(|e| redis_err("set inflight", &e))?;
        let _added_session: usize = conn
            .sadd(
                Self::key_inflight_by_session(session_id),
                run_id.to_string(),
            )
            .await
            .map_err(|e| redis_err("sadd inflight_by_session", &e))?;
        let _added_global: usize = conn
            .sadd(KEY_INFLIGHT_RUNS, run_id.to_string())
            .await
            .map_err(|e| redis_err("sadd inflight_runs", &e))?;
        Ok(())
    }

    async fn release_inflight(&self, session_id: Uuid) -> Result<Vec<Uuid>, ControlPlaneError> {
        let mut conn = self.conn.clone();
        let raw: Vec<String> = conn
            .smembers(Self::key_inflight_by_session(session_id))
            .await
            .map_err(|e| redis_err("smembers inflight_by_session", &e))?;
        let run_ids: Vec<Uuid> = raw.iter().filter_map(|s| Uuid::parse_str(s).ok()).collect();

        let _removed_session_set: usize = conn
            .del(Self::key_inflight_by_session(session_id))
            .await
            .map_err(|e| redis_err("del inflight_by_session", &e))?;

        for run_id in &run_ids {
            let _removed_inflight: usize = conn
                .del(Self::key_inflight(*run_id))
                .await
                .map_err(|e| redis_err("del inflight", &e))?;
            let _removed_global: usize = conn
                .srem(KEY_INFLIGHT_RUNS, run_id.to_string())
                .await
                .map_err(|e| redis_err("srem inflight_runs", &e))?;
        }
        Ok(run_ids)
    }

    async fn list_orphaned_inflight(
        &self,
        alive_sessions: &HashSet<Uuid>,
    ) -> Result<Vec<Uuid>, ControlPlaneError> {
        let mut conn = self.conn.clone();
        let raw_run_ids: Vec<String> = conn
            .smembers(KEY_INFLIGHT_RUNS)
            .await
            .map_err(|e| redis_err("smembers inflight_runs", &e))?;

        let mut orphans = Vec::new();
        for raw in raw_run_ids {
            let Ok(run_id) = Uuid::parse_str(&raw) else {
                continue;
            };
            let sid_str: Option<String> = conn
                .get(Self::key_inflight(run_id))
                .await
                .map_err(|e| redis_err("get inflight", &e))?;
            let Some(sid_str) = sid_str else { continue };
            let Ok(session_id) = Uuid::parse_str(&sid_str) else {
                continue;
            };
            if !alive_sessions.contains(&session_id) {
                orphans.push(run_id);
            }
        }
        Ok(orphans)
    }
}

#[cfg(test)]
mod tests {
    //! Round-trip tests against a real Valkey / Redis instance.
    //!
    //! The tests **compile** unconditionally (so workspace `clippy
    //! --tests` covers them) but **skip at runtime** unless the
    //! `BLAZEN_TEST_VALKEY_URL` environment variable is set, since most
    //! CI lanes do not run a Valkey side-car. To exercise locally:
    //!
    //! ```bash
    //! docker run --rm -p 6379:6379 -d valkey/valkey:latest
    //! BLAZEN_TEST_VALKEY_URL=redis://127.0.0.1:6379/ \
    //!   cargo nextest run -p blazen-controlplane --all-features --lib
    //! ```
    //!
    //! Each test scopes its keys to a fresh per-run prefix derived from
    //! a random UUID so concurrent test invocations against the same
    //! Valkey instance do not collide.

    use super::*;

    use blazen_core::distributed::RunStatus;

    use crate::protocol::ENVELOPE_VERSION;

    /// Pull the connection URL from `BLAZEN_TEST_VALKEY_URL`. Returns
    /// `None` (and emits a visible skip log) when unset.
    fn valkey_url() -> Option<String> {
        match std::env::var("BLAZEN_TEST_VALKEY_URL") {
            Ok(url) if !url.is_empty() => Some(url),
            _ => {
                eprintln!(
                    "skipping valkey_store test: set BLAZEN_TEST_VALKEY_URL=redis://… to run"
                );
                None
            }
        }
    }

    async fn fresh_store() -> Option<ValkeyAssignmentStore> {
        let url = valkey_url()?;
        Some(
            ValkeyAssignmentStore::new(&url)
                .await
                .expect("connect to valkey"),
        )
    }

    fn capability(kind: &str, version: u32) -> WorkerCapability {
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
    async fn assignment_put_get_delete_round_trip() {
        let Some(store) = fresh_store().await else {
            return;
        };
        let run_id = Uuid::new_v4();
        let assignment = make_assignment(run_id, "workflow:hello");

        store
            .put_assignment(run_id, &assignment)
            .await
            .expect("put_assignment");
        let fetched = store
            .get_assignment(run_id)
            .await
            .expect("get_assignment")
            .expect("Some(assignment)");
        assert_eq!(fetched.run_id, run_id);
        assert_eq!(fetched.workflow_name, "workflow:hello");

        store
            .delete_assignment(run_id)
            .await
            .expect("delete_assignment");
        let missing = store
            .get_assignment(run_id)
            .await
            .expect("get_assignment after delete");
        assert!(missing.is_none());
    }

    #[tokio::test]
    async fn state_put_get_with_terminal_ttl() {
        let Some(store) = fresh_store().await else {
            return;
        };
        let run_id = Uuid::new_v4();

        // Non-terminal: no TTL, persists indefinitely.
        let running = make_state(run_id, RunStatus::Running);
        store
            .put_state(run_id, &running, None)
            .await
            .expect("put_state running");
        let fetched = store
            .get_state(run_id)
            .await
            .expect("get_state")
            .expect("Some(state)");
        assert_eq!(fetched.run_id, run_id);
        assert_eq!(fetched.status, RunStatus::Running);

        // Terminal: applies the per-call TTL.
        let completed = make_state(run_id, RunStatus::Completed);
        store
            .put_state(run_id, &completed, Some(Duration::from_mins(1)))
            .await
            .expect("put_state completed");
        let fetched = store
            .get_state(run_id)
            .await
            .expect("get_state after terminal")
            .expect("Some(state)");
        assert_eq!(fetched.status, RunStatus::Completed);
    }

    #[tokio::test]
    async fn pending_fifo_via_rpush_lpop() {
        let Some(store) = fresh_store().await else {
            return;
        };
        // Use a UUID-scoped kind so this test doesn't collide with
        // anything else running against the same Valkey instance.
        let cap = capability(&format!("workflow:fifo-{}", Uuid::new_v4()), 1);
        let id1 = Uuid::new_v4();
        let id2 = Uuid::new_v4();
        let id3 = Uuid::new_v4();

        store
            .enqueue_pending(&cap, id1, &["tag=a".to_string()])
            .await
            .expect("enqueue 1");
        store
            .enqueue_pending(&cap, id2, &["tag=b".to_string()])
            .await
            .expect("enqueue 2");
        store
            .enqueue_pending(&cap, id3, &["tag=c".to_string()])
            .await
            .expect("enqueue 3");

        let first = store
            .dequeue_pending(&cap)
            .await
            .expect("dequeue 1")
            .expect("Some");
        let second = store
            .dequeue_pending(&cap)
            .await
            .expect("dequeue 2")
            .expect("Some");
        let third = store
            .dequeue_pending(&cap)
            .await
            .expect("dequeue 3")
            .expect("Some");
        let fourth = store.dequeue_pending(&cap).await.expect("dequeue 4");

        assert_eq!(first, (id1, vec!["tag=a".to_string()]));
        assert_eq!(second, (id2, vec!["tag=b".to_string()]));
        assert_eq!(third, (id3, vec!["tag=c".to_string()]));
        assert!(fourth.is_none());
    }

    #[tokio::test]
    async fn list_pending_capabilities_tracks_set() {
        let Some(store) = fresh_store().await else {
            return;
        };
        let cap_a = capability(&format!("workflow:caps-a-{}", Uuid::new_v4()), 1);
        let cap_b = capability(&format!("workflow:caps-b-{}", Uuid::new_v4()), 2);
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

        let caps = store
            .list_pending_capabilities()
            .await
            .expect("list capabilities");
        assert!(caps.contains(&cap_a), "cap_a present in {caps:?}");
        assert!(caps.contains(&cap_b), "cap_b present in {caps:?}");

        // Drain cap_a fully — it should drop out of the tracking set.
        let _drained = store
            .dequeue_pending(&cap_a)
            .await
            .expect("dequeue a")
            .expect("Some");
        let caps_after = store
            .list_pending_capabilities()
            .await
            .expect("list after drain");
        assert!(
            !caps_after.contains(&cap_a),
            "cap_a removed in {caps_after:?}"
        );
        assert!(
            caps_after.contains(&cap_b),
            "cap_b still present in {caps_after:?}"
        );

        // Clean up the surviving entry so the test leaves no garbage.
        let _ = store.dequeue_pending(&cap_b).await.expect("dequeue b");
    }

    #[tokio::test]
    async fn inflight_claim_release_round_trip() {
        let Some(store) = fresh_store().await else {
            return;
        };
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
        let Some(store) = fresh_store().await else {
            return;
        };
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
        assert!(
            orphans.contains(&orphan_run),
            "orphan_run present in {orphans:?}"
        );
        assert!(
            !orphans.contains(&live_run_a),
            "live_run_a absent in {orphans:?}"
        );
        assert!(
            !orphans.contains(&live_run_b),
            "live_run_b absent in {orphans:?}"
        );

        // Clean up so the test doesn't leak entries into KEY_INFLIGHT_RUNS.
        let _ = store
            .release_inflight(live_session)
            .await
            .expect("release live");
        let _ = store
            .release_inflight(dead_session)
            .await
            .expect("release dead");
    }
}
