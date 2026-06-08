//! P6 integration proof: tenant/place isolation across the control plane.
//!
//! Two assertions, both driven through a live [`ControlPlaneServer`] with
//! a token→place [`PlaceAuthenticator`] and NO real provider HTTP:
//!
//! 1. **Assignment routing is place-scoped.** Place-A and place-B workers
//!    both advertise the SAME capability (`workflow:x`). A submission made
//!    as place A lands ONLY on A's worker — B's handler never sees it —
//!    proving the queue's `(place, capability)` bucketing keeps tenants
//!    apart.
//! 2. **Key brokering is session-place-scoped.** A provider key seeded
//!    ONLY under place B is invisible to a place-A worker session: A's
//!    pre-warm round-trip returns `KeyResponse{key: None}`, so the worker
//!    resolves no key for that provider and falls through (here, to an
//!    error — nothing else supplies it).
//!
//! ## Global-state discipline
//!
//! The place-authenticator is process-global + set-once, and each Cargo
//! test binary is one process. This file installs exactly one
//! authenticator (mapping both `tok-a`→`a` and `tok-b`→`b`) and runs both
//! assertions as a single `#[tokio::test]`.

use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use tokio::sync::Mutex;

use blazen_controlplane::auth::{
    PeerIdentity, PeerKind, PlaceAuthenticator, install_place_authenticator,
};
use blazen_controlplane::protocol::{Assignment, ENVELOPE_VERSION, RunStatusWire, SubmitRequest};
use blazen_controlplane::server::ControlPlaneServer;
use blazen_controlplane::server::key_store::{EnvFileKeyStore, KeyStore};
use blazen_controlplane::worker::{
    AssignmentContext, AssignmentFailure, AssignmentHandler, Worker, WorkerConfig,
};
use blazen_core::distributed::{AdmissionMode, WorkerCapability};
use blazen_llm::{clear_key_resolvers, resolve_api_key, set_current_place};

use tonic::transport::Channel;

const TOKEN_A: &str = "tok-a";
const TOKEN_B: &str = "tok-b";
const PLACE_A: &str = "a";
const PLACE_B: &str = "b";
const CAP: &str = "x";
const KEY_PROVIDER: &str = "openai";
const PLACE_B_ONLY_KEY: &str = "sk-place-b-only";

/// Outcome of an in-handler key resolution, observable by the test.
#[derive(Clone)]
enum ResolveOutcome {
    /// The handler has not yet attempted resolution.
    Pending,
    /// Resolution finished; `Some(key)` on success, `None` on miss.
    Done(Option<String>),
}

type ResolveSlot = Arc<Mutex<ResolveOutcome>>;

/// Bind an ephemeral loopback port and return the resolved address.
async fn ephemeral_addr() -> SocketAddr {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    drop(listener);
    addr
}

fn encode<T: serde::Serialize>(v: &T) -> Vec<u8> {
    postcard::to_allocvec(v).expect("encode")
}

fn decode<T: serde::de::DeserializeOwned>(bytes: &[u8]) -> T {
    postcard::from_bytes(bytes).expect("decode")
}

/// Authenticator mapping a fixed set of `Bearer <token>` values to places.
/// Any other token (or none) is rejected, mirroring a real per-tenant
/// bearer scheme.
struct TokenTable {
    entries: Vec<(&'static str, &'static str)>,
}

impl PlaceAuthenticator for TokenTable {
    fn authenticate(&self, bearer: Option<&str>) -> Result<PeerIdentity, String> {
        let header = bearer.ok_or_else(|| "missing bearer".to_string())?;
        let presented = header
            .strip_prefix("Bearer ")
            .ok_or_else(|| "authorization header must use Bearer scheme".to_string())?;
        self.entries
            .iter()
            .find(|(tok, _)| *tok == presented)
            .map(|(_, place)| PeerIdentity {
                place: (*place).to_string(),
                kind: PeerKind::Worker,
            })
            .ok_or_else(|| format!("unknown token: {presented}"))
    }
}

/// Records every assignment this worker receives, so the test can assert
/// which place's worker was (or was NOT) routed to.
#[derive(Clone, Default)]
struct Recorder {
    inner: Arc<Mutex<Vec<Assignment>>>,
}

impl Recorder {
    async fn count(&self) -> usize {
        self.inner.lock().await.len()
    }

    async fn wait_for_one(&self) -> Assignment {
        loop {
            if let Some(a) = self.inner.lock().await.first().cloned() {
                return a;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    }
}

/// Handler that records each assignment and (optionally) resolves a
/// provider key through the resolver chain, stashing the outcome so the
/// key-scoping assertion can read it.
struct PlaceHandler {
    recorder: Recorder,
    /// Provider to resolve on assignment. `None` disables resolution
    /// (routing-only worker).
    resolve_provider: Option<String>,
    resolved: ResolveSlot,
}

#[async_trait]
impl AssignmentHandler for PlaceHandler {
    async fn handle(
        &self,
        assignment: Assignment,
        _ctx: AssignmentContext,
    ) -> Result<serde_json::Value, AssignmentFailure> {
        self.recorder.inner.lock().await.push(assignment.clone());
        if let Some(provider) = &self.resolve_provider {
            let outcome = resolve_api_key(provider, None).ok();
            *self.resolved.lock().await = ResolveOutcome::Done(outcome);
        }
        let input: serde_json::Value =
            serde_json::from_slice(&assignment.input_json).unwrap_or(serde_json::Value::Null);
        Ok(serde_json::json!({ "echo": input }))
    }
}

/// Connect a worker for `place` (authenticating with `token`), advertising
/// `workflow:x`, optionally pre-warming + resolving `KEY_PROVIDER`. Returns
/// its run-task handle, assignment recorder, and resolution slot.
fn spawn_worker(
    addr: SocketAddr,
    node_id: &'static str,
    place: &'static str,
    token: &'static str,
    resolve_key: bool,
) -> (tokio::task::JoinHandle<()>, Recorder, ResolveSlot) {
    let recorder = Recorder::default();
    let resolved: ResolveSlot = Arc::new(Mutex::new(ResolveOutcome::Pending));

    let mut cfg = WorkerConfig::new(format!("http://{addr}"), node_id)
        .with_capability(WorkerCapability {
            kind: format!("workflow:{CAP}"),
            version: 0,
        })
        .with_admission(AdmissionMode::Fixed { max_in_flight: 2 })
        .with_place(place)
        .with_bearer_token(token)
        .with_heartbeat_interval(Duration::from_millis(200));
    if resolve_key {
        cfg = cfg.with_prewarm_provider(KEY_PROVIDER);
    }
    let worker = Worker::connect(cfg).expect("validate worker");

    let handler = PlaceHandler {
        recorder: recorder.clone(),
        resolve_provider: resolve_key.then(|| KEY_PROVIDER.to_string()),
        resolved: resolved.clone(),
    };
    let run = tokio::spawn(async move {
        let _ = worker.run(handler).await;
    });
    (run, recorder, resolved)
}

#[tokio::test]
async fn place_isolation_routing_and_key_scoping() {
    clear_key_resolvers();
    set_current_place(None);

    // One authenticator for the whole binary: maps both tokens.
    install_place_authenticator(Arc::new(TokenTable {
        entries: vec![(TOKEN_A, PLACE_A), (TOKEN_B, PLACE_B)],
    }));

    // Key store seeded with an openai key ONLY under place B. Place A must
    // never resolve it.
    let keys_dir = tempfile::tempdir().expect("tempdir");
    let place_b_dir = keys_dir.path().join(PLACE_B);
    tokio::fs::create_dir_all(&place_b_dir)
        .await
        .expect("mkdir place-b dir");
    tokio::fs::write(
        place_b_dir.join(KEY_PROVIDER),
        format!("{PLACE_B_ONLY_KEY}\n"),
    )
    .await
    .expect("seed place-b key");
    let key_store: Arc<dyn KeyStore> =
        Arc::new(EnvFileKeyStore::with_keys_dir(keys_dir.path().to_path_buf()));

    let addr = ephemeral_addr().await;
    let server = ControlPlaneServer::new("cp-isolation").with_key_store(key_store);
    let server_handle = tokio::spawn(async move {
        let _ = server.serve(addr).await;
    });
    tokio::time::sleep(Duration::from_millis(150)).await;

    // Worker A (place a) resolves `openai`; Worker B (place b) records only.
    let (run_a, recorder_a, resolved_a) = spawn_worker(addr, "worker-a", PLACE_A, TOKEN_A, true);
    let (run_b, recorder_b, _resolved_b) = spawn_worker(addr, "worker-b", PLACE_B, TOKEN_B, false);

    // Let both workers register.
    tokio::time::sleep(Duration::from_millis(250)).await;

    // ---- Submit as place A. ----
    let channel = Channel::from_shared(format!("http://{addr}"))
        .unwrap()
        .connect()
        .await
        .expect("orchestrator connect");
    let mut client_a = client_as(channel, TOKEN_A);
    let run_id = submit_workflow(&mut client_a, CAP, b"{\"who\":\"a\"}").await;

    // ---- Assertion 1: routing landed ONLY on A. ----
    let a_assignment = tokio::time::timeout(Duration::from_secs(5), recorder_a.wait_for_one())
        .await
        .expect("place-A worker received the assignment");
    assert_eq!(a_assignment.run_id, run_id);
    assert_eq!(a_assignment.input_json, b"{\"who\":\"a\"}");

    // Drive the run to completion, then give B ample time to (wrongly)
    // pick it up — it must not.
    let final_status = poll_until_terminal(&mut client_a, run_id).await;
    assert_eq!(final_status.status, RunStatusWire::Completed);
    tokio::time::sleep(Duration::from_millis(200)).await;
    assert_eq!(
        recorder_b.count().await,
        0,
        "place-B worker must NEVER receive a place-A submission",
    );

    // ---- Assertion 2: place-A session cannot resolve place-B's key. ----
    // Worker A pre-warmed `openai` on its assignment; the key exists only
    // under place B, so the broker returned `KeyResponse{key: None}` and
    // the worker resolved nothing for that provider.
    let a_resolved = wait_resolved(&resolved_a).await;
    assert!(
        a_resolved.is_none(),
        "place A must NOT resolve place B's key (session-place scoping); got {a_resolved:?}",
    );

    // And to be explicit about the resolver-chain state: place A holds no
    // `openai` key anywhere, so a direct resolve also misses.
    set_current_place(Some(PLACE_A.to_string()));
    assert!(
        resolve_api_key(KEY_PROVIDER, None).is_err(),
        "no openai key is resolvable for place A",
    );

    run_a.abort();
    run_b.abort();
    server_handle.abort();
    clear_key_resolvers();
    set_current_place(None);
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Build a tonic client whose every request carries `Bearer <token>` so
/// the server derives the caller's place from it.
fn client_as(
    channel: Channel,
    token: &'static str,
) -> blazen_controlplane::pb::blazen_control_plane_client::BlazenControlPlaneClient<
    tonic::service::interceptor::InterceptedService<
        Channel,
        impl tonic::service::Interceptor + Clone,
    >,
> {
    blazen_controlplane::pb::blazen_control_plane_client::BlazenControlPlaneClient::with_interceptor(
        channel,
        move |mut req: tonic::Request<()>| {
            req.metadata_mut()
                .insert("authorization", format!("Bearer {token}").parse().unwrap());
            Ok(req)
        },
    )
}

/// Submit a workflow as the given client and return its run id, asserting
/// the server scoped it to the client's place.
async fn submit_workflow<I>(
    client: &mut blazen_controlplane::pb::blazen_control_plane_client::BlazenControlPlaneClient<
        tonic::service::interceptor::InterceptedService<Channel, I>,
    >,
    workflow_name: &str,
    input: &[u8],
) -> uuid::Uuid
where
    I: tonic::service::Interceptor,
{
    let submit = SubmitRequest {
        envelope_version: ENVELOPE_VERSION,
        workflow_name: workflow_name.into(),
        workflow_version: None,
        input_json: input.to_vec(),
        required_tags: vec![],
        idempotency_key: None,
        deadline_ms: None,
        wait_for_worker: true,
        resource_hint: None,
        place: None,
    };
    let resp = client
        .submit_workflow(blazen_controlplane::pb::PostcardRequest {
            postcard_payload: encode(&submit),
        })
        .await
        .expect("submit_workflow");
    let snap: blazen_controlplane::protocol::RunStateSnapshotWire =
        decode(&resp.into_inner().postcard_payload);
    assert_eq!(snap.place, PLACE_A, "submission scoped to place A");
    snap.run_id
}

/// Wait (up to 5s) for the handler's resolution outcome.
async fn wait_resolved(slot: &ResolveSlot) -> Option<String> {
    let deadline = std::time::Instant::now() + Duration::from_secs(5);
    loop {
        if let ResolveOutcome::Done(outcome) = slot.lock().await.clone() {
            return outcome;
        }
        assert!(
            std::time::Instant::now() < deadline,
            "handler never attempted key resolution within 5s",
        );
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
}

/// Poll `describe_workflow` until the run reaches a terminal state.
async fn poll_until_terminal<I>(
    client: &mut blazen_controlplane::pb::blazen_control_plane_client::BlazenControlPlaneClient<
        tonic::service::interceptor::InterceptedService<Channel, I>,
    >,
    run_id: uuid::Uuid,
) -> blazen_controlplane::protocol::RunStateSnapshotWire
where
    I: tonic::service::Interceptor,
{
    let deadline = std::time::Instant::now() + Duration::from_secs(5);
    loop {
        let req = blazen_controlplane::protocol::DescribeRequest {
            envelope_version: ENVELOPE_VERSION,
            run_id,
        };
        let resp = client
            .describe_workflow(blazen_controlplane::pb::PostcardRequest {
                postcard_payload: encode(&req),
            })
            .await
            .expect("describe_workflow");
        let snap: blazen_controlplane::protocol::RunStateSnapshotWire =
            decode(&resp.into_inner().postcard_payload);
        if matches!(
            snap.status,
            RunStatusWire::Completed | RunStatusWire::Failed | RunStatusWire::Cancelled,
        ) {
            return snap;
        }
        assert!(
            std::time::Instant::now() < deadline,
            "run {run_id} did not reach a terminal state within 5s (last={snap:?})",
        );
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
}
