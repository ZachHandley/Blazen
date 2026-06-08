//! P6 integration proof: ONE provider-key resolution flow, exercised
//! three ways, asserting that only the *configuration* differs between
//! them — the worker code path is identical.
//!
//! 1. **STANDALONE** — no control plane, no resolver chain. The key
//!    resolves straight from the [`blazen_llm`] terminal
//!    ([`resolve_api_key`]), exactly as a single-process Blazen caller
//!    gets it today.
//! 2. **CP-BROKERED** — a real [`ControlPlaneServer`] with a per-place
//!    [`KeyStore`], a [`PlaceAuthenticator`] mapping the worker's bearer
//!    token to place `"acme"`, and a connected [`Worker`] that pre-warms
//!    the brokered key over its authenticated session. The handler then
//!    resolves the key with **no env var and no local key configured**,
//!    proving it came from the broker.
//! 3. **LOCAL FALLBACK** — no key anywhere; a [`FallbackPolicy::WhenNoKey`]
//!    [`build_model`] call routes to a mock local factory and the remote
//!    builder is never invoked.
//!
//! ## Global-state discipline
//!
//! [`blazen_llm`]'s resolver chain + current-place and the control-plane's
//! place-authenticator are all process-global, and each Cargo test binary
//! is one process. This file runs the three flows in sequence within a
//! single `#[tokio::test]`, clearing the resolver chain + place scope
//! between them so one flow can never leak a key into the next. No
//! provider HTTP is ever issued — the remote builder is a closure that
//! records-and-must-not-be-called.

use std::net::SocketAddr;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use async_trait::async_trait;

use blazen_controlplane::auth::{PeerIdentity, PeerKind, PlaceAuthenticator};
use blazen_controlplane::protocol::{Assignment, ENVELOPE_VERSION, RunStatusWire, SubmitRequest};
use blazen_controlplane::server::ControlPlaneServer;
use blazen_controlplane::server::key_store::{EnvFileKeyStore, KeyStore, SharedKey};
use blazen_controlplane::worker::{
    AssignmentContext, AssignmentFailure, AssignmentHandler, Worker, WorkerConfig,
};
use blazen_core::distributed::{AdmissionMode, WorkerCapability};
use blazen_llm::error::BlazenError;
use blazen_llm::types::provider_options::ProviderOptions;
use blazen_llm::{
    FallbackPolicy, LocalModelFactory, LocalModelProbe, Model, build_model, clear_key_resolvers,
    resolve_api_key, set_current_place,
};

use tonic::transport::Channel;

// --- Fixtures shared across the three flows ---
const STANDALONE_PROVIDER: &str = "fal";
const STANDALONE_KEY: &str = "standalone-fal-key";
/// A provider with no env var in [`blazen_llm::keys::PROVIDER_ENV_VARS`],
/// so the env terminal can never satisfy it — used to prove "no key
/// anywhere" deterministically.
const NO_ENV_PROVIDER: &str = "provider-with-no-env-var";

const BROKER_PROVIDER: &str = "openai";
const BROKER_PLACE: &str = "acme";
const BROKER_TOKEN: &str = "tok-acme";
const BROKERED_KEY: &str = "sk-brokered-acme-openai";

/// Outcome of an in-handler key resolution, observable by the test.
#[derive(Clone)]
enum ResolveOutcome {
    /// The handler has not yet attempted resolution.
    Pending,
    /// Resolution finished; `Some(key)` on success, `None` on miss.
    Done(Option<String>),
}

type ResolveSlot = Arc<tokio::sync::Mutex<ResolveOutcome>>;

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

/// Restore pure standalone resolution: empty resolver chain + default
/// place. Called between flows so a brokered key can never bleed across.
fn reset_llm_globals() {
    clear_key_resolvers();
    set_current_place(None);
}

// ---------------------------------------------------------------------------
// CP-BROKERED flow support
// ---------------------------------------------------------------------------

/// A [`PlaceAuthenticator`] that maps a single known bearer token to a
/// place and rejects everything else. The control-plane interceptor calls
/// this AFTER [`validate_bearer`], so the place it returns is the
/// server-authenticated identity the key store is scoped to.
struct TokenToPlace {
    token: &'static str,
    place: &'static str,
}

impl PlaceAuthenticator for TokenToPlace {
    fn authenticate(&self, bearer: Option<&str>) -> Result<PeerIdentity, String> {
        let header = bearer.ok_or_else(|| "missing bearer".to_string())?;
        let presented = header
            .strip_prefix("Bearer ")
            .ok_or_else(|| "authorization header must use Bearer scheme".to_string())?;
        if presented == self.token {
            Ok(PeerIdentity {
                place: self.place.to_string(),
                kind: PeerKind::Worker,
            })
        } else {
            Err(format!("unknown token: {presented}"))
        }
    }
}

/// Worker handler that, on its assignment, resolves a provider key through
/// the (now CP-pre-warmed) [`blazen_llm`] resolver chain and stashes the
/// outcome for the test to assert. The worker drives `set_current_place` +
/// the control-plane `pre_warm` BEFORE invoking this handler, so a cache
/// hit here proves the broker served the key.
struct ResolveProbeHandler {
    provider: String,
    resolved: ResolveSlot,
}

#[async_trait]
impl AssignmentHandler for ResolveProbeHandler {
    async fn handle(
        &self,
        assignment: Assignment,
        _ctx: AssignmentContext,
    ) -> Result<serde_json::Value, AssignmentFailure> {
        let outcome = resolve_api_key(&self.provider, None).ok();
        *self.resolved.lock().await = ResolveOutcome::Done(outcome);
        let input: serde_json::Value =
            serde_json::from_slice(&assignment.input_json).unwrap_or(serde_json::Value::Null);
        Ok(serde_json::json!({ "echo": input }))
    }
}

// ---------------------------------------------------------------------------
// LOCAL FALLBACK flow support
// ---------------------------------------------------------------------------

/// Probe that reports a single provider as locally servable.
struct ServableProbe {
    provider: &'static str,
}

#[async_trait]
impl LocalModelProbe for ServableProbe {
    async fn is_locally_servable(&self, provider: &str, _model: Option<&str>) -> bool {
        provider == self.provider
    }
}

/// A trivial [`Model`] tagged with an id so the test can prove which
/// builder produced it.
struct TaggedModel {
    id: String,
}

#[async_trait]
impl Model for TaggedModel {
    fn model_id(&self) -> &str {
        &self.id
    }

    async fn complete(
        &self,
        _request: blazen_llm::types::ModelRequest,
    ) -> Result<blazen_llm::types::ModelResponse, BlazenError> {
        Err(BlazenError::unsupported("mock"))
    }

    async fn stream(
        &self,
        _request: blazen_llm::types::ModelRequest,
    ) -> Result<
        std::pin::Pin<
            Box<
                dyn futures_util::Stream<Item = Result<blazen_llm::types::StreamChunk, BlazenError>>
                    + Send,
            >,
        >,
        BlazenError,
    > {
        Err(BlazenError::unsupported("mock"))
    }
}

/// Local factory that records that it was called and returns a model
/// tagged `local:<provider>:<model>`.
struct RecordingLocalFactory {
    called: Arc<AtomicBool>,
}

#[async_trait]
impl LocalModelFactory for RecordingLocalFactory {
    async fn build_local(
        &self,
        provider: &str,
        model: &str,
    ) -> Result<Box<dyn Model>, BlazenError> {
        self.called.store(true, Ordering::SeqCst);
        Ok(Box::new(TaggedModel {
            id: format!("local:{provider}:{model}"),
        }))
    }
}

// ---------------------------------------------------------------------------
// The one flow, three ways.
// ---------------------------------------------------------------------------

#[tokio::test]
async fn place_key_broker_three_ways() {
    flow_standalone();
    flow_cp_brokered().await;
    flow_local_fallback().await;
}

/// (i) STANDALONE — no control plane, no resolver chain.
///
/// The standalone terminal is `resolve_api_key`'s explicit→chain→env
/// cascade. With an empty chain the key resolves with zero broker
/// machinery (this is exactly how an in-process Blazen caller that holds
/// its own key gets it). We deliberately do NOT mutate the process
/// environment: `std::env::set_var` is `unsafe` under edition 2024 and
/// would require an `#[allow(unsafe_code)]` bandaid; the empty-chain
/// cascade is the same code path the env terminal feeds into, and the env
/// terminal itself is covered by the unit tests in `blazen_llm::keys`.
fn flow_standalone() {
    reset_llm_globals();

    let resolved = resolve_api_key(STANDALONE_PROVIDER, Some(STANDALONE_KEY.to_string()))
        .expect("standalone resolves with no resolver chain");
    assert_eq!(
        resolved, STANDALONE_KEY,
        "standalone key resolves through the bare terminal, no CP",
    );
    // With the chain still empty and no key, an unknown provider reaches
    // the env terminal and errors — proving nothing in the chain is
    // shadowing resolution.
    assert!(
        resolve_api_key(NO_ENV_PROVIDER, None).is_err(),
        "empty chain + no key => env terminal error (no resolver shadowing)",
    );
}

/// (ii) CP-BROKERED — the same `resolve_api_key(provider, None)` call, but
/// now the key is served by the control plane over the worker's
/// authenticated session. No env var, no worker-local key: the only
/// possible source is the broker.
async fn flow_cp_brokered() {
    reset_llm_globals();

    // Per-place key store seeded on disk: `<dir>/acme/openai`.
    let keys_dir = tempfile::tempdir().expect("tempdir");
    seed_key(keys_dir.path(), BROKER_PLACE, BROKER_PROVIDER, BROKERED_KEY).await;
    let key_store: Arc<dyn KeyStore> =
        Arc::new(EnvFileKeyStore::with_keys_dir(keys_dir.path().to_path_buf()));

    // Sanity: the store resolves the seeded key for the right place and
    // refuses it for a different place (tenant isolation at the store).
    let probe: SharedKey = key_store
        .get_key(BROKER_PLACE, BROKER_PROVIDER)
        .await
        .expect("store ok")
        .expect("seeded key present");
    assert_eq!(probe.value, BROKERED_KEY);
    assert!(
        key_store
            .get_key("someone-else", BROKER_PROVIDER)
            .await
            .expect("store ok")
            .is_none(),
        "store must not serve acme's key to another place",
    );

    // Install the token→place authenticator (process-global, set-once).
    // This binary installs exactly one, mapping BROKER_TOKEN → acme.
    blazen_controlplane::auth::install_place_authenticator(Arc::new(TokenToPlace {
        token: BROKER_TOKEN,
        place: BROKER_PLACE,
    }));

    let addr = ephemeral_addr().await;
    let server = ControlPlaneServer::new("cp-broker").with_key_store(key_store);
    let server_handle = tokio::spawn(async move {
        let _ = server.serve(addr).await;
    });
    tokio::time::sleep(Duration::from_millis(150)).await;

    // Worker authenticates as place acme (via its bearer) and pre-warms
    // the openai key on each assignment.
    let cfg = WorkerConfig::new(format!("http://{addr}"), "worker-acme")
        .with_capability(WorkerCapability {
            kind: "workflow:broker".into(),
            version: 0,
        })
        .with_admission(AdmissionMode::Fixed { max_in_flight: 2 })
        .with_place(BROKER_PLACE)
        .with_bearer_token(BROKER_TOKEN)
        .with_prewarm_provider(BROKER_PROVIDER)
        .with_heartbeat_interval(Duration::from_millis(200));
    let worker = Worker::connect(cfg).expect("validate worker config");

    let resolved_slot: ResolveSlot = Arc::new(tokio::sync::Mutex::new(ResolveOutcome::Pending));
    let handler = ResolveProbeHandler {
        provider: BROKER_PROVIDER.to_string(),
        resolved: resolved_slot.clone(),
    };
    let run_handle = tokio::spawn(async move { worker.run(handler).await });
    tokio::time::sleep(Duration::from_millis(100)).await;

    // Submit through the orchestrator-side tonic client, authenticating as
    // the same place so the assignment routes to our worker.
    let channel = Channel::from_shared(format!("http://{addr}"))
        .unwrap()
        .connect()
        .await
        .expect("orchestrator connect");
    let mut client = client_as(channel, BROKER_TOKEN);

    let run_id = submit_workflow(&mut client, "broker", b"{\"n\":1}").await;

    // The worker resolves the key inside the handler. Wait for it.
    let resolved_outcome = wait_resolved(&resolved_slot).await;
    assert_eq!(
        resolved_outcome.as_deref(),
        Some(BROKERED_KEY),
        "worker resolved the brokered key (no env, no local key) — it came from the CP",
    );

    // And the run completes cleanly.
    let final_status = poll_until_terminal(&mut client, run_id).await;
    assert_eq!(final_status.status, RunStatusWire::Completed);

    run_handle.abort();
    server_handle.abort();
}

/// (iii) LOCAL FALLBACK — no key anywhere. `build_model` with `WhenNoKey`
/// plus a servable probe routes to the local factory and the remote
/// builder is NEVER called.
async fn flow_local_fallback() {
    reset_llm_globals();

    let factory_called = Arc::new(AtomicBool::new(false));
    let remote_called = Arc::new(AtomicBool::new(false));
    let remote_called_for_closure = remote_called.clone();

    let model = build_model(
        NO_ENV_PROVIDER,
        ProviderOptions::default(),
        FallbackPolicy::WhenNoKey,
        &ServableProbe {
            provider: NO_ENV_PROVIDER,
        },
        &RecordingLocalFactory {
            called: factory_called.clone(),
        },
        move |_key, _opts| {
            remote_called_for_closure.store(true, Ordering::SeqCst);
            Ok(Box::new(TaggedModel {
                id: "remote".into(),
            }) as Box<dyn Model>)
        },
    )
    .await
    .expect("local fallback builds a model");

    assert_eq!(
        model.model_id(),
        format!("local:{NO_ENV_PROVIDER}:default"),
        "build_model routed to the local factory",
    );
    assert!(
        factory_called.load(Ordering::SeqCst),
        "local factory must have been invoked",
    );
    assert!(
        !remote_called.load(Ordering::SeqCst),
        "remote builder must NOT have been called (no network)",
    );

    reset_llm_globals();
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Seed a per-place provider key file at `<dir>/<place>/<provider>`.
async fn seed_key(dir: &std::path::Path, place: &str, provider: &str, key: &str) {
    let place_dir = dir.join(place);
    tokio::fs::create_dir_all(&place_dir)
        .await
        .expect("mkdir place dir");
    tokio::fs::write(place_dir.join(provider), format!("{key}\n"))
        .await
        .expect("seed key file");
}

/// Build a tonic client whose every request carries `Bearer <token>`.
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

/// Submit a workflow and return its run id.
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
            "handler never resolved the key within 5s",
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
