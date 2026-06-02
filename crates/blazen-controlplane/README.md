# blazen-controlplane

Distributed **workflow** control plane for Blazen: a central gRPC server
that workers connect into. The control plane owns the authoritative view
of running workflows, the connected-worker registry, and the assignment
queue. Workers and orchestrators are both *clients* of it.

> **Not a model registry.** The control plane schedules *workflows* across
> *workers*. It does **not** centrally manage LLM provider instances or
> credentials — that is the job of the per-process `ModelManager`
> (a named registry of providers living inside one process). Workers hold
> their own provider credentials locally. See the cross-reference at the
> bottom.

## Topology

Unlike [`blazen-peer`](../blazen-peer), which models a flat mesh of equal
peers that dial each other directly, the control plane is a hub-and-spoke
server. **Workers always open the connection outbound**, which makes the
system NAT-friendly: only the control plane needs a reachable address.

```text
  orchestrator ──┐
                 │  unary + server-stream RPCs
                 │  (Submit / Cancel / Describe / SubscribeRunEvents /
                 ▼   SubscribeAll / ListWorkers / DrainWorker / RespondToInput)
        ControlPlaneServer
         ┌──────────────────────────────┐
         │ WorkerRegistry  (who's online,│
         │   capability + tag indexes)   │
         │ AssignmentQueue (priority WFQ,│
         │   per-capability pending pool)│
         │ Admission       (routing)     │
         └──────────────────────────────┘
                 ▲
                 │  bidi `WorkerSession` stream
  worker ────────┘  (worker dials in; Hello → Welcome)
```

## What it manages

- **Worker registry** (`server/registry.rs`) — the source of truth for
  which workers are connected, what capabilities they advertise, their
  tags/labels/taints, and their last heartbeat. Indexed by capability so
  the queue can find candidate workers in O(1).
- **Assignment queue** (`server/queue.rs`) — a per-capability pending pool
  with a **deficit-round-robin WFQ scheduler** across 8 priority bands
  (band width 32; lower numeric `priority` = served first; weights step
  down 256 → 32 so low-priority work never starves). Optionally durable
  via a pluggable `AssignmentStore` (in-memory default, or Valkey / the durable store
  behind feature flags) so queue state survives a control-plane restart.
- **Admission / routing** (`server/admission.rs`) — decides *which* worker
  gets a submitted assignment. It does **not** hold provider credentials.

The control plane does **not** run workflows itself and does **not** store
model weights or API keys. Workers do that.

## Capability-based routing

Every assignment names a required `WorkerCapability { kind, version }`.
Routing is a stateless filter pipeline:

1. **Capability match** — keep only workers advertising the exact
   `kind@version`.
2. **Tag predicate** — drop workers whose tags don't AND-match every
   `key=value` (or `key=*`) requirement on the submit.
3. **Node selector + taints/tolerations** — `NodeSelector.required` labels
   must all match, `forbidden` must none match; `preferred` adds to the
   tie-break score. A `NoSchedule` taint without a matching toleration
   excludes the worker; `PreferNoSchedule` only penalizes its score.
4. **Capacity filter** — drop workers whose per-worker admission strategy
   has no room (see below).
5. **Tie-break** — pick the least-loaded remaining worker (round-robin on
   ties); `model_residency` affinity gives a bonus to workers that already
   have the requested model loaded.

The capability `kind` namespace (`inference:llm:*`, `inference:vision:*`,
`media:*`, `agent:*`, …), plus the node-label / taint / priority
conventions, are the **authoritative registry** in
[`docs/CAPABILITIES.md`](../../docs/CAPABILITIES.md). Use kinds from there
rather than inventing strings.

### Per-worker admission strategy (`AdmissionMode`)

Each worker declares how its capacity is bounded
(`blazen_core::distributed::AdmissionMode`):

| Mode | Meaning |
|---|---|
| `Fixed { max_in_flight }` | Hard count cap. Best for fungible CPU work. |
| `VramBudget { max_vram_mb }` | VRAM-sum cap. Requires `resource_hint.vram_mb` on the assignment. |
| `Reactive` | Worker self-decides via offer / claim / decline negotiation. |

For `Reactive` workers the server sends an `Offer`; the worker replies
`Claim` or `Decline { reason }`, and the queue re-routes to the next
candidate on decline.

## Authentication

Two mechanisms are implemented today (`auth.rs`, `tls.rs`,
`server/interceptor.rs`), shared with `blazen-peer`:

- **Bearer token** — a shared secret carried as
  `authorization: Bearer <token>` metadata on every request. The server
  installs an interceptor that calls `validate_bearer` on every inbound
  RPC and returns `unauthenticated` on mismatch (constant-time compare).
  The token is read from `BLAZEN_PEER_TOKEN`, or supplied explicitly:
  orchestrators via the `bearer_token` argument to `Client::connect`,
  workers via `WorkerConfig::with_bearer_token`. **If no token is
  configured server-side, auth is effectively off** (intended for
  dev / loopback only).
- **mTLS** — configure the server with `ControlPlaneServer::with_tls` and
  clients with a `ClientTlsConfig` (or the `with_mtls` PEM-loading
  helpers). Recommended for production.

> The crate-level docs (`lib.rs`) also describe three *enrollment* policies
> — **Open**, **Allowlist**, and **Signed-handshake**. Only the bearer +
> mTLS surface above is implemented today; the allowlist / signed-handshake
> modes are described there as planned and are **not** yet enforced in
> code.

## Human-in-the-loop: the input-request round-trip

A running assignment can pause and ask the orchestrator (or a human behind
it) for input, then resume with the answer:

1. **Worker** calls
   `AssignmentContext::request_input(prompt, metadata, timeout_ms)`
   (`worker.rs`). This emits an `"input.request"` event carrying
   `{ request_id, prompt, metadata }` and blocks on a oneshot keyed by
   `request_id`.
2. The control plane fans the event out to subscribers
   (`SubscribeRunEvents` / `SubscribeAll`).
3. **Orchestrator** answers with
   `Client::respond_to_input(run_id, request_id, response_json)`.
4. The control plane routes a `ServerToWorker::InputResponse` frame to the
   worker currently assigned that run.
5. The worker's inbound pump fulfils the pending oneshot, and
   `request_input` returns the JSON answer to the handler.

`request_input` returns an error if the assignment is cancelled, the
worker disconnects, or `timeout_ms` elapses with no answer.

## Wire format

Every gRPC method takes a single `PostcardRequest { bytes }` and returns a
`PostcardResponse { bytes }`. The payload structs live in `protocol.rs`
and are [postcard](https://docs.rs/postcard)-encoded. Versioning is
per-message via `ENVELOPE_VERSION` (currently `2`) rather than by evolving
the `.proto`, so new fields are a source-only change. The handshake
negotiates the intersection of `supported_envelope_versions` and rejects
when versions don't overlap.

## Minimal usage

These signatures are taken from the crate source; consult the rustdoc for
the full surface.

**Run a server** (`server` feature, default):

```rust
use blazen_controlplane::ControlPlaneServer;

let server = ControlPlaneServer::new("cp-node-1");
// .with_tls(tls_config)  // optional mTLS
// .with_store(arc_store) // optional durable AssignmentStore
server.serve("0.0.0.0:7445".parse()?).await?;
```

**Connect a worker** (`client` feature, default):

```rust
use blazen_controlplane::{Worker, WorkerConfig, AssignmentHandler,
    AssignmentContext, AssignmentFailure};
use blazen_core::distributed::{AdmissionMode, WorkerCapability};
use async_trait::async_trait;
use serde_json::Value;

struct MyHandler;

#[async_trait]
impl AssignmentHandler for MyHandler {
    async fn handle(&self, assignment: blazen_controlplane::protocol::Assignment,
        ctx: AssignmentContext) -> Result<Value, AssignmentFailure> {
        // ... run the workflow; optionally ctx.request_input(...) / ctx.emit_event(...)
        Ok(serde_json::json!({ "ok": true }))
    }
}

let config = WorkerConfig::new("http://cp-host:7445", "worker-a")
    .with_capability(WorkerCapability { kind: "inference:llm:ollama".into(), version: 1 })
    .with_tag("region", "us-west")
    .with_admission(AdmissionMode::Fixed { max_in_flight: 4 });
    // .with_bearer_token("…") / .with_label(…) / .with_taint(…) / .with_descriptor(…)

Worker::connect(config)?.run(MyHandler).await?;
```

**Drive it from an orchestrator** (`client` feature) via `Client` /
`blazen_core::distributed::OrchestratorClient`: `connect`,
`submit_workflow`, `cancel_workflow`, `describe_workflow`,
`subscribe_run_events`, `list_workers`, plus `respond_to_input` and
`drain_worker` on `Client` directly.

## Feature flags

- `server` (default) — the tonic server-side service.
- `client` (default) — tonic client stubs for workers and orchestrators.
- `http-transport` — axum-based HTTP/SSE bridge for environments that
  cannot speak HTTP/2 (browsers, some serverless / wasi targets).
- `http-rest` — OpenAI-compatible REST + Blazen-admin endpoints.
- `valkey-store` / `durable-store` — durable `AssignmentStore` backends.
- `model-server` / `model-client` — the separate `blazen.modelserver.v1`
  service (remote `ModelManager`), independent of the workflow control
  plane.

## Related

- **Capability namespace registry:** [`docs/CAPABILITIES.md`](../../docs/CAPABILITIES.md).
- **Flat peer mesh (the alternative topology):** [`blazen-peer`](../blazen-peer).
- **Managing provider *instances* in one process:** that's the
  `ModelManager` (a per-process named registry of providers) — see the
  Providers / `ModelManager` docs. The control plane does not do this.
</content>
</invoke>
