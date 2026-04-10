# blazen-peer

Distributed sub-workflow execution layer for Blazen. This crate lets one Blazen process invoke workflows on another over gRPC, with optional mTLS, postcard wire encoding, and transparent session-ref proxying across machines.

## Architecture

A parent workflow running on machine A can delegate a sub-workflow to machine B:

1. **Machine A (client)** builds a `SubWorkflowRequest` containing the workflow name, an ordered list of step IDs, and a JSON input payload. It sends the request over a tonic gRPC channel to machine B.
2. **Machine B (server)** resolves each step ID against its local `blazen_core::step_registry`, assembles a `Workflow`, and runs it to completion. Any session refs the sub-workflow produces stay in the server's `SessionRefRegistry`.
3. **Machine B** returns a `SubWorkflowResponse` containing the terminal result, exported state values, and a set of `RemoteRefDescriptor` handles for session refs that could not be serialized inline.
4. **Machine A** can dereference those remote refs lazily over the same gRPC channel via `DerefSessionRef`, and release them via `ReleaseSessionRef` when done.

Session refs never leave the peer that created them. The parent holds lightweight `RemoteRefDescriptor` proxy handles and fetches the underlying bytes on demand.

### Wire format

The gRPC schema (`proto/blazen_peer.proto`) is intentionally minimal: every RPC takes a `PostcardRequest { bytes postcard_payload }` and returns a `PostcardResponse { bytes postcard_payload }`. The actual payload types live in `protocol.rs` and are serialized with [postcard](https://docs.rs/postcard). This means adding a field to `SubWorkflowRequest` never requires regenerating proto bindings -- versioning is handled by the `ENVELOPE_VERSION` constant on each payload.

## Tech stack

| Component | Crate | Version |
|---|---|---|
| gRPC transport | `tonic` | 0.14 |
| Wire encoding | `postcard` | 1.1 |
| TLS | `rustls` (aws-lc-rs backend) | 0.23 |
| Proto codegen | `tonic-prost-build` | 0.14 |

## Quick start

### Server side

Register your steps in the global step registry, then start the peer server:

```rust
use std::sync::Arc;
use blazen_core::register_step_builder;
use blazen_peer::BlazenPeerServer;

// Register steps that remote peers can invoke.
register_step_builder("my_app::analyze", my_analyze_step_builder);
register_step_builder("my_app::summarize", my_summarize_step_builder);

let server = BlazenPeerServer::new("node-1");

// Optionally share an existing SessionRefRegistry with in-process workflows.
// let server = server.with_session_refs(Arc::new(registry));

server.serve("0.0.0.0:50051".parse()?).await?;
```

### Client side

Connect to a remote peer and invoke a sub-workflow:

```rust
use blazen_peer::BlazenPeerClient;
use blazen_peer::SubWorkflowRequest;

let mut client = BlazenPeerClient::connect("http://peer-b:50051", "node-a").await?;

let input = serde_json::json!({ "document": "..." });
let request = SubWorkflowRequest::new(
    "analyze-wf",
    vec!["my_app::analyze".to_string(), "my_app::summarize".to_string()],
    &input,
    Some(60), // timeout in seconds
)?;

let response = client.invoke_sub_workflow(request).await?;

// Check for errors.
if let Some(err) = &response.error {
    eprintln!("remote workflow failed: {err}");
} else {
    // Read the terminal result.
    let result = response.result_value()?;
    println!("result: {result:?}");

    // Read exported state values.
    let state = response.state_values()?;
    println!("state: {state:?}");

    // Dereference any remote session refs.
    for (uuid, descriptor) in &response.remote_refs {
        println!("remote ref on {}: type={}", descriptor.origin_node_id, descriptor.type_tag);
        let key = blazen_core::session_ref::RegistryKey(*uuid);
        let bytes = client.deref_session_ref(key).await?;
        // Deserialize `bytes` according to `descriptor.type_tag`.
    }
}
```

## mTLS

The `tls` module provides `load_server_tls` and `load_client_tls` which read PEM-encoded certificate, key, and CA files from disk and produce tonic's `ServerTlsConfig` / `ClientTlsConfig`.

```rust
use std::path::Path;
use blazen_peer::tls::{load_server_tls, load_client_tls};

// Server side
let server_tls = load_server_tls(
    Path::new("/certs/server.crt"),
    Path::new("/certs/server.key"),
    Path::new("/certs/ca.crt"),
)?;

// Client side
let client_tls = load_client_tls(
    Path::new("/certs/client.crt"),
    Path::new("/certs/client.key"),
    Path::new("/certs/ca.crt"),
)?;
```

Both the server and any connecting clients must present certificates signed by the same CA for mutual TLS to succeed.

For Kubernetes deployments, [cert-manager](https://cert-manager.io/) can automate certificate issuance and rotation. Create a `Certificate` resource per peer pod with a shared `Issuer` CA.

When mTLS is not configured, peers can authenticate with a shared secret token via the `BLAZEN_PEER_TOKEN` environment variable. See `auth::resolve_peer_token`.

## Envelope versioning

Every wire payload carries an `envelope_version` field. The current version is `ENVELOPE_VERSION` (currently `1`).

**Forward-compatible changes** (no version bump required):
- Adding a new `Option<T>` field at the end of a struct. Postcard skips unknown trailing bytes on decode and fills missing fields with `None`.

**Breaking changes** (require a version bump):
- Renaming, reordering, or removing fields.

The server validates the envelope version on every incoming request. If the client sends a version newer than what the server supports, the server rejects the request with `FAILED_PRECONDITION`. Older versions are accepted -- the server is always backward-compatible with previous envelope versions.

## Failure modes

| Failure | Error variant | gRPC status |
|---|---|---|
| Step ID not registered on the peer | `PeerError::UnknownStep` | `NOT_FOUND` |
| Envelope version too new | `PeerError::EnvelopeVersion` | `FAILED_PRECONDITION` |
| Session ref expired or already released | -- | `NOT_FOUND` |
| Workflow execution error | `PeerError::Workflow` | returned in `SubWorkflowResponse.error` |
| Workflow timeout exceeded | -- | returned in `SubWorkflowResponse.error` |
| Network partition / connection refused | `PeerError::Transport` | transport-level failure |
| Postcard encode/decode failure | `PeerError::Encode` | `INVALID_ARGUMENT` (server) or client-side error |
| TLS misconfiguration | `PeerError::Tls` | connection refused at handshake |

## Feature flags

| Feature | Default | Description |
|---|---|---|
| `server` | yes | Builds the `BlazenPeerServer` and the tonic service implementation. Pulls in the generated `blazen_peer_server` module. |
| `client` | yes | Builds the `BlazenPeerClient`. Pulls in the generated `blazen_peer_client` module. |

Both features are independent. A node that only invokes remote workflows can disable `server`, and a worker node that only receives invocations can disable `client`:

```toml
[dependencies]
blazen-peer = { version = "...", default-features = false, features = ["client"] }
```

## Interaction with RefLifetime (Phase 11)

When a sub-workflow runs on the peer, any session refs it creates are registered in the server's `SessionRefRegistry`. The `RefLifetime` policy on each ref controls when it is purged:

- **`UntilContextDrop` (default)** -- purged when the sub-workflow's `Context` drops, i.e. immediately after the sub-workflow finishes. These refs are serialized into the `SubWorkflowResponse` if they implement `SessionRefSerializable`; otherwise they are lost.
- **`UntilParentFinish`** -- survives the sub-workflow and remains in the server's registry. The parent receives a `RemoteRefDescriptor` and can dereference the ref over gRPC via `DerefSessionRef` at any time. The ref is purged only when the parent workflow finishes and explicitly calls `ReleaseSessionRef`.
- **`UntilExplicitDrop`** -- never purged automatically. The parent must call `release_session_ref` to free it.

For distributed workflows, `UntilParentFinish` is the typical choice for values that the parent needs to access lazily (model weight caches, open file handles, GPU-resident tensors). The parent holds a lightweight `RemoteRefDescriptor` and only fetches the bytes when it actually needs them.

## License

AGPL-3.0 -- see [LICENSE](https://github.com/ZachHandley/Blazen/blob/main/LICENSE) for details.
