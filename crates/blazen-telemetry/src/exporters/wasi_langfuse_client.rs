//! Wasi Langfuse exporter dispatcher and `init_langfuse`.
//!
//! Mirrors the export protocol of the native (`reqwest`-backed)
//! [`super::langfuse`] exporter, but routes batches through
//! `Arc<dyn blazen_llm::http::HttpClient>` so the wasi build (Cloudflare
//! Workers / Deno via napi-rs's wasi runtime) can ship Langfuse traces
//! without `reqwest` (no socket access on wasi) or `web_sys` (no DOM
//! bindings on wasi).
//!
//! The host registers a fetch-backed `HttpClient` via
//! `setDefaultHttpClient(...)` at module load (see the `blazen-node` napi
//! binding); this dispatcher pulls it through
//! [`blazen_llm::http_napi_wasi::LazyHttpClient`].
//!
//! Wire format (`POST {host}/api/public/ingestion`, JSON body
//! `{ "batch": [...] }`, HTTP Basic auth with `public_key`/`secret_key`)
//! matches `super::langfuse::send_batch` exactly.

use std::sync::Arc;

use base64::Engine as _;
use base64::engine::general_purpose::STANDARD as BASE64;
use tokio::sync::mpsc;

use blazen_llm::http::{HttpClient as BlazenHttpClient, HttpRequest};
use blazen_llm::http_napi_wasi::LazyHttpClient;

use super::langfuse::{Envelope, LangfuseConfig, LangfuseLayer};
use crate::error::TelemetryError;

const DEFAULT_HOST: &str = "https://cloud.langfuse.com";

// ---------------------------------------------------------------------------
// Dispatcher
// ---------------------------------------------------------------------------

/// Spawn the wasi dispatcher task that drains the envelope channel into
/// Langfuse `POST /api/public/ingestion` calls.
///
/// On wasi we don't have `tokio::time::interval` (no time driver in the
/// napi-rs wasi runtime), so the dispatcher batches strictly by `batch_size`
/// and on channel close. Callers who want tight latency can build the
/// config with `with_batch_size(1)`.
fn spawn_dispatcher(
    client: Arc<dyn BlazenHttpClient>,
    host: String,
    public_key: String,
    secret_key: String,
    batch_size: usize,
    _flush_interval_ms: u64,
    mut rx: mpsc::UnboundedReceiver<Envelope>,
) -> Result<(), TelemetryError> {
    // `tokio::spawn` needs an active runtime; surface that as a typed error.
    let handle = tokio::runtime::Handle::try_current()
        .map_err(|e| TelemetryError::Langfuse(format!("no tokio runtime available: {e}")))?;

    handle.spawn(async move {
        let mut buffer: Vec<Envelope> = Vec::with_capacity(batch_size);
        while let Some(env) = rx.recv().await {
            buffer.push(env);
            if buffer.len() >= batch_size {
                let drained = std::mem::take(&mut buffer);
                send_batch(&*client, &host, &public_key, &secret_key, drained).await;
            }
        }
        // Channel closed: flush whatever's left.
        if !buffer.is_empty() {
            let drained = std::mem::take(&mut buffer);
            send_batch(&*client, &host, &public_key, &secret_key, drained).await;
        }
    });

    Ok(())
}

/// POST one batch of ingestion events to Langfuse.
async fn send_batch(
    client: &dyn BlazenHttpClient,
    host: &str,
    public_key: &str,
    secret_key: &str,
    events: Vec<Envelope>,
) {
    if events.is_empty() {
        return;
    }

    let url = format!("{host}/api/public/ingestion");
    let payload = serde_json::json!({ "batch": events });

    // Manual Basic auth: `Authorization: Basic base64(user:pass)` (matches
    // what `reqwest::RequestBuilder::basic_auth` produces).
    let credential = format!("{public_key}:{secret_key}");
    let basic_auth = format!("Basic {}", BASE64.encode(credential.as_bytes()));

    let json_body = match serde_json::to_vec(&payload) {
        Ok(bytes) => bytes,
        Err(err) => {
            tracing::warn!(
                error = %err,
                "langfuse(wasi): failed to serialize batch payload"
            );
            return;
        }
    };

    let req = HttpRequest::post(&url)
        .header("Authorization", basic_auth)
        .header("Content-Type", "application/json")
        .body(json_body);

    match client.send(req).await {
        Ok(resp) if resp.is_success() => {
            tracing::debug!(count = events.len(), "langfuse(wasi): flushed batch");
        }
        Ok(resp) => {
            let status = resp.status;
            let body = resp.text();
            tracing::warn!(
                status,
                body = %body,
                "langfuse(wasi): ingestion request returned non-success status"
            );
        }
        Err(err) => {
            tracing::warn!(
                error = %err,
                "langfuse(wasi): failed to send ingestion request"
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Public constructor
// ---------------------------------------------------------------------------

/// Initialize the Langfuse exporter layer on wasi.
///
/// Returns a [`LangfuseLayer`] whose dispatcher writes Langfuse ingestion
/// batches via the host-registered `Arc<dyn blazen_llm::http::HttpClient>`
/// (typically a fetch-backed client wired up by
/// `setDefaultHttpClient(...)` in the napi binding). A background tokio
/// task is spawned to drain the channel.
///
/// # Errors
///
/// Returns [`TelemetryError::Langfuse`] when no tokio runtime is available
/// to host the background dispatcher.
pub fn init_langfuse(config: LangfuseConfig) -> Result<LangfuseLayer, TelemetryError> {
    let client: Arc<dyn BlazenHttpClient> = Arc::new(LazyHttpClient::new());

    let host = config.host.unwrap_or_else(|| DEFAULT_HOST.to_owned());
    let LangfuseConfig {
        public_key,
        secret_key,
        batch_size,
        flush_interval_ms,
        ..
    } = config;

    let (tx, rx) = mpsc::unbounded_channel::<Envelope>();

    spawn_dispatcher(
        client,
        host,
        public_key,
        secret_key,
        batch_size,
        flush_interval_ms,
        rx,
    )?;

    Ok(LangfuseLayer::from_parts(tx, batch_size))
}
