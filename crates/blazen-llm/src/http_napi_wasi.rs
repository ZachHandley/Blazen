//! Wasi-only HTTP client registration hook.
//!
//! On wasi, [`crate::default_http_client`] returns a [`LazyHttpClient`] wrapper
//! that forwards every request to whatever client was registered via
//! [`register_default`]. The host (Node-on-wasi / Cloudflare Workers / Deno)
//! registers a fetch-backed client at module load via the
//! `setDefaultHttpClient()` napi binding. If no client is registered before a
//! request fires, every method returns a [`BlazenError::Unsupported`] with a
//! clear instruction.

use std::sync::{Arc, OnceLock};

use crate::error::BlazenError;
use crate::http::{ByteStream, HttpClient, HttpRequest, HttpResponse};

static DEFAULT: OnceLock<Arc<dyn HttpClient>> = OnceLock::new();

/// Register the process-wide default HTTP client.
///
/// # Errors
///
/// Returns `Err(client)` (carrying back the rejected client) if a client has
/// already been registered. This mirrors [`OnceLock::set`] semantics: the
/// first writer wins, and subsequent attempts get their value back.
pub fn register_default(client: Arc<dyn HttpClient>) -> Result<(), Arc<dyn HttpClient>> {
    DEFAULT.set(client)
}

/// Read the registered client, if any.
#[allow(dead_code)] // called by Wave 5E (setDefaultHttpClient napi binding) in blazen-node.
#[must_use]
pub fn registered() -> Option<&'static Arc<dyn HttpClient>> {
    DEFAULT.get()
}

/// HttpClient wrapper that lazily delegates to the registered default client.
/// Returns an unsupported-operation error from every method if no client is
/// registered.
#[derive(Debug, Default)]
pub struct LazyHttpClient;

impl LazyHttpClient {
    /// Construct a new lazy proxy. Cheap — holds no state.
    #[must_use]
    pub fn new() -> Self {
        Self
    }

    /// Box this client into a trait object suitable for
    /// [`crate::default_http_client`]'s return type.
    #[must_use]
    pub fn into_arc(self) -> Arc<dyn HttpClient> {
        Arc::new(self)
    }

    fn resolve(&self) -> Result<&'static Arc<dyn HttpClient>, BlazenError> {
        DEFAULT.get().ok_or_else(|| {
            BlazenError::unsupported(
                "blazen-llm: no HTTP client registered. \
                 Call setDefaultHttpClient(HttpClient.fromCallback(...)) before \
                 constructing cloud LLM providers, OTLP/Langfuse exporters, or \
                 distributed peer clients on wasi (Cloudflare Workers / Deno).",
            )
        })
    }
}

#[async_trait::async_trait]
impl HttpClient for LazyHttpClient {
    async fn send(&self, request: HttpRequest) -> Result<HttpResponse, BlazenError> {
        self.resolve()?.send(request).await
    }

    async fn send_streaming(
        &self,
        request: HttpRequest,
    ) -> Result<(u16, Vec<(String, String)>, ByteStream), BlazenError> {
        self.resolve()?.send_streaming(request).await
    }
}
