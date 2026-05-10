//! Upstash Redis REST-backed memory backend for the WASM SDK.
//!
//! Wraps [`blazen_memory_valkey::UpstashBackend`] using the SDK's host
//! `HttpClient` (the browser `fetch()`-backed
//! [`blazen_llm::FetchHttpClient`]). Wasi-compatible alternative to a raw-TCP
//! Redis backend for browsers, Cloudflare Workers, Deno, and other hosts that
//! cannot open arbitrary TCP sockets.
//!
//! ```javascript
//! import { UpstashBackend, Memory, EmbeddingModel } from '@blazen-dev/sdk';
//!
//! const backend = UpstashBackend.create(
//!     "https://us1-merry-cat-32242.upstash.io",
//!     "AYAg...",
//! );
//! const memory = Memory.fromBackend(EmbeddingModel.openai(), backend);
//! ```

use std::sync::Arc;

use wasm_bindgen::prelude::*;

use blazen_llm::FetchHttpClient;
use blazen_llm::http::HttpClient;
use blazen_memory_valkey::UpstashBackend;

/// Upstash Redis REST-backed memory backend.
///
/// Talks to Upstash's REST API over a `fetch()`-backed [`HttpClient`], so it
/// works in any host that supports `fetch` (browsers, Cloudflare Workers,
/// Deno, modern Node).
#[wasm_bindgen(js_name = "UpstashBackend")]
pub struct WasmUpstashBackend {
    // Held for the future `Memory.fromBackend` / `Memory.withUpstash` wiring.
    // Not read yet from Rust — it is exposed to JS through the
    // `#[wasm_bindgen]` factory below — so silence the dead-code lint until
    // the bridge call site lands.
    #[allow(dead_code)]
    pub(crate) inner: Arc<UpstashBackend>,
}

// SAFETY: WASM is single-threaded.
unsafe impl Send for WasmUpstashBackend {}
unsafe impl Sync for WasmUpstashBackend {}

#[wasm_bindgen(js_class = "UpstashBackend")]
#[allow(clippy::needless_pass_by_value)]
impl WasmUpstashBackend {
    /// Create a new Upstash REST backend.
    ///
    /// @param restUrl   - Upstash REST endpoint
    ///                    (e.g. `https://us1-merry-cat-32242.upstash.io`).
    /// @param restToken - REST token, sent as a `Bearer` token on every request.
    /// @param prefix    - Optional key prefix override. Defaults to
    ///                    `blazen:memory:`. Useful when running multiple
    ///                    logical stores against the same Upstash database.
    #[wasm_bindgen(js_name = "create")]
    #[must_use]
    pub fn create(
        rest_url: String,
        rest_token: String,
        prefix: Option<String>,
    ) -> WasmUpstashBackend {
        let http: Arc<dyn HttpClient> = FetchHttpClient::new().into_arc();
        let mut backend = UpstashBackend::new(rest_url, rest_token, http);
        if let Some(p) = prefix {
            backend = backend.with_prefix(p);
        }
        Self {
            inner: Arc::new(backend),
        }
    }
}
