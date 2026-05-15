//! Runtime fetch of pricing data from a remote endpoint (typically the
//! `blazen.dev` Cloudflare Worker fronting a KV-backed catalog).
//!
//! This is the *runtime* counterpart to [`crate::pricing::default_pricing`]
//! (which is the snapshot baked at build time). The full chain when an app
//! looks up a model:
//!
//! 1. Live `/models` responses for `OpenRouter` / Together overwrite the
//!    registry as those providers are first touched.
//! 2. If the app called `refresh_default` (or `fetch_one_default`) at
//!    startup, the Cloudflare Worker has already populated everything it
//!    knows about (zero `models.dev` traffic).
//! 3. Otherwise the build-time baked snapshot answers the query.
//!
//! Misses still return `None` from [`compute_cost`](crate::pricing::compute_cost)
//! — the cost path stays synchronous on purpose. Apps wanting "fetch on miss"
//! call `fetch_one_default` explicitly when they see a `None`.

use std::collections::HashMap;
use std::sync::Arc;

use serde::Deserialize;

use crate::error::BlazenError;
use crate::http::{HttpClient, HttpRequest};
use crate::pricing::{PricingEntry, register_pricing};

/// Bulk endpoint: returns the entire catalog as a single JSON document. Use
/// for one-shot startup hydration.
pub const DEFAULT_PRICING_URL: &str = "https://blazen.dev/api/pricing.json";

/// Per-model endpoint base. Append a normalized model id (after `/`) to get
/// the URL for a single entry. Use when you only need one model and want to
/// avoid downloading the whole catalog.
pub const DEFAULT_MODEL_PRICING_URL_BASE: &str = "https://blazen.dev/api/pricing/model/";

/// Wire format for the bulk endpoint. Mirrors the on-disk
/// `crates/blazen-llm/data/pricing.json` schema; `$`-prefixed metadata keys
/// at the top level are simply not deserialized here.
#[derive(Deserialize)]
struct PricingFile {
    models: HashMap<String, PricingEntry>,
}

/// Fetch the bulk catalog from `url` and bulk-register every entry into the
/// global pricing registry. Returns the number of entries registered.
///
/// Entries are normalized inside [`register_pricing`] before storage, so
/// either pre- or post-normalized keys in the response are handled correctly.
///
/// # Errors
/// Returns [`BlazenError`] if the HTTP request fails, returns a non-2xx
/// status, or the body cannot be parsed as the expected schema.
pub async fn refresh_from_url(
    client: Arc<dyn HttpClient>,
    url: &str,
) -> Result<usize, BlazenError> {
    let response = client.send(HttpRequest::get(url)).await?;
    if !response.is_success() {
        return Err(BlazenError::invalid_response(format!(
            "pricing fetch from {url} returned HTTP {}",
            response.status
        )));
    }
    let file: PricingFile = response.json()?;
    let count = file.models.len();
    for (id, entry) in file.models {
        register_pricing(&id, entry);
    }
    Ok(count)
}

/// Fetch a single model's pricing from `{url_base}{model_id}` and register
/// it. Returns the registered entry, or `Ok(None)` on a 404 (so callers can
/// distinguish "no such model" from a transport failure).
///
/// `url_base` is expected to already include a trailing slash (e.g.
/// [`DEFAULT_MODEL_PRICING_URL_BASE`]). `model_id` is appended verbatim;
/// callers should pass an already-normalized id.
///
/// # Errors
/// Returns [`BlazenError`] for network errors, non-(2xx|404) responses, or
/// JSON parse failures.
pub async fn fetch_one_from_url(
    client: Arc<dyn HttpClient>,
    url_base: &str,
    model_id: &str,
) -> Result<Option<PricingEntry>, BlazenError> {
    let url = format!("{url_base}{model_id}");
    let response = client.send(HttpRequest::get(&url)).await?;
    if response.status == 404 {
        return Ok(None);
    }
    if !response.is_success() {
        return Err(BlazenError::invalid_response(format!(
            "pricing fetch from {url} returned HTTP {}",
            response.status
        )));
    }
    let entry: PricingEntry = response.json()?;
    register_pricing(model_id, entry);
    Ok(Some(entry))
}

/// Convenience: bulk refresh from `url` using the platform-default HTTP
/// client (`reqwest` on native, `fetch` in browsers, host-bridged client
/// on WASI). Bindings call this rather than [`refresh_from_url`] so they
/// don't have to construct a client themselves.
///
/// # Errors
/// Same as [`refresh_from_url`].
#[cfg(any(
    all(not(target_arch = "wasm32"), feature = "reqwest"),
    target_arch = "wasm32"
))]
pub async fn refresh_default_with_url(url: &str) -> Result<usize, BlazenError> {
    let client = crate::default_http_client();
    refresh_from_url(client, url).await
}

/// Convenience: bulk refresh from [`DEFAULT_PRICING_URL`].
///
/// # Errors
/// Same as [`refresh_from_url`].
#[cfg(any(
    all(not(target_arch = "wasm32"), feature = "reqwest"),
    target_arch = "wasm32"
))]
pub async fn refresh_default() -> Result<usize, BlazenError> {
    refresh_default_with_url(DEFAULT_PRICING_URL).await
}

/// Convenience: per-model fetch from `url_base` (with the model id
/// appended) using the platform-default HTTP client.
///
/// # Errors
/// Same as [`fetch_one_from_url`].
#[cfg(any(
    all(not(target_arch = "wasm32"), feature = "reqwest"),
    target_arch = "wasm32"
))]
pub async fn fetch_one_default_with_url_base(
    url_base: &str,
    model_id: &str,
) -> Result<Option<PricingEntry>, BlazenError> {
    let client = crate::default_http_client();
    fetch_one_from_url(client, url_base, model_id).await
}

/// Convenience: per-model fetch from [`DEFAULT_MODEL_PRICING_URL_BASE`].
///
/// # Errors
/// Same as [`fetch_one_from_url`].
#[cfg(any(
    all(not(target_arch = "wasm32"), feature = "reqwest"),
    target_arch = "wasm32"
))]
pub async fn fetch_one_default(model_id: &str) -> Result<Option<PricingEntry>, BlazenError> {
    fetch_one_default_with_url_base(DEFAULT_MODEL_PRICING_URL_BASE, model_id).await
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::http::{ByteStream, HttpClientConfig, HttpResponse};
    use async_trait::async_trait;
    use std::sync::Mutex;

    #[derive(Debug)]
    struct StubHttpClient {
        body: Mutex<Option<(u16, String)>>,
        last_url: Mutex<Option<String>>,
        config: HttpClientConfig,
    }

    impl StubHttpClient {
        fn new(status: u16, body: impl Into<String>) -> Arc<Self> {
            Arc::new(Self {
                body: Mutex::new(Some((status, body.into()))),
                last_url: Mutex::new(None),
                config: HttpClientConfig::default(),
            })
        }
    }

    #[async_trait]
    impl HttpClient for StubHttpClient {
        async fn send(&self, request: HttpRequest) -> Result<HttpResponse, BlazenError> {
            *self.last_url.lock().unwrap() = Some(request.url.clone());
            let (status, body) = self.body.lock().unwrap().clone().unwrap();
            Ok(HttpResponse {
                status,
                body: body.into_bytes(),
                headers: vec![],
            })
        }

        async fn send_streaming(
            &self,
            _request: HttpRequest,
        ) -> Result<(u16, Vec<(String, String)>, ByteStream), BlazenError> {
            unimplemented!("not used in pricing tests")
        }

        fn config(&self) -> &HttpClientConfig {
            &self.config
        }
    }

    #[tokio::test]
    async fn refresh_from_url_registers_entries() {
        let body = r#"{
            "$schema_version": 1,
            "models": {
                "test-fetch-model-bulk": { "input_per_million": 7.5, "output_per_million": 22.5 }
            }
        }"#;
        let client = StubHttpClient::new(200, body);
        let count = refresh_from_url(client.clone(), "https://example.test/pricing.json")
            .await
            .unwrap();
        assert_eq!(count, 1);
        let registered = crate::pricing::lookup_pricing("test-fetch-model-bulk").unwrap();
        assert!((registered.input_per_million - 7.5).abs() < f64::EPSILON);
        assert!((registered.output_per_million - 22.5).abs() < f64::EPSILON);
        assert_eq!(
            client.last_url.lock().unwrap().as_deref(),
            Some("https://example.test/pricing.json"),
        );
    }

    #[tokio::test]
    async fn refresh_from_url_propagates_http_error() {
        let client = StubHttpClient::new(500, "boom");
        let err = refresh_from_url(client, "https://example.test/pricing.json")
            .await
            .unwrap_err();
        assert!(err.to_string().contains("HTTP 500"));
    }

    #[tokio::test]
    async fn fetch_one_returns_none_on_404() {
        let client = StubHttpClient::new(404, "");
        let got = fetch_one_from_url(client, "https://example.test/model/", "ghost-model")
            .await
            .unwrap();
        assert!(got.is_none());
    }

    #[tokio::test]
    async fn fetch_one_registers_and_returns_entry() {
        let body = r#"{ "input_per_million": 1.0, "output_per_million": 4.0 }"#;
        let client = StubHttpClient::new(200, body);
        let entry = fetch_one_from_url(client, "https://example.test/model/", "test-fetch-one")
            .await
            .unwrap()
            .unwrap();
        assert!((entry.input_per_million - 1.0).abs() < f64::EPSILON);
        assert!((entry.output_per_million - 4.0).abs() < f64::EPSILON);
        let registered = crate::pricing::lookup_pricing("test-fetch-one").unwrap();
        assert!((registered.input_per_million - 1.0).abs() < f64::EPSILON);
    }
}
