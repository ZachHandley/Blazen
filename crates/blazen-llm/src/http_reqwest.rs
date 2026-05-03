//! [`HttpClient`] implementation backed by [`reqwest::Client`].
//!
//! This module is only compiled when the `reqwest` dependency is available
//! (i.e. when any provider feature is enabled).

use futures_util::StreamExt;
use std::sync::Arc;

use crate::error::BlazenError;
use crate::http::{
    ByteStream, HttpClient, HttpClientConfig, HttpMethod, HttpRequest, HttpResponse,
};

/// An [`HttpClient`] backed by [`reqwest::Client`].
#[derive(Debug, Clone)]
pub struct ReqwestHttpClient {
    client: reqwest::Client,
    config: HttpClientConfig,
}

impl ReqwestHttpClient {
    /// Create with default timeouts (60s request, 10s connect).
    #[must_use]
    pub fn new() -> Self {
        Self::with_config(HttpClientConfig::default())
    }

    /// Create with the given timeout / user-agent configuration.
    #[must_use]
    pub fn with_config(config: HttpClientConfig) -> Self {
        let client = build_client(&config);
        Self { client, config }
    }

    /// Wrap an existing `reqwest::Client`. The `HttpClientConfig` returned
    /// by [`HttpClient::config`] will be `HttpClientConfig::default()` —
    /// callers using this constructor are responsible for matching their
    /// own client config with the values they want surfaced upstream.
    #[must_use]
    pub fn from_client(client: reqwest::Client) -> Self {
        Self {
            client,
            config: HttpClientConfig::default(),
        }
    }

    /// Wrap an existing `reqwest::Client` together with the
    /// [`HttpClientConfig`] that was used (or should be advertised) for it.
    #[must_use]
    pub fn from_client_and_config(client: reqwest::Client, config: HttpClientConfig) -> Self {
        Self { client, config }
    }

    /// Build a [`reqwest::RequestBuilder`] from our abstract [`HttpRequest`].
    fn build_request(&self, request: &HttpRequest) -> reqwest::RequestBuilder {
        let mut builder = match request.method {
            HttpMethod::Get => self.client.get(&request.url),
            HttpMethod::Post => self.client.post(&request.url),
            HttpMethod::Put => self.client.put(&request.url),
            HttpMethod::Delete => self.client.delete(&request.url),
            HttpMethod::Patch => self.client.patch(&request.url),
        };

        for (k, v) in &request.headers {
            builder = builder.header(k.as_str(), v.as_str());
        }

        for (k, v) in &request.query_params {
            builder = builder.query(&[(k.as_str(), v.as_str())]);
        }

        if let Some(ref body) = request.body {
            builder = builder.body(body.clone());
        }

        builder
    }

    /// Wrap `Self` in an `Arc` for use as `Arc<dyn HttpClient>`.
    #[must_use]
    pub fn into_arc(self) -> Arc<dyn HttpClient> {
        Arc::new(self)
    }
}

impl Default for ReqwestHttpClient {
    fn default() -> Self {
        Self::new()
    }
}

fn build_client(config: &HttpClientConfig) -> reqwest::Client {
    let mut builder = reqwest::Client::builder();
    if let Some(d) = config.request_timeout {
        builder = builder.timeout(d);
    }
    if let Some(d) = config.connect_timeout {
        builder = builder.connect_timeout(d);
    }
    if let Some(ref ua) = config.user_agent {
        builder = builder.user_agent(ua);
    }
    builder.build().unwrap_or_else(|_| reqwest::Client::new())
}

#[async_trait::async_trait]
impl HttpClient for ReqwestHttpClient {
    fn config(&self) -> &HttpClientConfig {
        &self.config
    }

    async fn send(&self, request: HttpRequest) -> Result<HttpResponse, BlazenError> {
        let builder = self.build_request(&request);
        let response = builder
            .send()
            .await
            .map_err(|e| BlazenError::request(e.to_string()))?;

        let status = response.status().as_u16();
        let headers: Vec<(String, String)> = response
            .headers()
            .iter()
            .map(|(k, v)| (k.as_str().to_owned(), v.to_str().unwrap_or("").to_owned()))
            .collect();
        let body = response
            .bytes()
            .await
            .map_err(|e| BlazenError::request(e.to_string()))?
            .to_vec();

        Ok(HttpResponse {
            status,
            headers,
            body,
        })
    }

    async fn send_streaming(
        &self,
        request: HttpRequest,
    ) -> Result<(u16, Vec<(String, String)>, ByteStream), BlazenError> {
        let builder = self.build_request(&request);
        let response = builder
            .send()
            .await
            .map_err(|e| BlazenError::request(e.to_string()))?;

        let status = response.status().as_u16();
        let headers: Vec<(String, String)> = response
            .headers()
            .iter()
            .map(|(k, v)| (k.as_str().to_owned(), v.to_str().unwrap_or("").to_owned()))
            .collect();

        // Map reqwest::Error -> Box<dyn Error + Send + Sync>
        let byte_stream = response.bytes_stream().map(|result| {
            result.map_err(|e| -> Box<dyn std::error::Error + Send + Sync> { Box::new(e) })
        });

        Ok((status, headers, Box::pin(byte_stream)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn config_round_trips_through_with_config() {
        let cfg = HttpClientConfig {
            request_timeout: Some(Duration::from_secs(7)),
            connect_timeout: Some(Duration::from_secs(3)),
            user_agent: Some("test-agent/1.0".to_owned()),
        };
        let client = ReqwestHttpClient::with_config(cfg.clone());
        assert_eq!(client.config().request_timeout, cfg.request_timeout);
        assert_eq!(client.config().connect_timeout, cfg.connect_timeout);
        assert_eq!(client.config().user_agent, cfg.user_agent);
    }

    #[test]
    fn unlimited_disables_timeouts() {
        let client = ReqwestHttpClient::with_config(HttpClientConfig::unlimited());
        assert!(client.config().request_timeout.is_none());
        assert!(client.config().connect_timeout.is_none());
    }

    #[test]
    fn default_construction_applies_default_timeouts() {
        let client = ReqwestHttpClient::new();
        assert_eq!(
            client.config().request_timeout,
            Some(Duration::from_mins(1))
        );
        assert_eq!(
            client.config().connect_timeout,
            Some(Duration::from_secs(10))
        );
    }
}
