//! [`HttpClient`] implementation backed by [`reqwest::Client`].
//!
//! This module is only compiled when the `reqwest` dependency is available
//! (i.e. when any provider feature is enabled).

use futures_util::StreamExt;
use std::sync::Arc;

use crate::error::BlazenError;
use crate::http::{ByteStream, HttpClient, HttpMethod, HttpRequest, HttpResponse};

/// An [`HttpClient`] backed by [`reqwest::Client`].
#[derive(Debug, Clone)]
pub struct ReqwestHttpClient {
    client: reqwest::Client,
}

impl ReqwestHttpClient {
    /// Create a new client with a default [`reqwest::Client`].
    #[must_use]
    pub fn new() -> Self {
        Self {
            client: reqwest::Client::new(),
        }
    }

    /// Create a new client wrapping an existing [`reqwest::Client`].
    #[must_use]
    pub fn from_client(client: reqwest::Client) -> Self {
        Self { client }
    }

    /// Build a [`reqwest::RequestBuilder`] from our abstract [`HttpRequest`].
    fn build_request(&self, request: &HttpRequest) -> reqwest::RequestBuilder {
        let mut builder = match request.method {
            HttpMethod::Get => self.client.get(&request.url),
            HttpMethod::Post => self.client.post(&request.url),
            HttpMethod::Put => self.client.put(&request.url),
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

#[async_trait::async_trait]
impl HttpClient for ReqwestHttpClient {
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
