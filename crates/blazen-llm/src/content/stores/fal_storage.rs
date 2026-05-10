//! [`FalStorageStore`] — content store backed by fal.ai's storage endpoint.
//!
//! fal.ai exposes a two-step "initiate-upload + PUT bytes" flow at
//! `https://rest.alpha.fal.ai/storage/upload/initiate`. The initiate response
//! returns a presigned upload URL plus the publicly-accessible `file_url` that
//! models can reference. Unlike provider file APIs that hand back opaque IDs,
//! fal serves the resulting bytes as a public URL under
//! `https://v3.fal.media/files/...` — so the handle ID we record IS the URL,
//! and resolution simply hands back [`MediaSource::Url`].
//!
//! Behaviour summary:
//!
//! - [`ContentBody::Bytes`] / [`ContentBody::LocalPath`] — perform the
//!   two-step upload and record the returned `file_url` as the handle id.
//! - [`ContentBody::Url`] — record the URL verbatim; fal accepts URL refs
//!   natively and there's no value in re-uploading.
//! - [`ContentBody::ProviderFile`] with `ProviderId::Fal` — record the id
//!   verbatim (for fal it IS the file URL).
//! - [`ContentBody::ProviderFile`] with any other provider — error;
//!   wrong-provider mismatch.
//!
//! [`delete`](ContentStore::delete) is a no-op: fal.ai does not expose a
//! public delete endpoint in their documented API.

use std::sync::Arc;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::content::detect::detect;
use crate::content::handle::ContentHandle;
use crate::content::kind::ContentKind;
use crate::content::store::{ContentBody, ContentHint, ContentStore};
use crate::error::BlazenError;
use crate::http::{HttpClient, HttpRequest};
use crate::types::{MediaSource, ProviderId};

/// Default base URL for fal.ai's storage endpoint.
const DEFAULT_BASE_URL: &str = "https://rest.alpha.fal.ai";

/// Body sent to the fal `storage/upload/initiate` endpoint.
#[derive(Debug, Serialize)]
struct InitiateRequest<'a> {
    file_name: &'a str,
    content_type: &'a str,
}

/// Response from the fal `storage/upload/initiate` endpoint.
#[derive(Debug, Deserialize)]
struct InitiateResponse {
    upload_url: String,
    file_url: String,
}

/// Content store backed by fal.ai's storage endpoint.
///
/// The store performs a two-step upload (initiate → PUT bytes) and records
/// the resulting public `file_url` as the handle ID. Resolution returns
/// [`MediaSource::Url`] directly; tools that want the raw bytes can call
/// [`fetch_bytes`](ContentStore::fetch_bytes), which performs a plain GET
/// against the public URL.
#[derive(Debug)]
pub struct FalStorageStore {
    client: Arc<dyn HttpClient>,
    api_key: String,
    base_url: String,
}

impl FalStorageStore {
    /// Create a new store with the platform-default HTTP client.
    ///
    /// Available on native targets with the `reqwest` feature and on browser
    /// WASM (which uses the `fetch` backend).
    #[cfg(any(
        all(target_arch = "wasm32", not(target_os = "wasi")),
        feature = "reqwest",
        target_os = "wasi"
    ))]
    #[must_use]
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            client: crate::default_http_client(),
            api_key: api_key.into(),
            base_url: DEFAULT_BASE_URL.to_owned(),
        }
    }

    /// Create a new store with an explicit HTTP client backend.
    #[must_use]
    pub fn new_with_client(api_key: impl Into<String>, client: Arc<dyn HttpClient>) -> Self {
        Self {
            client,
            api_key: api_key.into(),
            base_url: DEFAULT_BASE_URL.to_owned(),
        }
    }

    /// Builder: override the base URL (useful for testing against a mock).
    #[must_use]
    pub fn with_base_url(mut self, url: impl Into<String>) -> Self {
        self.base_url = url.into();
        self
    }

    /// Upload bytes via the two-step initiate + PUT flow.
    ///
    /// Returns the public `file_url` from the initiate response — this is the
    /// URL that models can reference and the value we record as the handle id.
    async fn upload_bytes(
        &self,
        bytes: Vec<u8>,
        file_name: &str,
        content_type: &str,
    ) -> Result<String, BlazenError> {
        // Step 1: initiate upload.
        let initiate_url = format!("{}/storage/upload/initiate", self.base_url);
        let initiate_body = InitiateRequest {
            file_name,
            content_type,
        };
        let initiate_request = HttpRequest::post(&initiate_url)
            .header("Authorization", format!("Key {}", self.api_key))
            .json_body(&initiate_body)?;
        let initiate_response = self.client.send(initiate_request).await?;
        if !initiate_response.is_success() {
            return Err(BlazenError::provider(
                "fal",
                format!(
                    "FalStorageStore: initiate upload failed (status {}): {}",
                    initiate_response.status,
                    initiate_response.text()
                ),
            ));
        }
        let initiated: InitiateResponse = initiate_response.json()?;

        // Step 2: PUT bytes to the presigned URL.
        let put_request = HttpRequest::put(&initiated.upload_url)
            .header("Content-Type", content_type)
            .body(bytes);
        let put_response = self.client.send(put_request).await?;
        if !put_response.is_success() {
            return Err(BlazenError::provider(
                "fal",
                format!(
                    "FalStorageStore: byte upload failed (status {}): {}",
                    put_response.status,
                    put_response.text()
                ),
            ));
        }

        Ok(initiated.file_url)
    }
}

#[async_trait]
impl ContentStore for FalStorageStore {
    async fn put(
        &self,
        body: ContentBody,
        hint: ContentHint,
    ) -> Result<ContentHandle, BlazenError> {
        match body {
            ContentBody::Bytes { data: bytes } => {
                let detected = detect(
                    Some(&bytes),
                    hint.mime_type.as_deref(),
                    hint.display_name.as_deref(),
                );
                let kind = hint.kind_hint.unwrap_or(detected.0);
                let mime = hint.mime_type.clone().or_else(|| detected.1.clone());
                let content_type = mime
                    .clone()
                    .unwrap_or_else(|| "application/octet-stream".to_owned());
                let file_name = hint
                    .display_name
                    .clone()
                    .unwrap_or_else(|| "upload.bin".to_owned());
                #[allow(clippy::cast_possible_truncation)]
                let size = Some(bytes.len() as u64);

                let file_url = self.upload_bytes(bytes, &file_name, &content_type).await?;

                let mut handle = ContentHandle::new(file_url, kind);
                handle.mime_type = mime;
                handle.byte_size = size;
                handle.display_name = hint.display_name;
                Ok(handle)
            }
            ContentBody::LocalPath { path } => {
                let bytes = read_local_file(&path)?;
                let file_name = hint
                    .display_name
                    .clone()
                    .or_else(|| path.file_name().and_then(|n| n.to_str()).map(str::to_owned))
                    .unwrap_or_else(|| "upload.bin".to_owned());
                let detected = detect(Some(&bytes), hint.mime_type.as_deref(), Some(&file_name));
                let kind = hint.kind_hint.unwrap_or(detected.0);
                let mime = hint.mime_type.clone().or_else(|| detected.1.clone());
                let content_type = mime
                    .clone()
                    .unwrap_or_else(|| "application/octet-stream".to_owned());
                #[allow(clippy::cast_possible_truncation)]
                let size = Some(bytes.len() as u64);

                let file_url = self.upload_bytes(bytes, &file_name, &content_type).await?;

                let mut handle = ContentHandle::new(file_url, kind);
                handle.mime_type = mime;
                handle.byte_size = size;
                handle.display_name = hint.display_name.or(Some(file_name));
                Ok(handle)
            }
            ContentBody::Url { url } => {
                // fal accepts URL refs natively — record verbatim.
                let kind = hint
                    .kind_hint
                    .unwrap_or_else(|| detect(None, hint.mime_type.as_deref(), Some(&url)).0);
                let mut handle = ContentHandle::new(url, kind);
                handle.mime_type = hint.mime_type;
                handle.byte_size = hint.byte_size;
                handle.display_name = hint.display_name;
                Ok(handle)
            }
            ContentBody::ProviderFile { provider, id } => {
                if provider != ProviderId::Fal {
                    return Err(BlazenError::validation(format!(
                        "FalStorageStore: cannot accept ProviderFile for provider {provider:?}; \
                         only ProviderId::Fal is supported"
                    )));
                }
                let kind = hint.kind_hint.unwrap_or(ContentKind::Other);
                let mut handle = ContentHandle::new(id, kind);
                handle.mime_type = hint.mime_type;
                handle.byte_size = hint.byte_size;
                handle.display_name = hint.display_name;
                Ok(handle)
            }
            ContentBody::Stream { stream, size_hint } => {
                use futures_util::StreamExt;
                let mut buf: Vec<u8> =
                    Vec::with_capacity(usize::try_from(size_hint.unwrap_or(0)).unwrap_or(0));
                let mut s = stream;
                while let Some(chunk) = s.next().await {
                    buf.extend_from_slice(&chunk?);
                }
                // TODO(blazen): true streaming PUT to fal storage.
                return self.put(ContentBody::Bytes { data: buf }, hint).await;
            }
        }
    }

    async fn resolve(&self, handle: &ContentHandle) -> Result<MediaSource, BlazenError> {
        // For fal storage, the handle ID IS the public file URL.
        Ok(MediaSource::Url {
            url: handle.id.clone(),
        })
    }

    async fn fetch_bytes(&self, handle: &ContentHandle) -> Result<Vec<u8>, BlazenError> {
        let request = HttpRequest::get(&handle.id);
        let response = self.client.send(request).await?;
        if !response.is_success() {
            return Err(BlazenError::provider(
                "fal",
                format!(
                    "FalStorageStore: fetch_bytes failed for '{}' (status {}): {}",
                    handle.id,
                    response.status,
                    response.text()
                ),
            ));
        }
        Ok(response.body)
    }

    async fn fetch_stream(
        &self,
        handle: &ContentHandle,
    ) -> Result<crate::content::store::ByteStream, BlazenError> {
        use futures_util::StreamExt;

        let request = HttpRequest::get(&handle.id);
        let (status, _headers, http_stream) = self.client.send_streaming(request).await?;
        if !(200..300).contains(&status) {
            return Err(BlazenError::provider(
                "fal",
                format!(
                    "FalStorageStore: fetch_stream failed for '{}' (status {})",
                    handle.id, status,
                ),
            ));
        }
        let mapped = http_stream.map(|res| {
            res.map_err(|e| {
                BlazenError::request(format!("FalStorageStore: stream read error: {e}"))
            })
        });
        Ok(Box::pin(mapped))
    }

    async fn delete(&self, _handle: &ContentHandle) -> Result<(), BlazenError> {
        tracing::warn!(
            "FalStorageStore: delete is a no-op; fal.ai storage does not expose a delete \
             endpoint in public APIs"
        );
        Ok(())
    }
}

/// Read a local file's bytes. Native targets only — WASM has no FS.
#[cfg(not(target_arch = "wasm32"))]
fn read_local_file(path: &std::path::Path) -> Result<Vec<u8>, BlazenError> {
    std::fs::read(path).map_err(|e| {
        BlazenError::request(format!(
            "FalStorageStore: failed to read local file '{}': {e}",
            path.display()
        ))
    })
}

/// WASM targets have no filesystem — `LocalPath` uploads are unsupported.
#[cfg(target_arch = "wasm32")]
fn read_local_file(_path: &std::path::Path) -> Result<Vec<u8>, BlazenError> {
    Err(BlazenError::unsupported(
        "FalStorageStore: ContentBody::LocalPath is not supported on WASM targets",
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::http::{ByteStream, HttpResponse};

    /// Mock HTTP client used by the smoke tests below. Captures requests so
    /// we can assert URL/header/body shape, and returns scripted responses.
    #[derive(Debug)]
    struct MockHttpClient {
        responses: std::sync::Mutex<Vec<HttpResponse>>,
        captured: std::sync::Mutex<Vec<HttpRequest>>,
    }

    impl MockHttpClient {
        fn new(responses: Vec<HttpResponse>) -> Self {
            Self {
                responses: std::sync::Mutex::new(responses),
                captured: std::sync::Mutex::new(Vec::new()),
            }
        }

        fn captured(&self) -> Vec<HttpRequest> {
            self.captured.lock().unwrap().clone()
        }
    }

    #[async_trait]
    impl HttpClient for MockHttpClient {
        async fn send(&self, request: HttpRequest) -> Result<HttpResponse, BlazenError> {
            self.captured.lock().unwrap().push(request);
            let mut responses = self.responses.lock().unwrap();
            if responses.is_empty() {
                return Err(BlazenError::internal(
                    "MockHttpClient: ran out of scripted responses",
                ));
            }
            Ok(responses.remove(0))
        }

        async fn send_streaming(
            &self,
            _request: HttpRequest,
        ) -> Result<(u16, Vec<(String, String)>, ByteStream), BlazenError> {
            Err(BlazenError::unsupported(
                "MockHttpClient: streaming not supported in tests",
            ))
        }
    }

    fn ok_response(body: Vec<u8>) -> HttpResponse {
        HttpResponse {
            status: 200,
            headers: Vec::new(),
            body,
        }
    }

    #[test]
    fn constructor_with_explicit_client_records_fields() {
        let client = Arc::new(MockHttpClient::new(Vec::new()));
        let store = FalStorageStore::new_with_client("test_key", client.clone())
            .with_base_url("https://example.test");
        assert_eq!(store.api_key, "test_key");
        assert_eq!(store.base_url, "https://example.test");
    }

    #[tokio::test]
    async fn put_bytes_runs_two_step_upload_and_records_file_url() {
        let initiate_body = serde_json::to_vec(&serde_json::json!({
            "upload_url": "https://upload.fal.test/presigned",
            "file_url": "https://v3.fal.media/files/cat.png",
        }))
        .unwrap();
        let mock = Arc::new(MockHttpClient::new(vec![
            ok_response(initiate_body),
            ok_response(Vec::new()),
        ]));
        let store = FalStorageStore::new_with_client("k", mock.clone());

        let handle = store
            .put(
                ContentBody::Bytes {
                    data: vec![1, 2, 3, 4],
                },
                ContentHint::default()
                    .with_mime_type("image/png")
                    .with_display_name("cat.png"),
            )
            .await
            .unwrap();

        assert_eq!(handle.id, "https://v3.fal.media/files/cat.png");
        assert_eq!(handle.kind, ContentKind::Image);
        assert_eq!(handle.byte_size, Some(4));

        // Verify two requests went out: initiate (POST to base+/storage/...)
        // and the PUT to the presigned URL.
        let captured = mock.captured();
        assert_eq!(captured.len(), 2);
        assert!(captured[0].url.ends_with("/storage/upload/initiate"));
        assert!(
            captured[0]
                .headers
                .iter()
                .any(|(k, v)| k == "Authorization" && v == "Key k")
        );
        assert_eq!(captured[1].url, "https://upload.fal.test/presigned");
        assert!(
            captured[1]
                .headers
                .iter()
                .any(|(k, v)| k == "Content-Type" && v == "image/png")
        );
        assert_eq!(captured[1].body.as_deref(), Some(&[1, 2, 3, 4][..]));
    }

    #[tokio::test]
    async fn put_url_records_url_without_uploading() {
        let mock = Arc::new(MockHttpClient::new(Vec::new()));
        let store = FalStorageStore::new_with_client("k", mock.clone());
        let handle = store
            .put(
                ContentBody::Url {
                    url: "https://example.com/x.png".into(),
                },
                ContentHint::default().with_mime_type("image/png"),
            )
            .await
            .unwrap();
        assert_eq!(handle.id, "https://example.com/x.png");
        let resolved = store.resolve(&handle).await.unwrap();
        match resolved {
            MediaSource::Url { url } => assert_eq!(url, "https://example.com/x.png"),
            other => panic!("expected url, got {other:?}"),
        }
        // No HTTP traffic.
        assert!(mock.captured().is_empty());
    }

    #[tokio::test]
    async fn put_provider_file_fal_round_trips() {
        let mock = Arc::new(MockHttpClient::new(Vec::new()));
        let store = FalStorageStore::new_with_client("k", mock);
        let handle = store
            .put(
                ContentBody::ProviderFile {
                    provider: ProviderId::Fal,
                    id: "https://v3.fal.media/files/x.png".into(),
                },
                ContentHint::default().with_kind(ContentKind::Image),
            )
            .await
            .unwrap();
        let resolved = store.resolve(&handle).await.unwrap();
        match resolved {
            MediaSource::Url { url } => {
                assert_eq!(url, "https://v3.fal.media/files/x.png");
            }
            other => panic!("expected url, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn put_provider_file_wrong_provider_errors() {
        let mock = Arc::new(MockHttpClient::new(Vec::new()));
        let store = FalStorageStore::new_with_client("k", mock);
        let err = store
            .put(
                ContentBody::ProviderFile {
                    provider: ProviderId::OpenAi,
                    id: "file_abc".into(),
                },
                ContentHint::default(),
            )
            .await
            .unwrap_err();
        assert!(format!("{err}").contains("FalStorageStore"));
    }

    #[tokio::test]
    async fn delete_is_no_op() {
        let mock = Arc::new(MockHttpClient::new(Vec::new()));
        let store = FalStorageStore::new_with_client("k", mock.clone());
        let handle = ContentHandle::new("https://v3.fal.media/files/x.png", ContentKind::Image);
        store.delete(&handle).await.unwrap();
        assert!(mock.captured().is_empty());
    }

    #[tokio::test]
    async fn fetch_bytes_does_a_plain_get() {
        let mock = Arc::new(MockHttpClient::new(vec![ok_response(vec![9, 8, 7])]));
        let store = FalStorageStore::new_with_client("k", mock.clone());
        let handle = ContentHandle::new("https://v3.fal.media/files/x.png", ContentKind::Image);
        let bytes = store.fetch_bytes(&handle).await.unwrap();
        assert_eq!(bytes, vec![9, 8, 7]);
        let captured = mock.captured();
        assert_eq!(captured.len(), 1);
        assert_eq!(captured[0].url, "https://v3.fal.media/files/x.png");
    }

    #[tokio::test]
    async fn stream_put_drains_to_bytes_path() {
        use bytes::Bytes;
        use futures_util::stream;
        let initiate_body = serde_json::to_vec(&serde_json::json!({
            "upload_url": "https://upload.fal.test/presigned",
            "file_url": "https://v3.fal.media/files/streamed.bin",
        }))
        .unwrap();
        let mock = Arc::new(MockHttpClient::new(vec![
            ok_response(initiate_body),
            ok_response(Vec::new()),
        ]));
        let store = FalStorageStore::new_with_client("test_key", mock.clone());
        let chunks: Vec<Result<Bytes, BlazenError>> = vec![
            Ok(Bytes::from_static(b"hello ")),
            Ok(Bytes::from_static(b"world")),
        ];
        let body = ContentBody::Stream {
            stream: Box::pin(stream::iter(chunks)),
            size_hint: Some(11),
        };
        let handle = store
            .put(body, ContentHint::default().with_kind(ContentKind::Other))
            .await
            .expect("put should succeed");
        assert!(!handle.id.is_empty());
        assert_eq!(handle.id, "https://v3.fal.media/files/streamed.bin");

        // The drain-to-bytes path should still issue the two-step upload, and
        // the PUT body should contain the concatenated stream chunks.
        let captured = mock.captured();
        assert_eq!(captured.len(), 2);
        assert_eq!(captured[1].body.as_deref(), Some(&b"hello world"[..]));
    }
}
