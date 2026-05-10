//! [`OpenAiFilesStore`] — content store backed by the `OpenAI` Files API.
//!
//! Bytes pushed into this store are uploaded to `POST /v1/files` (with
//! `purpose=user_data` by default) and the issued `file_…` ID is recorded as
//! the handle's `id`. Resolution always returns
//! [`MediaSource::ProviderFile { provider: ProviderId::OpenAi, id }`], so the
//! request builder serialises the upload as a native `OpenAI` file
//! reference rather than re-encoding the bytes inline.
//!
//! Supported [`ContentBody`] inputs:
//! - [`ContentBody::Bytes`] — uploaded as a multipart form-data part.
//! - [`ContentBody::LocalPath`] — bytes are read from disk, then uploaded.
//! - [`ContentBody::ProviderFile`] with `provider == OpenAi` — recorded
//!   verbatim; no upload is performed (the bytes already live in the
//!   `OpenAI` Files API).
//! - [`ContentBody::ProviderFile`] with any other provider — rejected;
//!   wrong provider for this store.
//! - [`ContentBody::Url`] — rejected; fetch the bytes via
//!   [`super::InMemoryContentStore`] (or a custom store) and re-`put` them.
//!
//! [`MediaSource::ProviderFile`]: crate::types::MediaSource

use std::sync::Arc;

use async_trait::async_trait;
use serde::Deserialize;

use crate::content::detect::detect;
use crate::content::handle::ContentHandle;
use crate::content::kind::ContentKind;
use crate::content::store::{ByteStream, ContentBody, ContentHint, ContentStore};
use crate::error::BlazenError;
use crate::http::{HttpClient, HttpRequest};
use crate::types::{MediaSource, ProviderId};

/// Default base URL for the `OpenAI` API.
const DEFAULT_BASE_URL: &str = "https://api.openai.com/v1";
/// Default `purpose` value sent with uploads.
///
/// `user_data` is the most permissive purpose accepted by the Files API and
/// covers the typical multimodal use case (PDFs, images, audio, etc.). Override
/// via [`OpenAiFilesStore::with_purpose`] for assistants / fine-tuning / batch.
const DEFAULT_PURPOSE: &str = "user_data";

// ---------------------------------------------------------------------------
// Store
// ---------------------------------------------------------------------------

/// Content store backed by the `OpenAI` Files API.
///
/// See the module-level docs for input semantics.
pub struct OpenAiFilesStore {
    client: Arc<dyn HttpClient>,
    api_key: String,
    base_url: String,
    purpose: String,
}

impl std::fmt::Debug for OpenAiFilesStore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OpenAiFilesStore")
            .field("base_url", &self.base_url)
            .field("purpose", &self.purpose)
            .finish_non_exhaustive()
    }
}

impl OpenAiFilesStore {
    /// Create a new store using the platform-default HTTP client.
    #[cfg(any(
        all(target_arch = "wasm32", not(target_os = "wasi")),
        feature = "reqwest",
        target_os = "wasi"
    ))]
    #[must_use]
    pub fn new(api_key: impl Into<String>) -> Self {
        Self::new_with_client(api_key, crate::default_http_client())
    }

    /// Create a new store with an explicit HTTP client backend.
    #[must_use]
    pub fn new_with_client(api_key: impl Into<String>, client: Arc<dyn HttpClient>) -> Self {
        Self {
            client,
            api_key: api_key.into(),
            base_url: DEFAULT_BASE_URL.to_owned(),
            purpose: DEFAULT_PURPOSE.to_owned(),
        }
    }

    /// Use a custom base URL (e.g. a corporate proxy or a mock server).
    /// A trailing slash is tolerated.
    #[must_use]
    pub fn with_base_url(mut self, url: impl Into<String>) -> Self {
        self.base_url = url.into();
        self
    }

    /// Override the `purpose` form field sent with each upload.
    ///
    /// The `OpenAI` Files API accepts `assistants`, `batch`, `fine-tune`,
    /// `vision`, and `user_data` (among others). The default,
    /// [`DEFAULT_PURPOSE`], is the broadest single value and works for the
    /// typical multimodal flow.
    #[must_use]
    pub fn with_purpose(mut self, purpose: impl Into<String>) -> Self {
        self.purpose = purpose.into();
        self
    }

    /// Trim a trailing `/` off the configured base URL once, lazily.
    fn base(&self) -> &str {
        self.base_url.trim_end_matches('/')
    }

    /// Upload `bytes` to `POST {base}/files` and return the parsed response.
    async fn upload_bytes(
        &self,
        bytes: &[u8],
        filename: &str,
        mime: Option<&str>,
    ) -> Result<OpenAiFileResponse, BlazenError> {
        let url = format!("{}/files", self.base());
        let boundary = generate_boundary();
        let body = build_multipart_body(
            &boundary,
            bytes,
            filename,
            mime.unwrap_or("application/octet-stream"),
            &self.purpose,
        );

        let request = HttpRequest::post(&url)
            .bearer_auth(&self.api_key)
            .header(
                "Content-Type",
                format!("multipart/form-data; boundary={boundary}"),
            )
            .body(body);

        let response = self.client.send(request).await?;
        if !response.is_success() {
            return Err(map_files_error(&url, &response));
        }
        response.json::<OpenAiFileResponse>()
    }
}

impl OpenAiFilesStore {
    /// Build a [`ContentHandle`] from a Files API response, layering caller
    /// hints on top of the values the server reported.
    fn build_handle(
        resp: OpenAiFileResponse,
        kind: ContentKind,
        mime: Option<String>,
        fallback_size: Option<u64>,
        fallback_filename: String,
        hint_display: Option<String>,
    ) -> ContentHandle {
        let mut handle = ContentHandle::new(resp.id, kind);
        if let Some(m) = mime {
            handle = handle.with_mime_type(m);
        }
        if let Some(s) = resp.bytes.or(fallback_size) {
            handle = handle.with_byte_size(s);
        }
        if let Some(d) = resp.filename.or(hint_display).or(Some(fallback_filename)) {
            handle = handle.with_display_name(d);
        }
        handle
    }

    /// Upload an in-memory byte buffer, returning the populated handle.
    async fn put_bytes_inner(
        &self,
        bytes: Vec<u8>,
        hint: ContentHint,
    ) -> Result<ContentHandle, BlazenError> {
        let (auto_kind, auto_mime) = if hint.kind_hint.is_some() {
            (ContentKind::Other, None)
        } else {
            detect(
                Some(&bytes),
                hint.mime_type.as_deref(),
                hint.display_name.as_deref(),
            )
        };
        let kind = hint.kind_hint.unwrap_or(auto_kind);
        let mime = hint.mime_type.clone().or(auto_mime);
        let filename = hint.display_name.clone().unwrap_or_else(|| "upload".into());
        let resp = self
            .upload_bytes(&bytes, &filename, mime.as_deref())
            .await?;
        let fallback_size = u64::try_from(bytes.len()).ok();
        Ok(Self::build_handle(
            resp,
            kind,
            mime,
            fallback_size,
            filename,
            hint.display_name,
        ))
    }

    /// Materialise a `ContentBody::ProviderFile { OpenAi, id }` into a
    /// handle without performing any HTTP traffic.
    fn put_provider_file(file_id: String, hint: ContentHint) -> ContentHandle {
        let kind = hint.kind_hint.unwrap_or(ContentKind::Other);
        let mut handle = ContentHandle::new(file_id, kind);
        if let Some(m) = hint.mime_type {
            handle = handle.with_mime_type(m);
        }
        if let Some(s) = hint.byte_size {
            handle = handle.with_byte_size(s);
        }
        if let Some(d) = hint.display_name {
            handle = handle.with_display_name(d);
        }
        handle
    }
}

#[async_trait]
impl ContentStore for OpenAiFilesStore {
    async fn put(
        &self,
        body: ContentBody,
        hint: ContentHint,
    ) -> Result<ContentHandle, BlazenError> {
        match body {
            ContentBody::Bytes { data: bytes } => self.put_bytes_inner(bytes, hint).await,

            #[cfg(not(target_arch = "wasm32"))]
            ContentBody::LocalPath { path } => {
                let bytes = tokio::fs::read(&path).await.map_err(|e| {
                    BlazenError::request(format!(
                        "OpenAiFilesStore: failed to read '{}': {e}",
                        path.display()
                    ))
                })?;
                let mut hint = hint;
                if hint.display_name.is_none() {
                    hint.display_name =
                        path.file_name().and_then(|n| n.to_str()).map(str::to_owned);
                }
                self.put_bytes_inner(bytes, hint).await
            }

            #[cfg(target_arch = "wasm32")]
            ContentBody::LocalPath { .. } => Err(BlazenError::unsupported(
                "OpenAiFilesStore: LocalPath ContentBody not supported on wasm32; \
                 read the file manually and pass ContentBody::Bytes",
            )),

            ContentBody::Url { .. } => Err(BlazenError::unsupported(
                "OpenAiFilesStore: URL ContentBody not supported; \
                 fetch first or use InMemoryContentStore",
            )),

            ContentBody::ProviderFile {
                provider: ProviderId::OpenAi,
                id: file_id,
            } => Ok(Self::put_provider_file(file_id, hint)),

            ContentBody::ProviderFile { provider, .. } => Err(BlazenError::unsupported(format!(
                "OpenAiFilesStore: ProviderFile from '{provider:?}' is not supported by this store \
                 (only ProviderId::OpenAi is accepted)"
            ))),

            ContentBody::Stream { stream, size_hint } => {
                use futures_util::StreamExt;
                let mut buf: Vec<u8> =
                    Vec::with_capacity(usize::try_from(size_hint.unwrap_or(0)).unwrap_or(0));
                let mut s = stream;
                while let Some(chunk) = s.next().await {
                    buf.extend_from_slice(&chunk?);
                }
                // TODO(blazen): true streaming multipart upload. For now we
                // drain the stream into a single buffer and route it through
                // the existing bytes-upload path, since the OpenAI Files API
                // multipart helper takes a `&[u8]` body. Provider-file
                // uploads are usually small enough that this is acceptable.
                self.put_bytes_inner(buf, hint).await
            }
        }
    }

    async fn resolve(&self, handle: &ContentHandle) -> Result<MediaSource, BlazenError> {
        Ok(MediaSource::ProviderFile {
            provider: ProviderId::OpenAi,
            id: handle.id.clone(),
        })
    }

    async fn fetch_bytes(&self, handle: &ContentHandle) -> Result<Vec<u8>, BlazenError> {
        let url = format!("{}/files/{}/content", self.base(), handle.id);
        let request = HttpRequest::get(&url).bearer_auth(&self.api_key);
        let response = self.client.send(request).await?;
        if !response.is_success() {
            return Err(map_files_error(&url, &response));
        }
        Ok(response.body)
    }

    async fn fetch_stream(&self, handle: &ContentHandle) -> Result<ByteStream, BlazenError> {
        use futures_util::StreamExt;

        let url = format!("{}/files/{}/content", self.base(), handle.id);
        let request = HttpRequest::get(&url).bearer_auth(&self.api_key);
        let (status, headers, byte_stream) = self.client.send_streaming(request).await?;
        if !(200..300).contains(&status) {
            // We don't drain the stream body to extract a richer error here;
            // status + headers is enough for the auth / rate-limit / generic
            // mapping. This mirrors how the streaming providers handle
            // pre-stream errors.
            return Err(match status {
                401 => BlazenError::auth("OpenAiFilesStore: authentication failed"),
                429 => BlazenError::RateLimit {
                    retry_after_ms: parse_retry_after(&headers),
                },
                _ => crate::providers::provider_http_error_parts(
                    "openai", &url, status, &headers, "",
                ),
            });
        }
        // Adapt the HttpClient's `Box<dyn Error>`-typed byte stream into a
        // `BlazenError`-typed `ContentStore` byte stream.
        let mapped =
            byte_stream.map(|chunk| chunk.map_err(|e| BlazenError::request(e.to_string())));
        Ok(Box::pin(mapped))
    }

    async fn delete(&self, handle: &ContentHandle) -> Result<(), BlazenError> {
        let url = format!("{}/files/{}", self.base(), handle.id);
        let request = HttpRequest::delete(&url).bearer_auth(&self.api_key);
        let response = self.client.send(request).await?;
        // 200 = deleted; 404 = already gone (idempotent).
        if response.is_success() || response.status == 404 {
            Ok(())
        } else {
            Err(map_files_error(&url, &response))
        }
    }
}

// ---------------------------------------------------------------------------
// Wire types
// ---------------------------------------------------------------------------

/// Subset of the `OpenAI` Files API response we care about.
///
/// The full payload also has `object`, `created_at`, `purpose`, `status`,
/// and `status_details`, but none of those are needed to populate a
/// [`ContentHandle`].
#[derive(Debug, Deserialize)]
struct OpenAiFileResponse {
    id: String,
    #[serde(default)]
    bytes: Option<u64>,
    #[serde(default)]
    filename: Option<String>,
}

// ---------------------------------------------------------------------------
// Multipart helpers
// ---------------------------------------------------------------------------

/// Generate a random multipart boundary string.
///
/// Using a UUID gives 122 bits of entropy — well above the multipart
/// requirement that the boundary not appear inside the encoded body, even
/// for adversarially-crafted file contents.
fn generate_boundary() -> String {
    format!("----BlazenBoundary{}", uuid::Uuid::new_v4().simple())
}

/// Build an RFC 7578 `multipart/form-data` body with two parts:
/// `purpose=<purpose>` and `file=<bytes>` (the latter with `filename`
/// + `Content-Type`).
fn build_multipart_body(
    boundary: &str,
    file_bytes: &[u8],
    filename: &str,
    mime: &str,
    purpose: &str,
) -> Vec<u8> {
    let mut body = Vec::with_capacity(file_bytes.len() + 512);

    // purpose part
    body.extend_from_slice(b"--");
    body.extend_from_slice(boundary.as_bytes());
    body.extend_from_slice(b"\r\n");
    body.extend_from_slice(b"Content-Disposition: form-data; name=\"purpose\"\r\n\r\n");
    body.extend_from_slice(purpose.as_bytes());
    body.extend_from_slice(b"\r\n");

    // file part
    body.extend_from_slice(b"--");
    body.extend_from_slice(boundary.as_bytes());
    body.extend_from_slice(b"\r\n");
    body.extend_from_slice(b"Content-Disposition: form-data; name=\"file\"; filename=\"");
    body.extend_from_slice(escape_quotes(filename).as_bytes());
    body.extend_from_slice(b"\"\r\n");
    body.extend_from_slice(b"Content-Type: ");
    body.extend_from_slice(mime.as_bytes());
    body.extend_from_slice(b"\r\n\r\n");
    body.extend_from_slice(file_bytes);
    body.extend_from_slice(b"\r\n");

    // closing boundary
    body.extend_from_slice(b"--");
    body.extend_from_slice(boundary.as_bytes());
    body.extend_from_slice(b"--\r\n");

    body
}

/// Minimal escape for filenames embedded in the `Content-Disposition`
/// header. `OpenAI` does not document a canonical escape; we follow the
/// common practice of escaping `"` and `\` with a leading backslash.
fn escape_quotes(s: &str) -> String {
    s.replace('\\', "\\\\").replace('"', "\\\"")
}

// ---------------------------------------------------------------------------
// Error mapping
// ---------------------------------------------------------------------------

fn map_files_error(url: &str, response: &crate::http::HttpResponse) -> BlazenError {
    match response.status {
        401 => BlazenError::auth("OpenAiFilesStore: authentication failed"),
        429 => BlazenError::RateLimit {
            retry_after_ms: parse_retry_after(&response.headers),
        },
        _ => crate::providers::provider_http_error("openai", url, response),
    }
}

/// Local copy of the `Retry-After` header parser so we don't need to make
/// `openai_format::parse_retry_after` `pub(crate)` for this module.
fn parse_retry_after(headers: &[(String, String)]) -> Option<u64> {
    headers
        .iter()
        .find(|(k, _)| k.eq_ignore_ascii_case("retry-after"))
        .and_then(|(_, v)| v.parse::<f64>().ok())
        .and_then(|secs| {
            if secs.is_finite() && secs >= 0.0 {
                #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
                Some((secs * 1000.0) as u64)
            } else {
                None
            }
        })
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::http::{ByteStream, HttpMethod, HttpResponse};
    use std::sync::Mutex;

    /// Mock client that records every request and returns canned responses
    /// from a queue (one response per `send` call).
    #[derive(Debug)]
    struct MockClient {
        sent: Arc<Mutex<Vec<HttpRequest>>>,
        responses: Arc<Mutex<Vec<HttpResponse>>>,
    }

    impl MockClient {
        fn new(responses: Vec<HttpResponse>) -> Self {
            Self {
                sent: Arc::new(Mutex::new(Vec::new())),
                responses: Arc::new(Mutex::new(responses)),
            }
        }

        fn last(&self) -> HttpRequest {
            self.sent
                .lock()
                .unwrap()
                .last()
                .cloned()
                .expect("no request captured")
        }
    }

    #[async_trait]
    impl HttpClient for MockClient {
        async fn send(&self, request: HttpRequest) -> Result<HttpResponse, BlazenError> {
            self.sent.lock().unwrap().push(request);
            let resp = self
                .responses
                .lock()
                .unwrap()
                .drain(..1)
                .next()
                .expect("no canned response left");
            Ok(resp)
        }

        async fn send_streaming(
            &self,
            _request: HttpRequest,
        ) -> Result<(u16, Vec<(String, String)>, ByteStream), BlazenError> {
            unimplemented!("not needed for tests")
        }
    }

    fn ok_files_response(id: &str, size: u64, filename: &str) -> HttpResponse {
        let body = serde_json::json!({
            "id": id,
            "object": "file",
            "bytes": size,
            "created_at": 1_700_000_000_u64,
            "filename": filename,
            "purpose": "user_data",
        })
        .to_string()
        .into_bytes();
        HttpResponse {
            status: 200,
            headers: vec![("content-type".to_owned(), "application/json".to_owned())],
            body,
        }
    }

    fn ok_bytes(bytes: Vec<u8>) -> HttpResponse {
        HttpResponse {
            status: 200,
            headers: vec![],
            body: bytes,
        }
    }

    fn ok_empty() -> HttpResponse {
        HttpResponse {
            status: 200,
            headers: vec![],
            body: b"{\"id\":\"file_x\",\"object\":\"file\",\"deleted\":true}".to_vec(),
        }
    }

    #[tokio::test]
    async fn put_bytes_uploads_multipart_and_records_file_id() {
        let client = Arc::new(MockClient::new(vec![ok_files_response(
            "file_abc123",
            7,
            "hello.txt",
        )]));
        let store =
            OpenAiFilesStore::new_with_client("sk-test", client.clone() as Arc<dyn HttpClient>);

        let handle = store
            .put(
                ContentBody::Bytes {
                    data: b"payload".to_vec(),
                },
                ContentHint::default()
                    .with_kind(ContentKind::Document)
                    .with_display_name("hello.txt")
                    .with_mime_type("text/plain"),
            )
            .await
            .unwrap();

        assert_eq!(handle.id, "file_abc123");
        assert_eq!(handle.kind, ContentKind::Document);
        assert_eq!(handle.byte_size, Some(7));
        assert_eq!(handle.display_name.as_deref(), Some("hello.txt"));

        let req = client.last();
        assert_eq!(req.method, HttpMethod::Post);
        assert_eq!(req.url, "https://api.openai.com/v1/files");
        assert!(
            req.headers
                .iter()
                .any(|(k, v)| k == "Authorization" && v == "Bearer sk-test"),
            "missing bearer header"
        );
        let ct = req
            .headers
            .iter()
            .find(|(k, _)| k.eq_ignore_ascii_case("content-type"))
            .map(|(_, v)| v.as_str())
            .unwrap();
        assert!(
            ct.starts_with("multipart/form-data; boundary="),
            "unexpected content-type: {ct}"
        );
        let body = req.body.as_ref().expect("missing body");
        let body_str = String::from_utf8_lossy(body);
        assert!(body_str.contains("name=\"purpose\""), "body: {body_str}");
        assert!(body_str.contains("user_data"), "body: {body_str}");
        assert!(
            body_str.contains("name=\"file\"; filename=\"hello.txt\""),
            "body: {body_str}"
        );
        assert!(body_str.contains("Content-Type: text/plain"));
        assert!(body_str.contains("payload"));
    }

    #[tokio::test]
    async fn resolve_returns_provider_file_for_openai() {
        let client = Arc::new(MockClient::new(vec![]));
        let store = OpenAiFilesStore::new_with_client("sk-test", client as Arc<dyn HttpClient>);

        let handle = ContentHandle::new("file_xyz", ContentKind::Image);
        let resolved = store.resolve(&handle).await.unwrap();
        match resolved {
            MediaSource::ProviderFile { provider, id } => {
                assert_eq!(provider, ProviderId::OpenAi);
                assert_eq!(id, "file_xyz");
            }
            other => panic!("expected provider file, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn put_provider_file_openai_skips_upload() {
        let client = Arc::new(MockClient::new(vec![]));
        let store =
            OpenAiFilesStore::new_with_client("sk-test", client.clone() as Arc<dyn HttpClient>);

        let handle = store
            .put(
                ContentBody::ProviderFile {
                    provider: ProviderId::OpenAi,
                    id: "file_existing".into(),
                },
                ContentHint::default().with_kind(ContentKind::Document),
            )
            .await
            .unwrap();

        assert_eq!(handle.id, "file_existing");
        assert_eq!(handle.kind, ContentKind::Document);
        assert!(
            client.sent.lock().unwrap().is_empty(),
            "no HTTP traffic should be issued for ProviderFile inputs"
        );
    }

    #[tokio::test]
    async fn put_provider_file_wrong_provider_errors() {
        let client = Arc::new(MockClient::new(vec![]));
        let store = OpenAiFilesStore::new_with_client("sk-test", client as Arc<dyn HttpClient>);

        let err = store
            .put(
                ContentBody::ProviderFile {
                    provider: ProviderId::Anthropic,
                    id: "anth_xxx".into(),
                },
                ContentHint::default(),
            )
            .await
            .expect_err("should reject non-OpenAI provider files");
        assert!(
            matches!(err, BlazenError::Unsupported { .. }),
            "got {err:?}"
        );
    }

    #[tokio::test]
    async fn put_url_is_unsupported() {
        let client = Arc::new(MockClient::new(vec![]));
        let store = OpenAiFilesStore::new_with_client("sk-test", client as Arc<dyn HttpClient>);

        let err = store
            .put(
                ContentBody::Url {
                    url: "https://example.com/x.png".into(),
                },
                ContentHint::default(),
            )
            .await
            .expect_err("URL inputs should be unsupported");
        assert!(
            matches!(err, BlazenError::Unsupported { .. }),
            "got {err:?}"
        );
    }

    #[tokio::test]
    async fn fetch_bytes_hits_content_endpoint() {
        let client = Arc::new(MockClient::new(vec![ok_bytes(b"hello".to_vec())]));
        let store =
            OpenAiFilesStore::new_with_client("sk-test", client.clone() as Arc<dyn HttpClient>);

        let handle = ContentHandle::new("file_dl", ContentKind::Document);
        let bytes = store.fetch_bytes(&handle).await.unwrap();
        assert_eq!(bytes, b"hello");

        let req = client.last();
        assert_eq!(req.method, HttpMethod::Get);
        assert_eq!(req.url, "https://api.openai.com/v1/files/file_dl/content");
    }

    #[tokio::test]
    async fn delete_treats_404_as_ok() {
        let client = Arc::new(MockClient::new(vec![HttpResponse {
            status: 404,
            headers: vec![],
            body: b"{\"error\":\"not found\"}".to_vec(),
        }]));
        let store =
            OpenAiFilesStore::new_with_client("sk-test", client.clone() as Arc<dyn HttpClient>);

        let handle = ContentHandle::new("file_gone", ContentKind::Document);
        store.delete(&handle).await.unwrap();

        let req = client.last();
        assert_eq!(req.method, HttpMethod::Delete);
        assert_eq!(req.url, "https://api.openai.com/v1/files/file_gone");
    }

    #[tokio::test]
    async fn delete_success_returns_ok() {
        let client = Arc::new(MockClient::new(vec![ok_empty()]));
        let store = OpenAiFilesStore::new_with_client("sk-test", client as Arc<dyn HttpClient>);
        store
            .delete(&ContentHandle::new("file_x", ContentKind::Document))
            .await
            .unwrap();
    }

    #[tokio::test]
    async fn upload_401_maps_to_auth_error() {
        let client = Arc::new(MockClient::new(vec![HttpResponse {
            status: 401,
            headers: vec![],
            body: b"bad key".to_vec(),
        }]));
        let store = OpenAiFilesStore::new_with_client("sk-bad", client as Arc<dyn HttpClient>);

        let err = store
            .put(
                ContentBody::Bytes {
                    data: vec![1, 2, 3],
                },
                ContentHint::default(),
            )
            .await
            .expect_err("expected auth error");
        assert!(matches!(err, BlazenError::Auth { .. }), "got {err:?}");
    }

    #[tokio::test]
    async fn with_base_url_and_with_purpose_round_trip() {
        let client = Arc::new(MockClient::new(vec![ok_files_response("file_p", 3, "x")]));
        let store = OpenAiFilesStore::new_with_client("sk", client.clone() as Arc<dyn HttpClient>)
            .with_base_url("https://proxy.example.com/v1/")
            .with_purpose("assistants");

        store
            .put(
                ContentBody::Bytes {
                    data: vec![1, 2, 3],
                },
                ContentHint::default(),
            )
            .await
            .unwrap();

        let req = client.last();
        assert_eq!(req.url, "https://proxy.example.com/v1/files");
        let body_str = String::from_utf8_lossy(req.body.as_ref().unwrap()).into_owned();
        assert!(
            body_str.contains("assistants"),
            "expected purpose=assistants in body: {body_str}"
        );
    }

    #[test]
    fn build_multipart_body_has_well_formed_parts() {
        let body = build_multipart_body(
            "XYZ",
            b"abc",
            "f.bin",
            "application/octet-stream",
            "user_data",
        );
        let s = String::from_utf8(body).unwrap();
        assert!(s.starts_with("--XYZ\r\n"));
        assert!(s.ends_with("--XYZ--\r\n"));
        assert!(s.contains("name=\"purpose\""));
        assert!(s.contains("name=\"file\"; filename=\"f.bin\""));
        assert!(s.contains("Content-Type: application/octet-stream"));
        assert!(s.contains("\r\n\r\nabc\r\n"));
    }

    #[test]
    fn escape_quotes_handles_quotes_and_backslashes() {
        assert_eq!(escape_quotes("a\"b"), "a\\\"b");
        assert_eq!(escape_quotes("a\\b"), "a\\\\b");
        assert_eq!(escape_quotes("plain.txt"), "plain.txt");
    }

    #[tokio::test]
    async fn stream_put_drains_to_bytes_path() {
        use bytes::Bytes;
        use futures_util::stream;

        let client = Arc::new(MockClient::new(vec![ok_files_response(
            "file_streamed",
            11,
            "stream.bin",
        )]));
        let store =
            OpenAiFilesStore::new_with_client("sk-test", client.clone() as Arc<dyn HttpClient>);

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

        assert_eq!(handle.id, "file_streamed");
        assert_eq!(handle.kind, ContentKind::Other);

        // Confirm the drained stream was forwarded into the multipart body
        // exactly once, via the same upload path used by `ContentBody::Bytes`.
        let req = client.last();
        assert_eq!(req.method, HttpMethod::Post);
        assert_eq!(req.url, "https://api.openai.com/v1/files");
        let body_bytes = req.body.as_ref().expect("missing body");
        let body_str = String::from_utf8_lossy(body_bytes);
        assert!(
            body_str.contains("hello world"),
            "drained stream payload missing from multipart body: {body_str}"
        );
        assert!(body_str.contains("name=\"file\""));
    }
}
