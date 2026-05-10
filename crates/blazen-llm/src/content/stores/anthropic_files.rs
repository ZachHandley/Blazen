//! [`AnthropicFilesStore`] — content store backed by the Anthropic Files API
//! (beta).
//!
//! Each `put` uploads bytes (or the contents of a local file) to
//! `POST /v1/files` via `multipart/form-data` and records the returned
//! `file_..` identifier. `resolve` returns
//! [`MediaSource::ProviderFile`] with `provider = ProviderId::Anthropic`,
//! which the Anthropic provider's request builder serializes natively as a
//! `"type": "file", "file_id": "..."` source.
//!
//! The Anthropic Files API requires a beta opt-in header
//! (`anthropic-beta: files-api-2025-04-14` at the time of writing). The
//! beta value is exposed via [`AnthropicFilesStore::with_beta_header`] so
//! the store can be re-targeted as the API graduates.
//!
//! # Multipart body construction
//!
//! Blazen has no shared `multipart/form-data` helper today — the existing
//! providers either send JSON (`text_to_speech_request`, the chat
//! endpoints) or pass a single binary blob with `Content-Type: audio/*`.
//! Rather than introduce a new shared helper for a single caller, this
//! module inlines a tiny single-field multipart builder ([`build_multipart`])
//! that emits the bytes for one `file` form field. If a second
//! provider-file store needs the same shape (likely Gemini Files), promote
//! [`build_multipart`] into `crates/blazen-llm/src/providers/multipart.rs`
//! as `pub(crate)` at that point.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use uuid::Uuid;

use crate::content::detect::detect;
use crate::content::handle::ContentHandle;
use crate::content::kind::ContentKind;
use crate::content::store::{ContentBody, ContentHint, ContentStore};
use crate::error::BlazenError;
use crate::http::{HttpClient, HttpRequest};
use crate::types::{MediaSource, ProviderId};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// The required Anthropic API version header value (matches
/// [`super::super::super::providers::anthropic`]).
const ANTHROPIC_VERSION: &str = "2023-06-01";
/// Default base URL for the Anthropic API.
const DEFAULT_BASE_URL: &str = "https://api.anthropic.com/v1";
/// Default beta opt-in header value for the Files API.
const DEFAULT_BETA_HEADER: &str = "files-api-2025-04-14";
/// Provider tag used in [`BlazenError::ProviderHttp`] errors.
const PROVIDER_TAG: &str = "anthropic_files";

// ---------------------------------------------------------------------------
// Internal record
// ---------------------------------------------------------------------------

/// What we remember about each handle: the Anthropic-issued file id plus
/// the metadata we surfaced on the [`ContentHandle`] at `put` time. We
/// never cache the bytes — `fetch_bytes` round-trips to Anthropic on
/// demand, matching the behaviour of a remote-only store.
#[derive(Debug, Clone)]
struct Stored {
    file_id: String,
}

// ---------------------------------------------------------------------------
// Store
// ---------------------------------------------------------------------------

/// [`ContentStore`] that uploads bytes to the Anthropic Files API and
/// resolves handles back to [`MediaSource::ProviderFile`] references.
pub struct AnthropicFilesStore {
    client: Arc<dyn HttpClient>,
    api_key: String,
    base_url: String,
    beta_header: String,
    index: Mutex<HashMap<String, Stored>>,
}

impl std::fmt::Debug for AnthropicFilesStore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AnthropicFilesStore")
            .field("base_url", &self.base_url)
            .field("beta_header", &self.beta_header)
            .field("api_key", &"<redacted>")
            .finish_non_exhaustive()
    }
}

impl AnthropicFilesStore {
    /// Create a new store using the platform-default HTTP client (reqwest
    /// on native targets, `fetch` on browser WASM).
    #[cfg(any(
        all(target_arch = "wasm32", not(target_os = "wasi")),
        feature = "reqwest",
        target_os = "wasi"
    ))]
    #[must_use]
    pub fn new(api_key: impl Into<String>) -> Self {
        Self::new_with_client(api_key, crate::default_http_client())
    }

    /// Create a new store with a caller-supplied HTTP client backend.
    #[must_use]
    pub fn new_with_client(api_key: impl Into<String>, client: Arc<dyn HttpClient>) -> Self {
        Self {
            client,
            api_key: api_key.into(),
            base_url: DEFAULT_BASE_URL.to_owned(),
            beta_header: DEFAULT_BETA_HEADER.to_owned(),
            index: Mutex::new(HashMap::new()),
        }
    }

    /// Override the Anthropic base URL (e.g. for a local proxy or
    /// integration test server).
    #[must_use]
    pub fn with_base_url(mut self, url: impl Into<String>) -> Self {
        self.base_url = url.into();
        self
    }

    /// Override the `anthropic-beta` header value used for Files API
    /// requests.
    #[must_use]
    pub fn with_beta_header(mut self, beta: impl Into<String>) -> Self {
        self.beta_header = beta.into();
        self
    }

    /// Read the configured base URL (test / introspection helper).
    #[must_use]
    pub fn base_url(&self) -> &str {
        &self.base_url
    }

    /// Read the configured beta header (test / introspection helper).
    #[must_use]
    pub fn beta_header(&self) -> &str {
        &self.beta_header
    }

    fn record(&self, handle_id: String, file_id: String) -> Result<(), BlazenError> {
        self.index
            .lock()
            .map_err(|_| BlazenError::internal("AnthropicFilesStore: index mutex poisoned"))?
            .insert(handle_id, Stored { file_id });
        Ok(())
    }

    fn lookup(&self, handle_id: &str) -> Result<Stored, BlazenError> {
        self.index
            .lock()
            .map_err(|_| BlazenError::internal("AnthropicFilesStore: index mutex poisoned"))?
            .get(handle_id)
            .cloned()
            .ok_or_else(|| {
                BlazenError::internal(format!(
                    "AnthropicFilesStore: handle '{handle_id}' not found"
                ))
            })
    }

    fn next_handle_id() -> String {
        let raw = Uuid::new_v4().simple().to_string();
        format!("blazen_{}", &raw[..16])
    }

    /// Apply the standard Anthropic auth + version + beta headers to a
    /// request. Centralised so every Files call uses the same set.
    fn with_anthropic_headers(&self, request: HttpRequest) -> HttpRequest {
        request
            .header("x-api-key", &self.api_key)
            .header("anthropic-version", ANTHROPIC_VERSION)
            .header("anthropic-beta", &self.beta_header)
    }

    /// Upload an in-memory byte buffer through the multipart endpoint and
    /// build the resulting [`ContentHandle`]. Shared by the
    /// [`ContentBody::Bytes`] and [`ContentBody::Stream`] arms of [`put`]
    /// so streaming uploads buffer-then-multipart through exactly the same
    /// code path as direct byte uploads.
    ///
    /// [`put`]: ContentStore::put
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
        let display = hint
            .display_name
            .clone()
            .unwrap_or_else(|| "upload.bin".to_owned());
        let upload = self.upload(bytes, &display, mime.as_deref()).await?;

        let handle_id = Self::next_handle_id();
        self.record(handle_id.clone(), upload.id.clone())?;

        let mut handle = ContentHandle::new(handle_id, kind);
        handle.mime_type = upload.mime_type.clone().or(mime);
        handle.byte_size = upload.size_bytes.or(hint.byte_size);
        handle.display_name = hint.display_name.or(upload.filename);
        Ok(handle)
    }

    /// Upload `bytes` to `POST /v1/files` and return the issued file id +
    /// (optional) server-reported MIME and byte size.
    async fn upload(
        &self,
        bytes: Vec<u8>,
        filename: &str,
        mime: Option<&str>,
    ) -> Result<UploadResponse, BlazenError> {
        let url = format!("{}/files", self.base_url.trim_end_matches('/'));
        let boundary = generate_boundary();
        let body = build_multipart(
            &boundary,
            "file",
            filename,
            mime.unwrap_or("application/octet-stream"),
            &bytes,
        );

        let content_type = format!("multipart/form-data; boundary={boundary}");
        let request = self
            .with_anthropic_headers(HttpRequest::post(&url))
            .header("Content-Type", content_type)
            .body(body);

        let response = self.client.send(request).await?;

        if !response.is_success() {
            return Err(crate::providers::provider_http_error(
                PROVIDER_TAG,
                &url,
                &response,
            ));
        }

        response.json::<UploadResponse>()
    }
}

// ---------------------------------------------------------------------------
// Wire types
// ---------------------------------------------------------------------------

/// Subset of the Anthropic Files API upload response that we consume.
#[derive(Debug, Clone, serde::Deserialize)]
struct UploadResponse {
    /// Anthropic-issued file id, e.g. `"file_011abcd…"`.
    id: String,
    /// MIME type the server detected (or echoed) for the upload.
    #[serde(default)]
    mime_type: Option<String>,
    /// Size in bytes the server recorded for the upload.
    #[serde(default)]
    size_bytes: Option<u64>,
    /// Original filename if the server echoed it.
    #[serde(default)]
    filename: Option<String>,
}

// ---------------------------------------------------------------------------
// ContentStore impl
// ---------------------------------------------------------------------------

#[async_trait]
impl ContentStore for AnthropicFilesStore {
    async fn put(
        &self,
        body: ContentBody,
        hint: ContentHint,
    ) -> Result<ContentHandle, BlazenError> {
        match body {
            ContentBody::Bytes { data: bytes } => self.put_bytes_inner(bytes, hint).await,
            ContentBody::LocalPath { path } => {
                #[cfg(target_arch = "wasm32")]
                {
                    let _ = path;
                    Err(BlazenError::unsupported(
                        "AnthropicFilesStore: LocalPath bodies are not supported on wasm32 \
                         targets. Read the file in your runtime and pass ContentBody::Bytes \
                         instead.",
                    ))
                }
                #[cfg(not(target_arch = "wasm32"))]
                {
                    let bytes = std::fs::read(&path).map_err(|e| {
                        BlazenError::request(format!(
                            "AnthropicFilesStore: failed to read '{}': {e}",
                            path.display()
                        ))
                    })?;
                    let detected = crate::content::detect::detect_from_path(&path);
                    let kind = hint.kind_hint.unwrap_or(detected.0);
                    let mime = hint.mime_type.clone().or_else(|| detected.1.clone());
                    let display = hint
                        .display_name
                        .clone()
                        .or_else(|| path.file_name().map(|n| n.to_string_lossy().into_owned()));
                    let display_for_upload =
                        display.clone().unwrap_or_else(|| "upload.bin".to_owned());
                    let upload = self
                        .upload(bytes, &display_for_upload, mime.as_deref())
                        .await?;

                    let handle_id = Self::next_handle_id();
                    self.record(handle_id.clone(), upload.id.clone())?;

                    let mut handle = ContentHandle::new(handle_id, kind);
                    handle.mime_type = upload.mime_type.clone().or(mime);
                    handle.byte_size = upload.size_bytes.or(hint.byte_size);
                    handle.display_name = display.or(upload.filename);
                    Ok(handle)
                }
            }
            ContentBody::Url { .. } => Err(BlazenError::unsupported(
                "AnthropicFilesStore: URL bodies are not supported. Fetch the bytes first \
                 and call put(ContentBody::Bytes(..)) instead.",
            )),
            ContentBody::ProviderFile { provider, id } => {
                if !matches!(provider, ProviderId::Anthropic) {
                    return Err(BlazenError::unsupported(format!(
                        "AnthropicFilesStore: ProviderFile body must reference \
                         ProviderId::Anthropic, got {provider:?}. Re-upload via the matching \
                         provider's store, or fetch the bytes and pass them directly."
                    )));
                }
                let handle_id = Self::next_handle_id();
                self.record(handle_id.clone(), id.clone())?;
                let kind = hint.kind_hint.unwrap_or(ContentKind::Other);
                let mut handle = ContentHandle::new(handle_id, kind);
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
                // TODO(blazen): true streaming multipart upload.
                self.put_bytes_inner(buf, hint).await
            }
        }
    }

    async fn resolve(&self, handle: &ContentHandle) -> Result<MediaSource, BlazenError> {
        let stored = self.lookup(&handle.id)?;
        Ok(MediaSource::ProviderFile {
            provider: ProviderId::Anthropic,
            id: stored.file_id,
        })
    }

    async fn fetch_bytes(&self, handle: &ContentHandle) -> Result<Vec<u8>, BlazenError> {
        let stored = self.lookup(&handle.id)?;
        let url = format!(
            "{}/files/{}/content",
            self.base_url.trim_end_matches('/'),
            stored.file_id
        );
        let request = self.with_anthropic_headers(HttpRequest::get(&url));
        let response = self.client.send(request).await?;
        if !response.is_success() {
            return Err(crate::providers::provider_http_error(
                PROVIDER_TAG,
                &url,
                &response,
            ));
        }
        Ok(response.body)
    }

    /// Stream the file body back from the Files API content endpoint.
    ///
    /// Uses [`HttpClient::send_streaming`] so the bytes flow back to the
    /// caller chunk-by-chunk instead of being buffered into a single
    /// `Vec<u8>` first. Errors mid-stream are surfaced as
    /// [`BlazenError::stream_error`] (matching the SSE parser convention).
    async fn fetch_stream(
        &self,
        handle: &ContentHandle,
    ) -> Result<crate::content::store::ByteStream, BlazenError> {
        use futures_util::StreamExt;

        let stored = self.lookup(&handle.id)?;
        let url = format!(
            "{}/files/{}/content",
            self.base_url.trim_end_matches('/'),
            stored.file_id
        );
        let request = self.with_anthropic_headers(HttpRequest::get(&url));
        let (status, headers, byte_stream) = self.client.send_streaming(request).await?;
        if !(200..300).contains(&status) {
            return Err(crate::providers::provider_http_error_parts(
                PROVIDER_TAG,
                &url,
                status,
                &headers,
                "",
            ));
        }
        Ok(Box::pin(byte_stream.map(|chunk| {
            chunk.map_err(|e| BlazenError::stream_error(e.to_string()))
        })))
    }

    async fn delete(&self, handle: &ContentHandle) -> Result<(), BlazenError> {
        // Best-effort: 404 is treated as success because the file may have
        // been TTL'd / removed out-of-band, which from the caller's
        // perspective is the same end state.
        let Ok(stored) = self.lookup(&handle.id) else {
            return Ok(());
        };
        let url = format!(
            "{}/files/{}",
            self.base_url.trim_end_matches('/'),
            stored.file_id
        );
        let request = self.with_anthropic_headers(HttpRequest::delete(&url));
        let response = self.client.send(request).await?;
        if !response.is_success() && response.status != 404 {
            return Err(crate::providers::provider_http_error(
                PROVIDER_TAG,
                &url,
                &response,
            ));
        }
        // Drop the local index entry regardless — the remote no longer
        // references the id (or never did).
        if let Ok(mut guard) = self.index.lock() {
            guard.remove(&handle.id);
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Multipart helpers (inlined; promote to providers::multipart if shared)
// ---------------------------------------------------------------------------

/// Generate a random multipart boundary token. The format is RFC-2046
/// compliant (token chars only) and prefixed so it's easy to spot in
/// captured wire traffic.
fn generate_boundary() -> String {
    let raw = Uuid::new_v4().simple().to_string();
    format!("----blazen-boundary-{raw}")
}

/// Build a single-field `multipart/form-data` body containing one file
/// upload. The body matches the shape the Anthropic Files API expects:
///
/// ```text
/// --<boundary>\r\n
/// Content-Disposition: form-data; name="<field>"; filename="<filename>"\r\n
/// Content-Type: <mime>\r\n
/// \r\n
/// <bytes>\r\n
/// --<boundary>--\r\n
/// ```
///
/// `filename` is included verbatim. Callers should sanitise it if it
/// originates from untrusted input — Anthropic does not interpret it
/// beyond echoing it back, but downstream tools may.
fn build_multipart(
    boundary: &str,
    field_name: &str,
    filename: &str,
    mime: &str,
    bytes: &[u8],
) -> Vec<u8> {
    let mut body = Vec::with_capacity(bytes.len() + 256);
    body.extend_from_slice(b"--");
    body.extend_from_slice(boundary.as_bytes());
    body.extend_from_slice(b"\r\n");
    body.extend_from_slice(b"Content-Disposition: form-data; name=\"");
    body.extend_from_slice(field_name.as_bytes());
    body.extend_from_slice(b"\"; filename=\"");
    body.extend_from_slice(filename.as_bytes());
    body.extend_from_slice(b"\"\r\n");
    body.extend_from_slice(b"Content-Type: ");
    body.extend_from_slice(mime.as_bytes());
    body.extend_from_slice(b"\r\n\r\n");
    body.extend_from_slice(bytes);
    body.extend_from_slice(b"\r\n--");
    body.extend_from_slice(boundary.as_bytes());
    body.extend_from_slice(b"--\r\n");
    body
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::http::{HttpMethod, HttpResponse};
    use std::sync::Mutex as StdMutex;

    /// Mock client that records every request and returns canned responses
    /// in FIFO order.
    #[derive(Debug)]
    struct MockClient {
        sent: Arc<StdMutex<Vec<HttpRequest>>>,
        responses: Arc<StdMutex<Vec<HttpResponse>>>,
    }

    impl MockClient {
        fn new(responses: Vec<HttpResponse>) -> Self {
            Self {
                sent: Arc::new(StdMutex::new(Vec::new())),
                responses: Arc::new(StdMutex::new(responses)),
            }
        }

        fn requests(&self) -> Vec<HttpRequest> {
            self.sent.lock().unwrap().clone()
        }
    }

    #[async_trait]
    impl HttpClient for MockClient {
        async fn send(&self, request: HttpRequest) -> Result<HttpResponse, BlazenError> {
            self.sent.lock().unwrap().push(request);
            let mut q = self.responses.lock().unwrap();
            assert!(!q.is_empty(), "MockClient ran out of canned responses");
            Ok(q.remove(0))
        }

        async fn send_streaming(
            &self,
            _request: HttpRequest,
        ) -> Result<(u16, Vec<(String, String)>, crate::http::ByteStream), BlazenError> {
            unimplemented!("AnthropicFilesStore tests do not stream")
        }
    }

    fn json_response(status: u16, value: &serde_json::Value) -> HttpResponse {
        HttpResponse {
            status,
            headers: vec![("content-type".to_owned(), "application/json".to_owned())],
            body: serde_json::to_vec(value).unwrap(),
        }
    }

    fn empty_response(status: u16) -> HttpResponse {
        HttpResponse {
            status,
            headers: vec![],
            body: Vec::new(),
        }
    }

    fn upload_ok(file_id: &str) -> HttpResponse {
        json_response(
            200,
            &serde_json::json!({
                "id": file_id,
                "type": "file",
                "filename": "upload.bin",
                "mime_type": "application/octet-stream",
                "size_bytes": 9,
                "created_at": "2025-04-14T00:00:00Z",
            }),
        )
    }

    #[test]
    fn defaults_match_anthropic_beta_endpoint() {
        let store =
            AnthropicFilesStore::new_with_client("sk-ant-test", Arc::new(MockClient::new(vec![])));
        assert_eq!(store.base_url(), "https://api.anthropic.com/v1");
        assert_eq!(store.beta_header(), "files-api-2025-04-14");
    }

    #[test]
    fn builders_override_defaults() {
        let store =
            AnthropicFilesStore::new_with_client("sk-ant-test", Arc::new(MockClient::new(vec![])))
                .with_base_url("http://localhost:9999/v1")
                .with_beta_header("files-api-9999-01-01");
        assert_eq!(store.base_url(), "http://localhost:9999/v1");
        assert_eq!(store.beta_header(), "files-api-9999-01-01");
    }

    #[test]
    fn build_multipart_emits_expected_envelope() {
        let body = build_multipart(
            "BOUNDARY",
            "file",
            "name.bin",
            "application/octet-stream",
            b"hello",
        );
        let s = String::from_utf8(body).unwrap();
        assert!(
            s.starts_with("--BOUNDARY\r\n"),
            "missing leading boundary: {s:?}"
        );
        assert!(
            s.contains("Content-Disposition: form-data; name=\"file\"; filename=\"name.bin\"\r\n"),
            "missing content-disposition: {s:?}"
        );
        assert!(
            s.contains("Content-Type: application/octet-stream\r\n\r\nhello\r\n"),
            "missing payload section: {s:?}"
        );
        assert!(
            s.ends_with("\r\n--BOUNDARY--\r\n"),
            "missing closing boundary: {s:?}"
        );
    }

    #[tokio::test]
    async fn put_bytes_uploads_and_records_file_id() {
        let client = Arc::new(MockClient::new(vec![upload_ok("file_011abc")]));
        let store = AnthropicFilesStore::new_with_client("sk-ant-test", client.clone());

        let handle = store
            .put(
                ContentBody::Bytes {
                    data: b"some-bytes".to_vec(),
                },
                ContentHint::default()
                    .with_mime_type("application/octet-stream")
                    .with_display_name("doc.bin"),
            )
            .await
            .expect("put should succeed");

        // Handle metadata reflects server response.
        assert_eq!(handle.byte_size, Some(9));
        assert_eq!(
            handle.mime_type.as_deref(),
            Some("application/octet-stream")
        );
        assert_eq!(handle.display_name.as_deref(), Some("doc.bin"));

        // Resolution returns a ProviderFile with the Anthropic-issued id.
        let resolved = store.resolve(&handle).await.unwrap();
        match resolved {
            MediaSource::ProviderFile { provider, id } => {
                assert_eq!(provider, ProviderId::Anthropic);
                assert_eq!(id, "file_011abc");
            }
            other => panic!("expected ProviderFile, got {other:?}"),
        }

        // Wire request was POST /v1/files with all three Anthropic headers.
        let reqs = client.requests();
        assert_eq!(reqs.len(), 1);
        assert_eq!(reqs[0].method, HttpMethod::Post);
        assert_eq!(reqs[0].url, "https://api.anthropic.com/v1/files");
        let header = |name: &str| -> Option<String> {
            reqs[0]
                .headers
                .iter()
                .find(|(k, _)| k.eq_ignore_ascii_case(name))
                .map(|(_, v)| v.clone())
        };
        assert_eq!(header("x-api-key").as_deref(), Some("sk-ant-test"));
        assert_eq!(
            header("anthropic-version").as_deref(),
            Some(ANTHROPIC_VERSION)
        );
        assert_eq!(
            header("anthropic-beta").as_deref(),
            Some(DEFAULT_BETA_HEADER)
        );
        let ct = header("Content-Type").expect("Content-Type missing");
        assert!(
            ct.starts_with("multipart/form-data; boundary=----blazen-boundary-"),
            "unexpected content-type: {ct}"
        );
        // Body contains the user's bytes inside the multipart envelope.
        let body = reqs[0].body.as_ref().expect("body missing");
        let body_str = String::from_utf8_lossy(body);
        assert!(body_str.contains("filename=\"doc.bin\""));
        assert!(body_str.contains("some-bytes"));
    }

    #[tokio::test]
    async fn provider_file_round_trips_when_provider_matches() {
        let client = Arc::new(MockClient::new(vec![]));
        let store = AnthropicFilesStore::new_with_client("sk-ant-test", client);
        let handle = store
            .put(
                ContentBody::ProviderFile {
                    provider: ProviderId::Anthropic,
                    id: "file_existing".into(),
                },
                ContentHint::default().with_kind(ContentKind::Document),
            )
            .await
            .unwrap();
        let resolved = store.resolve(&handle).await.unwrap();
        match resolved {
            MediaSource::ProviderFile { provider, id } => {
                assert_eq!(provider, ProviderId::Anthropic);
                assert_eq!(id, "file_existing");
            }
            other => panic!("expected ProviderFile, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn provider_file_rejects_other_providers() {
        let client = Arc::new(MockClient::new(vec![]));
        let store = AnthropicFilesStore::new_with_client("sk-ant-test", client);
        let err = store
            .put(
                ContentBody::ProviderFile {
                    provider: ProviderId::OpenAi,
                    id: "file_openai".into(),
                },
                ContentHint::default(),
            )
            .await
            .expect_err("expected error for cross-provider file id");
        assert!(
            matches!(err, BlazenError::Unsupported { .. }),
            "got {err:?}"
        );
    }

    #[tokio::test]
    async fn url_body_is_unsupported() {
        let client = Arc::new(MockClient::new(vec![]));
        let store = AnthropicFilesStore::new_with_client("sk-ant-test", client);
        let err = store
            .put(
                ContentBody::Url {
                    url: "https://example.com/x.png".into(),
                },
                ContentHint::default(),
            )
            .await
            .expect_err("URL bodies should be unsupported");
        assert!(
            matches!(err, BlazenError::Unsupported { .. }),
            "got {err:?}"
        );
    }

    #[tokio::test]
    async fn fetch_bytes_round_trips_through_content_endpoint() {
        let client = Arc::new(MockClient::new(vec![
            upload_ok("file_xyz"),
            HttpResponse {
                status: 200,
                headers: vec![(
                    "content-type".to_owned(),
                    "application/octet-stream".to_owned(),
                )],
                body: b"FETCHED-BYTES".to_vec(),
            },
        ]));
        let store = AnthropicFilesStore::new_with_client("sk-ant-test", client.clone());

        let handle = store
            .put(
                ContentBody::Bytes {
                    data: b"hi".to_vec(),
                },
                ContentHint::default(),
            )
            .await
            .unwrap();
        let bytes = store.fetch_bytes(&handle).await.unwrap();
        assert_eq!(bytes, b"FETCHED-BYTES");

        let reqs = client.requests();
        assert_eq!(reqs.len(), 2);
        assert_eq!(reqs[1].method, HttpMethod::Get);
        assert_eq!(
            reqs[1].url,
            "https://api.anthropic.com/v1/files/file_xyz/content"
        );
    }

    #[tokio::test]
    async fn delete_succeeds_on_200_and_404() {
        let client = Arc::new(MockClient::new(vec![
            upload_ok("file_one"),
            empty_response(200),
            upload_ok("file_two"),
            empty_response(404),
        ]));
        let store = AnthropicFilesStore::new_with_client("sk-ant-test", client.clone());

        let h1 = store
            .put(
                ContentBody::Bytes {
                    data: b"a".to_vec(),
                },
                ContentHint::default(),
            )
            .await
            .unwrap();
        store.delete(&h1).await.expect("200 delete should succeed");

        let h2 = store
            .put(
                ContentBody::Bytes {
                    data: b"b".to_vec(),
                },
                ContentHint::default(),
            )
            .await
            .unwrap();
        store
            .delete(&h2)
            .await
            .expect("404 delete should be best-effort");

        let reqs = client.requests();
        assert_eq!(reqs.len(), 4);
        assert_eq!(reqs[1].method, HttpMethod::Delete);
        assert_eq!(reqs[1].url, "https://api.anthropic.com/v1/files/file_one");
        assert_eq!(reqs[3].method, HttpMethod::Delete);
        assert_eq!(reqs[3].url, "https://api.anthropic.com/v1/files/file_two");
    }

    #[tokio::test]
    async fn upload_failure_surfaces_provider_http_error() {
        let client = Arc::new(MockClient::new(vec![HttpResponse {
            status: 500,
            headers: vec![],
            body: br#"{"error":{"message":"boom"}}"#.to_vec(),
        }]));
        let store = AnthropicFilesStore::new_with_client("sk-ant-test", client);
        let err = store
            .put(
                ContentBody::Bytes {
                    data: b"x".to_vec(),
                },
                ContentHint::default(),
            )
            .await
            .expect_err("expected upload failure");
        assert!(
            matches!(err, BlazenError::ProviderHttp(ref d) if d.status == 500),
            "got {err:?}"
        );
    }

    #[tokio::test]
    async fn stream_put_drains_to_bytes_path() {
        use bytes::Bytes;
        use futures_util::stream;

        let client = Arc::new(MockClient::new(vec![upload_ok("file_streamed")]));
        let store = AnthropicFilesStore::new_with_client("sk-ant-test", client.clone());

        let chunks = vec![
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

        // Streaming put must round-trip through the multipart upload path:
        // exactly one POST /v1/files request whose body contains the
        // concatenated stream bytes.
        let reqs = client.requests();
        assert_eq!(reqs.len(), 1, "expected one upload request");
        assert_eq!(reqs[0].method, HttpMethod::Post);
        assert_eq!(reqs[0].url, "https://api.anthropic.com/v1/files");
        let body_bytes = reqs[0].body.as_ref().expect("multipart body missing");
        let body_str = String::from_utf8_lossy(body_bytes);
        assert!(
            body_str.contains("hello world"),
            "drained stream bytes missing from multipart body: {body_str}"
        );

        // And the issued handle must round-trip back to the Anthropic file id.
        let resolved = store.resolve(&handle).await.unwrap();
        match resolved {
            MediaSource::ProviderFile { provider, id } => {
                assert_eq!(provider, ProviderId::Anthropic);
                assert_eq!(id, "file_streamed");
            }
            other => panic!("expected ProviderFile, got {other:?}"),
        }
    }
}
