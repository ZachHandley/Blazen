//! [`GeminiFilesStore`] — content store backed by the Google Gemini Files API.
//!
//! Bytes pushed into this store are uploaded to the Gemini Files API and the
//! returned file URI (e.g. `https://generativelanguage.googleapis.com/v1beta/files/abc-123`)
//! is recorded as the handle's `id`. Resolution always returns
//! [`MediaSource::ProviderFile { provider: ProviderId::Gemini, id: "<uri>" }`],
//! which the Gemini provider's request builder serializes natively as a
//! `fileData.fileUri` source.
//!
//! # Upload flow
//!
//! Gemini's Files API supports two upload paths:
//!
//! 1. A **resumable upload** initiated by `POST /upload/v1beta/files` with a
//!    JSON metadata body — returns an `X-Goog-Upload-URL` header that the
//!    caller then PUTs the bytes to.
//! 2. A **simple multipart upload** to `POST /upload/v1beta/files?uploadType=multipart`
//!    with a single multipart body containing a JSON metadata part and a
//!    binary file part.
//!
//! This store uses the simple multipart variant for code-path simplicity. See
//! the documentation on [`GeminiFilesStore::put`] for the inflight 20 MB
//! multipart upload ceiling Google enforces; bytes larger than that should
//! be uploaded via the resumable flow (not implemented here — file an issue
//! or supply a custom store if you need it).
//!
//! Auth: every request carries `x-goog-api-key: <api_key>` (matching
//! [`super::super::super::providers::gemini`]).
//!
//! Supported [`ContentBody`] inputs:
//! - [`ContentBody::Bytes`] — uploaded as multipart.
//! - [`ContentBody::LocalPath`] — bytes are read from disk, then uploaded
//!   (native targets only).
//! - [`ContentBody::ProviderFile`] with `provider == Gemini` — recorded
//!   verbatim; no upload is performed (the URI already references a file
//!   in Gemini's Files API).
//! - [`ContentBody::ProviderFile`] with any other provider — rejected;
//!   wrong provider for this store.
//! - [`ContentBody::Url`] — rejected; fetch the bytes via
//!   [`super::InMemoryContentStore`] (or a custom store) and re-`put` them.
//!
//! [`MediaSource::ProviderFile`]: crate::types::MediaSource

use std::sync::Arc;

use async_trait::async_trait;
use serde::Deserialize;
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

/// Default base URL for the Gemini API host. Both the upload endpoint
/// (`/upload/v1beta/files`) and the management endpoint (`/v1beta/files/...`)
/// hang off the same host, so the store stores the host root and joins the
/// path-prefix at call time.
const DEFAULT_BASE_URL: &str = "https://generativelanguage.googleapis.com";
/// Provider tag used for [`BlazenError::ProviderHttp`] errors.
const PROVIDER_TAG: &str = "gemini_files";

// ---------------------------------------------------------------------------
// Store
// ---------------------------------------------------------------------------

/// [`ContentStore`] that uploads bytes to the Google Gemini Files API and
/// resolves handles back to [`MediaSource::ProviderFile`] references.
///
/// See the module-level docs for input semantics.
pub struct GeminiFilesStore {
    client: Arc<dyn HttpClient>,
    api_key: String,
    base_url: String,
}

impl std::fmt::Debug for GeminiFilesStore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GeminiFilesStore")
            .field("base_url", &self.base_url)
            .field("api_key", &"<redacted>")
            .finish_non_exhaustive()
    }
}

impl GeminiFilesStore {
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
        }
    }

    /// Override the Gemini API host (e.g. for a local proxy, a regional
    /// endpoint, or a mock server). A trailing slash is tolerated.
    #[must_use]
    pub fn with_base_url(mut self, url: impl Into<String>) -> Self {
        self.base_url = url.into();
        self
    }

    /// Read the configured base URL (test / introspection helper).
    #[must_use]
    pub fn base_url(&self) -> &str {
        &self.base_url
    }

    /// Trim a trailing `/` off the configured base URL once.
    fn base(&self) -> &str {
        self.base_url.trim_end_matches('/')
    }

    /// Upload `bytes` via the simple multipart endpoint and parse the
    /// response.
    ///
    /// # Errors
    ///
    /// Returns [`BlazenError::ProviderHttp`] if Gemini responds with a
    /// non-2xx status. The 20 MB simple-upload ceiling typically surfaces
    /// here as a 400 with `INVALID_ARGUMENT`.
    async fn upload(
        &self,
        bytes: &[u8],
        display_name: &str,
        mime: &str,
    ) -> Result<GeminiFile, BlazenError> {
        let url = format!("{}/upload/v1beta/files?uploadType=multipart", self.base());
        let boundary = generate_boundary();
        let body = build_multipart(&boundary, display_name, mime, bytes);

        let request = HttpRequest::post(&url)
            .header("x-goog-api-key", &self.api_key)
            .header(
                "Content-Type",
                format!("multipart/related; boundary={boundary}"),
            )
            .body(body);

        let response = self.client.send(request).await?;

        if !response.is_success() {
            return Err(map_files_error(&url, &response));
        }

        let parsed: UploadEnvelope = response.json()?;
        Ok(parsed.file)
    }

    /// Materialise a `ContentBody::ProviderFile { Gemini, id }` into a
    /// handle without performing any HTTP traffic. The `id` is treated as
    /// the canonical file URI (what Gemini's `fileData.fileUri` expects).
    fn put_provider_file(uri: String, hint: ContentHint) -> ContentHandle {
        let kind = hint.kind_hint.unwrap_or(ContentKind::Other);
        let mut handle = ContentHandle::new(uri, kind);
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

    /// Upload an in-memory byte buffer and return the populated handle.
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
        let mime_owned = hint
            .mime_type
            .clone()
            .or(auto_mime)
            .unwrap_or_else(|| "application/octet-stream".to_owned());
        let display_for_upload = hint
            .display_name
            .clone()
            .unwrap_or_else(|| "upload.bin".to_owned());

        let file = self
            .upload(&bytes, &display_for_upload, &mime_owned)
            .await?;

        // Use the URI as the canonical handle id — that's what the Gemini
        // request builder expects to thread through to `fileData.fileUri`.
        let id = file.uri.clone().ok_or_else(|| {
            BlazenError::request(
                "GeminiFilesStore: Files API response missing required `file.uri` field",
            )
        })?;

        #[allow(clippy::cast_possible_truncation)]
        let local_size = bytes.len() as u64;
        let mut handle = ContentHandle::new(id, kind);
        handle.mime_type = file.mime_type.clone().or(Some(mime_owned));
        handle.byte_size = file
            .size_bytes_parsed()
            .or(hint.byte_size)
            .or(Some(local_size));
        handle.display_name = hint
            .display_name
            .clone()
            .or(file.display_name)
            .or(Some(display_for_upload));
        Ok(handle)
    }
}

#[async_trait]
impl ContentStore for GeminiFilesStore {
    /// Upload bytes to the Gemini Files API.
    ///
    /// # Size limit
    ///
    /// This store uses Gemini's simple multipart upload, which Google
    /// caps at roughly **20 MB per request**. Larger bytes will return a
    /// `400 INVALID_ARGUMENT` from Gemini. For larger files use the
    /// resumable upload flow documented at
    /// <https://ai.google.dev/gemini-api/docs/files> — not yet implemented
    /// in this store; supply a custom [`ContentStore`] or open an issue if
    /// you need it.
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
                        "GeminiFilesStore: failed to read '{}': {e}",
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
                "GeminiFilesStore: LocalPath ContentBody not supported on wasm32; \
                 read the file manually and pass ContentBody::Bytes",
            )),

            ContentBody::Url { .. } => Err(BlazenError::unsupported(
                "GeminiFilesStore: URL ContentBody not supported; \
                 fetch the bytes first and re-call put(ContentBody::Bytes(..))",
            )),

            ContentBody::ProviderFile {
                provider: ProviderId::Gemini,
                id: uri,
            } => Ok(Self::put_provider_file(uri, hint)),

            ContentBody::ProviderFile { provider, .. } => Err(BlazenError::unsupported(format!(
                "GeminiFilesStore: ProviderFile from '{provider:?}' is not supported by this \
                 store (only ProviderId::Gemini is accepted). Re-upload via the matching \
                 provider's store, or fetch the bytes and pass them directly."
            ))),

            ContentBody::Stream { stream, size_hint } => {
                use futures_util::StreamExt;
                let mut buf: Vec<u8> =
                    Vec::with_capacity(usize::try_from(size_hint.unwrap_or(0)).unwrap_or(0));
                let mut s = stream;
                while let Some(chunk) = s.next().await {
                    buf.extend_from_slice(&chunk?);
                }
                // TODO(blazen): true streaming multipart upload.
                self.put(ContentBody::Bytes { data: buf }, hint).await
            }
        }
    }

    async fn resolve(&self, handle: &ContentHandle) -> Result<MediaSource, BlazenError> {
        Ok(MediaSource::ProviderFile {
            provider: ProviderId::Gemini,
            id: handle.id.clone(),
        })
    }

    /// Always returns [`BlazenError::Unsupported`].
    ///
    /// The Gemini Files API exposes file *metadata* via `GET /v1beta/{name}`
    /// but does not expose a content-download endpoint — files are
    /// referenced by URI for inference and never streamed back to the
    /// caller. Tools that need raw bytes should keep a copy in a
    /// secondary store (e.g. [`super::InMemoryContentStore`]) at the time
    /// of upload.
    async fn fetch_bytes(&self, _handle: &ContentHandle) -> Result<Vec<u8>, BlazenError> {
        Err(BlazenError::unsupported(
            "GeminiFilesStore::fetch_bytes is not supported — the Gemini Files API does not \
             expose a content-download endpoint. Files are referenced by URI for inference \
             only. Keep a parallel copy in InMemoryContentStore (or similar) if your tools \
             need the raw bytes back.",
        ))
    }

    /// Best-effort delete: parses the resource name (`files/<id>`) out of
    /// the stored URI and issues `DELETE /v1beta/{name}`. 404 is treated
    /// as success (the file may have been TTL'd out).
    async fn delete(&self, handle: &ContentHandle) -> Result<(), BlazenError> {
        let Some(name) = parse_file_name(&handle.id) else {
            // Nothing we can do — the handle id is not a Gemini URI. Treat
            // as a no-op rather than erroring; callers may have stuffed an
            // arbitrary id in here via a custom workflow.
            return Ok(());
        };
        let url = format!("{}/v1beta/{name}", self.base());
        let request = HttpRequest::delete(&url).header("x-goog-api-key", &self.api_key);
        let response = self.client.send(request).await?;
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

/// Top-level upload response: `{"file": {...}}`.
#[derive(Debug, Deserialize)]
struct UploadEnvelope {
    file: GeminiFile,
}

/// Subset of the Gemini Files API `File` resource we care about.
#[derive(Debug, Deserialize)]
struct GeminiFile {
    /// Resource name, e.g. `"files/abc-123"`. Present on every response.
    #[serde(default)]
    #[allow(dead_code)] // retained for parity with the Files API resource shape
    name: Option<String>,
    /// Canonical URI, e.g.
    /// `"https://generativelanguage.googleapis.com/v1beta/files/abc-123"`.
    /// Stored as the handle id; `fileData.fileUri` consumes this directly.
    #[serde(default)]
    uri: Option<String>,
    /// MIME type the server detected (or echoed) for the upload.
    #[serde(default, rename = "mimeType")]
    mime_type: Option<String>,
    /// Server-reported byte size. Gemini returns this as a string-encoded
    /// int64 (`"sizeBytes": "1234"`), so it's parsed lazily via
    /// [`Self::size_bytes_parsed`].
    #[serde(default, rename = "sizeBytes")]
    size_bytes: Option<String>,
    /// Original display name if the server echoed it.
    #[serde(default, rename = "displayName")]
    display_name: Option<String>,
}

impl GeminiFile {
    /// Parse `sizeBytes` (a JSON string carrying an int64) into a `u64`.
    fn size_bytes_parsed(&self) -> Option<u64> {
        self.size_bytes.as_deref().and_then(|s| s.parse().ok())
    }
}

// ---------------------------------------------------------------------------
// Multipart helpers
// ---------------------------------------------------------------------------

/// Generate a random multipart boundary token. RFC-2046 compliant.
fn generate_boundary() -> String {
    let raw = Uuid::new_v4().simple().to_string();
    format!("----blazen-gemini-{raw}")
}

/// Build a Gemini-style `multipart/related` body with two parts:
///
/// 1. A JSON metadata part: `{"file": {"display_name": "..."}}`.
/// 2. The binary file part with `Content-Type: <mime>`.
///
/// This is the wire shape Gemini's `?uploadType=multipart` endpoint
/// accepts — different from the `multipart/form-data` shape that `OpenAI`
/// and Anthropic Files use.
fn build_multipart(boundary: &str, display_name: &str, mime: &str, bytes: &[u8]) -> Vec<u8> {
    let metadata = serde_json::json!({
        "file": {
            "display_name": display_name,
        }
    })
    .to_string();

    let mut body = Vec::with_capacity(bytes.len() + metadata.len() + 256);

    // metadata part
    body.extend_from_slice(b"--");
    body.extend_from_slice(boundary.as_bytes());
    body.extend_from_slice(b"\r\n");
    body.extend_from_slice(b"Content-Type: application/json; charset=UTF-8\r\n\r\n");
    body.extend_from_slice(metadata.as_bytes());
    body.extend_from_slice(b"\r\n");

    // file part
    body.extend_from_slice(b"--");
    body.extend_from_slice(boundary.as_bytes());
    body.extend_from_slice(b"\r\n");
    body.extend_from_slice(b"Content-Type: ");
    body.extend_from_slice(mime.as_bytes());
    body.extend_from_slice(b"\r\n\r\n");
    body.extend_from_slice(bytes);
    body.extend_from_slice(b"\r\n");

    // closing boundary
    body.extend_from_slice(b"--");
    body.extend_from_slice(boundary.as_bytes());
    body.extend_from_slice(b"--\r\n");

    body
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Pull the canonical resource name (`files/<id>`) out of a stored handle id.
///
/// Accepts:
/// - The full URI form: `https://generativelanguage.googleapis.com/v1beta/files/abc-123`
/// - The bare resource form: `files/abc-123`
///
/// Returns `None` for anything we can't recognise — caller treats that as
/// a no-op delete.
fn parse_file_name(id: &str) -> Option<String> {
    if let Some(idx) = id.find("/files/") {
        // Skip past the leading slash so we keep `files/<id>`.
        let after = &id[idx + 1..];
        if !after.is_empty() {
            return Some(after.to_owned());
        }
    }
    if id.starts_with("files/") {
        return Some(id.to_owned());
    }
    None
}

/// Map an unsuccessful Gemini Files API response onto a [`BlazenError`].
fn map_files_error(url: &str, response: &crate::http::HttpResponse) -> BlazenError {
    match response.status {
        401 | 403 => BlazenError::auth("GeminiFilesStore: authentication failed"),
        429 => BlazenError::RateLimit {
            retry_after_ms: parse_retry_after(&response.headers),
        },
        _ => crate::providers::provider_http_error(PROVIDER_TAG, url, response),
    }
}

/// Local copy of the `Retry-After` parser so this module doesn't need to
/// reach into `openai_format`.
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
    /// in FIFO order.
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
            let mut q = self.responses.lock().unwrap();
            assert!(!q.is_empty(), "MockClient ran out of canned responses");
            Ok(q.remove(0))
        }

        async fn send_streaming(
            &self,
            _request: HttpRequest,
        ) -> Result<(u16, Vec<(String, String)>, ByteStream), BlazenError> {
            unimplemented!("GeminiFilesStore tests do not stream")
        }
    }

    fn upload_ok(name: &str, uri: &str) -> HttpResponse {
        let body = serde_json::json!({
            "file": {
                "name": name,
                "displayName": "hello.txt",
                "mimeType": "text/plain",
                "sizeBytes": "7",
                "uri": uri,
                "state": "ACTIVE",
            }
        })
        .to_string()
        .into_bytes();
        HttpResponse {
            status: 200,
            headers: vec![("content-type".to_owned(), "application/json".to_owned())],
            body,
        }
    }

    #[test]
    fn defaults_match_gemini_endpoint() {
        let store = GeminiFilesStore::new_with_client("ai-test", Arc::new(MockClient::new(vec![])));
        assert_eq!(
            store.base_url(),
            "https://generativelanguage.googleapis.com"
        );
    }

    #[test]
    fn with_base_url_overrides_default() {
        let store = GeminiFilesStore::new_with_client("ai-test", Arc::new(MockClient::new(vec![])))
            .with_base_url("http://localhost:9999/");
        assert_eq!(store.base_url(), "http://localhost:9999/");
        assert_eq!(store.base(), "http://localhost:9999");
    }

    #[test]
    fn build_multipart_has_metadata_and_file_parts() {
        let body = build_multipart("XYZ", "hello.txt", "text/plain", b"payload");
        let s = String::from_utf8(body).unwrap();
        assert!(s.starts_with("--XYZ\r\n"), "body: {s}");
        assert!(s.ends_with("--XYZ--\r\n"), "body: {s}");
        assert!(
            s.contains("Content-Type: application/json; charset=UTF-8"),
            "body: {s}"
        );
        assert!(s.contains("\"display_name\":\"hello.txt\""), "body: {s}");
        assert!(s.contains("Content-Type: text/plain"), "body: {s}");
        assert!(s.contains("\r\n\r\npayload\r\n"), "body: {s}");
    }

    #[test]
    fn parse_file_name_handles_uri_and_bare_forms() {
        assert_eq!(
            parse_file_name("https://generativelanguage.googleapis.com/v1beta/files/abc-123"),
            Some("files/abc-123".to_owned())
        );
        assert_eq!(
            parse_file_name("files/abc-123"),
            Some("files/abc-123".to_owned())
        );
        assert_eq!(parse_file_name("not-a-gemini-id"), None);
        assert_eq!(parse_file_name(""), None);
    }

    #[tokio::test]
    async fn put_bytes_uploads_multipart_and_records_uri_as_id() {
        let client = Arc::new(MockClient::new(vec![upload_ok(
            "files/abc-123",
            "https://generativelanguage.googleapis.com/v1beta/files/abc-123",
        )]));
        let store =
            GeminiFilesStore::new_with_client("ai-test", client.clone() as Arc<dyn HttpClient>);

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

        assert_eq!(
            handle.id,
            "https://generativelanguage.googleapis.com/v1beta/files/abc-123"
        );
        assert_eq!(handle.kind, ContentKind::Document);
        assert_eq!(handle.byte_size, Some(7));
        assert_eq!(handle.mime_type.as_deref(), Some("text/plain"));
        assert_eq!(handle.display_name.as_deref(), Some("hello.txt"));

        let req = client.last();
        assert_eq!(req.method, HttpMethod::Post);
        assert_eq!(
            req.url,
            "https://generativelanguage.googleapis.com/upload/v1beta/files?uploadType=multipart"
        );
        assert!(
            req.headers
                .iter()
                .any(|(k, v)| k.eq_ignore_ascii_case("x-goog-api-key") && v == "ai-test"),
            "missing x-goog-api-key header: {:?}",
            req.headers
        );
        let ct = req
            .headers
            .iter()
            .find(|(k, _)| k.eq_ignore_ascii_case("content-type"))
            .map(|(_, v)| v.as_str())
            .unwrap();
        assert!(
            ct.starts_with("multipart/related; boundary="),
            "unexpected content-type: {ct}"
        );
        let body = req.body.as_ref().expect("missing body");
        let body_str = String::from_utf8_lossy(body);
        assert!(
            body_str.contains("\"display_name\":\"hello.txt\""),
            "body: {body_str}"
        );
        assert!(
            body_str.contains("Content-Type: text/plain"),
            "body: {body_str}"
        );
        assert!(body_str.contains("payload"), "body: {body_str}");
    }

    #[tokio::test]
    async fn resolve_returns_provider_file_with_uri_as_id() {
        let client = Arc::new(MockClient::new(vec![]));
        let store = GeminiFilesStore::new_with_client("ai-test", client as Arc<dyn HttpClient>);

        let handle = ContentHandle::new(
            "https://generativelanguage.googleapis.com/v1beta/files/xyz",
            ContentKind::Image,
        );
        let resolved = store.resolve(&handle).await.unwrap();
        match resolved {
            MediaSource::ProviderFile { provider, id } => {
                assert_eq!(provider, ProviderId::Gemini);
                assert_eq!(
                    id,
                    "https://generativelanguage.googleapis.com/v1beta/files/xyz"
                );
            }
            other => panic!("expected provider file, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn put_provider_file_gemini_records_uri_verbatim_and_skips_upload() {
        let client = Arc::new(MockClient::new(vec![]));
        let store =
            GeminiFilesStore::new_with_client("ai-test", client.clone() as Arc<dyn HttpClient>);

        let handle = store
            .put(
                ContentBody::ProviderFile {
                    provider: ProviderId::Gemini,
                    id: "https://generativelanguage.googleapis.com/v1beta/files/existing".into(),
                },
                ContentHint::default().with_kind(ContentKind::Document),
            )
            .await
            .unwrap();

        assert_eq!(
            handle.id,
            "https://generativelanguage.googleapis.com/v1beta/files/existing"
        );
        assert_eq!(handle.kind, ContentKind::Document);
        assert!(
            client.sent.lock().unwrap().is_empty(),
            "no HTTP traffic should be issued for ProviderFile inputs"
        );
    }

    #[tokio::test]
    async fn put_provider_file_wrong_provider_errors() {
        let client = Arc::new(MockClient::new(vec![]));
        let store = GeminiFilesStore::new_with_client("ai-test", client as Arc<dyn HttpClient>);

        let err = store
            .put(
                ContentBody::ProviderFile {
                    provider: ProviderId::OpenAi,
                    id: "file_oai".into(),
                },
                ContentHint::default(),
            )
            .await
            .expect_err("should reject non-Gemini provider files");
        assert!(
            matches!(err, BlazenError::Unsupported { .. }),
            "got {err:?}"
        );
    }

    #[tokio::test]
    async fn put_url_is_unsupported() {
        let client = Arc::new(MockClient::new(vec![]));
        let store = GeminiFilesStore::new_with_client("ai-test", client as Arc<dyn HttpClient>);

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
    async fn fetch_bytes_is_unsupported() {
        let client = Arc::new(MockClient::new(vec![]));
        let store = GeminiFilesStore::new_with_client("ai-test", client as Arc<dyn HttpClient>);

        let handle = ContentHandle::new(
            "https://generativelanguage.googleapis.com/v1beta/files/abc",
            ContentKind::Image,
        );
        let err = store
            .fetch_bytes(&handle)
            .await
            .expect_err("fetch_bytes should always fail");
        assert!(
            matches!(err, BlazenError::Unsupported { .. }),
            "got {err:?}"
        );
    }

    #[tokio::test]
    async fn delete_hits_v1beta_files_endpoint_and_treats_404_as_ok() {
        let client = Arc::new(MockClient::new(vec![HttpResponse {
            status: 404,
            headers: vec![],
            body: b"{\"error\":{\"code\":404,\"status\":\"NOT_FOUND\"}}".to_vec(),
        }]));
        let store =
            GeminiFilesStore::new_with_client("ai-test", client.clone() as Arc<dyn HttpClient>);

        let handle = ContentHandle::new(
            "https://generativelanguage.googleapis.com/v1beta/files/gone",
            ContentKind::Document,
        );
        store.delete(&handle).await.unwrap();

        let req = client.last();
        assert_eq!(req.method, HttpMethod::Delete);
        assert_eq!(
            req.url,
            "https://generativelanguage.googleapis.com/v1beta/files/gone"
        );
        assert!(
            req.headers
                .iter()
                .any(|(k, v)| k.eq_ignore_ascii_case("x-goog-api-key") && v == "ai-test"),
            "missing x-goog-api-key header on delete: {:?}",
            req.headers
        );
    }

    #[tokio::test]
    async fn delete_with_unrecognised_id_is_a_noop() {
        let client = Arc::new(MockClient::new(vec![]));
        let store =
            GeminiFilesStore::new_with_client("ai-test", client.clone() as Arc<dyn HttpClient>);

        let handle = ContentHandle::new("not-a-gemini-uri", ContentKind::Document);
        store.delete(&handle).await.unwrap();
        assert!(
            client.sent.lock().unwrap().is_empty(),
            "delete on unrecognised id should not issue HTTP traffic"
        );
    }

    #[tokio::test]
    async fn upload_401_maps_to_auth_error() {
        let client = Arc::new(MockClient::new(vec![HttpResponse {
            status: 401,
            headers: vec![],
            body: b"bad key".to_vec(),
        }]));
        let store = GeminiFilesStore::new_with_client("ai-bad", client as Arc<dyn HttpClient>);

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
    async fn upload_429_maps_to_rate_limit() {
        let client = Arc::new(MockClient::new(vec![HttpResponse {
            status: 429,
            headers: vec![("retry-after".to_owned(), "5".to_owned())],
            body: b"slow down".to_vec(),
        }]));
        let store = GeminiFilesStore::new_with_client("ai-test", client as Arc<dyn HttpClient>);

        let err = store
            .put(
                ContentBody::Bytes {
                    data: vec![1, 2, 3],
                },
                ContentHint::default(),
            )
            .await
            .expect_err("expected rate-limit error");
        match err {
            BlazenError::RateLimit { retry_after_ms } => {
                assert_eq!(retry_after_ms, Some(5_000));
            }
            other => panic!("expected RateLimit, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn stream_put_drains_to_bytes_path() {
        use bytes::Bytes;
        use futures_util::stream;

        let client = Arc::new(MockClient::new(vec![upload_ok(
            "files/streamed",
            "https://generativelanguage.googleapis.com/v1beta/files/streamed",
        )]));
        let store =
            GeminiFilesStore::new_with_client("ai-test", client.clone() as Arc<dyn HttpClient>);

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
        assert_eq!(
            handle.id,
            "https://generativelanguage.googleapis.com/v1beta/files/streamed"
        );

        // Confirm the drained stream was forwarded into the multipart body
        // exactly once, via the same upload path used by `ContentBody::Bytes`.
        let req = client.last();
        assert_eq!(req.method, HttpMethod::Post);
        assert_eq!(
            req.url,
            "https://generativelanguage.googleapis.com/upload/v1beta/files?uploadType=multipart"
        );
        let body_bytes = req.body.as_ref().expect("missing body");
        let body_str = String::from_utf8_lossy(body_bytes);
        assert!(
            body_str.contains("hello world"),
            "drained stream payload missing from multipart body: {body_str}"
        );
    }

    #[tokio::test]
    async fn upload_response_missing_uri_errors() {
        // Gemini server returned a file but no `uri` field — treat as a
        // protocol error, not a silent missing handle.
        let body = serde_json::json!({
            "file": {
                "name": "files/abc",
                "displayName": "x",
                "mimeType": "text/plain",
                "sizeBytes": "3",
            }
        })
        .to_string()
        .into_bytes();
        let client = Arc::new(MockClient::new(vec![HttpResponse {
            status: 200,
            headers: vec![("content-type".to_owned(), "application/json".to_owned())],
            body,
        }]));
        let store = GeminiFilesStore::new_with_client("ai-test", client as Arc<dyn HttpClient>);

        let err = store
            .put(
                ContentBody::Bytes {
                    data: vec![1, 2, 3],
                },
                ContentHint::default(),
            )
            .await
            .expect_err("expected protocol error from missing uri");
        assert!(matches!(err, BlazenError::Request { .. }), "got {err:?}");
    }
}
