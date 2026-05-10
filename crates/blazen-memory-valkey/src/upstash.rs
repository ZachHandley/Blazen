//! Upstash Redis REST backend. Wasi-compatible alternative to the native
//! TCP [`ValkeyBackend`](super::ValkeyBackend).
//!
//! Talks to Upstash's REST API over `Arc<dyn HttpClient>`. Uses the IDENTICAL
//! key layout as `ValkeyBackend`:
//!
//! | Key pattern                  | Type   | Contents                        |
//! |------------------------------|--------|---------------------------------|
//! | `{prefix}entry:{id}`         | STRING | JSON-serialized `StoredEntry`   |
//! | `{prefix}bands:{band_value}` | SET    | Entry IDs sharing this LSH band |
//! | `{prefix}ids`                | SET    | All entry IDs                   |
//!
//! Multi-key operations are pipelined via the Upstash `/pipeline` endpoint
//! (a JSON array of Redis command arrays). Note that Upstash's REST pipeline
//! is NOT atomic in the MULTI/EXEC sense — see Upstash docs for `/multi-exec`
//! if you need transactional semantics.

use std::sync::Arc;

use async_trait::async_trait;
use blazen_llm::http::{HttpClient, HttpMethod, HttpRequest};
use serde::Deserialize;
use serde_json::{Value, json};
use tracing::instrument;

use blazen_memory::error::{MemoryError, Result};
use blazen_memory::store::MemoryBackend;
use blazen_memory::types::StoredEntry;

/// Default key prefix used for namespacing in Upstash Redis.
const DEFAULT_PREFIX: &str = "blazen:memory:";

/// A [`MemoryBackend`] implementation backed by Upstash's Redis REST API.
///
/// Designed for environments where the native `redis` crate cannot be used
/// (Cloudflare Workers, Deno, browser-side wasm, wasi runtimes). All I/O is
/// dispatched through a user-supplied [`HttpClient`], so the same backend
/// works on any target that has an HTTP client implementation.
pub struct UpstashBackend {
    rest_url: String,
    rest_token: String,
    http: Arc<dyn HttpClient>,
    prefix: String,
}

impl std::fmt::Debug for UpstashBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("UpstashBackend")
            .field("rest_url", &self.rest_url)
            .field("prefix", &self.prefix)
            .finish_non_exhaustive()
    }
}

impl UpstashBackend {
    /// Create a new `UpstashBackend`.
    ///
    /// `rest_url` is the Upstash REST endpoint (e.g.
    /// `https://us1-merry-cat-32242.upstash.io`); a trailing slash, if
    /// present, is stripped.
    ///
    /// `rest_token` is the Upstash REST token, sent as a `Bearer` token on
    /// every request.
    ///
    /// `http` is the HTTP client used for all network I/O — typically an
    /// `Arc<ReqwestHttpClient>` on native or a `WasiHttpClient` on
    /// wasi-style targets.
    pub fn new(
        rest_url: impl Into<String>,
        rest_token: impl Into<String>,
        http: Arc<dyn HttpClient>,
    ) -> Self {
        let url: String = rest_url.into();
        Self {
            rest_url: url.trim_end_matches('/').to_owned(),
            rest_token: rest_token.into(),
            http,
            prefix: DEFAULT_PREFIX.to_owned(),
        }
    }

    /// Override the default key prefix (`blazen:memory:`).
    ///
    /// This is useful when running multiple logical stores against the same
    /// Upstash database.
    #[must_use]
    pub fn with_prefix(mut self, prefix: impl Into<String>) -> Self {
        self.prefix = prefix.into();
        self
    }

    // -- key helpers ----------------------------------------------------------

    fn entry_key(&self, id: &str) -> String {
        format!("{}entry:{id}", self.prefix)
    }

    fn band_key(&self, band: &str) -> String {
        format!("{}bands:{band}", self.prefix)
    }

    fn ids_key(&self) -> String {
        format!("{}ids", self.prefix)
    }

    // -- HTTP helpers ---------------------------------------------------------

    /// Send a single Redis command via Upstash's POST-with-array endpoint.
    ///
    /// The body is a JSON array `["CMD", "arg1", "arg2", ...]`, posted to the
    /// REST root. Returns the parsed `result` value.
    async fn command(&self, args: Vec<Value>) -> Result<Value> {
        let body = serde_json::to_vec(&args)?;
        let request = HttpRequest {
            method: HttpMethod::Post,
            url: self.rest_url.clone(),
            headers: vec![
                (
                    "Authorization".to_owned(),
                    format!("Bearer {}", self.rest_token),
                ),
                ("Content-Type".to_owned(), "application/json".to_owned()),
            ],
            body: Some(body),
            query_params: Vec::new(),
        };
        let resp = self
            .http
            .send(request)
            .await
            .map_err(|e| MemoryError::Backend(format!("Upstash HTTP error: {e}")))?;
        if !resp.is_success() {
            return Err(MemoryError::Backend(format!(
                "Upstash {} error: {}",
                resp.status,
                resp.text()
            )));
        }
        let reply: UpstashReply = serde_json::from_slice(&resp.body)?;
        if let Some(err) = reply.error {
            return Err(MemoryError::Backend(format!(
                "Upstash command error: {err}"
            )));
        }
        Ok(reply.result)
    }

    /// Send a JSON pipeline of Redis commands. Returns a vector of replies
    /// in the same order as the input commands.
    async fn pipeline(&self, commands: Vec<Vec<Value>>) -> Result<Vec<UpstashReply>> {
        let body = serde_json::to_vec(&commands)?;
        let request = HttpRequest {
            method: HttpMethod::Post,
            url: format!("{}/pipeline", self.rest_url),
            headers: vec![
                (
                    "Authorization".to_owned(),
                    format!("Bearer {}", self.rest_token),
                ),
                ("Content-Type".to_owned(), "application/json".to_owned()),
            ],
            body: Some(body),
            query_params: Vec::new(),
        };
        let resp = self
            .http
            .send(request)
            .await
            .map_err(|e| MemoryError::Backend(format!("Upstash HTTP error: {e}")))?;
        if !resp.is_success() {
            return Err(MemoryError::Backend(format!(
                "Upstash {} error: {}",
                resp.status,
                resp.text()
            )));
        }
        let replies: Vec<UpstashReply> = serde_json::from_slice(&resp.body)?;
        // Surface the first error, if any, so callers don't get silently-failed
        // pipeline steps.
        for r in &replies {
            if let Some(err) = &r.error {
                return Err(MemoryError::Backend(format!(
                    "Upstash pipeline error: {err}"
                )));
            }
        }
        Ok(replies)
    }
}

/// One element of an Upstash REST response (single-command body or pipeline
/// element). Either `result` or `error` is populated.
#[derive(Debug, Deserialize)]
struct UpstashReply {
    #[serde(default)]
    result: Value,
    #[serde(default)]
    error: Option<String>,
}

/// Decode an Upstash `result` value into `Option<String>`.
///
/// Upstash returns `null` for missing keys and a JSON string for present
/// values (even when the underlying value was bytes — Upstash base64-encodes
/// raw binary, but our payloads are always UTF-8 JSON, so a plain string is
/// what we get back).
fn result_as_opt_string(value: Value) -> Result<Option<String>> {
    match value {
        Value::Null => Ok(None),
        Value::String(s) => Ok(Some(s)),
        other => Err(MemoryError::Backend(format!(
            "Upstash: expected string|null, got {other}"
        ))),
    }
}

/// Decode an Upstash `result` value into `Vec<String>` (used for SMEMBERS,
/// SUNION, etc.).
fn result_as_string_vec(value: Value) -> Result<Vec<String>> {
    match value {
        Value::Null => Ok(Vec::new()),
        Value::Array(items) => items
            .into_iter()
            .map(|v| match v {
                Value::String(s) => Ok(s),
                Value::Null => Err(MemoryError::Backend(
                    "Upstash: unexpected null in string array".to_owned(),
                )),
                other => Err(MemoryError::Backend(format!(
                    "Upstash: expected string in array, got {other}"
                ))),
            })
            .collect(),
        other => Err(MemoryError::Backend(format!(
            "Upstash: expected array, got {other}"
        ))),
    }
}

/// Decode an Upstash `result` value into `usize` (used for SCARD, etc.).
fn result_as_usize(value: Value) -> Result<usize> {
    match value {
        Value::Number(n) => {
            let raw = n.as_u64().ok_or_else(|| {
                MemoryError::Backend(format!("Upstash: number {n} is not a valid u64"))
            })?;
            usize::try_from(raw).map_err(|_| {
                MemoryError::Backend(format!(
                    "Upstash: count {raw} exceeds usize on this platform"
                ))
            })
        }
        other => Err(MemoryError::Backend(format!(
            "Upstash: expected number, got {other}"
        ))),
    }
}

/// Decode an Upstash MGET `result` (an array of string|null) into a vector of
/// optional strings.
fn result_as_mget(value: Value) -> Result<Vec<Option<String>>> {
    match value {
        Value::Array(items) => items
            .into_iter()
            .map(|v| match v {
                Value::Null => Ok(None),
                Value::String(s) => Ok(Some(s)),
                other => Err(MemoryError::Backend(format!(
                    "Upstash: expected string|null in mget array, got {other}"
                ))),
            })
            .collect(),
        Value::Null => Ok(Vec::new()),
        other => Err(MemoryError::Backend(format!(
            "Upstash: expected array, got {other}"
        ))),
    }
}

#[async_trait]
impl MemoryBackend for UpstashBackend {
    #[instrument(skip(self, entry), fields(id = %entry.id))]
    async fn put(&self, entry: StoredEntry) -> Result<()> {
        let json_body = serde_json::to_string(&entry)?;
        let entry_key = self.entry_key(&entry.id);
        let ids_key = self.ids_key();

        // Pipeline: SET entry, SADD to ids set, SADD to each band set.
        let mut pipeline: Vec<Vec<Value>> = Vec::with_capacity(2 + entry.bands.len());
        pipeline.push(vec![
            json!("SET"),
            Value::String(entry_key),
            Value::String(json_body),
        ]);
        pipeline.push(vec![
            json!("SADD"),
            Value::String(ids_key),
            Value::String(entry.id.clone()),
        ]);
        for band in &entry.bands {
            pipeline.push(vec![
                json!("SADD"),
                Value::String(self.band_key(band)),
                Value::String(entry.id.clone()),
            ]);
        }

        self.pipeline(pipeline).await?;
        Ok(())
    }

    #[instrument(skip(self))]
    async fn get(&self, id: &str) -> Result<Option<StoredEntry>> {
        let value = self
            .command(vec![json!("GET"), Value::String(self.entry_key(id))])
            .await?;
        match result_as_opt_string(value)? {
            Some(raw) => {
                let entry: StoredEntry = serde_json::from_str(&raw)?;
                Ok(Some(entry))
            }
            None => Ok(None),
        }
    }

    #[instrument(skip(self))]
    async fn delete(&self, id: &str) -> Result<bool> {
        // Fetch the entry first so we know which band sets to clean up.
        let entry_key = self.entry_key(id);
        let value = self
            .command(vec![json!("GET"), Value::String(entry_key.clone())])
            .await?;
        let entry = match result_as_opt_string(value)? {
            Some(raw) => serde_json::from_str::<StoredEntry>(&raw)?,
            None => return Ok(false),
        };

        // Pipeline: DEL entry key, SREM from ids set, SREM from each band set.
        let mut pipeline: Vec<Vec<Value>> = Vec::with_capacity(2 + entry.bands.len());
        pipeline.push(vec![json!("DEL"), Value::String(entry_key)]);
        pipeline.push(vec![
            json!("SREM"),
            Value::String(self.ids_key()),
            Value::String(id.to_owned()),
        ]);
        for band in &entry.bands {
            pipeline.push(vec![
                json!("SREM"),
                Value::String(self.band_key(band)),
                Value::String(id.to_owned()),
            ]);
        }

        self.pipeline(pipeline).await?;
        Ok(true)
    }

    #[instrument(skip(self))]
    async fn list(&self) -> Result<Vec<StoredEntry>> {
        let value = self
            .command(vec![json!("SMEMBERS"), Value::String(self.ids_key())])
            .await?;
        let ids = result_as_string_vec(value)?;

        if ids.is_empty() {
            return Ok(Vec::new());
        }

        // MGET all entry keys at once.
        let mut args: Vec<Value> = Vec::with_capacity(1 + ids.len());
        args.push(json!("MGET"));
        for id in &ids {
            args.push(Value::String(self.entry_key(id)));
        }
        let value = self.command(args).await?;
        let values = result_as_mget(value)?;

        let mut entries = Vec::with_capacity(values.len());
        for raw in values.into_iter().flatten() {
            let entry: StoredEntry = serde_json::from_str(&raw)?;
            entries.push(entry);
        }

        Ok(entries)
    }

    #[instrument(skip(self))]
    async fn len(&self) -> Result<usize> {
        let value = self
            .command(vec![json!("SCARD"), Value::String(self.ids_key())])
            .await?;
        result_as_usize(value)
    }

    #[instrument(skip(self, bands))]
    async fn search_by_bands(&self, bands: &[String], limit: usize) -> Result<Vec<StoredEntry>> {
        if bands.is_empty() {
            return Ok(Vec::new());
        }

        // SUNION across all matching band sets to get candidate IDs.
        let mut args: Vec<Value> = Vec::with_capacity(1 + bands.len());
        args.push(json!("SUNION"));
        for b in bands {
            args.push(Value::String(self.band_key(b)));
        }
        let value = self.command(args).await?;
        let candidate_ids = result_as_string_vec(value)?;

        if candidate_ids.is_empty() {
            return Ok(Vec::new());
        }

        // Cap candidates to limit before fetching full entries.
        let capped: &[String] = if candidate_ids.len() > limit {
            &candidate_ids[..limit]
        } else {
            &candidate_ids
        };

        let mut args: Vec<Value> = Vec::with_capacity(1 + capped.len());
        args.push(json!("MGET"));
        for id in capped {
            args.push(Value::String(self.entry_key(id)));
        }
        let value = self.command(args).await?;
        let values = result_as_mget(value)?;

        let mut entries = Vec::with_capacity(values.len());
        for raw in values.into_iter().flatten() {
            let entry: StoredEntry = serde_json::from_str(&raw)?;
            entries.push(entry);
        }

        Ok(entries)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use blazen_llm::error::BlazenError;
    use blazen_llm::http::{ByteStream, HttpResponse};
    use std::sync::Mutex;

    /// A scripted HTTP client that returns canned responses keyed by request
    /// order. Captures every request sent for assertions.
    #[derive(Debug)]
    struct MockHttp {
        responses: Mutex<std::collections::VecDeque<HttpResponse>>,
        captured: Mutex<Vec<HttpRequest>>,
    }

    impl MockHttp {
        fn new(responses: Vec<HttpResponse>) -> Arc<Self> {
            Arc::new(Self {
                responses: Mutex::new(responses.into_iter().collect()),
                captured: Mutex::new(Vec::new()),
            })
        }

        fn captured(&self) -> Vec<HttpRequest> {
            self.captured.lock().unwrap().clone()
        }
    }

    #[async_trait]
    impl HttpClient for MockHttp {
        async fn send(
            &self,
            request: HttpRequest,
        ) -> std::result::Result<HttpResponse, BlazenError> {
            self.captured.lock().unwrap().push(request);
            self.responses
                .lock()
                .unwrap()
                .pop_front()
                .ok_or_else(|| BlazenError::Request {
                    message: "MockHttp: no more responses".to_owned(),
                    source: None,
                })
        }

        async fn send_streaming(
            &self,
            _request: HttpRequest,
        ) -> std::result::Result<(u16, Vec<(String, String)>, ByteStream), BlazenError> {
            Err(BlazenError::Request {
                message: "MockHttp::send_streaming not implemented".to_owned(),
                source: None,
            })
        }
    }

    fn ok_response(body: &Value) -> HttpResponse {
        HttpResponse {
            status: 200,
            headers: Vec::new(),
            body: serde_json::to_vec(body).unwrap(),
        }
    }

    fn make_entry(id: &str, text: &str, bands: Vec<String>) -> StoredEntry {
        StoredEntry {
            id: id.to_owned(),
            text: text.to_owned(),
            elid: None,
            simhash_hex: None,
            text_simhash: 0,
            bands,
            metadata: serde_json::Value::Null,
        }
    }

    fn backend(http: Arc<dyn HttpClient>) -> UpstashBackend {
        UpstashBackend::new("https://example.upstash.io/", "test-token", http).with_prefix("test:")
    }

    #[tokio::test]
    async fn put_pipelines_set_sadd_and_band_sadds() {
        // Pipeline returns one reply per command — 1 SET + 1 SADD ids + 2 SADD bands.
        let http = MockHttp::new(vec![ok_response(&json!([
            {"result": "OK"},
            {"result": 1},
            {"result": 1},
            {"result": 1},
        ]))]);
        let b = backend(http.clone());

        b.put(make_entry("e1", "hi", vec!["b0".into(), "b1".into()]))
            .await
            .unwrap();

        let captured = http.captured();
        assert_eq!(captured.len(), 1);
        let req = &captured[0];
        assert_eq!(req.method, HttpMethod::Post);
        assert_eq!(req.url, "https://example.upstash.io/pipeline");
        assert!(
            req.headers
                .iter()
                .any(|(k, v)| k == "Authorization" && v == "Bearer test-token")
        );

        // Decode the pipeline body and check command shapes.
        let body = req.body.as_ref().unwrap();
        let cmds: Vec<Vec<Value>> = serde_json::from_slice(body).unwrap();
        assert_eq!(cmds.len(), 4);
        assert_eq!(cmds[0][0], json!("SET"));
        assert_eq!(cmds[0][1], json!("test:entry:e1"));
        assert_eq!(cmds[1][0], json!("SADD"));
        assert_eq!(cmds[1][1], json!("test:ids"));
        assert_eq!(cmds[1][2], json!("e1"));
        assert_eq!(cmds[2][0], json!("SADD"));
        assert_eq!(cmds[2][1], json!("test:bands:b0"));
        assert_eq!(cmds[3][0], json!("SADD"));
        assert_eq!(cmds[3][1], json!("test:bands:b1"));
    }

    #[tokio::test]
    async fn get_returns_decoded_entry() {
        let entry = make_entry("e1", "hello", vec!["b".into()]);
        let blob = serde_json::to_string(&entry).unwrap();
        let http = MockHttp::new(vec![ok_response(&json!({"result": blob}))]);
        let b = backend(http.clone());

        let got = b.get("e1").await.unwrap().unwrap();
        assert_eq!(got.id, "e1");
        assert_eq!(got.text, "hello");

        let captured = http.captured();
        assert_eq!(captured.len(), 1);
        let cmd: Vec<Value> = serde_json::from_slice(captured[0].body.as_ref().unwrap()).unwrap();
        assert_eq!(cmd, vec![json!("GET"), json!("test:entry:e1")]);
        // Single-command path posts to the REST root, not /pipeline.
        assert_eq!(captured[0].url, "https://example.upstash.io");
    }

    #[tokio::test]
    async fn get_returns_none_for_missing() {
        let http = MockHttp::new(vec![ok_response(&json!({"result": null}))]);
        let b = backend(http);

        assert!(b.get("nope").await.unwrap().is_none());
    }

    #[tokio::test]
    async fn delete_returns_false_for_missing() {
        let http = MockHttp::new(vec![ok_response(&json!({"result": null}))]);
        let b = backend(http);

        assert!(!b.delete("nope").await.unwrap());
    }

    #[tokio::test]
    async fn delete_pipelines_del_srem_and_band_srems() {
        let entry = make_entry("e1", "bye", vec!["b0".into(), "b1".into()]);
        let blob = serde_json::to_string(&entry).unwrap();
        let http = MockHttp::new(vec![
            ok_response(&json!({"result": blob})), // GET
            ok_response(&json!([
                {"result": 1}, // DEL
                {"result": 1}, // SREM ids
                {"result": 1}, // SREM bands b0
                {"result": 1}, // SREM bands b1
            ])),
        ]);
        let b = backend(http.clone());

        assert!(b.delete("e1").await.unwrap());
        let captured = http.captured();
        assert_eq!(captured.len(), 2);

        // Second request is the pipeline.
        let cmds: Vec<Vec<Value>> =
            serde_json::from_slice(captured[1].body.as_ref().unwrap()).unwrap();
        assert_eq!(cmds.len(), 4);
        assert_eq!(cmds[0][0], json!("DEL"));
        assert_eq!(cmds[1][0], json!("SREM"));
        assert_eq!(cmds[1][1], json!("test:ids"));
        assert_eq!(cmds[2][0], json!("SREM"));
        assert_eq!(cmds[2][1], json!("test:bands:b0"));
        assert_eq!(cmds[3][1], json!("test:bands:b1"));
    }

    #[tokio::test]
    async fn len_decodes_scard() {
        let http = MockHttp::new(vec![ok_response(&json!({"result": 42}))]);
        let b = backend(http);

        assert_eq!(b.len().await.unwrap(), 42);
    }

    #[tokio::test]
    async fn list_smembers_then_mget() {
        let entry = make_entry("a", "alpha", vec![]);
        let blob = serde_json::to_string(&entry).unwrap();
        let http = MockHttp::new(vec![
            ok_response(&json!({"result": ["a"]})),  // SMEMBERS
            ok_response(&json!({"result": [blob]})), // MGET
        ]);
        let b = backend(http.clone());

        let entries = b.list().await.unwrap();
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].id, "a");

        let captured = http.captured();
        let smembers: Vec<Value> =
            serde_json::from_slice(captured[0].body.as_ref().unwrap()).unwrap();
        assert_eq!(smembers, vec![json!("SMEMBERS"), json!("test:ids")]);
        let mget: Vec<Value> = serde_json::from_slice(captured[1].body.as_ref().unwrap()).unwrap();
        assert_eq!(mget, vec![json!("MGET"), json!("test:entry:a")]);
    }

    #[tokio::test]
    async fn list_empty_returns_no_mget() {
        let http = MockHttp::new(vec![ok_response(&json!({"result": []}))]);
        let b = backend(http.clone());

        let entries = b.list().await.unwrap();
        assert!(entries.is_empty());
        // Only one call (SMEMBERS) — no MGET because there's nothing to fetch.
        assert_eq!(http.captured().len(), 1);
    }

    #[tokio::test]
    async fn search_by_bands_empty_input_returns_empty() {
        let http = MockHttp::new(vec![]);
        let b = backend(http.clone());

        let results = b.search_by_bands(&[], 10).await.unwrap();
        assert!(results.is_empty());
        // No HTTP call at all for empty input.
        assert!(http.captured().is_empty());
    }

    #[tokio::test]
    async fn search_by_bands_sunion_then_mget_capped_by_limit() {
        let entry = make_entry("a", "alpha", vec!["b".into()]);
        let blob = serde_json::to_string(&entry).unwrap();
        let http = MockHttp::new(vec![
            ok_response(&json!({"result": ["a", "b", "c"]})), // SUNION
            ok_response(&json!({"result": [blob]})),          // MGET (capped)
        ]);
        let b = backend(http.clone());

        let results = b.search_by_bands(&["x".into()], 1).await.unwrap();
        assert_eq!(results.len(), 1);

        let captured = http.captured();
        let sunion: Vec<Value> =
            serde_json::from_slice(captured[0].body.as_ref().unwrap()).unwrap();
        assert_eq!(sunion, vec![json!("SUNION"), json!("test:bands:x")]);
        let mget: Vec<Value> = serde_json::from_slice(captured[1].body.as_ref().unwrap()).unwrap();
        // MGET should be capped at limit=1, so only `a` is requested.
        assert_eq!(mget, vec![json!("MGET"), json!("test:entry:a")]);
    }

    #[tokio::test]
    async fn http_error_status_surfaces_as_backend_error() {
        let http = MockHttp::new(vec![HttpResponse {
            status: 500,
            headers: Vec::new(),
            body: b"upstream down".to_vec(),
        }]);
        let b = backend(http);

        let err = b.get("e1").await.unwrap_err();
        assert!(matches!(err, MemoryError::Backend(_)), "got {err:?}");
    }

    #[tokio::test]
    async fn pipeline_step_error_surfaces_as_backend_error() {
        let http = MockHttp::new(vec![ok_response(&json!([
            {"result": "OK"},
            {"error": "WRONGTYPE Operation against a key holding the wrong kind of value"},
            {"result": 1},
        ]))]);
        let b = backend(http);

        let err = b
            .put(make_entry("e1", "hi", vec!["b0".into()]))
            .await
            .unwrap_err();
        assert!(matches!(err, MemoryError::Backend(_)), "got {err:?}");
    }

    #[test]
    fn new_strips_trailing_slash() {
        let http = MockHttp::new(vec![]);
        let b = UpstashBackend::new("https://x.upstash.io/", "tok", http);
        assert_eq!(b.rest_url, "https://x.upstash.io");
    }

    #[test]
    fn key_helpers_use_default_prefix() {
        let http = MockHttp::new(vec![]);
        let b = UpstashBackend::new("https://x.upstash.io", "tok", http);
        assert_eq!(b.entry_key("e1"), "blazen:memory:entry:e1");
        assert_eq!(b.band_key("b0"), "blazen:memory:bands:b0");
        assert_eq!(b.ids_key(), "blazen:memory:ids");
    }
}
