//! Shared helper for `OpenAI`-compatible TTS (`POST /v1/audio/speech`).
//!
//! Both [`super::openai::OpenAiProvider`] and
//! [`super::openai_compat::OpenAiCompatProvider`] delegate to
//! [`text_to_speech_request`] in their `AudioGeneration::text_to_speech`
//! implementations. The same wire format is accepted by `OpenAI` proper and
//! by `OpenAI`-compatible services (zvoice / `VoxCPM2`, Groq, Together, etc.),
//! so the only per-provider config is the `base_url` and optional API key.

#[cfg(not(target_arch = "wasm32"))]
use std::time::Instant;
#[cfg(target_arch = "wasm32")]
use web_time::Instant;

use base64::Engine;
use serde_json::{Map, Value, json};

use super::openai_format::parse_retry_after;
use super::provider_http_error;
use crate::compute::requests::SpeechRequest;
use crate::compute::results::AudioResult;
use crate::error::BlazenError;
use crate::http::{HttpClient, HttpRequest};
use crate::media::{GeneratedAudio, MediaOutput, MediaType};
use crate::types::RequestTiming;

/// Default model string when the caller does not set `SpeechRequest.model`.
const DEFAULT_TTS_MODEL: &str = "tts-1";
/// Default voice when the caller does not set `SpeechRequest.voice`.
const DEFAULT_TTS_VOICE: &str = "alloy";
/// Default response format (`OpenAI`'s server honours `mp3`, `wav`, `flac`, `opus`, `pcm`).
const DEFAULT_RESPONSE_FORMAT: &str = "mp3";

/// Send an `OpenAI`-compatible `POST /v1/audio/speech` request and return the
/// audio as an [`AudioResult`]. Used by both the native `OpenAI` provider and
/// the generic `OpenAI`-compatible provider.
///
/// # Arguments
///
/// * `client` — HTTP client to use.
/// * `base_url` — provider base URL, e.g. `https://api.openai.com/v1` or
///   `http://beastpc.lan:8900/v1`. A trailing slash is tolerated.
/// * `api_key` — bearer token. Pass an empty string to skip the
///   `Authorization` header entirely (for unauth'd local services).
/// * `request` — typed speech request.
///
/// # Errors
///
/// Maps HTTP responses as follows:
/// - 401 → [`BlazenError::auth`]
/// - 429 → [`BlazenError::RateLimit`] with `Retry-After` if present
/// - any other non-2xx → [`BlazenError::ProviderHttp`] carrying status, endpoint, body detail
pub(crate) async fn text_to_speech_request(
    client: &dyn HttpClient,
    base_url: &str,
    api_key: &str,
    request: SpeechRequest,
) -> Result<AudioResult, BlazenError> {
    let start = Instant::now();

    let url = format!("{}/audio/speech", base_url.trim_end_matches('/'));

    // Build the OpenAI-spec body. `parameters` gets merged as additional
    // top-level fields so callers can pass zvoice/Groq extensions without
    // breaking the standard OpenAI shape.
    let model = request
        .model
        .as_deref()
        .unwrap_or(DEFAULT_TTS_MODEL)
        .to_owned();
    let voice = request
        .voice
        .as_deref()
        .unwrap_or(DEFAULT_TTS_VOICE)
        .to_owned();

    let mut body = json!({
        "model": model,
        "input": request.text,
        "voice": voice,
        "response_format": DEFAULT_RESPONSE_FORMAT,
    });
    if let Some(speed) = request.speed {
        body["speed"] = json!(speed);
    }
    if let Some(ref voice_url) = request.voice_url {
        // zvoice (and some other compat services) read this; OpenAI
        // proper ignores unknown fields.
        body["voice_url"] = json!(voice_url);
    }
    merge_parameters(&mut body, &request.parameters);

    // Build the HTTP request. Skip bearer auth if api_key is empty.
    let mut http_request = HttpRequest::post(&url).json_body(&body)?;
    if !api_key.is_empty() {
        http_request = http_request.bearer_auth(api_key);
    }

    let response = client.send(http_request).await?;

    if !response.is_success() {
        return Err(match response.status {
            401 => BlazenError::auth("authentication failed"),
            429 => BlazenError::RateLimit {
                retry_after_ms: parse_retry_after(&response.headers),
            },
            _ => provider_http_error("openai", &url, &response),
        });
    }

    // Extract content type and base64-encode the bytes. `MediaOutput.base64`
    // is the right field for binary payloads; `raw_content` is text-only.
    let content_type = response
        .header("content-type")
        .unwrap_or("audio/mpeg")
        .to_owned();
    let file_size = u64::try_from(response.body.len()).ok();
    let encoded = base64::engine::general_purpose::STANDARD.encode(&response.body);

    let elapsed_ms = u64::try_from(start.elapsed().as_millis()).unwrap_or(u64::MAX);

    Ok(AudioResult {
        audio: vec![GeneratedAudio {
            media: MediaOutput {
                url: None,
                base64: Some(encoded),
                raw_content: None,
                media_type: MediaType::from_mime(&content_type),
                file_size,
                metadata: Value::Object(Map::new()),
            },
            duration_seconds: None,
            sample_rate: None,
            channels: None,
        }],
        timing: RequestTiming {
            queue_ms: None,
            execution_ms: None,
            total_ms: Some(elapsed_ms),
        },
        cost: None,
        metadata: Value::Object(Map::new()),
    })
}

/// Merge a freeform parameters object into an `OpenAI`-spec body, preserving
/// existing keys. If `params` is not an object, it is ignored.
fn merge_parameters(body: &mut Value, params: &Value) {
    if let (Some(body_obj), Some(params_obj)) = (body.as_object_mut(), params.as_object()) {
        for (k, v) in params_obj {
            body_obj.entry(k.clone()).or_insert_with(|| v.clone());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::http::{HttpMethod, HttpResponse};
    use async_trait::async_trait;
    use std::sync::{Arc, Mutex};

    /// Mock client that records the last request and returns a canned response.
    #[derive(Debug)]
    struct MockClient {
        last: Arc<Mutex<Option<HttpRequest>>>,
        response: Arc<Mutex<HttpResponse>>,
    }

    impl MockClient {
        fn new(response: HttpResponse) -> Self {
            Self {
                last: Arc::new(Mutex::new(None)),
                response: Arc::new(Mutex::new(response)),
            }
        }

        fn last_request(&self) -> HttpRequest {
            self.last
                .lock()
                .unwrap()
                .clone()
                .expect("no request captured")
        }
    }

    #[async_trait]
    impl HttpClient for MockClient {
        async fn send(&self, request: HttpRequest) -> Result<HttpResponse, BlazenError> {
            *self.last.lock().unwrap() = Some(request);
            Ok(self.response.lock().unwrap().clone())
        }

        async fn send_streaming(
            &self,
            _request: HttpRequest,
        ) -> Result<(u16, Vec<(String, String)>, crate::http::ByteStream), BlazenError> {
            unimplemented!("not needed for tests")
        }
    }

    fn ok_audio_response() -> HttpResponse {
        HttpResponse {
            status: 200,
            headers: vec![("content-type".to_owned(), "audio/mpeg".to_owned())],
            body: b"fake-mp3-bytes".to_vec(),
        }
    }

    #[tokio::test]
    async fn posts_standard_openai_body() {
        let client = MockClient::new(ok_audio_response());
        let request = SpeechRequest::new("hello world");

        let result =
            text_to_speech_request(&client, "https://api.openai.com/v1", "sk-test", request)
                .await
                .unwrap();

        // Response correctness
        assert_eq!(result.audio.len(), 1);
        assert!(result.audio[0].media.base64.is_some());
        let decoded = base64::engine::general_purpose::STANDARD
            .decode(result.audio[0].media.base64.as_ref().unwrap())
            .unwrap();
        assert_eq!(decoded, b"fake-mp3-bytes");

        // Request correctness
        let req = client.last_request();
        assert_eq!(req.method, HttpMethod::Post);
        assert_eq!(req.url, "https://api.openai.com/v1/audio/speech");
        assert!(
            req.headers
                .iter()
                .any(|(k, v)| k == "Authorization" && v == "Bearer sk-test")
        );
        let body: Value = serde_json::from_slice(req.body.as_ref().unwrap()).unwrap();
        assert_eq!(body["model"], "tts-1");
        assert_eq!(body["input"], "hello world");
        assert_eq!(body["voice"], "alloy");
        assert_eq!(body["response_format"], "mp3");
    }

    #[tokio::test]
    async fn skips_auth_header_when_api_key_empty() {
        let client = MockClient::new(ok_audio_response());
        let request = SpeechRequest::new("zvoice test").with_voice("custom");

        text_to_speech_request(&client, "http://beastpc.lan:8900/v1", "", request)
            .await
            .unwrap();

        let req = client.last_request();
        assert!(
            req.headers.iter().all(|(k, _)| k != "Authorization"),
            "expected no Authorization header for zvoice (empty api_key)"
        );
        assert_eq!(req.url, "http://beastpc.lan:8900/v1/audio/speech");
        let body: Value = serde_json::from_slice(req.body.as_ref().unwrap()).unwrap();
        assert_eq!(body["voice"], "custom");
    }

    #[tokio::test]
    async fn merges_freeform_parameters_into_body() {
        let client = MockClient::new(ok_audio_response());
        let mut request = SpeechRequest::new("clone me");
        request.voice_url = Some("https://example.com/ref.wav".to_owned());
        request.parameters = json!({
            "stability": 0.6,
            "similarity_boost": 0.8,
        });

        text_to_speech_request(&client, "https://api.openai.com/v1", "sk-test", request)
            .await
            .unwrap();

        let req = client.last_request();
        let body: Value = serde_json::from_slice(req.body.as_ref().unwrap()).unwrap();
        assert_eq!(body["voice_url"], "https://example.com/ref.wav");
        assert_eq!(body["stability"], 0.6);
        assert_eq!(body["similarity_boost"], 0.8);
    }

    #[tokio::test]
    async fn maps_401_to_auth_error() {
        let client = MockClient::new(HttpResponse {
            status: 401,
            headers: vec![],
            body: b"invalid api key".to_vec(),
        });

        let err = text_to_speech_request(
            &client,
            "https://api.openai.com/v1",
            "sk-bad",
            SpeechRequest::new("test"),
        )
        .await
        .expect_err("expected auth error");

        assert!(matches!(err, BlazenError::Auth { .. }), "got {err:?}");
    }

    #[tokio::test]
    async fn maps_429_to_rate_limit() {
        let client = MockClient::new(HttpResponse {
            status: 429,
            headers: vec![("retry-after".to_owned(), "5".to_owned())],
            body: b"rate limited".to_vec(),
        });

        let err = text_to_speech_request(
            &client,
            "https://api.openai.com/v1",
            "sk-test",
            SpeechRequest::new("test"),
        )
        .await
        .expect_err("expected rate-limit error");

        assert!(matches!(err, BlazenError::RateLimit { .. }), "got {err:?}");
    }
}
