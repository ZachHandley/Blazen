//! Integration smoke tests for `OpenAiTtsBackend` against a `mockito`
//! HTTP server.
//!
//! These tests verify the wire-level request shape (URL, method,
//! headers, body) against the `OpenAI` spec for `/v1/audio/speech` and
//! the de-facto `/v1/voices/*` extension surface. They do not exercise
//! a real `OpenAI` account.

#![cfg(feature = "openai")]

use base64::Engine as _;
use blazen_audio_tts::{OpenAiTtsBackend, OpenAiTtsConfig, OpenAiTtsSpeechRequest, TtsError};
use mockito::Matcher;

fn backend_for(server: &mockito::ServerGuard, api_key: &str) -> OpenAiTtsBackend {
    let base_url = format!("{}/v1", server.url());
    OpenAiTtsBackend::new(OpenAiTtsConfig::new(base_url, api_key)).expect("backend should build")
}

// ---------------------------------------------------------------------------
// /v1/audio/speech
// ---------------------------------------------------------------------------

#[tokio::test]
async fn text_to_speech_posts_openai_spec_body() {
    let mut server = mockito::Server::new_async().await;
    let mock = server
        .mock("POST", "/v1/audio/speech")
        .match_header("authorization", "Bearer sk-test")
        .match_header("content-type", Matcher::Regex("application/json".into()))
        .match_body(Matcher::JsonString(
            r#"{"model":"tts-1","input":"hello","voice":"alloy","response_format":"mp3"}"#
                .to_owned(),
        ))
        .with_status(200)
        .with_header("content-type", "audio/mpeg")
        .with_body(b"fake-mp3-bytes")
        .create_async()
        .await;

    let backend = backend_for(&server, "sk-test");
    let response = backend
        .text_to_speech(OpenAiTtsSpeechRequest {
            text: "hello".to_owned(),
            ..OpenAiTtsSpeechRequest::default()
        })
        .await
        .expect("synthesis should succeed");

    mock.assert_async().await;
    assert_eq!(response.audio_bytes, b"fake-mp3-bytes");
    assert_eq!(response.content_type, "audio/mpeg");
}

#[tokio::test]
async fn text_to_speech_serializes_speed_and_voice_overrides() {
    let mut server = mockito::Server::new_async().await;
    let mock = server
        .mock("POST", "/v1/audio/speech")
        .match_body(Matcher::JsonString(
            r#"{"model":"tts-1-hd","input":"go","voice":"nova","response_format":"wav","speed":1.5}"#
                .to_owned(),
        ))
        .with_status(200)
        .with_header("content-type", "audio/wav")
        .with_body(b"WAVE")
        .create_async()
        .await;

    let backend = backend_for(&server, "sk-test");
    let response = backend
        .text_to_speech(OpenAiTtsSpeechRequest {
            text: "go".to_owned(),
            voice: Some("nova".to_owned()),
            model: Some("tts-1-hd".to_owned()),
            speed: Some(1.5),
            response_format: Some("wav".to_owned()),
            stream: None,
        })
        .await
        .expect("synthesis should succeed");

    mock.assert_async().await;
    assert_eq!(response.content_type, "audio/wav");
}

#[tokio::test]
async fn text_to_speech_omits_authorization_when_api_key_empty() {
    let mut server = mockito::Server::new_async().await;
    let mock = server
        .mock("POST", "/v1/audio/speech")
        .match_header("authorization", Matcher::Missing)
        .with_status(200)
        .with_header("content-type", "audio/mpeg")
        .with_body(b"x")
        .create_async()
        .await;

    let backend = backend_for(&server, "");
    backend
        .text_to_speech(OpenAiTtsSpeechRequest {
            text: "no auth".to_owned(),
            ..OpenAiTtsSpeechRequest::default()
        })
        .await
        .expect("synthesis should succeed");
    mock.assert_async().await;
}

#[tokio::test]
async fn text_to_speech_maps_401_to_auth_error() {
    let mut server = mockito::Server::new_async().await;
    let _mock = server
        .mock("POST", "/v1/audio/speech")
        .with_status(401)
        .with_body(r#"{"error":{"message":"bad key"}}"#)
        .create_async()
        .await;

    let backend = backend_for(&server, "sk-bad");
    let err = backend
        .text_to_speech(OpenAiTtsSpeechRequest {
            text: "hi".to_owned(),
            ..OpenAiTtsSpeechRequest::default()
        })
        .await
        .expect_err("expected auth error");
    match err {
        TtsError::Auth(msg) => assert_eq!(msg, "bad key"),
        other => panic!("expected Auth, got {other:?}"),
    }
}

#[tokio::test]
async fn text_to_speech_maps_429_to_rate_limit_with_retry_after() {
    let mut server = mockito::Server::new_async().await;
    let _mock = server
        .mock("POST", "/v1/audio/speech")
        .with_status(429)
        .with_header("retry-after", "7")
        .with_body(r#"{"error":{"message":"slow down"}}"#)
        .create_async()
        .await;

    let backend = backend_for(&server, "sk-test");
    let err = backend
        .text_to_speech(OpenAiTtsSpeechRequest {
            text: "hi".to_owned(),
            ..OpenAiTtsSpeechRequest::default()
        })
        .await
        .expect_err("expected rate-limit error");
    match err {
        TtsError::RateLimit {
            retry_after_seconds,
            message,
        } => {
            assert_eq!(retry_after_seconds, Some(7));
            assert_eq!(message, "slow down");
        }
        other => panic!("expected RateLimit, got {other:?}"),
    }
}

#[tokio::test]
async fn text_to_speech_maps_500_to_server_error() {
    let mut server = mockito::Server::new_async().await;
    let _mock = server
        .mock("POST", "/v1/audio/speech")
        .with_status(500)
        .with_body("upstream blew up")
        .create_async()
        .await;

    let backend = backend_for(&server, "sk-test");
    let err = backend
        .text_to_speech(OpenAiTtsSpeechRequest {
            text: "hi".to_owned(),
            ..OpenAiTtsSpeechRequest::default()
        })
        .await
        .expect_err("expected server error");
    match err {
        TtsError::ServerError {
            status, message, ..
        } => {
            assert_eq!(status, 500);
            assert!(message.contains("upstream"));
        }
        other => panic!("expected ServerError, got {other:?}"),
    }
}

// ---------------------------------------------------------------------------
// /v1/voices (list)
// ---------------------------------------------------------------------------

#[tokio::test]
async fn list_voices_parses_voices_array() {
    let mut server = mockito::Server::new_async().await;
    let mock = server
        .mock("GET", "/v1/voices")
        .match_header("authorization", "Bearer sk-test")
        .with_status(200)
        .with_header("content-type", "application/json")
        .with_body(r#"{"voices":[{"id":"alloy","name":"Alloy"},{"id":"nova","name":"Nova","language":"en"}]}"#)
        .create_async()
        .await;

    let backend = backend_for(&server, "sk-test");
    let voices = backend
        .list_voices_raw()
        .await
        .expect("list should succeed");
    mock.assert_async().await;
    assert_eq!(voices.len(), 2);
    assert_eq!(voices[0].id, "alloy");
    assert_eq!(voices[1].language.as_deref(), Some("en"));
}

// ---------------------------------------------------------------------------
// /v1/voices/clone (multipart)
// ---------------------------------------------------------------------------

#[tokio::test]
async fn clone_voice_posts_multipart_with_name_and_audio() {
    let mut server = mockito::Server::new_async().await;
    let mock = server
        .mock("POST", "/v1/voices/clone")
        .match_header("authorization", "Bearer sk-test")
        .match_header(
            "content-type",
            Matcher::Regex("^multipart/form-data".into()),
        )
        .match_body(Matcher::AllOf(vec![
            Matcher::Regex(r#"name="name""#.into()),
            Matcher::Regex(r#"name="audio""#.into()),
            Matcher::Regex(r#"name="transcript""#.into()),
        ]))
        .with_status(200)
        .with_header("content-type", "application/json")
        .with_body(r#"{"id":"clone-123","name":"Clone","language":"en"}"#)
        .create_async()
        .await;

    let backend = backend_for(&server, "sk-test");
    let voice = backend
        .clone_voice_raw(
            b"FAKE-AUDIO".to_vec(),
            "Clone",
            Some("the spoken transcript"),
        )
        .await
        .expect("clone should succeed");
    mock.assert_async().await;
    assert_eq!(voice.id, "clone-123");
    assert_eq!(voice.language.as_deref(), Some("en"));
}

#[tokio::test]
async fn clone_voice_accepts_wrapped_response_shape() {
    let mut server = mockito::Server::new_async().await;
    let _mock = server
        .mock("POST", "/v1/voices/clone")
        .with_status(200)
        .with_header("content-type", "application/json")
        .with_body(r#"{"voice":{"id":"wrapped-9","name":"Wrapped"}}"#)
        .create_async()
        .await;

    let backend = backend_for(&server, "sk-test");
    let voice = backend
        .clone_voice_raw(b"x".to_vec(), "Wrapped", None)
        .await
        .expect("clone should succeed");
    assert_eq!(voice.id, "wrapped-9");
    assert_eq!(voice.name, "Wrapped");
}

// ---------------------------------------------------------------------------
// /v1/voices/design
// ---------------------------------------------------------------------------

#[tokio::test]
async fn design_voice_posts_name_and_description() {
    let mut server = mockito::Server::new_async().await;
    let mock = server
        .mock("POST", "/v1/voices/design")
        .match_body(Matcher::JsonString(
            r#"{"name":"Narrator","description":"warm female narrator, mid-30s"}"#.to_owned(),
        ))
        .with_status(200)
        .with_header("content-type", "application/json")
        .with_body(r#"{"id":"designed-42","name":"Narrator"}"#)
        .create_async()
        .await;

    let backend = backend_for(&server, "sk-test");
    let voice = backend
        .design_voice_raw("Narrator", "warm female narrator, mid-30s")
        .await
        .expect("design should succeed");
    mock.assert_async().await;
    assert_eq!(voice.id, "designed-42");
    assert_eq!(voice.name, "Narrator");
}

// ---------------------------------------------------------------------------
// /v1/voices/{id} (delete)
// ---------------------------------------------------------------------------

#[tokio::test]
async fn delete_voice_calls_delete_endpoint_with_id() {
    let mut server = mockito::Server::new_async().await;
    let mock = server
        .mock("DELETE", "/v1/voices/abc-123")
        .match_header("authorization", "Bearer sk-test")
        .with_status(204)
        .create_async()
        .await;

    let backend = backend_for(&server, "sk-test");
    backend
        .delete_voice("abc-123")
        .await
        .expect("delete should succeed");
    mock.assert_async().await;
}

// ---------------------------------------------------------------------------
// Round-trip sanity: ensure the encoded bytes survive a base64 round-trip
// ---------------------------------------------------------------------------

#[tokio::test]
async fn audio_bytes_round_trip_through_base64() {
    let mut server = mockito::Server::new_async().await;
    let payload = b"binary\0\xffaudio\x01bytes".to_vec();
    let _mock = server
        .mock("POST", "/v1/audio/speech")
        .with_status(200)
        .with_header("content-type", "audio/mpeg")
        .with_body(payload.clone())
        .create_async()
        .await;

    let backend = backend_for(&server, "sk-test");
    let response = backend
        .text_to_speech(OpenAiTtsSpeechRequest {
            text: "x".to_owned(),
            ..OpenAiTtsSpeechRequest::default()
        })
        .await
        .unwrap();

    let decoded = base64::engine::general_purpose::STANDARD
        .decode(response.audio_base64())
        .unwrap();
    assert_eq!(decoded, payload);
}
