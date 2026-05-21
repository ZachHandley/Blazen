//! Smoke tests for [`blazen_llm_lmstudio::LmStudioProvider`] against a
//! mocked LM Studio server. Asserts request shapes match LM Studio's
//! documented HTTP API:
//!
//! - `POST /v1/chat/completions` — OAI-shaped body, streaming via SSE.
//! - `POST /v1/completions`      — OAI legacy single-prompt.
//! - `POST /v1/embeddings`       — OAI-shaped embeddings.
//! - `GET  /v1/models`           — OAI-shaped listing.
//! - `GET  /api/v0/models`       — LM-Studio-native listing (with state).
//! - `POST /api/v0/models/load`  — model load.
//! - `POST /api/v0/models/unload`— model unload.
//! - 404 maps to `LmStudioError::NotFound`; 409/422 to `NoModelLoaded`;
//!   500 to `LmStudioError::Http`.
//! - `Authorization: Bearer <key>` propagates when `api_key` is set.
//! - `load_adapter(...)` returns `LmStudioError::Unsupported` without
//!   contacting the server.

use std::path::PathBuf;

use blazen_llm_lmstudio::{
    LmStudioAdapterTransport, LmStudioError, LmStudioNativeModelState, LmStudioOptions,
    LmStudioProvider,
};

fn opts_for(server_url: &str) -> LmStudioOptions {
    LmStudioOptions::required(server_url.to_string(), "qwen2.5-7b-instruct-q4_k_m")
}

#[tokio::test]
async fn chat_completions_non_streaming_roundtrip() {
    let mut server = mockito::Server::new_async().await;
    let body = serde_json::json!({
        "id": "chat-1",
        "model": "qwen2.5-7b-instruct-q4_k_m",
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": "hello back"},
            "finish_reason": "stop"
        }],
        "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7}
    });
    let m = server
        .mock("POST", "/v1/chat/completions")
        .match_header("content-type", "application/json")
        .match_body(mockito::Matcher::PartialJson(serde_json::json!({
            "model": "qwen2.5-7b-instruct-q4_k_m",
            "stream": false,
        })))
        .with_status(200)
        .with_header("content-type", "application/json")
        .with_body(body.to_string())
        .create_async()
        .await;

    let provider = LmStudioProvider::from_options(opts_for(&server.url())).unwrap();
    // Caller passes stream:true and the client must force it to false.
    let req = serde_json::json!({
        "model": "qwen2.5-7b-instruct-q4_k_m",
        "messages": [{"role": "user", "content": "say hi"}],
        "stream": true,
    });
    let resp = provider.complete(req).await.expect("complete ok");
    m.assert_async().await;
    assert_eq!(resp["choices"][0]["message"]["content"], "hello back");
    assert_eq!(resp["usage"]["total_tokens"], 7);
}

#[tokio::test]
async fn chat_completions_streaming_sse_roundtrip() {
    let mut server = mockito::Server::new_async().await;
    // OpenAI-style SSE: each event prefixed with `data: ` and terminated
    // by a blank line, with `data: [DONE]` as the terminator.
    let sse = [
        r#"data: {"choices":[{"delta":{"content":"hel"},"finish_reason":null}]}"#,
        "",
        r#"data: {"choices":[{"delta":{"content":"lo"},"finish_reason":null}]}"#,
        "",
        r#"data: {"choices":[{"delta":{"content":""},"finish_reason":"stop"}]}"#,
        "",
        "data: [DONE]",
        "",
    ]
    .join("\n");

    let m = server
        .mock("POST", "/v1/chat/completions")
        .match_body(mockito::Matcher::PartialJson(serde_json::json!({
            "stream": true,
        })))
        .with_status(200)
        .with_header("content-type", "text/event-stream")
        .with_body(sse)
        .create_async()
        .await;

    let provider = LmStudioProvider::from_options(opts_for(&server.url())).unwrap();
    // Caller passes stream:false; helper must force true.
    let response = provider
        .stream(serde_json::json!({
            "model": "qwen2.5-7b-instruct-q4_k_m",
            "messages": [{"role": "user", "content": "say hi"}],
            "stream": false,
        }))
        .await
        .expect("stream ok");
    m.assert_async().await;
    let text = response.text().await.expect("read body");
    assert!(text.contains("[DONE]"));
    assert!(text.contains("\"hel\""));
}

#[tokio::test]
async fn legacy_completions_routes_through_v1_endpoint() {
    let mut server = mockito::Server::new_async().await;
    let m = server
        .mock("POST", "/v1/completions")
        .match_body(mockito::Matcher::PartialJson(serde_json::json!({
            "model": "qwen2.5-7b-instruct-q4_k_m",
            "prompt": "ping",
            "stream": false,
        })))
        .with_status(200)
        .with_header("content-type", "application/json")
        .with_body(
            serde_json::json!({
                "id": "cmpl-1",
                "model": "qwen2.5-7b-instruct-q4_k_m",
                "choices": [{"text": "pong", "finish_reason": "stop"}],
            })
            .to_string(),
        )
        .create_async()
        .await;

    let provider = LmStudioProvider::from_options(opts_for(&server.url())).unwrap();
    let req = serde_json::json!({
        "model": "qwen2.5-7b-instruct-q4_k_m",
        "prompt": "ping",
        // Caller passes stream:true; helper must force false.
        "stream": true,
    });
    let resp = provider.completions(req).await.expect("completions ok");
    m.assert_async().await;
    assert_eq!(resp["choices"][0]["text"], "pong");
}

#[tokio::test]
async fn embeddings_routes_through_v1_endpoint() {
    let mut server = mockito::Server::new_async().await;
    let m = server
        .mock("POST", "/v1/embeddings")
        .match_body(mockito::Matcher::PartialJson(serde_json::json!({
            "model": "nomic-embed-text-v1.5",
            "input": "hello",
        })))
        .with_status(200)
        .with_header("content-type", "application/json")
        .with_body(
            serde_json::json!({
                "object": "list",
                "model": "nomic-embed-text-v1.5",
                "data": [{
                    "object": "embedding",
                    "index": 0,
                    "embedding": [0.1, 0.2, 0.3],
                }],
                "usage": {"prompt_tokens": 1, "total_tokens": 1},
            })
            .to_string(),
        )
        .create_async()
        .await;

    let provider = LmStudioProvider::from_options(opts_for(&server.url())).unwrap();
    let req = serde_json::json!({
        "model": "nomic-embed-text-v1.5",
        "input": "hello",
    });
    let resp = provider.embed(req).await.expect("embed ok");
    m.assert_async().await;
    assert_eq!(resp["data"][0]["embedding"][0], 0.1);
}

#[tokio::test]
async fn list_models_v1_returns_id_only_rows() {
    let mut server = mockito::Server::new_async().await;
    let m = server
        .mock("GET", "/v1/models")
        .with_status(200)
        .with_header("content-type", "application/json")
        .with_body(
            serde_json::json!({
                "object": "list",
                "data": [
                    {"id": "qwen2.5-7b-instruct-q4_k_m", "object": "model"},
                    {"id": "nomic-embed-text-v1.5", "object": "model"},
                ]
            })
            .to_string(),
        )
        .create_async()
        .await;

    let provider = LmStudioProvider::from_options(opts_for(&server.url())).unwrap();
    let rows = provider.list_models().await.expect("list_models ok");
    m.assert_async().await;
    assert_eq!(rows.len(), 2);
    let ids: Vec<&str> = rows.iter().map(|r| r.id.as_str()).collect();
    assert!(ids.contains(&"qwen2.5-7b-instruct-q4_k_m"));
}

#[tokio::test]
async fn native_models_surface_carries_load_state() {
    let mut server = mockito::Server::new_async().await;
    let m = server
        .mock("GET", "/api/v0/models")
        .with_status(200)
        .with_header("content-type", "application/json")
        .with_body(
            serde_json::json!({
                "object": "list",
                "data": [
                    {
                        "id": "qwen2.5-7b-instruct-q4_k_m",
                        "object": "model",
                        "type": "llm",
                        "arch": "qwen2",
                        "state": "loaded",
                    },
                    {
                        "id": "llama-3.2-3b-instruct-q4_k_m",
                        "object": "model",
                        "type": "llm",
                        "arch": "llama",
                        "state": "not-loaded",
                    },
                ]
            })
            .to_string(),
        )
        .create_async()
        .await;

    let provider = LmStudioProvider::from_options(opts_for(&server.url())).unwrap();
    let rows = provider.native_models().await.expect("native_models ok");
    m.assert_async().await;
    assert_eq!(rows.len(), 2);
    let qwen = rows.iter().find(|r| r.id.contains("qwen")).unwrap();
    assert_eq!(qwen.state, LmStudioNativeModelState::Loaded);
    assert_eq!(qwen.r#type.as_deref(), Some("llm"));
    assert_eq!(qwen.arch.as_deref(), Some("qwen2"));
    let llama = rows.iter().find(|r| r.id.contains("llama")).unwrap();
    assert_eq!(llama.state, LmStudioNativeModelState::NotLoaded);
}

#[tokio::test]
async fn model_load_then_unload_roundtrip() {
    let mut server = mockito::Server::new_async().await;
    let load = server
        .mock("POST", "/api/v0/models/load")
        .match_body(mockito::Matcher::Json(serde_json::json!({
            "model": "qwen2.5-7b-instruct-q4_k_m",
        })))
        .with_status(200)
        .with_body(r#"{"status":"ok"}"#)
        .create_async()
        .await;
    let unload = server
        .mock("POST", "/api/v0/models/unload")
        .match_body(mockito::Matcher::Json(serde_json::json!({
            "model": "qwen2.5-7b-instruct-q4_k_m",
        })))
        .with_status(200)
        .with_body(r#"{"status":"ok"}"#)
        .create_async()
        .await;

    let provider = LmStudioProvider::from_options(opts_for(&server.url())).unwrap();
    provider
        .load_model("qwen2.5-7b-instruct-q4_k_m")
        .await
        .expect("load ok");
    provider
        .unload_model("qwen2.5-7b-instruct-q4_k_m")
        .await
        .expect("unload ok");
    load.assert_async().await;
    unload.assert_async().await;
}

#[tokio::test]
async fn load_model_propagates_404_as_not_found() {
    let mut server = mockito::Server::new_async().await;
    let _m = server
        .mock("POST", "/api/v0/models/load")
        .with_status(404)
        .with_body(r#"{"error":"model not installed"}"#)
        .create_async()
        .await;

    let provider = LmStudioProvider::from_options(opts_for(&server.url())).unwrap();
    let err = provider
        .load_model("nonexistent")
        .await
        .expect_err("404 must surface as NotFound");
    assert!(matches!(err, LmStudioError::NotFound(_)));
}

#[tokio::test]
async fn load_model_propagates_500_as_load_failed() {
    let mut server = mockito::Server::new_async().await;
    let _m = server
        .mock("POST", "/api/v0/models/load")
        .with_status(500)
        .with_body(r#"{"error":"out of memory"}"#)
        .create_async()
        .await;

    let provider = LmStudioProvider::from_options(opts_for(&server.url())).unwrap();
    let err = provider
        .load_model("qwen2.5-7b-instruct-q4_k_m")
        .await
        .expect_err("500 must surface as LoadFailed");
    assert!(matches!(err, LmStudioError::LoadFailed(_)));
}

#[tokio::test]
async fn chat_completions_409_surfaces_as_no_model_loaded() {
    let mut server = mockito::Server::new_async().await;
    let _m = server
        .mock("POST", "/v1/chat/completions")
        .with_status(409)
        .with_body(r#"{"error":"no model loaded"}"#)
        .create_async()
        .await;

    let provider = LmStudioProvider::from_options(opts_for(&server.url())).unwrap();
    let err = provider
        .complete(serde_json::json!({
            "model": "qwen2.5-7b-instruct-q4_k_m",
            "messages": [{"role": "user", "content": "hi"}],
        }))
        .await
        .expect_err("409 must surface as NoModelLoaded");
    assert!(matches!(err, LmStudioError::NoModelLoaded(_)));
}

#[tokio::test]
async fn load_adapter_returns_unsupported_without_hitting_server() {
    // No mocks registered: the test fails loudly if the provider actually
    // tries to POST anything.
    let server = mockito::Server::new_async().await;
    let provider = LmStudioProvider::from_options(opts_for(&server.url())).unwrap();
    let err = provider
        .load_adapter("sql-lora", &PathBuf::from("/srv/loras/sql-lora"))
        .await
        .expect_err("LM Studio runtime LoRA is not supported");
    assert!(matches!(err, LmStudioError::Unsupported(_)));
    let msg = err.to_string();
    // Caller should see actionable guidance, not a generic refusal.
    assert!(msg.to_lowercase().contains("gguf"));
    assert!(msg.contains("merge_lora_into_base") || msg.contains("merge"));
}

#[tokio::test]
async fn load_adapter_unsupported_for_hf_hub_transport() {
    let server = mockito::Server::new_async().await;
    let opts = LmStudioOptions {
        adapter_transport: LmStudioAdapterTransport::HfHub {
            repo: "tloen/alpaca-lora-7b".into(),
            revision: Some("v1.0".into()),
        },
        ..opts_for(&server.url())
    };
    let provider = LmStudioProvider::from_options(opts).unwrap();
    let err = provider
        .load_adapter("alpaca", &PathBuf::from("/unused"))
        .await
        .expect_err("HfHub variant also unsupported");
    assert!(matches!(err, LmStudioError::Unsupported(_)));
}

#[tokio::test]
async fn bearer_auth_propagates_when_api_key_set() {
    let mut server = mockito::Server::new_async().await;
    let m = server
        .mock("GET", "/v1/models")
        .match_header("authorization", "Bearer sk-test-key")
        .with_status(200)
        .with_header("content-type", "application/json")
        .with_body(r#"{"object":"list","data":[]}"#)
        .create_async()
        .await;

    let opts = LmStudioOptions {
        api_key: Some("sk-test-key".into()),
        ..opts_for(&server.url())
    };
    let provider = LmStudioProvider::from_options(opts).unwrap();
    let _rows = provider.list_models().await.expect("ok");
    m.assert_async().await;
}

#[tokio::test]
async fn is_model_loaded_returns_true_when_state_is_loaded() {
    let mut server = mockito::Server::new_async().await;
    let _m = server
        .mock("GET", "/api/v0/models")
        .with_status(200)
        .with_header("content-type", "application/json")
        .with_body(
            serde_json::json!({
                "object": "list",
                "data": [{
                    "id": "qwen2.5-7b-instruct-q4_k_m",
                    "object": "model",
                    "type": "llm",
                    "arch": "qwen2",
                    "state": "loaded",
                }]
            })
            .to_string(),
        )
        .create_async()
        .await;

    let provider = LmStudioProvider::from_options(opts_for(&server.url())).unwrap();
    let loaded = provider.is_model_loaded().await.expect("ok");
    assert!(loaded);
}
