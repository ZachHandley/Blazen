//! Smoke tests for [`blazen_llm_llamacpp_server::LlamacppServerProvider`]
//! against a mocked `llama-server`. Asserts request shapes match the
//! documented `llama.cpp` HTTP API:
//!
//! - `POST /v1/chat/completions` accepts the OpenAI-shaped body and an
//!   adapter alias can be supplied via the `model` field.
//! - `POST /v1/completions` accepts a `{model, prompt}` body.
//! - `POST /v1/embeddings` accepts a `{model, input}` body and returns
//!   `{data: [{embedding: [...]}]}`.
//! - `GET  /v1/models` returns `{data: [{id, object}, ...]}`.
//! - `GET  /health` returns `{status: "ok"}` on a ready server.
//! - `GET  /slots` returns one row per running slot.
//! - `GET  /lora-adapters` returns `[{id, path, scale}, ...]`.
//! - `POST /lora-adapters` accepts a top-level array of `{id, scale}`
//!   toggles.
//! - 404 maps to `LlamacppServerError::NotFound`; 500 maps to
//!   `LlamacppServerError::Http`.
//! - `Authorization: Bearer <key>` propagates when `api_key` is set.

use std::path::PathBuf;

use blazen_llm_llamacpp_server::{
    LlamacppServerAdapterTransport, LlamacppServerError, LlamacppServerOptions,
    LlamacppServerProvider,
};

fn opts_for(server_url: &str) -> LlamacppServerOptions {
    LlamacppServerOptions::required(server_url.to_string(), "llama-3.2")
}

#[tokio::test]
async fn chat_completions_non_streaming_roundtrip() {
    let mut server = mockito::Server::new_async().await;
    let m = server
        .mock("POST", "/v1/chat/completions")
        .match_header("content-type", "application/json")
        .match_body(mockito::Matcher::PartialJson(serde_json::json!({
            "model": "llama-3.2",
            "stream": false,
        })))
        .with_status(200)
        .with_header("content-type", "application/json")
        .with_body(
            serde_json::json!({
                "id": "chatcmpl-xxx",
                "model": "llama-3.2",
                "choices": [{
                    "message": {"role": "assistant", "content": "hello back"},
                    "finish_reason": "stop"
                }],
                "usage": {"prompt_tokens": 6, "completion_tokens": 3, "total_tokens": 9}
            })
            .to_string(),
        )
        .create_async()
        .await;

    let provider = LlamacppServerProvider::from_options(opts_for(&server.url())).unwrap();
    let body = serde_json::json!({
        "model": "llama-3.2",
        "messages": [{"role": "user", "content": "say hi"}],
        "stream": false,
    });
    let resp = provider
        .chat_completions(body)
        .await
        .expect("chat_completions ok");
    m.assert_async().await;
    assert_eq!(resp["choices"][0]["message"]["content"], "hello back");
    assert_eq!(resp["usage"]["total_tokens"], 9);
}

#[tokio::test]
async fn chat_completions_streaming_sse_roundtrip_with_auth_header() {
    let mut server = mockito::Server::new_async().await;
    // OAI-shaped SSE: each frame is `data: <json>\n\n`, terminated by
    // `data: [DONE]\n\n`.
    let sse = [
        "data: {\"choices\":[{\"delta\":{\"content\":\"hel\"}}]}\n\n",
        "data: {\"choices\":[{\"delta\":{\"content\":\"lo\"}}]}\n\n",
        "data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n",
        "data: [DONE]\n\n",
    ]
    .concat();

    let m = server
        .mock("POST", "/v1/chat/completions")
        .match_header("authorization", "Bearer secret-token")
        .match_body(mockito::Matcher::PartialJson(serde_json::json!({
            "stream": true,
        })))
        .with_status(200)
        .with_header("content-type", "text/event-stream")
        .with_body(sse)
        .create_async()
        .await;

    let opts = LlamacppServerOptions {
        api_key: Some("secret-token".into()),
        ..opts_for(&server.url())
    };
    let provider = LlamacppServerProvider::from_options(opts).unwrap();
    let resp = provider
        .chat_completions_stream(serde_json::json!({
            "model": "llama-3.2",
            "messages": [{"role": "user", "content": "say hi"}],
            "stream": true,
        }))
        .await
        .expect("stream open ok");
    m.assert_async().await;

    let body = resp.text().await.expect("drain body");
    assert!(body.contains("[DONE]"));
    assert!(body.contains("hel"));
}

#[tokio::test]
async fn completions_roundtrip() {
    let mut server = mockito::Server::new_async().await;
    let m = server
        .mock("POST", "/v1/completions")
        .match_body(mockito::Matcher::PartialJson(serde_json::json!({
            "model": "llama-3.2",
            "prompt": "Once upon a time",
        })))
        .with_status(200)
        .with_body(
            serde_json::json!({
                "model": "llama-3.2",
                "choices": [{"text": " there was a llama.", "finish_reason": "length"}]
            })
            .to_string(),
        )
        .create_async()
        .await;

    let provider = LlamacppServerProvider::from_options(opts_for(&server.url())).unwrap();
    let resp = provider
        .completions(serde_json::json!({
            "model": "llama-3.2",
            "prompt": "Once upon a time",
            "max_tokens": 16,
        }))
        .await
        .expect("completions ok");

    m.assert_async().await;
    assert_eq!(resp["choices"][0]["text"], " there was a llama.");
}

#[tokio::test]
async fn embeddings_roundtrip() {
    let mut server = mockito::Server::new_async().await;
    let m = server
        .mock("POST", "/v1/embeddings")
        .match_body(mockito::Matcher::PartialJson(serde_json::json!({
            "model": "llama-3.2",
            "input": ["embed me"],
        })))
        .with_status(200)
        .with_body(
            serde_json::json!({
                "model": "llama-3.2",
                "data": [{"embedding": [0.1f32, 0.2, 0.3, 0.4], "index": 0}]
            })
            .to_string(),
        )
        .create_async()
        .await;

    let provider = LlamacppServerProvider::from_options(opts_for(&server.url())).unwrap();
    let resp = provider
        .embeddings(serde_json::json!({
            "model": "llama-3.2",
            "input": ["embed me"],
        }))
        .await
        .expect("embeddings ok");

    m.assert_async().await;
    assert_eq!(resp["data"][0]["embedding"].as_array().unwrap().len(), 4);
}

#[tokio::test]
async fn list_models_returns_loaded_model() {
    let mut server = mockito::Server::new_async().await;
    let m = server
        .mock("GET", "/v1/models")
        .with_status(200)
        .with_header("content-type", "application/json")
        .with_body(
            serde_json::json!({
                "data": [{"id": "llama-3.2", "object": "model"}],
                "object": "list"
            })
            .to_string(),
        )
        .create_async()
        .await;

    let provider = LlamacppServerProvider::from_options(opts_for(&server.url())).unwrap();
    let listing = provider.models().await.expect("models ok");
    m.assert_async().await;
    assert_eq!(listing.len(), 1);
    assert_eq!(listing[0].id, "llama-3.2");
}

#[tokio::test]
async fn health_returns_status_ok() {
    let mut server = mockito::Server::new_async().await;
    let m = server
        .mock("GET", "/health")
        .with_status(200)
        .with_body(r#"{"status":"ok"}"#)
        .create_async()
        .await;

    let provider = LlamacppServerProvider::from_options(opts_for(&server.url())).unwrap();
    let health = provider.health().await.expect("health ok");
    m.assert_async().await;
    assert_eq!(health.status, "ok");
}

#[tokio::test]
async fn slots_returns_one_row_per_slot() {
    let mut server = mockito::Server::new_async().await;
    let m = server
        .mock("GET", "/slots")
        .with_status(200)
        .with_body(
            serde_json::json!([
                {"id": 0, "is_processing": false, "prompt": ""},
                {"id": 1, "is_processing": true, "prompt": "say hi"}
            ])
            .to_string(),
        )
        .create_async()
        .await;

    let provider = LlamacppServerProvider::from_options(opts_for(&server.url())).unwrap();
    let rows = provider.slots().await.expect("slots ok");
    m.assert_async().await;
    assert_eq!(rows.len(), 2);
    assert_eq!(rows[0].id, 0);
    assert_eq!(rows[1].id, 1);
    // Free-form fields ride through `raw`.
    assert_eq!(rows[1].raw["is_processing"], true);
}

#[tokio::test]
async fn list_lora_adapters_returns_preloaded_rows() {
    let mut server = mockito::Server::new_async().await;
    let m = server
        .mock("GET", "/lora-adapters")
        .with_status(200)
        .with_body(
            serde_json::json!([
                {"id": 0, "path": "/srv/loras/sql.gguf", "scale": 0.0},
                {"id": 1, "path": "/srv/loras/code.gguf", "scale": 0.0}
            ])
            .to_string(),
        )
        .create_async()
        .await;

    let provider = LlamacppServerProvider::from_options(opts_for(&server.url())).unwrap();
    let rows = provider
        .client()
        .list_lora_adapters()
        .await
        .expect("list_lora_adapters ok");
    m.assert_async().await;
    assert_eq!(rows.len(), 2);
    assert_eq!(rows[0].id, 0);
    assert_eq!(rows[1].path, "/srv/loras/code.gguf");
}

#[tokio::test]
async fn load_adapter_toggles_active_set_via_post() {
    let mut server = mockito::Server::new_async().await;
    let list = server
        .mock("GET", "/lora-adapters")
        .with_status(200)
        .with_body(
            serde_json::json!([
                {"id": 0, "path": "/srv/loras/sql.gguf", "scale": 0.0},
                {"id": 1, "path": "/srv/loras/code.gguf", "scale": 0.0}
            ])
            .to_string(),
        )
        .create_async()
        .await;
    let toggle = server
        .mock("POST", "/lora-adapters")
        // Top-level array of {id, scale} — the new active set.
        .match_body(mockito::Matcher::Json(serde_json::json!([
            {"id": 0, "scale": 1.0}
        ])))
        .with_status(200)
        .with_body("{}")
        .create_async()
        .await;

    let provider = LlamacppServerProvider::from_options(opts_for(&server.url())).unwrap();
    let mounted = provider
        .load_adapter("sql", &PathBuf::from("/srv/loras/sql.gguf"))
        .await
        .expect("load_adapter ok");

    list.assert_async().await;
    toggle.assert_async().await;
    assert_eq!(mounted.adapter_id, "sql");
    assert_eq!(mounted.server_index, 0);
    let listed = provider.list_adapters().await;
    assert_eq!(listed.len(), 1);
}

#[tokio::test]
async fn load_adapter_missing_preload_surfaces_as_adapter_failed() {
    let mut server = mockito::Server::new_async().await;
    // Only sql.gguf is preloaded — asking for math.gguf must fail BEFORE
    // any POST hits the wire.
    let _list = server
        .mock("GET", "/lora-adapters")
        .with_status(200)
        .with_body(
            serde_json::json!([
                {"id": 0, "path": "/srv/loras/sql.gguf", "scale": 0.0}
            ])
            .to_string(),
        )
        .create_async()
        .await;

    let provider = LlamacppServerProvider::from_options(opts_for(&server.url())).unwrap();
    let err = provider
        .load_adapter("math", &PathBuf::from("/srv/loras/math.gguf"))
        .await
        .expect_err("missing preload must error");
    assert!(matches!(err, LlamacppServerError::AdapterFailed(_)));
    assert!(provider.list_adapters().await.is_empty());
}

#[tokio::test]
async fn load_adapter_rejects_http_push_without_hitting_server() {
    let server = mockito::Server::new_async().await;
    // No mocks registered: the test fails loudly if the provider
    // actually tries to POST or GET anything.
    let opts = LlamacppServerOptions {
        adapter_transport: LlamacppServerAdapterTransport::HttpPush(vec![0, 1, 2]),
        ..opts_for(&server.url())
    };
    let provider = LlamacppServerProvider::from_options(opts).unwrap();
    let err = provider
        .load_adapter("a1", &PathBuf::from("/srv/loras/a1.gguf"))
        .await
        .expect_err("HttpPush must be rejected up-front");
    assert!(matches!(err, LlamacppServerError::Unsupported(_)));
}

#[tokio::test]
async fn load_adapter_rejects_hf_hub_without_hitting_server() {
    let server = mockito::Server::new_async().await;
    let opts = LlamacppServerOptions {
        adapter_transport: LlamacppServerAdapterTransport::HfHub {
            repo: "tloen/alpaca-lora-7b".into(),
            revision: None,
        },
        ..opts_for(&server.url())
    };
    let provider = LlamacppServerProvider::from_options(opts).unwrap();
    let err = provider
        .load_adapter("a1", &PathBuf::from("/unused"))
        .await
        .expect_err("HfHub must be rejected up-front");
    assert!(matches!(err, LlamacppServerError::Unsupported(_)));
}

#[tokio::test]
async fn unload_adapter_posts_empty_active_set_after_single_mount() {
    let mut server = mockito::Server::new_async().await;
    let _list = server
        .mock("GET", "/lora-adapters")
        .with_status(200)
        .with_body(
            serde_json::json!([
                {"id": 0, "path": "/srv/loras/sql.gguf", "scale": 0.0}
            ])
            .to_string(),
        )
        .create_async()
        .await;
    let mount_post = server
        .mock("POST", "/lora-adapters")
        .match_body(mockito::Matcher::Json(serde_json::json!([
            {"id": 0, "scale": 1.0}
        ])))
        .with_status(200)
        .with_body("{}")
        .create_async()
        .await;
    let unmount_post = server
        .mock("POST", "/lora-adapters")
        // After unload, the active set is empty -> empty array.
        .match_body(mockito::Matcher::Json(serde_json::json!([])))
        .with_status(200)
        .with_body("{}")
        .create_async()
        .await;

    let provider = LlamacppServerProvider::from_options(opts_for(&server.url())).unwrap();
    provider
        .load_adapter("sql", &PathBuf::from("/srv/loras/sql.gguf"))
        .await
        .unwrap();
    provider.unload_adapter("sql").await.unwrap();

    mount_post.assert_async().await;
    unmount_post.assert_async().await;
    assert!(provider.list_adapters().await.is_empty());
}

#[tokio::test]
async fn missing_model_404_maps_to_not_found() {
    let mut server = mockito::Server::new_async().await;
    let _m = server
        .mock("POST", "/v1/chat/completions")
        .with_status(404)
        .with_body(r#"{"error":"model 'nope' not found"}"#)
        .create_async()
        .await;

    let provider = LlamacppServerProvider::from_options(opts_for(&server.url())).unwrap();
    let err = provider
        .chat_completions(serde_json::json!({
            "model": "nope",
            "messages": [{"role": "user", "content": "x"}]
        }))
        .await
        .expect_err("404 must surface");
    assert!(matches!(err, LlamacppServerError::NotFound(_)));
}

#[tokio::test]
async fn chat_completions_500_maps_to_http() {
    let mut server = mockito::Server::new_async().await;
    let _m = server
        .mock("POST", "/v1/chat/completions")
        .with_status(500)
        .with_body("kaboom")
        .create_async()
        .await;

    let provider = LlamacppServerProvider::from_options(opts_for(&server.url())).unwrap();
    let err = provider
        .chat_completions(serde_json::json!({
            "model": "llama-3.2",
            "messages": [{"role": "user", "content": "x"}]
        }))
        .await
        .expect_err("500 must surface");
    assert!(matches!(err, LlamacppServerError::Http { status: 500, .. }));
}

#[tokio::test]
async fn chat_completions_malformed_json_maps_to_decode() {
    let mut server = mockito::Server::new_async().await;
    let _m = server
        .mock("POST", "/v1/chat/completions")
        .with_status(200)
        .with_body("not even close to JSON {")
        .create_async()
        .await;

    let provider = LlamacppServerProvider::from_options(opts_for(&server.url())).unwrap();
    let err = provider
        .chat_completions(serde_json::json!({
            "model": "llama-3.2",
            "messages": [{"role": "user", "content": "x"}]
        }))
        .await
        .expect_err("malformed JSON must surface as Decode");
    assert!(matches!(err, LlamacppServerError::Decode(_)));
}

#[tokio::test]
async fn auth_header_omitted_when_no_api_key_set() {
    let mut server = mockito::Server::new_async().await;
    let m = server
        .mock("GET", "/health")
        .match_header("authorization", mockito::Matcher::Missing)
        .with_status(200)
        .with_body(r#"{"status":"ok"}"#)
        .create_async()
        .await;

    let provider = LlamacppServerProvider::from_options(opts_for(&server.url())).unwrap();
    provider.health().await.expect("health ok");
    m.assert_async().await;
}

#[tokio::test]
async fn refresh_adapters_filters_inactive_rows() {
    let mut server = mockito::Server::new_async().await;
    let _m = server
        .mock("GET", "/lora-adapters")
        .with_status(200)
        .with_body(
            serde_json::json!([
                {"id": 0, "path": "/srv/loras/sql.gguf", "scale": 1.0},
                {"id": 1, "path": "/srv/loras/code.gguf", "scale": 0.0},
                {"id": 2, "path": "/srv/loras/math.gguf", "scale": 0.5}
            ])
            .to_string(),
        )
        .create_async()
        .await;

    let provider = LlamacppServerProvider::from_options(opts_for(&server.url())).unwrap();
    let upstream = provider.refresh_adapters_from_server().await.unwrap();
    assert_eq!(upstream.len(), 3);
    let cached = provider.list_adapters().await;
    // Only rows with scale > 0.0 are mirrored locally.
    assert_eq!(cached.len(), 2);
    let ids: Vec<u32> = cached.iter().map(|a| a.server_index).collect();
    assert!(ids.contains(&0));
    assert!(ids.contains(&2));
    assert!(!ids.contains(&1));
}
