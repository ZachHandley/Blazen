//! Smoke tests for [`blazen_llm_tgi::TgiProvider`] against a mocked TGI
//! server. Asserts request shapes match TGI's documented HTTP API:
//!
//! - `POST /generate`               — `{inputs, parameters, adapter_id?}`
//! - `POST /generate_stream`        — SSE token frames, `data: <json>\n\n`
//! - `POST /v1/chat/completions`    — OAI shape; `stream` controlled by helper
//! - `POST /v1/completions`         — OAI legacy shape
//! - `GET  /info`                   — runtime config
//! - `GET  /v1/models`              — base + preloaded adapters
//! - 404 → `TgiError::NotFound`; 422 → `TgiError::Validation`;
//!   500 → `TgiError::Http`.
//! - `Authorization: Bearer <key>` propagates when `api_key` is set.
//! - `adapter_id` is attached to outgoing requests when an adapter is
//!   active.

use std::path::PathBuf;

use blazen_llm_tgi::{TgiAdapterTransport, TgiError, TgiOptions, TgiProvider};

fn opts_for(server_url: &str) -> TgiOptions {
    TgiOptions::required(server_url.to_string(), "meta-llama/Llama-3.2-3B")
}

#[tokio::test]
async fn generate_native_roundtrip() {
    let mut server = mockito::Server::new_async().await;
    let m = server
        .mock("POST", "/generate")
        .match_header("content-type", "application/json")
        .match_body(mockito::Matcher::PartialJson(serde_json::json!({
            "inputs": "say hi",
        })))
        .with_status(200)
        .with_header("content-type", "application/json")
        .with_body(
            serde_json::json!({
                "generated_text": "hello back",
                "details": {
                    "finish_reason": "stop_sequence",
                    "generated_tokens": 3,
                    "prefill": [],
                },
            })
            .to_string(),
        )
        .create_async()
        .await;

    let provider = TgiProvider::from_options(opts_for(&server.url())).unwrap();
    let body = serde_json::json!({
        "inputs": "say hi",
        "parameters": {"max_new_tokens": 16},
    });
    let resp = provider.generate(body).await.expect("generate ok");
    m.assert_async().await;
    assert_eq!(resp["generated_text"], "hello back");
    assert_eq!(resp["details"]["finish_reason"], "stop_sequence");
}

#[tokio::test]
async fn generate_stream_sse_roundtrip() {
    let mut server = mockito::Server::new_async().await;
    // TGI uses SSE for /generate_stream: data: <json>\n\n per frame.
    // The terminal frame carries `details` and `generated_text`.
    let sse = [
        "data: {\"token\":{\"id\":1,\"text\":\"hel\"}}\n\n",
        "data: {\"token\":{\"id\":2,\"text\":\"lo\"}}\n\n",
        "data: {\"token\":{\"id\":3,\"text\":\"\"},\"details\":{\"finish_reason\":\"length\",\"generated_tokens\":3},\"generated_text\":\"hello\"}\n\n",
    ]
    .concat();

    let m = server
        .mock("POST", "/generate_stream")
        .match_body(mockito::Matcher::PartialJson(serde_json::json!({
            "inputs": "say hi",
        })))
        .with_status(200)
        .with_header("content-type", "text/event-stream")
        .with_body(&sse)
        .create_async()
        .await;

    let provider = TgiProvider::from_options(opts_for(&server.url())).unwrap();
    let resp = provider
        .generate_stream(serde_json::json!({
            "inputs": "say hi",
            "parameters": {"max_new_tokens": 16},
        }))
        .await
        .expect("stream open ok");
    m.assert_async().await;

    let body = resp.text().await.expect("drain body");
    // Three SSE events were delivered.
    assert_eq!(body.matches("data:").count(), 3);
    // Terminal frame includes generated_text.
    assert!(body.contains("\"generated_text\":\"hello\""));
}

#[tokio::test]
async fn chat_completions_non_streaming_with_auth_header() {
    let mut server = mockito::Server::new_async().await;
    let m = server
        .mock("POST", "/v1/chat/completions")
        .match_header("authorization", "Bearer secret-token")
        .match_body(mockito::Matcher::PartialJson(serde_json::json!({
            "model": "meta-llama/Llama-3.2-3B",
            "stream": false,
        })))
        .with_status(200)
        .with_body(
            serde_json::json!({
                "id": "chatcmpl-tgi-1",
                "model": "meta-llama/Llama-3.2-3B",
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": "hi there"},
                    "finish_reason": "stop",
                }],
                "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8},
            })
            .to_string(),
        )
        .create_async()
        .await;

    let opts = TgiOptions {
        api_key: Some("secret-token".into()),
        ..opts_for(&server.url())
    };
    let provider = TgiProvider::from_options(opts).unwrap();
    let resp = provider
        .chat(serde_json::json!({
            "model": "meta-llama/Llama-3.2-3B",
            "messages": [{"role": "user", "content": "hi"}],
        }))
        .await
        .expect("chat ok");

    m.assert_async().await;
    assert_eq!(resp["choices"][0]["message"]["content"], "hi there");
    assert_eq!(resp["usage"]["total_tokens"], 8);
}

#[tokio::test]
async fn chat_completions_streaming_sse_roundtrip() {
    let mut server = mockito::Server::new_async().await;
    // OpenAI-shaped SSE: data: <json>\n\n with [DONE] terminator.
    let sse = [
        "data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"hel\"}}]}\n\n",
        "data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"lo\"},\"finish_reason\":\"stop\"}]}\n\n",
        "data: [DONE]\n\n",
    ]
    .concat();

    let m = server
        .mock("POST", "/v1/chat/completions")
        .match_body(mockito::Matcher::PartialJson(serde_json::json!({
            "stream": true,
        })))
        .with_status(200)
        .with_header("content-type", "text/event-stream")
        .with_body(&sse)
        .create_async()
        .await;

    let provider = TgiProvider::from_options(opts_for(&server.url())).unwrap();
    let resp = provider
        .chat_stream(serde_json::json!({
            "model": "meta-llama/Llama-3.2-3B",
            "messages": [{"role": "user", "content": "hi"}],
            // Caller passes stream:false; helper must force true.
            "stream": false,
        }))
        .await
        .expect("chat-stream open ok");
    m.assert_async().await;
    let body = resp.text().await.expect("drain body");
    assert!(body.contains("[DONE]"));
    assert!(body.contains("\"content\":\"hel\""));
}

#[tokio::test]
async fn completions_legacy_roundtrip() {
    let mut server = mockito::Server::new_async().await;
    let m = server
        .mock("POST", "/v1/completions")
        .match_body(mockito::Matcher::PartialJson(serde_json::json!({
            "model": "meta-llama/Llama-3.2-3B",
            "prompt": "say hi",
            "stream": false,
        })))
        .with_status(200)
        .with_body(
            serde_json::json!({
                "id": "cmpl-tgi-1",
                "model": "meta-llama/Llama-3.2-3B",
                "choices": [{"index": 0, "text": " hello", "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 2, "completion_tokens": 2, "total_tokens": 4},
            })
            .to_string(),
        )
        .create_async()
        .await;

    let provider = TgiProvider::from_options(opts_for(&server.url())).unwrap();
    let resp = provider
        .complete(serde_json::json!({
            "model": "meta-llama/Llama-3.2-3B",
            "prompt": "say hi",
            "max_tokens": 16,
        }))
        .await
        .expect("completions ok");

    m.assert_async().await;
    assert_eq!(resp["choices"][0]["text"], " hello");
}

#[tokio::test]
async fn info_returns_runtime_config() {
    let mut server = mockito::Server::new_async().await;
    let m = server
        .mock("GET", "/info")
        .with_status(200)
        .with_header("content-type", "application/json")
        .with_body(
            serde_json::json!({
                "model_id": "meta-llama/Llama-3.2-3B",
                "version": "2.3.1",
                "sha": "deadbeef",
                "model_dtype": "torch.bfloat16",
                "max_input_tokens": 4096,
                "max_total_tokens": 4128,
            })
            .to_string(),
        )
        .create_async()
        .await;

    let provider = TgiProvider::from_options(opts_for(&server.url())).unwrap();
    let info = provider.info().await.expect("info ok");
    m.assert_async().await;
    assert_eq!(info.model_id.as_deref(), Some("meta-llama/Llama-3.2-3B"));
    assert_eq!(info.version.as_deref(), Some("2.3.1"));
    assert_eq!(info.max_input_tokens, Some(4096));
    assert_eq!(info.max_total_tokens, Some(4128));
}

#[tokio::test]
async fn list_models_returns_base_and_adapters() {
    let mut server = mockito::Server::new_async().await;
    let m = server
        .mock("GET", "/v1/models")
        .with_status(200)
        .with_body(
            serde_json::json!({
                "object": "list",
                "data": [
                    {"id": "meta-llama/Llama-3.2-3B", "object": "model"},
                    {"id": "sql-lora", "object": "lora"},
                    {"id": "code-lora", "object": "lora"},
                ],
            })
            .to_string(),
        )
        .create_async()
        .await;

    let provider = TgiProvider::from_options(opts_for(&server.url())).unwrap();
    let listing = provider.list_models().await.expect("list_models ok");
    m.assert_async().await;
    assert_eq!(listing.len(), 3);
    assert_eq!(listing[0].id, "meta-llama/Llama-3.2-3B");
    assert_eq!(listing[1].id, "sql-lora");
    assert_eq!(listing[1].object.as_deref(), Some("lora"));
}

#[tokio::test]
async fn load_adapter_registers_preloaded_id_and_propagates_on_next_request() {
    let mut server = mockito::Server::new_async().await;
    // 1. `/v1/models` returns the preloaded adapter set.
    let models = server
        .mock("GET", "/v1/models")
        .with_status(200)
        .with_body(
            serde_json::json!({
                "object": "list",
                "data": [
                    {"id": "meta-llama/Llama-3.2-3B", "object": "model"},
                    {"id": "sql-lora", "object": "lora"},
                ],
            })
            .to_string(),
        )
        .create_async()
        .await;
    // 2. After load_adapter, the next /generate call must carry
    //    `adapter_id: "sql-lora"`.
    let generate = server
        .mock("POST", "/generate")
        .match_body(mockito::Matcher::PartialJson(serde_json::json!({
            "inputs": "select * from t",
            "adapter_id": "sql-lora",
        })))
        .with_status(200)
        .with_body(
            serde_json::json!({
                "generated_text": "SELECT * FROM t LIMIT 1;",
                "details": {"finish_reason": "stop_sequence", "generated_tokens": 7},
            })
            .to_string(),
        )
        .create_async()
        .await;

    let opts = TgiOptions {
        adapter_transport: TgiAdapterTransport::LocalFs(PathBuf::from("/srv/loras/sql-lora")),
        ..opts_for(&server.url())
    };
    let provider = TgiProvider::from_options(opts).unwrap();
    let active = provider
        .load_adapter("sql-lora", &PathBuf::from("/srv/loras/sql-lora"))
        .await
        .expect("load_adapter ok");
    assert_eq!(active.adapter_id, "sql-lora");
    assert_eq!(
        provider.active_adapter_id().await.as_deref(),
        Some("sql-lora")
    );

    let resp = provider
        .generate(serde_json::json!({
            "inputs": "select * from t",
            "parameters": {"max_new_tokens": 32},
        }))
        .await
        .expect("generate-with-adapter ok");
    assert_eq!(resp["generated_text"], "SELECT * FROM t LIMIT 1;");

    models.assert_async().await;
    generate.assert_async().await;
}

#[tokio::test]
async fn load_adapter_rejects_id_not_preloaded() {
    let mut server = mockito::Server::new_async().await;
    let _m = server
        .mock("GET", "/v1/models")
        .with_status(200)
        .with_body(
            serde_json::json!({
                "object": "list",
                "data": [{"id": "meta-llama/Llama-3.2-3B", "object": "model"}],
            })
            .to_string(),
        )
        .create_async()
        .await;

    let provider = TgiProvider::from_options(opts_for(&server.url())).unwrap();
    let err = provider
        .load_adapter("nope-lora", &PathBuf::from("/unused"))
        .await
        .expect_err("unknown adapter must be rejected");
    assert!(matches!(err, TgiError::NotFound(_)));
    assert!(provider.active_adapter_id().await.is_none());
}

#[tokio::test]
async fn load_adapter_rejects_http_push_without_hitting_server() {
    let server = mockito::Server::new_async().await;
    // No mocks registered: the test fails loudly if the provider actually
    // tries to GET /v1/models or POST anything.
    let opts = TgiOptions {
        adapter_transport: TgiAdapterTransport::HttpPush(vec![0, 1, 2]),
        ..opts_for(&server.url())
    };
    let provider = TgiProvider::from_options(opts).unwrap();
    let err = provider
        .load_adapter("a1", &PathBuf::from("/srv/loras/a1"))
        .await
        .expect_err("HttpPush must be rejected up-front");
    assert!(matches!(err, TgiError::Unsupported(_)));
}

#[tokio::test]
async fn unload_adapter_clears_active_id() {
    let mut server = mockito::Server::new_async().await;
    let _models = server
        .mock("GET", "/v1/models")
        .with_status(200)
        .with_body(
            serde_json::json!({
                "object": "list",
                "data": [
                    {"id": "meta-llama/Llama-3.2-3B", "object": "model"},
                    {"id": "sql-lora", "object": "lora"},
                ],
            })
            .to_string(),
        )
        .create_async()
        .await;

    let provider = TgiProvider::from_options(opts_for(&server.url())).unwrap();
    provider
        .load_adapter("sql-lora", &PathBuf::from("/unused"))
        .await
        .expect("register ok");
    assert!(provider.active_adapter_id().await.is_some());

    provider
        .unload_adapter("sql-lora")
        .await
        .expect("unload ok");
    assert!(provider.active_adapter_id().await.is_none());
    assert!(provider.list_adapters().await.is_empty());
}

#[tokio::test]
async fn missing_endpoint_404_maps_to_not_found() {
    let mut server = mockito::Server::new_async().await;
    let _m = server
        .mock("POST", "/generate")
        .with_status(404)
        .with_body(r#"{"error":"Not Found","error_type":"not_found"}"#)
        .create_async()
        .await;

    let provider = TgiProvider::from_options(opts_for(&server.url())).unwrap();
    let err = provider
        .generate(serde_json::json!({"inputs": "x"}))
        .await
        .expect_err("404 must surface");
    assert!(matches!(err, TgiError::NotFound(_)));
}

#[tokio::test]
async fn validation_422_maps_to_validation_variant() {
    let mut server = mockito::Server::new_async().await;
    let _m = server
        .mock("POST", "/generate")
        .with_status(422)
        .with_body(
            r#"{"error":"Input validation error: `inputs` tokens + `max_new_tokens` must be <= `max_total_tokens`","error_type":"validation"}"#,
        )
        .create_async()
        .await;

    let provider = TgiProvider::from_options(opts_for(&server.url())).unwrap();
    let err = provider
        .generate(serde_json::json!({"inputs": "x", "parameters": {"max_new_tokens": 999_999}}))
        .await
        .expect_err("422 must surface");
    match err {
        TgiError::Validation(body) => assert!(body.contains("max_total_tokens")),
        other => panic!("expected Validation, got {other:?}"),
    }
}

#[tokio::test]
async fn server_500_maps_to_http() {
    let mut server = mockito::Server::new_async().await;
    let _m = server
        .mock("POST", "/v1/chat/completions")
        .with_status(500)
        .with_body("kaboom")
        .create_async()
        .await;

    let provider = TgiProvider::from_options(opts_for(&server.url())).unwrap();
    let err = provider
        .chat(serde_json::json!({
            "model": "x",
            "messages": [{"role": "user", "content": "hi"}],
        }))
        .await
        .expect_err("500 must surface");
    assert!(matches!(err, TgiError::Http { status: 500, .. }));
}

#[tokio::test]
async fn refresh_adapters_filters_out_base_model() {
    let mut server = mockito::Server::new_async().await;
    let _m = server
        .mock("GET", "/v1/models")
        .with_status(200)
        .with_body(
            serde_json::json!({
                "object": "list",
                "data": [
                    {"id": "meta-llama/Llama-3.2-3B", "object": "model"},
                    {"id": "sql-lora", "object": "lora"},
                    {"id": "code-lora", "object": "lora"},
                ],
            })
            .to_string(),
        )
        .create_async()
        .await;

    let provider = TgiProvider::from_options(opts_for(&server.url())).unwrap();
    let upstream = provider.refresh_adapters_from_server().await.unwrap();
    assert_eq!(upstream.len(), 3);
    let cached = provider.list_adapters().await;
    let ids: Vec<&str> = cached.iter().map(|a| a.adapter_id.as_str()).collect();
    assert!(ids.contains(&"sql-lora"));
    assert!(ids.contains(&"code-lora"));
    assert!(!ids.contains(&"meta-llama/Llama-3.2-3B"));
}
