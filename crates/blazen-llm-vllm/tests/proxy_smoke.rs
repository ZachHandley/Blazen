//! Smoke tests for [`blazen_llm_vllm::VllmProvider`] against a mocked
//! vLLM server. Asserts request shapes match vLLM's documented HTTP API:
//!
//! - `POST /v1/load_lora_adapter` with `{lora_name, lora_path}` body.
//! - `POST /v1/unload_lora_adapter` with `{lora_name}` body.
//! - `POST /v1/chat/completions` accepts the OpenAI-shaped body and an
//!   adapter id can be supplied via the `model` field (vLLM's
//!   per-request adapter selection contract).
//! - `GET  /v1/models` returns the base + mounted adapter rows; the
//!   `parent` field identifies adapters.

use std::path::PathBuf;

use blazen_llm_vllm::{VllmAdapterTransport, VllmError, VllmOptions, VllmProvider};

fn opts_for(server_url: &str) -> VllmOptions {
    VllmOptions::required(server_url.to_string(), "meta-llama/Llama-3.2-3B-Instruct")
}

#[tokio::test]
async fn load_lora_adapter_sends_canonical_body_shape() {
    let mut server = mockito::Server::new_async().await;
    let m = server
        .mock("POST", "/v1/load_lora_adapter")
        .match_header("content-type", "application/json")
        .match_body(mockito::Matcher::Json(serde_json::json!({
            "lora_name": "sql-lora",
            "lora_path": "/srv/loras/sql-lora",
        })))
        .with_status(200)
        .with_body("Success: LoRA adapter 'sql-lora' added successfully.")
        .create_async()
        .await;

    let provider = VllmProvider::from_options(opts_for(&server.url())).unwrap();
    provider
        .load_adapter("sql-lora", &PathBuf::from("/srv/loras/sql-lora"))
        .await
        .expect("load_adapter should succeed against mock");

    m.assert_async().await;

    let adapters = provider.list_adapters().await;
    assert_eq!(adapters.len(), 1);
    assert_eq!(adapters[0].adapter_id, "sql-lora");
}

#[tokio::test]
async fn load_lora_adapter_propagates_server_4xx() {
    let mut server = mockito::Server::new_async().await;
    let _m = server
        .mock("POST", "/v1/load_lora_adapter")
        .with_status(400)
        .with_body("VLLM_ALLOW_RUNTIME_LORA_UPDATING is not set")
        .create_async()
        .await;

    let provider = VllmProvider::from_options(opts_for(&server.url())).unwrap();
    let err = provider
        .load_adapter("a1", &PathBuf::from("/srv/loras/a1"))
        .await
        .expect_err("non-2xx must surface as AdapterFailed");
    assert!(matches!(err, VllmError::AdapterFailed(_)));

    // Ensure the failure did NOT add the adapter to the local cache.
    assert!(provider.list_adapters().await.is_empty());
}

#[tokio::test]
async fn unload_lora_adapter_sends_canonical_body_shape() {
    let mut server = mockito::Server::new_async().await;
    let load_mock = server
        .mock("POST", "/v1/load_lora_adapter")
        .with_status(200)
        .with_body("ok")
        .create_async()
        .await;
    let unload_mock = server
        .mock("POST", "/v1/unload_lora_adapter")
        .match_body(mockito::Matcher::Json(serde_json::json!({
            "lora_name": "sql-lora",
        })))
        .with_status(200)
        .with_body("ok")
        .create_async()
        .await;

    let provider = VllmProvider::from_options(opts_for(&server.url())).unwrap();
    provider
        .load_adapter("sql-lora", &PathBuf::from("/srv/loras/sql-lora"))
        .await
        .unwrap();
    provider.unload_adapter("sql-lora").await.unwrap();

    load_mock.assert_async().await;
    unload_mock.assert_async().await;

    assert!(provider.list_adapters().await.is_empty());
}

#[tokio::test]
async fn chat_completions_routes_through_v1_endpoint() {
    let mut server = mockito::Server::new_async().await;
    let body = serde_json::json!({
        "id": "chat-1",
        "model": "sql-lora",
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": "SELECT 1;"},
            "finish_reason": "stop"
        }],
        "usage": {"prompt_tokens": 10, "completion_tokens": 4, "total_tokens": 14}
    });
    let m = server
        .mock("POST", "/v1/chat/completions")
        .match_body(mockito::Matcher::PartialJson(serde_json::json!({
            "model": "sql-lora",
        })))
        .with_status(200)
        .with_header("content-type", "application/json")
        .with_body(body.to_string())
        .create_async()
        .await;

    let provider = VllmProvider::from_options(opts_for(&server.url())).unwrap();
    // Per-request adapter selection: clients put the LoRA name in `model`.
    let req = serde_json::json!({
        "model": "sql-lora",
        "messages": [{"role": "user", "content": "give me a query"}],
        "stream": false,
    });
    let resp = provider.complete(req).await.expect("complete ok");

    m.assert_async().await;
    assert_eq!(resp["choices"][0]["message"]["content"], "SELECT 1;");
    assert_eq!(resp["usage"]["total_tokens"], 14);
}

#[tokio::test]
async fn list_models_surface_distinguishes_adapters_via_parent_field() {
    let mut server = mockito::Server::new_async().await;
    let m = server
        .mock("GET", "/v1/models")
        .with_status(200)
        .with_header("content-type", "application/json")
        .with_body(
            serde_json::json!({
                "object": "list",
                "data": [
                    {"id": "meta-llama/Llama-3.2-3B-Instruct", "object": "model"},
                    {
                        "id": "sql-lora",
                        "object": "model",
                        "parent": "meta-llama/Llama-3.2-3B-Instruct"
                    },
                    {
                        "id": "code-lora",
                        "object": "model",
                        "parent": "meta-llama/Llama-3.2-3B-Instruct"
                    }
                ]
            })
            .to_string(),
        )
        .create_async()
        .await;

    let provider = VllmProvider::from_options(opts_for(&server.url())).unwrap();
    let upstream = provider.refresh_adapters_from_server().await.unwrap();
    m.assert_async().await;

    assert_eq!(upstream.len(), 3);
    let cached = provider.list_adapters().await;
    let ids: Vec<&str> = cached.iter().map(|a| a.adapter_id.as_str()).collect();
    assert!(ids.contains(&"sql-lora"));
    assert!(ids.contains(&"code-lora"));
    assert!(!ids.contains(&"meta-llama/Llama-3.2-3B-Instruct"));
}

#[tokio::test]
async fn http_push_transport_returns_unsupported_without_hitting_server() {
    let server = mockito::Server::new_async().await;
    // No mocks registered: the test fails loudly if the provider actually
    // tries to POST anything.
    let opts = VllmOptions {
        adapter_transport: VllmAdapterTransport::HttpPush(vec![0, 1, 2]),
        ..opts_for(&server.url())
    };
    let provider = VllmProvider::from_options(opts).unwrap();
    let err = provider
        .load_adapter("a1", &PathBuf::from("/srv/loras/a1"))
        .await
        .expect_err("HttpPush must be rejected up-front");
    assert!(matches!(err, VllmError::UnsupportedTransport(_)));
}

#[tokio::test]
async fn runtime_lora_disabled_short_circuits_before_http() {
    let server = mockito::Server::new_async().await;
    let opts = VllmOptions {
        runtime_lora_updating: false,
        ..opts_for(&server.url())
    };
    let provider = VllmProvider::from_options(opts).unwrap();
    let err = provider
        .load_adapter("a1", &PathBuf::from("/srv/loras/a1"))
        .await
        .expect_err("must short-circuit with AdapterFailed");
    assert!(matches!(err, VllmError::AdapterFailed(_)));
}

#[tokio::test]
async fn hf_hub_transport_translates_to_repo_at_revision_path() {
    let mut server = mockito::Server::new_async().await;
    let m = server
        .mock("POST", "/v1/load_lora_adapter")
        .match_body(mockito::Matcher::Json(serde_json::json!({
            "lora_name": "alpaca",
            "lora_path": "tloen/alpaca-lora-7b@v1.0",
        })))
        .with_status(200)
        .with_body("ok")
        .create_async()
        .await;

    let opts = VllmOptions {
        adapter_transport: VllmAdapterTransport::HfHub {
            repo: "tloen/alpaca-lora-7b".into(),
            revision: Some("v1.0".into()),
        },
        ..opts_for(&server.url())
    };
    let provider = VllmProvider::from_options(opts).unwrap();
    // The `path_or_dir` argument is ignored when transport != LocalFs
    // with a populated path, so we pass any value here.
    provider
        .load_adapter("alpaca", &PathBuf::from("/unused"))
        .await
        .unwrap();
    m.assert_async().await;
}
