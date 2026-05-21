//! Smoke tests for [`blazen_llm_ollama::OllamaProvider`] against a
//! mocked Ollama server. Asserts request shapes match Ollama's
//! documented HTTP API:
//!
//! - `POST /api/generate` accepts `{model, prompt}` with `stream` forced
//!   by the helper to whichever mode is being used.
//! - `POST /api/chat` accepts `{model, messages}` with the same
//!   stream-flag semantics.
//! - `POST /api/embeddings` accepts `{model, prompt}` (Ollama 0.1.x) or
//!   `{model, input}` (Ollama 0.1.40+).
//! - `GET  /api/tags` returns `{models: [...]}`.
//! - `POST /api/pull` streams `{status, digest, total, completed}` NDJSON
//!   frames; `{status: "success"}` is terminal.
//! - `POST /api/create` accepts a Modelfile body and streams `{status}`
//!   frames; `{status: "error", ...}` is failure.
//! - 404 maps to `OllamaError::NotFound`; 500 maps to `OllamaError::Http`.
//! - `Authorization: Bearer <key>` propagates when `api_key` is set.

use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use blazen_llm_ollama::{
    OllamaAdapterTransport, OllamaError, OllamaOptions, OllamaProvider, OllamaPullProgress,
};

fn opts_for(server_url: &str) -> OllamaOptions {
    OllamaOptions::required(server_url.to_string(), "llama3.2")
}

#[tokio::test]
async fn generate_non_streaming_roundtrip() {
    let mut server = mockito::Server::new_async().await;
    let m = server
        .mock("POST", "/api/generate")
        .match_header("content-type", "application/json")
        .match_body(mockito::Matcher::PartialJson(serde_json::json!({
            "model": "llama3.2",
            "stream": false,
        })))
        .with_status(200)
        .with_header("content-type", "application/json")
        .with_body(
            serde_json::json!({
                "model": "llama3.2",
                "response": "hello back",
                "done": true,
                "done_reason": "stop",
                "prompt_eval_count": 6,
                "eval_count": 3,
            })
            .to_string(),
        )
        .create_async()
        .await;

    let provider = OllamaProvider::from_options(opts_for(&server.url())).unwrap();
    let body = serde_json::json!({
        "model": "llama3.2",
        "prompt": "say hi",
        // Caller passes stream:true and the client must force it to false.
        "stream": true,
    });
    let resp = provider.generate(body).await.expect("generate ok");
    m.assert_async().await;
    assert_eq!(resp["response"], "hello back");
    assert_eq!(resp["done"], true);
    assert_eq!(resp["eval_count"], 3);
}

#[tokio::test]
async fn generate_streaming_ndjson_roundtrip() {
    let mut server = mockito::Server::new_async().await;
    // Ollama streams NDJSON: each frame is one JSON object terminated
    // by a newline. The terminal frame sets done:true.
    let ndjson = [
        r#"{"model":"llama3.2","response":"hel","done":false}"#,
        r#"{"model":"llama3.2","response":"lo","done":false}"#,
        r#"{"model":"llama3.2","response":"","done":true,"done_reason":"stop","eval_count":2}"#,
    ]
    .join("\n");

    let m = server
        .mock("POST", "/api/generate")
        .match_body(mockito::Matcher::PartialJson(serde_json::json!({
            "stream": true,
        })))
        .with_status(200)
        .with_header("content-type", "application/x-ndjson")
        .with_body(ndjson)
        .create_async()
        .await;

    let provider = OllamaProvider::from_options(opts_for(&server.url())).unwrap();
    let resp = provider
        .generate_stream(serde_json::json!({
            "model": "llama3.2",
            "prompt": "say hi",
            // Caller passes stream:false; helper must force true.
            "stream": false,
        }))
        .await
        .expect("stream open ok");
    m.assert_async().await;

    let body = resp.text().await.expect("drain body");
    let frames: Vec<&str> = body.lines().filter(|l| !l.trim().is_empty()).collect();
    assert_eq!(frames.len(), 3);
    // First two are mid-stream; last is terminal.
    let last: serde_json::Value = serde_json::from_str(frames[2]).unwrap();
    assert_eq!(last["done"], true);
    assert_eq!(last["done_reason"], "stop");
}

#[tokio::test]
async fn chat_non_streaming_roundtrip_with_auth_header() {
    let mut server = mockito::Server::new_async().await;
    let m = server
        .mock("POST", "/api/chat")
        .match_header("authorization", "Bearer secret-token")
        .match_body(mockito::Matcher::PartialJson(serde_json::json!({
            "model": "llama3.2",
            "stream": false,
        })))
        .with_status(200)
        .with_body(
            serde_json::json!({
                "model": "llama3.2",
                "message": {"role": "assistant", "content": "hi there"},
                "done": true,
                "done_reason": "stop",
            })
            .to_string(),
        )
        .create_async()
        .await;

    let opts = OllamaOptions {
        api_key: Some("secret-token".into()),
        ..opts_for(&server.url())
    };
    let provider = OllamaProvider::from_options(opts).unwrap();
    let resp = provider
        .chat(serde_json::json!({
            "model": "llama3.2",
            "messages": [{"role": "user", "content": "hi"}],
        }))
        .await
        .expect("chat ok");

    m.assert_async().await;
    assert_eq!(resp["message"]["content"], "hi there");
}

#[tokio::test]
async fn embeddings_roundtrip() {
    let mut server = mockito::Server::new_async().await;
    let m = server
        .mock("POST", "/api/embeddings")
        .match_body(mockito::Matcher::PartialJson(serde_json::json!({
            "model": "nomic-embed-text",
            "prompt": "embed me",
        })))
        .with_status(200)
        .with_body(
            serde_json::json!({
                "embedding": [0.1f32, 0.2, 0.3, 0.4]
            })
            .to_string(),
        )
        .create_async()
        .await;

    let provider = OllamaProvider::from_options(opts_for(&server.url())).unwrap();
    let resp = provider
        .embed(serde_json::json!({
            "model": "nomic-embed-text",
            "prompt": "embed me",
        }))
        .await
        .expect("embeddings ok");

    m.assert_async().await;
    assert_eq!(resp["embedding"].as_array().unwrap().len(), 4);
}

#[tokio::test]
async fn list_tags_returns_installed_models() {
    let mut server = mockito::Server::new_async().await;
    let m = server
        .mock("GET", "/api/tags")
        .with_status(200)
        .with_header("content-type", "application/json")
        .with_body(
            serde_json::json!({
                "models": [
                    {"name": "llama3.2", "digest": "sha256:abc", "size": 4_200_000_000u64},
                    {"name": "llama3.2-sql-lora", "digest": "sha256:def", "size": 4_300_000_000u64}
                ]
            })
            .to_string(),
        )
        .create_async()
        .await;

    let provider = OllamaProvider::from_options(opts_for(&server.url())).unwrap();
    let listing = provider.tags().await.expect("tags ok");
    m.assert_async().await;
    assert_eq!(listing.len(), 2);
    assert_eq!(listing[0].name, "llama3.2");
    assert_eq!(listing[1].name, "llama3.2-sql-lora");
}

#[tokio::test]
async fn pull_progress_frames_drain_through_callback() {
    let mut server = mockito::Server::new_async().await;
    let ndjson = [
        r#"{"status":"pulling manifest"}"#,
        r#"{"status":"downloading","digest":"sha256:abc","total":1000,"completed":250}"#,
        r#"{"status":"downloading","digest":"sha256:abc","total":1000,"completed":1000}"#,
        r#"{"status":"verifying sha256 digest"}"#,
        r#"{"status":"success"}"#,
    ]
    .join("\n");

    let m = server
        .mock("POST", "/api/pull")
        .match_body(mockito::Matcher::PartialJson(serde_json::json!({
            "name": "llama3.2",
            "stream": true,
        })))
        .with_status(200)
        .with_header("content-type", "application/x-ndjson")
        .with_body(ndjson)
        .create_async()
        .await;

    let provider = OllamaProvider::from_options(opts_for(&server.url())).unwrap();
    let captured: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));
    let cap_clone = Arc::clone(&captured);
    provider
        .pull("llama3.2", move |p: &OllamaPullProgress| {
            cap_clone.lock().unwrap().push(p.status.clone());
        })
        .await
        .expect("pull ok");

    m.assert_async().await;
    let saw = captured.lock().unwrap();
    assert_eq!(saw.len(), 5);
    assert_eq!(saw[0], "pulling manifest");
    assert_eq!(saw[4], "success");
}

#[tokio::test]
async fn pull_error_frame_surfaces_as_adapter_failed() {
    let mut server = mockito::Server::new_async().await;
    let ndjson = [
        r#"{"status":"pulling manifest"}"#,
        r#"{"status":"error","error":"pull manifest: repository not found"}"#,
    ]
    .join("\n");

    let _m = server
        .mock("POST", "/api/pull")
        .with_status(200)
        .with_body(ndjson)
        .create_async()
        .await;

    let provider = OllamaProvider::from_options(opts_for(&server.url())).unwrap();
    let err = provider
        .pull("nope/missing", |_| {})
        .await
        .expect_err("error frame must propagate");
    assert!(matches!(err, OllamaError::AdapterFailed(_)));
}

#[tokio::test]
async fn load_adapter_localfs_sends_modelfile_via_create() {
    let mut server = mockito::Server::new_async().await;
    // Ollama responds with NDJSON frames terminated by {status:"success"}.
    let m = server
        .mock("POST", "/api/create")
        .match_body(mockito::Matcher::AllOf(vec![
            mockito::Matcher::PartialJson(serde_json::json!({
                "name": "llama3.2-sql-lora",
                "stream": true,
            })),
            // The Modelfile body should contain both FROM and ADAPTER lines.
            mockito::Matcher::Regex("FROM llama3\\.2".into()),
            mockito::Matcher::Regex("ADAPTER /srv/loras/sql-lora".into()),
        ]))
        .with_status(200)
        .with_body(
            [
                r#"{"status":"reading model metadata"}"#,
                r#"{"status":"creating system layer"}"#,
                r#"{"status":"success"}"#,
            ]
            .join("\n"),
        )
        .create_async()
        .await;

    let provider = OllamaProvider::from_options(opts_for(&server.url())).unwrap();
    let mounted = provider
        .load_adapter("llama3.2-sql-lora", &PathBuf::from("/srv/loras/sql-lora"))
        .await
        .expect("load_adapter ok");

    m.assert_async().await;
    assert_eq!(mounted.adapter_id, "llama3.2-sql-lora");
    let listed = provider.list_adapters().await;
    assert_eq!(listed.len(), 1);
}

#[tokio::test]
async fn load_adapter_create_error_frame_surfaces_as_adapter_failed() {
    let mut server = mockito::Server::new_async().await;
    let _m = server
        .mock("POST", "/api/create")
        .with_status(200)
        .with_body(
            [
                r#"{"status":"reading model metadata"}"#,
                r#"{"status":"error","error":"adapter file not found"}"#,
            ]
            .join("\n"),
        )
        .create_async()
        .await;

    let provider = OllamaProvider::from_options(opts_for(&server.url())).unwrap();
    let err = provider
        .load_adapter("derived-name", &PathBuf::from("/missing/path"))
        .await
        .expect_err("error frame must surface");
    assert!(matches!(err, OllamaError::AdapterFailed(_)));
    // Cache must not contain the failed adapter.
    assert!(provider.list_adapters().await.is_empty());
}

#[tokio::test]
async fn load_adapter_create_non_2xx_surfaces_as_adapter_failed() {
    let mut server = mockito::Server::new_async().await;
    let _m = server
        .mock("POST", "/api/create")
        .with_status(500)
        .with_body("internal error")
        .create_async()
        .await;

    let provider = OllamaProvider::from_options(opts_for(&server.url())).unwrap();
    let err = provider
        .load_adapter("derived-name", &PathBuf::from("/srv/loras/sql-lora"))
        .await
        .expect_err("500 must surface as AdapterFailed");
    assert!(matches!(err, OllamaError::AdapterFailed(_)));
}

#[tokio::test]
async fn load_adapter_hfhub_pulls_then_creates() {
    let mut server = mockito::Server::new_async().await;
    let pull = server
        .mock("POST", "/api/pull")
        .match_body(mockito::Matcher::PartialJson(serde_json::json!({
            "name": "hf://tloen/alpaca-lora-7b:v1.0",
            "stream": true,
        })))
        .with_status(200)
        .with_body(r#"{"status":"success"}"#)
        .create_async()
        .await;
    let create = server
        .mock("POST", "/api/create")
        .match_body(mockito::Matcher::AllOf(vec![
            mockito::Matcher::PartialJson(serde_json::json!({
                "name": "alpaca",
                "stream": true,
            })),
            mockito::Matcher::Regex("ADAPTER hf://tloen/alpaca-lora-7b:v1\\.0".into()),
        ]))
        .with_status(200)
        .with_body(r#"{"status":"success"}"#)
        .create_async()
        .await;

    let opts = OllamaOptions {
        adapter_transport: OllamaAdapterTransport::HfHub {
            repo: "tloen/alpaca-lora-7b".into(),
            revision: Some("v1.0".into()),
        },
        ..opts_for(&server.url())
    };
    let provider = OllamaProvider::from_options(opts).unwrap();
    // path_or_dir is ignored under HfHub transport.
    provider
        .load_adapter("alpaca", &PathBuf::from("/unused"))
        .await
        .expect("hf-hub mount ok");

    pull.assert_async().await;
    create.assert_async().await;
}

#[tokio::test]
async fn load_adapter_rejects_http_push_without_hitting_server() {
    let server = mockito::Server::new_async().await;
    // No mocks registered: the test fails loudly if the provider actually
    // tries to POST anything.
    let opts = OllamaOptions {
        adapter_transport: OllamaAdapterTransport::HttpPush(vec![0, 1, 2]),
        ..opts_for(&server.url())
    };
    let provider = OllamaProvider::from_options(opts).unwrap();
    let err = provider
        .load_adapter("a1", &PathBuf::from("/srv/loras/a1"))
        .await
        .expect_err("HttpPush must be rejected up-front");
    assert!(matches!(err, OllamaError::Unsupported(_)));
}

#[tokio::test]
async fn unload_adapter_deletes_derived_model() {
    let mut server = mockito::Server::new_async().await;
    let create = server
        .mock("POST", "/api/create")
        .with_status(200)
        .with_body(r#"{"status":"success"}"#)
        .create_async()
        .await;
    let delete = server
        .mock("DELETE", "/api/delete")
        .match_body(mockito::Matcher::PartialJson(serde_json::json!({
            "name": "llama3.2-sql-lora",
        })))
        .with_status(200)
        .with_body("")
        .create_async()
        .await;

    let provider = OllamaProvider::from_options(opts_for(&server.url())).unwrap();
    provider
        .load_adapter("llama3.2-sql-lora", &PathBuf::from("/srv/loras/sql-lora"))
        .await
        .unwrap();
    provider.unload_adapter("llama3.2-sql-lora").await.unwrap();

    create.assert_async().await;
    delete.assert_async().await;
    assert!(provider.list_adapters().await.is_empty());
}

#[tokio::test]
async fn missing_model_404_maps_to_not_found() {
    let mut server = mockito::Server::new_async().await;
    let _m = server
        .mock("POST", "/api/show")
        .with_status(404)
        .with_body(r#"{"error":"model 'nope' not found"}"#)
        .create_async()
        .await;

    let provider = OllamaProvider::from_options(opts_for(&server.url())).unwrap();
    let err = provider
        .client()
        .show("nope")
        .await
        .expect_err("404 must surface");
    assert!(matches!(err, OllamaError::NotFound(_)));
}

#[tokio::test]
async fn generate_500_maps_to_http() {
    let mut server = mockito::Server::new_async().await;
    let _m = server
        .mock("POST", "/api/generate")
        .with_status(500)
        .with_body("kaboom")
        .create_async()
        .await;

    let provider = OllamaProvider::from_options(opts_for(&server.url())).unwrap();
    let err = provider
        .generate(serde_json::json!({"model": "llama3.2", "prompt": "x"}))
        .await
        .expect_err("500 must surface");
    assert!(matches!(err, OllamaError::Http { status: 500, .. }));
}

#[tokio::test]
async fn refresh_adapters_filters_out_base_model_from_tags() {
    let mut server = mockito::Server::new_async().await;
    let _m = server
        .mock("GET", "/api/tags")
        .with_status(200)
        .with_body(
            serde_json::json!({
                "models": [
                    {"name": "llama3.2"},
                    {"name": "llama3.2-sql-lora"},
                    {"name": "llama3.2-code-lora"}
                ]
            })
            .to_string(),
        )
        .create_async()
        .await;

    let provider = OllamaProvider::from_options(opts_for(&server.url())).unwrap();
    let upstream = provider.refresh_adapters_from_server().await.unwrap();
    assert_eq!(upstream.len(), 3);
    let cached = provider.list_adapters().await;
    let ids: Vec<&str> = cached.iter().map(|a| a.adapter_id.as_str()).collect();
    assert!(ids.contains(&"llama3.2-sql-lora"));
    assert!(ids.contains(&"llama3.2-code-lora"));
    assert!(!ids.contains(&"llama3.2"));
}
