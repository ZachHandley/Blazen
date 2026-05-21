//! End-to-end smoke test for the PR5 phase 2 REST surface.
//!
//! Wires a `MockManagerHandle`-style fixture into [`build_router`], then
//! drives a few canonical requests via `tower::ServiceExt::oneshot` —
//! exactly how a real-world HTTP server would be exercised. Unlike the
//! per-route unit tests inside `http/openai_compat.rs`, this file uses
//! the public [`build_router`] entry point and a hand-rolled
//! `ManagerHandle` impl to verify the wiring at the crate boundary.

#![cfg(feature = "http-rest")]

use std::sync::Arc;

use async_trait::async_trait;
use axum::body::{Body, to_bytes};
use axum::http::{Request, StatusCode};
use blazen_controlplane::ManagerHandle;
use blazen_controlplane::build_router;
use blazen_controlplane::model_protocol::{
    AdapterMountStrategyWire, AdapterStatusWire, BackendHintWire, CompleteRequest,
    CompleteResponse, EmbedRequest, EmbedResponse, FetchBlobRequest, GenerateImageRequest,
    GenerateImageResponse, GenerateMusicRequest, GenerateMusicResponse, ImageBlobWire,
    IsLoadedRequest, IsLoadedResponse, ListAdaptersRequest, ListAdaptersResponse,
    LoadAdapterRequest, LoadAdapterResponse, LoadFromHfRequest, LoadFromHfResponse, LoadRequest,
    LoadResponse, MODEL_ENVELOPE_VERSION, ModelStatusWire, PoolWire, RpcError, StatusRequest,
    StatusResponse, StreamCompleteChunk, TextToSpeechRequest, TextToSpeechResponse,
    TranscribeRequest, TranscribeResponse, UnloadAdapterRequest, UnloadAdapterResponse,
    UnloadRequest, UnloadResponse, UploadBlobChunk, UploadBlobResponse,
};
use blazen_controlplane::server::model_manager::{FetchBlobStream, StreamCompleteStream};
use serde_json::Value;
use tokio::sync::mpsc;
use tower::ServiceExt;

/// Minimal `ManagerHandle` for the smoke test. Records nothing, returns
/// deterministic canned responses so the assertions stay tight.
struct SmokeHandle;

#[async_trait]
impl ManagerHandle for SmokeHandle {
    async fn load(&self, _: LoadRequest) -> Result<LoadResponse, RpcError> {
        Ok(LoadResponse {
            envelope_version: MODEL_ENVELOPE_VERSION,
        })
    }
    async fn unload(&self, _: UnloadRequest) -> Result<UnloadResponse, RpcError> {
        Ok(UnloadResponse {
            envelope_version: MODEL_ENVELOPE_VERSION,
        })
    }
    async fn is_loaded(&self, _: IsLoadedRequest) -> Result<IsLoadedResponse, RpcError> {
        Ok(IsLoadedResponse {
            envelope_version: MODEL_ENVELOPE_VERSION,
            loaded: true,
        })
    }
    async fn status(&self, _: StatusRequest) -> Result<StatusResponse, RpcError> {
        Ok(StatusResponse {
            envelope_version: MODEL_ENVELOPE_VERSION,
            models: vec![ModelStatusWire {
                id: "qwen3-7b".to_owned(),
                loaded: true,
                memory_estimate_bytes: 15 * 1024 * 1024 * 1024,
                pool: PoolWire::Gpu(0),
                adapters: vec![],
            }],
        })
    }
    async fn load_from_hf(&self, _: LoadFromHfRequest) -> Result<LoadFromHfResponse, RpcError> {
        Ok(LoadFromHfResponse {
            envelope_version: MODEL_ENVELOPE_VERSION,
            chosen_backend: BackendHintWire::Candle,
        })
    }
    async fn load_adapter(&self, req: LoadAdapterRequest) -> Result<LoadAdapterResponse, RpcError> {
        Ok(LoadAdapterResponse {
            envelope_version: MODEL_ENVELOPE_VERSION,
            adapter_id: req.adapter_id,
            memory_bytes: 2048,
            mount_strategy: AdapterMountStrategyWire::Merged,
        })
    }
    async fn unload_adapter(
        &self,
        _: UnloadAdapterRequest,
    ) -> Result<UnloadAdapterResponse, RpcError> {
        Ok(UnloadAdapterResponse {
            envelope_version: MODEL_ENVELOPE_VERSION,
        })
    }
    async fn list_adapters(
        &self,
        _: ListAdaptersRequest,
    ) -> Result<ListAdaptersResponse, RpcError> {
        Ok(ListAdaptersResponse {
            envelope_version: MODEL_ENVELOPE_VERSION,
            adapters: vec![AdapterStatusWire {
                adapter_id: "lora-1".into(),
                scale: 0.5,
                source_dir: "/srv/adapters/lora-1".into(),
                memory_bytes: 4096,
            }],
        })
    }
    async fn complete(&self, _: CompleteRequest) -> Result<CompleteResponse, RpcError> {
        Ok(CompleteResponse {
            envelope_version: MODEL_ENVELOPE_VERSION,
            text: "hello from smoke".into(),
            prompt_tokens: Some(5),
            completion_tokens: Some(3),
            finish_reason: Some("stop".into()),
            tool_calls_json: Vec::new(),
        })
    }
    async fn stream_complete(&self, _: CompleteRequest) -> Result<StreamCompleteStream, RpcError> {
        let chunks = vec![
            Ok(StreamCompleteChunk::Delta {
                envelope_version: MODEL_ENVELOPE_VERSION,
                text: "hello ".into(),
            }),
            Ok(StreamCompleteChunk::Delta {
                envelope_version: MODEL_ENVELOPE_VERSION,
                text: "from ".into(),
            }),
            Ok(StreamCompleteChunk::Delta {
                envelope_version: MODEL_ENVELOPE_VERSION,
                text: "smoke".into(),
            }),
            Ok(StreamCompleteChunk::Done {
                envelope_version: MODEL_ENVELOPE_VERSION,
                prompt_tokens: Some(5),
                completion_tokens: Some(3),
                finish_reason: Some("stop".into()),
            }),
        ];
        Ok(Box::pin(tokio_stream::iter(chunks)))
    }
    async fn embed(&self, req: EmbedRequest) -> Result<EmbedResponse, RpcError> {
        #[allow(clippy::cast_precision_loss)]
        let vectors = req
            .inputs
            .iter()
            .enumerate()
            .map(|(i, _)| vec![i as f32, (i + 1) as f32])
            .collect();
        Ok(EmbedResponse {
            envelope_version: MODEL_ENVELOPE_VERSION,
            vectors,
            prompt_tokens: Some(7),
        })
    }
    async fn generate_image(
        &self,
        _: GenerateImageRequest,
    ) -> Result<GenerateImageResponse, RpcError> {
        Ok(GenerateImageResponse {
            envelope_version: MODEL_ENVELOPE_VERSION,
            images: vec![ImageBlobWire {
                mime: "image/png".into(),
                data: vec![0x89, 0x50, 0x4e, 0x47],
            }],
        })
    }
    async fn text_to_speech(
        &self,
        _: TextToSpeechRequest,
    ) -> Result<TextToSpeechResponse, RpcError> {
        Ok(TextToSpeechResponse {
            envelope_version: MODEL_ENVELOPE_VERSION,
            mime: "audio/mpeg".into(),
            data: vec![0x49, 0x44, 0x33],
            sample_rate_hz: Some(24_000),
        })
    }
    async fn generate_music(
        &self,
        _: GenerateMusicRequest,
    ) -> Result<GenerateMusicResponse, RpcError> {
        Ok(GenerateMusicResponse {
            envelope_version: MODEL_ENVELOPE_VERSION,
            mime: "audio/wav".into(),
            data: Vec::new(),
            sample_rate_hz: None,
        })
    }
    async fn transcribe(&self, _: TranscribeRequest) -> Result<TranscribeResponse, RpcError> {
        Ok(TranscribeResponse {
            envelope_version: MODEL_ENVELOPE_VERSION,
            text: "smoke transcript".into(),
            language: Some("en".into()),
            segments_json: Vec::new(),
        })
    }
    async fn upload_blob(
        &self,
        _: mpsc::Receiver<UploadBlobChunk>,
    ) -> Result<UploadBlobResponse, RpcError> {
        Ok(UploadBlobResponse {
            envelope_version: MODEL_ENVELOPE_VERSION,
            blob_id: "smoke".into(),
            bytes_received: 0,
        })
    }
    async fn fetch_blob(&self, _: FetchBlobRequest) -> Result<FetchBlobStream, RpcError> {
        Ok(Box::pin(tokio_stream::iter(Vec::new())))
    }
}

fn router() -> axum::Router {
    build_router(Arc::new(SmokeHandle))
}

async fn body_json(resp: axum::http::Response<Body>) -> Value {
    let bytes = to_bytes(resp.into_body(), 8 * 1024 * 1024).await.unwrap();
    serde_json::from_slice(&bytes).unwrap()
}

async fn body_bytes(resp: axum::http::Response<Body>) -> Vec<u8> {
    to_bytes(resp.into_body(), 8 * 1024 * 1024)
        .await
        .unwrap()
        .to_vec()
}

fn post_json(path: &str, body: &Value) -> Request<Body> {
    Request::builder()
        .method("POST")
        .uri(path)
        .header("content-type", "application/json")
        .body(Body::from(serde_json::to_vec(body).unwrap()))
        .unwrap()
}

#[tokio::test]
async fn chat_completions_end_to_end() {
    let r = router();
    let body = serde_json::json!({
        "model": "qwen3-7b",
        "messages": [{"role":"user","content":"hi"}],
    });
    let resp = r
        .oneshot(post_json("/v1/chat/completions", &body))
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let v = body_json(resp).await;
    assert_eq!(v["model"], "qwen3-7b");
    assert_eq!(v["choices"][0]["message"]["content"], "hello from smoke");
    assert_eq!(v["usage"]["total_tokens"], 8);
}

#[tokio::test]
async fn chat_completions_stream_end_to_end() {
    let r = router();
    let body = serde_json::json!({
        "model": "qwen3-7b",
        "messages": [{"role":"user","content":"hi"}],
        "stream": true,
    });
    let resp = r
        .oneshot(post_json("/v1/chat/completions", &body))
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let text = String::from_utf8(body_bytes(resp).await).unwrap();
    let data_lines: Vec<_> = text.lines().filter(|l| l.starts_with("data:")).collect();
    // Three deltas + one done frame + `[DONE]` terminator = 5 events.
    assert_eq!(data_lines.len(), 5, "saw: {data_lines:?}");
    assert!(text.contains("hello "));
    assert!(text.contains("from "));
    assert!(text.contains("smoke"));
    assert!(text.contains("[DONE]"));
}

#[tokio::test]
async fn embeddings_end_to_end() {
    let r = router();
    let body = serde_json::json!({"model":"bge","input":["a","b"]});
    let resp = r.oneshot(post_json("/v1/embeddings", &body)).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let v = body_json(resp).await;
    assert_eq!(v["data"].as_array().unwrap().len(), 2);
    assert_eq!(v["data"][1]["embedding"][1], 2.0);
    assert_eq!(v["usage"]["prompt_tokens"], 7);
}

#[tokio::test]
async fn list_models_end_to_end() {
    let r = router();
    let req = Request::builder()
        .method("GET")
        .uri("/v1/models")
        .body(Body::empty())
        .unwrap();
    let resp = r.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let v = body_json(resp).await;
    assert_eq!(v["data"][0]["id"], "qwen3-7b");
}

#[tokio::test]
async fn images_generations_end_to_end() {
    let r = router();
    let body = serde_json::json!({
        "model": "sdxl",
        "prompt": "a cat",
        "size": "512x512",
        "n": 1,
    });
    let resp = r
        .oneshot(post_json("/v1/images/generations", &body))
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let v = body_json(resp).await;
    let b64 = v["data"][0]["b64_json"].as_str().unwrap();
    // PNG magic header `\x89PNG` base64-encodes to `iVBO...`.
    assert!(b64.starts_with("iVBO"), "got {b64}");
}

#[tokio::test]
async fn audio_speech_end_to_end() {
    let r = router();
    let body = serde_json::json!({"model":"kokoro","input":"hi"});
    let resp = r
        .oneshot(post_json("/v1/audio/speech", &body))
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let ct = resp
        .headers()
        .get(axum::http::header::CONTENT_TYPE)
        .unwrap()
        .clone();
    assert_eq!(ct, "audio/mpeg");
    let bytes = body_bytes(resp).await;
    assert_eq!(bytes, vec![0x49, 0x44, 0x33]);
}

#[tokio::test]
async fn admin_list_adapters_end_to_end() {
    let r = router();
    let req = Request::builder()
        .method("GET")
        .uri("/v1/blazen/adapters/qwen3-7b")
        .body(Body::empty())
        .unwrap();
    let resp = r.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let v = body_json(resp).await;
    let adapters = v["adapters"].as_array().unwrap();
    assert_eq!(adapters.len(), 1);
    assert_eq!(adapters[0]["adapter_id"], "lora-1");
}

#[tokio::test]
async fn admin_load_adapter_local_fs_roundtrip() {
    let r = router();
    let body = serde_json::json!({
        "adapter_id": "lora-1",
        "scale": 0.75,
        "source": { "type": "local_fs", "path": "/srv/adapters/lora-1" },
    });
    let resp = r
        .oneshot(post_json("/v1/blazen/adapters/qwen3-7b/load", &body))
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let v = body_json(resp).await;
    assert_eq!(v["adapter_id"], "lora-1");
    assert_eq!(v["mount_strategy"], "merged");
}

#[tokio::test]
async fn admin_load_from_hf_end_to_end() {
    let r = router();
    let body = serde_json::json!({
        "model_id": "qwen-7b",
        "repo": "Qwen/Qwen3-7B",
        "backend_hint": "candle",
    });
    let resp = r
        .oneshot(post_json("/v1/blazen/models/load_from_hf", &body))
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let v = body_json(resp).await;
    assert_eq!(v["chosen_backend"], "candle");
}

#[tokio::test]
async fn admin_model_status_end_to_end() {
    let r = router();
    let req = Request::builder()
        .method("GET")
        .uri("/v1/blazen/models/qwen3-7b/status")
        .body(Body::empty())
        .unwrap();
    let resp = r.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let v = body_json(resp).await;
    assert_eq!(v["id"], "qwen3-7b");
    assert_eq!(v["loaded"], true);
    assert_eq!(v["pool"]["kind"], "gpu");
    assert_eq!(v["pool"]["device"], 0);
}

#[tokio::test]
async fn admin_health_end_to_end() {
    let r = router();
    let req = Request::builder()
        .method("GET")
        .uri("/v1/blazen/health")
        .body(Body::empty())
        .unwrap();
    let resp = r.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let v = body_json(resp).await;
    assert_eq!(v["status"], "ok");
}

#[tokio::test]
async fn admin_metrics_emit_after_dispatch() {
    let r = router();
    // Drive one dispatch through `/v1/embeddings` to bump the counter.
    let body = serde_json::json!({"model":"bge","input":"hi"});
    let resp = r
        .clone()
        .oneshot(post_json("/v1/embeddings", &body))
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);

    let req = Request::builder()
        .method("GET")
        .uri("/v1/blazen/metrics")
        .body(Body::empty())
        .unwrap();
    let resp = r.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let text = String::from_utf8(body_bytes(resp).await).unwrap();
    assert!(text.contains("blazen_rest_calls_total"));
    assert!(text.contains("rpc=\"embed\""));
}

#[tokio::test]
async fn unknown_route_returns_404() {
    let r = router();
    let req = Request::builder()
        .method("GET")
        .uri("/v1/does-not-exist")
        .body(Body::empty())
        .unwrap();
    let resp = r.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::NOT_FOUND);
}
