//! Blazen-specific admin endpoints.
//!
//! These complement the OpenAI-compat surface in
//! [`super::openai_compat`]: they expose lifecycle / introspection verbs
//! the `OpenAI` API doesn't model (adapter load/unload, register-and-load
//! from Hugging Face Hub, status, health, metrics).
//!
//! Routes (all under `/v1/blazen`):
//!
//! - `POST   /adapters/{model_id}/load` — body specifies the adapter
//!   source. Maps to [`crate::server::model_manager::ManagerHandle::load_adapter`].
//! - `DELETE /adapters/{model_id}/{adapter_id}` — unmount an adapter.
//! - `GET    /adapters/{model_id}` — list mounted adapters.
//! - `POST   /models/load_from_hf` — register-and-load from a HF repo.
//! - `GET    /models/{id}/status` — per-model snapshot.
//! - `GET    /health` — liveness probe (no upstream call).
//! - `GET    /metrics` — Prometheus exposition.

use std::sync::Arc;

use axum::Json;
use axum::Router;
use axum::extract::{Path, State};
use axum::http::{HeaderValue, StatusCode, header};
use axum::response::{IntoResponse, Response};
use axum::routing::{delete, get, post};
use serde::{Deserialize, Serialize};
use serde_json::Value as Json2;

use crate::model_protocol::{
    AdapterMountStrategyWire, AdapterStatusWire, BackendHintWire, ListAdaptersRequest,
    LoadAdapterRequest, LoadFromHfRequest, MODEL_ENVELOPE_VERSION, ModelStatusWire, PoolWire,
    StatusRequest, UnloadAdapterRequest,
};

use super::error::HttpError;
use super::rest_state::RestState;

/// Build the `/v1/blazen/*` sub-router.
pub fn router(state: Arc<RestState>) -> Router {
    Router::new()
        .route("/v1/blazen/adapters/{model_id}/load", post(load_adapter))
        .route(
            "/v1/blazen/adapters/{model_id}/{adapter_id}",
            delete(unload_adapter),
        )
        .route("/v1/blazen/adapters/{model_id}", get(list_adapters))
        .route("/v1/blazen/models/load_from_hf", post(load_from_hf))
        .route("/v1/blazen/models/{model_id}/status", get(model_status))
        .route("/v1/blazen/health", get(health))
        .route("/v1/blazen/metrics", get(metrics))
        .with_state(state)
}

// ---------------------------------------------------------------------------
// Adapter load
// ---------------------------------------------------------------------------

/// Source descriptor for an adapter load. Mirrors the `AdapterSource`
/// enum in `blazen_manager` without taking that crate as a dep — both
/// sides serialise to the same JSON shape.
#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum AdapterSourceSpec {
    /// On-disk path the server can read directly.
    LocalFs { path: String },
    /// Hugging Face Hub repo (caller is responsible for prior auth).
    HfHub {
        repo: String,
        #[serde(default)]
        revision: Option<String>,
        #[serde(default)]
        hf_token: Option<String>,
    },
    /// Adapter previously staged in the [`super::ContentStore`] via a
    /// chunked upload. The id is the UUID returned from
    /// `POST /v1/blazen/content`.
    ContentStore { handle: String },
}

#[derive(Debug, Deserialize)]
pub struct LoadAdapterBody {
    pub adapter_id: String,
    pub scale: f32,
    pub source: AdapterSourceSpec,
}

#[derive(Debug, Serialize)]
struct LoadAdapterResponseBody {
    adapter_id: String,
    memory_bytes: u64,
    mount_strategy: &'static str,
}

async fn load_adapter(
    State(state): State<Arc<RestState>>,
    Path(model_id): Path<String>,
    Json(body): Json<LoadAdapterBody>,
) -> Result<Json<LoadAdapterResponseBody>, HttpError> {
    state.metrics.record("load_adapter");
    let adapter_dir = resolve_adapter_dir(&state, &body.source)?;
    let wire = LoadAdapterRequest {
        envelope_version: MODEL_ENVELOPE_VERSION,
        model_id,
        adapter_dir,
        adapter_id: body.adapter_id.clone(),
        scale: body.scale,
    };
    let resp = state.handle.load_adapter(wire).await?;
    Ok(Json(LoadAdapterResponseBody {
        adapter_id: resp.adapter_id,
        memory_bytes: resp.memory_bytes,
        mount_strategy: mount_strategy_name(resp.mount_strategy),
    }))
}

fn resolve_adapter_dir(state: &RestState, source: &AdapterSourceSpec) -> Result<String, HttpError> {
    match source {
        AdapterSourceSpec::LocalFs { path } => Ok(path.clone()),
        AdapterSourceSpec::HfHub { repo, revision, .. } => {
            // Hosts that wire `LoadAdapter` through a Hugging Face puller
            // expect a `hf://repo[@revision]` URI in `adapter_dir`. This
            // mirrors the convention used by `blazen-manager`'s loader.
            let mut s = format!("hf://{repo}");
            if let Some(rev) = revision {
                s.push('@');
                s.push_str(rev);
            }
            Ok(s)
        }
        AdapterSourceSpec::ContentStore { handle } => {
            let id = uuid::Uuid::parse_str(handle).map_err(|e| {
                HttpError::bad_request(format!("invalid content-store handle: {e}"))
            })?;
            let blob = state.content_store.get(id).ok_or_else(|| {
                HttpError::not_found(format!("unknown content-store handle {id}"))
            })?;
            // For the MVP we materialise the blob into a temp dir under
            // the system tmp root and hand the path to the manager. The
            // manager removes it after consuming. This avoids forcing
            // every backend to learn a new "in-memory blob" code path.
            let dir = std::env::temp_dir().join(format!("blazen-adapter-{id}"));
            std::fs::create_dir_all(&dir).map_err(|e| {
                HttpError::internal(format!("create adapter dir {}: {e}", dir.display()))
            })?;
            let filename = blob.filename.as_deref().unwrap_or("adapter.bin");
            let target = dir.join(filename);
            std::fs::write(&target, blob.data.as_slice()).map_err(|e| {
                HttpError::internal(format!("write adapter blob {}: {e}", target.display()))
            })?;
            Ok(dir.to_string_lossy().into_owned())
        }
    }
}

fn mount_strategy_name(s: AdapterMountStrategyWire) -> &'static str {
    match s {
        AdapterMountStrategyWire::Attached => "attached",
        AdapterMountStrategyWire::Rebuilt => "rebuilt",
        AdapterMountStrategyWire::Merged => "merged",
    }
}

// ---------------------------------------------------------------------------
// Adapter unload + list
// ---------------------------------------------------------------------------

async fn unload_adapter(
    State(state): State<Arc<RestState>>,
    Path((model_id, adapter_id)): Path<(String, String)>,
) -> Result<StatusCode, HttpError> {
    state.metrics.record("unload_adapter");
    let wire = UnloadAdapterRequest {
        envelope_version: MODEL_ENVELOPE_VERSION,
        model_id,
        adapter_id,
    };
    state.handle.unload_adapter(wire).await?;
    Ok(StatusCode::NO_CONTENT)
}

#[derive(Debug, Serialize)]
struct AdapterListResponse {
    adapters: Vec<AdapterDescriptor>,
}

#[derive(Debug, Serialize)]
struct AdapterDescriptor {
    adapter_id: String,
    scale: f32,
    source_dir: String,
    memory_bytes: u64,
}

impl From<AdapterStatusWire> for AdapterDescriptor {
    fn from(a: AdapterStatusWire) -> Self {
        Self {
            adapter_id: a.adapter_id,
            scale: a.scale,
            source_dir: a.source_dir,
            memory_bytes: a.memory_bytes,
        }
    }
}

async fn list_adapters(
    State(state): State<Arc<RestState>>,
    Path(model_id): Path<String>,
) -> Result<Json<AdapterListResponse>, HttpError> {
    state.metrics.record("list_adapters");
    let wire = ListAdaptersRequest {
        envelope_version: MODEL_ENVELOPE_VERSION,
        model_id,
    };
    let resp = state.handle.list_adapters(wire).await?;
    Ok(Json(AdapterListResponse {
        adapters: resp.adapters.into_iter().map(Into::into).collect(),
    }))
}

// ---------------------------------------------------------------------------
// Models: load_from_hf
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
pub struct LoadFromHfBody {
    pub model_id: String,
    pub repo: String,
    #[serde(default)]
    pub memory_estimate_bytes: Option<u64>,
    #[serde(default)]
    pub backend_hint: Option<BackendHintSpec>,
    #[serde(default)]
    pub gguf_file: Option<String>,
    #[serde(default)]
    pub revision: Option<String>,
    #[serde(default)]
    pub hf_token: Option<String>,
    #[serde(default)]
    pub extra_options: Option<Json2>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BackendHintSpec {
    Auto,
    MistralRs,
    Candle,
    LlamaCpp,
}

impl From<BackendHintSpec> for BackendHintWire {
    fn from(h: BackendHintSpec) -> Self {
        match h {
            BackendHintSpec::Auto => Self::Auto,
            BackendHintSpec::MistralRs => Self::MistralRs,
            BackendHintSpec::Candle => Self::Candle,
            BackendHintSpec::LlamaCpp => Self::LlamaCpp,
        }
    }
}

#[derive(Debug, Serialize)]
struct LoadFromHfResponseBody {
    chosen_backend: &'static str,
}

async fn load_from_hf(
    State(state): State<Arc<RestState>>,
    Json(body): Json<LoadFromHfBody>,
) -> Result<Json<LoadFromHfResponseBody>, HttpError> {
    state.metrics.record("load_from_hf");
    let extra_options_json = match &body.extra_options {
        Some(v) => serde_json::to_vec(v)?,
        None => Vec::new(),
    };
    let wire = LoadFromHfRequest {
        envelope_version: MODEL_ENVELOPE_VERSION,
        model_id: body.model_id,
        repo: body.repo,
        memory_estimate_bytes: body.memory_estimate_bytes,
        backend_hint: body.backend_hint.map(Into::into),
        gguf_file: body.gguf_file,
        revision: body.revision,
        hf_token: body.hf_token,
        extra_options_json,
    };
    let resp = state.handle.load_from_hf(wire).await?;
    Ok(Json(LoadFromHfResponseBody {
        chosen_backend: backend_name(resp.chosen_backend),
    }))
}

fn backend_name(b: BackendHintWire) -> &'static str {
    match b {
        BackendHintWire::Auto => "auto",
        BackendHintWire::MistralRs => "mistralrs",
        BackendHintWire::Candle => "candle",
        BackendHintWire::LlamaCpp => "llamacpp",
    }
}

// ---------------------------------------------------------------------------
// Per-model status
// ---------------------------------------------------------------------------

#[derive(Debug, Serialize)]
struct ModelStatusBody {
    id: String,
    loaded: bool,
    memory_estimate_bytes: u64,
    pool: PoolDescriptor,
    adapters: Vec<AdapterDescriptor>,
}

#[derive(Debug, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
enum PoolDescriptor {
    Cpu,
    Gpu { device: u32 },
    Remote,
}

impl From<PoolWire> for PoolDescriptor {
    fn from(p: PoolWire) -> Self {
        match p {
            PoolWire::Cpu => Self::Cpu,
            PoolWire::Gpu(d) => Self::Gpu { device: d },
            PoolWire::Remote => Self::Remote,
        }
    }
}

impl From<ModelStatusWire> for ModelStatusBody {
    fn from(m: ModelStatusWire) -> Self {
        Self {
            id: m.id,
            loaded: m.loaded,
            memory_estimate_bytes: m.memory_estimate_bytes,
            pool: m.pool.into(),
            adapters: m.adapters.into_iter().map(Into::into).collect(),
        }
    }
}

async fn model_status(
    State(state): State<Arc<RestState>>,
    Path(model_id): Path<String>,
) -> Result<Json<ModelStatusBody>, HttpError> {
    state.metrics.record("status");
    let req = StatusRequest {
        envelope_version: MODEL_ENVELOPE_VERSION,
    };
    let snap = state.handle.status(req).await?;
    snap.models
        .into_iter()
        .find(|m| m.id == model_id)
        .map(|m| Json(m.into()))
        .ok_or_else(|| HttpError::not_found(format!("unknown model '{model_id}'")))
}

// ---------------------------------------------------------------------------
// Health + metrics
// ---------------------------------------------------------------------------

#[derive(Debug, Serialize)]
struct Health {
    status: &'static str,
    envelope_version: u32,
}

async fn health() -> Json<Health> {
    Json(Health {
        status: "ok",
        envelope_version: MODEL_ENVELOPE_VERSION,
    })
}

async fn metrics(State(state): State<Arc<RestState>>) -> Response {
    let body = state.metrics.render_prometheus();
    let mut resp = (StatusCode::OK, body).into_response();
    resp.headers_mut().insert(
        header::CONTENT_TYPE,
        HeaderValue::from_static("text/plain; version=0.0.4"),
    );
    resp
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::server::model_manager::test_support::MockManagerHandle;
    use axum::body::{Body, to_bytes};
    use axum::http::Request;
    use tower::ServiceExt;

    fn router_with_mock() -> (axum::Router, Arc<RestState>) {
        let mock = MockManagerHandle::new();
        let state = Arc::new(RestState::new(mock));
        (router(state.clone()), state)
    }

    async fn body_json(resp: axum::http::Response<Body>) -> Json2 {
        let bytes = to_bytes(resp.into_body(), 1024 * 1024).await.unwrap();
        serde_json::from_slice(&bytes).unwrap()
    }

    async fn body_text(resp: axum::http::Response<Body>) -> String {
        let bytes = to_bytes(resp.into_body(), 1024 * 1024).await.unwrap();
        String::from_utf8(bytes.to_vec()).unwrap()
    }

    fn json_request(method: &str, path: &str, body: &Json2) -> Request<Body> {
        Request::builder()
            .method(method)
            .uri(path)
            .header("content-type", "application/json")
            .body(Body::from(serde_json::to_vec(body).unwrap()))
            .unwrap()
    }

    #[tokio::test]
    async fn health_returns_ok() {
        let (r, _) = router_with_mock();
        let req = Request::builder()
            .method("GET")
            .uri("/v1/blazen/health")
            .body(Body::empty())
            .unwrap();
        let resp = r.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let v = body_json(resp).await;
        assert_eq!(v["status"], "ok");
        assert_eq!(v["envelope_version"], MODEL_ENVELOPE_VERSION);
    }

    #[tokio::test]
    async fn metrics_are_prometheus_shaped() {
        let (r, state) = router_with_mock();
        // Drive one dispatch through health (doesn't record) and one
        // through list_adapters (does record).
        let req = Request::builder()
            .method("GET")
            .uri("/v1/blazen/adapters/qwen")
            .body(Body::empty())
            .unwrap();
        let _ = r.clone().oneshot(req).await.unwrap();

        let req = Request::builder()
            .method("GET")
            .uri("/v1/blazen/metrics")
            .body(Body::empty())
            .unwrap();
        let resp = r.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let txt = body_text(resp).await;
        assert!(txt.contains("blazen_rest_calls_total"));
        assert!(txt.contains("rpc=\"list_adapters\""));
        let _ = state; // ensure state Arc lives through assert
    }

    #[tokio::test]
    async fn load_adapter_local_fs_roundtrip() {
        let (r, _) = router_with_mock();
        let body = serde_json::json!({
            "adapter_id": "lora-1",
            "scale": 0.5,
            "source": { "type": "local_fs", "path": "/srv/adapters/lora-1" },
        });
        let resp = r
            .oneshot(json_request("POST", "/v1/blazen/adapters/qwen/load", &body))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let v = body_json(resp).await;
        assert_eq!(v["adapter_id"], "lora-1");
        assert_eq!(v["mount_strategy"], "attached");
    }

    #[tokio::test]
    async fn load_adapter_hf_hub_builds_uri() {
        let (r, _) = router_with_mock();
        let body = serde_json::json!({
            "adapter_id": "lora-1",
            "scale": 1.0,
            "source": {
                "type": "hf_hub",
                "repo": "user/lora",
                "revision": "main",
            },
        });
        let resp = r
            .oneshot(json_request("POST", "/v1/blazen/adapters/qwen/load", &body))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn load_adapter_unknown_content_store_handle_404() {
        let (r, _) = router_with_mock();
        let body = serde_json::json!({
            "adapter_id": "lora-1",
            "scale": 0.5,
            "source": { "type": "content_store", "handle": "00000000-0000-0000-0000-000000000000" },
        });
        let resp = r
            .oneshot(json_request("POST", "/v1/blazen/adapters/qwen/load", &body))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn load_adapter_bad_content_store_handle_400() {
        let (r, _) = router_with_mock();
        let body = serde_json::json!({
            "adapter_id": "lora-1",
            "scale": 0.5,
            "source": { "type": "content_store", "handle": "not-a-uuid" },
        });
        let resp = r
            .oneshot(json_request("POST", "/v1/blazen/adapters/qwen/load", &body))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn unload_adapter_returns_no_content() {
        let (r, _) = router_with_mock();
        let req = Request::builder()
            .method("DELETE")
            .uri("/v1/blazen/adapters/qwen/lora-1")
            .body(Body::empty())
            .unwrap();
        let resp = r.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::NO_CONTENT);
    }

    #[tokio::test]
    async fn list_adapters_empty_ok() {
        let (r, _) = router_with_mock();
        let req = Request::builder()
            .method("GET")
            .uri("/v1/blazen/adapters/qwen")
            .body(Body::empty())
            .unwrap();
        let resp = r.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let v = body_json(resp).await;
        assert!(v["adapters"].as_array().unwrap().is_empty());
    }

    #[tokio::test]
    async fn load_from_hf_returns_backend() {
        let (r, _) = router_with_mock();
        let body = serde_json::json!({
            "model_id": "qwen-7b",
            "repo": "Qwen/Qwen3-7B",
        });
        let resp = r
            .oneshot(json_request(
                "POST",
                "/v1/blazen/models/load_from_hf",
                &body,
            ))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let v = body_json(resp).await;
        assert_eq!(v["chosen_backend"], "mistralrs");
    }

    #[tokio::test]
    async fn model_status_returns_404_for_unknown() {
        let (r, _) = router_with_mock();
        let req = Request::builder()
            .method("GET")
            .uri("/v1/blazen/models/nope/status")
            .body(Body::empty())
            .unwrap();
        let resp = r.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn content_store_blob_materialises_to_tmp() {
        let (r, state) = router_with_mock();
        // Stage a tiny blob in the store.
        let id = state.content_store.put(super::super::uploads::StoredBlob {
            filename: Some("adapter.bin".into()),
            content_type: "application/octet-stream".into(),
            data: Arc::new(b"hello".to_vec()),
        });
        let body = serde_json::json!({
            "adapter_id": "lora-1",
            "scale": 0.5,
            "source": { "type": "content_store", "handle": id.to_string() },
        });
        let resp = r
            .oneshot(json_request("POST", "/v1/blazen/adapters/qwen/load", &body))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        // The temp dir + file should now exist.
        let dir = std::env::temp_dir().join(format!("blazen-adapter-{id}"));
        let path = dir.join("adapter.bin");
        assert!(path.exists(), "expected {} to exist", path.display());
        // Cleanup.
        let _ = std::fs::remove_dir_all(&dir);
    }
}
