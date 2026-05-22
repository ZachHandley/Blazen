//! UniFFI surface for [`blazen_controlplane::ModelClient`].
//!
//! Mirrors the structure of [`crate::controlplane`]: a thin
//! `#[derive(uniffi::Object)]` wrapper around the upstream gRPC client,
//! exposed under `#[uniffi::export(async_runtime = "tokio")]` so foreign
//! callers see language-native async (Swift `async`, Kotlin `suspend
//! fun`, Go blocking).
//!
//! Only a minimal lifecycle surface is exposed in this wave:
//! [`ModelClient::connect`], [`ModelClient::connect_with_tls`],
//! [`ModelClient::status`], and [`ModelClient::is_loaded`]. The
//! remaining inference / adapter / blob verbs land in later waves.
//!
//! ## TLS
//!
//! `connect_with_tls` accepts PEM strings in-memory rather than file
//! paths — the same as how the Node / Python bindings handle TLS — so
//! foreign callers can source the material from whatever store they
//! prefer (Keychain, Windows cert store, env var, etc.) without writing
//! it to disk first.

use std::sync::Arc;

use futures_util::StreamExt;
use tonic::transport::{Certificate, ClientTlsConfig, Identity};

use blazen_controlplane::ModelClient as CoreModelClient;
use blazen_controlplane::model_protocol::{
    BackendHintWire, CompleteRequest, EmbedRequest, FetchBlobChunk, FetchBlobRequest,
    GenerateImageRequest, GenerateMusicRequest, IsLoadedRequest, ListAdaptersRequest,
    LoadAdapterRequest, LoadFromHfRequest, LoadRequest, MODEL_ENVELOPE_VERSION, ModelStatusWire,
    PoolWire, StatusRequest, StreamCompleteChunk, TextToSpeechRequest, TranscribeRequest,
    UnloadAdapterRequest, UnloadRequest, UploadBlobChunk,
};

use crate::errors::{BlazenError, BlazenResult};
use crate::llm::TokenUsage;
use crate::streaming::{CompletionStreamSink, StreamChunk};

// ===========================================================================
// Records
// ===========================================================================

/// Pool a model is registered against. Mirrors
/// [`blazen_controlplane::model_protocol::PoolWire`] in a UniFFI-friendly
/// shape — Rust's `Gpu(u32)` payload becomes a discriminator + optional
/// `device_index` so the enum can cross the FFI boundary cleanly.
#[derive(Debug, Clone, uniffi::Enum)]
pub enum ModelPool {
    /// Host RAM pool.
    Cpu,
    /// GPU VRAM pool at `device_index`. Metal collapses to index `0`.
    Gpu { device_index: u32 },
    /// Off-host pool — memory lives in another process / host.
    Remote,
}

impl From<PoolWire> for ModelPool {
    fn from(p: PoolWire) -> Self {
        match p {
            PoolWire::Cpu => Self::Cpu,
            PoolWire::Gpu(device_index) => Self::Gpu { device_index },
            PoolWire::Remote => Self::Remote,
        }
    }
}

/// Foreign-facing snapshot of a single registered model.
///
/// Mirrors [`ModelStatusWire`] from the model protocol. The upstream
/// `adapters: Vec<AdapterStatusWire>` field is omitted in this wave —
/// adapter introspection lands with the adapter RPCs in a later wave.
#[derive(Debug, Clone, uniffi::Record)]
pub struct ModelClientStatusRecord {
    /// Identifier under which the model was registered.
    pub id: String,
    /// Whether the model is currently loaded into its pool.
    pub loaded: bool,
    /// Estimated memory footprint in bytes (includes any mounted adapters).
    pub memory_estimate_bytes: u64,
    /// Pool the model is charged against.
    pub pool: ModelPool,
}

impl From<&ModelStatusWire> for ModelClientStatusRecord {
    fn from(s: &ModelStatusWire) -> Self {
        Self {
            id: s.id.clone(),
            loaded: s.loaded,
            memory_estimate_bytes: s.memory_estimate_bytes,
            pool: s.pool.into(),
        }
    }
}

/// Foreign-facing response from
/// [`ModelClient::status`]. Mirrors
/// [`blazen_controlplane::model_protocol::StatusResponse`] but filters
/// to a single model when the caller scopes the query with a
/// `model_id`.
#[derive(Debug, Clone, uniffi::Record)]
pub struct StatusRecord {
    /// Snapshot of each registered model (or just the requested one,
    /// when `status(Some(id))` was called).
    pub models: Vec<ModelClientStatusRecord>,
}

/// Foreign-facing request for [`ModelClient::load`].
///
/// Mirrors [`blazen_controlplane::model_protocol::LoadRequest`] minus the
/// `envelope_version` (the wrapper fills that in from
/// [`MODEL_ENVELOPE_VERSION`]).
#[derive(Debug, Clone, uniffi::Record)]
pub struct LoadRecord {
    /// Id under which the target model was previously registered.
    pub model_id: String,
}

/// Foreign-facing response from [`ModelClient::load`] and
/// [`ModelClient::load_from_hf`].
///
/// `LoadResponse` is empty on the wire (failures travel via the
/// `Result`), but `LoadFromHfResponse` reports the chosen backend; we
/// surface both through a single record with optional fields so foreign
/// callers see one shape regardless of which loader they invoke.
#[derive(Debug, Clone, uniffi::Record)]
pub struct LoadResultRecord {
    /// Model id that was loaded. Echoes the request's `model_id` so
    /// foreign callers don't have to thread the value through their own
    /// state.
    pub model_id: String,
    /// Whether the load succeeded. Always `true` on the success branch
    /// of the `Result`; provided for forward-compat with future wire
    /// schemas that may carry a richer status.
    pub loaded: bool,
    /// Backend the loader chose. Only populated by `load_from_hf`;
    /// `None` for the plain `load` path which does not negotiate a
    /// backend.
    pub chosen_backend: Option<HfBackendHint>,
}

/// Backend selector used by [`ModelClient::load_from_hf`]. Mirrors
/// [`BackendHintWire`].
#[derive(Debug, Clone, Copy, uniffi::Enum)]
pub enum HfBackendHint {
    /// Auto-detect from the repo layout.
    Auto,
    /// Force the `mistral.rs` backend.
    MistralRs,
    /// Force the candle-llm backend.
    Candle,
    /// Force the llama.cpp backend.
    LlamaCpp,
}

impl From<BackendHintWire> for HfBackendHint {
    fn from(b: BackendHintWire) -> Self {
        match b {
            BackendHintWire::Auto => Self::Auto,
            BackendHintWire::MistralRs => Self::MistralRs,
            BackendHintWire::Candle => Self::Candle,
            BackendHintWire::LlamaCpp => Self::LlamaCpp,
        }
    }
}

impl From<HfBackendHint> for BackendHintWire {
    fn from(b: HfBackendHint) -> Self {
        match b {
            HfBackendHint::Auto => Self::Auto,
            HfBackendHint::MistralRs => Self::MistralRs,
            HfBackendHint::Candle => Self::Candle,
            HfBackendHint::LlamaCpp => Self::LlamaCpp,
        }
    }
}

/// Foreign-facing request for [`ModelClient::load_from_hf`]. Mirrors
/// [`LoadFromHfRequest`] minus the `envelope_version`.
#[derive(Debug, Clone, uniffi::Record)]
pub struct LoadFromHfRecord {
    /// Id under which to register the resulting model.
    pub model_id: String,
    /// Hugging Face repo slug (`org/name`).
    pub repo: String,
    /// Optional explicit memory estimate in bytes; `None` asks the
    /// loader to estimate from repo metadata.
    pub memory_estimate_bytes: Option<u64>,
    /// Optional backend override.
    pub backend_hint: Option<HfBackendHint>,
    /// Optional GGUF file name when the backend is llama.cpp.
    pub gguf_file: Option<String>,
    /// Optional HF revision (branch / tag / commit).
    pub revision: Option<String>,
    /// Optional bearer token for gated repos.
    pub hf_token: Option<String>,
    /// Pre-serialised JSON for any backend-specific extra options the
    /// host should honor. Empty string = none.
    pub extra_options_json: String,
}

// ===========================================================================
// ModelClient
// ===========================================================================

/// gRPC client for the `BlazenModelServer` service, exposed to Go /
/// Swift / Kotlin / Ruby via UniFFI.
///
/// Wraps [`blazen_controlplane::ModelClient`] one-to-one; the upstream
/// type is already cheaply cloneable and serialises concurrent RPCs
/// internally, so the wrapper does not need its own mutex.
#[derive(uniffi::Object)]
pub struct ModelClient {
    inner: CoreModelClient,
}

#[uniffi::export(async_runtime = "tokio")]
impl ModelClient {
    /// Open a plaintext connection to a `BlazenModelServer` at
    /// `endpoint` (e.g. `"http://127.0.0.1:7070"`).
    ///
    /// # Errors
    /// Returns a [`BlazenError::Peer`] with `kind =
    /// "ControlPlaneTransport"` when the endpoint URI is invalid or the
    /// TCP / HTTP-2 handshake fails.
    #[uniffi::constructor]
    pub async fn connect(endpoint: String) -> BlazenResult<Arc<Self>> {
        let inner = CoreModelClient::connect(endpoint)
            .await
            .map_err(BlazenError::from)?;
        Ok(Arc::new(Self { inner }))
    }

    /// Open a TLS / mTLS connection. `ca_cert_pem` is the trust root
    /// the client uses to verify the server; `client_cert_pem` and
    /// `client_key_pem` are the client identity for mutual TLS (pass
    /// both or neither).
    ///
    /// # Errors
    /// Returns a [`BlazenError::Peer`] with `kind = "ControlPlaneTls"`
    /// when the PEM material can't be parsed, or with
    /// `kind = "ControlPlaneTransport"` for handshake failures.
    /// [`BlazenError::Validation`] if exactly one of `client_cert_pem`
    /// / `client_key_pem` is supplied.
    #[uniffi::constructor]
    pub async fn connect_with_tls(
        endpoint: String,
        ca_cert_pem: String,
        client_cert_pem: Option<String>,
        client_key_pem: Option<String>,
    ) -> BlazenResult<Arc<Self>> {
        let identity = match (client_cert_pem, client_key_pem) {
            (Some(cert), Some(key)) => Some(Identity::from_pem(cert, key)),
            (None, None) => None,
            _ => {
                return Err(BlazenError::Validation {
                    message: "client_cert_pem and client_key_pem must be supplied together".into(),
                });
            }
        };
        let mut tls = ClientTlsConfig::new().ca_certificate(Certificate::from_pem(ca_cert_pem));
        if let Some(id) = identity {
            tls = tls.identity(id);
        }
        let inner = CoreModelClient::connect_with_tls(endpoint, Some(tls))
            .await
            .map_err(BlazenError::from)?;
        Ok(Arc::new(Self { inner }))
    }

    /// Fetch the server's view of registered models.
    ///
    /// When `model_id` is `Some(id)`, the response is filtered to just
    /// that model (empty `models` vec if the id is unknown). When
    /// `None`, every registered model is returned.
    ///
    /// # Errors
    /// Returns [`BlazenError::Peer`] (`ControlPlaneTransport` /
    /// `ControlPlaneRpc`) for wire or model-layer failures.
    pub async fn status(self: Arc<Self>, model_id: Option<String>) -> BlazenResult<StatusRecord> {
        let resp = self
            .inner
            .status(StatusRequest {
                envelope_version: MODEL_ENVELOPE_VERSION,
            })
            .await
            .map_err(BlazenError::from)?;
        let models: Vec<ModelClientStatusRecord> = match model_id {
            Some(filter) => resp
                .models
                .iter()
                .filter(|m| m.id == filter)
                .map(Into::into)
                .collect(),
            None => resp.models.iter().map(Into::into).collect(),
        };
        Ok(StatusRecord { models })
    }

    /// Load a previously-registered model into its pool.
    ///
    /// # Errors
    /// Returns [`BlazenError::Peer`] (`ControlPlaneTransport` /
    /// `ControlPlaneRpc`) for wire or model-layer failures (e.g. unknown
    /// `model_id`).
    pub async fn load(self: Arc<Self>, request: LoadRecord) -> BlazenResult<LoadResultRecord> {
        let model_id = request.model_id.clone();
        let _resp = self
            .inner
            .load(LoadRequest {
                envelope_version: MODEL_ENVELOPE_VERSION,
                model_id: request.model_id,
            })
            .await
            .map_err(BlazenError::from)?;
        Ok(LoadResultRecord {
            model_id,
            loaded: true,
            chosen_backend: None,
        })
    }

    /// Drop a previously-loaded model from memory.
    ///
    /// # Errors
    /// See [`Self::load`].
    pub async fn unload(self: Arc<Self>, model_id: String) -> BlazenResult<()> {
        let _resp = self
            .inner
            .unload(UnloadRequest {
                envelope_version: MODEL_ENVELOPE_VERSION,
                model_id,
            })
            .await
            .map_err(BlazenError::from)?;
        Ok(())
    }

    /// Register-and-load a model directly from a Hugging Face Hub repo.
    /// Returns the backend the loader chose (never
    /// [`HfBackendHint::Auto`]).
    ///
    /// # Errors
    /// See [`Self::load`]. Additionally surfaces loader-side failures
    /// (HF fetch errors, unsupported repo layouts) via
    /// `BlazenError::Peer` with `kind = "ControlPlaneRpc"`.
    pub async fn load_from_hf(
        self: Arc<Self>,
        request: LoadFromHfRecord,
    ) -> BlazenResult<LoadResultRecord> {
        let model_id = request.model_id.clone();
        let resp = self
            .inner
            .load_from_hf(LoadFromHfRequest {
                envelope_version: MODEL_ENVELOPE_VERSION,
                model_id: request.model_id,
                repo: request.repo,
                memory_estimate_bytes: request.memory_estimate_bytes,
                backend_hint: request.backend_hint.map(Into::into),
                gguf_file: request.gguf_file,
                revision: request.revision,
                hf_token: request.hf_token,
                extra_options_json: request.extra_options_json.into_bytes(),
            })
            .await
            .map_err(BlazenError::from)?;
        Ok(LoadResultRecord {
            model_id,
            loaded: true,
            chosen_backend: Some(resp.chosen_backend.into()),
        })
    }

    /// Liveness check for a single model.
    ///
    /// # Errors
    /// See [`Self::status`].
    pub async fn is_loaded(self: Arc<Self>, model_id: String) -> BlazenResult<bool> {
        let resp = self
            .inner
            .is_loaded(IsLoadedRequest {
                envelope_version: MODEL_ENVELOPE_VERSION,
                model_id,
            })
            .await
            .map_err(BlazenError::from)?;
        Ok(resp.loaded)
    }

    // ----- JSON-string surface ------------------------------------------------
    //
    // The remaining RPCs accept and return JSON strings rather than typed
    // [`uniffi::Record`]s. Wave 1.1/1.2 used typed records for the
    // high-frequency lifecycle methods (`load`, `unload`, `status`,
    // `load_from_hf`, `is_loaded`) where the schema is small, stable, and
    // hot-path enough to deserve first-class FFI types. The 9 methods below
    // cover adapter management, inference, and multimodal generation —
    // their wire types are large (e.g. `CompleteRequest` carries messages,
    // tool definitions, sampling params, streaming flags) and would explode
    // the UDL surface if mirrored one-to-one as records. Instead, foreign
    // callers serialize the wire struct as JSON in their own language
    // (Go's `encoding/json`, Swift's `JSONEncoder`, Kotlin's
    // `kotlinx.serialization`, Ruby's `JSON.dump`) and we deserialize on
    // the Rust side. The wire structs in
    // `blazen_controlplane::model_protocol` are the canonical schema; their
    // `serde` impls drive both directions.
    //
    // The `envelope_version` field is overwritten with
    // `MODEL_ENVELOPE_VERSION` after deserialization so foreign callers
    // never have to know about envelope versioning — they can omit the
    // field (it'll default to whatever serde fills in) and we stamp the
    // correct value before the RPC fires.

    /// Mount a LoRA / adapter onto a loaded model.
    ///
    /// `request_json` is the JSON form of
    /// [`blazen_controlplane::model_protocol::LoadAdapterRequest`]; the
    /// `envelope_version` field is filled in automatically and may be
    /// omitted by the caller. Returns the JSON form of
    /// [`blazen_controlplane::model_protocol::LoadAdapterResponse`].
    ///
    /// # Errors
    /// Returns [`BlazenError::Validation`] when the request JSON cannot be
    /// parsed or the response cannot be serialized;
    /// [`BlazenError::Peer`] for control-plane / transport failures.
    pub async fn load_adapter(self: Arc<Self>, request_json: String) -> BlazenResult<String> {
        let mut req: LoadAdapterRequest = serde_json::from_str(&request_json)?;
        req.envelope_version = MODEL_ENVELOPE_VERSION;
        let resp = self
            .inner
            .load_adapter(req)
            .await
            .map_err(BlazenError::from)?;
        Ok(serde_json::to_string(&resp)?)
    }

    /// Drop a previously-mounted adapter.
    ///
    /// `request_json` is the JSON form of
    /// [`blazen_controlplane::model_protocol::UnloadAdapterRequest`]; the
    /// `envelope_version` field is filled in automatically. Returns the
    /// JSON form of
    /// [`blazen_controlplane::model_protocol::UnloadAdapterResponse`].
    ///
    /// # Errors
    /// See [`Self::load_adapter`].
    pub async fn unload_adapter(self: Arc<Self>, request_json: String) -> BlazenResult<String> {
        let mut req: UnloadAdapterRequest = serde_json::from_str(&request_json)?;
        req.envelope_version = MODEL_ENVELOPE_VERSION;
        let resp = self
            .inner
            .unload_adapter(req)
            .await
            .map_err(BlazenError::from)?;
        Ok(serde_json::to_string(&resp)?)
    }

    /// List adapters mounted on a model.
    ///
    /// `request_json` is the JSON form of
    /// [`blazen_controlplane::model_protocol::ListAdaptersRequest`]; the
    /// `envelope_version` field is filled in automatically. Returns the
    /// JSON form of
    /// [`blazen_controlplane::model_protocol::ListAdaptersResponse`].
    ///
    /// # Errors
    /// See [`Self::load_adapter`].
    pub async fn list_adapters(self: Arc<Self>, request_json: String) -> BlazenResult<String> {
        let mut req: ListAdaptersRequest = serde_json::from_str(&request_json)?;
        req.envelope_version = MODEL_ENVELOPE_VERSION;
        let resp = self
            .inner
            .list_adapters(req)
            .await
            .map_err(BlazenError::from)?;
        Ok(serde_json::to_string(&resp)?)
    }

    /// Issue a non-streaming completion.
    ///
    /// `request_json` is the JSON form of
    /// [`blazen_controlplane::model_protocol::CompleteRequest`]; the
    /// `envelope_version` field is filled in automatically. Returns the
    /// JSON form of
    /// [`blazen_controlplane::model_protocol::CompleteResponse`].
    ///
    /// For streaming completions use a future wave's `stream_complete`
    /// surface — this method always buffers the full response.
    ///
    /// # Errors
    /// See [`Self::load_adapter`].
    pub async fn complete(self: Arc<Self>, request_json: String) -> BlazenResult<String> {
        let mut req: CompleteRequest = serde_json::from_str(&request_json)?;
        req.envelope_version = MODEL_ENVELOPE_VERSION;
        let resp = self.inner.complete(req).await.map_err(BlazenError::from)?;
        Ok(serde_json::to_string(&resp)?)
    }

    /// Issue a streaming completion, delivering each token-delta to `sink`.
    ///
    /// `request_json` is the JSON form of
    /// [`blazen_controlplane::model_protocol::CompleteRequest`]; the
    /// `envelope_version` field is filled in automatically. As frames arrive
    /// from the server, the [`StreamCompleteChunk::Delta`]'s `text` is
    /// forwarded to [`CompletionStreamSink::on_chunk`] as the chunk's
    /// `content_delta`; the terminal [`StreamCompleteChunk::Done`] triggers
    /// [`CompletionStreamSink::on_done`] with the reported `finish_reason`
    /// (empty string when the provider didn't supply one) and a
    /// [`TokenUsage`] built from the `prompt_tokens` / `completion_tokens`
    /// fields.
    ///
    /// Errors observed mid-stream are *delivered* via
    /// [`CompletionStreamSink::on_error`] and the method returns `Ok(())`,
    /// mirroring the symmetry of
    /// [`crate::streaming::complete_streaming`]: the sink owns both
    /// happy-path and error-path observation. The only way this method
    /// itself returns `Err` is when the initial request JSON cannot be
    /// parsed or the upstream `stream_complete` call fails to *start* the
    /// stream.
    ///
    /// Reuses the existing text-only [`CompletionStreamSink`] so Go / Swift
    /// / Kotlin / Ruby callers see a uniform streaming surface across both
    /// the in-process [`crate::llm::Model`] path and the gRPC
    /// [`ModelClient`] path. The wire-level
    /// [`StreamCompleteChunk`] carries only `text` payloads today (no
    /// per-frame tool-call deltas, citations, or reasoning trace), so the
    /// text-only sink loses no information; if a future wire schema grows
    /// structured fields, callers that need them should drive the gRPC
    /// client directly.
    ///
    /// # Errors
    /// Returns [`BlazenError::Validation`] when the request JSON cannot be
    /// parsed; [`BlazenError::Peer`] for control-plane / transport failures
    /// starting the stream.
    pub async fn stream_complete(
        self: Arc<Self>,
        request_json: String,
        sink: Arc<dyn CompletionStreamSink>,
    ) -> BlazenResult<()> {
        let mut req: CompleteRequest = serde_json::from_str(&request_json)?;
        req.envelope_version = MODEL_ENVELOPE_VERSION;

        let stream = match self.inner.stream_complete(req).await {
            Ok(s) => s,
            Err(err) => return Err(BlazenError::from(err)),
        };

        let mut stream = std::pin::pin!(stream);
        let mut last_finish_reason = String::new();
        let mut usage = TokenUsage::default();
        let mut pending: Option<StreamChunk> = None;

        while let Some(item) = stream.next().await {
            match item {
                Ok(StreamCompleteChunk::Delta { text, .. }) => {
                    let wire = StreamChunk {
                        content_delta: text,
                        tool_calls: Vec::new(),
                        is_final: false,
                    };
                    // Why: defer dispatch by one step so the last
                    // content-bearing chunk can be flagged `is_final = true`
                    // when we observe the terminal `Done` frame (or stream
                    // end) without having to peek ahead.
                    if let Some(prev) = pending.take()
                        && let Err(sink_err) = sink.on_chunk(prev).await
                    {
                        let _ = sink.on_error(sink_err).await;
                        return Ok(());
                    }
                    pending = Some(wire);
                }
                Ok(StreamCompleteChunk::Done {
                    prompt_tokens,
                    completion_tokens,
                    finish_reason,
                    ..
                }) => {
                    if let Some(reason) = finish_reason {
                        last_finish_reason = reason;
                    }
                    let prompt = u64::from(prompt_tokens.unwrap_or(0));
                    let completion = u64::from(completion_tokens.unwrap_or(0));
                    usage = TokenUsage {
                        prompt_tokens: prompt,
                        completion_tokens: completion,
                        total_tokens: prompt + completion,
                        ..TokenUsage::default()
                    };
                }
                Err(err) => {
                    if let Some(prev) = pending.take() {
                        let _ = sink.on_chunk(prev).await;
                    }
                    let _ = sink.on_error(BlazenError::from(err)).await;
                    return Ok(());
                }
            }
        }

        if let Some(mut last) = pending.take() {
            last.is_final = true;
            if let Err(sink_err) = sink.on_chunk(last).await {
                let _ = sink.on_error(sink_err).await;
                return Ok(());
            }
        }

        if let Err(sink_err) = sink.on_done(last_finish_reason, usage).await {
            let _ = sink.on_error(sink_err).await;
        }
        Ok(())
    }

    /// Compute embeddings for one or more inputs.
    ///
    /// `request_json` is the JSON form of
    /// [`blazen_controlplane::model_protocol::EmbedRequest`]; the
    /// `envelope_version` field is filled in automatically. Returns the
    /// JSON form of
    /// [`blazen_controlplane::model_protocol::EmbedResponse`].
    ///
    /// # Errors
    /// See [`Self::load_adapter`].
    pub async fn embed(self: Arc<Self>, request_json: String) -> BlazenResult<String> {
        let mut req: EmbedRequest = serde_json::from_str(&request_json)?;
        req.envelope_version = MODEL_ENVELOPE_VERSION;
        let resp = self.inner.embed(req).await.map_err(BlazenError::from)?;
        Ok(serde_json::to_string(&resp)?)
    }

    /// Generate one or more images.
    ///
    /// `request_json` is the JSON form of
    /// [`blazen_controlplane::model_protocol::GenerateImageRequest`]; the
    /// `envelope_version` field is filled in automatically. Returns the
    /// JSON form of
    /// [`blazen_controlplane::model_protocol::GenerateImageResponse`].
    ///
    /// # Errors
    /// See [`Self::load_adapter`].
    pub async fn generate_image(self: Arc<Self>, request_json: String) -> BlazenResult<String> {
        let mut req: GenerateImageRequest = serde_json::from_str(&request_json)?;
        req.envelope_version = MODEL_ENVELOPE_VERSION;
        let resp = self
            .inner
            .generate_image(req)
            .await
            .map_err(BlazenError::from)?;
        Ok(serde_json::to_string(&resp)?)
    }

    /// Synthesize speech from text.
    ///
    /// `request_json` is the JSON form of
    /// [`blazen_controlplane::model_protocol::TextToSpeechRequest`]; the
    /// `envelope_version` field is filled in automatically. Returns the
    /// JSON form of
    /// [`blazen_controlplane::model_protocol::TextToSpeechResponse`].
    ///
    /// # Errors
    /// See [`Self::load_adapter`].
    pub async fn text_to_speech(self: Arc<Self>, request_json: String) -> BlazenResult<String> {
        let mut req: TextToSpeechRequest = serde_json::from_str(&request_json)?;
        req.envelope_version = MODEL_ENVELOPE_VERSION;
        let resp = self
            .inner
            .text_to_speech(req)
            .await
            .map_err(BlazenError::from)?;
        Ok(serde_json::to_string(&resp)?)
    }

    /// Generate music from a textual prompt.
    ///
    /// `request_json` is the JSON form of
    /// [`blazen_controlplane::model_protocol::GenerateMusicRequest`]; the
    /// `envelope_version` field is filled in automatically. Returns the
    /// JSON form of
    /// [`blazen_controlplane::model_protocol::GenerateMusicResponse`].
    ///
    /// # Errors
    /// See [`Self::load_adapter`].
    pub async fn generate_music(self: Arc<Self>, request_json: String) -> BlazenResult<String> {
        let mut req: GenerateMusicRequest = serde_json::from_str(&request_json)?;
        req.envelope_version = MODEL_ENVELOPE_VERSION;
        let resp = self
            .inner
            .generate_music(req)
            .await
            .map_err(BlazenError::from)?;
        Ok(serde_json::to_string(&resp)?)
    }

    /// Transcribe audio to text.
    ///
    /// `request_json` is the JSON form of
    /// [`blazen_controlplane::model_protocol::TranscribeRequest`]; the
    /// `envelope_version` field is filled in automatically. Returns the
    /// JSON form of
    /// [`blazen_controlplane::model_protocol::TranscribeResponse`].
    ///
    /// # Errors
    /// See [`Self::load_adapter`].
    pub async fn transcribe(self: Arc<Self>, request_json: String) -> BlazenResult<String> {
        let mut req: TranscribeRequest = serde_json::from_str(&request_json)?;
        req.envelope_version = MODEL_ENVELOPE_VERSION;
        let resp = self
            .inner
            .transcribe(req)
            .await
            .map_err(BlazenError::from)?;
        Ok(serde_json::to_string(&resp)?)
    }

    /// Upload a blob in one shot.
    ///
    /// The entire `data` payload is buffered in memory and sent as a single
    /// `UploadBlobChunk::Data` frame between a `Start` (carrying `blob_id`
    /// + `mime`) and `End` frame. Returns the JSON form of
    /// [`blazen_controlplane::model_protocol::UploadBlobResponse`] (the
    /// server's ack, echoing the blob id + bytes received).
    ///
    /// This buffered surface is the simple path for the UniFFI bindings —
    /// the whole payload must fit in process memory on both sides. Callers
    /// pushing multi-gigabyte blobs (e.g. base model weights) should drive
    /// [`blazen_controlplane::ModelClient::upload_blob`] directly from Rust
    /// where they can construct the chunk stream incrementally.
    ///
    /// # Errors
    /// Returns [`BlazenError::Peer`] for control-plane / transport
    /// failures, or [`BlazenError::Validation`] when the response cannot be
    /// serialized.
    pub async fn upload_blob(
        self: Arc<Self>,
        blob_id: String,
        mime: String,
        data: Vec<u8>,
    ) -> BlazenResult<String> {
        // 3 frames: Start, Data (whole payload), End. Buffered fits a
        // single-shot upload by design — see the doc comment.
        let (tx, rx) = tokio::sync::mpsc::channel::<UploadBlobChunk>(3);
        let total_bytes = data.len() as u64;
        let content_type = if mime.is_empty() { None } else { Some(mime) };
        tx.send(UploadBlobChunk::Start {
            envelope_version: MODEL_ENVELOPE_VERSION,
            blob_id,
            total_bytes: Some(total_bytes),
            content_type,
        })
        .await
        .map_err(|e| BlazenError::Validation {
            message: format!("upload_blob channel closed before Start: {e}"),
        })?;
        tx.send(UploadBlobChunk::Data {
            envelope_version: MODEL_ENVELOPE_VERSION,
            bytes: data,
        })
        .await
        .map_err(|e| BlazenError::Validation {
            message: format!("upload_blob channel closed before Data: {e}"),
        })?;
        tx.send(UploadBlobChunk::End {
            envelope_version: MODEL_ENVELOPE_VERSION,
        })
        .await
        .map_err(|e| BlazenError::Validation {
            message: format!("upload_blob channel closed before End: {e}"),
        })?;
        drop(tx);
        let resp = self
            .inner
            .upload_blob(rx)
            .await
            .map_err(BlazenError::from)?;
        Ok(serde_json::to_string(&resp)?)
    }

    /// Fetch a blob in one shot.
    ///
    /// `request_json` is the JSON form of
    /// [`blazen_controlplane::model_protocol::FetchBlobRequest`]; the
    /// `envelope_version` field is filled in automatically. The whole
    /// response stream is buffered in memory: each
    /// [`FetchBlobChunk::Data`] frame's bytes are concatenated and returned
    /// as a single `Vec<u8>`; `Start` and `End` frames carry only metadata
    /// and are not surfaced through this API.
    ///
    /// Callers that need to stream multi-gigabyte blobs without buffering
    /// should drive [`blazen_controlplane::ModelClient::fetch_blob`]
    /// directly from Rust.
    ///
    /// # Errors
    /// Returns [`BlazenError::Validation`] when the request JSON cannot be
    /// parsed; [`BlazenError::Peer`] for control-plane / transport failures
    /// (either starting the stream or mid-stream).
    pub async fn fetch_blob(self: Arc<Self>, request_json: String) -> BlazenResult<Vec<u8>> {
        let mut req: FetchBlobRequest = serde_json::from_str(&request_json)?;
        req.envelope_version = MODEL_ENVELOPE_VERSION;
        let stream = self
            .inner
            .fetch_blob(req)
            .await
            .map_err(BlazenError::from)?;
        let mut stream = std::pin::pin!(stream);
        let mut out: Vec<u8> = Vec::new();
        while let Some(item) = stream.next().await {
            match item.map_err(BlazenError::from)? {
                FetchBlobChunk::Data { bytes, .. } => out.extend_from_slice(&bytes),
                FetchBlobChunk::Start { .. } | FetchBlobChunk::End { .. } => {}
            }
        }
        Ok(out)
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use blazen_controlplane::model_protocol::ModelStatusWire;

    #[test]
    fn pool_cpu_maps_to_cpu() {
        let mapped: ModelPool = PoolWire::Cpu.into();
        assert!(matches!(mapped, ModelPool::Cpu));
    }

    #[test]
    fn pool_gpu_carries_device_index() {
        let mapped: ModelPool = PoolWire::Gpu(3).into();
        match mapped {
            ModelPool::Gpu { device_index } => assert_eq!(device_index, 3),
            other => panic!("expected Gpu, got {other:?}"),
        }
    }

    #[test]
    fn pool_remote_maps_to_remote() {
        let mapped: ModelPool = PoolWire::Remote.into();
        assert!(matches!(mapped, ModelPool::Remote));
    }

    #[test]
    fn model_status_record_round_trips_fields() {
        let wire = ModelStatusWire {
            id: "qwen3-7b".into(),
            loaded: true,
            memory_estimate_bytes: 15 * 1024 * 1024 * 1024,
            pool: PoolWire::Gpu(0),
            adapters: Vec::new(),
        };
        let record: ModelClientStatusRecord = (&wire).into();
        assert_eq!(record.id, "qwen3-7b");
        assert!(record.loaded);
        assert_eq!(record.memory_estimate_bytes, 15 * 1024 * 1024 * 1024);
        assert!(matches!(record.pool, ModelPool::Gpu { device_index: 0 }));
    }
}
