//! Node bindings for [`blazen_controlplane::client::ModelClient`].
//!
//! Exposes a thin [`JsModelClient`] (visible to JS as `ModelClient`) that
//! wraps the inherent-method [`ModelClient`] gRPC client. This wave
//! implements only the connection factories plus the two cheapest status
//! RPCs (`status` and `isLoaded`); the remaining RPCs (`load`, `unload`,
//! `complete`, `embed`, media generators, blob streams, adapters) land in
//! later waves.
//!
//! The wrapper stores the underlying [`ModelClient`] behind an [`Arc`] so
//! the JS handle is cheaply cloneable conceptually (each `#[napi]` async
//! method clones the arc, never the channel).

use std::path::Path;
use std::pin::Pin;
use std::sync::Arc;

use futures_util::StreamExt;
use napi::Status;
use napi::bindgen_prelude::*;
use napi_derive::napi;
use tokio::sync::Mutex;
use tonic::transport::{Certificate, ClientTlsConfig, Identity};

use blazen_controlplane::client::ModelClient;
use blazen_controlplane::client::model_client::{FetchBlobRecvStream, StreamCompleteRecvStream};
use blazen_controlplane::error::ControlPlaneError;
use blazen_controlplane::model_protocol::{
    CompleteRequest, CompleteResponse, EmbedRequest, EmbedResponse, FetchBlobChunk,
    FetchBlobRequest, GenerateImageRequest, GenerateImageResponse, GenerateMusicRequest,
    GenerateMusicResponse, IsLoadedRequest, ListAdaptersRequest, ListAdaptersResponse,
    LoadAdapterRequest, LoadAdapterResponse, LoadFromHfRequest, LoadFromHfResponse, LoadRequest,
    LoadResponse, MODEL_ENVELOPE_VERSION, StatusRequest, StatusResponse, TextToSpeechRequest,
    TextToSpeechResponse, TranscribeRequest, TranscribeResponse, UnloadAdapterRequest,
    UnloadAdapterResponse, UnloadRequest, UnloadResponse, UploadBlobChunk, UploadBlobResponse,
};
use tokio::sync::mpsc;

// ---------------------------------------------------------------------------
// Error mapping
// ---------------------------------------------------------------------------

#[allow(clippy::needless_pass_by_value)]
fn controlplane_error_to_napi(err: ControlPlaneError) -> napi::Error {
    let prefix = match &err {
        ControlPlaneError::Encode(_) => "ControlPlaneEncodeError",
        ControlPlaneError::Json(_) => "ControlPlaneJsonError",
        ControlPlaneError::Transport(_) => "ControlPlaneTransportError",
        ControlPlaneError::EnvelopeVersion { .. } => "ControlPlaneEnvelopeVersionError",
        ControlPlaneError::Tls(_) => "ControlPlaneTlsError",
        ControlPlaneError::Unauthenticated(_) => "ControlPlaneUnauthenticatedError",
        ControlPlaneError::NoMatchingWorker { .. } => "ControlPlaneNoMatchingWorkerError",
        ControlPlaneError::MissingVramHint => "ControlPlaneMissingVramHintError",
        ControlPlaneError::UnknownRun(_) => "ControlPlaneUnknownRunError",
        ControlPlaneError::UnknownWorker(_) => "ControlPlaneUnknownWorkerError",
        ControlPlaneError::Workflow(_) => "ControlPlaneWorkflowError",
        ControlPlaneError::Rpc(_) => "ControlPlaneRpcError",
    };
    napi::Error::new(Status::GenericFailure, format!("[{prefix}] {err}"))
}

/// Build a [`ClientTlsConfig`] from PEM file paths. `client_cert` /
/// `client_key` may be `None` for server-auth-only TLS; both must be
/// present together for mTLS.
fn build_tls_config(
    ca_cert: &str,
    client_cert: Option<&str>,
    client_key: Option<&str>,
) -> Result<ClientTlsConfig> {
    let ca_pem = std::fs::read(Path::new(ca_cert))
        .map_err(|e| napi::Error::from_reason(format!("failed to read ca cert {ca_cert}: {e}")))?;
    let mut cfg = ClientTlsConfig::new().ca_certificate(Certificate::from_pem(ca_pem));
    match (client_cert, client_key) {
        (Some(cert_path), Some(key_path)) => {
            let cert_pem = std::fs::read(Path::new(cert_path)).map_err(|e| {
                napi::Error::from_reason(format!("failed to read client cert {cert_path}: {e}"))
            })?;
            let key_pem = std::fs::read(Path::new(key_path)).map_err(|e| {
                napi::Error::from_reason(format!("failed to read client key {key_path}: {e}"))
            })?;
            cfg = cfg.identity(Identity::from_pem(cert_pem, key_pem));
        }
        (None, None) => {}
        _ => {
            return Err(napi::Error::from_reason(
                "clientCert and clientKey must be provided together",
            ));
        }
    }
    Ok(cfg)
}

// ---------------------------------------------------------------------------
// Connect options object
// ---------------------------------------------------------------------------

/// TLS options accepted by [`JsModelClient::connect_with_tls`].
#[napi(object)]
pub struct JsModelClientTlsOptions {
    /// Filesystem path to the PEM-encoded CA certificate used to verify
    /// the server.
    pub ca_cert: String,
    /// Optional path to the PEM-encoded client certificate (mTLS). Must
    /// be paired with [`Self::client_key`].
    pub client_cert: Option<String>,
    /// Optional path to the PEM-encoded client private key (mTLS). Must
    /// be paired with [`Self::client_cert`].
    pub client_key: Option<String>,
}

// ---------------------------------------------------------------------------
// JsModelClient
// ---------------------------------------------------------------------------

/// gRPC client for the `BlazenModelServer` service.
///
/// Connect with [`connect`](Self::connect) (plaintext) or
/// [`connectWithTls`](Self::connect_with_tls) (TLS / mTLS), then issue
/// RPCs. This wave exposes only the status RPCs; later waves add load /
/// unload / completions / embeddings / media / blobs.
///
/// ```typescript
/// const client = await ModelClient.connect("http://model-server:50051");
/// const status = await client.status();
/// if (await client.isLoaded("gpt-oss-120b")) {
///   // ...
/// }
/// ```
#[napi(js_name = "ModelClient")]
pub struct JsModelClient {
    /// Stored behind `Arc` so each `#[napi]` method clones a handle
    /// without re-establishing the gRPC channel. `ModelClient` itself is
    /// already cheaply cloneable (it wraps an `Arc<Mutex<Grpc>>`), so the
    /// extra `Arc` is purely for cross-method ergonomics.
    inner: Arc<ModelClient>,
}

#[napi]
#[allow(
    clippy::must_use_candidate,
    clippy::missing_errors_doc,
    clippy::missing_panics_doc,
    clippy::needless_pass_by_value
)]
impl JsModelClient {
    /// Open a plaintext connection to a `BlazenModelServer` at
    /// `endpoint` (e.g. `"http://localhost:50051"`).
    #[napi(factory)]
    pub async fn connect(endpoint: String) -> Result<Self> {
        let client = ModelClient::connect(endpoint)
            .await
            .map_err(controlplane_error_to_napi)?;
        Ok(Self {
            inner: Arc::new(client),
        })
    }

    /// Open a TLS (or mTLS, when `opts.clientCert` + `opts.clientKey`
    /// are supplied) connection to a `BlazenModelServer` at `endpoint`.
    #[napi(factory, js_name = "connectWithTls")]
    pub async fn connect_with_tls(endpoint: String, opts: JsModelClientTlsOptions) -> Result<Self> {
        let tls = build_tls_config(
            &opts.ca_cert,
            opts.client_cert.as_deref(),
            opts.client_key.as_deref(),
        )?;
        let client = ModelClient::connect_with_tls(endpoint, Some(tls))
            .await
            .map_err(controlplane_error_to_napi)?;
        Ok(Self {
            inner: Arc::new(client),
        })
    }

    /// Fetch a snapshot of every registered model on the server.
    ///
    /// `model_id` is currently unused (the server returns every model
    /// either way) but is reserved for a future per-model filter. Pass
    /// `undefined` / omit the argument for the full snapshot.
    ///
    /// Returns a plain JS object with the wire shape of
    /// [`StatusResponse`] (`{ envelopeVersion, models: [...] }`),
    /// produced via `serde_json` so the model + adapter wires serialize
    /// recursively without per-field napi glue.
    #[napi(js_name = "status", ts_args_type = "modelId?: string")]
    pub async fn status(&self, _model_id: Option<String>) -> Result<serde_json::Value> {
        let req = StatusRequest {
            envelope_version: MODEL_ENVELOPE_VERSION,
        };
        let resp: StatusResponse = self
            .inner
            .status(req)
            .await
            .map_err(controlplane_error_to_napi)?;
        serde_json::to_value(resp).map_err(|e| {
            napi::Error::from_reason(format!("failed to serialize StatusResponse: {e}"))
        })
    }

    /// Liveness check for a specific registered model.
    #[napi(js_name = "isLoaded")]
    pub async fn is_loaded(&self, model_id: String) -> Result<bool> {
        let req = IsLoadedRequest {
            envelope_version: MODEL_ENVELOPE_VERSION,
            model_id,
        };
        let resp = self
            .inner
            .is_loaded(req)
            .await
            .map_err(controlplane_error_to_napi)?;
        Ok(resp.loaded)
    }

    /// Issue a `Load` RPC for a previously-registered model.
    ///
    /// `request` is a plain JS object matching the wire shape of
    /// [`LoadRequest`] (`{ envelopeVersion?, modelId }`). The
    /// `envelopeVersion` field is filled in from
    /// [`MODEL_ENVELOPE_VERSION`] when omitted by the caller — pass the
    /// shorthand `{ modelId: "qwen3-7b" }` and the binding will set the
    /// rest. Returns the wire-shaped [`LoadResponse`] as a plain JS
    /// object.
    #[napi(js_name = "load")]
    pub async fn load(&self, request: serde_json::Value) -> Result<serde_json::Value> {
        let mut req: LoadRequest = serde_json::from_value(request)
            .map_err(|e| napi::Error::from_reason(format!("failed to decode LoadRequest: {e}")))?;
        if req.envelope_version == 0 {
            req.envelope_version = MODEL_ENVELOPE_VERSION;
        }
        let resp: LoadResponse = self
            .inner
            .load(req)
            .await
            .map_err(controlplane_error_to_napi)?;
        serde_json::to_value(resp)
            .map_err(|e| napi::Error::from_reason(format!("failed to serialize LoadResponse: {e}")))
    }

    /// Issue an `Unload` RPC to drop a loaded model from memory.
    ///
    /// `request` mirrors [`UnloadRequest`] on the wire
    /// (`{ envelopeVersion?, modelId }`). Returns the wire-shaped
    /// [`UnloadResponse`] as a plain JS object.
    #[napi]
    pub async fn unload(&self, request: serde_json::Value) -> Result<serde_json::Value> {
        let mut req: UnloadRequest = serde_json::from_value(request).map_err(|e| {
            napi::Error::from_reason(format!("failed to decode UnloadRequest: {e}"))
        })?;
        if req.envelope_version == 0 {
            req.envelope_version = MODEL_ENVELOPE_VERSION;
        }
        let resp: UnloadResponse = self
            .inner
            .unload(req)
            .await
            .map_err(controlplane_error_to_napi)?;
        serde_json::to_value(resp).map_err(|e| {
            napi::Error::from_reason(format!("failed to serialize UnloadResponse: {e}"))
        })
    }

    /// Issue a `LoadFromHf` RPC — register-and-load a model from a
    /// Hugging Face Hub repo. Whether the server actually honors the
    /// request depends on it having been built with the `hf-loader`
    /// feature; the client speaks the wire either way.
    ///
    /// `request` matches [`LoadFromHfRequest`] on the wire
    /// (`{ envelopeVersion?, modelId, repo, memoryEstimateBytes?,
    /// backendHint?, ggufFile?, revision?, hfToken?,
    /// extraOptionsJson? }`). Returns the wire-shaped
    /// [`LoadFromHfResponse`] as a plain JS object.
    #[napi(js_name = "loadFromHf")]
    pub async fn load_from_hf(&self, request: serde_json::Value) -> Result<serde_json::Value> {
        let mut req: LoadFromHfRequest = serde_json::from_value(request).map_err(|e| {
            napi::Error::from_reason(format!("failed to decode LoadFromHfRequest: {e}"))
        })?;
        if req.envelope_version == 0 {
            req.envelope_version = MODEL_ENVELOPE_VERSION;
        }
        let resp: LoadFromHfResponse = self
            .inner
            .load_from_hf(req)
            .await
            .map_err(controlplane_error_to_napi)?;
        serde_json::to_value(resp).map_err(|e| {
            napi::Error::from_reason(format!("failed to serialize LoadFromHfResponse: {e}"))
        })
    }

    // ----- Adapters -----

    /// Issue a `LoadAdapter` RPC.
    ///
    /// `request` matches [`LoadAdapterRequest`] on the wire. Returns the
    /// wire-shaped [`LoadAdapterResponse`] as a plain JS object.
    #[napi(js_name = "loadAdapter")]
    pub async fn load_adapter(&self, request: serde_json::Value) -> Result<serde_json::Value> {
        let mut req: LoadAdapterRequest = serde_json::from_value(request).map_err(|e| {
            napi::Error::from_reason(format!("failed to decode LoadAdapterRequest: {e}"))
        })?;
        if req.envelope_version == 0 {
            req.envelope_version = MODEL_ENVELOPE_VERSION;
        }
        let resp: LoadAdapterResponse = self
            .inner
            .load_adapter(req)
            .await
            .map_err(controlplane_error_to_napi)?;
        serde_json::to_value(resp).map_err(|e| {
            napi::Error::from_reason(format!("failed to serialize LoadAdapterResponse: {e}"))
        })
    }

    /// Issue an `UnloadAdapter` RPC.
    ///
    /// `request` matches [`UnloadAdapterRequest`] on the wire. Returns the
    /// wire-shaped [`UnloadAdapterResponse`] as a plain JS object.
    #[napi(js_name = "unloadAdapter")]
    pub async fn unload_adapter(&self, request: serde_json::Value) -> Result<serde_json::Value> {
        let mut req: UnloadAdapterRequest = serde_json::from_value(request).map_err(|e| {
            napi::Error::from_reason(format!("failed to decode UnloadAdapterRequest: {e}"))
        })?;
        if req.envelope_version == 0 {
            req.envelope_version = MODEL_ENVELOPE_VERSION;
        }
        let resp: UnloadAdapterResponse = self
            .inner
            .unload_adapter(req)
            .await
            .map_err(controlplane_error_to_napi)?;
        serde_json::to_value(resp).map_err(|e| {
            napi::Error::from_reason(format!("failed to serialize UnloadAdapterResponse: {e}"))
        })
    }

    /// Issue a `ListAdapters` RPC.
    ///
    /// `request` matches [`ListAdaptersRequest`] on the wire. Returns the
    /// wire-shaped [`ListAdaptersResponse`] as a plain JS object.
    #[napi(js_name = "listAdapters")]
    pub async fn list_adapters(&self, request: serde_json::Value) -> Result<serde_json::Value> {
        let mut req: ListAdaptersRequest = serde_json::from_value(request).map_err(|e| {
            napi::Error::from_reason(format!("failed to decode ListAdaptersRequest: {e}"))
        })?;
        if req.envelope_version == 0 {
            req.envelope_version = MODEL_ENVELOPE_VERSION;
        }
        let resp: ListAdaptersResponse = self
            .inner
            .list_adapters(req)
            .await
            .map_err(controlplane_error_to_napi)?;
        serde_json::to_value(resp).map_err(|e| {
            napi::Error::from_reason(format!("failed to serialize ListAdaptersResponse: {e}"))
        })
    }

    // ----- Inference -----

    /// Issue a `Complete` RPC.
    ///
    /// `request` matches [`CompleteRequest`] on the wire. Returns the
    /// wire-shaped [`CompleteResponse`] as a plain JS object.
    #[napi(js_name = "complete")]
    pub async fn complete(&self, request: serde_json::Value) -> Result<serde_json::Value> {
        let mut req: CompleteRequest = serde_json::from_value(request).map_err(|e| {
            napi::Error::from_reason(format!("failed to decode CompleteRequest: {e}"))
        })?;
        if req.envelope_version == 0 {
            req.envelope_version = MODEL_ENVELOPE_VERSION;
        }
        let resp: CompleteResponse = self
            .inner
            .complete(req)
            .await
            .map_err(controlplane_error_to_napi)?;
        serde_json::to_value(resp).map_err(|e| {
            napi::Error::from_reason(format!("failed to serialize CompleteResponse: {e}"))
        })
    }

    /// Issue an `Embed` RPC.
    ///
    /// `request` matches [`EmbedRequest`] on the wire. Returns the
    /// wire-shaped [`EmbedResponse`] as a plain JS object.
    #[napi(js_name = "embed")]
    pub async fn embed(&self, request: serde_json::Value) -> Result<serde_json::Value> {
        let mut req: EmbedRequest = serde_json::from_value(request)
            .map_err(|e| napi::Error::from_reason(format!("failed to decode EmbedRequest: {e}")))?;
        if req.envelope_version == 0 {
            req.envelope_version = MODEL_ENVELOPE_VERSION;
        }
        let resp: EmbedResponse = self
            .inner
            .embed(req)
            .await
            .map_err(controlplane_error_to_napi)?;
        serde_json::to_value(resp).map_err(|e| {
            napi::Error::from_reason(format!("failed to serialize EmbedResponse: {e}"))
        })
    }

    // ----- Multimodal -----

    /// Issue a `GenerateImage` RPC.
    ///
    /// `request` matches [`GenerateImageRequest`] on the wire. Returns the
    /// wire-shaped [`GenerateImageResponse`] as a plain JS object.
    #[napi(js_name = "generateImage")]
    pub async fn generate_image(&self, request: serde_json::Value) -> Result<serde_json::Value> {
        let mut req: GenerateImageRequest = serde_json::from_value(request).map_err(|e| {
            napi::Error::from_reason(format!("failed to decode GenerateImageRequest: {e}"))
        })?;
        if req.envelope_version == 0 {
            req.envelope_version = MODEL_ENVELOPE_VERSION;
        }
        let resp: GenerateImageResponse = self
            .inner
            .generate_image(req)
            .await
            .map_err(controlplane_error_to_napi)?;
        serde_json::to_value(resp).map_err(|e| {
            napi::Error::from_reason(format!("failed to serialize GenerateImageResponse: {e}"))
        })
    }

    /// Issue a `TextToSpeech` RPC.
    ///
    /// `request` matches [`TextToSpeechRequest`] on the wire. Returns the
    /// wire-shaped [`TextToSpeechResponse`] as a plain JS object.
    #[napi(js_name = "textToSpeech")]
    pub async fn text_to_speech(&self, request: serde_json::Value) -> Result<serde_json::Value> {
        let mut req: TextToSpeechRequest = serde_json::from_value(request).map_err(|e| {
            napi::Error::from_reason(format!("failed to decode TextToSpeechRequest: {e}"))
        })?;
        if req.envelope_version == 0 {
            req.envelope_version = MODEL_ENVELOPE_VERSION;
        }
        let resp: TextToSpeechResponse = self
            .inner
            .text_to_speech(req)
            .await
            .map_err(controlplane_error_to_napi)?;
        serde_json::to_value(resp).map_err(|e| {
            napi::Error::from_reason(format!("failed to serialize TextToSpeechResponse: {e}"))
        })
    }

    /// Issue a `GenerateMusic` RPC.
    ///
    /// `request` matches [`GenerateMusicRequest`] on the wire. Returns the
    /// wire-shaped [`GenerateMusicResponse`] as a plain JS object.
    #[napi(js_name = "generateMusic")]
    pub async fn generate_music(&self, request: serde_json::Value) -> Result<serde_json::Value> {
        let mut req: GenerateMusicRequest = serde_json::from_value(request).map_err(|e| {
            napi::Error::from_reason(format!("failed to decode GenerateMusicRequest: {e}"))
        })?;
        if req.envelope_version == 0 {
            req.envelope_version = MODEL_ENVELOPE_VERSION;
        }
        let resp: GenerateMusicResponse = self
            .inner
            .generate_music(req)
            .await
            .map_err(controlplane_error_to_napi)?;
        serde_json::to_value(resp).map_err(|e| {
            napi::Error::from_reason(format!("failed to serialize GenerateMusicResponse: {e}"))
        })
    }

    /// Issue a `Transcribe` RPC.
    ///
    /// `request` matches [`TranscribeRequest`] on the wire. Returns the
    /// wire-shaped [`TranscribeResponse`] as a plain JS object.
    #[napi(js_name = "transcribe")]
    pub async fn transcribe(&self, request: serde_json::Value) -> Result<serde_json::Value> {
        let mut req: TranscribeRequest = serde_json::from_value(request).map_err(|e| {
            napi::Error::from_reason(format!("failed to decode TranscribeRequest: {e}"))
        })?;
        if req.envelope_version == 0 {
            req.envelope_version = MODEL_ENVELOPE_VERSION;
        }
        let resp: TranscribeResponse = self
            .inner
            .transcribe(req)
            .await
            .map_err(controlplane_error_to_napi)?;
        serde_json::to_value(resp).map_err(|e| {
            napi::Error::from_reason(format!("failed to serialize TranscribeResponse: {e}"))
        })
    }

    /// Issue a `StreamComplete` server-streaming RPC.
    ///
    /// `request` matches [`CompleteRequest`] on the wire (same shape as
    /// the unary `complete` method). Returns a JS
    /// `AsyncIterableIterator` that yields wire-shaped
    /// [`StreamCompleteChunk`](blazen_controlplane::model_protocol::StreamCompleteChunk)
    /// objects (each a plain JS object — `{ kind: "delta", ... }` or
    /// `{ kind: "done", ... }` depending on the variant) until the
    /// server closes the stream.
    ///
    /// The stream is opened lazily on the first `next()` call so the
    /// initial RPC error (if any) surfaces to the consumer rather than
    /// to the synchronous call site.
    ///
    /// Mirrors the lazy-open `AsyncIterableIterator` pattern used by
    /// [`crate::controlplane::client::JsControlPlaneClient::subscribe_run_events`].
    #[napi(
        js_name = "streamComplete",
        ts_args_type = "request: object",
        ts_return_type = "AsyncIterableIterator<object>"
    )]
    pub fn stream_complete<'env>(
        &self,
        env: &'env Env,
        request: serde_json::Value,
    ) -> Result<Object<'env>> {
        let mut req: CompleteRequest = serde_json::from_value(request).map_err(|e| {
            napi::Error::from_reason(format!("failed to decode CompleteRequest: {e}"))
        })?;
        if req.envelope_version == 0 {
            req.envelope_version = MODEL_ENVELOPE_VERSION;
        }
        let client = Arc::clone(&self.inner);
        let fut: OpenChunkStreamFuture = Box::pin(async move {
            let stream = client
                .stream_complete(req)
                .await
                .map_err(controlplane_error_to_napi)?;
            Ok(stream)
        });
        let state: PendingChunkStream = Arc::new(Mutex::new(ChunkStreamState::Pending(Some(fut))));
        build_stream_complete_iterable(env, &state)
    }

    /// Issue an `UploadBlob` client-streaming RPC.
    ///
    /// `chunks` is the pre-collected blob payload split into one or more
    /// `Buffer` (or `Uint8Array`) pieces. The binding wraps them in the
    /// canonical `Start` / `Data*` / `End` frame sequence — callers do
    /// not need to construct envelope frames themselves. `options.blobId`
    /// names the upload (defaults to a freshly-generated UUID-shaped
    /// string when omitted); `options.mime` is forwarded as the
    /// `content_type` hint on the `Start` frame.
    ///
    /// This binding uses the **pre-collected `Vec<Buffer>`** approach
    /// rather than consuming a JS `AsyncIterable`: napi-rs does not yet
    /// surface a first-class JS-async-iterator → Rust-`Stream` adapter,
    /// and a hand-rolled `iterator.next()`-pumping bridge would have run
    /// well past the ~50-line budget called out in the wave plan. The
    /// streaming-over-the-wire shape (multiple postcard frames) is
    /// preserved — only the JS-side ergonomics differ from a true async
    /// iterable. Returns the wire-shaped [`UploadBlobResponse`] as a
    /// plain JS object.
    #[napi(
        js_name = "uploadBlob",
        ts_args_type = "chunks: Array<Buffer | Uint8Array>, options?: { blobId?: string, mime?: string }"
    )]
    pub async fn upload_blob(
        &self,
        chunks: Vec<Buffer>,
        options: Option<serde_json::Value>,
    ) -> Result<serde_json::Value> {
        // Pull blob_id / mime out of the optional options bag.
        let (blob_id, mime) = match options {
            Some(v) => {
                let blob_id = v
                    .get("blobId")
                    .and_then(serde_json::Value::as_str)
                    .map(str::to_owned);
                let mime = v
                    .get("mime")
                    .and_then(serde_json::Value::as_str)
                    .map(str::to_owned);
                (blob_id, mime)
            }
            None => (None, None),
        };
        let blob_id = blob_id.unwrap_or_else(|| format!("blob-{}", uuid::Uuid::new_v4()));
        let total_bytes: u64 = chunks.iter().map(|b| b.len() as u64).sum();

        // Build the frame sequence: Start, Data*, End.
        let (tx, rx) = mpsc::channel::<UploadBlobChunk>(16);
        let send_blob_id = blob_id.clone();
        let send_chunks: Vec<Vec<u8>> = chunks.into_iter().map(|b| b.to_vec()).collect();
        tokio::spawn(async move {
            if tx
                .send(UploadBlobChunk::Start {
                    envelope_version: MODEL_ENVELOPE_VERSION,
                    blob_id: send_blob_id,
                    total_bytes: Some(total_bytes),
                    content_type: mime,
                })
                .await
                .is_err()
            {
                return;
            }
            for bytes in send_chunks {
                if tx
                    .send(UploadBlobChunk::Data {
                        envelope_version: MODEL_ENVELOPE_VERSION,
                        bytes,
                    })
                    .await
                    .is_err()
                {
                    return;
                }
            }
            let _ = tx
                .send(UploadBlobChunk::End {
                    envelope_version: MODEL_ENVELOPE_VERSION,
                })
                .await;
        });
        let resp: UploadBlobResponse = self
            .inner
            .upload_blob(rx)
            .await
            .map_err(controlplane_error_to_napi)?;
        debug_assert_eq!(resp.blob_id, blob_id);
        serde_json::to_value(resp).map_err(|e| {
            napi::Error::from_reason(format!("failed to serialize UploadBlobResponse: {e}"))
        })
    }

    /// Issue a `FetchBlob` server-streaming RPC.
    ///
    /// `request` matches [`FetchBlobRequest`] on the wire
    /// (`{ envelopeVersion?, blobId, offset?, chunkSize? }`). Returns a
    /// JS `AsyncIterableIterator<Buffer>` that yields the blob body in
    /// order — only the `Data` frames are surfaced as `Buffer` values;
    /// the `Start` / `End` envelope frames are consumed transparently
    /// (`Start` is dropped so callers see only bytes, `End` terminates
    /// iteration). The stream is opened lazily on the first `next()`
    /// call so the initial RPC error (if any) surfaces to the consumer.
    ///
    /// Mirrors the lazy-open pattern used by [`Self::stream_complete`].
    #[napi(
        js_name = "fetchBlob",
        ts_args_type = "request: object",
        ts_return_type = "AsyncIterableIterator<Buffer>"
    )]
    pub fn fetch_blob<'env>(
        &self,
        env: &'env Env,
        request: serde_json::Value,
    ) -> Result<Object<'env>> {
        let mut req: FetchBlobRequest = serde_json::from_value(request).map_err(|e| {
            napi::Error::from_reason(format!("failed to decode FetchBlobRequest: {e}"))
        })?;
        if req.envelope_version == 0 {
            req.envelope_version = MODEL_ENVELOPE_VERSION;
        }
        let client = Arc::clone(&self.inner);
        let fut: OpenFetchBlobStreamFuture = Box::pin(async move {
            let stream = client
                .fetch_blob(req)
                .await
                .map_err(controlplane_error_to_napi)?;
            Ok(stream)
        });
        let state: PendingFetchBlobStream =
            Arc::new(Mutex::new(FetchBlobStreamState::Pending(Some(fut))));
        build_fetch_blob_iterable(env, &state)
    }
}

// ---------------------------------------------------------------------------
// streamComplete async-iterable bridge
// ---------------------------------------------------------------------------

/// Future that lazily opens the `StreamComplete` RPC on first `next()`
/// call. Mirrors the run-event stream pattern in
/// [`crate::controlplane::client`] but specialised for the chunk
/// stream (which is already `'static`, so no lifetime cast is needed).
type OpenChunkStreamFuture =
    Pin<Box<dyn std::future::Future<Output = Result<StreamCompleteRecvStream>> + Send>>;

enum ChunkStreamState {
    Pending(Option<OpenChunkStreamFuture>),
    Active(StreamCompleteRecvStream),
    Closed,
}

type PendingChunkStream = Arc<Mutex<ChunkStreamState>>;

/// Build the JS-side `AsyncIterableIterator<object>` object backed by a
/// [`StreamCompleteRecvStream`]. Each yielded value is the chunk
/// re-encoded as a plain JS object via `serde_json`.
fn build_stream_complete_iterable<'env>(
    env: &'env Env,
    state: &PendingChunkStream,
) -> Result<Object<'env>> {
    let mut iter_obj = Object::new(env)?;

    let next_state = Arc::clone(state);
    let next_fn =
        env.create_function_from_closure::<(), napi::sys::napi_value, _>("next", move |ctx| {
            let state = Arc::clone(&next_state);
            let promise = ctx.env.spawn_future_with_callback(
                async move { pull_next_chunk(state).await },
                |env, val: Option<serde_json::Value>| {
                    let mut obj = Object::new(env)?;
                    if let Some(chunk) = val {
                        obj.set("value", chunk)?;
                        obj.set("done", false)?;
                    } else {
                        obj.set("value", ())?;
                        obj.set("done", true)?;
                    }
                    Ok(obj)
                },
            )?;
            Ok(promise.raw())
        })?;
    iter_obj.set("next", next_fn)?;

    // `[Symbol.asyncIterator]()` returns the iterator itself.
    let iter_raw_value = iter_obj.value();
    let self_returning_fn = env.create_function_from_closure::<(), napi::sys::napi_value, _>(
        "[Symbol.asyncIterator]",
        move |_ctx| Ok(iter_raw_value.value),
    )?;
    let global = env.get_global()?;
    let symbol_obj = global.get_named_property_unchecked::<Object>("Symbol")?;
    let async_iterator_symbol =
        symbol_obj.get_named_property_unchecked::<Unknown>("asyncIterator")?;
    iter_obj.set_property(async_iterator_symbol, self_returning_fn)?;

    Ok(iter_obj)
}

/// Pull the next [`StreamCompleteChunk`](blazen_controlplane::model_protocol::StreamCompleteChunk)
/// from the wrapped stream as a `serde_json::Value`, opening the stream
/// lazily on the first call.
async fn pull_next_chunk(state: PendingChunkStream) -> Result<Option<serde_json::Value>> {
    // Stage 1: open the stream if still pending. Done outside the
    // long-lived lock so concurrent `next()` callers serialize cleanly.
    let pending_future = {
        let mut guard = state.lock().await;
        match &mut *guard {
            ChunkStreamState::Pending(slot) => slot.take(),
            ChunkStreamState::Active(_) | ChunkStreamState::Closed => None,
        }
    };
    if let Some(fut) = pending_future {
        match fut.await {
            Ok(stream) => {
                let mut guard = state.lock().await;
                *guard = ChunkStreamState::Active(stream);
            }
            Err(e) => {
                let mut guard = state.lock().await;
                *guard = ChunkStreamState::Closed;
                return Err(e);
            }
        }
    }

    // Stage 2: pull a single chunk from the active stream.
    let mut guard = state.lock().await;
    match &mut *guard {
        ChunkStreamState::Active(stream) => match stream.next().await {
            Some(Ok(chunk)) => {
                let value = serde_json::to_value(&chunk).map_err(|e| {
                    napi::Error::from_reason(format!(
                        "failed to serialize StreamCompleteChunk: {e}"
                    ))
                })?;
                Ok(Some(value))
            }
            Some(Err(e)) => {
                *guard = ChunkStreamState::Closed;
                Err(controlplane_error_to_napi(e))
            }
            None => {
                *guard = ChunkStreamState::Closed;
                Ok(None)
            }
        },
        ChunkStreamState::Closed => Ok(None),
        ChunkStreamState::Pending(_) => {
            // Unreachable — Stage 1 either transitioned to Active or
            // returned the open error.
            *guard = ChunkStreamState::Closed;
            Err(napi::Error::from_reason(
                "stream state inconsistent: still pending after open future",
            ))
        }
    }
}

// ---------------------------------------------------------------------------
// fetchBlob async-iterable bridge
// ---------------------------------------------------------------------------

/// Future that lazily opens the `FetchBlob` RPC on first `next()` call.
type OpenFetchBlobStreamFuture =
    Pin<Box<dyn std::future::Future<Output = Result<FetchBlobRecvStream>> + Send>>;

enum FetchBlobStreamState {
    Pending(Option<OpenFetchBlobStreamFuture>),
    Active(FetchBlobRecvStream),
    Closed,
}

type PendingFetchBlobStream = Arc<Mutex<FetchBlobStreamState>>;

/// Build the JS-side `AsyncIterableIterator<Buffer>` object backed by a
/// [`FetchBlobRecvStream`]. Only the `Data` frames surface to the JS
/// consumer; `Start` is consumed transparently on first poll and `End`
/// terminates iteration.
fn build_fetch_blob_iterable<'env>(
    env: &'env Env,
    state: &PendingFetchBlobStream,
) -> Result<Object<'env>> {
    let mut iter_obj = Object::new(env)?;

    let next_state = Arc::clone(state);
    let next_fn =
        env.create_function_from_closure::<(), napi::sys::napi_value, _>("next", move |ctx| {
            let state = Arc::clone(&next_state);
            let promise = ctx.env.spawn_future_with_callback(
                async move { pull_next_fetch_blob_data(state).await },
                |env, val: Option<Vec<u8>>| {
                    let mut obj = Object::new(env)?;
                    if let Some(bytes) = val {
                        obj.set("value", Buffer::from(bytes))?;
                        obj.set("done", false)?;
                    } else {
                        obj.set("value", ())?;
                        obj.set("done", true)?;
                    }
                    Ok(obj)
                },
            )?;
            Ok(promise.raw())
        })?;
    iter_obj.set("next", next_fn)?;

    // `[Symbol.asyncIterator]()` returns the iterator itself.
    let iter_raw_value = iter_obj.value();
    let self_returning_fn = env.create_function_from_closure::<(), napi::sys::napi_value, _>(
        "[Symbol.asyncIterator]",
        move |_ctx| Ok(iter_raw_value.value),
    )?;
    let global = env.get_global()?;
    let symbol_obj = global.get_named_property_unchecked::<Object>("Symbol")?;
    let async_iterator_symbol =
        symbol_obj.get_named_property_unchecked::<Unknown>("asyncIterator")?;
    iter_obj.set_property(async_iterator_symbol, self_returning_fn)?;

    Ok(iter_obj)
}

/// Pull the next blob body chunk from the wrapped stream as a
/// `Vec<u8>`, opening the stream lazily on the first call and
/// transparently consuming envelope (`Start` / `End`) frames.
async fn pull_next_fetch_blob_data(state: PendingFetchBlobStream) -> Result<Option<Vec<u8>>> {
    // Stage 1: open the stream if still pending.
    let pending_future = {
        let mut guard = state.lock().await;
        match &mut *guard {
            FetchBlobStreamState::Pending(slot) => slot.take(),
            FetchBlobStreamState::Active(_) | FetchBlobStreamState::Closed => None,
        }
    };
    if let Some(fut) = pending_future {
        match fut.await {
            Ok(stream) => {
                let mut guard = state.lock().await;
                *guard = FetchBlobStreamState::Active(stream);
            }
            Err(e) => {
                let mut guard = state.lock().await;
                *guard = FetchBlobStreamState::Closed;
                return Err(e);
            }
        }
    }

    // Stage 2: pull frames until we surface a Data frame, hit End / EOF,
    // or encounter an error. Start frames are dropped silently.
    let mut guard = state.lock().await;
    loop {
        match &mut *guard {
            FetchBlobStreamState::Active(stream) => match stream.next().await {
                Some(Ok(FetchBlobChunk::Start { .. })) => {}
                Some(Ok(FetchBlobChunk::Data { bytes, .. })) => return Ok(Some(bytes)),
                Some(Ok(FetchBlobChunk::End { .. })) | None => {
                    *guard = FetchBlobStreamState::Closed;
                    return Ok(None);
                }
                Some(Err(e)) => {
                    *guard = FetchBlobStreamState::Closed;
                    return Err(controlplane_error_to_napi(e));
                }
            },
            FetchBlobStreamState::Closed => return Ok(None),
            FetchBlobStreamState::Pending(_) => {
                *guard = FetchBlobStreamState::Closed;
                return Err(napi::Error::from_reason(
                    "stream state inconsistent: still pending after open future",
                ));
            }
        }
    }
}
