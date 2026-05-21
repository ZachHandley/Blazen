//! Handler functions for the `BlazenModelServer` gRPC service.
//!
//! Each handler:
//!
//! 1. Decodes the postcard payload into a typed request from
//!    [`crate::model_protocol`].
//! 2. Validates the envelope version.
//! 3. Calls the matching method on the supplied [`ManagerHandle`].
//! 4. Encodes the result (a [`RpcResult<T>`]) back into a
//!    `PostcardResponse`.
//!
//! The host crate (typically `blazen-manager` or whatever crate
//! actually owns a `ModelManager`) implements [`ManagerHandle`] and
//! plugs its trait object into [`ModelServerState`]. This keeps the
//! dependency direction one-way: blazen-controlplane does NOT depend on
//! blazen-llm / blazen-manager — bindings and host binaries depend on
//! blazen-controlplane, never the reverse.

use std::sync::Arc;

use async_trait::async_trait;
use futures_util::{Stream, StreamExt};
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;
use tonic::{Response, Status, Streaming};

use crate::model_pb::{PostcardRequest, PostcardResponse};
use crate::model_protocol::{
    CompleteRequest, CompleteResponse, EmbedRequest, EmbedResponse, FetchBlobChunk,
    FetchBlobRequest, GenerateImageRequest, GenerateImageResponse, GenerateMusicRequest,
    GenerateMusicResponse, IsLoadedRequest, IsLoadedResponse, ListAdaptersRequest,
    ListAdaptersResponse, LoadAdapterRequest, LoadAdapterResponse, LoadFromHfRequest,
    LoadFromHfResponse, LoadRequest, LoadResponse, RpcError, RpcResult, StatusRequest,
    StatusResponse, StreamCompleteChunk, TextToSpeechRequest, TextToSpeechResponse,
    TranscribeRequest, TranscribeResponse, UnloadAdapterRequest, UnloadAdapterResponse,
    UnloadRequest, UnloadResponse, UploadBlobChunk, UploadBlobResponse,
    validate_model_envelope_version,
};

/// Pinned boxed `Stream` of decoded `StreamCompleteChunk`s — what
/// `ManagerHandle::stream_complete` produces. The server module wraps
/// each frame in postcard + `PostcardResponse` before pushing it down
/// the gRPC stream.
pub type StreamCompleteStream =
    std::pin::Pin<Box<dyn Stream<Item = Result<StreamCompleteChunk, RpcError>> + Send>>;

/// Pinned boxed `Stream` of decoded `FetchBlobChunk`s, matching the
/// shape of [`StreamCompleteStream`].
pub type FetchBlobStream =
    std::pin::Pin<Box<dyn Stream<Item = Result<FetchBlobChunk, RpcError>> + Send>>;

/// Host-supplied interface that the model-server handlers call into.
///
/// Implementors live in the host crate (typically `blazen-manager`) and
/// wrap a real `ModelManager`. This crate intentionally knows nothing
/// about `ModelManager`'s concrete shape — the dependency direction is
/// `host -> controlplane`, never the reverse.
///
/// Every method takes typed wire structs as input and returns wire
/// structs (wrapped in [`Result<_, RpcError>`]) as output. The
/// envelope-version check has already passed by the time the handler
/// reaches the trait method, but implementors must still set
/// `envelope_version = MODEL_ENVELOPE_VERSION` on whatever response
/// they construct.
#[async_trait]
pub trait ManagerHandle: Send + Sync {
    /// Load a registered model into memory. See `ModelManager::load`.
    async fn load(&self, req: LoadRequest) -> Result<LoadResponse, RpcError>;

    /// Drop a loaded model from memory. See `ModelManager::unload`.
    async fn unload(&self, req: UnloadRequest) -> Result<UnloadResponse, RpcError>;

    /// Liveness check. See `ModelManager::is_loaded`.
    async fn is_loaded(&self, req: IsLoadedRequest) -> Result<IsLoadedResponse, RpcError>;

    /// Snapshot of every registered model. See `ModelManager::status`.
    async fn status(&self, req: StatusRequest) -> Result<StatusResponse, RpcError>;

    /// Register-and-load from a Hugging Face Hub repo. See
    /// `ModelManager::load_from_hf`.
    async fn load_from_hf(&self, req: LoadFromHfRequest) -> Result<LoadFromHfResponse, RpcError>;

    /// Mount a `PEFT`-format `LoRA` adapter. See `ModelManager::load_adapter`.
    async fn load_adapter(&self, req: LoadAdapterRequest) -> Result<LoadAdapterResponse, RpcError>;

    /// Remove a previously-mounted adapter. See `ModelManager::unload_adapter`.
    async fn unload_adapter(
        &self,
        req: UnloadAdapterRequest,
    ) -> Result<UnloadAdapterResponse, RpcError>;

    /// List currently-mounted adapters. See `ModelManager::list_adapters`.
    async fn list_adapters(
        &self,
        req: ListAdaptersRequest,
    ) -> Result<ListAdaptersResponse, RpcError>;

    /// Single-shot chat completion.
    async fn complete(&self, req: CompleteRequest) -> Result<CompleteResponse, RpcError>;

    /// Server-streamed chat completion. Implementors return a stream of
    /// [`StreamCompleteChunk`]s; the wire handler postcard-encodes each
    /// one as it arrives.
    async fn stream_complete(&self, req: CompleteRequest)
    -> Result<StreamCompleteStream, RpcError>;

    /// Batched embedding.
    async fn embed(&self, req: EmbedRequest) -> Result<EmbedResponse, RpcError>;

    /// Image generation.
    async fn generate_image(
        &self,
        req: GenerateImageRequest,
    ) -> Result<GenerateImageResponse, RpcError>;

    /// Text-to-speech.
    async fn text_to_speech(
        &self,
        req: TextToSpeechRequest,
    ) -> Result<TextToSpeechResponse, RpcError>;

    /// Music generation.
    async fn generate_music(
        &self,
        req: GenerateMusicRequest,
    ) -> Result<GenerateMusicResponse, RpcError>;

    /// Audio transcription.
    async fn transcribe(&self, req: TranscribeRequest) -> Result<TranscribeResponse, RpcError>;

    /// Client-streamed blob upload. The handler decodes the
    /// `UploadBlobChunk` frames as they arrive and forwards them via
    /// the `chunks` channel. Implementor returns once it has consumed
    /// every chunk (channel closed) and is ready to acknowledge.
    async fn upload_blob(
        &self,
        chunks: mpsc::Receiver<UploadBlobChunk>,
    ) -> Result<UploadBlobResponse, RpcError>;

    /// Server-streamed blob fetch.
    async fn fetch_blob(&self, req: FetchBlobRequest) -> Result<FetchBlobStream, RpcError>;
}

/// Shared state for the model server. Today this is only the trait
/// object — the type exists so we can grow the state struct later
/// (auth, metrics, broadcast bus, …) without breaking callers.
#[derive(Clone)]
pub struct ModelServerState {
    pub handle: Arc<dyn ManagerHandle>,
}

impl ModelServerState {
    /// Build a server state around the given host handle.
    #[must_use]
    pub fn new(handle: Arc<dyn ManagerHandle>) -> Self {
        Self { handle }
    }
}

// ---------------------------------------------------------------------------
// Unary handlers
// ---------------------------------------------------------------------------

/// Handle the `Load` RPC.
///
/// # Errors
/// Returns a tonic `Status::invalid_argument` if the postcard payload
/// fails to decode, or `Status::internal` if the response cannot be
/// encoded. Model-layer failures travel inside the postcard response as
/// `RpcResult::Err(RpcError)` rather than as a `Status`.
pub async fn handle_load(
    state: &ModelServerState,
    request: PostcardRequest,
) -> Result<Response<PostcardResponse>, Status> {
    let req: LoadRequest = decode(&request.postcard_payload, "LoadRequest")?;
    if let Err(e) = validate_model_envelope_version(req.envelope_version) {
        return encode_resp(&RpcResult::<LoadResponse>::Err(e));
    }
    let result: RpcResult<LoadResponse> = state.handle.load(req).await.into();
    encode_resp(&result)
}

/// Handle the `Unload` RPC.
///
/// # Errors
/// See [`handle_load`].
pub async fn handle_unload(
    state: &ModelServerState,
    request: PostcardRequest,
) -> Result<Response<PostcardResponse>, Status> {
    let req: UnloadRequest = decode(&request.postcard_payload, "UnloadRequest")?;
    if let Err(e) = validate_model_envelope_version(req.envelope_version) {
        return encode_resp(&RpcResult::<UnloadResponse>::Err(e));
    }
    let result: RpcResult<UnloadResponse> = state.handle.unload(req).await.into();
    encode_resp(&result)
}

/// Handle the `IsLoaded` RPC.
///
/// # Errors
/// See [`handle_load`].
pub async fn handle_is_loaded(
    state: &ModelServerState,
    request: PostcardRequest,
) -> Result<Response<PostcardResponse>, Status> {
    let req: IsLoadedRequest = decode(&request.postcard_payload, "IsLoadedRequest")?;
    if let Err(e) = validate_model_envelope_version(req.envelope_version) {
        return encode_resp(&RpcResult::<IsLoadedResponse>::Err(e));
    }
    let result: RpcResult<IsLoadedResponse> = state.handle.is_loaded(req).await.into();
    encode_resp(&result)
}

/// Handle the `Status` RPC.
///
/// # Errors
/// See [`handle_load`].
pub async fn handle_status(
    state: &ModelServerState,
    request: PostcardRequest,
) -> Result<Response<PostcardResponse>, Status> {
    let req: StatusRequest = decode(&request.postcard_payload, "StatusRequest")?;
    if let Err(e) = validate_model_envelope_version(req.envelope_version) {
        return encode_resp(&RpcResult::<StatusResponse>::Err(e));
    }
    let result: RpcResult<StatusResponse> = state.handle.status(req).await.into();
    encode_resp(&result)
}

/// Handle the `LoadFromHf` RPC.
///
/// # Errors
/// See [`handle_load`].
pub async fn handle_load_from_hf(
    state: &ModelServerState,
    request: PostcardRequest,
) -> Result<Response<PostcardResponse>, Status> {
    let req: LoadFromHfRequest = decode(&request.postcard_payload, "LoadFromHfRequest")?;
    if let Err(e) = validate_model_envelope_version(req.envelope_version) {
        return encode_resp(&RpcResult::<LoadFromHfResponse>::Err(e));
    }
    let result: RpcResult<LoadFromHfResponse> = state.handle.load_from_hf(req).await.into();
    encode_resp(&result)
}

/// Handle the `LoadAdapter` RPC.
///
/// # Errors
/// See [`handle_load`].
pub async fn handle_load_adapter(
    state: &ModelServerState,
    request: PostcardRequest,
) -> Result<Response<PostcardResponse>, Status> {
    let req: LoadAdapterRequest = decode(&request.postcard_payload, "LoadAdapterRequest")?;
    if let Err(e) = validate_model_envelope_version(req.envelope_version) {
        return encode_resp(&RpcResult::<LoadAdapterResponse>::Err(e));
    }
    let result: RpcResult<LoadAdapterResponse> = state.handle.load_adapter(req).await.into();
    encode_resp(&result)
}

/// Handle the `UnloadAdapter` RPC.
///
/// # Errors
/// See [`handle_load`].
pub async fn handle_unload_adapter(
    state: &ModelServerState,
    request: PostcardRequest,
) -> Result<Response<PostcardResponse>, Status> {
    let req: UnloadAdapterRequest = decode(&request.postcard_payload, "UnloadAdapterRequest")?;
    if let Err(e) = validate_model_envelope_version(req.envelope_version) {
        return encode_resp(&RpcResult::<UnloadAdapterResponse>::Err(e));
    }
    let result: RpcResult<UnloadAdapterResponse> = state.handle.unload_adapter(req).await.into();
    encode_resp(&result)
}

/// Handle the `ListAdapters` RPC.
///
/// # Errors
/// See [`handle_load`].
pub async fn handle_list_adapters(
    state: &ModelServerState,
    request: PostcardRequest,
) -> Result<Response<PostcardResponse>, Status> {
    let req: ListAdaptersRequest = decode(&request.postcard_payload, "ListAdaptersRequest")?;
    if let Err(e) = validate_model_envelope_version(req.envelope_version) {
        return encode_resp(&RpcResult::<ListAdaptersResponse>::Err(e));
    }
    let result: RpcResult<ListAdaptersResponse> = state.handle.list_adapters(req).await.into();
    encode_resp(&result)
}

/// Handle the unary `Complete` RPC.
///
/// # Errors
/// See [`handle_load`].
pub async fn handle_complete(
    state: &ModelServerState,
    request: PostcardRequest,
) -> Result<Response<PostcardResponse>, Status> {
    let req: CompleteRequest = decode(&request.postcard_payload, "CompleteRequest")?;
    if let Err(e) = validate_model_envelope_version(req.envelope_version) {
        return encode_resp(&RpcResult::<CompleteResponse>::Err(e));
    }
    let result: RpcResult<CompleteResponse> = state.handle.complete(req).await.into();
    encode_resp(&result)
}

/// Handle the `Embed` RPC.
///
/// # Errors
/// See [`handle_load`].
pub async fn handle_embed(
    state: &ModelServerState,
    request: PostcardRequest,
) -> Result<Response<PostcardResponse>, Status> {
    let req: EmbedRequest = decode(&request.postcard_payload, "EmbedRequest")?;
    if let Err(e) = validate_model_envelope_version(req.envelope_version) {
        return encode_resp(&RpcResult::<EmbedResponse>::Err(e));
    }
    let result: RpcResult<EmbedResponse> = state.handle.embed(req).await.into();
    encode_resp(&result)
}

/// Handle the `GenerateImage` RPC.
///
/// # Errors
/// See [`handle_load`].
pub async fn handle_generate_image(
    state: &ModelServerState,
    request: PostcardRequest,
) -> Result<Response<PostcardResponse>, Status> {
    let req: GenerateImageRequest = decode(&request.postcard_payload, "GenerateImageRequest")?;
    if let Err(e) = validate_model_envelope_version(req.envelope_version) {
        return encode_resp(&RpcResult::<GenerateImageResponse>::Err(e));
    }
    let result: RpcResult<GenerateImageResponse> = state.handle.generate_image(req).await.into();
    encode_resp(&result)
}

/// Handle the `TextToSpeech` RPC.
///
/// # Errors
/// See [`handle_load`].
pub async fn handle_text_to_speech(
    state: &ModelServerState,
    request: PostcardRequest,
) -> Result<Response<PostcardResponse>, Status> {
    let req: TextToSpeechRequest = decode(&request.postcard_payload, "TextToSpeechRequest")?;
    if let Err(e) = validate_model_envelope_version(req.envelope_version) {
        return encode_resp(&RpcResult::<TextToSpeechResponse>::Err(e));
    }
    let result: RpcResult<TextToSpeechResponse> = state.handle.text_to_speech(req).await.into();
    encode_resp(&result)
}

/// Handle the `GenerateMusic` RPC.
///
/// # Errors
/// See [`handle_load`].
pub async fn handle_generate_music(
    state: &ModelServerState,
    request: PostcardRequest,
) -> Result<Response<PostcardResponse>, Status> {
    let req: GenerateMusicRequest = decode(&request.postcard_payload, "GenerateMusicRequest")?;
    if let Err(e) = validate_model_envelope_version(req.envelope_version) {
        return encode_resp(&RpcResult::<GenerateMusicResponse>::Err(e));
    }
    let result: RpcResult<GenerateMusicResponse> = state.handle.generate_music(req).await.into();
    encode_resp(&result)
}

/// Handle the `Transcribe` RPC.
///
/// # Errors
/// See [`handle_load`].
pub async fn handle_transcribe(
    state: &ModelServerState,
    request: PostcardRequest,
) -> Result<Response<PostcardResponse>, Status> {
    let req: TranscribeRequest = decode(&request.postcard_payload, "TranscribeRequest")?;
    if let Err(e) = validate_model_envelope_version(req.envelope_version) {
        return encode_resp(&RpcResult::<TranscribeResponse>::Err(e));
    }
    let result: RpcResult<TranscribeResponse> = state.handle.transcribe(req).await.into();
    encode_resp(&result)
}

// ---------------------------------------------------------------------------
// Streaming handlers
// ---------------------------------------------------------------------------

/// Type returned to tonic for every server-streaming RPC on this
/// service.
pub type PostcardOutStream =
    std::pin::Pin<Box<dyn Stream<Item = Result<PostcardResponse, Status>> + Send>>;

/// Handle the `StreamComplete` server-streaming RPC. Builds an mpsc
/// pump so the handler's stream task is decoupled from tonic's send
/// loop.
///
/// # Errors
/// Returns `Status::invalid_argument` if the initial postcard payload
/// fails to decode. Per-frame errors travel as
/// `RpcResult::Err(RpcError)` inside the postcard payload.
pub async fn handle_stream_complete(
    state: ModelServerState,
    request: PostcardRequest,
) -> Result<Response<PostcardOutStream>, Status> {
    let req: CompleteRequest = decode(&request.postcard_payload, "CompleteRequest")?;
    if let Err(e) = validate_model_envelope_version(req.envelope_version) {
        let (tx, rx) = mpsc::channel::<Result<PostcardResponse, Status>>(1);
        let payload = postcard::to_allocvec(&RpcResult::<StreamCompleteChunk>::Err(e))
            .map_err(|e| Status::internal(format!("encode response: {e}")))?;
        let _ = tx
            .send(Ok(PostcardResponse {
                postcard_payload: payload,
            }))
            .await;
        let stream: PostcardOutStream = Box::pin(ReceiverStream::new(rx));
        return Ok(Response::new(stream));
    }

    let inner_stream = match state.handle.stream_complete(req).await {
        Ok(s) => s,
        Err(rpc_err) => {
            let (tx, rx) = mpsc::channel::<Result<PostcardResponse, Status>>(1);
            let payload = postcard::to_allocvec(&RpcResult::<StreamCompleteChunk>::Err(rpc_err))
                .map_err(|e| Status::internal(format!("encode response: {e}")))?;
            let _ = tx
                .send(Ok(PostcardResponse {
                    postcard_payload: payload,
                }))
                .await;
            let stream: PostcardOutStream = Box::pin(ReceiverStream::new(rx));
            return Ok(Response::new(stream));
        }
    };

    let (tx, rx) = mpsc::channel::<Result<PostcardResponse, Status>>(16);
    tokio::spawn(async move {
        let mut inner = inner_stream;
        while let Some(frame) = inner.next().await {
            let result: RpcResult<StreamCompleteChunk> = frame.into();
            let payload = match postcard::to_allocvec(&result) {
                Ok(b) => b,
                Err(e) => {
                    let _ = tx
                        .send(Err(Status::internal(format!("encode frame: {e}"))))
                        .await;
                    return;
                }
            };
            if tx
                .send(Ok(PostcardResponse {
                    postcard_payload: payload,
                }))
                .await
                .is_err()
            {
                return;
            }
        }
    });

    let stream: PostcardOutStream = Box::pin(ReceiverStream::new(rx));
    Ok(Response::new(stream))
}

/// Handle the `UploadBlob` client-streaming RPC. Decodes each incoming
/// frame into an [`UploadBlobChunk`] and forwards via the channel.
///
/// # Errors
/// Returns `Status::invalid_argument` for decode failures on individual
/// frames. Application-level failures travel back as
/// `RpcResult::Err(RpcError)`.
pub async fn handle_upload_blob(
    state: ModelServerState,
    mut request: Streaming<PostcardRequest>,
) -> Result<Response<PostcardResponse>, Status> {
    let (tx, rx) = mpsc::channel::<UploadBlobChunk>(16);

    // Spawn a forwarder so we can call into the trait method
    // concurrently with reading. The trait method is responsible for
    // draining the channel; it returns once we close `tx`.
    let handle = state.handle.clone();
    let join = tokio::spawn(async move { handle.upload_blob(rx).await });

    while let Some(frame) = request.message().await? {
        let chunk: UploadBlobChunk = postcard::from_bytes(&frame.postcard_payload)
            .map_err(|e| Status::invalid_argument(format!("decode UploadBlobChunk: {e}")))?;
        if let Err(e) = validate_model_envelope_version(envelope_of_upload(&chunk)) {
            drop(tx);
            return encode_resp(&RpcResult::<UploadBlobResponse>::Err(e));
        }
        if tx.send(chunk).await.is_err() {
            // Receiver hung up — bail out with whatever the trait
            // method ends up returning.
            break;
        }
    }
    drop(tx);

    let result = join.await.map_err(|e| {
        Status::internal(format!(
            "upload_blob handler panicked or was cancelled: {e}"
        ))
    })?;
    let wire: RpcResult<UploadBlobResponse> = result.into();
    encode_resp(&wire)
}

/// Handle the `FetchBlob` server-streaming RPC.
///
/// # Errors
/// See [`handle_stream_complete`].
pub async fn handle_fetch_blob(
    state: ModelServerState,
    request: PostcardRequest,
) -> Result<Response<PostcardOutStream>, Status> {
    let req: FetchBlobRequest = decode(&request.postcard_payload, "FetchBlobRequest")?;
    if let Err(e) = validate_model_envelope_version(req.envelope_version) {
        let (tx, rx) = mpsc::channel::<Result<PostcardResponse, Status>>(1);
        let payload = postcard::to_allocvec(&RpcResult::<FetchBlobChunk>::Err(e))
            .map_err(|e| Status::internal(format!("encode response: {e}")))?;
        let _ = tx
            .send(Ok(PostcardResponse {
                postcard_payload: payload,
            }))
            .await;
        let stream: PostcardOutStream = Box::pin(ReceiverStream::new(rx));
        return Ok(Response::new(stream));
    }

    let inner_stream = match state.handle.fetch_blob(req).await {
        Ok(s) => s,
        Err(rpc_err) => {
            let (tx, rx) = mpsc::channel::<Result<PostcardResponse, Status>>(1);
            let payload = postcard::to_allocvec(&RpcResult::<FetchBlobChunk>::Err(rpc_err))
                .map_err(|e| Status::internal(format!("encode response: {e}")))?;
            let _ = tx
                .send(Ok(PostcardResponse {
                    postcard_payload: payload,
                }))
                .await;
            let stream: PostcardOutStream = Box::pin(ReceiverStream::new(rx));
            return Ok(Response::new(stream));
        }
    };

    let (tx, rx) = mpsc::channel::<Result<PostcardResponse, Status>>(16);
    tokio::spawn(async move {
        let mut inner = inner_stream;
        while let Some(frame) = inner.next().await {
            let result: RpcResult<FetchBlobChunk> = frame.into();
            let payload = match postcard::to_allocvec(&result) {
                Ok(b) => b,
                Err(e) => {
                    let _ = tx
                        .send(Err(Status::internal(format!("encode frame: {e}"))))
                        .await;
                    return;
                }
            };
            if tx
                .send(Ok(PostcardResponse {
                    postcard_payload: payload,
                }))
                .await
                .is_err()
            {
                return;
            }
        }
    });

    let stream: PostcardOutStream = Box::pin(ReceiverStream::new(rx));
    Ok(Response::new(stream))
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn decode<T: serde::de::DeserializeOwned>(bytes: &[u8], what: &str) -> Result<T, Status> {
    postcard::from_bytes(bytes).map_err(|e| Status::invalid_argument(format!("decode {what}: {e}")))
}

fn encode_resp<T: serde::Serialize>(value: &T) -> Result<Response<PostcardResponse>, Status> {
    let bytes = postcard::to_allocvec(value)
        .map_err(|e| Status::internal(format!("encode response: {e}")))?;
    Ok(Response::new(PostcardResponse {
        postcard_payload: bytes,
    }))
}

fn envelope_of_upload(chunk: &UploadBlobChunk) -> u32 {
    match chunk {
        UploadBlobChunk::Start {
            envelope_version, ..
        }
        | UploadBlobChunk::Data {
            envelope_version, ..
        }
        | UploadBlobChunk::End { envelope_version } => *envelope_version,
    }
}

// ---------------------------------------------------------------------------
// Mock handle + tests
// ---------------------------------------------------------------------------

#[cfg(test)]
pub(crate) mod test_support {
    //! Mock `ManagerHandle` used by both this module's unit tests and
    //! the higher-level integration tests in `model_service.rs`.

    use super::*;
    use crate::model_protocol::MODEL_ENVELOPE_VERSION;
    use std::sync::Mutex;

    /// Minimal in-memory `ManagerHandle` that records every call and
    /// returns canned responses. Stream-style methods produce a small
    /// fixed sequence so streaming end-to-end tests can assert on the
    /// shape.
    pub struct MockManagerHandle {
        pub calls: Mutex<Vec<String>>,
    }

    impl MockManagerHandle {
        pub fn new() -> Arc<Self> {
            Arc::new(Self {
                calls: Mutex::new(Vec::new()),
            })
        }
        fn record(&self, name: &str) {
            self.calls.lock().unwrap().push(name.to_owned());
        }
    }

    #[async_trait]
    impl ManagerHandle for MockManagerHandle {
        async fn load(&self, _req: LoadRequest) -> Result<LoadResponse, RpcError> {
            self.record("load");
            Ok(LoadResponse {
                envelope_version: MODEL_ENVELOPE_VERSION,
            })
        }
        async fn unload(&self, _req: UnloadRequest) -> Result<UnloadResponse, RpcError> {
            self.record("unload");
            Ok(UnloadResponse {
                envelope_version: MODEL_ENVELOPE_VERSION,
            })
        }
        async fn is_loaded(&self, _req: IsLoadedRequest) -> Result<IsLoadedResponse, RpcError> {
            self.record("is_loaded");
            Ok(IsLoadedResponse {
                envelope_version: MODEL_ENVELOPE_VERSION,
                loaded: true,
            })
        }
        async fn status(&self, _req: StatusRequest) -> Result<StatusResponse, RpcError> {
            self.record("status");
            Ok(StatusResponse {
                envelope_version: MODEL_ENVELOPE_VERSION,
                models: Vec::new(),
            })
        }
        async fn load_from_hf(
            &self,
            _req: LoadFromHfRequest,
        ) -> Result<LoadFromHfResponse, RpcError> {
            self.record("load_from_hf");
            Ok(LoadFromHfResponse {
                envelope_version: MODEL_ENVELOPE_VERSION,
                chosen_backend: crate::model_protocol::BackendHintWire::MistralRs,
            })
        }
        async fn load_adapter(
            &self,
            req: LoadAdapterRequest,
        ) -> Result<LoadAdapterResponse, RpcError> {
            self.record("load_adapter");
            Ok(LoadAdapterResponse {
                envelope_version: MODEL_ENVELOPE_VERSION,
                adapter_id: req.adapter_id,
                memory_bytes: 4096,
                mount_strategy: crate::model_protocol::AdapterMountStrategyWire::Attached,
            })
        }
        async fn unload_adapter(
            &self,
            _req: UnloadAdapterRequest,
        ) -> Result<UnloadAdapterResponse, RpcError> {
            self.record("unload_adapter");
            Ok(UnloadAdapterResponse {
                envelope_version: MODEL_ENVELOPE_VERSION,
            })
        }
        async fn list_adapters(
            &self,
            _req: ListAdaptersRequest,
        ) -> Result<ListAdaptersResponse, RpcError> {
            self.record("list_adapters");
            Ok(ListAdaptersResponse {
                envelope_version: MODEL_ENVELOPE_VERSION,
                adapters: Vec::new(),
            })
        }
        async fn complete(&self, req: CompleteRequest) -> Result<CompleteResponse, RpcError> {
            self.record("complete");
            Ok(CompleteResponse {
                envelope_version: MODEL_ENVELOPE_VERSION,
                text: format!("echo:{}", req.messages.len()),
                prompt_tokens: Some(3),
                completion_tokens: Some(2),
                finish_reason: Some("stop".to_owned()),
                tool_calls_json: Vec::new(),
            })
        }
        async fn stream_complete(
            &self,
            _req: CompleteRequest,
        ) -> Result<StreamCompleteStream, RpcError> {
            self.record("stream_complete");
            let chunks = vec![
                Ok(StreamCompleteChunk::Delta {
                    envelope_version: MODEL_ENVELOPE_VERSION,
                    text: "hello".to_owned(),
                }),
                Ok(StreamCompleteChunk::Delta {
                    envelope_version: MODEL_ENVELOPE_VERSION,
                    text: " world".to_owned(),
                }),
                Ok(StreamCompleteChunk::Done {
                    envelope_version: MODEL_ENVELOPE_VERSION,
                    prompt_tokens: Some(3),
                    completion_tokens: Some(2),
                    finish_reason: Some("stop".to_owned()),
                }),
            ];
            Ok(Box::pin(tokio_stream::iter(chunks)))
        }
        async fn embed(&self, req: EmbedRequest) -> Result<EmbedResponse, RpcError> {
            self.record("embed");
            Ok(EmbedResponse {
                envelope_version: MODEL_ENVELOPE_VERSION,
                vectors: req.inputs.iter().map(|_| vec![0.0_f32; 4]).collect(),
                prompt_tokens: None,
            })
        }
        async fn generate_image(
            &self,
            _req: GenerateImageRequest,
        ) -> Result<GenerateImageResponse, RpcError> {
            self.record("generate_image");
            Ok(GenerateImageResponse {
                envelope_version: MODEL_ENVELOPE_VERSION,
                images: vec![crate::model_protocol::ImageBlobWire {
                    mime: "image/png".to_owned(),
                    data: vec![0xDE, 0xAD],
                }],
            })
        }
        async fn text_to_speech(
            &self,
            _req: TextToSpeechRequest,
        ) -> Result<TextToSpeechResponse, RpcError> {
            self.record("text_to_speech");
            Ok(TextToSpeechResponse {
                envelope_version: MODEL_ENVELOPE_VERSION,
                mime: "audio/wav".to_owned(),
                data: vec![0; 8],
                sample_rate_hz: Some(24_000),
            })
        }
        async fn generate_music(
            &self,
            _req: GenerateMusicRequest,
        ) -> Result<GenerateMusicResponse, RpcError> {
            self.record("generate_music");
            Ok(GenerateMusicResponse {
                envelope_version: MODEL_ENVELOPE_VERSION,
                mime: "audio/wav".to_owned(),
                data: vec![0; 16],
                sample_rate_hz: Some(32_000),
            })
        }
        async fn transcribe(
            &self,
            _req: TranscribeRequest,
        ) -> Result<TranscribeResponse, RpcError> {
            self.record("transcribe");
            Ok(TranscribeResponse {
                envelope_version: MODEL_ENVELOPE_VERSION,
                text: "transcribed".to_owned(),
                language: Some("en".to_owned()),
                segments_json: Vec::new(),
            })
        }
        async fn upload_blob(
            &self,
            mut chunks: mpsc::Receiver<UploadBlobChunk>,
        ) -> Result<UploadBlobResponse, RpcError> {
            self.record("upload_blob");
            let mut total: u64 = 0;
            let mut id = String::new();
            while let Some(c) = chunks.recv().await {
                match c {
                    UploadBlobChunk::Start { blob_id, .. } => id = blob_id,
                    UploadBlobChunk::Data { bytes, .. } => {
                        total = total.saturating_add(bytes.len() as u64);
                    }
                    UploadBlobChunk::End { .. } => {}
                }
            }
            Ok(UploadBlobResponse {
                envelope_version: MODEL_ENVELOPE_VERSION,
                blob_id: id,
                bytes_received: total,
            })
        }
        async fn fetch_blob(&self, req: FetchBlobRequest) -> Result<FetchBlobStream, RpcError> {
            self.record("fetch_blob");
            let chunks = vec![
                Ok(FetchBlobChunk::Start {
                    envelope_version: MODEL_ENVELOPE_VERSION,
                    blob_id: req.blob_id,
                    total_bytes: Some(4),
                    content_type: None,
                }),
                Ok(FetchBlobChunk::Data {
                    envelope_version: MODEL_ENVELOPE_VERSION,
                    bytes: vec![1, 2, 3, 4],
                }),
                Ok(FetchBlobChunk::End {
                    envelope_version: MODEL_ENVELOPE_VERSION,
                }),
            ];
            Ok(Box::pin(tokio_stream::iter(chunks)))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::test_support::MockManagerHandle;
    use super::*;
    use crate::model_protocol::{
        BackendHintWire, ChatMessageWire, MODEL_ENVELOPE_VERSION, RPC_ERR_INCOMPATIBLE,
    };

    fn state() -> (ModelServerState, Arc<MockManagerHandle>) {
        let mock = MockManagerHandle::new();
        let state = ModelServerState::new(mock.clone());
        (state, mock)
    }

    fn pack<T: serde::Serialize>(v: &T) -> PostcardRequest {
        PostcardRequest {
            postcard_payload: postcard::to_allocvec(v).unwrap(),
        }
    }

    fn unpack<T: serde::de::DeserializeOwned>(resp: Response<PostcardResponse>) -> T {
        postcard::from_bytes(&resp.into_inner().postcard_payload).unwrap()
    }

    #[tokio::test]
    async fn load_handler_dispatches_to_handle() {
        let (state, mock) = state();
        let req = LoadRequest {
            envelope_version: MODEL_ENVELOPE_VERSION,
            model_id: "m".to_owned(),
        };
        let resp = handle_load(&state, pack(&req)).await.unwrap();
        let out: RpcResult<LoadResponse> = unpack(resp);
        assert!(matches!(out, RpcResult::Ok(_)));
        assert_eq!(mock.calls.lock().unwrap().as_slice(), &["load"]);
    }

    #[tokio::test]
    async fn load_handler_rejects_future_envelope() {
        let (state, _mock) = state();
        let req = LoadRequest {
            envelope_version: MODEL_ENVELOPE_VERSION + 1,
            model_id: "m".to_owned(),
        };
        let resp = handle_load(&state, pack(&req)).await.unwrap();
        let out: RpcResult<LoadResponse> = unpack(resp);
        match out {
            RpcResult::Err(e) => assert_eq!(e.code, RPC_ERR_INCOMPATIBLE),
            RpcResult::Ok(_) => panic!("expected incompatible-envelope error"),
        }
    }

    #[tokio::test]
    async fn complete_handler_roundtrip() {
        let (state, _mock) = state();
        let req = CompleteRequest {
            envelope_version: MODEL_ENVELOPE_VERSION,
            model_id: "m".to_owned(),
            messages: vec![ChatMessageWire {
                role: "user".to_owned(),
                text: "hi".to_owned(),
                content_json: Vec::new(),
            }],
            max_tokens: Some(8),
            temperature: None,
            top_p: None,
            stop: Vec::new(),
            response_format_json: Vec::new(),
            extra_json: Vec::new(),
            tags: std::collections::BTreeMap::new(),
        };
        let resp = handle_complete(&state, pack(&req)).await.unwrap();
        let out: RpcResult<CompleteResponse> = unpack(resp);
        if let RpcResult::Ok(c) = out {
            assert!(c.text.starts_with("echo:"));
        } else {
            panic!("expected ok");
        }
    }

    #[tokio::test]
    async fn stream_complete_emits_all_frames() {
        let (state, _mock) = state();
        let req = CompleteRequest {
            envelope_version: MODEL_ENVELOPE_VERSION,
            model_id: "m".to_owned(),
            messages: Vec::new(),
            max_tokens: None,
            temperature: None,
            top_p: None,
            stop: Vec::new(),
            response_format_json: Vec::new(),
            extra_json: Vec::new(),
            tags: std::collections::BTreeMap::new(),
        };
        let resp = handle_stream_complete(state, pack(&req)).await.unwrap();
        let mut stream = resp.into_inner();
        let mut frames = Vec::new();
        while let Some(f) = stream.next().await {
            let f = f.unwrap();
            let chunk: RpcResult<StreamCompleteChunk> =
                postcard::from_bytes(&f.postcard_payload).unwrap();
            frames.push(chunk);
        }
        assert_eq!(frames.len(), 3);
        assert!(matches!(
            frames[2],
            RpcResult::Ok(StreamCompleteChunk::Done { .. })
        ));
    }

    #[tokio::test]
    async fn embed_handler_returns_canned_vectors() {
        let (state, _mock) = state();
        let req = EmbedRequest {
            envelope_version: MODEL_ENVELOPE_VERSION,
            model_id: "bge".to_owned(),
            inputs: vec!["a".to_owned(), "b".to_owned(), "c".to_owned()],
            dimensions: None,
            extra_json: Vec::new(),
        };
        let resp = handle_embed(&state, pack(&req)).await.unwrap();
        let out: RpcResult<EmbedResponse> = unpack(resp);
        match out {
            RpcResult::Ok(e) => assert_eq!(e.vectors.len(), 3),
            RpcResult::Err(_) => panic!("expected ok"),
        }
    }

    #[tokio::test]
    async fn load_from_hf_returns_chosen_backend() {
        let (state, _mock) = state();
        let req = LoadFromHfRequest {
            envelope_version: MODEL_ENVELOPE_VERSION,
            model_id: "m".to_owned(),
            repo: "Qwen/Qwen3-7B".to_owned(),
            memory_estimate_bytes: None,
            backend_hint: None,
            gguf_file: None,
            revision: None,
            hf_token: None,
            extra_options_json: Vec::new(),
        };
        let resp = handle_load_from_hf(&state, pack(&req)).await.unwrap();
        let out: RpcResult<LoadFromHfResponse> = unpack(resp);
        match out {
            RpcResult::Ok(r) => assert_eq!(r.chosen_backend, BackendHintWire::MistralRs),
            RpcResult::Err(_) => panic!("expected ok"),
        }
    }
}
