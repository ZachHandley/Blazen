//! Tonic client wrapper for the `BlazenModelServer` gRPC service.
//!
//! Mirrors [`super::Client`] (which targets the workflow control
//! plane) — every method postcard-encodes a typed request, sends it
//! through the gRPC channel, and postcard-decodes the response into a
//! typed wire struct. Semantic errors travel inside the response as
//! [`RpcError`]; transport failures travel as
//! [`ControlPlaneError::Transport`].
//!
//! The whole thing wraps a `Channel` behind an `Arc<Mutex<...>>` so
//! cloning the client is cheap and concurrent RPCs serialize behind the
//! underlying gRPC client (each generated `&mut self` method requires
//! it).

use std::pin::Pin;
use std::sync::Arc;
use std::time::Duration;

use futures_util::Stream;
use futures_util::StreamExt;
use tokio::sync::Mutex;
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;
use tonic::transport::{Channel, ClientTlsConfig, Endpoint};

use crate::error::ControlPlaneError;
use crate::model_pb;
use crate::model_pb::PostcardRequest;
use crate::model_protocol::{
    CompleteRequest, CompleteResponse, EmbedRequest, EmbedResponse, FetchBlobChunk,
    FetchBlobRequest, GenerateImageRequest, GenerateImageResponse, GenerateMusicRequest,
    GenerateMusicResponse, IsLoadedRequest, IsLoadedResponse, ListAdaptersRequest,
    ListAdaptersResponse, LoadAdapterRequest, LoadAdapterResponse, LoadFromHfRequest,
    LoadFromHfResponse, LoadRequest, LoadResponse, RpcError, RpcResult, StatusRequest,
    StatusResponse, StreamCompleteChunk, TextToSpeechRequest, TextToSpeechResponse,
    TranscribeRequest, TranscribeResponse, UnloadAdapterRequest, UnloadAdapterResponse,
    UnloadRequest, UnloadResponse, UploadBlobChunk, UploadBlobResponse,
};

type Grpc = model_pb::blazen_model_server_client::BlazenModelServerClient<Channel>;

/// Boxed `Stream` of [`StreamCompleteChunk`]s yielded by
/// [`ModelClient::stream_complete`]. Each frame is already
/// postcard-decoded; errors are surfaced as [`RpcError`].
pub type StreamCompleteRecvStream =
    Pin<Box<dyn Stream<Item = Result<StreamCompleteChunk, ControlPlaneError>> + Send>>;

/// Boxed `Stream` of [`FetchBlobChunk`]s, matching the shape of
/// [`StreamCompleteRecvStream`].
pub type FetchBlobRecvStream =
    Pin<Box<dyn Stream<Item = Result<FetchBlobChunk, ControlPlaneError>> + Send>>;

/// gRPC client for the `BlazenModelServer` service.
///
/// Cheaply cloneable.
#[derive(Clone)]
pub struct ModelClient {
    inner: Arc<Mutex<Grpc>>,
}

impl ModelClient {
    /// Open a connection to a model server at `endpoint`. Pass
    /// `tls = None` for plaintext.
    ///
    /// # Errors
    /// Returns [`ControlPlaneError::Transport`] if the endpoint URI is
    /// invalid or the TCP / HTTP-2 handshake fails. Returns
    /// [`ControlPlaneError::Tls`] when a TLS config cannot be applied.
    pub async fn connect(endpoint: impl Into<String>) -> Result<Self, ControlPlaneError> {
        Self::connect_with_tls(endpoint, None).await
    }

    /// Open a connection with an optional [`ClientTlsConfig`].
    ///
    /// # Errors
    /// See [`Self::connect`].
    pub async fn connect_with_tls(
        endpoint: impl Into<String>,
        tls: Option<ClientTlsConfig>,
    ) -> Result<Self, ControlPlaneError> {
        let endpoint_str = endpoint.into();
        let mut endpoint = Endpoint::from_shared(endpoint_str)
            .map_err(|e| ControlPlaneError::Transport(format!("invalid endpoint URI: {e}")))?
            .tcp_keepalive(Some(Duration::from_secs(30)))
            .http2_keep_alive_interval(Duration::from_secs(20))
            .keep_alive_while_idle(true);
        if let Some(t) = tls {
            endpoint = endpoint
                .tls_config(t)
                .map_err(|e| ControlPlaneError::Tls(e.to_string()))?;
        }
        let channel = endpoint
            .connect()
            .await
            .map_err(|e| ControlPlaneError::Transport(format!("connect: {e}")))?;
        Ok(Self {
            inner: Arc::new(Mutex::new(Grpc::new(channel))),
        })
    }

    // ----- Lifecycle -----

    /// Issue a `Load` RPC.
    ///
    /// # Errors
    /// Returns [`ControlPlaneError::Transport`] for wire failures and
    /// [`ControlPlaneError::Rpc`] for model-layer failures.
    pub async fn load(&self, req: LoadRequest) -> Result<LoadResponse, ControlPlaneError> {
        let resp = self
            .inner
            .lock()
            .await
            .load(pack(&req)?)
            .await
            .map_err(|e| ControlPlaneError::Transport(e.to_string()))?;
        unpack::<LoadResponse>(&resp.into_inner().postcard_payload)
    }

    /// Issue an `Unload` RPC.
    ///
    /// # Errors
    /// See [`Self::load`].
    pub async fn unload(&self, req: UnloadRequest) -> Result<UnloadResponse, ControlPlaneError> {
        let resp = self
            .inner
            .lock()
            .await
            .unload(pack(&req)?)
            .await
            .map_err(|e| ControlPlaneError::Transport(e.to_string()))?;
        unpack::<UnloadResponse>(&resp.into_inner().postcard_payload)
    }

    /// Issue an `IsLoaded` RPC.
    ///
    /// # Errors
    /// See [`Self::load`].
    pub async fn is_loaded(
        &self,
        req: IsLoadedRequest,
    ) -> Result<IsLoadedResponse, ControlPlaneError> {
        let resp = self
            .inner
            .lock()
            .await
            .is_loaded(pack(&req)?)
            .await
            .map_err(|e| ControlPlaneError::Transport(e.to_string()))?;
        unpack::<IsLoadedResponse>(&resp.into_inner().postcard_payload)
    }

    /// Issue a `Status` RPC.
    ///
    /// # Errors
    /// See [`Self::load`].
    pub async fn status(&self, req: StatusRequest) -> Result<StatusResponse, ControlPlaneError> {
        let resp = self
            .inner
            .lock()
            .await
            .status(pack(&req)?)
            .await
            .map_err(|e| ControlPlaneError::Transport(e.to_string()))?;
        unpack::<StatusResponse>(&resp.into_inner().postcard_payload)
    }

    /// Issue a `LoadFromHf` RPC.
    ///
    /// # Errors
    /// See [`Self::load`].
    pub async fn load_from_hf(
        &self,
        req: LoadFromHfRequest,
    ) -> Result<LoadFromHfResponse, ControlPlaneError> {
        let resp = self
            .inner
            .lock()
            .await
            .load_from_hf(pack(&req)?)
            .await
            .map_err(|e| ControlPlaneError::Transport(e.to_string()))?;
        unpack::<LoadFromHfResponse>(&resp.into_inner().postcard_payload)
    }

    // ----- Adapters -----

    /// Issue a `LoadAdapter` RPC.
    ///
    /// # Errors
    /// See [`Self::load`].
    pub async fn load_adapter(
        &self,
        req: LoadAdapterRequest,
    ) -> Result<LoadAdapterResponse, ControlPlaneError> {
        let resp = self
            .inner
            .lock()
            .await
            .load_adapter(pack(&req)?)
            .await
            .map_err(|e| ControlPlaneError::Transport(e.to_string()))?;
        unpack::<LoadAdapterResponse>(&resp.into_inner().postcard_payload)
    }

    /// Issue an `UnloadAdapter` RPC.
    ///
    /// # Errors
    /// See [`Self::load`].
    pub async fn unload_adapter(
        &self,
        req: UnloadAdapterRequest,
    ) -> Result<UnloadAdapterResponse, ControlPlaneError> {
        let resp = self
            .inner
            .lock()
            .await
            .unload_adapter(pack(&req)?)
            .await
            .map_err(|e| ControlPlaneError::Transport(e.to_string()))?;
        unpack::<UnloadAdapterResponse>(&resp.into_inner().postcard_payload)
    }

    /// Issue a `ListAdapters` RPC.
    ///
    /// # Errors
    /// See [`Self::load`].
    pub async fn list_adapters(
        &self,
        req: ListAdaptersRequest,
    ) -> Result<ListAdaptersResponse, ControlPlaneError> {
        let resp = self
            .inner
            .lock()
            .await
            .list_adapters(pack(&req)?)
            .await
            .map_err(|e| ControlPlaneError::Transport(e.to_string()))?;
        unpack::<ListAdaptersResponse>(&resp.into_inner().postcard_payload)
    }

    // ----- Inference -----

    /// Issue a `Complete` RPC.
    ///
    /// # Errors
    /// See [`Self::load`].
    pub async fn complete(
        &self,
        req: CompleteRequest,
    ) -> Result<CompleteResponse, ControlPlaneError> {
        let resp = self
            .inner
            .lock()
            .await
            .complete(pack(&req)?)
            .await
            .map_err(|e| ControlPlaneError::Transport(e.to_string()))?;
        unpack::<CompleteResponse>(&resp.into_inner().postcard_payload)
    }

    /// Issue a `StreamComplete` server-streaming RPC. Returns a stream of
    /// decoded [`StreamCompleteChunk`]s.
    ///
    /// # Errors
    /// Returns [`ControlPlaneError::Transport`] for the initial wire
    /// failure; per-frame errors surface inside the stream.
    pub async fn stream_complete(
        &self,
        req: CompleteRequest,
    ) -> Result<StreamCompleteRecvStream, ControlPlaneError> {
        let resp = self
            .inner
            .lock()
            .await
            .stream_complete(pack(&req)?)
            .await
            .map_err(|e| ControlPlaneError::Transport(e.to_string()))?;
        let inner = resp.into_inner();
        let mapped = inner.map(|frame| match frame {
            Ok(f) => unpack::<StreamCompleteChunk>(&f.postcard_payload),
            Err(s) => Err(ControlPlaneError::Transport(s.to_string())),
        });
        Ok(Box::pin(mapped))
    }

    /// Issue an `Embed` RPC.
    ///
    /// # Errors
    /// See [`Self::load`].
    pub async fn embed(&self, req: EmbedRequest) -> Result<EmbedResponse, ControlPlaneError> {
        let resp = self
            .inner
            .lock()
            .await
            .embed(pack(&req)?)
            .await
            .map_err(|e| ControlPlaneError::Transport(e.to_string()))?;
        unpack::<EmbedResponse>(&resp.into_inner().postcard_payload)
    }

    /// Issue a `GenerateImage` RPC.
    ///
    /// # Errors
    /// See [`Self::load`].
    pub async fn generate_image(
        &self,
        req: GenerateImageRequest,
    ) -> Result<GenerateImageResponse, ControlPlaneError> {
        let resp = self
            .inner
            .lock()
            .await
            .generate_image(pack(&req)?)
            .await
            .map_err(|e| ControlPlaneError::Transport(e.to_string()))?;
        unpack::<GenerateImageResponse>(&resp.into_inner().postcard_payload)
    }

    /// Issue a `TextToSpeech` RPC.
    ///
    /// # Errors
    /// See [`Self::load`].
    pub async fn text_to_speech(
        &self,
        req: TextToSpeechRequest,
    ) -> Result<TextToSpeechResponse, ControlPlaneError> {
        let resp = self
            .inner
            .lock()
            .await
            .text_to_speech(pack(&req)?)
            .await
            .map_err(|e| ControlPlaneError::Transport(e.to_string()))?;
        unpack::<TextToSpeechResponse>(&resp.into_inner().postcard_payload)
    }

    /// Issue a `GenerateMusic` RPC.
    ///
    /// # Errors
    /// See [`Self::load`].
    pub async fn generate_music(
        &self,
        req: GenerateMusicRequest,
    ) -> Result<GenerateMusicResponse, ControlPlaneError> {
        let resp = self
            .inner
            .lock()
            .await
            .generate_music(pack(&req)?)
            .await
            .map_err(|e| ControlPlaneError::Transport(e.to_string()))?;
        unpack::<GenerateMusicResponse>(&resp.into_inner().postcard_payload)
    }

    /// Issue a `Transcribe` RPC.
    ///
    /// # Errors
    /// See [`Self::load`].
    pub async fn transcribe(
        &self,
        req: TranscribeRequest,
    ) -> Result<TranscribeResponse, ControlPlaneError> {
        let resp = self
            .inner
            .lock()
            .await
            .transcribe(pack(&req)?)
            .await
            .map_err(|e| ControlPlaneError::Transport(e.to_string()))?;
        unpack::<TranscribeResponse>(&resp.into_inner().postcard_payload)
    }

    // ----- Blobs -----

    /// Issue an `UploadBlob` client-streaming RPC. Drains `chunks` and
    /// returns the server's acknowledgement once the stream is closed.
    ///
    /// # Errors
    /// See [`Self::load`].
    pub async fn upload_blob(
        &self,
        mut chunks: mpsc::Receiver<UploadBlobChunk>,
    ) -> Result<UploadBlobResponse, ControlPlaneError> {
        let (tx, rx) = mpsc::channel::<PostcardRequest>(16);
        tokio::spawn(async move {
            while let Some(c) = chunks.recv().await {
                let Ok(bytes) = postcard::to_allocvec(&c) else {
                    return;
                };
                if tx
                    .send(PostcardRequest {
                        postcard_payload: bytes,
                    })
                    .await
                    .is_err()
                {
                    return;
                }
            }
        });
        let outbound = ReceiverStream::new(rx);
        let resp = self
            .inner
            .lock()
            .await
            .upload_blob(outbound)
            .await
            .map_err(|e| ControlPlaneError::Transport(e.to_string()))?;
        unpack::<UploadBlobResponse>(&resp.into_inner().postcard_payload)
    }

    /// Issue a `FetchBlob` server-streaming RPC.
    ///
    /// # Errors
    /// See [`Self::stream_complete`].
    pub async fn fetch_blob(
        &self,
        req: FetchBlobRequest,
    ) -> Result<FetchBlobRecvStream, ControlPlaneError> {
        let resp = self
            .inner
            .lock()
            .await
            .fetch_blob(pack(&req)?)
            .await
            .map_err(|e| ControlPlaneError::Transport(e.to_string()))?;
        let inner = resp.into_inner();
        let mapped = inner.map(|frame| match frame {
            Ok(f) => unpack::<FetchBlobChunk>(&f.postcard_payload),
            Err(s) => Err(ControlPlaneError::Transport(s.to_string())),
        });
        Ok(Box::pin(mapped))
    }
}

fn pack<T: serde::Serialize>(v: &T) -> Result<PostcardRequest, ControlPlaneError> {
    let bytes = postcard::to_allocvec(v)?;
    Ok(PostcardRequest {
        postcard_payload: bytes,
    })
}

fn unpack<T: serde::de::DeserializeOwned>(bytes: &[u8]) -> Result<T, ControlPlaneError> {
    let result: RpcResult<T> = postcard::from_bytes(bytes)?;
    result.into_result().map_err(ControlPlaneError::from)
}

impl From<RpcError> for ControlPlaneError {
    fn from(e: RpcError) -> Self {
        Self::Rpc(e)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model_protocol::MODEL_ENVELOPE_VERSION;

    #[test]
    fn pack_unpack_round_trip_ok() {
        let resp = LoadResponse {
            envelope_version: MODEL_ENVELOPE_VERSION,
        };
        let ok: RpcResult<LoadResponse> = RpcResult::Ok(resp.clone());
        let bytes = postcard::to_allocvec(&ok).unwrap();
        let out = unpack::<LoadResponse>(&bytes).unwrap();
        assert_eq!(out, resp);
    }

    #[test]
    fn pack_unpack_propagates_rpc_error() {
        let err: RpcResult<LoadResponse> = RpcResult::Err(RpcError::not_found("missing"));
        let bytes = postcard::to_allocvec(&err).unwrap();
        match unpack::<LoadResponse>(&bytes) {
            Err(ControlPlaneError::Rpc(rpc)) => {
                assert_eq!(rpc.message, "missing");
            }
            other => panic!("expected Rpc error, got {other:?}"),
        }
    }
}
