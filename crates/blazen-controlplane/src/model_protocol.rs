//! Postcard wire types for the `BlazenModelServer` gRPC service.
//!
//! Mirrors the design of [`crate::protocol`]: every RPC carries a single
//! `postcard_payload: Vec<u8>` whose contents are one of the request /
//! response structs defined here, encoded with [`postcard`]. Per-message
//! versioning lives on the [`MODEL_ENVELOPE_VERSION`] field of each
//! struct so the proto schema never has to evolve.
//!
//! ## A note on JSON inside postcard
//!
//! Postcard is a non-self-describing format and cannot round-trip
//! `serde_json::Value` directly (the untagged enum needs
//! `deserialize_any`, which postcard explicitly does not implement). For
//! payloads that originate as user-supplied JSON (`response_format`
//! schemas, `image_config`, `audio_config`, free-form provider
//! parameters) we carry the **already-serialised JSON bytes** as
//! `Vec<u8>` and let the host decode them at the trait boundary. Empty
//! `Vec<u8>` means "no JSON supplied".
//!
//! ## `RpcError`
//!
//! Errors that originate inside the host's `ModelManager` are returned
//! to the client as `Response<...> = Err(RpcError)` rather than as a
//! tonic `Status`, so the client can distinguish *transport* failures
//! (the wire) from *semantic* failures (the model). The `retryable`
//! flag is forensic only — the client decides what to do with it.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

/// Current envelope version for `BlazenModelServer` wire types. Bump
/// whenever a struct in this module changes shape in a way that isn't
/// forward-compatible (rename, reorder, remove).
///
/// Adding a new trailing `Option<_>` field is forward-compatible —
/// postcard tolerates missing trailing bytes — so does not require a
/// bump.
pub const MODEL_ENVELOPE_VERSION: u32 = 1;

/// Returns `Err(RpcError::incompatible)` if `got > MODEL_ENVELOPE_VERSION`.
///
/// Older payloads are always accepted; the field-by-field decode of
/// postcard tolerates missing trailing fields.
///
/// # Errors
/// Returns [`RpcError`] with code [`RPC_ERR_INCOMPATIBLE`] when the
/// incoming version is newer than this build understands.
pub fn validate_model_envelope_version(got: u32) -> Result<(), RpcError> {
    if got > MODEL_ENVELOPE_VERSION {
        Err(RpcError::incompatible(format!(
            "envelope version {got} is newer than supported {MODEL_ENVELOPE_VERSION}",
        )))
    } else {
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Error envelope
// ---------------------------------------------------------------------------

/// Numeric error code carried on [`RpcError`]. Lets the client switch on
/// a stable integer rather than string-matching the message.
pub const RPC_ERR_INTERNAL: u32 = 1;
/// Caller supplied a bad request (decode failure, missing field, etc.).
pub const RPC_ERR_INVALID: u32 = 2;
/// Requested model / adapter is not registered.
pub const RPC_ERR_NOT_FOUND: u32 = 3;
/// Backend reports it can't honor this verb at all.
pub const RPC_ERR_UNSUPPORTED: u32 = 4;
/// Operation timed out before the backend produced a result.
pub const RPC_ERR_TIMEOUT: u32 = 5;
/// Caller would exceed a pool budget / quota.
pub const RPC_ERR_QUOTA: u32 = 6;
/// Caller's envelope version is newer than the server understands.
pub const RPC_ERR_INCOMPATIBLE: u32 = 7;

/// Semantic error returned inside a successful gRPC response (`Ok(...)`
/// at the tonic layer) so the client can distinguish wire failures from
/// model failures.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct RpcError {
    /// Numeric code; one of the `RPC_ERR_*` constants above.
    pub code: u32,
    /// Human-readable message, suitable for logging.
    pub message: String,
    /// Hint to the client about whether retrying is likely to succeed.
    /// Forensic only — the client decides policy.
    pub retryable: bool,
}

impl RpcError {
    /// Construct an `RPC_ERR_INTERNAL` error (catch-all, non-retryable).
    #[must_use]
    pub fn internal(msg: impl Into<String>) -> Self {
        Self {
            code: RPC_ERR_INTERNAL,
            message: msg.into(),
            retryable: false,
        }
    }
    /// Construct an `RPC_ERR_INVALID` error.
    #[must_use]
    pub fn invalid(msg: impl Into<String>) -> Self {
        Self {
            code: RPC_ERR_INVALID,
            message: msg.into(),
            retryable: false,
        }
    }
    /// Construct an `RPC_ERR_NOT_FOUND` error.
    #[must_use]
    pub fn not_found(msg: impl Into<String>) -> Self {
        Self {
            code: RPC_ERR_NOT_FOUND,
            message: msg.into(),
            retryable: false,
        }
    }
    /// Construct an `RPC_ERR_UNSUPPORTED` error.
    #[must_use]
    pub fn unsupported(msg: impl Into<String>) -> Self {
        Self {
            code: RPC_ERR_UNSUPPORTED,
            message: msg.into(),
            retryable: false,
        }
    }
    /// Construct an `RPC_ERR_TIMEOUT` error (retryable by default).
    #[must_use]
    pub fn timeout(msg: impl Into<String>) -> Self {
        Self {
            code: RPC_ERR_TIMEOUT,
            message: msg.into(),
            retryable: true,
        }
    }
    /// Construct an `RPC_ERR_QUOTA` error.
    #[must_use]
    pub fn quota(msg: impl Into<String>) -> Self {
        Self {
            code: RPC_ERR_QUOTA,
            message: msg.into(),
            retryable: false,
        }
    }
    /// Construct an `RPC_ERR_INCOMPATIBLE` error.
    #[must_use]
    pub fn incompatible(msg: impl Into<String>) -> Self {
        Self {
            code: RPC_ERR_INCOMPATIBLE,
            message: msg.into(),
            retryable: false,
        }
    }
}

/// Convenience wrapper used on every response payload — postcard-encoded
/// `Result<T, RpcError>` so a single decode at the client surfaces both
/// success and semantic-failure paths.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RpcResult<T> {
    /// Request succeeded; carries the typed response payload.
    Ok(T),
    /// Request failed at the model layer; see [`RpcError`].
    Err(RpcError),
}

impl<T> RpcResult<T> {
    /// Convert into a `std::result::Result` for ergonomic `?` usage on
    /// the client side.
    ///
    /// # Errors
    /// Returns the inner [`RpcError`] when this is the `Err` variant.
    pub fn into_result(self) -> Result<T, RpcError> {
        match self {
            Self::Ok(v) => Ok(v),
            Self::Err(e) => Err(e),
        }
    }
}

impl<T> From<Result<T, RpcError>> for RpcResult<T> {
    fn from(r: Result<T, RpcError>) -> Self {
        match r {
            Ok(v) => Self::Ok(v),
            Err(e) => Self::Err(e),
        }
    }
}

// ---------------------------------------------------------------------------
// Lifecycle (5)
// ---------------------------------------------------------------------------

/// Pool a model is registered against. Mirrors `blazen_llm::Pool` but
/// kept independent so this crate doesn't grow a hard dep on blazen-llm.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum PoolWire {
    /// Host RAM pool.
    Cpu,
    /// GPU VRAM pool at the given device index. Metal collapses to `Gpu(0)`.
    Gpu(u32),
    /// Off-host pool — the memory lives in another process / host.
    Remote,
}

/// Snapshot of a single registered model — mirrors the host's
/// `blazen_manager::ModelStatus`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ModelStatusWire {
    /// Identifier under which the model was registered.
    pub id: String,
    /// Whether the model is currently loaded into its pool.
    pub loaded: bool,
    /// Estimated memory footprint in bytes, including any mounted adapters.
    pub memory_estimate_bytes: u64,
    /// Pool the model is charged against.
    pub pool: PoolWire,
    /// Adapters currently mounted on this model (possibly empty).
    pub adapters: Vec<AdapterStatusWire>,
}

/// Snapshot of a single mounted adapter on a model.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct AdapterStatusWire {
    /// Caller-chosen adapter id.
    pub adapter_id: String,
    /// Scaling factor applied to the adapter's delta-weights.
    pub scale: f32,
    /// On-disk path the adapter was loaded from.
    pub source_dir: String,
    /// Bytes the adapter adds on top of the base model.
    pub memory_bytes: u64,
}

/// `Load` RPC request — instruct the host to load a previously-
/// registered model.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct LoadRequest {
    pub envelope_version: u32,
    /// Id under which the target model was registered.
    pub model_id: String,
}

/// `Load` RPC response — empty body on success; failures travel via
/// [`RpcResult::Err`].
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct LoadResponse {
    pub envelope_version: u32,
}

/// `Unload` RPC request — drop a previously-loaded model from memory.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct UnloadRequest {
    pub envelope_version: u32,
    pub model_id: String,
}

/// `Unload` RPC response.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct UnloadResponse {
    pub envelope_version: u32,
}

/// `IsLoaded` RPC request — boolean liveness check.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct IsLoadedRequest {
    pub envelope_version: u32,
    pub model_id: String,
}

/// `IsLoaded` RPC response.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct IsLoadedResponse {
    pub envelope_version: u32,
    pub loaded: bool,
}

/// `Status` RPC request — snapshot of every registered model.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct StatusRequest {
    pub envelope_version: u32,
}

/// `Status` RPC response.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct StatusResponse {
    pub envelope_version: u32,
    pub models: Vec<ModelStatusWire>,
}

/// Backend selector used by `LoadFromHf`. Mirrors
/// `blazen_manager::hf_loader::BackendHint`; kept independent so this
/// crate doesn't acquire a blazen-manager dep.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum BackendHintWire {
    /// Auto-detect from the repo layout.
    Auto,
    /// Force the `mistral.rs` backend.
    MistralRs,
    /// Force the candle-llm backend.
    Candle,
    /// Force the llama.cpp backend.
    LlamaCpp,
}

/// `LoadFromHf` RPC request — register-and-load a model from a Hugging
/// Face Hub repo. The host implements the actual probe + provider build;
/// here we only carry the user inputs.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct LoadFromHfRequest {
    pub envelope_version: u32,
    /// Id under which to register the resulting model.
    pub model_id: String,
    /// Hugging Face repo slug (`org/name`).
    pub repo: String,
    /// Optional explicit memory estimate in bytes; `None` asks the loader
    /// to estimate from repo metadata.
    pub memory_estimate_bytes: Option<u64>,
    /// Optional backend override.
    pub backend_hint: Option<BackendHintWire>,
    /// Optional GGUF file name when the backend is llama.cpp.
    pub gguf_file: Option<String>,
    /// Optional HF revision (branch / tag / commit).
    pub revision: Option<String>,
    /// Optional bearer token for gated repos.
    pub hf_token: Option<String>,
    /// Pre-serialised JSON for any backend-specific extra options the
    /// host wants to honor. Empty `Vec<u8>` = none.
    #[serde(with = "serde_bytes")]
    pub extra_options_json: Vec<u8>,
}

/// `LoadFromHf` RPC response — reports which backend the loader chose.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct LoadFromHfResponse {
    pub envelope_version: u32,
    /// Backend the loader picked (never `Auto`).
    pub chosen_backend: BackendHintWire,
}

// ---------------------------------------------------------------------------
// Adapters (3)
// ---------------------------------------------------------------------------

/// Backend's report of how a `load_adapter` request was honored. Mirrors
/// `blazen_llm::AdapterMountStrategy`.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum AdapterMountStrategyWire {
    /// Adapter was hot-attached to the live model.
    Attached,
    /// Engine was rebuilt with the adapter.
    Rebuilt,
    /// Adapter weights were merged in place.
    Merged,
}

/// `LoadAdapter` RPC request.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct LoadAdapterRequest {
    pub envelope_version: u32,
    pub model_id: String,
    /// Server-side path to the adapter directory (PEFT layout). The
    /// `UploadBlob` RPC can be used to stage one beforehand.
    pub adapter_dir: String,
    /// Caller-chosen adapter id (must be unique per model).
    pub adapter_id: String,
    /// Scaling factor applied to the adapter's delta-weights.
    pub scale: f32,
}

/// `LoadAdapter` RPC response — carries the host's `AdapterHandle`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct LoadAdapterResponse {
    pub envelope_version: u32,
    pub adapter_id: String,
    pub memory_bytes: u64,
    pub mount_strategy: AdapterMountStrategyWire,
}

/// `UnloadAdapter` RPC request.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct UnloadAdapterRequest {
    pub envelope_version: u32,
    pub model_id: String,
    pub adapter_id: String,
}

/// `UnloadAdapter` RPC response.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct UnloadAdapterResponse {
    pub envelope_version: u32,
}

/// `ListAdapters` RPC request.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ListAdaptersRequest {
    pub envelope_version: u32,
    pub model_id: String,
}

/// `ListAdapters` RPC response.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ListAdaptersResponse {
    pub envelope_version: u32,
    pub adapters: Vec<AdapterStatusWire>,
}

// ---------------------------------------------------------------------------
// Inference (7)
// ---------------------------------------------------------------------------

/// A single chat message — postcard-friendly mirror of the host
/// completion provider's message struct. Kept minimal here; structured
/// content (tool calls, multi-modal) travels via `content_json`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ChatMessageWire {
    /// Role: `"system"`, `"user"`, `"assistant"`, `"tool"`, …
    pub role: String,
    /// Plain-text content. Empty when `content_json` carries the real
    /// payload.
    pub text: String,
    /// Pre-serialised JSON for structured content (tool calls, parts,
    /// multimodal). Empty `Vec<u8>` = none.
    #[serde(with = "serde_bytes")]
    pub content_json: Vec<u8>,
}

/// `Complete` RPC request. Carries the conversation, sampler knobs, and
/// any provider-specific extras as pre-serialised JSON so we don't have
/// to redefine every host config struct here.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct CompleteRequest {
    pub envelope_version: u32,
    pub model_id: String,
    pub messages: Vec<ChatMessageWire>,
    pub max_tokens: Option<u32>,
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    /// Stop sequences applied to the decoded text.
    pub stop: Vec<String>,
    /// Pre-serialised JSON of the host's `ResponseFormat` (JSON-schema
    /// constraints, etc.). Empty = none.
    #[serde(with = "serde_bytes")]
    pub response_format_json: Vec<u8>,
    /// Pre-serialised JSON of any extra provider-specific options.
    /// Empty = none.
    #[serde(with = "serde_bytes")]
    pub extra_json: Vec<u8>,
    /// Caller-supplied tag map for tracing/metrics.
    pub tags: BTreeMap<String, String>,
}

/// `Complete` RPC response.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct CompleteResponse {
    pub envelope_version: u32,
    /// Decoded completion text.
    pub text: String,
    /// Prompt-token count reported by the backend.
    pub prompt_tokens: Option<u32>,
    /// Completion-token count reported by the backend.
    pub completion_tokens: Option<u32>,
    /// Reason the generation stopped (`"stop"`, `"length"`, `"tool_calls"`, …).
    pub finish_reason: Option<String>,
    /// Pre-serialised JSON of any structured tool-call payload. Empty = none.
    #[serde(with = "serde_bytes")]
    pub tool_calls_json: Vec<u8>,
}

/// Single frame in a `StreamComplete` server-stream.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum StreamCompleteChunk {
    /// Incremental token (or token group) emitted by the backend.
    Delta {
        /// Envelope version of this frame.
        envelope_version: u32,
        /// Text fragment to append to the running response.
        text: String,
    },
    /// Terminal frame — carries final usage / finish reason.
    Done {
        envelope_version: u32,
        prompt_tokens: Option<u32>,
        completion_tokens: Option<u32>,
        finish_reason: Option<String>,
    },
}

/// `Embed` RPC request — vector embedding over a batch of inputs.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct EmbedRequest {
    pub envelope_version: u32,
    pub model_id: String,
    pub inputs: Vec<String>,
    /// Optional target dimensionality (some models support truncation).
    pub dimensions: Option<u32>,
    /// Pre-serialised JSON of any extra provider-specific options.
    #[serde(with = "serde_bytes")]
    pub extra_json: Vec<u8>,
}

/// `Embed` RPC response.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct EmbedResponse {
    pub envelope_version: u32,
    /// One vector per input; backends preserve input order.
    pub vectors: Vec<Vec<f32>>,
    pub prompt_tokens: Option<u32>,
}

/// `GenerateImage` RPC request.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct GenerateImageRequest {
    pub envelope_version: u32,
    pub model_id: String,
    pub prompt: String,
    pub negative_prompt: Option<String>,
    pub width: Option<u32>,
    pub height: Option<u32>,
    pub num_images: Option<u32>,
    pub seed: Option<u64>,
    /// Pre-serialised JSON of the host's `ImageConfig`. Empty = none.
    #[serde(with = "serde_bytes")]
    pub image_config_json: Vec<u8>,
}

/// `GenerateImage` RPC response. Image bytes are inlined here for the
/// MVP; very large outputs should use the `FetchBlob` server-stream
/// instead.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct GenerateImageResponse {
    pub envelope_version: u32,
    /// One entry per generated image.
    pub images: Vec<ImageBlobWire>,
}

/// Single inline image carried inside a `GenerateImage` response.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ImageBlobWire {
    /// MIME type (`"image/png"`, `"image/jpeg"`, …).
    pub mime: String,
    /// Raw image bytes.
    #[serde(with = "serde_bytes")]
    pub data: Vec<u8>,
}

/// `TextToSpeech` RPC request.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct TextToSpeechRequest {
    pub envelope_version: u32,
    pub model_id: String,
    pub text: String,
    pub voice: Option<String>,
    pub language: Option<String>,
    pub sample_rate_hz: Option<u32>,
    /// Pre-serialised JSON of the host's `AudioConfig`. Empty = none.
    #[serde(with = "serde_bytes")]
    pub audio_config_json: Vec<u8>,
}

/// `TextToSpeech` RPC response — raw audio inlined.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct TextToSpeechResponse {
    pub envelope_version: u32,
    /// MIME type (`"audio/wav"`, `"audio/mpeg"`, `"audio/ogg"`, …).
    pub mime: String,
    #[serde(with = "serde_bytes")]
    pub data: Vec<u8>,
    pub sample_rate_hz: Option<u32>,
}

/// `GenerateMusic` RPC request.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct GenerateMusicRequest {
    pub envelope_version: u32,
    pub model_id: String,
    pub prompt: String,
    pub duration_secs: Option<f32>,
    pub seed: Option<u64>,
    /// Pre-serialised JSON of any extra options. Empty = none.
    #[serde(with = "serde_bytes")]
    pub extra_json: Vec<u8>,
}

/// `GenerateMusic` RPC response.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct GenerateMusicResponse {
    pub envelope_version: u32,
    pub mime: String,
    #[serde(with = "serde_bytes")]
    pub data: Vec<u8>,
    pub sample_rate_hz: Option<u32>,
}

/// `Transcribe` RPC request.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct TranscribeRequest {
    pub envelope_version: u32,
    pub model_id: String,
    /// Raw audio bytes (caller's responsibility to match `mime`).
    #[serde(with = "serde_bytes")]
    pub audio: Vec<u8>,
    pub mime: String,
    pub language: Option<String>,
    /// Pre-serialised JSON of any extra options. Empty = none.
    #[serde(with = "serde_bytes")]
    pub extra_json: Vec<u8>,
}

/// `Transcribe` RPC response.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct TranscribeResponse {
    pub envelope_version: u32,
    pub text: String,
    pub language: Option<String>,
    /// Pre-serialised JSON of structured segments (timestamps, speaker
    /// labels, …). Empty = backend produced none.
    #[serde(with = "serde_bytes")]
    pub segments_json: Vec<u8>,
}

// ---------------------------------------------------------------------------
// Blob transfer (2)
// ---------------------------------------------------------------------------

/// Single frame inside an `UploadBlob` client-stream.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum UploadBlobChunk {
    /// First frame — names the blob and (optionally) declares total size.
    Start {
        envelope_version: u32,
        /// Caller-chosen blob id; subsequent verbs reference this id.
        blob_id: String,
        /// Total byte count if known; informational only.
        total_bytes: Option<u64>,
        /// MIME-style content hint (e.g. `"application/octet-stream"`).
        content_type: Option<String>,
    },
    /// One chunk of bytes. May appear zero or more times between
    /// `Start` and `End`.
    Data {
        envelope_version: u32,
        #[serde(with = "serde_bytes")]
        bytes: Vec<u8>,
    },
    /// Final frame — closes the upload.
    End { envelope_version: u32 },
}

/// `UploadBlob` RPC response (sent once the server reads `End`).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct UploadBlobResponse {
    pub envelope_version: u32,
    /// Echoes `UploadBlobChunk::Start::blob_id`.
    pub blob_id: String,
    /// Total bytes the server received.
    pub bytes_received: u64,
}

/// `FetchBlob` RPC request.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct FetchBlobRequest {
    pub envelope_version: u32,
    pub blob_id: String,
    /// Optional byte offset for resuming a partial fetch.
    pub offset: Option<u64>,
    /// Optional per-frame chunk size hint.
    pub chunk_size: Option<u32>,
}

/// Single frame inside a `FetchBlob` server-stream.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum FetchBlobChunk {
    /// First frame — declares total size + content type.
    Start {
        envelope_version: u32,
        blob_id: String,
        total_bytes: Option<u64>,
        content_type: Option<String>,
    },
    /// Body bytes; may repeat.
    Data {
        envelope_version: u32,
        #[serde(with = "serde_bytes")]
        bytes: Vec<u8>,
    },
    /// Terminal frame — signals successful EOF.
    End { envelope_version: u32 },
}

// ---------------------------------------------------------------------------
// Tests — postcard roundtrip + envelope-version handling
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn roundtrip<T>(value: &T) -> T
    where
        T: Serialize + serde::de::DeserializeOwned + PartialEq + std::fmt::Debug,
    {
        let bytes = postcard::to_allocvec(value).expect("encode");
        postcard::from_bytes(&bytes).expect("decode")
    }

    #[test]
    fn load_request_postcard_roundtrip() {
        let req = LoadRequest {
            envelope_version: MODEL_ENVELOPE_VERSION,
            model_id: "qwen3-7b".to_owned(),
        };
        assert_eq!(roundtrip(&req), req);
    }

    #[test]
    fn unload_request_postcard_roundtrip() {
        let req = UnloadRequest {
            envelope_version: MODEL_ENVELOPE_VERSION,
            model_id: "qwen3-7b".to_owned(),
        };
        assert_eq!(roundtrip(&req), req);
    }

    #[test]
    fn is_loaded_request_postcard_roundtrip() {
        let req = IsLoadedRequest {
            envelope_version: MODEL_ENVELOPE_VERSION,
            model_id: "qwen3-7b".to_owned(),
        };
        assert_eq!(roundtrip(&req), req);
    }

    #[test]
    fn status_request_postcard_roundtrip() {
        let req = StatusRequest {
            envelope_version: MODEL_ENVELOPE_VERSION,
        };
        assert_eq!(roundtrip(&req), req);
    }

    #[test]
    fn load_from_hf_request_postcard_roundtrip() {
        let req = LoadFromHfRequest {
            envelope_version: MODEL_ENVELOPE_VERSION,
            model_id: "qwen".to_owned(),
            repo: "Qwen/Qwen3-7B".to_owned(),
            memory_estimate_bytes: Some(15 * 1024 * 1024 * 1024),
            backend_hint: Some(BackendHintWire::MistralRs),
            gguf_file: None,
            revision: Some("main".to_owned()),
            hf_token: None,
            extra_options_json: b"{\"flash_attn\":true}".to_vec(),
        };
        assert_eq!(roundtrip(&req), req);
    }

    #[test]
    fn load_adapter_request_postcard_roundtrip() {
        let req = LoadAdapterRequest {
            envelope_version: MODEL_ENVELOPE_VERSION,
            model_id: "qwen".to_owned(),
            adapter_dir: "/srv/adapters/finetune-1".to_owned(),
            adapter_id: "finetune-1".to_owned(),
            scale: 0.75,
        };
        assert_eq!(roundtrip(&req), req);
    }

    #[test]
    fn unload_adapter_request_postcard_roundtrip() {
        let req = UnloadAdapterRequest {
            envelope_version: MODEL_ENVELOPE_VERSION,
            model_id: "qwen".to_owned(),
            adapter_id: "finetune-1".to_owned(),
        };
        assert_eq!(roundtrip(&req), req);
    }

    #[test]
    fn list_adapters_request_postcard_roundtrip() {
        let req = ListAdaptersRequest {
            envelope_version: MODEL_ENVELOPE_VERSION,
            model_id: "qwen".to_owned(),
        };
        assert_eq!(roundtrip(&req), req);
    }

    #[test]
    fn complete_request_postcard_roundtrip() {
        let req = CompleteRequest {
            envelope_version: MODEL_ENVELOPE_VERSION,
            model_id: "qwen".to_owned(),
            messages: vec![ChatMessageWire {
                role: "user".to_owned(),
                text: "hi".to_owned(),
                content_json: Vec::new(),
            }],
            max_tokens: Some(128),
            temperature: Some(0.7),
            top_p: None,
            stop: vec!["</s>".to_owned()],
            response_format_json: Vec::new(),
            extra_json: Vec::new(),
            tags: BTreeMap::from([("trace".to_owned(), "abc".to_owned())]),
        };
        assert_eq!(roundtrip(&req), req);
    }

    #[test]
    fn stream_complete_chunk_postcard_roundtrip() {
        let delta = StreamCompleteChunk::Delta {
            envelope_version: MODEL_ENVELOPE_VERSION,
            text: "Hello, ".to_owned(),
        };
        assert_eq!(roundtrip(&delta), delta);
        let done = StreamCompleteChunk::Done {
            envelope_version: MODEL_ENVELOPE_VERSION,
            prompt_tokens: Some(3),
            completion_tokens: Some(5),
            finish_reason: Some("stop".to_owned()),
        };
        assert_eq!(roundtrip(&done), done);
    }

    #[test]
    fn embed_request_postcard_roundtrip() {
        let req = EmbedRequest {
            envelope_version: MODEL_ENVELOPE_VERSION,
            model_id: "bge".to_owned(),
            inputs: vec!["alpha".to_owned(), "beta".to_owned()],
            dimensions: Some(768),
            extra_json: Vec::new(),
        };
        assert_eq!(roundtrip(&req), req);
    }

    #[test]
    fn generate_image_request_postcard_roundtrip() {
        let req = GenerateImageRequest {
            envelope_version: MODEL_ENVELOPE_VERSION,
            model_id: "sdxl".to_owned(),
            prompt: "a cat".to_owned(),
            negative_prompt: Some("blurry".to_owned()),
            width: Some(1024),
            height: Some(1024),
            num_images: Some(1),
            seed: Some(42),
            image_config_json: b"{\"steps\":30}".to_vec(),
        };
        assert_eq!(roundtrip(&req), req);
    }

    #[test]
    fn text_to_speech_request_postcard_roundtrip() {
        let req = TextToSpeechRequest {
            envelope_version: MODEL_ENVELOPE_VERSION,
            model_id: "kokoro".to_owned(),
            text: "hello world".to_owned(),
            voice: Some("af_heart".to_owned()),
            language: Some("en".to_owned()),
            sample_rate_hz: Some(24_000),
            audio_config_json: Vec::new(),
        };
        assert_eq!(roundtrip(&req), req);
    }

    #[test]
    fn generate_music_request_postcard_roundtrip() {
        let req = GenerateMusicRequest {
            envelope_version: MODEL_ENVELOPE_VERSION,
            model_id: "musicgen".to_owned(),
            prompt: "lofi beats".to_owned(),
            duration_secs: Some(15.0),
            seed: Some(7),
            extra_json: Vec::new(),
        };
        assert_eq!(roundtrip(&req), req);
    }

    #[test]
    fn transcribe_request_postcard_roundtrip() {
        let req = TranscribeRequest {
            envelope_version: MODEL_ENVELOPE_VERSION,
            model_id: "whisper".to_owned(),
            audio: vec![0u8; 1024],
            mime: "audio/wav".to_owned(),
            language: Some("en".to_owned()),
            extra_json: Vec::new(),
        };
        assert_eq!(roundtrip(&req), req);
    }

    #[test]
    fn upload_blob_chunk_postcard_roundtrip() {
        let start = UploadBlobChunk::Start {
            envelope_version: MODEL_ENVELOPE_VERSION,
            blob_id: "adapter-1".to_owned(),
            total_bytes: Some(4_096),
            content_type: Some("application/octet-stream".to_owned()),
        };
        assert_eq!(roundtrip(&start), start);
        let data = UploadBlobChunk::Data {
            envelope_version: MODEL_ENVELOPE_VERSION,
            bytes: vec![1, 2, 3, 4],
        };
        assert_eq!(roundtrip(&data), data);
        let end = UploadBlobChunk::End {
            envelope_version: MODEL_ENVELOPE_VERSION,
        };
        assert_eq!(roundtrip(&end), end);
    }

    #[test]
    fn fetch_blob_request_postcard_roundtrip() {
        let req = FetchBlobRequest {
            envelope_version: MODEL_ENVELOPE_VERSION,
            blob_id: "adapter-1".to_owned(),
            offset: Some(1024),
            chunk_size: Some(64 * 1024),
        };
        assert_eq!(roundtrip(&req), req);
    }

    #[test]
    fn fetch_blob_chunk_postcard_roundtrip() {
        let start = FetchBlobChunk::Start {
            envelope_version: MODEL_ENVELOPE_VERSION,
            blob_id: "adapter-1".to_owned(),
            total_bytes: Some(4_096),
            content_type: None,
        };
        assert_eq!(roundtrip(&start), start);
        let data = FetchBlobChunk::Data {
            envelope_version: MODEL_ENVELOPE_VERSION,
            bytes: vec![9, 8, 7],
        };
        assert_eq!(roundtrip(&data), data);
        let end = FetchBlobChunk::End {
            envelope_version: MODEL_ENVELOPE_VERSION,
        };
        assert_eq!(roundtrip(&end), end);
    }

    #[test]
    fn rpc_error_postcard_roundtrip() {
        let err = RpcError::not_found("no such model 'qwen'");
        assert_eq!(roundtrip(&err), err);
        assert_eq!(err.code, RPC_ERR_NOT_FOUND);
    }

    #[test]
    fn rpc_result_postcard_roundtrip() {
        let ok: RpcResult<LoadResponse> = RpcResult::Ok(LoadResponse {
            envelope_version: MODEL_ENVELOPE_VERSION,
        });
        let bytes = postcard::to_allocvec(&ok).unwrap();
        let decoded: RpcResult<LoadResponse> = postcard::from_bytes(&bytes).unwrap();
        assert!(matches!(decoded, RpcResult::Ok(_)));

        let err: RpcResult<LoadResponse> = RpcResult::Err(RpcError::invalid("bad model id"));
        let bytes = postcard::to_allocvec(&err).unwrap();
        let decoded: RpcResult<LoadResponse> = postcard::from_bytes(&bytes).unwrap();
        assert!(matches!(decoded, RpcResult::Err(e) if e.code == RPC_ERR_INVALID));
    }

    #[test]
    fn envelope_version_accepts_current() {
        assert!(validate_model_envelope_version(MODEL_ENVELOPE_VERSION).is_ok());
        assert!(validate_model_envelope_version(0).is_ok());
    }

    #[test]
    fn envelope_version_rejects_newer() {
        let err = validate_model_envelope_version(MODEL_ENVELOPE_VERSION + 1).unwrap_err();
        assert_eq!(err.code, RPC_ERR_INCOMPATIBLE);
        assert!(!err.retryable);
    }
}
