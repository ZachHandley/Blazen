//! Error conversion utilities for napi-rs.
//!
//! Converts internal `Blazen` errors into [`napi::Error`] for the Node.js side.
//!
//! ## Provider error sentinel protocol
//!
//! napi-rs 3's [`napi::Error`] cannot carry arbitrary typed fields. For
//! [`BlazenError::Provider`] and [`BlazenError::ProviderHttp`] we embed a
//! JSON payload in the error message, prefixed with
//! [`PROVIDER_ERROR_SENTINEL`]. A hand-written JS wrapper
//! (`crates/blazen-node/errors.js`) detects the sentinel and re-throws a
//! typed `ProviderError` with `.provider`, `.status`, `.endpoint`,
//! `.requestId`, `.detail`, `.retryAfterMs` attributes.
//!
//! Raw message format:
//!
//! ```text
//! __BLAZEN_PROVIDER_ERROR__ {"provider":"fal","status":503,...}
//! [ProviderError] fal HTTP 503 at https://fal.run/x: service unavailable (request-id=abc)
//! ```
//!
//! Consumers who don't use the wrapper still get a readable message at
//! the end (minus the sentinel line).

use napi::Status;
use serde::Serialize;

/// Sentinel prefix on provider-error messages. The JS wrapper at
/// `crates/blazen-node/errors.js` pattern-matches on this. Keep in sync.
pub const PROVIDER_ERROR_SENTINEL: &str = "__BLAZEN_PROVIDER_ERROR__";

/// Structured payload embedded in a provider-error message's JSON line.
/// Field names use camelCase to match the receiving JS convention.
#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct ProviderErrorPayload<'a> {
    provider: &'a str,
    status: Option<u16>,
    endpoint: Option<&'a str>,
    request_id: Option<&'a str>,
    detail: Option<&'a str>,
    retry_after_ms: Option<u64>,
    // raw_body intentionally omitted — 4 KiB of JSON in an error message
    // is noisy. JS consumers who need it can re-inspect the Rust error.
}

/// Convert any `Display`-able error into a [`napi::Error`].
pub fn to_napi_error(err: impl std::fmt::Display) -> napi::Error {
    napi::Error::new(Status::GenericFailure, err.to_string())
}

/// Convert a [`WorkflowError`](blazen_core::WorkflowError) into a [`napi::Error`].
///
/// Intentionally takes by value for use with `map_err`.
#[must_use]
#[allow(clippy::needless_pass_by_value)]
pub fn workflow_error_to_napi(err: blazen_core::WorkflowError) -> napi::Error {
    napi::Error::new(Status::GenericFailure, err.to_string())
}

/// Convert a [`blazen_pipeline::PipelineError`] to a [`napi::Error`].
#[must_use]
#[allow(clippy::needless_pass_by_value)]
pub fn pipeline_error_to_napi(err: blazen_pipeline::PipelineError) -> napi::Error {
    napi::Error::new(Status::GenericFailure, err.to_string())
}

/// Convert a [`blazen_peer::PeerError`] to a [`napi::Error`].
///
/// The error class name is included as a prefix so JS consumers can
/// distinguish transport, encoding, TLS, envelope-version, workflow,
/// and unknown-step failures from the message text.
///
/// Available on every target — `PeerError`'s variants are wasi-compatible
/// (no tonic / rustls types in the public surface), so the wasi
/// HTTP/JSON peer client can route errors through the same mapper.
#[must_use]
#[allow(clippy::needless_pass_by_value)]
pub fn peer_error_to_napi(err: blazen_peer::PeerError) -> napi::Error {
    use blazen_peer::PeerError;
    let prefix = match &err {
        PeerError::Encode(_) => "PeerEncodeError",
        PeerError::Transport(_) => "PeerTransportError",
        PeerError::EnvelopeVersion { .. } => "PeerEnvelopeVersionError",
        PeerError::Workflow(_) => "PeerWorkflowError",
        PeerError::Tls(_) => "PeerTlsError",
        PeerError::UnknownStep { .. } => "PeerUnknownStepError",
    };
    napi::Error::new(Status::GenericFailure, format!("[{prefix}] {err}"))
}

/// Convert a [`BlazenError`](blazen_llm::BlazenError) into a [`napi::Error`].
///
/// For `Provider` / `ProviderHttp` variants, embeds a JSON payload with
/// structured fields (see module docs). For other variants, prefixes the
/// message with the error class name for readable JS logs.
///
/// Intentionally takes by value for use with `map_err`.
#[must_use]
#[allow(clippy::needless_pass_by_value)]
pub fn blazen_error_to_napi(err: blazen_llm::BlazenError) -> napi::Error {
    use blazen_llm::BlazenError;

    // Fast path — provider errors get a structured sentinel payload.
    if let BlazenError::ProviderHttp(d) = &err {
        let payload = ProviderErrorPayload {
            provider: d.provider.as_ref(),
            status: Some(d.status),
            endpoint: Some(d.endpoint.as_str()),
            request_id: d.request_id.as_deref(),
            detail: d.detail.as_deref(),
            retry_after_ms: d.retry_after_ms,
        };
        let json = serde_json::to_string(&payload).unwrap_or_else(|_| "{}".to_string());
        let readable = err.to_string();
        return napi::Error::new(
            Status::GenericFailure,
            format!("{PROVIDER_ERROR_SENTINEL} {json}\n[ProviderError] {readable}"),
        );
    }

    if let BlazenError::Provider {
        provider,
        status_code,
        ..
    } = &err
    {
        let payload = ProviderErrorPayload {
            provider: provider.as_str(),
            status: *status_code,
            endpoint: None,
            request_id: None,
            detail: None,
            retry_after_ms: None,
        };
        let json = serde_json::to_string(&payload).unwrap_or_else(|_| "{}".to_string());
        let readable = err.to_string();
        return napi::Error::new(
            Status::GenericFailure,
            format!("{PROVIDER_ERROR_SENTINEL} {json}\n[ProviderError] {readable}"),
        );
    }

    let prefix = match &err {
        BlazenError::Auth { .. } => "AuthError",
        BlazenError::RateLimit { .. } => "RateLimitError",
        BlazenError::Timeout { .. } => "TimeoutError",
        BlazenError::Validation { .. } => "ValidationError",
        BlazenError::ContentPolicy { .. } => "ContentPolicyError",
        BlazenError::Unsupported { .. } => "UnsupportedError",
        _ => "BlazenError",
    };
    napi::Error::new(Status::GenericFailure, format!("[{prefix}] {err}"))
}

/// Backwards-compatible alias for [`blazen_error_to_napi`].
#[must_use]
#[allow(clippy::needless_pass_by_value)]
pub fn llm_error_to_napi(err: blazen_llm::BlazenError) -> napi::Error {
    blazen_error_to_napi(err)
}

/// Convert a [`blazen_persist::PersistError`] to a [`napi::Error`].
#[cfg(not(target_os = "wasi"))]
#[must_use]
#[allow(clippy::needless_pass_by_value)]
pub fn persist_error_to_napi(err: blazen_persist::PersistError) -> napi::Error {
    napi::Error::new(Status::GenericFailure, format!("[PersistError] {err}"))
}

/// Convert a [`blazen_prompts::PromptError`] to a [`napi::Error`].
///
/// The error class name is included as a prefix so JS consumers can
/// distinguish missing-variable, not-found, version-not-found, IO,
/// YAML/JSON parse, and validation failures from the message text.
#[must_use]
#[allow(clippy::needless_pass_by_value)]
pub fn prompt_error_to_napi(err: blazen_prompts::PromptError) -> napi::Error {
    use blazen_prompts::PromptError;
    let prefix = match &err {
        PromptError::MissingVariable { .. } => "PromptMissingVariableError",
        PromptError::NotFound { .. } => "PromptNotFoundError",
        PromptError::VersionNotFound { .. } => "PromptVersionNotFoundError",
        PromptError::Io(_) => "PromptIoError",
        PromptError::Yaml(_) => "PromptYamlError",
        PromptError::Json(_) => "PromptJsonError",
        PromptError::Validation(_) => "PromptValidationError",
    };
    napi::Error::new(Status::GenericFailure, format!("[{prefix}] {err}"))
}

/// Convert a [`blazen_memory::MemoryError`] to a [`napi::Error`].
///
/// Prefixes the message with the variant name so JS consumers can
/// distinguish missing-embedder, ELID, embedding, not-found, serialization,
/// I/O, and backend failures from the message text.
#[must_use]
#[allow(clippy::needless_pass_by_value)]
pub fn memory_error_to_napi(err: blazen_memory::MemoryError) -> napi::Error {
    use blazen_memory::MemoryError;
    let prefix = match &err {
        MemoryError::NoEmbedder => "MemoryNoEmbedderError",
        MemoryError::Elid(_) => "MemoryElidError",
        MemoryError::Embedding(_) => "MemoryEmbeddingError",
        MemoryError::NotFound(_) => "MemoryNotFoundError",
        MemoryError::Serialization(_) => "MemorySerializationError",
        MemoryError::Io(_) => "MemoryIoError",
        MemoryError::Backend(_) => "MemoryBackendError",
    };
    napi::Error::new(Status::GenericFailure, format!("[{prefix}] {err}"))
}

// ---------------------------------------------------------------------------
// Local backend provider error mappers (feature-gated).
//
// Each mapper wraps the upstream variant name as a class-name prefix so JS
// consumers can pattern-match on the leading `[XxxError]` token without
// parsing the full message body. The provider modules themselves currently
// stringify via `e.to_string()` for `from_options` setup errors; these
// helpers exist for downstream callers (and any future per-call error
// conversion) that need richer classification.
// ---------------------------------------------------------------------------

/// Convert a [`blazen_llm::MistralRsError`] to a [`napi::Error`].
#[cfg(feature = "mistralrs")]
#[must_use]
#[allow(clippy::needless_pass_by_value)]
pub fn mistralrs_error_to_napi(err: blazen_llm::MistralRsError) -> napi::Error {
    use blazen_llm::MistralRsError;
    let prefix = match &err {
        MistralRsError::InvalidOptions(_) => "MistralRsInvalidOptionsError",
        MistralRsError::Init(_) => "MistralRsInitError",
        MistralRsError::Inference(_) => "MistralRsInferenceError",
        MistralRsError::EngineNotAvailable => "MistralRsEngineNotAvailableError",
    };
    napi::Error::new(Status::GenericFailure, format!("[{prefix}] {err}"))
}

/// Convert a [`blazen_llm::CandleLlmError`] to a [`napi::Error`].
#[cfg(feature = "candle-llm")]
#[must_use]
#[allow(clippy::needless_pass_by_value)]
pub fn candle_llm_error_to_napi(err: blazen_llm::CandleLlmError) -> napi::Error {
    use blazen_llm::CandleLlmError;
    let prefix = match &err {
        CandleLlmError::InvalidOptions(_) => "CandleLlmInvalidOptionsError",
        CandleLlmError::ModelLoad(_) => "CandleLlmModelLoadError",
        CandleLlmError::Inference(_) => "CandleLlmInferenceError",
        CandleLlmError::EngineNotAvailable => "CandleLlmEngineNotAvailableError",
    };
    napi::Error::new(Status::GenericFailure, format!("[{prefix}] {err}"))
}

/// Convert a [`blazen_llm::CandleEmbedError`] to a [`napi::Error`].
#[cfg(feature = "candle-embed")]
#[must_use]
#[allow(clippy::needless_pass_by_value)]
pub fn candle_embed_error_to_napi(err: blazen_llm::CandleEmbedError) -> napi::Error {
    use blazen_llm::CandleEmbedError;
    let prefix = match &err {
        CandleEmbedError::InvalidOptions(_) => "CandleEmbedInvalidOptionsError",
        CandleEmbedError::ModelLoad(_) => "CandleEmbedModelLoadError",
        CandleEmbedError::Embedding(_) => "CandleEmbedEmbeddingError",
        CandleEmbedError::EngineNotAvailable => "CandleEmbedEngineNotAvailableError",
        CandleEmbedError::TaskPanicked(_) => "CandleEmbedTaskPanickedError",
    };
    napi::Error::new(Status::GenericFailure, format!("[{prefix}] {err}"))
}

/// Convert a [`blazen_llm::LlamaCppError`] to a [`napi::Error`].
#[cfg(feature = "llamacpp")]
#[must_use]
#[allow(clippy::needless_pass_by_value)]
pub fn llamacpp_error_to_napi(err: blazen_llm::LlamaCppError) -> napi::Error {
    use blazen_llm::LlamaCppError;
    let prefix = match &err {
        LlamaCppError::InvalidOptions(_) => "LlamaCppInvalidOptionsError",
        LlamaCppError::ModelLoad(_) => "LlamaCppModelLoadError",
        LlamaCppError::Inference(_) => "LlamaCppInferenceError",
        LlamaCppError::EngineNotAvailable => "LlamaCppEngineNotAvailableError",
    };
    napi::Error::new(Status::GenericFailure, format!("[{prefix}] {err}"))
}

/// Convert a [`blazen_llm::WhisperError`] to a [`napi::Error`].
#[cfg(feature = "whispercpp")]
#[must_use]
#[allow(clippy::needless_pass_by_value)]
pub fn whisper_error_to_napi(err: blazen_llm::WhisperError) -> napi::Error {
    use blazen_llm::WhisperError;
    let prefix = match &err {
        WhisperError::EngineNotAvailable => "WhisperEngineNotAvailableError",
        WhisperError::InvalidOptions(_) => "WhisperInvalidOptionsError",
        WhisperError::ModelLoad(_) => "WhisperModelLoadError",
        WhisperError::Transcription(_) => "WhisperTranscriptionError",
        WhisperError::Io(_) => "WhisperIoError",
    };
    napi::Error::new(Status::GenericFailure, format!("[{prefix}] {err}"))
}

/// Convert a [`blazen_llm::DiffusionError`] to a [`napi::Error`].
#[cfg(feature = "diffusion")]
#[must_use]
#[allow(clippy::needless_pass_by_value)]
pub fn diffusion_error_to_napi(err: blazen_llm::DiffusionError) -> napi::Error {
    use blazen_llm::DiffusionError;
    let prefix = match &err {
        DiffusionError::InvalidOptions(_) => "DiffusionInvalidOptionsError",
        DiffusionError::ModelLoad(_) => "DiffusionModelLoadError",
        DiffusionError::Generation(_) => "DiffusionGenerationError",
    };
    napi::Error::new(Status::GenericFailure, format!("[{prefix}] {err}"))
}

/// Convert a [`blazen_llm::PiperError`] to a [`napi::Error`].
#[cfg(feature = "piper")]
#[must_use]
#[allow(clippy::needless_pass_by_value)]
pub fn piper_error_to_napi(err: blazen_llm::PiperError) -> napi::Error {
    use blazen_llm::PiperError;
    let prefix = match &err {
        PiperError::InvalidOptions(_) => "PiperInvalidOptionsError",
        PiperError::ModelLoad(_) => "PiperModelLoadError",
        PiperError::Synthesis(_) => "PiperSynthesisError",
        PiperError::EngineNotAvailable => "PiperEngineNotAvailableError",
    };
    napi::Error::new(Status::GenericFailure, format!("[{prefix}] {err}"))
}

/// Convert a [`blazen_llm::EmbedError`] (fastembed on non-musl, tract on musl)
/// to a [`napi::Error`].
#[cfg(feature = "embed")]
#[must_use]
#[allow(clippy::needless_pass_by_value)]
pub fn embed_error_to_napi(err: blazen_llm::EmbedError) -> napi::Error {
    use blazen_llm::EmbedError;
    let prefix = match &err {
        EmbedError::UnknownModel(_) => "EmbedUnknownModelError",
        EmbedError::Init(_) => "EmbedInitError",
        EmbedError::Embed(_) => "EmbedEmbedError",
        EmbedError::MutexPoisoned(_) => "EmbedMutexPoisonedError",
        EmbedError::TaskPanicked(_) => "EmbedTaskPanickedError",
    };
    napi::Error::new(Status::GenericFailure, format!("[{prefix}] {err}"))
}
