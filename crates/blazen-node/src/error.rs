//! Error conversion utilities for napi-rs.
//!
//! Converts internal `Blazen` errors into [`napi::Error`] tagged with a
//! registered class name via [`napi::Error::with_class`]. The conversion
//! path in napi-patched looks the class up in the runtime registry
//! (populated at module load by
//! [`crate::error_classes::register_all_classes`]) and constructs an
//! instance of that class as the JS throw value — so JS callers see
//! `instanceof ProviderError === true` with structured fields as own
//! properties, without any post-build shim.
//!
//! ## Caller errors
//!
//! When a user-supplied JS callback (tool handler, etc.) throws or rejects,
//! the napi binding stashes the original `napi::Error` (whose `maybe_raw`
//! field points at the original JS exception) in
//! [`crate::error_classes::CALLER_ERROR_STASH`] keyed by a fresh UUID, and
//! carries that UUID through `BlazenError::CallerError` until the agent
//! loop bubbles back to the napi boundary. [`blazen_caller_error_to_napi`]
//! then pops the original error out of the stash and returns it directly,
//! so the JS caller sees its exact original error instance — preserving
//! `instanceof MyError`, custom prototype chain, and own-properties.

use napi::Status;

use crate::error_classes::take_caller_error;

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

/// Convert a [`blazen_peer::PeerError`] to a [`napi::Error`] using the
/// runtime error-class registry.
///
/// Available on every target — `PeerError`'s variants are wasi-compatible
/// (no tonic / rustls types in the public surface), so the wasi
/// HTTP/JSON peer client can route errors through the same mapper.
#[must_use]
#[allow(clippy::needless_pass_by_value)]
pub fn peer_error_to_napi(err: blazen_peer::PeerError) -> napi::Error {
    use blazen_peer::PeerError;
    let class = match &err {
        PeerError::Encode(_) => "PeerEncodeError",
        PeerError::Transport(_) => "PeerTransportError",
        PeerError::EnvelopeVersion { .. } => "PeerEnvelopeVersionError",
        PeerError::Workflow(_) => "PeerWorkflowError",
        PeerError::Tls(_) => "PeerTlsError",
        PeerError::UnknownStep { .. } => "PeerUnknownStepError",
    };
    napi::Error::with_class(class, err.to_string())
}

/// Convert a [`BlazenError`](blazen_llm::BlazenError) into a [`napi::Error`]
/// tagged with the appropriate registered class. `ProviderHttp` carries the
/// full set of structured fields (provider, status, endpoint, requestId,
/// detail, retryAfterMs); other variants carry just the class tag.
///
/// Intentionally takes by value for use with `map_err`.
#[must_use]
#[allow(clippy::needless_pass_by_value)]
pub fn blazen_error_to_napi(err: blazen_llm::BlazenError) -> napi::Error {
    use blazen_llm::BlazenError;

    if let BlazenError::ProviderHttp(d) = &err {
        return napi::Error::with_class("ProviderError", err.to_string())
            .with_field("provider", d.provider.as_ref())
            .with_field("status", d.status)
            .with_field("endpoint", d.endpoint.as_str())
            .with_field_opt("requestId", d.request_id.as_deref())
            .with_field_opt("detail", d.detail.as_deref())
            .with_field_opt("retryAfterMs", d.retry_after_ms);
    }

    if let BlazenError::Provider {
        provider,
        status_code,
        ..
    } = &err
    {
        return napi::Error::with_class("ProviderError", err.to_string())
            .with_field("provider", provider.as_str())
            .with_field_opt("status", *status_code);
    }

    let class = match &err {
        BlazenError::Auth { .. } => "AuthError",
        BlazenError::RateLimit { .. } => "RateLimitError",
        BlazenError::Timeout { .. } => "TimeoutError",
        BlazenError::Validation { .. } => "ValidationError",
        BlazenError::ContentPolicy { .. } => "ContentPolicyError",
        BlazenError::Unsupported { .. } => "UnsupportedError",
        _ => "BlazenError",
    };
    napi::Error::with_class(class, err.to_string())
}

/// Backwards-compatible alias for [`blazen_error_to_napi`].
#[must_use]
#[allow(clippy::needless_pass_by_value)]
pub fn llm_error_to_napi(err: blazen_llm::BlazenError) -> napi::Error {
    blazen_error_to_napi(err)
}

/// Convert a [`BlazenError::CallerError`] carrying a UUID source (set by
/// the napi tool wrapper after a JS callback threw or rejected) back into
/// the original `napi::Error` — preserving the JS error instance via the
/// `maybe_raw` reference.
///
/// On a `CallerError` whose source downcasts to a stash UUID, looks the
/// original error up in [`crate::error_classes::CALLER_ERROR_STASH`] and
/// returns it directly. JS callers see their exact `MyError` instance with
/// `instanceof MyError === true` and all own-properties intact.
///
/// Other variants fall through to [`blazen_error_to_napi`].
#[must_use]
#[allow(clippy::needless_pass_by_value)]
pub fn blazen_caller_error_to_napi(err: blazen_llm::BlazenError) -> napi::Error {
    if let blazen_llm::BlazenError::CallerError {
        source: Some(s), ..
    } = &err
        && let Some(uuid) = (**s).downcast_ref::<String>()
        && let Some(original) = take_caller_error(uuid)
    {
        return original;
    }
    blazen_error_to_napi(err)
}

/// Convert a [`blazen_persist::PersistError`] to a [`napi::Error`].
#[cfg(not(target_os = "wasi"))]
#[must_use]
#[allow(clippy::needless_pass_by_value)]
pub fn persist_error_to_napi(err: blazen_persist::PersistError) -> napi::Error {
    napi::Error::with_class("PersistError", err.to_string())
}

/// Convert a [`blazen_prompts::PromptError`] to a [`napi::Error`] tagged
/// with the appropriate subclass.
#[must_use]
#[allow(clippy::needless_pass_by_value)]
pub fn prompt_error_to_napi(err: blazen_prompts::PromptError) -> napi::Error {
    use blazen_prompts::PromptError;
    let class = match &err {
        PromptError::MissingVariable { .. } => "PromptMissingVariableError",
        PromptError::NotFound { .. } => "PromptNotFoundError",
        PromptError::VersionNotFound { .. } => "PromptVersionNotFoundError",
        PromptError::Io(_) => "PromptIoError",
        PromptError::Yaml(_) => "PromptYamlError",
        PromptError::Json(_) => "PromptJsonError",
        PromptError::Validation(_) => "PromptValidationError",
    };
    napi::Error::with_class(class, err.to_string())
}

/// Convert a [`blazen_memory::MemoryError`] to a [`napi::Error`] tagged
/// with the appropriate subclass.
#[must_use]
#[allow(clippy::needless_pass_by_value)]
pub fn memory_error_to_napi(err: blazen_memory::MemoryError) -> napi::Error {
    use blazen_memory::MemoryError;
    let class = match &err {
        MemoryError::NoEmbedder => "MemoryNoEmbedderError",
        MemoryError::Elid(_) => "MemoryElidError",
        MemoryError::Embedding(_) => "MemoryEmbeddingError",
        MemoryError::NotFound(_) => "MemoryNotFoundError",
        MemoryError::Serialization(_) => "MemorySerializationError",
        MemoryError::Io(_) => "MemoryIoError",
        MemoryError::Backend(_) => "MemoryBackendError",
    };
    napi::Error::with_class(class, err.to_string())
}

// ---------------------------------------------------------------------------
// Local backend provider error mappers (feature-gated).
// ---------------------------------------------------------------------------

/// Convert a [`blazen_llm::MistralRsError`] to a [`napi::Error`].
#[cfg(feature = "mistralrs")]
#[must_use]
#[allow(clippy::needless_pass_by_value)]
pub fn mistralrs_error_to_napi(err: blazen_llm::MistralRsError) -> napi::Error {
    use blazen_llm::MistralRsError;
    let class = match &err {
        MistralRsError::InvalidOptions(_) => "MistralRsInvalidOptionsError",
        MistralRsError::Init(_) => "MistralRsInitError",
        MistralRsError::Inference(_) => "MistralRsInferenceError",
        MistralRsError::EngineNotAvailable => "MistralRsEngineNotAvailableError",
        MistralRsError::AdapterFailed(_) => "MistralRsAdapterFailedError",
    };
    napi::Error::with_class(class, err.to_string())
}

/// Convert a [`blazen_llm::CandleLlmError`] to a [`napi::Error`].
#[cfg(feature = "candle-llm")]
#[must_use]
#[allow(clippy::needless_pass_by_value)]
pub fn candle_llm_error_to_napi(err: blazen_llm::CandleLlmError) -> napi::Error {
    use blazen_llm::CandleLlmError;
    let class = match &err {
        CandleLlmError::InvalidOptions(_) => "CandleLlmInvalidOptionsError",
        CandleLlmError::ModelLoad(_) => "CandleLlmModelLoadError",
        CandleLlmError::Inference(_) => "CandleLlmInferenceError",
        CandleLlmError::Unsupported(_) => "CandleLlmUnsupportedError",
        CandleLlmError::EngineNotAvailable => "CandleLlmEngineNotAvailableError",
    };
    napi::Error::with_class(class, err.to_string())
}

/// Convert a [`blazen_llm::CandleEmbedError`] to a [`napi::Error`].
#[cfg(feature = "candle-embed")]
#[must_use]
#[allow(clippy::needless_pass_by_value)]
pub fn candle_embed_error_to_napi(err: blazen_llm::CandleEmbedError) -> napi::Error {
    use blazen_llm::CandleEmbedError;
    let class = match &err {
        CandleEmbedError::InvalidOptions(_) => "CandleEmbedInvalidOptionsError",
        CandleEmbedError::ModelLoad(_) => "CandleEmbedModelLoadError",
        CandleEmbedError::Embedding(_) => "CandleEmbedEmbeddingError",
        CandleEmbedError::EngineNotAvailable => "CandleEmbedEngineNotAvailableError",
        CandleEmbedError::TaskPanicked(_) => "CandleEmbedTaskPanickedError",
    };
    napi::Error::with_class(class, err.to_string())
}

/// Convert a [`blazen_llm::LlamaCppError`] to a [`napi::Error`].
#[cfg(feature = "llamacpp")]
#[must_use]
#[allow(clippy::needless_pass_by_value)]
pub fn llamacpp_error_to_napi(err: blazen_llm::LlamaCppError) -> napi::Error {
    use blazen_llm::LlamaCppError;
    let class = match &err {
        LlamaCppError::InvalidOptions(_) => "LlamaCppInvalidOptionsError",
        LlamaCppError::ModelLoad(_) => "LlamaCppModelLoadError",
        LlamaCppError::Inference(_) => "LlamaCppInferenceError",
        LlamaCppError::EngineNotAvailable => "LlamaCppEngineNotAvailableError",
        LlamaCppError::AdapterFailed(_) => "LlamaCppAdapterFailedError",
    };
    napi::Error::with_class(class, err.to_string())
}

/// Convert a [`blazen_llm::WhisperError`] to a [`napi::Error`].
#[cfg(feature = "whispercpp")]
#[must_use]
#[allow(clippy::needless_pass_by_value)]
pub fn whisper_error_to_napi(err: blazen_llm::WhisperError) -> napi::Error {
    use blazen_llm::WhisperError;
    let class = match &err {
        WhisperError::EngineNotAvailable => "WhisperEngineNotAvailableError",
        WhisperError::InvalidOptions(_) => "WhisperInvalidOptionsError",
        WhisperError::ModelLoad(_) => "WhisperModelLoadError",
        WhisperError::Transcription(_) => "WhisperTranscriptionError",
        WhisperError::Io(_) => "WhisperIoError",
    };
    napi::Error::with_class(class, err.to_string())
}

/// Convert a [`blazen_llm::DiffusionError`] to a [`napi::Error`].
#[cfg(feature = "diffusion")]
#[must_use]
#[allow(clippy::needless_pass_by_value)]
pub fn diffusion_error_to_napi(err: blazen_llm::DiffusionError) -> napi::Error {
    use blazen_llm::DiffusionError;
    let class = match &err {
        DiffusionError::InvalidOptions(_) => "DiffusionInvalidOptionsError",
        DiffusionError::ModelLoad(_) => "DiffusionModelLoadError",
        DiffusionError::Generation(_) => "DiffusionGenerationError",
        DiffusionError::EngineNotAvailable => "DiffusionEngineNotAvailableError",
    };
    napi::Error::with_class(class, err.to_string())
}

/// Convert a [`blazen_llm::TtsError`] to a [`napi::Error`].
#[cfg(feature = "tts")]
#[must_use]
#[allow(clippy::needless_pass_by_value)]
pub fn tts_error_to_napi(err: blazen_llm::TtsError) -> napi::Error {
    use blazen_llm::TtsError;
    let class = match &err {
        TtsError::InvalidOptions(_) => "TtsInvalidOptionsError",
        TtsError::ModelLoad(_) => "TtsModelLoadError",
        TtsError::Synthesis(_) => "TtsSynthesisError",
        TtsError::EngineNotAvailable => "TtsEngineNotAvailableError",
    };
    napi::Error::with_class(class, err.to_string())
}

/// Convert a [`blazen_llm::EmbedError`] (fastembed on non-musl, tract on musl)
/// to a [`napi::Error`].
#[cfg(feature = "embed")]
#[must_use]
#[allow(clippy::needless_pass_by_value)]
pub fn embed_error_to_napi(err: blazen_llm::EmbedError) -> napi::Error {
    use blazen_llm::EmbedError;
    let class = match &err {
        EmbedError::UnknownModel(_) => "EmbedUnknownModelError",
        EmbedError::Init(_) => "EmbedInitError",
        EmbedError::Embed(_) => "EmbedEmbedError",
        EmbedError::MutexPoisoned(_) => "EmbedMutexPoisonedError",
        EmbedError::TaskPanicked(_) => "EmbedTaskPanickedError",
    };
    napi::Error::with_class(class, err.to_string())
}
