'use strict'

// Typed JS error classes for Blazen.
//
// The Rust binding (see `crates/blazen-node/src/error.rs`) prefixes every
// `napi::Error` message with a `[ClassName]` tag. This file:
//   1. Defines a class hierarchy that mirrors the Python binding's
//      `pyo3::create_exception!` tree (the source of truth).
//   2. Exposes `enrichError(err)` which, given a plain `Error` thrown by a
//      napi-rs function, parses the leading `[Tag]` prefix and rethrows a
//      typed instance whose `message` is the original message minus the
//      prefix.
//
// Consumers can do `err instanceof BlazenError`, `err instanceof ProviderError`,
// or check the concrete leaf type (e.g. `LlamaCppModelLoadError`).
//
// The `[ProviderError]` tag is also emitted alongside a sentinel JSON line
// for HTTP/provider errors -- see `error.rs` `PROVIDER_ERROR_SENTINEL`.
// `enrichError` extracts the sentinel JSON (when present), attaches the
// structured fields (`provider`, `status`, `endpoint`, `requestId`,
// `detail`, `retryAfterMs`) onto the resulting `ProviderError` instance,
// and strips the sentinel line from the user-visible message.

const PROVIDER_ERROR_SENTINEL = '__BLAZEN_PROVIDER_ERROR__'

// ---------------------------------------------------------------------------
// Class hierarchy
// ---------------------------------------------------------------------------

class BlazenError extends Error {
  constructor(message) {
    super(message)
    this.name = 'BlazenError'
  }
}

class AuthError extends BlazenError {
  constructor(message) {
    super(message)
    this.name = 'AuthError'
  }
}

class RateLimitError extends BlazenError {
  constructor(message) {
    super(message)
    this.name = 'RateLimitError'
  }
}

class TimeoutError extends BlazenError {
  constructor(message) {
    super(message)
    this.name = 'TimeoutError'
  }
}

class ValidationError extends BlazenError {
  constructor(message) {
    super(message)
    this.name = 'ValidationError'
  }
}

class ContentPolicyError extends BlazenError {
  constructor(message) {
    super(message)
    this.name = 'ContentPolicyError'
  }
}

class UnsupportedError extends BlazenError {
  constructor(message) {
    super(message)
    this.name = 'UnsupportedError'
  }
}

class ComputeError extends BlazenError {
  constructor(message) {
    super(message)
    this.name = 'ComputeError'
  }
}

class MediaError extends BlazenError {
  constructor(message) {
    super(message)
    this.name = 'MediaError'
  }
}

class ProviderError extends BlazenError {
  constructor(message) {
    super(message)
    this.name = 'ProviderError'
    // Structured fields populated from the sentinel JSON when present.
    // Always defined (defaulting to `null`) so consumers can rely on the
    // shape without `in` / `hasOwnProperty` checks.
    this.provider = null
    this.status = null
    this.endpoint = null
    this.requestId = null
    this.detail = null
    this.retryAfterMs = null
  }
}

// Per-backend provider subclasses. All extend `ProviderError` so callers
// that don't care which backend failed can keep one `instanceof
// ProviderError` arm.

class LlamaCppError extends ProviderError {
  constructor(message) {
    super(message)
    this.name = 'LlamaCppError'
  }
}
class LlamaCppInvalidOptionsError extends LlamaCppError {
  constructor(message) {
    super(message)
    this.name = 'LlamaCppInvalidOptionsError'
  }
}
class LlamaCppModelLoadError extends LlamaCppError {
  constructor(message) {
    super(message)
    this.name = 'LlamaCppModelLoadError'
  }
}
class LlamaCppInferenceError extends LlamaCppError {
  constructor(message) {
    super(message)
    this.name = 'LlamaCppInferenceError'
  }
}
class LlamaCppEngineNotAvailableError extends LlamaCppError {
  constructor(message) {
    super(message)
    this.name = 'LlamaCppEngineNotAvailableError'
  }
}

class CandleLlmError extends ProviderError {
  constructor(message) {
    super(message)
    this.name = 'CandleLlmError'
  }
}
class CandleLlmInvalidOptionsError extends CandleLlmError {
  constructor(message) {
    super(message)
    this.name = 'CandleLlmInvalidOptionsError'
  }
}
class CandleLlmModelLoadError extends CandleLlmError {
  constructor(message) {
    super(message)
    this.name = 'CandleLlmModelLoadError'
  }
}
class CandleLlmInferenceError extends CandleLlmError {
  constructor(message) {
    super(message)
    this.name = 'CandleLlmInferenceError'
  }
}
class CandleLlmEngineNotAvailableError extends CandleLlmError {
  constructor(message) {
    super(message)
    this.name = 'CandleLlmEngineNotAvailableError'
  }
}

class CandleEmbedError extends ProviderError {
  constructor(message) {
    super(message)
    this.name = 'CandleEmbedError'
  }
}
class CandleEmbedInvalidOptionsError extends CandleEmbedError {
  constructor(message) {
    super(message)
    this.name = 'CandleEmbedInvalidOptionsError'
  }
}
class CandleEmbedModelLoadError extends CandleEmbedError {
  constructor(message) {
    super(message)
    this.name = 'CandleEmbedModelLoadError'
  }
}
class CandleEmbedEmbeddingError extends CandleEmbedError {
  constructor(message) {
    super(message)
    this.name = 'CandleEmbedEmbeddingError'
  }
}
class CandleEmbedEngineNotAvailableError extends CandleEmbedError {
  constructor(message) {
    super(message)
    this.name = 'CandleEmbedEngineNotAvailableError'
  }
}
class CandleEmbedTaskPanickedError extends CandleEmbedError {
  constructor(message) {
    super(message)
    this.name = 'CandleEmbedTaskPanickedError'
  }
}

class MistralRsError extends ProviderError {
  constructor(message) {
    super(message)
    this.name = 'MistralRsError'
  }
}
class MistralRsInvalidOptionsError extends MistralRsError {
  constructor(message) {
    super(message)
    this.name = 'MistralRsInvalidOptionsError'
  }
}
class MistralRsInitError extends MistralRsError {
  constructor(message) {
    super(message)
    this.name = 'MistralRsInitError'
  }
}
class MistralRsInferenceError extends MistralRsError {
  constructor(message) {
    super(message)
    this.name = 'MistralRsInferenceError'
  }
}
class MistralRsEngineNotAvailableError extends MistralRsError {
  constructor(message) {
    super(message)
    this.name = 'MistralRsEngineNotAvailableError'
  }
}

class WhisperError extends ProviderError {
  constructor(message) {
    super(message)
    this.name = 'WhisperError'
  }
}
class WhisperInvalidOptionsError extends WhisperError {
  constructor(message) {
    super(message)
    this.name = 'WhisperInvalidOptionsError'
  }
}
class WhisperModelLoadError extends WhisperError {
  constructor(message) {
    super(message)
    this.name = 'WhisperModelLoadError'
  }
}
class WhisperTranscriptionError extends WhisperError {
  constructor(message) {
    super(message)
    this.name = 'WhisperTranscriptionError'
  }
}
class WhisperEngineNotAvailableError extends WhisperError {
  constructor(message) {
    super(message)
    this.name = 'WhisperEngineNotAvailableError'
  }
}
class WhisperIoError extends WhisperError {
  constructor(message) {
    super(message)
    this.name = 'WhisperIoError'
  }
}

class PiperError extends ProviderError {
  constructor(message) {
    super(message)
    this.name = 'PiperError'
  }
}
class PiperInvalidOptionsError extends PiperError {
  constructor(message) {
    super(message)
    this.name = 'PiperInvalidOptionsError'
  }
}
class PiperModelLoadError extends PiperError {
  constructor(message) {
    super(message)
    this.name = 'PiperModelLoadError'
  }
}
class PiperSynthesisError extends PiperError {
  constructor(message) {
    super(message)
    this.name = 'PiperSynthesisError'
  }
}
class PiperEngineNotAvailableError extends PiperError {
  constructor(message) {
    super(message)
    this.name = 'PiperEngineNotAvailableError'
  }
}

class DiffusionError extends ProviderError {
  constructor(message) {
    super(message)
    this.name = 'DiffusionError'
  }
}
class DiffusionInvalidOptionsError extends DiffusionError {
  constructor(message) {
    super(message)
    this.name = 'DiffusionInvalidOptionsError'
  }
}
class DiffusionModelLoadError extends DiffusionError {
  constructor(message) {
    super(message)
    this.name = 'DiffusionModelLoadError'
  }
}
class DiffusionGenerationError extends DiffusionError {
  constructor(message) {
    super(message)
    this.name = 'DiffusionGenerationError'
  }
}

// fastembed (non-musl) and tract (musl) both surface as `Embed*` tags
// from the Rust side. We expose them both as `FastEmbedError` (matching
// the Python binding's name) AND as the underlying `Embed*` tag classes
// so consumers who already check the per-tag class keep working.
class FastEmbedError extends ProviderError {
  constructor(message) {
    super(message)
    this.name = 'FastEmbedError'
  }
}
class EmbedUnknownModelError extends FastEmbedError {
  constructor(message) {
    super(message)
    this.name = 'EmbedUnknownModelError'
  }
}
class EmbedInitError extends FastEmbedError {
  constructor(message) {
    super(message)
    this.name = 'EmbedInitError'
  }
}
class EmbedEmbedError extends FastEmbedError {
  constructor(message) {
    super(message)
    this.name = 'EmbedEmbedError'
  }
}
class EmbedMutexPoisonedError extends FastEmbedError {
  constructor(message) {
    super(message)
    this.name = 'EmbedMutexPoisonedError'
  }
}
class EmbedTaskPanickedError extends FastEmbedError {
  constructor(message) {
    super(message)
    this.name = 'EmbedTaskPanickedError'
  }
}

class TractError extends ProviderError {
  constructor(message) {
    super(message)
    this.name = 'TractError'
  }
}

// Peer subsystem
class PeerEncodeError extends BlazenError {
  constructor(message) {
    super(message)
    this.name = 'PeerEncodeError'
  }
}
class PeerTransportError extends BlazenError {
  constructor(message) {
    super(message)
    this.name = 'PeerTransportError'
  }
}
class PeerEnvelopeVersionError extends BlazenError {
  constructor(message) {
    super(message)
    this.name = 'PeerEnvelopeVersionError'
  }
}
class PeerWorkflowError extends BlazenError {
  constructor(message) {
    super(message)
    this.name = 'PeerWorkflowError'
  }
}
class PeerTlsError extends BlazenError {
  constructor(message) {
    super(message)
    this.name = 'PeerTlsError'
  }
}
class PeerUnknownStepError extends BlazenError {
  constructor(message) {
    super(message)
    this.name = 'PeerUnknownStepError'
  }
}

// Persist
class PersistError extends BlazenError {
  constructor(message) {
    super(message)
    this.name = 'PersistError'
  }
}

// Prompts
class PromptError extends BlazenError {
  constructor(message) {
    super(message)
    this.name = 'PromptError'
  }
}
class PromptMissingVariableError extends PromptError {
  constructor(message) {
    super(message)
    this.name = 'PromptMissingVariableError'
  }
}
class PromptNotFoundError extends PromptError {
  constructor(message) {
    super(message)
    this.name = 'PromptNotFoundError'
  }
}
class PromptVersionNotFoundError extends PromptError {
  constructor(message) {
    super(message)
    this.name = 'PromptVersionNotFoundError'
  }
}
class PromptIoError extends PromptError {
  constructor(message) {
    super(message)
    this.name = 'PromptIoError'
  }
}
class PromptYamlError extends PromptError {
  constructor(message) {
    super(message)
    this.name = 'PromptYamlError'
  }
}
class PromptJsonError extends PromptError {
  constructor(message) {
    super(message)
    this.name = 'PromptJsonError'
  }
}
class PromptValidationError extends PromptError {
  constructor(message) {
    super(message)
    this.name = 'PromptValidationError'
  }
}

// Memory
class MemoryError extends BlazenError {
  constructor(message) {
    super(message)
    this.name = 'MemoryError'
  }
}
class MemoryNoEmbedderError extends MemoryError {
  constructor(message) {
    super(message)
    this.name = 'MemoryNoEmbedderError'
  }
}
class MemoryElidError extends MemoryError {
  constructor(message) {
    super(message)
    this.name = 'MemoryElidError'
  }
}
class MemoryEmbeddingError extends MemoryError {
  constructor(message) {
    super(message)
    this.name = 'MemoryEmbeddingError'
  }
}
class MemoryNotFoundError extends MemoryError {
  constructor(message) {
    super(message)
    this.name = 'MemoryNotFoundError'
  }
}
class MemorySerializationError extends MemoryError {
  constructor(message) {
    super(message)
    this.name = 'MemorySerializationError'
  }
}
class MemoryIoError extends MemoryError {
  constructor(message) {
    super(message)
    this.name = 'MemoryIoError'
  }
}
class MemoryBackendError extends MemoryError {
  constructor(message) {
    super(message)
    this.name = 'MemoryBackendError'
  }
}

// Model cache
class CacheError extends BlazenError {
  constructor(message) {
    super(message)
    this.name = 'CacheError'
  }
}
class DownloadError extends CacheError {
  constructor(message) {
    super(message)
    this.name = 'DownloadError'
  }
}
class CacheDirError extends CacheError {
  constructor(message) {
    super(message)
    this.name = 'CacheDirError'
  }
}
class IoError extends CacheError {
  constructor(message) {
    super(message)
    this.name = 'IoError'
  }
}

// ---------------------------------------------------------------------------
// Tag -> class registry
// ---------------------------------------------------------------------------

const TAG_TO_CLASS = {
  // Core
  BlazenError,
  AuthError,
  RateLimitError,
  TimeoutError,
  ValidationError,
  ContentPolicyError,
  UnsupportedError,
  ComputeError,
  MediaError,
  ProviderError,

  // LlamaCpp
  LlamaCppError,
  LlamaCppInvalidOptionsError,
  LlamaCppModelLoadError,
  LlamaCppInferenceError,
  LlamaCppEngineNotAvailableError,

  // Candle LLM
  CandleLlmError,
  CandleLlmInvalidOptionsError,
  CandleLlmModelLoadError,
  CandleLlmInferenceError,
  CandleLlmEngineNotAvailableError,

  // Candle Embed
  CandleEmbedError,
  CandleEmbedInvalidOptionsError,
  CandleEmbedModelLoadError,
  CandleEmbedEmbeddingError,
  CandleEmbedEngineNotAvailableError,
  CandleEmbedTaskPanickedError,

  // MistralRs
  MistralRsError,
  MistralRsInvalidOptionsError,
  MistralRsInitError,
  MistralRsInferenceError,
  MistralRsEngineNotAvailableError,

  // Whisper
  WhisperError,
  WhisperInvalidOptionsError,
  WhisperModelLoadError,
  WhisperTranscriptionError,
  WhisperEngineNotAvailableError,
  WhisperIoError,

  // Piper
  PiperError,
  PiperInvalidOptionsError,
  PiperModelLoadError,
  PiperSynthesisError,
  PiperEngineNotAvailableError,

  // Diffusion
  DiffusionError,
  DiffusionInvalidOptionsError,
  DiffusionModelLoadError,
  DiffusionGenerationError,

  // Fastembed / Tract
  FastEmbedError,
  EmbedUnknownModelError,
  EmbedInitError,
  EmbedEmbedError,
  EmbedMutexPoisonedError,
  EmbedTaskPanickedError,
  TractError,

  // Peer
  PeerEncodeError,
  PeerTransportError,
  PeerEnvelopeVersionError,
  PeerWorkflowError,
  PeerTlsError,
  PeerUnknownStepError,

  // Persist
  PersistError,

  // Prompts
  PromptError,
  PromptMissingVariableError,
  PromptNotFoundError,
  PromptVersionNotFoundError,
  PromptIoError,
  PromptYamlError,
  PromptJsonError,
  PromptValidationError,

  // Memory
  MemoryError,
  MemoryNoEmbedderError,
  MemoryElidError,
  MemoryEmbeddingError,
  MemoryNotFoundError,
  MemorySerializationError,
  MemoryIoError,
  MemoryBackendError,

  // Model cache
  CacheError,
  DownloadError,
  CacheDirError,
  IoError,
}

// Tag prefix regex. Matches `[Tag] rest-of-message`. The tag is restricted
// to ASCII letters/digits to avoid eating bracketed user content.
const TAG_RE = /^\[([A-Za-z][A-Za-z0-9]*)\]\s*([\s\S]*)$/

/**
 * Given an `Error` thrown by a napi-rs function, parse the Rust-side
 * `[Tag]` prefix and return a typed instance from the hierarchy above.
 *
 * If `err` is not an `Error`, or its message is not tagged, or the tag
 * is unknown, the original error is returned unchanged.
 */
function enrichError(err) {
  if (!(err instanceof Error)) {
    return err
  }
  let message = err.message || ''
  let providerPayload = null

  // Detect and strip the provider-error sentinel (a JSON line followed by
  // the human-readable `[ProviderError] ...` line).
  if (message.startsWith(PROVIDER_ERROR_SENTINEL)) {
    const newlineIdx = message.indexOf('\n')
    if (newlineIdx !== -1) {
      const jsonPart = message
        .slice(PROVIDER_ERROR_SENTINEL.length, newlineIdx)
        .trim()
      try {
        providerPayload = JSON.parse(jsonPart)
      } catch {
        providerPayload = null
      }
      message = message.slice(newlineIdx + 1)
    }
  }

  const match = TAG_RE.exec(message)
  if (!match) {
    return err
  }
  const tag = match[1]
  const rest = match[2]
  const Cls = TAG_TO_CLASS[tag]
  if (!Cls) {
    return err
  }
  const enriched = new Cls(rest)
  enriched.stack = err.stack
  // Preserve the original napi `code` (e.g. 'GenericFailure') if present.
  if (err.code !== undefined) {
    enriched.code = err.code
  }
  if (providerPayload && enriched instanceof ProviderError) {
    if (providerPayload.provider !== undefined) {
      enriched.provider = providerPayload.provider
    }
    if (providerPayload.status !== undefined) {
      enriched.status = providerPayload.status
    }
    if (providerPayload.endpoint !== undefined) {
      enriched.endpoint = providerPayload.endpoint
    }
    if (providerPayload.requestId !== undefined) {
      enriched.requestId = providerPayload.requestId
    }
    if (providerPayload.detail !== undefined) {
      enriched.detail = providerPayload.detail
    }
    if (providerPayload.retryAfterMs !== undefined) {
      enriched.retryAfterMs = providerPayload.retryAfterMs
    }
  }
  return enriched
}

module.exports = {
  // Core
  BlazenError,
  AuthError,
  RateLimitError,
  TimeoutError,
  ValidationError,
  ContentPolicyError,
  UnsupportedError,
  ComputeError,
  MediaError,
  ProviderError,

  // LlamaCpp
  LlamaCppError,
  LlamaCppInvalidOptionsError,
  LlamaCppModelLoadError,
  LlamaCppInferenceError,
  LlamaCppEngineNotAvailableError,

  // Candle LLM
  CandleLlmError,
  CandleLlmInvalidOptionsError,
  CandleLlmModelLoadError,
  CandleLlmInferenceError,
  CandleLlmEngineNotAvailableError,

  // Candle Embed
  CandleEmbedError,
  CandleEmbedInvalidOptionsError,
  CandleEmbedModelLoadError,
  CandleEmbedEmbeddingError,
  CandleEmbedEngineNotAvailableError,
  CandleEmbedTaskPanickedError,

  // MistralRs
  MistralRsError,
  MistralRsInvalidOptionsError,
  MistralRsInitError,
  MistralRsInferenceError,
  MistralRsEngineNotAvailableError,

  // Whisper
  WhisperError,
  WhisperInvalidOptionsError,
  WhisperModelLoadError,
  WhisperTranscriptionError,
  WhisperEngineNotAvailableError,
  WhisperIoError,

  // Piper
  PiperError,
  PiperInvalidOptionsError,
  PiperModelLoadError,
  PiperSynthesisError,
  PiperEngineNotAvailableError,

  // Diffusion
  DiffusionError,
  DiffusionInvalidOptionsError,
  DiffusionModelLoadError,
  DiffusionGenerationError,

  // Fastembed / Tract
  FastEmbedError,
  EmbedUnknownModelError,
  EmbedInitError,
  EmbedEmbedError,
  EmbedMutexPoisonedError,
  EmbedTaskPanickedError,
  TractError,

  // Peer
  PeerEncodeError,
  PeerTransportError,
  PeerEnvelopeVersionError,
  PeerWorkflowError,
  PeerTlsError,
  PeerUnknownStepError,

  // Persist
  PersistError,

  // Prompts
  PromptError,
  PromptMissingVariableError,
  PromptNotFoundError,
  PromptVersionNotFoundError,
  PromptIoError,
  PromptYamlError,
  PromptJsonError,
  PromptValidationError,

  // Memory
  MemoryError,
  MemoryNoEmbedderError,
  MemoryElidError,
  MemoryEmbeddingError,
  MemoryNotFoundError,
  MemorySerializationError,
  MemoryIoError,
  MemoryBackendError,

  // Model cache
  CacheError,
  DownloadError,
  CacheDirError,
  IoError,

  // Helper
  enrichError,
}
