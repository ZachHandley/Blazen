//! Native Error-class registration + caller-error UUID stash for the Node
//! binding. Replaces the JS post-build shim that previously did this work
//! via `[Tag]` / `__BLAZEN_PROVIDER_ERROR__` / `__BLAZEN_CALLER_ERROR__`
//! sentinels.
//!
//! ## How it works
//!
//! 1. At module load (`module_exports` in `lib.rs`), [`register_all_classes`]
//!    walks the full error hierarchy and calls
//!    [`napi::register_error_class`] for each one. Children are always
//!    registered after their parent, so JS sees a proper inheritance chain:
//!
//!    ```text
//!    BlazenError extends Error
//!      ├── AuthError, RateLimitError, TimeoutError, ValidationError,
//!      │   ContentPolicyError, UnsupportedError, ComputeError, MediaError
//!      ├── ProviderError
//!      │     ├── LlamaCppError, LlamaCppInvalidOptionsError, ...
//!      │     ├── CandleLlmError, CandleEmbedError, MistralRsError,
//!      │     │   WhisperError, PiperError, DiffusionError, FastEmbedError,
//!      │     │   TractError (and their per-variant subclasses)
//!      ├── PeerEncodeError, PeerTransportError, PeerEnvelopeVersionError,
//!      │   PeerWorkflowError, PeerTlsError, PeerUnknownStepError
//!      ├── PersistError, CacheError (DownloadError, CacheDirError, IoError)
//!      ├── PromptError (and 7 variant subclasses)
//!      └── MemoryError (and 7 variant subclasses)
//!    ```
//!
//! 2. After registration, Rust code in `src/error.rs` constructs errors via
//!    `napi::Error::with_class("ProviderError", reason).with_field(...)`.
//!    napi-patched's `ToNapiValue for Error` looks up the class name in
//!    the registry and constructs `new ClassName(reason, props)` as the
//!    throw value — so JS callers see the right instance with structured
//!    fields as own properties.
//!
//! 3. For errors that originate inside user-supplied JS callbacks
//!    (tool handlers, custom-provider methods, ...), napi-patched's
//!    `call_async_catch` (in napi 3.9.0) preserves the original JS error
//!    object via `napi::Error::has_js_value()`. Because Blazen's agent loop
//!    sits between the napi boundary and the loop body, we stash the
//!    captured `napi::Error` in [`CALLER_ERROR_STASH`] keyed by a UUID and
//!    carry the UUID through `BlazenError::CallerError` until we can throw
//!    it back through napi. See [`stash_caller_error`] / [`take_caller_error`].
//!
//! 4. After all this, the post-build JS shim has nothing left to do and is
//!    deleted in the next commit.
//!
//! ## Why a Rust-side stash instead of letting the napi Error flow through?
//!
//! The `Tool` trait in `blazen-llm` requires returning
//! `Result<ToolOutput, BlazenError>`. `BlazenError` is in `blazen-llm` which
//! cannot depend on `napi`. So we cannot put a `napi::Error` directly inside
//! a `BlazenError` variant. The next-best thing is a UUID handle that the
//! Node binding owns on both ends of the agent loop.

use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};

use napi::bindgen_prelude::Object;
use napi::{Env, Result, export_class_to, register_error_class};

// ----------------------------------------------------------------------------
// Class registration
// ----------------------------------------------------------------------------

/// The full hierarchy: `(class_name, parent_or_None)`, ordered so every
/// parent appears before any of its children. [`register_all_classes`]
/// walks this in order, registering each class via
/// [`napi::register_error_class`] and binding it onto the module's
/// `exports` object so JS callers can do `require('blazen').ProviderError`
/// and `err instanceof ProviderError`.
const ERROR_CLASS_HIERARCHY: &[(&str, Option<&str>)] = &[
    // Root
    ("BlazenError", None),
    // Direct subclasses of BlazenError
    ("AuthError", Some("BlazenError")),
    ("RateLimitError", Some("BlazenError")),
    ("TimeoutError", Some("BlazenError")),
    ("ValidationError", Some("BlazenError")),
    ("ContentPolicyError", Some("BlazenError")),
    ("UnsupportedError", Some("BlazenError")),
    ("ComputeError", Some("BlazenError")),
    ("MediaError", Some("BlazenError")),
    // ProviderError + nine backend-family trees
    ("ProviderError", Some("BlazenError")),
    ("LlamaCppError", Some("ProviderError")),
    ("LlamaCppInvalidOptionsError", Some("LlamaCppError")),
    ("LlamaCppModelLoadError", Some("LlamaCppError")),
    ("LlamaCppInferenceError", Some("LlamaCppError")),
    ("LlamaCppEngineNotAvailableError", Some("LlamaCppError")),
    ("LlamaCppAdapterFailedError", Some("LlamaCppError")),
    ("CandleLlmError", Some("ProviderError")),
    ("CandleLlmInvalidOptionsError", Some("CandleLlmError")),
    ("CandleLlmModelLoadError", Some("CandleLlmError")),
    ("CandleLlmInferenceError", Some("CandleLlmError")),
    ("CandleLlmEngineNotAvailableError", Some("CandleLlmError")),
    ("CandleLlmUnsupportedError", Some("CandleLlmError")),
    ("CandleEmbedError", Some("ProviderError")),
    ("CandleEmbedInvalidOptionsError", Some("CandleEmbedError")),
    ("CandleEmbedModelLoadError", Some("CandleEmbedError")),
    ("CandleEmbedEmbeddingError", Some("CandleEmbedError")),
    (
        "CandleEmbedEngineNotAvailableError",
        Some("CandleEmbedError"),
    ),
    ("CandleEmbedTaskPanickedError", Some("CandleEmbedError")),
    ("MistralRsError", Some("ProviderError")),
    ("MistralRsInvalidOptionsError", Some("MistralRsError")),
    ("MistralRsInitError", Some("MistralRsError")),
    ("MistralRsInferenceError", Some("MistralRsError")),
    ("MistralRsEngineNotAvailableError", Some("MistralRsError")),
    ("MistralRsAdapterFailedError", Some("MistralRsError")),
    ("WhisperError", Some("ProviderError")),
    ("WhisperInvalidOptionsError", Some("WhisperError")),
    ("WhisperModelLoadError", Some("WhisperError")),
    ("WhisperTranscriptionError", Some("WhisperError")),
    ("WhisperEngineNotAvailableError", Some("WhisperError")),
    ("WhisperIoError", Some("WhisperError")),
    ("PiperError", Some("ProviderError")),
    ("PiperInvalidOptionsError", Some("PiperError")),
    ("PiperModelLoadError", Some("PiperError")),
    ("PiperSynthesisError", Some("PiperError")),
    ("PiperEngineNotAvailableError", Some("PiperError")),
    ("DiffusionError", Some("ProviderError")),
    ("DiffusionInvalidOptionsError", Some("DiffusionError")),
    ("DiffusionModelLoadError", Some("DiffusionError")),
    ("DiffusionGenerationError", Some("DiffusionError")),
    ("DiffusionEngineNotAvailableError", Some("DiffusionError")),
    ("FastEmbedError", Some("ProviderError")),
    ("EmbedUnknownModelError", Some("FastEmbedError")),
    ("EmbedInitError", Some("FastEmbedError")),
    ("EmbedEmbedError", Some("FastEmbedError")),
    ("EmbedMutexPoisonedError", Some("FastEmbedError")),
    ("EmbedTaskPanickedError", Some("FastEmbedError")),
    ("TractError", Some("ProviderError")),
    // Peer transport
    ("PeerEncodeError", Some("BlazenError")),
    ("PeerTransportError", Some("BlazenError")),
    ("PeerEnvelopeVersionError", Some("BlazenError")),
    ("PeerWorkflowError", Some("BlazenError")),
    ("PeerTlsError", Some("BlazenError")),
    ("PeerUnknownStepError", Some("BlazenError")),
    // Persist
    ("PersistError", Some("BlazenError")),
    // Cache
    ("CacheError", Some("BlazenError")),
    ("DownloadError", Some("CacheError")),
    ("CacheDirError", Some("CacheError")),
    ("IoError", Some("CacheError")),
    // Prompts
    ("PromptError", Some("BlazenError")),
    ("PromptMissingVariableError", Some("PromptError")),
    ("PromptNotFoundError", Some("PromptError")),
    ("PromptVersionNotFoundError", Some("PromptError")),
    ("PromptIoError", Some("PromptError")),
    ("PromptYamlError", Some("PromptError")),
    ("PromptJsonError", Some("PromptError")),
    ("PromptValidationError", Some("PromptError")),
    // Memory
    ("MemoryError", Some("BlazenError")),
    ("MemoryNoEmbedderError", Some("MemoryError")),
    ("MemoryElidError", Some("MemoryError")),
    ("MemoryEmbeddingError", Some("MemoryError")),
    ("MemoryNotFoundError", Some("MemoryError")),
    ("MemorySerializationError", Some("MemoryError")),
    ("MemoryIoError", Some("MemoryError")),
    ("MemoryBackendError", Some("MemoryError")),
];

/// Register every JS error class Blazen exposes and bind each onto the
/// module's `exports` object. Called once from `module_exports` in `lib.rs`.
///
/// All classes are registered unconditionally — even feature-gated backend
/// subclasses (`LlamaCpp*`, `Mistral*`, etc.) get a class declaration so
/// that catch-all JS like `catch (e) { if (e instanceof LlamaCppError) ... }`
/// type-checks regardless of which Rust features the build was compiled
/// with. The Rust mappers that *emit* those classes are still feature-gated.
///
/// # Errors
/// Propagates any napi error from the underlying class-factory script
/// evaluation, reference creation, or export binding. In practice this
/// fails only if the JS context is in an unrecoverable state.
pub fn register_all_classes(env: &Env, exports: &Object<'_>) -> Result<()> {
    for (name, parent) in ERROR_CLASS_HIERARCHY {
        register_error_class(env, name, *parent)?;
        export_class_to(env, exports, name)?;
    }
    Ok(())
}

// ----------------------------------------------------------------------------
// Caller-error UUID stash
// ----------------------------------------------------------------------------

/// Process-global stash for `napi::Error` values that originated inside
/// user JS callbacks (most commonly tool handlers). The Blazen agent loop
/// in `blazen-llm` doesn't know about napi and so can't carry these errors
/// directly — it carries a `BlazenError::CallerError { source: Some(uuid) }`
/// instead, and the Node binding looks the original error back up here
/// when emitting the napi error back to JS.
///
/// Stashed entries are removed by [`take_caller_error`] when reissued, so
/// the map size is bounded by the number of in-flight rejected tool calls.
static CALLER_ERROR_STASH: OnceLock<Mutex<HashMap<String, napi::Error>>> = OnceLock::new();

fn stash() -> &'static Mutex<HashMap<String, napi::Error>> {
    CALLER_ERROR_STASH.get_or_init(|| Mutex::new(HashMap::new()))
}

/// Stash a `napi::Error` and return a UUID handle. The error stays in the
/// stash until [`take_caller_error`] is called with the same UUID, or the
/// process exits.
#[must_use]
pub fn stash_caller_error(err: napi::Error) -> String {
    let uuid = uuid::Uuid::new_v4().to_string();
    stash()
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .insert(uuid.clone(), err);
    uuid
}

/// Look up and remove a previously-stashed `napi::Error` by UUID. Returns
/// `None` if no entry exists for that UUID (already taken, or the UUID was
/// fabricated).
#[must_use]
pub fn take_caller_error(uuid: &str) -> Option<napi::Error> {
    stash()
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .remove(uuid)
}
