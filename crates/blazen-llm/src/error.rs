//! Unified error types for all Blazen LLM and compute operations.

use std::time::Duration;

/// Maximum bytes retained on `BlazenError::ProviderHttp.raw_body`.
/// Anything larger is truncated; a " ... [truncated N bytes]" marker
/// is appended. Prevents a 20 MB HTML error page from turning
/// `BlazenError` into a memory-bloat vector.
pub const PROVIDER_ERROR_BODY_CAP: usize = 4 * 1024;

/// The unified error type for all Blazen LLM and compute operations.
#[derive(Debug, thiserror::Error)]
pub enum BlazenError {
    // ---- Shared (applies to completions, compute, media, tools) ----
    #[error("authentication failed: {message}")]
    Auth { message: String },

    #[error("rate limited{}", retry_after_ms.map(|ms| format!(": retry after {ms}ms")).unwrap_or_default())]
    RateLimit { retry_after_ms: Option<u64> },

    #[error("timed out after {elapsed_ms}ms")]
    Timeout { elapsed_ms: u64 },

    #[error("{provider} error: {message}")]
    Provider {
        provider: String,
        message: String,
        status_code: Option<u16>,
    },

    /// Upstream HTTP provider returned a non-success response.
    ///
    /// Populated for !2xx responses from real HTTP calls (fal, `OpenRouter`,
    /// `OpenAI`, Anthropic, Gemini, Azure, groq, etc). Do NOT use for:
    /// - Auth failures (use `Auth`)
    /// - Rate-limit where the provider sent a clean `Retry-After` and no body (use `RateLimit`)
    /// - Network/transport failures (keep `Request`)
    /// - Subclass/custom-provider dispatch (keep `Provider`)
    ///
    /// The payload is boxed to keep `BlazenError` under `clippy::result_large_err`'s
    /// 128-byte threshold; access via the `Box<ProviderHttpDetails>` tuple field.
    #[error("{} HTTP {} at {}: {}",
        _0.provider, _0.status, _0.endpoint,
        crate::providers::format_provider_http_tail(
            _0.detail.as_deref(), &_0.raw_body, _0.request_id.as_deref()
        )
    )]
    ProviderHttp(Box<ProviderHttpDetails>),

    #[error("invalid input: {message}")]
    Validation {
        field: Option<String>,
        message: String,
    },

    #[error("content policy violation: {message}")]
    ContentPolicy { message: String },

    #[error("unsupported: {message}")]
    Unsupported { message: String },

    #[error("serialization error: {0}")]
    Serialization(String),

    #[error("request failed: {message}")]
    Request {
        message: String,
        source: Option<Box<dyn std::error::Error + Send + Sync>>,
    },

    // ---- LLM completion-specific ----
    #[error("completion error: {0}")]
    Model(ModelErrorKind),

    // ---- Compute job-specific ----
    #[error("compute error: {0}")]
    Compute(ComputeErrorKind),

    // ---- Media-specific ----
    #[error("media error: {0}")]
    Media(MediaErrorKind),

    // ---- Tool-specific ----
    #[error("tool error: {message}")]
    Tool {
        name: Option<String>,
        message: String,
    },

    /// Caller-side error captured from a binding (Node / Python / WASM /
    /// `UniFFI` / cabi). Carries an opaque type-erased handle to the *original*
    /// caller error value so the binding's error-conversion path can downcast
    /// and re-throw it, preserving class identity and custom properties where
    /// the FFI mechanism allows.
    ///
    /// - Node: `source` holds a `napi::bindgen_prelude::Reference<Unknown<'static>>`.
    /// - Python: `source` holds a `pyo3::Py<pyo3::PyAny>`.
    /// - WASM: `source` holds a `send_wrapper::SendWrapper<wasm_bindgen::JsValue>`.
    /// - `UniFFI` / cabi: `source` holds a binding-specific struct carrying
    ///   `name` / `message` / `properties_json` for structural re-raise.
    ///
    /// `name` is the optional tool / source name. `message` is a fallback
    /// string used by `Display` and by callers that don't downcast the source.
    #[error("caller error{name_in_paren}: {message}", name_in_paren = name.as_deref().map(|n| format!(" in `{n}`")).unwrap_or_default())]
    CallerError {
        name: Option<String>,
        message: String,
        /// Type-erased original caller-side error value. `None` when no
        /// binding-side handle was attached (internal callers). Wrapped in
        /// [`CallerErrorSource`] (a thin `std::error::Error`-implementing
        /// newtype around `Box<dyn Any + Send + Sync>`) so that thiserror's
        /// auto-`#[source]` detection on the literal field name `source`
        /// can find a compatible trait impl. Downcasting goes through the
        /// newtype's `Deref<Target = dyn Any + Send + Sync>` impl, so
        /// `source.as_ref().unwrap().downcast_ref::<T>()` works as-is.
        #[source]
        source: Option<CallerErrorSource>,
    },
}

/// Opaque, type-erased caller-side error payload for
/// [`BlazenError::CallerError`].
///
/// This is a transparent newtype over `Box<dyn Any + Send + Sync>`. It
/// exists for one reason: thiserror auto-treats fields named `source` as
/// `#[source]` and requires the field type to implement `std::error::Error`.
/// `dyn Any` does not, so we wrap it in a type that does.
///
/// `Deref<Target = dyn Any + Send + Sync>` is implemented so callers can
/// downcast the inner value directly:
///
/// ```ignore
/// if let BlazenError::CallerError { source: Some(s), .. } = &err {
///     if let Some(orig) = s.downcast_ref::<MyError>() { /* ... */ }
/// }
/// ```
pub struct CallerErrorSource(pub Box<dyn std::any::Any + Send + Sync>);

impl std::fmt::Debug for CallerErrorSource {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_tuple("CallerErrorSource")
            .field(&"<opaque>")
            .finish()
    }
}

impl std::fmt::Display for CallerErrorSource {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("<opaque caller error>")
    }
}

impl std::error::Error for CallerErrorSource {}

impl std::ops::Deref for CallerErrorSource {
    type Target = dyn std::any::Any + Send + Sync;
    fn deref(&self) -> &Self::Target {
        &*self.0
    }
}

/// Structured payload for [`BlazenError::ProviderHttp`].
///
/// Carries enough HTTP context that a caller can decide retry/fail/report
/// without string-matching the error message. Boxed from the enum variant
/// to keep `BlazenError` under `clippy::result_large_err`'s 128-byte
/// threshold (~138 bytes inline -> 16 bytes boxed).
#[derive(Debug)]
pub struct ProviderHttpDetails {
    /// Provider identifier (e.g. `"fal"`, `"openrouter"`). `Cow` so that
    /// compile-time literals AND runtime `String`s (openai-compat wrappers)
    /// fit without leaking.
    pub provider: std::borrow::Cow<'static, str>,
    /// Full URL of the request that failed. Callers should keep PII out of
    /// URLs — we do not sanitize.
    pub endpoint: String,
    /// HTTP status code.
    pub status: u16,
    /// Provider-supplied request/trace id, looked up as
    /// `x-fal-request-id`, `x-request-id`, `request-id` in that order.
    pub request_id: Option<String>,
    /// Human-readable detail extracted from a JSON error body, if the
    /// body was JSON and matched one of the known shapes.
    pub detail: Option<String>,
    /// Raw response body, capped at `PROVIDER_ERROR_BODY_CAP` bytes.
    /// Longer bodies end with `" ... [truncated N bytes]"`.
    pub raw_body: String,
    /// Parsed from the `Retry-After` response header, when present.
    /// Populated on any status (not just 429) so callers can honor a
    /// `503 + Retry-After`.
    pub retry_after_ms: Option<u64>,
}

/// LLM completion-specific error variants.
#[derive(Debug, thiserror::Error)]
pub enum ModelErrorKind {
    #[error("model returned no content")]
    NoContent,
    #[error("model not found: {0}")]
    ModelNotFound(String),
    #[error("invalid response: {0}")]
    InvalidResponse(String),
    #[error("stream error: {0}")]
    Stream(String),
}

/// Compute job-specific error variants.
#[derive(Debug, thiserror::Error)]
pub enum ComputeErrorKind {
    #[error("job failed: {message}")]
    JobFailed {
        message: String,
        error_type: Option<String>,
        retryable: bool,
    },
    #[error("job cancelled")]
    Cancelled,
    #[error("quota exceeded: {message}")]
    QuotaExceeded { message: String },
}

/// Media-specific error variants.
#[derive(Debug, thiserror::Error)]
pub enum MediaErrorKind {
    #[error("invalid media: {message}")]
    Invalid {
        media_type: Option<String>,
        message: String,
    },
    #[error("media too large: {size_bytes} bytes (max {max_bytes})")]
    TooLarge { size_bytes: u64, max_bytes: u64 },
}

impl BlazenError {
    /// Whether this error is likely transient and the request could be retried.
    #[must_use]
    pub fn is_retryable(&self) -> bool {
        match self {
            Self::RateLimit { .. } | Self::Timeout { .. } | Self::Request { .. } => true,
            Self::Provider { status_code, .. } => status_code.is_none_or(|code| code >= 500),
            Self::ProviderHttp(d) => d.status >= 500 || d.status == 429,
            Self::Compute(ComputeErrorKind::JobFailed { retryable, .. }) => *retryable,
            // CallerError is non-retryable — caller-side errors are user-originated
            // and not the framework's to retry. Folds into the wildcard below.
            _ => false,
        }
    }

    // Convenience constructors

    pub fn auth(message: impl Into<String>) -> Self {
        Self::Auth {
            message: message.into(),
        }
    }

    #[must_use]
    pub fn timeout(elapsed_ms: u64) -> Self {
        Self::Timeout { elapsed_ms }
    }

    #[must_use]
    #[allow(clippy::cast_possible_truncation)]
    pub fn timeout_from_duration(elapsed: Duration) -> Self {
        let ms = elapsed.as_millis();
        Self::Timeout {
            elapsed_ms: if ms > u128::from(u64::MAX) {
                u64::MAX
            } else {
                ms as u64
            },
        }
    }

    pub fn request(message: impl Into<String>) -> Self {
        Self::Request {
            message: message.into(),
            source: None,
        }
    }

    pub fn unsupported(message: impl Into<String>) -> Self {
        Self::Unsupported {
            message: message.into(),
        }
    }

    /// Internal-state error suitable for invariants that should not happen
    /// in correct callers (e.g. mutex poisoning, missing-but-expected
    /// records). Maps to a `Request` variant under the hood.
    pub fn internal(message: impl Into<String>) -> Self {
        Self::Request {
            message: format!("internal: {}", message.into()),
            source: None,
        }
    }

    pub fn provider(provider: impl Into<String>, message: impl Into<String>) -> Self {
        Self::Provider {
            provider: provider.into(),
            message: message.into(),
            status_code: None,
        }
    }

    #[must_use]
    #[allow(clippy::too_many_arguments)]
    pub fn provider_http(
        provider: impl Into<std::borrow::Cow<'static, str>>,
        endpoint: impl Into<String>,
        status: u16,
        request_id: Option<String>,
        detail: Option<String>,
        raw_body: impl Into<String>,
        retry_after_ms: Option<u64>,
    ) -> Self {
        Self::ProviderHttp(Box::new(ProviderHttpDetails {
            provider: provider.into(),
            endpoint: endpoint.into(),
            status,
            request_id,
            detail,
            raw_body: raw_body.into(),
            retry_after_ms,
        }))
    }

    pub fn validation(message: impl Into<String>) -> Self {
        Self::Validation {
            field: None,
            message: message.into(),
        }
    }

    pub fn tool_error(message: impl Into<String>) -> Self {
        Self::Tool {
            name: None,
            message: message.into(),
        }
    }

    /// Build a [`BlazenError::CallerError`] from a caller-side opaque source.
    ///
    /// The `source` is boxed as `dyn Any + Send + Sync` and round-tripped back
    /// to the binding's error-conversion path, which downcasts it to re-throw
    /// the original caller error. See the `CallerError` variant docs.
    pub fn caller_error<E>(message: impl Into<String>, source: E) -> Self
    where
        E: std::any::Any + Send + Sync + 'static,
    {
        Self::CallerError {
            name: None,
            message: message.into(),
            source: Some(CallerErrorSource(Box::new(source))),
        }
    }

    /// Builder: attach a tool/source name to a [`BlazenError::CallerError`].
    /// Panics with `debug_assert!` if called on any other variant — callers
    /// should chain immediately after `caller_error(...)`.
    #[must_use]
    pub fn with_caller_name(mut self, name: impl Into<String>) -> Self {
        if let Self::CallerError { name: slot, .. } = &mut self {
            *slot = Some(name.into());
        } else {
            debug_assert!(false, "with_caller_name called on non-CallerError variant");
        }
        self
    }

    #[must_use]
    pub fn no_content() -> Self {
        Self::Model(ModelErrorKind::NoContent)
    }

    pub fn model_not_found(model: impl Into<String>) -> Self {
        Self::Model(ModelErrorKind::ModelNotFound(model.into()))
    }

    pub fn invalid_response(message: impl Into<String>) -> Self {
        Self::Model(ModelErrorKind::InvalidResponse(message.into()))
    }

    pub fn stream_error(message: impl Into<String>) -> Self {
        Self::Model(ModelErrorKind::Stream(message.into()))
    }

    pub fn job_failed(message: impl Into<String>) -> Self {
        Self::Compute(ComputeErrorKind::JobFailed {
            message: message.into(),
            error_type: None,
            retryable: false,
        })
    }

    #[must_use]
    pub fn cancelled() -> Self {
        Self::Compute(ComputeErrorKind::Cancelled)
    }
}

impl From<serde_json::Error> for BlazenError {
    fn from(e: serde_json::Error) -> Self {
        Self::Serialization(e.to_string())
    }
}

// ---------------------------------------------------------------------------
// blazen-3d capability-trait errors
// ---------------------------------------------------------------------------
//
// The four post-proc capability traits in `blazen-3d` (`Texturizer3dBackend`
// / `Rigger3dBackend` / `Refiner3dBackend` / `Animator3dBackend`) each
// return a parallel error enum with the same variant set
// (`Io` / `EngineNotAvailable` / `InvalidInput` / `Backend` / `Unsupported`).
// The `From` impls below let providers in `crates/blazen-llm/src/providers/`
// delegate to those backends with `?` and surface a `BlazenError` with a
// stage-tagged message that callers can match on.

#[cfg(feature = "threed")]
impl From<blazen_3d::Texturizer3dError> for BlazenError {
    fn from(err: blazen_3d::Texturizer3dError) -> Self {
        use blazen_3d::Texturizer3dError as E;
        match err {
            E::Io(e) => Self::provider("compat-3d", format!("texturize io: {e}")),
            E::EngineNotAvailable(s) => {
                Self::unsupported(format!("texturize engine not available: {s}"))
            }
            E::InvalidInput(s) => Self::validation(format!("invalid texturize input: {s}")),
            E::Backend(s) => Self::provider("compat-3d", format!("texturize backend: {s}")),
            E::Unsupported(s) => {
                Self::unsupported(format!("texturize capability not supported: {s}"))
            }
        }
    }
}

#[cfg(feature = "threed")]
impl From<blazen_3d::Rigger3dError> for BlazenError {
    fn from(err: blazen_3d::Rigger3dError) -> Self {
        use blazen_3d::Rigger3dError as E;
        match err {
            E::Io(e) => Self::provider("compat-3d", format!("rig io: {e}")),
            E::EngineNotAvailable(s) => Self::unsupported(format!("rig engine not available: {s}")),
            E::InvalidInput(s) => Self::validation(format!("invalid rig input: {s}")),
            E::Backend(s) => Self::provider("compat-3d", format!("rig backend: {s}")),
            E::Unsupported(s) => Self::unsupported(format!("rig capability not supported: {s}")),
        }
    }
}

#[cfg(feature = "threed")]
impl From<blazen_3d::Refiner3dError> for BlazenError {
    fn from(err: blazen_3d::Refiner3dError) -> Self {
        use blazen_3d::Refiner3dError as E;
        match err {
            E::Io(e) => Self::provider("compat-3d", format!("refine io: {e}")),
            E::EngineNotAvailable(s) => {
                Self::unsupported(format!("refine engine not available: {s}"))
            }
            E::InvalidInput(s) => Self::validation(format!("invalid refine input: {s}")),
            E::Backend(s) => Self::provider("compat-3d", format!("refine backend: {s}")),
            E::Unsupported(s) => Self::unsupported(format!("refine capability not supported: {s}")),
        }
    }
}

#[cfg(feature = "threed")]
impl From<blazen_3d::Animator3dError> for BlazenError {
    fn from(err: blazen_3d::Animator3dError) -> Self {
        use blazen_3d::Animator3dError as E;
        match err {
            E::Io(e) => Self::provider("compat-3d", format!("animate io: {e}")),
            E::EngineNotAvailable(s) => {
                Self::unsupported(format!("animate engine not available: {s}"))
            }
            E::InvalidInput(s) => Self::validation(format!("invalid animate input: {s}")),
            E::Backend(s) => Self::provider("compat-3d", format!("animate backend: {s}")),
            E::Unsupported(s) => {
                Self::unsupported(format!("animate capability not supported: {s}"))
            }
        }
    }
}

/// Backwards-compatible alias.
#[deprecated(note = "use BlazenError instead")]
pub type LlmError = BlazenError;

/// Backwards-compatible alias.
#[deprecated(note = "use BlazenError instead")]
pub type ComputeError = BlazenError;

/// Result type alias for Blazen operations.
pub type Result<T, E = BlazenError> = std::result::Result<T, E>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn blazen_error_stays_under_size_threshold() {
        // `clippy::result_large_err` defaults to a 128-byte threshold; the
        // existing variants are boxed accordingly (see comment near line 95).
        // Catch regressions if a new variant inflates the enum.
        assert!(
            std::mem::size_of::<BlazenError>() <= 128,
            "BlazenError grew to {} bytes; box the offending variant",
            std::mem::size_of::<BlazenError>()
        );
    }

    #[test]
    fn caller_error_constructor_stores_source() {
        struct Marker(u32);
        let err = BlazenError::caller_error("boom", Marker(42));
        match err {
            BlazenError::CallerError {
                ref source,
                ref message,
                ..
            } => {
                assert_eq!(message, "boom");
                let downcast = source.as_ref().unwrap().downcast_ref::<Marker>().unwrap();
                assert_eq!(downcast.0, 42);
            }
            _ => panic!("expected CallerError variant"),
        }
    }

    #[test]
    fn caller_error_is_not_retryable() {
        let err = BlazenError::caller_error("x", 0u8);
        assert!(!err.is_retryable());
    }
}
