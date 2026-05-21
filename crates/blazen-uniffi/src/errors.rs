//! Error types exposed across the UniFFI boundary.
//!
//! ## Design
//!
//! Internally, Blazen has a sprawling error hierarchy: `blazen_llm::BlazenError`
//! with ~20 top-level variants, `blazen_core::WorkflowError`, `blazen_peer::PeerError`,
//! `blazen_pipeline::PipelineError`, plus per-backend sub-errors (LlamaCpp, Candle,
//! MistralRs, Whisper, Piper, Diffusion, FastEmbed, Tract). The Node binding flattens
//! all of these into ~80 classes; the Python binding does the same via PyO3 exceptions.
//!
//! UniFFI's idiomatic error model is a flat enum with structured payloads. Trying
//! to encode 80 nested classes in UDL would fight UniFFI's grain and produce
//! brittle bindings. Instead, this module exposes a **single `BlazenError` enum**
//! with the principal error categories as variants, and a `kind` discriminator
//! string on `Provider` / `Peer` / `Prompt` / `Memory` / `Cache` for sub-classification.
//!
//! Consumers get full typed errors at the top level (`if let BlazenError::Provider {
//! kind, status, ... }`), and structured detail via the payload — functionally
//! equivalent to the Node/Python class hierarchies, idiomatic for Go switch-typing,
//! Swift `case`-matching, Kotlin sealed-class branches, and Ruby `case`/`in`.

use thiserror::Error;

/// Structured payload stored in `blazen_llm::BlazenError::CallerError.source`
/// when a UniFFI tool handler raises a typed caller error.
///
/// Foreign-language handlers can't carry their exception class identity
/// across the UniFFI IDL boundary (FFI mechanism is genuinely missing),
/// so we preserve `(name, message, properties_json)` instead. The
/// foreign-side `name` is the exception class name (e.g. `"SubmitSignal"`);
/// `properties_json` is a JSON-encoded blob of any custom attributes the
/// foreign side wants to round-trip.
///
/// `crates/blazen-uniffi/src/agent.rs` constructs this payload when a
/// UniFFI `ToolHandler` returns `Err(BlazenError::CallerError { ... })`;
/// the conversion back to UniFFI [`BlazenError::CallerError`] in this
/// file downcasts the opaque `Box<dyn Any>` to recover the fields.
#[derive(Debug, Clone)]
pub struct UniffiCallerErrorPayload {
    pub name: Option<String>,
    pub message: String,
    pub properties_json: String,
}

/// Canonical error type for all Blazen UniFFI bindings.
///
/// Each variant carries a `message` field with a human-readable description
/// (matching the corresponding Node/Python error's `.message`). Variants with
/// sub-types carry a `kind` string discriminator (e.g. `Provider.kind = "LlamaCppModelLoad"`).
#[derive(Debug, Error, uniffi::Error)]
pub enum BlazenError {
    /// Authentication / credentials failure (missing API key, invalid token, etc.).
    #[error("auth: {message}")]
    Auth { message: String },

    /// Rate limit exceeded. `retry_after_ms` is set when the provider returned a
    /// Retry-After hint.
    #[error("rate limit: {message}")]
    RateLimit {
        message: String,
        retry_after_ms: Option<u64>,
    },

    /// Operation timed out before the provider responded.
    #[error("timeout: {message} (elapsed {elapsed_ms} ms)")]
    Timeout { message: String, elapsed_ms: u64 },

    /// Input validation failed (bad schema, missing required field, etc.).
    #[error("validation: {message}")]
    Validation { message: String },

    /// Content policy violation (provider refused due to safety filters).
    #[error("content policy: {message}")]
    ContentPolicy { message: String },

    /// Operation unsupported on this platform / build / provider.
    #[error("unsupported: {message}")]
    Unsupported { message: String },

    /// Compute error (CPU/GPU/accelerator failure, OOM, etc.).
    #[error("compute: {message}")]
    Compute { message: String },

    /// Media-handling error (decode, encode, transcoding).
    #[error("media: {message}")]
    Media { message: String },

    /// Provider / backend error. `kind` identifies the specific backend and failure
    /// mode, mirroring the Node binding's `[ProviderError]` sentinel JSON shape.
    /// Examples of `kind`: `"LlamaCppModelLoad"`, `"DiffusionGeneration"`,
    /// `"CandleEmbedInference"`, `"OpenAIHttp"`, `"AnthropicHttp"`.
    #[error("provider {kind}: {message}")]
    Provider {
        kind: String,
        message: String,
        provider: Option<String>,
        status: Option<u32>,
        endpoint: Option<String>,
        request_id: Option<String>,
        detail: Option<String>,
        retry_after_ms: Option<u64>,
    },

    /// Workflow execution error (step panic, deadlock, missing context, etc.).
    #[error("workflow: {message}")]
    Workflow { message: String },

    /// Tool / function-call error during LLM agent execution.
    #[error("tool: {message}")]
    Tool { message: String },

    /// Caller-side error raised by a foreign-language tool handler.
    ///
    /// Carries structural error data — `name` (foreign-language exception
    /// class name, e.g. `"SubmitSignal"`), `message`, and `properties_json`
    /// (JSON-encoded custom attributes). Foreign consumers pattern-match on
    /// `name` and decode `properties_json` to recover custom payload data.
    ///
    /// Full exception class identity is not preserved across the UniFFI
    /// boundary (the Node/Python/WASM bindings get full `instanceof`
    /// preservation because they have native object references; UniFFI does
    /// not). This variant is the structural equivalent.
    #[error("caller{name_in_paren}: {message}", name_in_paren = name.as_deref().map(|n| format!(" `{n}`")).unwrap_or_default())]
    CallerError {
        name: Option<String>,
        message: String,
        properties_json: String,
    },

    /// Distributed peer-to-peer error and (folded in) distributed
    /// control-plane error. For peer-mesh failures `kind` is one of
    /// `"Encode"`, `"Transport"`, `"EnvelopeVersion"`, `"Workflow"`,
    /// `"Tls"`, `"UnknownStep"`. For control-plane failures `kind` is
    /// prefixed `"ControlPlane"` (e.g. `"ControlPlaneTransport"`,
    /// `"ControlPlaneEncode"`, `"ControlPlaneTls"`,
    /// `"ControlPlaneEnvelopeVersion"`, `"ControlPlaneNoMatchingWorker"`,
    /// `"ControlPlaneMissingVramHint"`, `"ControlPlaneUnknownRun"`,
    /// `"ControlPlaneUnknownWorker"`) so foreign consumers can discriminate
    /// without juggling a second top-level variant.
    #[error("peer {kind}: {message}")]
    Peer { kind: String, message: String },

    /// Persistence layer error (redb / valkey checkpoint store).
    #[error("persist: {message}")]
    Persist { message: String },

    /// Prompt template error. `kind`: `"MissingVariable"`, `"NotFound"`, `"VersionNotFound"`,
    /// `"Io"`, `"Yaml"`, `"Json"`, `"Validation"`.
    #[error("prompt {kind}: {message}")]
    Prompt { kind: String, message: String },

    /// Memory subsystem error. `kind`: `"NoEmbedder"`, `"Elid"`, `"Embedding"`,
    /// `"NotFound"`, `"Serialization"`, `"Io"`, `"Backend"`.
    #[error("memory {kind}: {message}")]
    Memory { kind: String, message: String },

    /// Model-cache / download error. `kind`: `"Download"`, `"CacheDir"`, `"Io"`.
    #[error("cache {kind}: {message}")]
    Cache { kind: String, message: String },

    /// Operation was cancelled (e.g. via a foreign-language `context.Context`
    /// or `Task.cancel()` request). Mapped to `context.Canceled` /
    /// `Task.CancellationError` / `Kotlin CancellationException` on the foreign side.
    #[error("cancelled")]
    Cancelled,

    /// Fallback for errors that don't fit any other variant — should be rare;
    /// new errors should usually get their own variant or a `kind` extension.
    #[error("internal: {message}")]
    Internal { message: String },
}

/// Result alias used throughout the UniFFI layer.
pub type BlazenResult<T> = Result<T, BlazenError>;

// ---------------------------------------------------------------------------
// Conversions from internal Blazen error types.
//
// These are the bridge between Blazen's internal error enums and the UniFFI
// surface. Adding a new internal variant means adding a match arm here.
// ---------------------------------------------------------------------------

impl From<blazen_llm::BlazenError> for BlazenError {
    fn from(err: blazen_llm::BlazenError) -> Self {
        use blazen_llm::BlazenError as L;
        // Pre-format Display BEFORE moving fields out, so we can reference
        // it inside variants that need a textual fallback (Display impl on
        // BlazenError covers all variants via its #[error(...)] attrs).
        let display = err.to_string();
        match err {
            L::Auth { message } => Self::Auth { message },
            L::RateLimit { retry_after_ms } => Self::RateLimit {
                message: display,
                retry_after_ms,
            },
            L::Timeout { elapsed_ms } => Self::Timeout {
                message: display,
                elapsed_ms,
            },
            L::Validation { message, .. } => Self::Validation { message },
            L::ContentPolicy { message } => Self::ContentPolicy { message },
            L::Unsupported { message } => Self::Unsupported { message },
            L::Tool { message, .. } => Self::Tool { message },
            L::Provider {
                provider,
                message,
                status_code,
            } => Self::Provider {
                kind: "Generic".into(),
                message,
                provider: Some(provider),
                status: status_code.map(u32::from),
                endpoint: None,
                request_id: None,
                detail: None,
                retry_after_ms: None,
            },
            L::ProviderHttp(details) => Self::Provider {
                kind: "Http".into(),
                message: display,
                provider: Some(details.provider.to_string()),
                status: Some(u32::from(details.status)),
                endpoint: Some(details.endpoint.clone()),
                request_id: details.request_id.clone(),
                detail: details.detail.clone(),
                retry_after_ms: details.retry_after_ms,
            },
            L::Compute(_) => Self::Compute { message: display },
            L::Media(_) => Self::Media { message: display },
            L::CallerError {
                source,
                name,
                message,
            } => {
                // Downcast the opaque source to recover UniFFI's structured payload.
                // If the source came from a different binding (Node/Python/WASM
                // typed handles, or no source at all), fall back to `name`/`message`
                // and an empty properties_json.
                let (final_name, properties_json) = source
                    .as_ref()
                    .and_then(|s| s.downcast_ref::<UniffiCallerErrorPayload>())
                    .map(|p| {
                        (
                            p.name.clone().or_else(|| name.clone()),
                            p.properties_json.clone(),
                        )
                    })
                    .unwrap_or_else(|| (name, "{}".to_string()));
                Self::CallerError {
                    name: final_name,
                    message,
                    properties_json,
                }
            }
            // Serialization / Request / Completion all fold into Internal; if a
            // caller needs to discriminate these we can promote them later.
            _ => Self::Internal { message: display },
        }
    }
}

impl From<blazen_core::WorkflowError> for BlazenError {
    fn from(err: blazen_core::WorkflowError) -> Self {
        Self::Workflow {
            message: err.to_string(),
        }
    }
}

impl From<blazen_pipeline::PipelineError> for BlazenError {
    fn from(err: blazen_pipeline::PipelineError) -> Self {
        Self::Workflow {
            message: err.to_string(),
        }
    }
}

impl From<blazen_peer::PeerError> for BlazenError {
    fn from(err: blazen_peer::PeerError) -> Self {
        use blazen_peer::PeerError as P;
        let (kind, message) = match &err {
            P::Encode(_) => ("Encode", err.to_string()),
            P::Transport(_) => ("Transport", err.to_string()),
            P::EnvelopeVersion { .. } => ("EnvelopeVersion", err.to_string()),
            P::Workflow(_) => ("Workflow", err.to_string()),
            P::Tls(_) => ("Tls", err.to_string()),
            P::UnknownStep { .. } => ("UnknownStep", err.to_string()),
        };
        Self::Peer {
            kind: kind.into(),
            message,
        }
    }
}

#[cfg(feature = "distributed")]
impl From<blazen_controlplane::ControlPlaneError> for BlazenError {
    fn from(err: blazen_controlplane::ControlPlaneError) -> Self {
        use blazen_controlplane::ControlPlaneError as C;
        // Why fold into `Peer`: keeping a single transport-error variant
        // across peer-mesh and control-plane callers means foreign
        // bindings get a stable error surface to switch on, and the
        // workspace's other consumers of `BlazenError` (blazen-cabi)
        // don't have to grow a parallel variant for every new
        // distributed-transport error type. The `kind` discriminator
        // namespaces control-plane errors with the `ControlPlane` prefix.
        let kind = match &err {
            C::Encode(_) => "ControlPlaneEncode",
            C::Json(_) => "ControlPlaneJson",
            C::Transport(_) => "ControlPlaneTransport",
            C::EnvelopeVersion { .. } => "ControlPlaneEnvelopeVersion",
            C::Tls(_) => "ControlPlaneTls",
            C::Unauthenticated(_) => "ControlPlaneUnauthenticated",
            C::NoMatchingWorker { .. } => "ControlPlaneNoMatchingWorker",
            C::MissingVramHint => "ControlPlaneMissingVramHint",
            C::UnknownRun(_) => "ControlPlaneUnknownRun",
            C::UnknownWorker(_) => "ControlPlaneUnknownWorker",
            C::Workflow(_) => "ControlPlaneWorkflow",
            C::Rpc(_) => "ControlPlaneRpc",
        };
        Self::Peer {
            kind: kind.into(),
            message: err.to_string(),
        }
    }
}

impl From<serde_json::Error> for BlazenError {
    fn from(err: serde_json::Error) -> Self {
        Self::Validation {
            message: format!("json: {err}"),
        }
    }
}

impl From<std::io::Error> for BlazenError {
    fn from(err: std::io::Error) -> Self {
        Self::Internal {
            message: format!("io: {err}"),
        }
    }
}
