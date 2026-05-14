//! Provider-defaults records for the UniFFI bindings.
//!
//! Mirrors the typed defaults hierarchy in [`blazen_llm::providers::defaults`]:
//! a universal [`BaseProviderDefaults`] plus 11 role-specific defaults that
//! compose `BaseProviderDefaults` as a `base` field.
//!
//! ## V1 scope: data fields only — no hooks
//!
//! The upstream defaults carry async hook closures (`before_request` /
//! `before_completion` / role-specific `before` hooks). Those hooks are
//! `Arc<dyn Fn(...) -> BoxFuture<...>>` trait objects, which UniFFI Records
//! cannot represent. They are deferred to Phase C, which will introduce
//! foreign-implementable hook callback traits via
//! `#[uniffi::export(with_foreign)]` (see [`crate::streaming`] for the
//! pattern).
//!
//! For Phase A, the defaults Records carry the **data** fields only:
//!
//! - [`CompletionProviderDefaults`]: `system_prompt`, `tools_json`,
//!   `response_format_json`.
//! - [`EmbeddingProviderDefaults`]: composes `base` only.
//! - 9 role-specific defaults: compose `base` only (their typed hook is
//!   deferred to Phase C).
//!
//! ## Wire-format shape
//!
//! UniFFI doesn't expose `serde_json::Value` or `Vec<ToolDefinition>`
//! directly, so:
//!
//! - `response_format: Option<serde_json::Value>` crosses as
//!   `response_format_json: Option<String>` (JSON-encoded). `None` and
//!   `Some(String::new())` both round-trip to upstream `None`.
//! - `tools: Vec<ToolDefinition>` crosses as `tools_json: Option<String>`
//!   (JSON-encoded array). `None` and `Some(String::new())` both round-trip
//!   to upstream `Vec::new()`.
//!
//! These records are consumed by Phase B's `CustomProvider` factories and
//! by Phase A's [`crate::provider_base::BaseProvider`].

use blazen_llm::providers::defaults::{
    AudioMusicProviderDefaults as CoreAudioMusicProviderDefaults,
    AudioSpeechProviderDefaults as CoreAudioSpeechProviderDefaults,
    BackgroundRemovalProviderDefaults as CoreBackgroundRemovalProviderDefaults,
    BaseProviderDefaults as CoreBaseProviderDefaults,
    CompletionProviderDefaults as CoreCompletionProviderDefaults,
    EmbeddingProviderDefaults as CoreEmbeddingProviderDefaults,
    ImageGenerationProviderDefaults as CoreImageGenerationProviderDefaults,
    ImageUpscaleProviderDefaults as CoreImageUpscaleProviderDefaults,
    ThreeDProviderDefaults as CoreThreeDProviderDefaults,
    TranscriptionProviderDefaults as CoreTranscriptionProviderDefaults,
    VideoProviderDefaults as CoreVideoProviderDefaults,
    VoiceCloningProviderDefaults as CoreVoiceCloningProviderDefaults,
};
use blazen_llm::types::ToolDefinition;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Encode an `Option<serde_json::Value>` as `Option<String>`.
///
/// `None` upstream maps to `None` here. `Some(Value::Null)` maps to
/// `Some("null")` — the explicit JSON null token, distinguishable from
/// `None` on the foreign side.
fn json_value_to_opt_string(v: Option<&serde_json::Value>) -> Option<String> {
    v.and_then(|val| serde_json::to_string(val).ok())
}

/// Decode `Option<String>` back into `Option<serde_json::Value>`.
///
/// `None` and `Some("")` both decode to `None`. Malformed JSON also decodes
/// to `None` (defensive: foreign callers can't always validate the JSON
/// before sending it across the FFI).
fn opt_string_to_json_value(s: Option<&str>) -> Option<serde_json::Value> {
    let s = s?.trim();
    if s.is_empty() {
        return None;
    }
    serde_json::from_str(s).ok()
}

/// Encode `Vec<ToolDefinition>` as `Option<String>` (JSON-encoded array).
///
/// An empty `Vec` maps to `None` (cleaner foreign-side ergonomics than
/// `Some("[]")`).
fn tools_to_opt_string(tools: &[ToolDefinition]) -> Option<String> {
    if tools.is_empty() {
        return None;
    }
    serde_json::to_string(tools).ok()
}

/// Decode `Option<String>` back into `Vec<ToolDefinition>`.
///
/// `None`, `Some("")`, and malformed input all decode to `Vec::new()` —
/// matching the upstream `#[derive(Default)]` behavior.
fn opt_string_to_tools(s: Option<&str>) -> Vec<ToolDefinition> {
    let Some(s) = s else {
        return Vec::new();
    };
    let s = s.trim();
    if s.is_empty() {
        return Vec::new();
    }
    serde_json::from_str(s).unwrap_or_default()
}

// ---------------------------------------------------------------------------
// BaseProviderDefaults
// ---------------------------------------------------------------------------

/// Universal provider defaults applicable across every provider role.
///
/// V1 carries no data fields — the upstream `before_request` hook is
/// deferred to Phase C. A placeholder boolean field is included so the
/// generated foreign-language struct is non-empty (UniFFI Records with
/// zero fields generate slightly awkward foreign-side code).
#[derive(Debug, Clone, Default, uniffi::Record)]
pub struct BaseProviderDefaults {
    /// Reserved for future use. Currently ignored on both sides of the FFI.
    /// V1 carries no universal defaults data — the upstream `before_request`
    /// hook is exposed via Phase C's foreign-implementable callback trait.
    #[uniffi(default = false)]
    pub reserved: bool,
}

impl From<&CoreBaseProviderDefaults> for BaseProviderDefaults {
    fn from(_core: &CoreBaseProviderDefaults) -> Self {
        Self::default()
    }
}

impl From<BaseProviderDefaults> for CoreBaseProviderDefaults {
    fn from(_ffi: BaseProviderDefaults) -> Self {
        Self::default()
    }
}

// ---------------------------------------------------------------------------
// CompletionProviderDefaults
// ---------------------------------------------------------------------------

/// Completion-role defaults: system prompt, default tools, default
/// `response_format`. Hooks (`before_completion`) deferred to Phase C.
#[derive(Debug, Clone, Default, uniffi::Record)]
pub struct CompletionProviderDefaults {
    #[uniffi(default = None)]
    pub base: Option<BaseProviderDefaults>,
    /// Prepended as a system message if the request lacks one.
    #[uniffi(default = None)]
    pub system_prompt: Option<String>,
    /// JSON-encoded `Vec<ToolDefinition>`. Merged into the request's tool
    /// list — request-supplied tools win on name collision.
    #[uniffi(default = None)]
    pub tools_json: Option<String>,
    /// JSON-encoded `serde_json::Value` for the OpenAI-style
    /// `response_format` field. Set only if the request lacks one.
    #[uniffi(default = None)]
    pub response_format_json: Option<String>,
}

impl From<&CoreCompletionProviderDefaults> for CompletionProviderDefaults {
    fn from(core: &CoreCompletionProviderDefaults) -> Self {
        Self {
            base: Some(BaseProviderDefaults::from(&core.base)),
            system_prompt: core.system_prompt.clone(),
            tools_json: tools_to_opt_string(&core.tools),
            response_format_json: json_value_to_opt_string(core.response_format.as_ref()),
        }
    }
}

impl From<CompletionProviderDefaults> for CoreCompletionProviderDefaults {
    fn from(ffi: CompletionProviderDefaults) -> Self {
        let base = ffi
            .base
            .map(CoreBaseProviderDefaults::from)
            .unwrap_or_default();
        Self {
            base,
            system_prompt: ffi.system_prompt,
            tools: opt_string_to_tools(ffi.tools_json.as_deref()),
            response_format: opt_string_to_json_value(ffi.response_format_json.as_deref()),
            before_completion: None,
        }
    }
}

// ---------------------------------------------------------------------------
// EmbeddingProviderDefaults
// ---------------------------------------------------------------------------

/// Embedding-role defaults. V1 composes only `base`.
#[derive(Debug, Clone, Default, uniffi::Record)]
pub struct EmbeddingProviderDefaults {
    #[uniffi(default = None)]
    pub base: Option<BaseProviderDefaults>,
}

impl From<&CoreEmbeddingProviderDefaults> for EmbeddingProviderDefaults {
    fn from(core: &CoreEmbeddingProviderDefaults) -> Self {
        Self {
            base: Some(BaseProviderDefaults::from(&core.base)),
        }
    }
}

impl From<EmbeddingProviderDefaults> for CoreEmbeddingProviderDefaults {
    fn from(ffi: EmbeddingProviderDefaults) -> Self {
        let base = ffi
            .base
            .map(CoreBaseProviderDefaults::from)
            .unwrap_or_default();
        Self { base }
    }
}

// ---------------------------------------------------------------------------
// Role-specific defaults (9 roles)
// ---------------------------------------------------------------------------
//
// Each role's defaults Record composes `base` only — the typed `before`
// hook is deferred to Phase C. The `From` impls drop the upstream hook
// (data-only round-trip) and reconstruct the upstream struct with
// `before = None`.

/// Generate a UniFFI Record mirroring a role-specific defaults type.
macro_rules! ffi_role_defaults {
    ($name:ident, $core:ty) => {
        #[derive(Debug, Clone, Default, uniffi::Record)]
        pub struct $name {
            #[uniffi(default = None)]
            pub base: Option<BaseProviderDefaults>,
        }

        impl From<&$core> for $name {
            fn from(core: &$core) -> Self {
                Self {
                    base: Some(BaseProviderDefaults::from(&core.base)),
                }
            }
        }

        impl From<$name> for $core {
            fn from(ffi: $name) -> Self {
                let base = ffi
                    .base
                    .map(CoreBaseProviderDefaults::from)
                    .unwrap_or_default();
                Self { base, before: None }
            }
        }
    };
}

ffi_role_defaults!(AudioSpeechProviderDefaults, CoreAudioSpeechProviderDefaults);
ffi_role_defaults!(AudioMusicProviderDefaults, CoreAudioMusicProviderDefaults);
ffi_role_defaults!(
    VoiceCloningProviderDefaults,
    CoreVoiceCloningProviderDefaults
);
ffi_role_defaults!(
    ImageGenerationProviderDefaults,
    CoreImageGenerationProviderDefaults
);
ffi_role_defaults!(
    ImageUpscaleProviderDefaults,
    CoreImageUpscaleProviderDefaults
);
ffi_role_defaults!(VideoProviderDefaults, CoreVideoProviderDefaults);
ffi_role_defaults!(
    TranscriptionProviderDefaults,
    CoreTranscriptionProviderDefaults
);
ffi_role_defaults!(ThreeDProviderDefaults, CoreThreeDProviderDefaults);
ffi_role_defaults!(
    BackgroundRemovalProviderDefaults,
    CoreBackgroundRemovalProviderDefaults
);
