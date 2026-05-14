//! JavaScript bindings for the provider-defaults hierarchy.
//!
//! Exposes one class per Rust defaults struct from
//! [`blazen_llm::providers::defaults`]:
//!
//! - [`JsBaseProviderDefaults`] — universal fields (`beforeRequest`).
//! - [`JsCompletionProviderDefaults`] — completion role.
//! - [`JsEmbeddingProviderDefaults`] — embedding role.
//! - One class per role-specific defaults type (audio speech / music,
//!   voice cloning, image generation / upscale, video, transcription,
//!   3D, background removal).
//!
//! ## V1 hook handling
//!
//! Each `before*` hook is stored as `Option<ThreadsafeFunction<...>>` so
//! that JS callbacks captured at construction time survive being shipped
//! across threads to the framework's async runtime. Today the hooks are
//! **stored only** — Phase B wires them into actual dispatch through
//! `BeforeRequestHook` / `BeforeCompletionRequestHook` / role-specific
//! `Before*RequestHook` aliases. Until then the getters return whether a
//! hook is present and the setters replace it.

use std::sync::Arc;
use std::sync::Mutex;

use napi::bindgen_prelude::*;
use napi::threadsafe_function::ThreadsafeFunction;
use napi_derive::napi;

use blazen_llm::providers::defaults::{
    AudioMusicProviderDefaults, AudioSpeechProviderDefaults, BackgroundRemovalProviderDefaults,
    BaseProviderDefaults, CompletionProviderDefaults, EmbeddingProviderDefaults,
    ImageGenerationProviderDefaults, ImageUpscaleProviderDefaults, ThreeDProviderDefaults,
    TranscriptionProviderDefaults, VideoProviderDefaults, VoiceCloningProviderDefaults,
};
use blazen_llm::types::ToolDefinition;

use crate::generated::JsToolDefinition;

// ---------------------------------------------------------------------------
// Hook TSF type aliases
// ---------------------------------------------------------------------------

/// Universal `beforeRequest` hook TSF.
///
/// JS callback signature (as seen by the user):
/// `(method: string, request: any) => Promise<any | void>`
///
/// V1 only stores the handle — Phase B will dispatch to it from the
/// framework's async runtime through `BeforeRequestHook`.
pub(crate) type BeforeRequestTsfn = ThreadsafeFunction<
    (String, serde_json::Value),
    Promise<Option<serde_json::Value>>,
    (String, serde_json::Value),
    napi::Status,
    false,
    true,
>;

/// Typed `beforeCompletion` hook TSF.
///
/// JS callback signature: `(request: any) => Promise<any | void>`.
pub(crate) type BeforeCompletionTsfn = ThreadsafeFunction<
    serde_json::Value,
    Promise<Option<serde_json::Value>>,
    serde_json::Value,
    napi::Status,
    false,
    true,
>;

/// Role-specific typed `before*` hook TSF (same shape as
/// `BeforeCompletionTsfn`).
pub(crate) type BeforeRoleTsfn = ThreadsafeFunction<
    serde_json::Value,
    Promise<Option<serde_json::Value>>,
    serde_json::Value,
    napi::Status,
    false,
    true,
>;

// ---------------------------------------------------------------------------
// JsBaseProviderDefaults
// ---------------------------------------------------------------------------

/// Universal provider defaults applicable to every provider role.
///
/// Carries cross-cutting fields (currently just the `beforeRequest`
/// hook). Embedded as a `base` field on every role-specific defaults
/// class.
///
/// ```javascript
/// import { BaseProviderDefaults } from "blazen";
///
/// const d = new BaseProviderDefaults(async (method, request) => {
///   console.log("request via", method);
/// });
/// ```
#[napi(js_name = "BaseProviderDefaults")]
pub struct JsBaseProviderDefaults {
    /// Inner Rust value carrying the typed Rust-side hook closures. For
    /// V1 the closures are never populated from JS (Phase B), so the
    /// inner struct's `before_request` field is always `None`.
    pub(crate) inner: Arc<Mutex<BaseProviderDefaults>>,
    /// The raw JS callback handle. Kept around so getters can answer
    /// "is a hook configured?" without poking the inner struct, and so
    /// Phase B can invoke it directly.
    pub(crate) before_request: Arc<Mutex<Option<BeforeRequestTsfn>>>,
}

#[napi]
#[allow(clippy::must_use_candidate, clippy::needless_pass_by_value)]
impl JsBaseProviderDefaults {
    /// Construct a new [`BaseProviderDefaults`].
    ///
    /// `beforeRequest` is an optional async callback fired before any
    /// provider request (V1: stored only, not yet dispatched).
    #[napi(constructor)]
    pub fn new(before_request: Option<BeforeRequestTsfn>) -> Self {
        Self {
            inner: Arc::new(Mutex::new(BaseProviderDefaults::default())),
            before_request: Arc::new(Mutex::new(before_request)),
        }
    }

    /// Returns `true` when a `beforeRequest` hook is configured.
    #[napi(getter, js_name = "hasBeforeRequest")]
    pub fn has_before_request(&self) -> bool {
        self.before_request.lock().is_ok_and(|g| g.is_some())
    }

    /// Replace the `beforeRequest` hook. Pass `null` to clear.
    #[napi(setter, js_name = "beforeRequest")]
    pub fn set_before_request(&self, hook: Option<BeforeRequestTsfn>) {
        if let Ok(mut g) = self.before_request.lock() {
            *g = hook;
        }
    }
}

impl JsBaseProviderDefaults {
    /// Internal: build a Rust [`BaseProviderDefaults`] snapshot. Today
    /// the snapshot never carries a hook (Phase B will populate one).
    pub(crate) fn to_rust(&self) -> BaseProviderDefaults {
        self.inner
            .lock()
            .map_or_else(|_| BaseProviderDefaults::default(), |g| g.clone())
    }
}

// ---------------------------------------------------------------------------
// JsCompletionProviderDefaults
// ---------------------------------------------------------------------------

/// Completion-role provider defaults: system prompt, default tools,
/// `responseFormat`, and a typed `beforeCompletion` hook.
///
/// ```javascript
/// import { BaseProviderDefaults, CompletionProviderDefaults } from "blazen";
///
/// const d = new CompletionProviderDefaults(
///   new BaseProviderDefaults(),
///   "Be terse.",
///   [], // default tools
///   { type: "json_object" },
///   async (request) => { /* mutate request */ },
/// );
/// ```
#[napi(js_name = "CompletionProviderDefaults")]
pub struct JsCompletionProviderDefaults {
    pub(crate) base: Arc<Mutex<JsBaseProviderDefaults>>,
    pub(crate) system_prompt: Arc<Mutex<Option<String>>>,
    pub(crate) tools: Arc<Mutex<Vec<JsToolDefinition>>>,
    pub(crate) response_format: Arc<Mutex<Option<serde_json::Value>>>,
    pub(crate) before_completion: Arc<Mutex<Option<BeforeCompletionTsfn>>>,
}

#[napi]
#[allow(clippy::must_use_candidate, clippy::needless_pass_by_value)]
impl JsCompletionProviderDefaults {
    /// Construct completion-role defaults.
    #[napi(constructor)]
    pub fn new(
        base: Option<&JsBaseProviderDefaults>,
        system_prompt: Option<String>,
        tools: Option<Vec<JsToolDefinition>>,
        response_format: Option<serde_json::Value>,
        before_completion: Option<BeforeCompletionTsfn>,
    ) -> Self {
        let base_owned = base.map_or_else(|| JsBaseProviderDefaults::new(None), clone_base);
        Self {
            base: Arc::new(Mutex::new(base_owned)),
            system_prompt: Arc::new(Mutex::new(system_prompt)),
            tools: Arc::new(Mutex::new(tools.unwrap_or_default())),
            response_format: Arc::new(Mutex::new(response_format)),
            before_completion: Arc::new(Mutex::new(before_completion)),
        }
    }

    /// The system prompt prepended to requests when the request itself
    /// carries no system message.
    #[napi(getter, js_name = "systemPrompt")]
    pub fn system_prompt(&self) -> Option<String> {
        self.system_prompt.lock().ok().and_then(|g| g.clone())
    }

    /// Replace the system prompt. Pass `null` to clear.
    #[napi(setter, js_name = "systemPrompt")]
    pub fn set_system_prompt(&self, value: Option<String>) {
        if let Ok(mut g) = self.system_prompt.lock() {
            *g = value;
        }
    }

    /// The default tools appended to every completion request.
    #[napi(getter)]
    pub fn tools(&self) -> Vec<JsToolDefinition> {
        self.tools.lock().map_or_else(
            |_| Vec::new(),
            |g| {
                g.iter()
                    .map(|t| JsToolDefinition {
                        name: t.name.clone(),
                        description: t.description.clone(),
                        parameters: t.parameters.clone(),
                    })
                    .collect()
            },
        )
    }

    /// Replace the default tools.
    #[napi(setter)]
    pub fn set_tools(&self, value: Option<Vec<JsToolDefinition>>) {
        if let Ok(mut g) = self.tools.lock() {
            *g = value.unwrap_or_default();
        }
    }

    /// Default `response_format` (JSON Schema or similar object).
    #[napi(getter, js_name = "responseFormat")]
    pub fn response_format(&self) -> Option<serde_json::Value> {
        self.response_format.lock().ok().and_then(|g| g.clone())
    }

    /// Replace the default `responseFormat`. Pass `null` to clear.
    #[napi(setter, js_name = "responseFormat")]
    pub fn set_response_format(&self, value: Option<serde_json::Value>) {
        if let Ok(mut g) = self.response_format.lock() {
            *g = value;
        }
    }

    /// Returns `true` when a `beforeCompletion` hook is configured.
    #[napi(getter, js_name = "hasBeforeCompletion")]
    pub fn has_before_completion(&self) -> bool {
        self.before_completion.lock().is_ok_and(|g| g.is_some())
    }

    /// Replace the typed `beforeCompletion` hook. Pass `null` to clear.
    #[napi(setter, js_name = "beforeCompletion")]
    pub fn set_before_completion(&self, hook: Option<BeforeCompletionTsfn>) {
        if let Ok(mut g) = self.before_completion.lock() {
            *g = hook;
        }
    }
}

impl JsCompletionProviderDefaults {
    /// Internal: snapshot the JS-side defaults into a Rust
    /// [`CompletionProviderDefaults`] for use by `BaseProvider`. V1
    /// snapshots field values only — hooks are dropped because they
    /// require the Phase B dispatch wiring to convert TSFs into the
    /// Rust closure aliases.
    pub(crate) fn to_rust(&self) -> CompletionProviderDefaults {
        let base = self
            .base
            .lock()
            .map_or_else(|_| BaseProviderDefaults::default(), |g| g.to_rust());
        let system_prompt = self.system_prompt.lock().ok().and_then(|g| g.clone());
        let tools: Vec<ToolDefinition> = self.tools.lock().map_or_else(
            |_| Vec::new(),
            |g| {
                g.iter()
                    .map(|t| ToolDefinition {
                        name: t.name.clone(),
                        description: t.description.clone(),
                        parameters: t.parameters.clone(),
                    })
                    .collect()
            },
        );
        let response_format = self.response_format.lock().ok().and_then(|g| g.clone());

        let mut d = CompletionProviderDefaults::new().with_base(base);
        if let Some(sp) = system_prompt {
            d = d.with_system_prompt(sp);
        }
        if !tools.is_empty() {
            d = d.with_tools(tools);
        }
        if let Some(rf) = response_format {
            d = d.with_response_format(rf);
        }
        d
    }

    /// Internal: clone the JS object preserving the shared Arc<Mutex>
    /// pointers so getters/setters on the original surface here too.
    pub(crate) fn clone_shared(&self) -> Self {
        Self {
            base: Arc::clone(&self.base),
            system_prompt: Arc::clone(&self.system_prompt),
            tools: Arc::clone(&self.tools),
            response_format: Arc::clone(&self.response_format),
            before_completion: Arc::clone(&self.before_completion),
        }
    }
}

// ---------------------------------------------------------------------------
// JsEmbeddingProviderDefaults
// ---------------------------------------------------------------------------

/// Embedding-role provider defaults. V1 just wraps a
/// [`JsBaseProviderDefaults`].
#[napi(js_name = "EmbeddingProviderDefaults")]
pub struct JsEmbeddingProviderDefaults {
    // Held for parity with the other role-defaults wrappers. Phase B
    // routes embedding-side `beforeRequest` through this slot.
    #[allow(dead_code)]
    pub(crate) base: Arc<Mutex<JsBaseProviderDefaults>>,
}

#[napi]
#[allow(clippy::must_use_candidate, clippy::needless_pass_by_value)]
impl JsEmbeddingProviderDefaults {
    /// Construct embedding-role defaults.
    #[napi(constructor)]
    pub fn new(base: Option<&JsBaseProviderDefaults>) -> Self {
        let base_owned = base.map_or_else(|| JsBaseProviderDefaults::new(None), clone_base);
        Self {
            base: Arc::new(Mutex::new(base_owned)),
        }
    }
}

impl JsEmbeddingProviderDefaults {
    #[allow(dead_code)]
    pub(crate) fn to_rust(&self) -> EmbeddingProviderDefaults {
        let base = self
            .base
            .lock()
            .map_or_else(|_| BaseProviderDefaults::default(), |g| g.to_rust());
        EmbeddingProviderDefaults::new().with_base(base)
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn clone_base(b: &JsBaseProviderDefaults) -> JsBaseProviderDefaults {
    JsBaseProviderDefaults {
        inner: Arc::clone(&b.inner),
        before_request: Arc::clone(&b.before_request),
    }
}

// ---------------------------------------------------------------------------
// Role-specific defaults — generated via macro
// ---------------------------------------------------------------------------
//
// Each role wraps `BaseProviderDefaults` plus a typed before-`<role>`-request
// hook. V1 stores the hook handle without dispatching it.

macro_rules! js_role_defaults {
    ($js_name:ident, $rust_ty:ty, $class_name:expr) => {
        #[doc = concat!(
            "Role-specific defaults wrapping a [`BaseProviderDefaults`] plus a typed `before` hook. See [`",
            stringify!($rust_ty),
            "`] in the core crate."
        )]
        #[napi(js_name = $class_name)]
        pub struct $js_name {
            pub(crate) base: Arc<Mutex<JsBaseProviderDefaults>>,
            pub(crate) before: Arc<Mutex<Option<BeforeRoleTsfn>>>,
        }

        #[napi]
        #[allow(clippy::must_use_candidate, clippy::needless_pass_by_value)]
        impl $js_name {
            /// Construct role-specific defaults.
            #[napi(constructor)]
            pub fn new(
                base: Option<&JsBaseProviderDefaults>,
                before: Option<BeforeRoleTsfn>,
            ) -> Self {
                let base_owned = base
                    .map_or_else(|| JsBaseProviderDefaults::new(None), clone_base);
                Self {
                    base: Arc::new(Mutex::new(base_owned)),
                    before: Arc::new(Mutex::new(before)),
                }
            }

            /// Returns `true` when a `before` hook is configured.
            #[napi(getter, js_name = "hasBefore")]
            pub fn has_before(&self) -> bool {
                self.before.lock().is_ok_and(|g| g.is_some())
            }

            /// Replace the typed `before` hook. Pass `null` to clear.
            #[napi(setter, js_name = "before")]
            pub fn set_before(&self, hook: Option<BeforeRoleTsfn>) {
                if let Ok(mut g) = self.before.lock() {
                    *g = hook;
                }
            }
        }

        impl $js_name {
            #[allow(dead_code)]
            pub(crate) fn to_rust(&self) -> $rust_ty {
                let base = self
                    .base
                    .lock()
                    .map_or_else(|_| BaseProviderDefaults::default(), |g| g.to_rust());
                <$rust_ty>::new().with_base(base)
            }
        }
    };
}

js_role_defaults!(
    JsAudioSpeechProviderDefaults,
    AudioSpeechProviderDefaults,
    "AudioSpeechProviderDefaults"
);
js_role_defaults!(
    JsAudioMusicProviderDefaults,
    AudioMusicProviderDefaults,
    "AudioMusicProviderDefaults"
);
js_role_defaults!(
    JsVoiceCloningProviderDefaults,
    VoiceCloningProviderDefaults,
    "VoiceCloningProviderDefaults"
);
js_role_defaults!(
    JsImageGenerationProviderDefaults,
    ImageGenerationProviderDefaults,
    "ImageGenerationProviderDefaults"
);
js_role_defaults!(
    JsImageUpscaleProviderDefaults,
    ImageUpscaleProviderDefaults,
    "ImageUpscaleProviderDefaults"
);
js_role_defaults!(
    JsVideoProviderDefaults,
    VideoProviderDefaults,
    "VideoProviderDefaults"
);
js_role_defaults!(
    JsTranscriptionProviderDefaults,
    TranscriptionProviderDefaults,
    "TranscriptionProviderDefaults"
);
js_role_defaults!(
    JsThreeDProviderDefaults,
    ThreeDProviderDefaults,
    "ThreeDProviderDefaults"
);
js_role_defaults!(
    JsBackgroundRemovalProviderDefaults,
    BackgroundRemovalProviderDefaults,
    "BackgroundRemovalProviderDefaults"
);
