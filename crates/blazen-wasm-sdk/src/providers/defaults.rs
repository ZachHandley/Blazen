//! `wasm-bindgen` wrappers for the provider-defaults hierarchy from
//! [`blazen_llm::providers::defaults`].
//!
//! Eleven classes total:
//! - [`WasmBaseProviderDefaults`] — universal `before_request` hook holder.
//! - [`WasmCompletionProviderDefaults`] — completion-role defaults (system
//!   prompt, default tools, response format, `before_completion` hook).
//! - [`WasmEmbeddingProviderDefaults`] — embedding-role defaults (only
//!   `base` for V1).
//! - Nine role-specific defaults wrappers (audio speech, audio music, voice
//!   cloning, image generation, image upscale, video, transcription, 3D,
//!   background removal).
//!
//! V1 surface: hooks are stored as [`js_sys::Function`] but not invoked here
//! — Phase B threads `JsFuture`-aware dispatch through `CustomProvider`. The
//! getters/setters on these classes give JS callers a typed handle to attach
//! and inspect the hook references.

use std::cell::RefCell;
use std::rc::Rc;

use js_sys::{Array, Function};
use wasm_bindgen::prelude::*;

// ---------------------------------------------------------------------------
// BaseProviderDefaults
// ---------------------------------------------------------------------------

/// Universal provider defaults applicable to any provider role.
///
/// Carries the universal `before_request` hook — a JS function that the
/// framework invokes before any completion / compute call with the method
/// name (e.g. `"complete"`, `"text_to_speech"`) and a mutable JSON view of
/// the request body.
///
/// V1 only stores the hook reference; Phase B wires actual `await` calls
/// through `CustomProvider`'s dispatch path.
#[wasm_bindgen(js_name = "BaseProviderDefaults")]
#[derive(Default)]
pub struct WasmBaseProviderDefaults {
    before_request: Rc<RefCell<Option<Function>>>,
}

impl Clone for WasmBaseProviderDefaults {
    fn clone(&self) -> Self {
        Self {
            before_request: Rc::clone(&self.before_request),
        }
    }
}

#[wasm_bindgen(js_class = "BaseProviderDefaults")]
impl WasmBaseProviderDefaults {
    /// Create a new instance. `beforeRequest` is an optional JS function
    /// that the framework will invoke before any provider request.
    #[wasm_bindgen(constructor)]
    #[must_use]
    pub fn new(before_request: Option<Function>) -> WasmBaseProviderDefaults {
        Self {
            before_request: Rc::new(RefCell::new(before_request)),
        }
    }

    /// The configured `before_request` hook, or `undefined`.
    #[wasm_bindgen(getter, js_name = "beforeRequest")]
    #[must_use]
    pub fn before_request(&self) -> Option<Function> {
        self.before_request.borrow().clone()
    }

    /// Replace the `before_request` hook.
    #[wasm_bindgen(setter, js_name = "beforeRequest")]
    pub fn set_before_request(&self, hook: Option<Function>) {
        *self.before_request.borrow_mut() = hook;
    }
}

// ---------------------------------------------------------------------------
// CompletionProviderDefaults
// ---------------------------------------------------------------------------

/// Completion-role defaults. Carries the universal `base` plus completion-
/// specific fields: `systemPrompt`, default `tools`, default `responseFormat`,
/// and a typed `beforeCompletion` hook.
#[wasm_bindgen(js_name = "CompletionProviderDefaults")]
#[derive(Default)]
pub struct WasmCompletionProviderDefaults {
    base: Rc<RefCell<WasmBaseProviderDefaults>>,
    system_prompt: Rc<RefCell<Option<String>>>,
    tools: Rc<RefCell<Array>>,
    response_format: Rc<RefCell<JsValue>>,
    before_completion: Rc<RefCell<Option<Function>>>,
}

impl Clone for WasmCompletionProviderDefaults {
    fn clone(&self) -> Self {
        Self {
            base: Rc::clone(&self.base),
            system_prompt: Rc::clone(&self.system_prompt),
            tools: Rc::clone(&self.tools),
            response_format: Rc::clone(&self.response_format),
            before_completion: Rc::clone(&self.before_completion),
        }
    }
}

#[wasm_bindgen(js_class = "CompletionProviderDefaults")]
impl WasmCompletionProviderDefaults {
    /// Create a new instance. All arguments are optional.
    ///
    /// - `base`: universal defaults (defaults to an empty
    ///   [`WasmBaseProviderDefaults`]).
    /// - `systemPrompt`: prepended as a system message when the request has none.
    /// - `tools`: JS array of tool definitions appended to the request's tools.
    /// - `responseFormat`: applied when the request lacks a `responseFormat`.
    /// - `beforeCompletion`: typed completion hook (JS function returning a Promise).
    #[wasm_bindgen(constructor)]
    #[must_use]
    pub fn new(
        base: Option<WasmBaseProviderDefaults>,
        system_prompt: Option<String>,
        tools: Option<Array>,
        response_format: JsValue,
        before_completion: Option<Function>,
    ) -> WasmCompletionProviderDefaults {
        Self {
            base: Rc::new(RefCell::new(base.unwrap_or_default())),
            system_prompt: Rc::new(RefCell::new(system_prompt)),
            tools: Rc::new(RefCell::new(tools.unwrap_or_else(Array::new))),
            response_format: Rc::new(RefCell::new(response_format)),
            before_completion: Rc::new(RefCell::new(before_completion)),
        }
    }

    /// The configured universal `base` defaults.
    #[wasm_bindgen(getter)]
    #[must_use]
    pub fn base(&self) -> WasmBaseProviderDefaults {
        self.base.borrow().clone()
    }

    /// Replace the universal `base` defaults.
    #[wasm_bindgen(setter)]
    pub fn set_base(&self, base: WasmBaseProviderDefaults) {
        *self.base.borrow_mut() = base;
    }

    /// The configured default system prompt, or `undefined`.
    #[wasm_bindgen(getter, js_name = "systemPrompt")]
    #[must_use]
    pub fn system_prompt(&self) -> Option<String> {
        self.system_prompt.borrow().clone()
    }

    /// Replace the default system prompt.
    #[wasm_bindgen(setter, js_name = "systemPrompt")]
    pub fn set_system_prompt(&self, value: Option<String>) {
        *self.system_prompt.borrow_mut() = value;
    }

    /// The configured default tools as a JS array (may be empty).
    #[wasm_bindgen(getter)]
    #[must_use]
    pub fn tools(&self) -> Array {
        self.tools.borrow().clone()
    }

    /// Replace the default tools.
    #[wasm_bindgen(setter)]
    pub fn set_tools(&self, tools: Array) {
        *self.tools.borrow_mut() = tools;
    }

    /// The configured default `responseFormat`, or `undefined`.
    #[wasm_bindgen(getter, js_name = "responseFormat")]
    #[must_use]
    pub fn response_format(&self) -> JsValue {
        self.response_format.borrow().clone()
    }

    /// Replace the default `responseFormat`.
    #[wasm_bindgen(setter, js_name = "responseFormat")]
    pub fn set_response_format(&self, value: JsValue) {
        *self.response_format.borrow_mut() = value;
    }

    /// The configured `beforeCompletion` hook, or `undefined`.
    #[wasm_bindgen(getter, js_name = "beforeCompletion")]
    #[must_use]
    pub fn before_completion(&self) -> Option<Function> {
        self.before_completion.borrow().clone()
    }

    /// Replace the `beforeCompletion` hook.
    #[wasm_bindgen(setter, js_name = "beforeCompletion")]
    pub fn set_before_completion(&self, hook: Option<Function>) {
        *self.before_completion.borrow_mut() = hook;
    }
}

// ---------------------------------------------------------------------------
// EmbeddingProviderDefaults
// ---------------------------------------------------------------------------

/// Embedding-role defaults. V1 carries only the universal `base` defaults.
#[wasm_bindgen(js_name = "EmbeddingProviderDefaults")]
#[derive(Default)]
pub struct WasmEmbeddingProviderDefaults {
    base: Rc<RefCell<WasmBaseProviderDefaults>>,
}

impl Clone for WasmEmbeddingProviderDefaults {
    fn clone(&self) -> Self {
        Self {
            base: Rc::clone(&self.base),
        }
    }
}

#[wasm_bindgen(js_class = "EmbeddingProviderDefaults")]
impl WasmEmbeddingProviderDefaults {
    /// Create a new instance. `base` defaults to an empty
    /// [`WasmBaseProviderDefaults`] when omitted.
    #[wasm_bindgen(constructor)]
    #[must_use]
    pub fn new(base: Option<WasmBaseProviderDefaults>) -> WasmEmbeddingProviderDefaults {
        Self {
            base: Rc::new(RefCell::new(base.unwrap_or_default())),
        }
    }

    /// The configured universal `base` defaults.
    #[wasm_bindgen(getter)]
    #[must_use]
    pub fn base(&self) -> WasmBaseProviderDefaults {
        self.base.borrow().clone()
    }

    /// Replace the universal `base` defaults.
    #[wasm_bindgen(setter)]
    pub fn set_base(&self, base: WasmBaseProviderDefaults) {
        *self.base.borrow_mut() = base;
    }
}

// ---------------------------------------------------------------------------
// Role-specific defaults (9 classes)
// ---------------------------------------------------------------------------

/// Generate a role-specific defaults wrapper class that composes
/// [`WasmBaseProviderDefaults`] plus an optional typed `before` hook.
macro_rules! role_defaults {
    ($name:ident, $js_name:literal) => {
        #[wasm_bindgen(js_name = $js_name)]
        #[derive(Default)]
        pub struct $name {
            base: Rc<RefCell<WasmBaseProviderDefaults>>,
            before: Rc<RefCell<Option<Function>>>,
        }

        impl Clone for $name {
            fn clone(&self) -> Self {
                Self {
                    base: Rc::clone(&self.base),
                    before: Rc::clone(&self.before),
                }
            }
        }

        #[wasm_bindgen(js_class = $js_name)]
        impl $name {
            /// Create a new instance. Both arguments are optional.
            #[wasm_bindgen(constructor)]
            #[must_use]
            pub fn new(base: Option<WasmBaseProviderDefaults>, before: Option<Function>) -> $name {
                Self {
                    base: Rc::new(RefCell::new(base.unwrap_or_default())),
                    before: Rc::new(RefCell::new(before)),
                }
            }

            /// The configured universal `base` defaults.
            #[wasm_bindgen(getter)]
            #[must_use]
            pub fn base(&self) -> WasmBaseProviderDefaults {
                self.base.borrow().clone()
            }

            /// Replace the universal `base` defaults.
            #[wasm_bindgen(setter)]
            pub fn set_base(&self, base: WasmBaseProviderDefaults) {
                *self.base.borrow_mut() = base;
            }

            /// The configured typed `before` hook, or `undefined`.
            #[wasm_bindgen(getter)]
            #[must_use]
            pub fn before(&self) -> Option<Function> {
                self.before.borrow().clone()
            }

            /// Replace the typed `before` hook.
            #[wasm_bindgen(setter)]
            pub fn set_before(&self, hook: Option<Function>) {
                *self.before.borrow_mut() = hook;
            }
        }
    };
}

role_defaults!(
    WasmAudioSpeechProviderDefaults,
    "AudioSpeechProviderDefaults"
);
role_defaults!(WasmAudioMusicProviderDefaults, "AudioMusicProviderDefaults");
role_defaults!(
    WasmVoiceCloningProviderDefaults,
    "VoiceCloningProviderDefaults"
);
role_defaults!(
    WasmImageGenerationProviderDefaults,
    "ImageGenerationProviderDefaults"
);
role_defaults!(
    WasmImageUpscaleProviderDefaults,
    "ImageUpscaleProviderDefaults"
);
role_defaults!(WasmVideoProviderDefaults, "VideoProviderDefaults");
role_defaults!(
    WasmTranscriptionProviderDefaults,
    "TranscriptionProviderDefaults"
);
role_defaults!(WasmThreeDProviderDefaults, "ThreeDProviderDefaults");
role_defaults!(
    WasmBackgroundRemovalProviderDefaults,
    "BackgroundRemovalProviderDefaults"
);
