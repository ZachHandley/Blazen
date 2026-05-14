//! `wasm-bindgen` wrapper for [`blazen_llm::providers::custom::ApiProtocol`].
//!
//! Exposes a JS-friendly tagged class with two static factory methods:
//! `ApiProtocol.openai(config)` and `ApiProtocol.custom()`.
//!
//! The wrapped Rust enum is `ApiProtocol::OpenAi(OpenAiCompatConfig)` or
//! `ApiProtocol::Custom`. The `kind` getter returns a stable string tag.

use wasm_bindgen::prelude::*;

use super::openai_compat::WasmOpenAiCompatConfig;

/// Selects how a `CustomProvider` talks to its backend for completion calls.
///
/// V1 has two variants:
/// - `openai(config)` — `OpenAI` Chat Completions wire format using the
///   supplied [`WasmOpenAiCompatConfig`].
/// - `custom()` — completion calls dispatch to a host-language object (wired
///   up in Phase B via `CustomProvider.withDispatch`).
///
/// ```js
/// const cfg = new WasmOpenAiCompatConfig('my-llm', 'https://api.example.com/v1', 'sk-...', 'gpt-4o');
/// const proto = ApiProtocol.openai(cfg);
/// // ... or ...
/// const proto = ApiProtocol.custom();
/// ```
#[wasm_bindgen(js_name = "ApiProtocol")]
pub struct WasmApiProtocol {
    kind: ApiProtocolKind,
    config: Option<WasmOpenAiCompatConfig>,
}

#[derive(Clone, Copy)]
enum ApiProtocolKind {
    OpenAi,
    Custom,
}

impl ApiProtocolKind {
    fn as_str(self) -> &'static str {
        match self {
            Self::OpenAi => "openai",
            Self::Custom => "custom",
        }
    }
}

impl Clone for WasmApiProtocol {
    fn clone(&self) -> Self {
        Self {
            kind: self.kind,
            config: self.config.clone(),
        }
    }
}

impl WasmApiProtocol {
    /// Crate-internal: returns `true` when this protocol is the `OpenAi`
    /// variant. Phase B `WasmCustomProvider` factories use this to decide
    /// whether the `openaiCompat` factory should accept the wrapped config.
    #[allow(dead_code)]
    pub(crate) fn is_openai(&self) -> bool {
        matches!(self.kind, ApiProtocolKind::OpenAi)
    }

    /// Crate-internal: returns a clone of the wrapped
    /// [`WasmOpenAiCompatConfig`] when this protocol is the `OpenAi`
    /// variant.
    #[allow(dead_code)]
    pub(crate) fn config_clone(&self) -> Option<WasmOpenAiCompatConfig> {
        self.config.clone()
    }
}

#[wasm_bindgen(js_class = "ApiProtocol")]
impl WasmApiProtocol {
    /// `OpenAI` Chat Completions wire format. The supplied config carries the
    /// base URL, model, API key, and auth method.
    #[wasm_bindgen(js_name = "openai")]
    #[must_use]
    pub fn openai(config: &WasmOpenAiCompatConfig) -> WasmApiProtocol {
        Self {
            kind: ApiProtocolKind::OpenAi,
            config: Some(config.clone()),
        }
    }

    /// User-defined protocol. Completion calls dispatch to the host-language
    /// object configured on the `CustomProvider` (wired up in Phase B).
    #[wasm_bindgen(js_name = "custom")]
    #[must_use]
    pub fn custom() -> WasmApiProtocol {
        Self {
            kind: ApiProtocolKind::Custom,
            config: None,
        }
    }

    /// Stable string tag for the variant: `"openai"` or `"custom"`.
    #[wasm_bindgen(getter)]
    #[must_use]
    pub fn kind(&self) -> String {
        self.kind.as_str().to_owned()
    }

    /// The wrapped [`WasmOpenAiCompatConfig`] when `kind === "openai"`,
    /// otherwise `undefined`.
    #[wasm_bindgen(getter)]
    #[must_use]
    pub fn config(&self) -> Option<WasmOpenAiCompatConfig> {
        self.config.clone()
    }
}
