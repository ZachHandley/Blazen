//! API key resolution for the WASM component.
//!
//! The [`KeyProvider`] reads API keys based on a configured [`KeyStrategy`].
//! For the MVP, only [`KeyStrategy::RuntimeInjection`] is implemented, which
//! reads keys from environment variables at startup.

use std::collections::HashMap;

/// Strategy for resolving API keys at runtime.
#[derive(Debug, Clone)]
pub enum KeyStrategy {
    /// Read keys from environment variables injected by the runtime.
    /// This is the simplest approach: the host sets env vars like
    /// `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, etc.
    RuntimeInjection,

    /// Read keys from `ZLayer`'s secret store.
    /// TODO: Implement via `wasi:keyvalue` or a ZLayer-specific interface.
    ZLayerSecrets,

    /// Keys are stored encrypted and decrypted at startup with a master key.
    /// TODO: Implement envelope encryption with a runtime-provided master key.
    Encrypted,

    /// All outbound requests are routed through a key-injecting proxy.
    /// The component never sees raw API keys; the proxy adds auth headers.
    /// TODO: Implement proxy-mode where we skip auth header injection entirely.
    Proxy,
}

use blazen_llm::keys::PROVIDER_ENV_VARS;

/// Resolves API keys for LLM providers.
#[derive(Debug, Clone)]
pub struct KeyProvider {
    strategy: KeyStrategy,
    /// Cached keys, keyed by provider name (lowercase).
    keys: HashMap<String, String>,
}

impl KeyProvider {
    /// Create a new key provider with the given strategy.
    ///
    /// For [`KeyStrategy::RuntimeInjection`], this immediately reads all
    /// known environment variables and caches the values.
    #[must_use]
    pub fn new(strategy: KeyStrategy) -> Self {
        let keys = match &strategy {
            KeyStrategy::RuntimeInjection => Self::load_from_env(),
            KeyStrategy::ZLayerSecrets => {
                // TODO: Load from ZLayer secret store
                HashMap::new()
            }
            KeyStrategy::Encrypted => {
                // TODO: Load and decrypt keys
                HashMap::new()
            }
            KeyStrategy::Proxy => {
                // No keys needed -- proxy handles auth
                HashMap::new()
            }
        };

        Self { strategy, keys }
    }

    /// Create a key provider using runtime injection (env vars).
    #[must_use]
    pub fn from_env() -> Self {
        Self::new(KeyStrategy::RuntimeInjection)
    }

    /// Look up the API key for a given provider.
    #[must_use]
    pub fn get(&self, provider: &str) -> Option<&str> {
        self.keys.get(provider).map(String::as_str)
    }

    /// Check whether we have a key for the given provider.
    #[must_use]
    pub fn has_key(&self, provider: &str) -> bool {
        self.keys.contains_key(provider)
    }

    /// Return the active strategy.
    #[must_use]
    pub fn strategy(&self) -> &KeyStrategy {
        &self.strategy
    }

    /// List all providers for which we have keys.
    pub fn available_providers(&self) -> impl Iterator<Item = &str> {
        self.keys.keys().map(String::as_str)
    }

    /// Read all known provider env vars and return the ones that are set.
    fn load_from_env() -> HashMap<String, String> {
        let mut keys = HashMap::new();
        for &(provider, env_var) in PROVIDER_ENV_VARS {
            if let Ok(value) = std::env::var(env_var) {
                if !value.is_empty() {
                    keys.insert(provider.to_owned(), value);
                }
            }
        }
        keys
    }
}
