//! API key resolution for LLM providers.
//!
//! Maps provider names to well-known environment variables and provides
//! a single [`resolve_api_key`] entry-point. Resolution order is:
//! explicit key → installed [`KeyResolver`] chain (scoped to
//! [`current_place`]) → the corresponding environment variable.
//!
//! With no resolvers installed (the default), behaviour is identical to a
//! plain explicit-then-env lookup, so standalone Blazen is unaffected.

use std::sync::{Arc, LazyLock, RwLock};

use crate::BlazenError;

/// Where a resolved key came from. Surfaced for telemetry/auditing; a
/// resolver implementation MUST never log the key value itself.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KeySource {
    /// The caller passed the key explicitly.
    Explicit,
    /// An installed [`KeyResolver`] in the chain returned the key.
    Resolver,
    /// The provider's environment variable supplied the key.
    Env,
}

/// A pluggable link in the key-resolution cascade.
///
/// Resolvers are consulted, in install order, AFTER the explicit-key check
/// and BEFORE the environment-variable terminal. The first resolver to
/// return `Some(key)` wins. `place` is the current tenant scope (see
/// [`set_current_place`]); `None` means the default/standalone place.
///
/// Implementations MUST NOT log the key value.
pub trait KeyResolver: Send + Sync {
    /// Return a key for `provider` in tenant `place`, or `None` to defer to
    /// the next link in the chain.
    fn resolve(&self, provider: &str, place: Option<&str>) -> Option<String>;
}

/// The process-global resolver chain, consulted by [`resolve_api_key`].
///
/// Empty by default (standalone behaviour). Re-installable + clearable
/// (RwLock-backed, unlike a set-once `OnceLock`) so a worker can install its
/// chain at startup and tests can reset between cases.
static RESOLVER_CHAIN: LazyLock<RwLock<Vec<Arc<dyn KeyResolver>>>> =
    LazyLock::new(|| RwLock::new(Vec::new()));

/// The current tenant/place scope, passed to resolvers by [`resolve_api_key`].
///
/// A worker process serves a single place, so a process-global value is
/// sufficient and correct here; standalone leaves it `None`. `RwLock::new`
/// is const, so no lazy init is needed.
static CURRENT_PLACE: RwLock<Option<String>> = RwLock::new(None);

/// Install the resolver chain, replacing any prior install. Order is
/// priority — earlier resolvers are consulted first.
///
/// # Panics
///
/// Panics if the resolver-chain lock has been poisoned by a prior panic
/// while it was held.
pub fn install_key_resolvers(chain: Vec<Arc<dyn KeyResolver>>) {
    *RESOLVER_CHAIN.write().expect("key resolver chain poisoned") = chain;
}

/// Append one resolver to the end of the chain (consulted after existing
/// resolvers but still before the env terminal).
///
/// # Panics
///
/// Panics if the resolver-chain lock has been poisoned by a prior panic
/// while it was held.
pub fn push_key_resolver(resolver: Arc<dyn KeyResolver>) {
    RESOLVER_CHAIN
        .write()
        .expect("key resolver chain poisoned")
        .push(resolver);
}

/// Clear the resolver chain, restoring pure explicit-then-env behaviour.
///
/// # Panics
///
/// Panics if the resolver-chain lock has been poisoned by a prior panic
/// while it was held.
pub fn clear_key_resolvers() {
    RESOLVER_CHAIN
        .write()
        .expect("key resolver chain poisoned")
        .clear();
}

/// Set the current tenant/place scope. Resolvers receive this via their
/// [`KeyResolver::resolve`] call. `None` selects the default place.
///
/// # Panics
///
/// Panics if the current-place lock has been poisoned by a prior panic
/// while it was held.
pub fn set_current_place(place: Option<String>) {
    *CURRENT_PLACE.write().expect("current place poisoned") = place;
}

/// Read the current tenant/place scope.
///
/// # Panics
///
/// Panics if the current-place lock has been poisoned by a prior panic
/// while it was held.
#[must_use]
pub fn current_place() -> Option<String> {
    CURRENT_PLACE
        .read()
        .expect("current place poisoned")
        .clone()
}

/// Well-known provider names and their corresponding environment variables.
pub const PROVIDER_ENV_VARS: &[(&str, &str)] = &[
    ("openai", "OPENAI_API_KEY"),
    ("anthropic", "ANTHROPIC_API_KEY"),
    ("gemini", "GEMINI_API_KEY"),
    ("azure", "AZURE_OPENAI_API_KEY"),
    ("fal", "FAL_KEY"),
    ("openrouter", "OPENROUTER_API_KEY"),
    ("groq", "GROQ_API_KEY"),
    ("together", "TOGETHER_API_KEY"),
    ("mistral", "MISTRAL_API_KEY"),
    ("deepseek", "DEEPSEEK_API_KEY"),
    ("fireworks", "FIREWORKS_API_KEY"),
    ("perplexity", "PERPLEXITY_API_KEY"),
    ("xai", "XAI_API_KEY"),
    ("cohere", "COHERE_API_KEY"),
    ("bedrock", "AWS_ACCESS_KEY_ID"),
];

/// Return the environment variable name for a given provider, if known.
#[must_use]
pub fn env_var_for_provider(provider: &str) -> Option<&'static str> {
    PROVIDER_ENV_VARS
        .iter()
        .find(|(name, _)| *name == provider)
        .map(|(_, var)| *var)
}

/// Resolve an API key for `provider`.
///
/// Resolution order:
/// 1. `explicit` — if the caller passed a non-empty key directly, use it.
/// 2. Installed [`KeyResolver`] chain — consulted in order, scoped to
///    [`current_place`]; the first `Some` wins.
/// 3. Environment variable — looked up via [`PROVIDER_ENV_VARS`].
/// 4. Error — returns [`BlazenError::Auth`] with a helpful message.
///
/// # Panics
///
/// Panics if the resolver-chain or current-place lock has been poisoned by
/// a prior panic while it was held.
///
/// # Errors
///
/// Returns [`BlazenError::Auth`] if no explicit key is provided, no resolver
/// in the chain returns a key, and the environment variable for the provider
/// is either unset or empty.
pub fn resolve_api_key(provider: &str, explicit: Option<String>) -> Result<String, BlazenError> {
    if let Some(key) = explicit.filter(|k| !k.is_empty()) {
        return Ok(key);
    }

    // Resolver chain (scoped to the current place). Snapshot the Arcs under a
    // short read lock, then release it before calling resolvers so a resolver
    // is free to touch the chain without deadlocking.
    let place = current_place();
    let resolvers: Vec<Arc<dyn KeyResolver>> = RESOLVER_CHAIN
        .read()
        .expect("key resolver chain poisoned")
        .clone();
    for resolver in &resolvers {
        if let Some(key) = resolver
            .resolve(provider, place.as_deref())
            .filter(|k| !k.is_empty())
        {
            return Ok(key);
        }
    }

    if let Some(env_var) = env_var_for_provider(provider) {
        if let Ok(key) = std::env::var(env_var)
            && !key.is_empty()
        {
            return Ok(key);
        }
        Err(BlazenError::Auth {
            message: format!(
                "no API key for {provider}: set the {env_var} environment variable \
                 or pass api_key in options"
            ),
        })
    } else {
        Err(BlazenError::Auth {
            message: format!(
                "no API key for {provider}: pass api_key in options \
                 (no known environment variable for this provider)"
            ),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn explicit_key_takes_priority() {
        // Even if env var is set, explicit wins.
        let result = resolve_api_key("openai", Some("sk-explicit".into()));
        assert_eq!(result.unwrap(), "sk-explicit");
    }

    #[test]
    fn empty_explicit_key_falls_through() {
        let result = resolve_api_key("openai", Some(String::new()));
        // Should try env var, then error (env var likely not set in tests).
        assert!(result.is_err() || !result.unwrap().is_empty());
    }

    #[test]
    fn unknown_provider_errors() {
        let result = resolve_api_key("nonexistent", None);
        assert!(result.is_err());
        let msg = result.unwrap_err().to_string();
        assert!(msg.contains("nonexistent"));
    }

    #[test]
    fn env_var_mapping_complete() {
        // Spot-check a few mappings.
        assert_eq!(env_var_for_provider("openai"), Some("OPENAI_API_KEY"));
        assert_eq!(env_var_for_provider("fal"), Some("FAL_KEY"));
        assert_eq!(env_var_for_provider("bedrock"), Some("AWS_ACCESS_KEY_ID"));
        assert_eq!(env_var_for_provider("nonexistent"), None);
    }

    // ---- Resolver chain (P1) ----
    //
    // The resolver chain + current place are process-global, so these tests
    // must not run concurrently. Serialize them with a mutex and always tear
    // down the global state (clear chain + reset place) before asserting.

    static SERIAL: std::sync::Mutex<()> = std::sync::Mutex::new(());

    /// A resolver returning a fixed key for a specific (provider, place) pair.
    struct MockResolver {
        provider: String,
        place: Option<String>,
        key: String,
    }

    impl KeyResolver for MockResolver {
        fn resolve(&self, provider: &str, place: Option<&str>) -> Option<String> {
            if provider == self.provider && place.map(str::to_owned) == self.place {
                Some(self.key.clone())
            } else {
                None
            }
        }
    }

    fn reset() {
        clear_key_resolvers();
        set_current_place(None);
    }

    #[test]
    fn empty_chain_preserves_explicit_and_env_behavior() {
        let _g = SERIAL
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        reset();
        // Explicit still wins with an empty chain.
        assert_eq!(
            resolve_api_key("openai", Some("sk-explicit".into())).unwrap(),
            "sk-explicit"
        );
        // Unknown provider with no key still errors (env terminal unchanged).
        assert!(resolve_api_key("nonexistent", None).is_err());
        reset();
    }

    #[test]
    fn explicit_beats_resolver() {
        let _g = SERIAL
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        reset();
        install_key_resolvers(vec![Arc::new(MockResolver {
            provider: "openai".into(),
            place: None,
            key: "sk-from-resolver".into(),
        })]);
        assert_eq!(
            resolve_api_key("openai", Some("sk-explicit".into())).unwrap(),
            "sk-explicit"
        );
        reset();
    }

    #[test]
    fn resolver_beats_env() {
        let _g = SERIAL
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        reset();
        // `fal` has an env var (FAL_KEY); the resolver must win over it even
        // when set, and supply a key when it is not.
        install_key_resolvers(vec![Arc::new(MockResolver {
            provider: "fal".into(),
            place: None,
            key: "sk-resolver-fal".into(),
        })]);
        assert_eq!(resolve_api_key("fal", None).unwrap(), "sk-resolver-fal");
        reset();
    }

    #[test]
    fn first_resolver_in_chain_wins() {
        let _g = SERIAL
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        reset();
        install_key_resolvers(vec![
            Arc::new(MockResolver {
                provider: "openai".into(),
                place: None,
                key: "sk-first".into(),
            }),
            Arc::new(MockResolver {
                provider: "openai".into(),
                place: None,
                key: "sk-second".into(),
            }),
        ]);
        assert_eq!(resolve_api_key("openai", None).unwrap(), "sk-first");
        reset();
    }

    #[test]
    fn resolver_receives_current_place() {
        let _g = SERIAL
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        reset();
        install_key_resolvers(vec![
            Arc::new(MockResolver {
                provider: "openai".into(),
                place: Some("acme".into()),
                key: "sk-acme".into(),
            }),
            Arc::new(MockResolver {
                provider: "openai".into(),
                place: None,
                key: "sk-default".into(),
            }),
        ]);
        set_current_place(Some("acme".into()));
        assert_eq!(resolve_api_key("openai", None).unwrap(), "sk-acme");
        set_current_place(None);
        assert_eq!(resolve_api_key("openai", None).unwrap(), "sk-default");
        reset();
    }

    #[test]
    fn clear_restores_env_terminal() {
        let _g = SERIAL
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        reset();
        install_key_resolvers(vec![Arc::new(MockResolver {
            provider: "nonexistent".into(),
            place: None,
            key: "sk-resolver".into(),
        })]);
        assert_eq!(resolve_api_key("nonexistent", None).unwrap(), "sk-resolver");
        clear_key_resolvers();
        assert!(resolve_api_key("nonexistent", None).is_err());
        reset();
    }
}
