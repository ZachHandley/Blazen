//! API key resolution for LLM providers.
//!
//! Maps provider names to well-known environment variables and provides
//! a single [`resolve_api_key`] entry-point that checks an explicit key
//! first, then falls back to the corresponding env var.

use crate::BlazenError;

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
/// 1. `explicit` — if the caller passed a key directly, use it.
/// 2. Environment variable — looked up via [`PROVIDER_ENV_VARS`].
/// 3. Error — returns [`BlazenError::Auth`] with a helpful message.
///
/// # Errors
///
/// Returns [`BlazenError::Auth`] if no explicit key is provided and the
/// environment variable for the provider is either unset or empty.
pub fn resolve_api_key(provider: &str, explicit: Option<String>) -> Result<String, BlazenError> {
    if let Some(key) = explicit.filter(|k| !k.is_empty()) {
        return Ok(key);
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
}
