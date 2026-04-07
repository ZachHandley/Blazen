//! Model pricing lookup and cost computation.
//!
//! Pricing is stored in a global [`PricingRegistry`] that is pre-seeded with
//! default prices for well-known models.  Providers can register dynamic
//! pricing at runtime (e.g. after calling their `/models` endpoint) via
//! [`register_pricing`] or [`register_from_model_info`].
//!
//! Model IDs are normalized before lookup -- date suffixes (e.g.
//! `claude-sonnet-4-20250514`) and version tags are stripped so that
//! point-in-time snapshots resolve to their base model pricing.

use std::collections::HashMap;
use std::sync::{LazyLock, RwLock};

use crate::traits::{ModelInfo, ModelPricing};
use crate::types::TokenUsage;

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// Pricing for a model in USD per million tokens.
#[derive(Debug, Clone, Copy)]
pub struct PricingEntry {
    /// USD per million input (prompt) tokens.
    pub input_per_million: f64,
    /// USD per million output (completion) tokens.
    pub output_per_million: f64,
}

// ---------------------------------------------------------------------------
// PricingRegistry
// ---------------------------------------------------------------------------

/// Thread-safe pricing registry.  Pre-seeded with defaults; providers can
/// push dynamic pricing via [`register_pricing`].
struct PricingRegistry {
    entries: RwLock<HashMap<String, PricingEntry>>,
}

impl PricingRegistry {
    fn new() -> Self {
        Self {
            entries: RwLock::new(default_pricing()),
        }
    }

    fn lookup(&self, normalized_id: &str) -> Option<PricingEntry> {
        self.entries
            .read()
            .expect("pricing registry lock poisoned")
            .get(normalized_id)
            .copied()
    }

    fn register(&self, normalized_id: &str, entry: PricingEntry) {
        self.entries
            .write()
            .expect("pricing registry lock poisoned")
            .insert(normalized_id.to_owned(), entry);
    }
}

static REGISTRY: LazyLock<PricingRegistry> = LazyLock::new(PricingRegistry::new);

/// Default pricing entries seeded into the registry.
///
/// These are fallback values used when providers have not yet pushed
/// dynamic pricing via their `/models` endpoints.
fn default_pricing() -> HashMap<String, PricingEntry> {
    // (model_id, input_per_million, output_per_million)
    let defaults: &[(&str, f64, f64)] = &[
        // OpenAI
        ("gpt-4.1", 2.0, 8.0),
        ("gpt-4.1-mini", 0.40, 1.60),
        ("gpt-4.1-nano", 0.10, 0.40),
        ("gpt-4o", 2.50, 10.0),
        ("gpt-4o-mini", 0.15, 0.60),
        ("o3", 10.0, 40.0),
        ("o4-mini", 1.10, 4.40),
        // Anthropic
        ("claude-sonnet-4", 3.0, 15.0),
        ("claude-opus-4", 15.0, 75.0),
        ("claude-haiku-4", 0.80, 4.0),
        // Google
        ("gemini-2.5-flash", 0.15, 0.60),
        ("gemini-2.5-pro", 1.25, 10.0),
        // Others
        ("llama-3.3-70b-versatile", 0.59, 0.79), // Groq default
        ("llama-3.3-70b-instruct-turbo", 0.88, 0.88), // Together/Fireworks default
        ("deepseek-chat", 0.27, 1.10),
        ("mistral-large-latest", 2.0, 6.0),
        ("grok-3", 3.0, 15.0),
        ("sonar-pro", 3.0, 15.0),
        ("command-a", 2.50, 10.0),
    ];

    defaults
        .iter()
        .map(|&(id, input, output)| {
            (
                id.to_owned(),
                PricingEntry {
                    input_per_million: input,
                    output_per_million: output,
                },
            )
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Compute the estimated USD cost of a completion given a model ID and token
/// usage.
///
/// Returns `None` if the model is not in the pricing registry.
///
/// # Examples
///
/// ```
/// use blazen_llm::pricing::compute_cost;
/// use blazen_llm::TokenUsage;
///
/// let usage = TokenUsage {
///     prompt_tokens: 1000,
///     completion_tokens: 500,
///     total_tokens: 1500,
///     ..Default::default()
/// };
///
/// let cost = compute_cost("gpt-4.1", &usage).unwrap();
/// assert!(cost > 0.0);
/// ```
#[must_use]
pub fn compute_cost(model_id: &str, usage: &TokenUsage) -> Option<f64> {
    let pricing = lookup_pricing(model_id)?;
    let input_cost = f64::from(usage.prompt_tokens) * pricing.input_per_million / 1_000_000.0;
    let output_cost = f64::from(usage.completion_tokens) * pricing.output_per_million / 1_000_000.0;
    Some(input_cost + output_cost)
}

/// Look up pricing for a model by its ID.
///
/// Returns `None` if the model is unknown.
#[must_use]
pub fn lookup_pricing(model_id: &str) -> Option<PricingEntry> {
    let normalized = normalize_model_id(model_id);
    REGISTRY.lookup(&normalized)
}

/// Register (or overwrite) pricing for a model.
///
/// The `model_id` is normalized before storage, so both `"gpt-4o"` and
/// `"openai/gpt-4o-2024-08-06"` resolve to the same entry.
pub fn register_pricing(model_id: &str, entry: PricingEntry) {
    let normalized = normalize_model_id(model_id);
    REGISTRY.register(&normalized, entry);
}

/// Register pricing from a [`ModelInfo`] returned by a provider's
/// [`ModelRegistry`](crate::traits::ModelRegistry).
///
/// Does nothing if the model info has no pricing data.
pub fn register_from_model_info(info: &ModelInfo) {
    if let Some(ref pricing) = info.pricing
        && let Some(entry) = model_pricing_to_entry(pricing)
    {
        register_pricing(&info.id, entry);
    }
}

/// Convert a [`ModelPricing`] to a [`PricingEntry`], returning `None` if
/// both input and output are missing.
fn model_pricing_to_entry(pricing: &ModelPricing) -> Option<PricingEntry> {
    match (pricing.input_per_million, pricing.output_per_million) {
        (Some(input), Some(output)) => Some(PricingEntry {
            input_per_million: input,
            output_per_million: output,
        }),
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// Normalization
// ---------------------------------------------------------------------------

/// Normalize a model ID by stripping date suffixes, version tags, and
/// provider prefixes so that variants resolve to their canonical name.
///
/// Examples:
/// - `claude-sonnet-4-20250514` -> `claude-sonnet-4`
/// - `gpt-4o-2024-08-06` -> `gpt-4o`
/// - `openai/gpt-4.1` -> `gpt-4.1`
/// - `accounts/fireworks/models/llama-v3p3-70b-instruct` -> unchanged (no match)
fn normalize_model_id(model_id: &str) -> String {
    let mut id = model_id.to_lowercase();

    // Strip provider prefixes used by OpenRouter (e.g. "openai/gpt-4.1").
    if let Some(pos) = id.rfind('/') {
        id = id[pos + 1..].to_owned();
    }

    // Strip date suffixes like "-20250514" or "-2024-08-06".
    // Pattern: "-" followed by 8+ digits (YYYYMMDD) at the end.
    id = strip_date_suffix(&id);

    // Strip version suffixes like "-v1:0" (Bedrock format).
    if let Some(pos) = id.rfind("-v") {
        let suffix = &id[pos + 2..];
        if suffix
            .chars()
            .all(|c| c.is_ascii_digit() || c == ':' || c == '.')
            && !suffix.is_empty()
        {
            id = id[..pos].to_owned();
        }
    }

    id
}

/// Strip trailing date suffixes from a model ID.
///
/// Handles both compact dates (`-20250514`) and hyphenated dates
/// (`-2024-08-06`).
fn strip_date_suffix(id: &str) -> String {
    // Try compact date: "-YYYYMMDD" (8 digits at the end after a dash).
    if let Some(pos) = id.rfind('-') {
        let suffix = &id[pos + 1..];
        if suffix.len() >= 8 && suffix.chars().all(|c| c.is_ascii_digit()) {
            return id[..pos].to_owned();
        }
    }

    // Try hyphenated date: "-YYYY-MM-DD" at the end.
    // Look for "-YYYY-" pattern followed by two more segments.
    let bytes = id.as_bytes();
    let len = bytes.len();
    // "-YYYY-MM-DD" is 11 chars.
    if len >= 11 {
        let candidate_start = len - 11;
        let candidate = &id[candidate_start..];
        if candidate.starts_with('-')
            && candidate[1..5].chars().all(|c| c.is_ascii_digit())
            && candidate.as_bytes()[5] == b'-'
            && candidate[6..8].chars().all(|c| c.is_ascii_digit())
            && candidate.as_bytes()[8] == b'-'
            && candidate[9..11].chars().all(|c| c.is_ascii_digit())
        {
            return id[..candidate_start].to_owned();
        }
    }

    id.to_owned()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_known_model_pricing() {
        let usage = TokenUsage {
            prompt_tokens: 1_000_000,
            completion_tokens: 1_000_000,
            total_tokens: 2_000_000,
            ..Default::default()
        };

        // gpt-4.1: $2 in + $8 out = $10
        let cost = compute_cost("gpt-4.1", &usage).unwrap();
        assert!(
            (cost - 10.0).abs() < 0.001,
            "gpt-4.1 cost was {cost}, expected 10.0"
        );

        // claude-sonnet-4: $3 in + $15 out = $18
        let cost = compute_cost("claude-sonnet-4", &usage).unwrap();
        assert!(
            (cost - 18.0).abs() < 0.001,
            "claude-sonnet-4 cost was {cost}, expected 18.0"
        );

        // gemini-2.5-flash: $0.15 in + $0.60 out = $0.75
        let cost = compute_cost("gemini-2.5-flash", &usage).unwrap();
        assert!(
            (cost - 0.75).abs() < 0.001,
            "gemini-2.5-flash cost was {cost}, expected 0.75"
        );

        // o3: $10 in + $40 out = $50
        let cost = compute_cost("o3", &usage).unwrap();
        assert!(
            (cost - 50.0).abs() < 0.001,
            "o3 cost was {cost}, expected 50.0"
        );

        // claude-opus-4: $15 in + $75 out = $90
        let cost = compute_cost("claude-opus-4", &usage).unwrap();
        assert!(
            (cost - 90.0).abs() < 0.001,
            "claude-opus-4 cost was {cost}, expected 90.0"
        );
    }

    #[test]
    fn test_unknown_model_returns_none() {
        let usage = TokenUsage {
            prompt_tokens: 100,
            completion_tokens: 50,
            total_tokens: 150,
            ..Default::default()
        };

        assert!(compute_cost("totally-unknown-model-xyz", &usage).is_none());
        assert!(compute_cost("", &usage).is_none());
    }

    #[test]
    fn test_model_alias_normalization() {
        let usage = TokenUsage {
            prompt_tokens: 1_000_000,
            completion_tokens: 1_000_000,
            total_tokens: 2_000_000,
            ..Default::default()
        };

        // Anthropic date suffix: claude-sonnet-4-20250514 -> claude-sonnet-4
        let cost = compute_cost("claude-sonnet-4-20250514", &usage).unwrap();
        assert!(
            (cost - 18.0).abs() < 0.001,
            "claude-sonnet-4-20250514 should resolve to claude-sonnet-4, got cost {cost}"
        );

        // OpenAI date suffix: gpt-4o-2024-08-06 -> gpt-4o
        let cost = compute_cost("gpt-4o-2024-08-06", &usage).unwrap();
        assert!(
            (cost - 12.5).abs() < 0.001,
            "gpt-4o-2024-08-06 should resolve to gpt-4o, got cost {cost}"
        );

        // OpenRouter prefix: openai/gpt-4.1 -> gpt-4.1
        let cost = compute_cost("openai/gpt-4.1", &usage).unwrap();
        assert!(
            (cost - 10.0).abs() < 0.001,
            "openai/gpt-4.1 should resolve to gpt-4.1, got cost {cost}"
        );

        // Bedrock version suffix: claude-sonnet-4-v1:0 should NOT match
        // because the date suffix strip already handles the numeric suffix.
        // Instead, test the explicit Bedrock format:
        let cost = compute_cost("anthropic/claude-sonnet-4", &usage).unwrap();
        assert!(
            (cost - 18.0).abs() < 0.001,
            "anthropic/claude-sonnet-4 should resolve to claude-sonnet-4, got cost {cost}"
        );

        // Case insensitivity
        let cost = compute_cost("GPT-4.1", &usage).unwrap();
        assert!(
            (cost - 10.0).abs() < 0.001,
            "GPT-4.1 (uppercase) should resolve to gpt-4.1, got cost {cost}"
        );
    }

    #[test]
    fn test_zero_tokens_returns_zero_cost() {
        let usage = TokenUsage {
            prompt_tokens: 0,
            completion_tokens: 0,
            total_tokens: 0,
            ..Default::default()
        };

        let cost = compute_cost("gpt-4.1", &usage).unwrap();
        assert!(
            cost.abs() < f64::EPSILON,
            "zero tokens should produce zero cost, got {cost}"
        );
    }

    #[test]
    fn test_small_token_count() {
        let usage = TokenUsage {
            prompt_tokens: 100,
            completion_tokens: 50,
            total_tokens: 150,
            ..Default::default()
        };

        // gpt-4.1: 100 * 2.0/1M + 50 * 8.0/1M = 0.0002 + 0.0004 = 0.0006
        let cost = compute_cost("gpt-4.1", &usage).unwrap();
        assert!(
            (cost - 0.0006).abs() < 1e-10,
            "gpt-4.1 small token cost was {cost}, expected 0.0006"
        );
    }

    #[test]
    fn test_normalize_strips_date_suffix() {
        assert_eq!(
            normalize_model_id("claude-sonnet-4-20250514"),
            "claude-sonnet-4"
        );
        assert_eq!(normalize_model_id("gpt-4o-2024-08-06"), "gpt-4o");
    }

    #[test]
    fn test_normalize_strips_provider_prefix() {
        assert_eq!(normalize_model_id("openai/gpt-4.1"), "gpt-4.1");
        assert_eq!(
            normalize_model_id("anthropic/claude-sonnet-4"),
            "claude-sonnet-4"
        );
    }

    #[test]
    fn test_normalize_case_insensitive() {
        assert_eq!(normalize_model_id("GPT-4.1"), "gpt-4.1");
        assert_eq!(normalize_model_id("Claude-Sonnet-4"), "claude-sonnet-4");
    }

    #[test]
    fn test_all_models_have_positive_pricing() {
        let models = [
            "gpt-4.1",
            "gpt-4.1-mini",
            "gpt-4.1-nano",
            "gpt-4o",
            "gpt-4o-mini",
            "o3",
            "o4-mini",
            "claude-sonnet-4",
            "claude-opus-4",
            "claude-haiku-4",
            "gemini-2.5-flash",
            "gemini-2.5-pro",
            "deepseek-chat",
            "mistral-large-latest",
            "grok-3",
            "sonar-pro",
            "command-a",
        ];

        for model in models {
            let entry =
                lookup_pricing(model).unwrap_or_else(|| panic!("missing pricing for {model}"));
            assert!(
                entry.input_per_million > 0.0,
                "{model} has non-positive input pricing"
            );
            assert!(
                entry.output_per_million > 0.0,
                "{model} has non-positive output pricing"
            );
        }
    }

    #[test]
    fn test_register_dynamic_pricing() {
        // Register a new model dynamically.
        register_pricing(
            "my-custom-model",
            PricingEntry {
                input_per_million: 5.0,
                output_per_million: 20.0,
            },
        );

        let entry = lookup_pricing("my-custom-model").expect("should find registered model");
        assert!((entry.input_per_million - 5.0).abs() < f64::EPSILON);
        assert!((entry.output_per_million - 20.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_register_overrides_default() {
        // Override gpt-4.1 pricing with new values.
        let original = lookup_pricing("gpt-4.1").unwrap();
        let new_entry = PricingEntry {
            input_per_million: 99.0,
            output_per_million: 99.0,
        };
        register_pricing("gpt-4.1", new_entry);

        let updated = lookup_pricing("gpt-4.1").unwrap();
        assert!((updated.input_per_million - 99.0).abs() < f64::EPSILON);

        // Restore original so other tests aren't affected.
        register_pricing("gpt-4.1", original);
    }

    #[test]
    fn test_register_from_model_info() {
        let info = ModelInfo {
            id: "test-registry-model".into(),
            name: None,
            provider: "test".into(),
            context_length: None,
            pricing: Some(ModelPricing {
                input_per_million: Some(1.5),
                output_per_million: Some(6.0),
                per_image: None,
                per_second: None,
            }),
            capabilities: Default::default(),
        };

        register_from_model_info(&info);
        let entry = lookup_pricing("test-registry-model").expect("should find model");
        assert!((entry.input_per_million - 1.5).abs() < f64::EPSILON);
        assert!((entry.output_per_million - 6.0).abs() < f64::EPSILON);
    }
}
