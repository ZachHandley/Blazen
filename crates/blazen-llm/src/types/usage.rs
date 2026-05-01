//! Token usage statistics and request timing metadata.

use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Token usage
// ---------------------------------------------------------------------------

/// Token usage statistics for a completion request.
#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq, Eq)]
#[cfg_attr(feature = "tsify", derive(tsify_next::Tsify))]
#[cfg_attr(feature = "tsify", tsify(into_wasm_abi, from_wasm_abi))]
pub struct TokenUsage {
    /// Number of tokens in the prompt / input.
    pub prompt_tokens: u32,
    /// Number of tokens in the completion / output.
    pub completion_tokens: u32,
    /// Total tokens consumed (prompt + completion).
    pub total_tokens: u32,
    /// Tokens spent on hidden reasoning (o-series, R1, Anthropic thinking).
    #[serde(default)]
    pub reasoning_tokens: u32,
    /// Tokens served from a prompt cache, if the provider reports them.
    #[serde(default)]
    pub cached_input_tokens: u32,
    /// Tokens consumed by audio input.
    #[serde(default)]
    pub audio_input_tokens: u32,
    /// Tokens consumed by audio output.
    #[serde(default)]
    pub audio_output_tokens: u32,
}

impl TokenUsage {
    /// Returns a `TokenUsage` with every counter set to zero.
    ///
    /// Equivalent to `TokenUsage::default()`, but kept as an explicit
    /// constructor for readability at call sites that build running tallies.
    #[must_use]
    pub fn zero() -> Self {
        Self::default()
    }

    /// Adds every field of `other` into `self` using saturating arithmetic.
    ///
    /// All seven token counters are summed: `prompt_tokens`,
    /// `completion_tokens`, `total_tokens`, `reasoning_tokens`,
    /// `cached_input_tokens`, `audio_input_tokens`, `audio_output_tokens`.
    /// Saturation prevents wraparound on degenerate inputs; overflow is not
    /// expected in practice.
    pub fn add(&mut self, other: &TokenUsage) {
        self.prompt_tokens = self.prompt_tokens.saturating_add(other.prompt_tokens);
        self.completion_tokens = self
            .completion_tokens
            .saturating_add(other.completion_tokens);
        self.total_tokens = self.total_tokens.saturating_add(other.total_tokens);
        self.reasoning_tokens = self.reasoning_tokens.saturating_add(other.reasoning_tokens);
        self.cached_input_tokens = self
            .cached_input_tokens
            .saturating_add(other.cached_input_tokens);
        self.audio_input_tokens = self
            .audio_input_tokens
            .saturating_add(other.audio_input_tokens);
        self.audio_output_tokens = self
            .audio_output_tokens
            .saturating_add(other.audio_output_tokens);
    }
}

// ---------------------------------------------------------------------------
// Request timing
// ---------------------------------------------------------------------------

/// Timing metadata for a request.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "tsify", derive(tsify_next::Tsify))]
#[cfg_attr(feature = "tsify", tsify(into_wasm_abi, from_wasm_abi))]
pub struct RequestTiming {
    /// Time spent waiting in queue (ms), if applicable.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub queue_ms: Option<u64>,
    /// Time spent executing the request (ms).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub execution_ms: Option<u64>,
    /// Total wall-clock time from submit to response (ms).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub total_ms: Option<u64>,
}

// ---------------------------------------------------------------------------
// Re-exports
// ---------------------------------------------------------------------------

pub use blazen_events::Modality;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zero_returns_default() {
        let z = TokenUsage::zero();
        assert_eq!(z.prompt_tokens, 0);
        assert_eq!(z.completion_tokens, 0);
        assert_eq!(z.total_tokens, 0);
        assert_eq!(z.reasoning_tokens, 0);
        assert_eq!(z.cached_input_tokens, 0);
        assert_eq!(z.audio_input_tokens, 0);
        assert_eq!(z.audio_output_tokens, 0);
    }

    #[test]
    fn add_sums_every_field() {
        let mut a = TokenUsage {
            prompt_tokens: 1,
            completion_tokens: 2,
            total_tokens: 3,
            reasoning_tokens: 4,
            cached_input_tokens: 5,
            audio_input_tokens: 6,
            audio_output_tokens: 7,
        };
        let b = TokenUsage {
            prompt_tokens: 10,
            completion_tokens: 20,
            total_tokens: 30,
            reasoning_tokens: 40,
            cached_input_tokens: 50,
            audio_input_tokens: 60,
            audio_output_tokens: 70,
        };
        a.add(&b);
        assert_eq!(a.prompt_tokens, 11);
        assert_eq!(a.completion_tokens, 22);
        assert_eq!(a.total_tokens, 33);
        assert_eq!(a.reasoning_tokens, 44);
        assert_eq!(a.cached_input_tokens, 55);
        assert_eq!(a.audio_input_tokens, 66);
        assert_eq!(a.audio_output_tokens, 77);
    }

    #[test]
    fn add_saturates_on_overflow() {
        let mut a = TokenUsage {
            prompt_tokens: u32::MAX - 1,
            ..Default::default()
        };
        let b = TokenUsage {
            prompt_tokens: 100,
            ..Default::default()
        };
        a.add(&b);
        assert_eq!(a.prompt_tokens, u32::MAX);
    }
}
