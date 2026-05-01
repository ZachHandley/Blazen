//! Token counting for LLM messages.
//!
//! Provides a [`TokenCounter`] trait with two implementations:
//!
//! - **[`TiktokenCounter`]** (feature = `"tiktoken"`) -- exact BPE-based counting
//!   via `tiktoken-rs`, matching `OpenAI`'s tokeniser for GPT-3.5 / GPT-4 / GPT-4.1 /
//!   o-series models.
//! - **[`EstimateCounter`]** -- a lightweight heuristic that needs no external data
//!   files and works everywhere (including WASM). Good enough for budget checks
//!   when exact counts aren't critical.

#[cfg(feature = "tiktoken")]
use crate::error::BlazenError;
use crate::types::ChatMessage;
#[cfg(feature = "tiktoken")]
use crate::types::Role;

// ---------------------------------------------------------------------------
// Trait
// ---------------------------------------------------------------------------

/// Trait for counting tokens in messages.
///
/// All implementations must be `Send + Sync` so they can be shared across
/// async tasks.
pub trait TokenCounter: Send + Sync {
    /// Count tokens in a raw text string.
    fn count_tokens(&self, text: &str) -> usize;

    /// Count tokens for a chat message array, including per-message overhead
    /// tokens that `OpenAI` charges (role markers, separators, etc.).
    fn count_message_tokens(&self, messages: &[ChatMessage]) -> usize;

    /// Get the context window size (maximum tokens) for the model this counter
    /// was built for.
    fn context_size(&self) -> usize;

    /// Calculate the number of tokens remaining after the given prompt.
    ///
    /// Returns `0` if the prompt already exceeds the context window.
    fn remaining_tokens(&self, messages: &[ChatMessage]) -> usize {
        let used = self.count_message_tokens(messages);
        self.context_size().saturating_sub(used)
    }
}

// ---------------------------------------------------------------------------
// Context window lookup
// ---------------------------------------------------------------------------

/// Best-effort context window size for a model identifier.
///
/// Falls back to 128 000 when the model string doesn't match any known
/// pattern -- a reasonable default for most modern LLMs.
#[must_use]
pub fn get_context_window(model: &str) -> usize {
    match model {
        m if m.contains("gpt-4.1") => 1_048_576, // 1M
        m if m.contains("gpt-4o") => 128_000,
        m if m.contains("o3") || m.contains("o4") => 200_000,
        m if m.contains("claude") => 200_000,
        m if m.contains("gemini-2.5") => 1_048_576,
        m if m.contains("gemini-2.0") => 1_048_576,
        _ => 128_000, // reasonable default
    }
}

// ---------------------------------------------------------------------------
// Helper: Role -> string
// ---------------------------------------------------------------------------

#[cfg(feature = "tiktoken")]
fn role_str(role: &Role) -> &'static str {
    match role {
        Role::System => "system",
        Role::User => "user",
        Role::Assistant => "assistant",
        Role::Tool => "tool",
    }
}

// ---------------------------------------------------------------------------
// TiktokenCounter (feature-gated)
// ---------------------------------------------------------------------------

/// Exact BPE token counter backed by [`tiktoken_rs`].
///
/// Mirrors the per-message overhead rules documented in
/// <https://platform.openai.com/docs/guides/text-generation>.
#[cfg(feature = "tiktoken")]
pub struct TiktokenCounter {
    bpe: tiktoken_rs::CoreBPE,
    /// Extra tokens added for each message (role, separators).
    tokens_per_message: i32,
    /// Extra tokens when a `name` field is present (`1` for GPT-4+, `-1` for
    /// GPT-3.5).  Reserved for when `ChatMessage` gains a `name` field.
    #[allow(dead_code)]
    tokens_per_name: i32,
    /// Maximum tokens the model supports.
    context_window: usize,
}

#[cfg(feature = "tiktoken")]
impl TiktokenCounter {
    /// Create a counter tuned for a specific model.
    ///
    /// # Errors
    ///
    /// Returns [`BlazenError::Unsupported`] when `tiktoken-rs` does not
    /// recognise the model name.
    pub fn for_model(model: &str) -> Result<Self, BlazenError> {
        let bpe = tiktoken_rs::get_bpe_from_model(model).map_err(|e| {
            BlazenError::unsupported(format!("Unknown model for token counting: {model}: {e}"))
        })?;

        let (tokens_per_message, tokens_per_name) = if model.starts_with("gpt-3.5") {
            (4, -1)
        } else {
            (3, 1) // GPT-4, GPT-4.1, o3, o4, etc.
        };

        let context_window = get_context_window(model);

        Ok(Self {
            bpe,
            tokens_per_message,
            tokens_per_name,
            context_window,
        })
    }
}

#[cfg(feature = "tiktoken")]
impl TokenCounter for TiktokenCounter {
    fn count_tokens(&self, text: &str) -> usize {
        self.bpe.encode_with_special_tokens(text).len()
    }

    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_possible_wrap,
        clippy::cast_sign_loss
    )]
    fn count_message_tokens(&self, messages: &[ChatMessage]) -> usize {
        let mut total: i32 = 0;

        for msg in messages {
            total += self.tokens_per_message;

            // Role token.
            total += self
                .bpe
                .encode_with_special_tokens(role_str(&msg.role))
                .len() as i32;

            // Text content.
            if let Some(text) = msg.content.text_content() {
                total += self.bpe.encode_with_special_tokens(&text).len() as i32;
            }

            // Tool-call ID (present on Role::Tool result messages).
            if let Some(ref id) = msg.tool_call_id {
                total += self.bpe.encode_with_special_tokens(id).len() as i32;
            }

            // Tool calls requested by the assistant.
            for tc in &msg.tool_calls {
                total += self.bpe.encode_with_special_tokens(&tc.name).len() as i32;
                total += self
                    .bpe
                    .encode_with_special_tokens(&tc.arguments.to_string())
                    .len() as i32;
            }
        }

        // Every reply is primed with <|start|>assistant<|message|> (3 tokens).
        total += 3;

        total.max(0) as usize
    }

    fn context_size(&self) -> usize {
        self.context_window
    }
}

// ---------------------------------------------------------------------------
// EstimateCounter (always available)
// ---------------------------------------------------------------------------

/// Heuristic token counter that uses a characters-per-token ratio.
///
/// Default ratio is **3.5 characters per token**, which is a reasonable
/// approximation for English text tokenised with BPE.  This counter requires
/// no external data files and compiles to WASM without issues.
pub struct EstimateCounter {
    chars_per_token: f64,
    context_window: usize,
}

impl EstimateCounter {
    /// Create an estimate counter with the default 3.5 chars/token ratio.
    #[must_use]
    pub fn new(context_window: usize) -> Self {
        Self {
            chars_per_token: 3.5,
            context_window,
        }
    }

    /// Create an estimate counter with a custom chars-per-token ratio.
    #[must_use]
    pub fn with_ratio(context_window: usize, chars_per_token: f64) -> Self {
        Self {
            chars_per_token,
            context_window,
        }
    }
}

impl TokenCounter for EstimateCounter {
    #[allow(
        clippy::cast_precision_loss,
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss
    )]
    fn count_tokens(&self, text: &str) -> usize {
        (text.len() as f64 / self.chars_per_token).ceil() as usize
    }

    fn count_message_tokens(&self, messages: &[ChatMessage]) -> usize {
        let mut total = 0usize;
        for msg in messages {
            total += 4; // per-message overhead estimate
            if let Some(text) = msg.content.text_content() {
                total += self.count_tokens(&text);
            }
        }
        total + 3 // assistant priming
    }

    fn context_size(&self) -> usize {
        self.context_window
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::ChatMessage;

    // -- EstimateCounter tests (always run) ---------------------------------

    #[test]
    fn test_estimate_counter() {
        let counter = EstimateCounter::new(128_000);
        // "Hello, world!" is 13 chars. At 3.5 chars/token => ceil(13/3.5) = ceil(3.71) = 4
        let count = counter.count_tokens("Hello, world!");
        assert_eq!(count, 4, "expected 4 estimated tokens for 'Hello, world!'");

        // Empty string should be 0 tokens.
        assert_eq!(counter.count_tokens(""), 0);

        // Context size.
        assert_eq!(counter.context_size(), 128_000);
    }

    #[test]
    fn test_estimate_message_count() {
        let counter = EstimateCounter::new(128_000);
        let messages = vec![
            ChatMessage::system("You are a helpful assistant."),
            ChatMessage::user("Hello!"),
        ];
        let count = counter.count_message_tokens(&messages);

        // Each message: 4 overhead + text tokens.
        // "You are a helpful assistant." = 28 chars => ceil(28/3.5) = 8
        // "Hello!" = 6 chars => ceil(6/3.5) = ceil(1.71) = 2
        // Total: (4+8) + (4+2) + 3 priming = 21
        assert_eq!(count, 21);

        // remaining_tokens should work.
        let remaining = counter.remaining_tokens(&messages);
        assert_eq!(remaining, 128_000 - 21);
    }

    // -- TiktokenCounter tests (feature-gated) ------------------------------

    #[cfg(feature = "tiktoken")]
    mod tiktoken_tests {
        use super::*;

        #[test]
        fn test_tiktoken_basic_count() {
            let counter = TiktokenCounter::for_model("gpt-4.1").expect("model should be known");
            let count = counter.count_tokens("Hello, world!");
            // tiktoken should produce a small, nonzero number of tokens for this string.
            assert!(
                count > 0 && count < 20,
                "unexpected token count {count} for 'Hello, world!'"
            );
        }

        #[test]
        fn test_tiktoken_message_overhead() {
            let counter = TiktokenCounter::for_model("gpt-4.1").expect("model should be known");
            let text = "Hello";
            let raw_tokens = counter.count_tokens(text);

            let messages = vec![ChatMessage::user(text)];
            let message_tokens = counter.count_message_tokens(&messages);

            // message_tokens should be strictly greater than raw_tokens because
            // of per-message overhead (tokens_per_message) + role token +
            // assistant priming (3).
            assert!(
                message_tokens > raw_tokens,
                "message tokens ({message_tokens}) should exceed raw tokens ({raw_tokens})"
            );

            // The overhead for a single user message with GPT-4 rules is:
            //   tokens_per_message(3) + role("user" = 1 token) + 3 priming = 7
            let overhead = message_tokens - raw_tokens;
            assert!(
                (4..=10).contains(&overhead),
                "unexpected overhead {overhead}"
            );
        }

        #[test]
        fn test_tiktoken_context_window() {
            // GPT-4.1 should be 1M.
            let counter = TiktokenCounter::for_model("gpt-4.1").expect("model should be known");
            assert_eq!(counter.context_size(), 1_048_576);

            // GPT-4o should be 128K.
            let counter = TiktokenCounter::for_model("gpt-4o").expect("model should be known");
            assert_eq!(counter.context_size(), 128_000);
        }

        #[test]
        fn test_tiktoken_remaining_tokens() {
            let counter = TiktokenCounter::for_model("gpt-4o").expect("model should be known");
            let messages = vec![
                ChatMessage::system("You are a helpful assistant."),
                ChatMessage::user("What is 2+2?"),
            ];
            let used = counter.count_message_tokens(&messages);
            let remaining = counter.remaining_tokens(&messages);
            assert_eq!(remaining, 128_000 - used);
            assert!(
                remaining > 0,
                "remaining should be positive for short prompt"
            );
        }
    }

    // -- Context window lookup tests ----------------------------------------

    #[test]
    fn test_context_window_known_models() {
        assert_eq!(get_context_window("gpt-4.1"), 1_048_576);
        assert_eq!(get_context_window("gpt-4o-mini"), 128_000);
        assert_eq!(get_context_window("o3-mini"), 200_000);
        assert_eq!(get_context_window("o4-mini"), 200_000);
        assert_eq!(get_context_window("claude-3.5-sonnet"), 200_000);
        assert_eq!(get_context_window("gemini-2.5-pro"), 1_048_576);
        assert_eq!(get_context_window("gemini-2.0-flash"), 1_048_576);
    }

    #[test]
    fn test_context_window_unknown_model() {
        assert_eq!(get_context_window("some-unknown-model"), 128_000);
    }
}
