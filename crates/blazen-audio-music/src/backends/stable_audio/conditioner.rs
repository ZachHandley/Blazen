//! Text + numeric conditioner for the Stable Audio Open Small DiT.
//!
//! Two pieces of conditioning feed into the diffusion transformer:
//!
//! 1. **Text** — the prompt is tokenized with the T5-base SentencePiece
//!    tokenizer, run through the FP32 T5 encoder (109 M params) shipped
//!    with the open release, and the resulting `(B, T_text, 768)` hidden
//!    states are consumed by the DiT cross-attention layers. An attention
//!    mask is returned alongside so the DiT can ignore padded positions.
//! 2. **Numeric** — two scalars (`seconds_start`, `seconds_total`) describe
//!    where in the original clip this generation sits and how long the
//!    full clip is (typically `0.0` and `10.5` for the Small variant).
//!    These get Fourier-feature embedded inside the DiT for AdaLN
//!    modulation, so this module only emits the raw `(B,)` f32 tensors.
//!
//! Reference implementations: HuggingFace `diffusers`
//! `StableAudioProjectionModel` and Stability AI
//! `stable-audio-tools/stable_audio_tools/models/conditioners.py`.

use std::path::Path;

use candle_core::{DType, Device, Result, Tensor};
use candle_nn::{Linear, VarBuilder, linear};
use candle_transformers::models::t5::{Config as T5Config, T5EncoderModel};
use tokenizers::{PaddingDirection, PaddingParams, PaddingStrategy, Tokenizer, TruncationParams};

/// Default text-token budget for the Stable Audio Open Small variant.
pub const SMALL_MAX_TEXT_TOKENS: usize = 64;

/// Text + numeric conditioner for the Stable Audio Small DiT.
pub struct Conditioner {
    t5_encoder: T5EncoderModel,
    t5_tokenizer: Tokenizer,
    /// Optional projection from T5 hidden_dim → DiT cross-attn dim.
    /// `None` on the Small variant (both dims are 768 — identity); kept
    /// here so the 1.0 variant can plug a learned projection in later
    /// without changing the public surface.
    t5_proj: Option<Linear>,
    max_text_tokens: usize,
    /// Pad token id pinned at construction. Exposed via [`Self::pad_token_id`]
    /// for callers that build unconditional prompts; kept as a field even
    /// though the encode path uses the tokenizer's own pad-id table.
    #[allow(
        dead_code,
        reason = "Surfaced through pad_token_id() for the unconditional-prompt \
                  / CFG branches added in later waves; the encode-text path \
                  reads the id from the tokenizer config directly."
    )]
    pad_token_id: u32,
    device: Device,
}

impl Conditioner {
    /// Build a conditioner from a `VarBuilder` rooted at the T5 encoder
    /// weights and a path to the SentencePiece `tokenizer.json` shipped
    /// with the model repo.
    ///
    /// # Errors
    ///
    /// Returns a candle error if the T5 weights can't be loaded or the
    /// tokenizer file can't be parsed.
    pub fn new(
        vb: &VarBuilder,
        t5_config: &T5Config,
        tokenizer_path: &Path,
        max_text_tokens: usize,
    ) -> Result<Self> {
        let pad_token_id =
            u32::try_from(t5_config.pad_token_id).map_err(candle_core::Error::wrap)?;
        let t5_encoder = T5EncoderModel::load(vb.pp("text_encoder"), t5_config)?;
        let mut t5_tokenizer = Tokenizer::from_file(tokenizer_path)
            .map_err(|e| candle_core::Error::Msg(format!("tokenizer load failed: {e}")))?;

        // Configure padding + truncation to `max_text_tokens` so a single
        // prompt and a batch share the same shape contract.
        t5_tokenizer
            .with_padding(Some(PaddingParams {
                strategy: PaddingStrategy::Fixed(max_text_tokens),
                direction: PaddingDirection::Right,
                pad_to_multiple_of: None,
                pad_id: pad_token_id,
                pad_type_id: 0,
                pad_token: "<pad>".to_string(),
            }))
            .with_truncation(Some(TruncationParams {
                max_length: max_text_tokens,
                ..TruncationParams::default()
            }))
            .map_err(|e| candle_core::Error::Msg(format!("tokenizer config failed: {e}")))?;

        // For the Small variant T5 hidden_dim == DiT cross-attn dim (768),
        // so the projection is identity. If the VarBuilder happens to
        // expose a `t5_proj` weight (the 1.0 variant does in the official
        // checkpoint), load it; otherwise leave the projection as `None`.
        let t5_proj = if vb.contains_tensor("t5_proj.weight") {
            Some(linear(
                t5_config.d_model,
                t5_config.d_model,
                vb.pp("t5_proj"),
            )?)
        } else {
            None
        };

        Ok(Self {
            t5_encoder,
            t5_tokenizer,
            t5_proj,
            max_text_tokens,
            pad_token_id,
            device: vb.device().clone(),
        })
    }

    /// Encode a single prompt.
    ///
    /// Returns `(hidden_states, attention_mask)` of shape
    /// `(1, max_text_tokens, d_model)` and `(1, max_text_tokens)`.
    ///
    /// # Errors
    ///
    /// Returns a candle error if tokenization fails or the T5 forward
    /// pass fails.
    pub fn encode_text(&mut self, prompt: &str) -> Result<(Tensor, Tensor)> {
        self.encode_batch(&[prompt])
    }

    /// Encode a batch of prompts.
    ///
    /// Returns `(hidden_states, attention_mask)` of shape
    /// `(B, max_text_tokens, d_model)` and `(B, max_text_tokens)`.
    ///
    /// # Errors
    ///
    /// Returns a candle error if tokenization fails or the T5 forward
    /// pass fails.
    pub fn encode_batch(&mut self, prompts: &[&str]) -> Result<(Tensor, Tensor)> {
        if prompts.is_empty() {
            return Err(candle_core::Error::Msg(
                "encode_batch called with empty prompt slice".to_string(),
            ));
        }

        // `encode_batch` accepts `Vec<EncodeInput>`; for plain strings the
        // `&str` → `EncodeInput` conversion is automatic via `Into`.
        let inputs: Vec<tokenizers::EncodeInput> = prompts
            .iter()
            .map(|p| tokenizers::EncodeInput::Single((*p).into()))
            .collect();
        let encodings = self
            .t5_tokenizer
            .encode_batch(inputs, true)
            .map_err(|e| candle_core::Error::Msg(format!("tokenizer encode failed: {e}")))?;

        let batch = encodings.len();
        let seq = self.max_text_tokens;

        // Flatten ids + attention_mask into contiguous Vec<u32>s for
        // tensor construction. With Fixed-strategy padding configured
        // above every encoding is exactly `seq` long.
        let mut ids: Vec<u32> = Vec::with_capacity(batch * seq);
        let mut mask: Vec<u32> = Vec::with_capacity(batch * seq);
        for enc in &encodings {
            debug_assert_eq!(enc.get_ids().len(), seq, "tokenizer padding misconfigured");
            ids.extend_from_slice(enc.get_ids());
            mask.extend_from_slice(enc.get_attention_mask());
        }

        let input_ids = Tensor::from_vec(ids, (batch, seq), &self.device)?;
        let attention_mask =
            Tensor::from_vec(mask, (batch, seq), &self.device)?.to_dtype(DType::F32)?;

        // T5 caches relative position bias across calls; reset before
        // each forward to keep the encoder stateless from the caller's
        // perspective.
        self.t5_encoder.clear_kv_cache();
        let mut hidden = self.t5_encoder.forward(&input_ids)?;
        if let Some(proj) = self.t5_proj.as_ref() {
            hidden = hidden.apply(proj)?;
        }
        Ok((hidden, attention_mask))
    }

    /// Read-only view of the configured pad token id (handy for tests
    /// and for callers that need to construct an unconditional prompt).
    #[must_use]
    #[allow(
        dead_code,
        reason = "Placeholder accessor for the unconditional-prompt / CFG \
                  branches added in later waves."
    )]
    pub const fn pad_token_id(&self) -> u32 {
        self.pad_token_id
    }

    /// Configured text-token budget.
    #[must_use]
    #[allow(
        dead_code,
        reason = "Surfaced for diagnostics + future CFG support; the \
                  encode path uses the tokenizer's configured padding \
                  strategy directly."
    )]
    pub const fn max_text_tokens(&self) -> usize {
        self.max_text_tokens
    }
}

/// Helper that produces the raw numeric scalars the DiT consumes.
///
/// The actual Fourier-feature embedding + projection happens inside
/// `dit::DiT::forward` (because the embedding weights live with the DiT,
/// not the conditioner), so this just broadcasts the two scalars to
/// `(B,)` f32 tensors on the target device.
///
/// # Errors
///
/// Returns a candle error if tensor allocation fails on the target
/// device (typically only happens on GPU OOM).
pub fn build_numeric_conds(
    seconds_start: f32,
    seconds_total: f32,
    batch_size: usize,
    device: &Device,
) -> Result<(Tensor, Tensor)> {
    if batch_size == 0 {
        return Err(candle_core::Error::Msg(
            "build_numeric_conds called with batch_size=0".to_string(),
        ));
    }
    let start = Tensor::from_vec(vec![seconds_start; batch_size], batch_size, device)?;
    let total = Tensor::from_vec(vec![seconds_total; batch_size], batch_size, device)?;
    Ok((start, total))
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    /// `build_numeric_conds` smoke test — pure tensor allocation, no
    /// external dependencies.
    #[test]
    fn build_numeric_conds_smoke() {
        let device = Device::Cpu;
        let (start, total) = build_numeric_conds(0.0, 10.5, 2, &device).unwrap();
        assert_eq!(start.dims(), &[2]);
        assert_eq!(total.dims(), &[2]);
        assert_eq!(start.dtype(), DType::F32);
        assert_eq!(total.dtype(), DType::F32);

        let start_vec: Vec<f32> = start.to_vec1().unwrap();
        let total_vec: Vec<f32> = total.to_vec1().unwrap();
        assert_eq!(start_vec, vec![0.0, 0.0]);
        assert_eq!(total_vec, vec![10.5, 10.5]);
    }

    #[test]
    fn build_numeric_conds_rejects_empty_batch() {
        let err = build_numeric_conds(0.0, 10.5, 0, &Device::Cpu).unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("batch_size=0"), "got: {msg}");
    }

    /// End-to-end shape smoke test for the text encoder. Requires a real
    /// T5 `tokenizer.json` + matching safetensors on disk; gated behind
    /// `BLAZEN_T5_BASE_DIR` env-var so it stays hermetic in CI. Set the
    /// var to a directory containing `tokenizer.json` and `model.safetensors`
    /// for a T5-base checkpoint to enable.
    #[test]
    #[ignore = "needs T5-base weights + tokenizer; set BLAZEN_T5_BASE_DIR to enable"]
    fn conditioner_dimensions_smoke() {
        let Ok(dir) = std::env::var("BLAZEN_T5_BASE_DIR") else {
            return;
        };
        let dir = std::path::PathBuf::from(dir);
        let weights = dir.join("model.safetensors");
        let tokenizer = dir.join("tokenizer.json");

        let cfg = T5Config {
            vocab_size: 32128,
            d_model: 768,
            d_kv: 64,
            d_ff: 3072,
            num_layers: 12,
            num_heads: 12,
            pad_token_id: 0,
            eos_token_id: 1,
            ..T5Config::default()
        };

        // SAFETY: mmap requires the file to outlive the mapping and not
        // be mutated; the test-controlled directory satisfies both.
        #[allow(unsafe_code)]
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[&weights], DType::F32, &Device::Cpu).unwrap()
        };

        let mut cond = Conditioner::new(&vb, &cfg, &tokenizer, SMALL_MAX_TEXT_TOKENS).unwrap();
        let (hidden, mask) = cond.encode_text("hello").unwrap();
        assert_eq!(hidden.dims(), &[1, SMALL_MAX_TEXT_TOKENS, 768]);
        assert_eq!(mask.dims(), &[1, SMALL_MAX_TEXT_TOKENS]);
    }
}
