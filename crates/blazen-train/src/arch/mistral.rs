//! Trainable Mistral model wrapper for LoRA fine-tuning.
//!
//! Wave 2C of PR7 mirrors `candle_transformers::models::mistral::Model`'s
//! forward pass but substitutes [`crate::lora::LoraLinear`] at every
//! attention projection (q/k/v/o) and, when requested, the MLP
//! projections (gate/up/down) listed in
//! [`crate::config::LoraConfig::target_modules`].
//!
//! The trained adapter round-trips through
//! `blazen_llm_candle::lora::LoadedAdapter` because the var names this
//! wrapper writes match the PEFT-canonical layout:
//! `base_model.model.model.layers.N.self_attn.{q,k,v,o}_proj.{lora_A,lora_B}.weight`.
//!
//! Mistral-specific notes:
//! - **Sliding-window attention.** `Config::sliding_window` (typically
//!   4096) caps each token's attention span; the prepared mask sets
//!   `-inf` whenever `j + sliding_window < i`, matching upstream.
//! - **GQA.** `num_key_value_heads < num_attention_heads`; we repeat KV
//!   via [`candle_transformers::utils::repeat_kv`].
//! - **No biases.** Attention and MLP projections are bias-free.
//! - **RoPE.** Standard rotary, `theta = cfg.rope_theta`.
//! - **Tied embeddings.** Mistral does NOT tie embeddings; `lm_head` is a
//!   separate frozen `Linear` loaded from `lm_head.weight`.

// Why: candle's idiomatic VarBuilder ergonomics take the builder by value
// (the typical caller passes `vb.pp("...")` which is already a fresh
// scoped clone). LoraLinear::wrap uses the same allow for the same reason.
#![allow(clippy::needless_pass_by_value)]

pub use candle_transformers::models::mistral::Config;

use std::collections::HashSet;
use std::sync::Arc;

use candle_core::{D, DType, Device, Module, Result, Tensor};
use candle_nn::{Embedding, Linear, RmsNorm, VarBuilder, VarMap};
use candle_transformers::utils::repeat_kv;

use crate::arch::{BaseLoader, TrainMode, build_embedding, build_rms_norm};
use crate::config::LoraConfig;
use crate::lora::LoraLinear;

/// Frozen or LoRA-augmented linear, switched at load time based on
/// whether the projection's suffix matches `LoraConfig::target_modules`.
enum MaybeLora {
    Frozen(Linear),
    Lora(LoraLinear),
}

impl Module for MaybeLora {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        match self {
            Self::Frozen(l) => l.forward(xs),
            Self::Lora(l) => l.forward(xs),
        }
    }
}

// Why: every argument is load-bearing — module name + LoRA target set,
// LoRA cfg (rank/alpha), two VarBuilders (frozen vs trainable scopes),
// mode-aware loader, and the absolute key prefix for FFT varmap
// insertion. Mirrors `maybe_lora_linear` in `llama.rs`.
#[allow(clippy::too_many_arguments)]
fn maybe_lora(
    in_dim: usize,
    out_dim: usize,
    suffix: &str,
    targets: &HashSet<String>,
    base_vb: &VarBuilder,
    lora_vb: &VarBuilder,
    lora_cfg: &LoraConfig,
    loader: &BaseLoader<'_>,
    abs_path: &str,
) -> Result<MaybeLora> {
    let base = loader.linear_no_bias(
        in_dim,
        out_dim,
        base_vb.pp(suffix),
        &format!("{abs_path}.{suffix}"),
    )?;
    if targets.contains(suffix) {
        let wrapped = LoraLinear::wrap(
            base,
            in_dim,
            out_dim,
            lora_cfg.rank,
            lora_cfg.alpha,
            lora_vb.pp(suffix),
        )?;
        Ok(MaybeLora::Lora(wrapped))
    } else {
        Ok(MaybeLora::Frozen(base))
    }
}

struct RotaryEmbedding {
    sin: Tensor,
    cos: Tensor,
}

impl RotaryEmbedding {
    fn new(dtype: DType, cfg: &Config, dev: &Device) -> Result<Self> {
        #[allow(clippy::cast_possible_truncation)]
        let rope_theta = cfg.rope_theta as f32;
        let dim = head_dim(cfg);
        let max_seq_len = cfg.max_position_embeddings;
        #[allow(clippy::cast_precision_loss)]
        let inv_freq: Vec<f32> = (0..dim)
            .step_by(2)
            .map(|i| 1f32 / rope_theta.powf(i as f32 / dim as f32))
            .collect();
        let inv_freq_len = inv_freq.len();
        let inv_freq = Tensor::from_vec(inv_freq, (1, inv_freq_len), dev)?.to_dtype(DType::F32)?;
        #[allow(clippy::cast_possible_truncation)]
        let t = Tensor::arange(0u32, max_seq_len as u32, dev)?
            .to_dtype(DType::F32)?
            .reshape((max_seq_len, 1))?;
        let freqs = t.matmul(&inv_freq)?;
        Ok(Self {
            sin: freqs.sin()?.to_dtype(dtype)?,
            cos: freqs.cos()?.to_dtype(dtype)?,
        })
    }

    fn apply(&self, q: &Tensor, k: &Tensor) -> Result<(Tensor, Tensor)> {
        let (_b, _h, seq_len, _d) = q.dims4()?;
        let cos = self.cos.narrow(0, 0, seq_len)?;
        let sin = self.sin.narrow(0, 0, seq_len)?;
        // Why: `candle_nn::rotary_emb::rope` is a no-bwd CustomOp3 and severs
        // the autograd graph at attention. `rope_slow` is the primitive-op
        // equivalent that backpropagates.
        let q_embed = candle_nn::rotary_emb::rope_slow(q, &cos, &sin)?;
        let k_embed = candle_nn::rotary_emb::rope_slow(k, &cos, &sin)?;
        Ok((q_embed, k_embed))
    }
}

fn head_dim(cfg: &Config) -> usize {
    cfg.head_dim
        .unwrap_or(cfg.hidden_size / cfg.num_attention_heads)
}

struct Attention {
    q_proj: MaybeLora,
    k_proj: MaybeLora,
    v_proj: MaybeLora,
    o_proj: MaybeLora,
    num_heads: usize,
    num_kv_heads: usize,
    num_kv_groups: usize,
    head_dim: usize,
    rotary: Arc<RotaryEmbedding>,
}

impl Attention {
    #[allow(clippy::needless_pass_by_value, clippy::too_many_arguments)]
    fn load(
        rotary: Arc<RotaryEmbedding>,
        cfg: &Config,
        base_vb: VarBuilder,
        lora_vb: VarBuilder,
        lora_cfg: &LoraConfig,
        targets: &HashSet<String>,
        loader: &BaseLoader<'_>,
        abs_path: &str,
    ) -> Result<Self> {
        let hidden = cfg.hidden_size;
        let num_heads = cfg.num_attention_heads;
        let num_kv_heads = cfg.num_key_value_heads;
        let num_kv_groups = num_heads / num_kv_heads;
        let d_head = head_dim(cfg);
        let q_proj = maybe_lora(
            hidden,
            num_heads * d_head,
            "q_proj",
            targets,
            &base_vb,
            &lora_vb,
            lora_cfg,
            loader,
            abs_path,
        )?;
        let k_proj = maybe_lora(
            hidden,
            num_kv_heads * d_head,
            "k_proj",
            targets,
            &base_vb,
            &lora_vb,
            lora_cfg,
            loader,
            abs_path,
        )?;
        let v_proj = maybe_lora(
            hidden,
            num_kv_heads * d_head,
            "v_proj",
            targets,
            &base_vb,
            &lora_vb,
            lora_cfg,
            loader,
            abs_path,
        )?;
        let o_proj = maybe_lora(
            num_heads * d_head,
            hidden,
            "o_proj",
            targets,
            &base_vb,
            &lora_vb,
            lora_cfg,
            loader,
            abs_path,
        )?;
        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            num_heads,
            num_kv_heads,
            num_kv_groups,
            head_dim: d_head,
            rotary,
        })
    }

    fn forward(&self, xs: &Tensor, attention_mask: Option<&Tensor>) -> Result<Tensor> {
        let (b_sz, q_len, _) = xs.dims3()?;

        let q = self.q_proj.forward(xs)?;
        let k = self.k_proj.forward(xs)?;
        let v = self.v_proj.forward(xs)?;

        let q = q
            .reshape((b_sz, q_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let k = k
            .reshape((b_sz, q_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let v = v
            .reshape((b_sz, q_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;

        let (q, k) = self.rotary.apply(&q, &k)?;

        let k = repeat_kv(k, self.num_kv_groups)?;
        let v = repeat_kv(v, self.num_kv_groups)?;

        #[allow(clippy::cast_precision_loss)]
        let scale = 1f64 / f64::sqrt(self.head_dim as f64);
        let attn_weights = (q.matmul(&k.transpose(2, 3)?)? * scale)?;

        let attn_weights = match attention_mask {
            None => attn_weights,
            Some(mask) => attn_weights.broadcast_add(mask)?,
        };
        // Why: `softmax_last_dim` is a no-backprop CustomOp1; training needs
        // the primitive-op `softmax` so gradients flow through attention.
        let attn_weights = candle_nn::ops::softmax(&attn_weights, candle_core::D::Minus1)?;
        let attn = attn_weights.matmul(&v)?;

        attn.transpose(1, 2)?
            .reshape((b_sz, q_len, self.num_heads * self.head_dim))?
            .apply(&self.o_proj)
    }
}

struct Mlp {
    gate_proj: MaybeLora,
    up_proj: MaybeLora,
    down_proj: MaybeLora,
    act: candle_nn::Activation,
}

impl Mlp {
    #[allow(clippy::needless_pass_by_value, clippy::too_many_arguments)]
    fn load(
        cfg: &Config,
        base_vb: VarBuilder,
        lora_vb: VarBuilder,
        lora_cfg: &LoraConfig,
        targets: &HashSet<String>,
        loader: &BaseLoader<'_>,
        abs_path: &str,
    ) -> Result<Self> {
        let hidden = cfg.hidden_size;
        let inter = cfg.intermediate_size;
        let gate_proj = maybe_lora(
            hidden,
            inter,
            "gate_proj",
            targets,
            &base_vb,
            &lora_vb,
            lora_cfg,
            loader,
            abs_path,
        )?;
        let up_proj = maybe_lora(
            hidden, inter, "up_proj", targets, &base_vb, &lora_vb, lora_cfg, loader, abs_path,
        )?;
        let down_proj = maybe_lora(
            inter,
            hidden,
            "down_proj",
            targets,
            &base_vb,
            &lora_vb,
            lora_cfg,
            loader,
            abs_path,
        )?;
        Ok(Self {
            gate_proj,
            up_proj,
            down_proj,
            act: cfg.hidden_act,
        })
    }
}

impl Module for Mlp {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let lhs = xs.apply(&self.gate_proj)?.apply(&self.act)?;
        let rhs = xs.apply(&self.up_proj)?;
        (lhs * rhs)?.apply(&self.down_proj)
    }
}

struct DecoderLayer {
    self_attn: Attention,
    mlp: Mlp,
    input_layernorm: RmsNorm,
    post_attention_layernorm: RmsNorm,
}

impl DecoderLayer {
    #[allow(clippy::needless_pass_by_value, clippy::too_many_arguments)]
    fn load(
        rotary: Arc<RotaryEmbedding>,
        cfg: &Config,
        base_vb: VarBuilder,
        lora_vb: VarBuilder,
        lora_cfg: &LoraConfig,
        targets: &HashSet<String>,
        loader: &BaseLoader<'_>,
        abs_path: &str,
    ) -> Result<Self> {
        let self_attn = Attention::load(
            rotary,
            cfg,
            base_vb.pp("self_attn"),
            lora_vb.pp("self_attn"),
            lora_cfg,
            targets,
            loader,
            &format!("{abs_path}.self_attn"),
        )?;
        let mlp = Mlp::load(
            cfg,
            base_vb.pp("mlp"),
            lora_vb.pp("mlp"),
            lora_cfg,
            targets,
            loader,
            &format!("{abs_path}.mlp"),
        )?;
        let input_layernorm = build_rms_norm(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            base_vb.pp("input_layernorm"),
            loader,
            &format!("{abs_path}.input_layernorm"),
        )?;
        let post_attention_layernorm = build_rms_norm(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            base_vb.pp("post_attention_layernorm"),
            loader,
            &format!("{abs_path}.post_attention_layernorm"),
        )?;
        Ok(Self {
            self_attn,
            mlp,
            input_layernorm,
            post_attention_layernorm,
        })
    }

    fn forward(&self, xs: &Tensor, attention_mask: Option<&Tensor>) -> Result<Tensor> {
        let residual = xs;
        // Why: `RmsNorm::forward` takes the fast `apply_op2_no_bwd` path
        // when input is contiguous, severing the autograd graph at every
        // layernorm. `forward_diff` always uses the primitive LayerNorm
        // forward which backpropagates.
        let xs = self.input_layernorm.forward_diff(xs)?;
        let xs = self.self_attn.forward(&xs, attention_mask)?;
        let xs = (xs + residual)?;
        let residual = &xs;
        let xs = self
            .post_attention_layernorm
            .forward_diff(&xs)?
            .apply(&self.mlp)?;
        residual + xs
    }
}

/// Trainable Mistral model: frozen base + injected `LoraLinear` adapters.
///
/// Load via [`Self::load`] with two separate `VarBuilder`s — `base_vb`
/// reads the frozen safetensor weights, `lora_vb` is built from the
/// trainer's `VarMap` so the LoRA A/B matrices register as trainable
/// `Var`s automatically.
pub struct TrainableMistral {
    embed_tokens: Embedding,
    layers: Vec<DecoderLayer>,
    norm: RmsNorm,
    lm_head: Linear,
    sliding_window: Option<usize>,
    device: Device,
    dtype: DType,
}

impl TrainableMistral {
    /// Load a `TrainableMistral` from frozen base weights and a
    /// gradient-tracking LoRA `VarBuilder`.
    ///
    /// The LoRA prefix `lora_vb` is expected to be the trainer's
    /// var-map builder already scoped to `base_model.model.` so that the
    /// final var names match PEFT canonical
    /// (`base_model.model.model.layers.N.self_attn.q_proj.lora_A.weight`,
    /// etc.) for round-trip with
    /// `blazen_llm_candle::lora::LoadedAdapter::from_dir`.
    ///
    /// # Errors
    ///
    /// Returns any candle error from missing tensors, shape mismatches,
    /// or `VarBuilder`/`VarMap` failures.
    pub fn load(
        base_vb: VarBuilder,
        lora_vb: VarBuilder,
        cfg: &Config,
        lora_cfg: &LoraConfig,
    ) -> Result<Self> {
        Self::load_with_mode(base_vb, lora_vb, None, cfg, lora_cfg, TrainMode::LoraOnly)
    }

    /// Build the trainable Mistral wrapper with an explicit [`TrainMode`].
    ///
    /// In [`TrainMode::LoraOnly`] this is identical to [`Self::load`]:
    /// base weights are read frozen from `base_vb` and only LoRA params
    /// land in the varmap behind `lora_vb`.
    ///
    /// In [`TrainMode::FullFineTune`] every linear's base weight is read
    /// from `base_vb` and copied into `train_varmap` as a trainable [`Var`]
    /// (under safetensors-canonical keys like
    /// `model.layers.0.self_attn.q_proj.weight`). `lora_cfg.target_modules`
    /// is ignored — no LoRA layers are constructed. `lora_vb` is unused
    /// but kept in the signature so callers can pass the same builder
    /// they would use for LoRA without conditional plumbing.
    ///
    /// `train_varmap` is required in `FullFineTune` mode and may be
    /// `None` in `LoraOnly` mode. Passing `None` in `FullFineTune` mode
    /// panics — the caller must always be able to provide it because
    /// the trainer owns the varmap that will receive AdamW updates.
    ///
    /// Mistral has NO `tie_word_embeddings` field in its upstream
    /// `Config` — `lm_head` is always a separately-registered linear
    /// loaded from `lm_head.weight`, so FFT mode always emits a distinct
    /// `lm_head.weight` `Var` (independent from `model.embed_tokens.weight`).
    ///
    /// # Errors
    ///
    /// Forwards candle/varbuilder errors from any underlying load.
    ///
    /// # Panics
    ///
    /// Panics if `mode == TrainMode::FullFineTune` and `train_varmap` is
    /// `None`. This is a programming error, not a runtime input failure.
    #[allow(clippy::needless_pass_by_value)]
    pub fn load_with_mode(
        base_vb: VarBuilder,
        lora_vb: VarBuilder,
        train_varmap: Option<&VarMap>,
        cfg: &Config,
        lora_cfg: &LoraConfig,
        mode: TrainMode,
    ) -> Result<Self> {
        let dtype = base_vb.dtype();
        let device = base_vb.device().clone();

        // Why: FullFineTune ignores LoRA targets entirely — no adapters
        // are constructed. LoraOnly honors the user's target list.
        let targets: HashSet<String> = match mode {
            TrainMode::LoraOnly => lora_cfg.target_modules.iter().cloned().collect(),
            TrainMode::FullFineTune => HashSet::new(),
        };

        if mode == TrainMode::FullFineTune {
            assert!(
                train_varmap.is_some(),
                "FullFineTune requires train_varmap; got None",
            );
        }

        let loader = BaseLoader { mode, train_varmap };

        let base_m = base_vb.pp("model");
        let lora_m = lora_vb.pp("model");

        let embed_tokens = build_embedding(
            cfg.vocab_size,
            cfg.hidden_size,
            base_m.pp("embed_tokens"),
            &loader,
            "model.embed_tokens",
        )?;
        let rotary = Arc::new(RotaryEmbedding::new(dtype, cfg, &device)?);

        let base_layers = base_m.pp("layers");
        let lora_layers = lora_m.pp("layers");
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for i in 0..cfg.num_hidden_layers {
            layers.push(DecoderLayer::load(
                rotary.clone(),
                cfg,
                base_layers.pp(i),
                lora_layers.pp(i),
                lora_cfg,
                &targets,
                &loader,
                &format!("model.layers.{i}"),
            )?);
        }
        let norm = build_rms_norm(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            base_m.pp("norm"),
            &loader,
            "model.norm",
        )?;
        // Why: Mistral does NOT tie embeddings — `lm_head` is always its
        // own untied linear loaded from `lm_head.weight`. In FFT mode this
        // means the train varmap gets a distinct `lm_head.weight` Var
        // (independent from `model.embed_tokens.weight`).
        let lm_head = loader.linear_no_bias(
            cfg.hidden_size,
            cfg.vocab_size,
            base_vb.pp("lm_head"),
            "lm_head",
        )?;

        Ok(Self {
            embed_tokens,
            layers,
            norm,
            lm_head,
            sliding_window: cfg.sliding_window,
            device,
            dtype,
        })
    }

    /// Forward pass returning logits over the full sequence
    /// (`[b, seq, vocab]`).
    ///
    /// Unlike the upstream inference model (which narrows to the last
    /// token for sampling), the trainer needs per-token logits so the
    /// shifted cross-entropy loss can be computed at every position.
    ///
    /// # Errors
    ///
    /// Returns any candle error from the embedding lookup, attention,
    /// MLP, or final projection.
    pub fn forward(&self, input_ids: &Tensor) -> Result<Tensor> {
        let (_b, seq_len) = input_ids.dims2()?;
        let attention_mask = if seq_len <= 1 {
            None
        } else {
            Some(self.prepare_attention_mask(seq_len)?)
        };
        let mut xs = self.embed_tokens.forward(input_ids)?;
        for layer in &self.layers {
            xs = layer.forward(&xs, attention_mask.as_ref())?;
        }
        // Why: see DecoderLayer::forward — use `forward_diff` to bypass
        // the no-bwd fast-path RmsNorm.
        self.norm.forward_diff(&xs)?.apply(&self.lm_head)
    }

    /// Sliding-window-aware causal mask. Mirrors upstream
    /// `Model::prepare_decoder_attention_mask`: token `i` may attend to
    /// token `j` iff `j <= i` AND `j + sliding_window >= i`. With no
    /// sliding window configured the second clause is vacuous (only
    /// causal masking applies).
    fn prepare_attention_mask(&self, tgt_len: usize) -> Result<Tensor> {
        let sliding_window = self.sliding_window.unwrap_or(tgt_len + 1);
        let mask: Vec<f32> = (0..tgt_len)
            .flat_map(|i| {
                (0..tgt_len).map(move |j| {
                    if i < j || j + sliding_window < i {
                        f32::NEG_INFINITY
                    } else {
                        0.0
                    }
                })
            })
            .collect();
        let mask = Tensor::from_slice(&mask, (tgt_len, tgt_len), &self.device)?;
        mask.expand((1, 1, tgt_len, tgt_len))?.to_dtype(self.dtype)
    }

    /// Inspect the configured sliding window (mirrors upstream `Config`).
    #[must_use]
    pub fn sliding_window(&self) -> Option<usize> {
        self.sliding_window
    }

    /// Number of decoder layers loaded.
    #[must_use]
    pub fn num_layers(&self) -> usize {
        self.layers.len()
    }
}

// Why: D is imported by the inference model but unused in the training
// forward path (no kv-cache concat on dim -1); keep the import suppressed
// rather than dropping it because future training-time extensions
// (e.g. packing multiple sequences) will need it.
#[allow(dead_code)]
const _D_USED: Option<D> = None;

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device, Tensor};
    use candle_nn::{Activation, VarMap};

    fn tiny_mistral_config() -> Config {
        Config {
            vocab_size: 32,
            hidden_size: 32,
            intermediate_size: 64,
            num_hidden_layers: 2,
            num_attention_heads: 4,
            head_dim: None,
            num_key_value_heads: 2,
            hidden_act: Activation::Silu,
            max_position_embeddings: 64,
            rms_norm_eps: 1e-5,
            rope_theta: 10_000.0,
            sliding_window: Some(8),
            use_flash_attn: false,
        }
    }

    fn lora_cfg(target_modules: &[&str]) -> LoraConfig {
        LoraConfig {
            rank: 4,
            alpha: 8.0,
            dropout: 0.0,
            target_modules: target_modules.iter().map(|s| (*s).to_string()).collect(),
        }
    }

    fn build_model(
        cfg: &Config,
        lora_cfg: &LoraConfig,
    ) -> (TrainableMistral, VarMap, VarMap, Device) {
        let device = Device::Cpu;
        let base_map = VarMap::new();
        let lora_map = VarMap::new();
        // Why: pre-populate every base weight to zero so the load path
        // succeeds without a real safetensors file. RmsNorm scale gets
        // ones via candle_nn::rms_norm's default initializer when reading
        // from a writable VarBuilder, so we need a writable builder here.
        let base_vb = VarBuilder::from_varmap(&base_map, DType::F32, &device);
        let lora_vb = VarBuilder::from_varmap(&lora_map, DType::F32, &device);
        let model = TrainableMistral::load(base_vb, lora_vb, cfg, lora_cfg)
            .expect("load trainable mistral");
        (model, base_map, lora_map, device)
    }

    #[test]
    fn forward_shape_correct() {
        let cfg = tiny_mistral_config();
        let (model, _b, _l, device) = build_model(&cfg, &lora_cfg(&["q_proj", "v_proj"]));
        let input_ids =
            Tensor::from_vec(vec![0u32, 1, 2, 3, 4], (1, 5), &device).expect("input_ids");
        let logits = model.forward(&input_ids).expect("forward");
        let dims = logits.dims();
        assert_eq!(dims, &[1, 5, cfg.vocab_size], "logits shape mismatch");
    }

    #[test]
    fn lora_target_modules_only_creates_lora_at_targets() {
        let cfg = tiny_mistral_config();
        let targets = ["q_proj", "v_proj"];
        let (_model, _base_map, lora_map, _device) = build_model(&cfg, &lora_cfg(&targets));

        let guard = lora_map.data().lock().expect("lora varmap mutex poisoned");
        let lora_keys: Vec<&String> = guard
            .keys()
            .filter(|k| k.ends_with(".lora_A.weight") || k.ends_with(".lora_B.weight"))
            .collect();

        // 2 layers * 2 target modules * 2 mats (A,B) = 8
        assert_eq!(
            lora_keys.len(),
            cfg.num_hidden_layers * targets.len() * 2,
            "wrong number of lora keys; got {lora_keys:?}",
        );

        for key in &lora_keys {
            assert!(
                key.contains(".self_attn.q_proj.") || key.contains(".self_attn.v_proj."),
                "lora key at non-target site: {key}",
            );
            assert!(
                !key.contains(".k_proj.") && !key.contains(".o_proj."),
                "lora key created at non-target site: {key}",
            );
            assert!(
                !key.contains(".mlp."),
                "mlp module shouldn't have lora when targets are q/v only: {key}",
            );
        }
    }

    #[test]
    fn zero_init_lora_b_means_initial_logits_match_base_only() {
        let cfg = tiny_mistral_config();
        let device = Device::Cpu;

        let base_map = VarMap::new();
        let base_vb = VarBuilder::from_varmap(&base_map, DType::F32, &device);
        let lora_map_with = VarMap::new();
        let lora_vb_with = VarBuilder::from_varmap(&lora_map_with, DType::F32, &device);
        let model_with = TrainableMistral::load(
            base_vb,
            lora_vb_with,
            &cfg,
            &lora_cfg(&["q_proj", "k_proj", "v_proj", "o_proj"]),
        )
        .expect("model with lora");

        // Why: build a parallel model with the SAME base varmap but an
        // empty target-module list — no LoraLinear gets instantiated, so
        // forward() represents the pure base-only path.
        let lora_map_without = VarMap::new();
        let lora_vb_without = VarBuilder::from_varmap(&lora_map_without, DType::F32, &device);
        let base_vb2 = VarBuilder::from_varmap(&base_map, DType::F32, &device);
        let model_without = TrainableMistral::load(base_vb2, lora_vb_without, &cfg, &lora_cfg(&[]))
            .expect("model without lora");

        let input_ids = Tensor::from_vec(vec![1u32, 2, 3, 4], (1, 4), &device).expect("input_ids");
        let logits_with = model_with.forward(&input_ids).expect("forward with");
        let logits_without = model_without.forward(&input_ids).expect("forward without");

        let diff = (&logits_with - &logits_without).expect("diff");
        let max_abs = diff
            .abs()
            .expect("abs")
            .flatten_all()
            .expect("flatten")
            .max(0)
            .expect("max")
            .to_scalar::<f32>()
            .expect("scalar");
        assert!(
            max_abs < 1e-5,
            "B=0 must make LoRA forward identical to base-only; got max-abs {max_abs}",
        );
    }

    #[test]
    fn sliding_window_attention_mask_applied() {
        let cfg = tiny_mistral_config();
        let sw = cfg.sliding_window.expect("sliding window set");
        let (model, _b, _l, _d) = build_model(&cfg, &lora_cfg(&["q_proj"]));

        // tgt_len large enough that the sliding-window clause actually
        // bites (i = sw + 2 vs j = 1 must be masked).
        let tgt_len = sw + 4;
        let mask = model.prepare_attention_mask(tgt_len).expect("mask");
        let dims = mask.dims();
        assert_eq!(dims, &[1, 1, tgt_len, tgt_len], "mask shape");

        let mask_f32 = mask
            .to_dtype(DType::F32)
            .expect("to f32")
            .squeeze(0)
            .expect("squeeze batch")
            .squeeze(0)
            .expect("squeeze head");
        let m: Vec<Vec<f32>> = mask_f32.to_vec2().expect("to_vec2");

        // Causal: i < j must be -inf
        let upper = m[2][5];
        assert!(
            upper.is_infinite() && upper.is_sign_negative(),
            "causal violated"
        );

        // Sliding-window: j + sw < i must be -inf. Pick i = sw + 3, j = 1.
        let i = sw + 3;
        let j = 1usize;
        assert!(
            j + sw < i,
            "test precondition: need j + sw < i (sw={sw} i={i} j={j})",
        );
        let blocked = m[i][j];
        assert!(
            blocked.is_infinite() && blocked.is_sign_negative(),
            "sliding-window mask not applied at [i={i}][j={j}]: got {blocked}",
        );

        // Diagonal and near-diagonal within the window must be zero.
        assert!((m[1][1] - 0.0).abs() < 1e-6, "self-attention masked");
        assert!((m[5][4] - 0.0).abs() < 1e-6, "in-window past masked");
    }

    // -----------------------------------------------------------------
    // FullFineTune mode tests (Wave 8c)
    // -----------------------------------------------------------------

    /// Build a Mistral in `FullFineTune` mode, returning the train VarMap
    /// that received every base weight as a fresh `Var`.
    ///
    /// `base_map` is provided by the caller so two builds can share a
    /// source-of-truth set of base weights (used by the
    /// `forward_shape_matches_lora_mode` test).
    fn build_full_finetune(
        base_map: &VarMap,
        cfg: &Config,
    ) -> (TrainableMistral, VarMap, VarMap, Device) {
        let device = Device::Cpu;
        let lora_map = VarMap::new();
        let train_map = VarMap::new();
        let base_vb = VarBuilder::from_varmap(base_map, DType::F32, &device);
        let lora_vb = VarBuilder::from_varmap(&lora_map, DType::F32, &device);

        let model = TrainableMistral::load_with_mode(
            base_vb,
            lora_vb,
            Some(&train_map),
            cfg,
            &lora_cfg(&[]),
            TrainMode::FullFineTune,
        )
        .expect("full-FT model loads");

        (model, lora_map, train_map, device)
    }

    /// Expected absolute key list (every trainable `.weight`) for the
    /// `tiny_mistral_config()` model. Mirrors the safetensors naming
    /// convention so `VarMap::save` output round-trips through the
    /// inference-side loader.
    ///
    /// Mistral has NO biases on q/k/v/o or MLP projections (like Llama,
    /// unlike Qwen2). `lm_head.weight` is always present because Mistral
    /// does NOT tie embeddings.
    fn expected_full_finetune_keys(cfg: &Config) -> Vec<String> {
        let mut keys = Vec::new();
        keys.push("model.embed_tokens.weight".to_string());
        for li in 0..cfg.num_hidden_layers {
            for proj in ["q_proj", "k_proj", "v_proj", "o_proj"] {
                keys.push(format!("model.layers.{li}.self_attn.{proj}.weight"));
            }
            for proj in ["gate_proj", "up_proj", "down_proj"] {
                keys.push(format!("model.layers.{li}.mlp.{proj}.weight"));
            }
            keys.push(format!("model.layers.{li}.input_layernorm.weight"));
            keys.push(format!("model.layers.{li}.post_attention_layernorm.weight"));
        }
        keys.push("model.norm.weight".to_string());
        keys.push("lm_head.weight".to_string());
        keys
    }

    #[test]
    fn mistral_full_finetune_loads_base_into_varmap() {
        let cfg = tiny_mistral_config();
        let base_map = VarMap::new();
        let (_model, _lora, train_map, _dev) = build_full_finetune(&base_map, &cfg);

        let guard = train_map
            .data()
            .lock()
            .expect("train varmap mutex poisoned by another thread");
        let keys: std::collections::HashSet<String> = guard.keys().cloned().collect();
        drop(guard);

        // Minimal sanity probe: at least one well-known weight is present.
        assert!(
            keys.contains("model.layers.0.self_attn.q_proj.weight"),
            "FullFineTune train varmap missing q_proj.weight; have: {keys:?}",
        );
        // FullFineTune ignores LoRA targets, so NO lora_A/lora_B should be
        // present in the train varmap. (The separate LoRA varmap stays
        // empty because the wrapper never constructs LoraLinear in FFT.)
        for k in &keys {
            assert!(
                !k.contains("lora_A") && !k.contains("lora_B"),
                "FullFineTune train varmap unexpectedly has LoRA key: {k}",
            );
        }
    }

    #[test]
    fn mistral_full_finetune_forward_shape_matches_lora_mode() {
        let device = Device::Cpu;
        let cfg = tiny_mistral_config();

        // Build the LoraOnly reference with NO LoRA targets so it's a
        // base-only forward — values will match FullFineTune at step 0
        // because the FFT path copies the same base weights into Vars
        // without modifying them.
        let base_map = VarMap::new();
        let lora_map_ref = VarMap::new();
        let base_vb_ref = VarBuilder::from_varmap(&base_map, DType::F32, &device);
        let lora_vb_ref = VarBuilder::from_varmap(&lora_map_ref, DType::F32, &device);
        let lora_only = TrainableMistral::load(base_vb_ref, lora_vb_ref, &cfg, &lora_cfg(&[]))
            .expect("lora load");

        // Reuse the same base_map for FullFineTune so the source weights
        // are bit-identical between the two models.
        let (full_ft, _lora_map, _train_map, _dev) = build_full_finetune(&base_map, &cfg);

        let input =
            Tensor::from_vec(vec![1u32, 2, 3, 4, 5, 6, 7, 8], (1, 8), &device).expect("input ids");

        let out_lora = lora_only.forward(&input).expect("lora forward");
        let out_ft = full_ft.forward(&input).expect("ft forward");

        // Why: shapes must match exactly — same arch wraps the same cfg.
        assert_eq!(out_ft.dims(), out_lora.dims());
        assert_eq!(out_ft.dims(), &[1, 8, cfg.vocab_size]);

        // Bonus: at step 0 (no weight updates yet), the FFT path is just
        // copy-in-place of the same base weights, so logits should match
        // numerically.
        let diff = (&out_ft - &out_lora).expect("diff");
        let max_abs = diff
            .abs()
            .expect("abs")
            .flatten_all()
            .expect("flatten")
            .max(0)
            .expect("max")
            .to_scalar::<f32>()
            .expect("scalar");
        assert!(
            max_abs < 1e-5,
            "FullFineTune at step 0 should match LoraOnly base-only forward; got max-abs delta {max_abs}",
        );
    }

    #[test]
    fn mistral_full_finetune_every_linear_has_var() {
        let cfg = tiny_mistral_config();
        let base_map = VarMap::new();
        let (_model, _lora, train_map, _dev) = build_full_finetune(&base_map, &cfg);

        let guard = train_map
            .data()
            .lock()
            .expect("train varmap mutex poisoned by another thread");
        let actual: std::collections::HashSet<String> = guard.keys().cloned().collect();
        drop(guard);

        let expected = expected_full_finetune_keys(&cfg);
        for key in &expected {
            assert!(
                actual.contains(key),
                "FullFineTune train varmap missing expected key {key}; have: {actual:?}",
            );
        }

        // Why: also assert no surprise extras — every key in the varmap
        // must appear in our expected list. This catches accidental
        // double-registration or stale prefix bugs.
        let expected_set: std::collections::HashSet<&String> = expected.iter().collect();
        for key in &actual {
            assert!(
                expected_set.contains(key),
                "FullFineTune train varmap has unexpected key {key}; expected one of: {expected:?}",
            );
        }
    }

    #[test]
    fn mistral_full_finetune_lm_head_is_independent_var() {
        // Why: Mistral's upstream Config has no `tie_word_embeddings`
        // field — `lm_head` is always loaded independently from
        // `lm_head.weight`. In FFT mode the train varmap must contain
        // BOTH `model.embed_tokens.weight` AND `lm_head.weight` as
        // distinct entries, and the underlying tensors must NOT share
        // autograd identity (otherwise tying would silently appear).
        let cfg = tiny_mistral_config();
        let base_map = VarMap::new();
        let (_model, _lora, train_map, _dev) = build_full_finetune(&base_map, &cfg);

        let guard = train_map
            .data()
            .lock()
            .expect("train varmap mutex poisoned by another thread");

        let embed = guard
            .get("model.embed_tokens.weight")
            .expect("embed_tokens Var registered")
            .clone();
        let lm_head = guard
            .get("lm_head.weight")
            .expect("lm_head Var registered (Mistral does not tie)")
            .clone();
        drop(guard);

        // Why: `Var::as_tensor().id()` is the autograd identity used by
        // candle's backward pass. Two independent Vars must have distinct
        // IDs; a tied setup would reuse the same id.
        assert_ne!(
            embed.as_tensor().id(),
            lm_head.as_tensor().id(),
            "Mistral FFT must keep lm_head independent from embed_tokens (got same Var id)",
        );
    }
}
