//! Trainable Llama 2 / Llama 3 model wrapper.
//!
//! Wave 2B of PR7 mirrors `candle_transformers::models::llama::Llama`'s
//! forward pass but substitutes [`crate::lora::LoraLinear`] at every
//! attention projection (q/k/v/o) and, when requested, the MLP
//! projections (gate/up/down).
//!
//! The wrapper is a *training* forward: no KV cache, full causal mask,
//! per-token logits (`[B, T, vocab]`, not just the final position).
//! Inference still uses the upstream `candle_transformers::models::llama::Llama`.
//!
//! # Var-path naming
//!
//! Base weights load from `base_vb` at the upstream Llama paths
//! (`model.embed_tokens.weight`,
//! `model.layers.N.self_attn.{q,k,v,o}_proj.weight`,
//! `model.layers.N.mlp.{gate,up,down}_proj.weight`,
//! `model.layers.N.input_layernorm.weight`,
//! `model.layers.N.post_attention_layernorm.weight`,
//! `model.norm.weight`, `lm_head.weight`).
//!
//! LoRA A/B weights load from `lora_vb` at the PEFT-canonical paths so
//! [`crate::export::adapter_state_dict`] round-trips through
//! `crates/blazen-llm-candle/src/lora.rs::LoadedAdapter::from_dir`:
//!
//! ```text
//! base_model.model.model.layers.{i}.self_attn.{q,k,v,o}_proj.{lora_A,lora_B}.weight
//! base_model.model.model.layers.{i}.mlp.{gate,up,down}_proj.{lora_A,lora_B}.weight
//! ```
//!
//! Callers are expected to push the `base_model.model` prefix on
//! `lora_vb` themselves (Wave 4's `Trainer::load_arch` does this) so the
//! per-site `push_prefix` here only adds the layer/module suffix.

pub use candle_transformers::models::llama::{
    Config, Llama3RopeConfig, Llama3RopeType, LlamaConfig, LlamaEosToks,
};

use std::collections::HashSet;
use std::f32::consts::PI;

use candle_core::{DType, IndexOp, Result, Tensor};
use candle_nn::{Embedding, Linear, Module, RmsNorm, VarBuilder, VarMap};

use crate::arch::{BaseLoader, TrainMode, build_embedding, build_rms_norm};
use crate::config::LoraConfig;
use crate::lora::LoraLinear;
use crate::qlora::QLoraLinear;

// ---------------------------------------------------------------------------
// MaybeLora
// ---------------------------------------------------------------------------

/// Per-projection dispatch. A target linear is either wrapped with
/// trainable LoRA adapters on a dense base ([`MaybeLora::Lora`]), wrapped
/// with LoRA adapters on a *quantized* base ([`MaybeLora::Qlora`] —
/// QLoRA mode), or left as a plain frozen [`Linear`] ([`MaybeLora::Plain`]).
enum MaybeLora {
    Plain(Linear),
    Lora(LoraLinear),
    Qlora(QLoraLinear),
}

impl MaybeLora {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        match self {
            Self::Plain(l) => l.forward(x),
            Self::Lora(l) => l.forward(x),
            Self::Qlora(l) => l.forward(x),
        }
    }
}

/// Build a `Linear` from `base_vb` at `name` (mode-aware via `loader` —
/// either frozen mmap or copied into `train_varmap` as a fresh `Var`),
/// then optionally wrap with `LoraLinear` if `name` is in `targets`.
///
/// `lora_vb` must already be scoped to the *containing* path (e.g.
/// `base_model.model.model.layers.0.self_attn`); this helper pushes the
/// target name as its own prefix so the resulting weight keys are
/// `<lora_vb>/<name>/lora_{A,B}.weight`.
///
/// In `TrainMode::FullFineTune` mode `targets` is expected to be empty,
/// so `LoraLinear` wrapping is skipped and only the FFT base var is
/// registered under `{abs_path}.{name}.weight`.
// Why: every argument is load-bearing — module name + LoRA target set,
// LoRA cfg (rank/alpha), two VarBuilders (frozen vs trainable scopes),
// mode-aware loader, and the absolute key prefix for FFT varmap
// insertion. Bundling into a struct would just move the line count.
#[allow(clippy::too_many_arguments)]
fn maybe_lora_linear(
    in_dim: usize,
    out_dim: usize,
    name: &str,
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
        base_vb.pp(name),
        &format!("{abs_path}.{name}"),
    )?;
    if !targets.contains(name) {
        return Ok(MaybeLora::Plain(base));
    }
    // Why: target linear — wrap with LoRA, optionally quantizing the
    // frozen base if the trainer asked for QLoRA mode. FullFineTune
    // contains an empty target set so it never reaches this branch.
    match loader.mode {
        TrainMode::Qlora { quant } => {
            let wrapped = QLoraLinear::wrap(
                base,
                in_dim,
                out_dim,
                lora_cfg.rank,
                lora_cfg.alpha,
                quant,
                lora_vb.pp(name),
            )?;
            Ok(MaybeLora::Qlora(wrapped))
        }
        TrainMode::LoraOnly | TrainMode::FullFineTune => {
            let wrapped = LoraLinear::wrap(
                base,
                in_dim,
                out_dim,
                lora_cfg.rank,
                lora_cfg.alpha,
                lora_vb.pp(name),
            )?;
            Ok(MaybeLora::Lora(wrapped))
        }
    }
}

// ---------------------------------------------------------------------------
// RoPE cache (training-mode: position-independent, full-sequence)
// ---------------------------------------------------------------------------

#[derive(Clone)]
struct RopeCache {
    cos: Tensor,
    sin: Tensor,
}

#[allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)]
fn calculate_default_inv_freq(cfg: &Config) -> Vec<f32> {
    let head_dim = cfg.hidden_size / cfg.num_attention_heads;
    (0..head_dim)
        .step_by(2)
        .map(|i| 1f32 / cfg.rope_theta.powf(i as f32 / head_dim as f32))
        .collect()
}

#[allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)]
fn build_rope_cache(cfg: &Config, dtype: DType, device: &candle_core::Device) -> Result<RopeCache> {
    // Why: mirrors candle_transformers::models::llama::Cache::new — including
    // the Llama3 rope_scaling smooth-interpolation path. We rebuild instead
    // of reusing the upstream Cache because that type also owns kv-cache /
    // mask state we don't want in the training graph.
    let theta = match &cfg.rope_scaling {
        None
        | Some(Llama3RopeConfig {
            rope_type: Llama3RopeType::Default,
            ..
        }) => calculate_default_inv_freq(cfg),
        Some(rope_scaling) => {
            let low_freq_wavelen =
                rope_scaling.original_max_position_embeddings as f32 / rope_scaling.low_freq_factor;
            let high_freq_wavelen = rope_scaling.original_max_position_embeddings as f32
                / rope_scaling.high_freq_factor;

            calculate_default_inv_freq(cfg)
                .into_iter()
                .map(|freq| {
                    let wavelen = 2. * PI / freq;
                    if wavelen < high_freq_wavelen {
                        freq
                    } else if wavelen > low_freq_wavelen {
                        freq / rope_scaling.factor
                    } else {
                        let smooth = (rope_scaling.original_max_position_embeddings as f32
                            / wavelen
                            - rope_scaling.low_freq_factor)
                            / (rope_scaling.high_freq_factor - rope_scaling.low_freq_factor);
                        (1. - smooth) * freq / rope_scaling.factor + smooth * freq
                    }
                })
                .collect::<Vec<_>>()
        }
    };

    let theta = Tensor::new(theta, device)?;
    let idx_theta = Tensor::arange(
        0,
        u32::try_from(cfg.max_position_embeddings).unwrap_or(u32::MAX),
        device,
    )?
    .to_dtype(DType::F32)?
    .reshape((cfg.max_position_embeddings, 1))?
    .matmul(&theta.reshape((1, theta.elem_count()))?)?;
    let cos = idx_theta.cos()?.to_dtype(dtype)?;
    let sin = idx_theta.sin()?.to_dtype(dtype)?;
    Ok(RopeCache { cos, sin })
}

// ---------------------------------------------------------------------------
// CausalSelfAttention
// ---------------------------------------------------------------------------

struct CausalSelfAttention {
    q_proj: MaybeLora,
    k_proj: MaybeLora,
    v_proj: MaybeLora,
    o_proj: MaybeLora,
    num_attention_heads: usize,
    num_key_value_heads: usize,
    head_dim: usize,
}

impl CausalSelfAttention {
    // Why: VarBuilder is a cheap Arc-cloning handle; candle's idiomatic
    // call sites move freshly `pp`-scoped builders in (mirrors
    // candle_nn::linear and the qwen2 wrapper). The extra `loader` +
    // `abs_path` + `targets` arguments are load-bearing for FFT.
    #[allow(clippy::needless_pass_by_value, clippy::too_many_arguments)]
    fn load(
        base_vb: VarBuilder,
        lora_vb: VarBuilder,
        cfg: &Config,
        lora_cfg: &LoraConfig,
        targets: &HashSet<String>,
        loader: &BaseLoader<'_>,
        abs_path: &str,
    ) -> Result<Self> {
        let hidden = cfg.hidden_size;
        let head_dim = hidden / cfg.num_attention_heads;
        let size_q = head_dim * cfg.num_attention_heads;
        let size_kv = head_dim * cfg.num_key_value_heads;

        let q_proj = maybe_lora_linear(
            hidden, size_q, "q_proj", targets, &base_vb, &lora_vb, lora_cfg, loader, abs_path,
        )?;
        let k_proj = maybe_lora_linear(
            hidden, size_kv, "k_proj", targets, &base_vb, &lora_vb, lora_cfg, loader, abs_path,
        )?;
        let v_proj = maybe_lora_linear(
            hidden, size_kv, "v_proj", targets, &base_vb, &lora_vb, lora_cfg, loader, abs_path,
        )?;
        let o_proj = maybe_lora_linear(
            size_q, hidden, "o_proj", targets, &base_vb, &lora_vb, lora_cfg, loader, abs_path,
        )?;

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            num_attention_heads: cfg.num_attention_heads,
            num_key_value_heads: cfg.num_key_value_heads,
            head_dim,
        })
    }

    fn repeat_kv(&self, x: Tensor) -> Result<Tensor> {
        candle_transformers::utils::repeat_kv(
            x,
            self.num_attention_heads / self.num_key_value_heads,
        )
    }

    #[allow(clippy::many_single_char_names)]
    fn forward(&self, xs: &Tensor, rope: &RopeCache, mask: &Tensor) -> Result<Tensor> {
        let (b_sz, seq_len, hidden) = xs.dims3()?;

        let q = self.q_proj.forward(xs)?;
        let k = self.k_proj.forward(xs)?;
        let v = self.v_proj.forward(xs)?;

        let q = q
            .reshape((b_sz, seq_len, self.num_attention_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let k = k
            .reshape((b_sz, seq_len, self.num_key_value_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let v = v
            .reshape((b_sz, seq_len, self.num_key_value_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;

        let q = apply_rope(&q, rope, seq_len)?;
        let k = apply_rope(&k, rope, seq_len)?;

        let k = self.repeat_kv(k)?;
        let v = self.repeat_kv(v)?;

        let in_dtype = q.dtype();
        let q = q.to_dtype(DType::F32)?;
        let k = k.to_dtype(DType::F32)?;
        let v = v.to_dtype(DType::F32)?;

        #[allow(clippy::cast_precision_loss)]
        let scale = 1f64 / (self.head_dim as f64).sqrt();
        let att = (q.matmul(&k.t()?)? * scale)?;

        let att = if seq_len == 1 {
            att
        } else {
            let mask = mask.broadcast_as(att.shape())?;
            masked_fill(&att, &mask, f32::NEG_INFINITY)?
        };

        // Why: `softmax_last_dim` is a no-backprop CustomOp1; training needs
        // the primitive-op `softmax` so gradients flow through attention.
        let att = candle_nn::ops::softmax(&att, candle_core::D::Minus1)?;
        let y = att.matmul(&v.contiguous()?)?.to_dtype(in_dtype)?;
        let y = y.transpose(1, 2)?.reshape((b_sz, seq_len, hidden))?;
        self.o_proj.forward(&y)
    }
}

fn apply_rope(x: &Tensor, rope: &RopeCache, seq_len: usize) -> Result<Tensor> {
    let cos = rope.cos.narrow(0, 0, seq_len)?;
    let sin = rope.sin.narrow(0, 0, seq_len)?;
    // Why: `candle_nn::rotary_emb::rope` is a no-bwd CustomOp3 and severs
    // the autograd graph at attention. `rope_slow` is the primitive-op
    // equivalent that backpropagates.
    candle_nn::rotary_emb::rope_slow(x, &cos, &sin)
}

fn masked_fill(on_false: &Tensor, mask: &Tensor, on_true: f32) -> Result<Tensor> {
    let shape = mask.shape();
    let on_true = Tensor::new(on_true, on_false.device())?.broadcast_as(shape.dims())?;
    mask.where_cond(&on_true, on_false)
}

fn build_causal_mask(seq_len: usize, device: &candle_core::Device) -> Result<Tensor> {
    candle_transformers::utils::build_causal_mask(seq_len, 0, device)
}

// ---------------------------------------------------------------------------
// MLP (Llama uses SwiGLU: silu(gate) * up, then down)
// ---------------------------------------------------------------------------

#[allow(clippy::struct_field_names)]
struct Mlp {
    gate_proj: MaybeLora,
    up_proj: MaybeLora,
    down_proj: MaybeLora,
}

impl Mlp {
    #[allow(clippy::needless_pass_by_value, clippy::too_many_arguments)]
    fn load(
        base_vb: VarBuilder,
        lora_vb: VarBuilder,
        cfg: &Config,
        lora_cfg: &LoraConfig,
        targets: &HashSet<String>,
        loader: &BaseLoader<'_>,
        abs_path: &str,
    ) -> Result<Self> {
        let h = cfg.hidden_size;
        let i = cfg.intermediate_size;
        let gate_proj = maybe_lora_linear(
            h,
            i,
            "gate_proj",
            targets,
            &base_vb,
            &lora_vb,
            lora_cfg,
            loader,
            abs_path,
        )?;
        let up_proj = maybe_lora_linear(
            h, i, "up_proj", targets, &base_vb, &lora_vb, lora_cfg, loader, abs_path,
        )?;
        let down_proj = maybe_lora_linear(
            i,
            h,
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
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let g = candle_nn::ops::silu(&self.gate_proj.forward(x)?)?;
        let u = self.up_proj.forward(x)?;
        self.down_proj.forward(&(g * u)?)
    }
}

// ---------------------------------------------------------------------------
// DecoderLayer (Block)
// ---------------------------------------------------------------------------

struct DecoderLayer {
    input_layernorm: RmsNorm,
    self_attn: CausalSelfAttention,
    post_attention_layernorm: RmsNorm,
    mlp: Mlp,
}

impl DecoderLayer {
    #[allow(clippy::needless_pass_by_value, clippy::too_many_arguments)]
    fn load(
        base_vb: VarBuilder,
        lora_vb: VarBuilder,
        cfg: &Config,
        lora_cfg: &LoraConfig,
        targets: &HashSet<String>,
        loader: &BaseLoader<'_>,
        abs_path: &str,
    ) -> Result<Self> {
        let self_attn = CausalSelfAttention::load(
            base_vb.pp("self_attn"),
            lora_vb.pp("self_attn"),
            cfg,
            lora_cfg,
            targets,
            loader,
            &format!("{abs_path}.self_attn"),
        )?;
        let mlp = Mlp::load(
            base_vb.pp("mlp"),
            lora_vb.pp("mlp"),
            cfg,
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
            input_layernorm,
            self_attn,
            post_attention_layernorm,
            mlp,
        })
    }

    fn forward(&self, x: &Tensor, rope: &RopeCache, mask: &Tensor) -> Result<Tensor> {
        let residual = x;
        // Why: `RmsNorm::forward` takes the fast `apply_op2_no_bwd` path
        // when input is contiguous, severing the autograd graph at every
        // layernorm. `forward_diff` always uses the primitive LayerNorm
        // forward which backpropagates.
        let h = self.input_layernorm.forward_diff(x)?;
        let h = self.self_attn.forward(&h, rope, mask)?;
        let h = (h + residual)?;
        let residual = &h;
        let n = self.post_attention_layernorm.forward_diff(&h)?;
        let m = self.mlp.forward(&n)?;
        m + residual
    }
}

// ---------------------------------------------------------------------------
// TrainableLlama
// ---------------------------------------------------------------------------

/// Trainable Llama 2 / Llama 3 wrapper with LoRA on selected projections.
///
/// Forward returns per-token logits `[B, T, vocab_size]` for use with a
/// next-token cross-entropy loss. There is no KV cache, no
/// position-indexed slicing, and no final-token-only optimization — the
/// training graph wants gradients on every position.
pub struct TrainableLlama {
    embed_tokens: Embedding,
    layers: Vec<DecoderLayer>,
    norm: RmsNorm,
    // Why: `Linear` (when tied) and `Linear` (untied via linear_no_bias)
    // both go through candle_nn::Linear, so a single Linear field works
    // either way; the tying logic only affects construction.
    lm_head: Linear,
    rope: RopeCache,
    tied: bool,
}

impl TrainableLlama {
    /// Load the model from two `VarBuilder`s: `base_vb` holds the frozen
    /// pretrained weights, `lora_vb` is the [`VarBuilder::from_varmap`]
    /// where new LoRA A/B vars are registered.
    ///
    /// `lora_vb` should already include the PEFT prefix
    /// (`base_model.model`) so the resulting weight keys round-trip
    /// through `crates/blazen-llm-candle/src/lora.rs::LoadedAdapter`.
    ///
    /// # Errors
    ///
    /// Forwards candle/varbuilder errors from any underlying load. Returns
    /// an error if `cfg.hidden_size` is not divisible by
    /// `cfg.num_attention_heads`.
    pub fn load(
        base_vb: VarBuilder,
        lora_vb: VarBuilder,
        cfg: &Config,
        lora_cfg: &LoraConfig,
    ) -> Result<Self> {
        Self::load_with_mode(base_vb, lora_vb, None, cfg, lora_cfg, TrainMode::LoraOnly)
    }

    /// Build the trainable Llama wrapper with an explicit [`TrainMode`].
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
    /// Tied-embedding handling: when `cfg.tie_word_embeddings` is true
    /// the embedding tensor is registered in the varmap exactly once
    /// under `model.embed_tokens.weight`, and `lm_head` reuses the same
    /// underlying tensor via `Linear::new(embed_tokens.embeddings().clone(), None)`.
    /// No separate `lm_head.weight` entry is created.
    ///
    /// # Errors
    ///
    /// Forwards candle/varbuilder errors from any underlying load. Returns
    /// an error if `cfg.hidden_size` is not divisible by
    /// `cfg.num_attention_heads`.
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
        if !cfg.hidden_size.is_multiple_of(cfg.num_attention_heads) {
            return Err(candle_core::Error::Msg(format!(
                "hidden_size {} not divisible by num_attention_heads {}",
                cfg.hidden_size, cfg.num_attention_heads
            )));
        }
        if !cfg
            .num_attention_heads
            .is_multiple_of(cfg.num_key_value_heads)
        {
            return Err(candle_core::Error::Msg(format!(
                "num_attention_heads {} not divisible by num_key_value_heads {}",
                cfg.num_attention_heads, cfg.num_key_value_heads
            )));
        }

        // Why: FullFineTune ignores LoRA targets entirely — no adapters
        // are constructed. LoraOnly and Qlora both honor the user's
        // target list; Qlora additionally quantizes the base of each
        // target linear (the maybe_lora_linear branch keys on
        // `loader.mode` to pick the wrapper type).
        let targets: HashSet<String> = match mode {
            TrainMode::LoraOnly | TrainMode::Qlora { .. } => {
                lora_cfg.target_modules.iter().cloned().collect()
            }
            TrainMode::FullFineTune => HashSet::new(),
        };

        if mode == TrainMode::FullFineTune {
            assert!(
                train_varmap.is_some(),
                "FullFineTune requires train_varmap; got None",
            );
        }

        let loader = BaseLoader { mode, train_varmap };

        let embed_tokens = build_embedding(
            cfg.vocab_size,
            cfg.hidden_size,
            base_vb.pp("model.embed_tokens"),
            &loader,
            "model.embed_tokens",
        )?;

        let lm_head = if cfg.tie_word_embeddings {
            // Why: PEFT-style tying — lm_head shares the embedding weight
            // matrix. No bias on Llama lm_head. In FFT mode the embedding
            // tensor was already registered in the train varmap via
            // `build_embedding`; reusing `embeddings().clone()` here keeps
            // it a single entry so gradients flow back to one Var only
            // (preventing the double-registration footgun).
            Linear::new(embed_tokens.embeddings().clone(), None)
        } else {
            loader.linear_no_bias(
                cfg.hidden_size,
                cfg.vocab_size,
                base_vb.pp("lm_head"),
                "lm_head",
            )?
        };

        let norm = build_rms_norm(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            base_vb.pp("model.norm"),
            &loader,
            "model.norm",
        )?;

        let layers_vb_base = base_vb.pp("model.layers");
        let layers_vb_lora = lora_vb.pp("model.layers");
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for i in 0..cfg.num_hidden_layers {
            let layer = DecoderLayer::load(
                layers_vb_base.pp(i.to_string()),
                layers_vb_lora.pp(i.to_string()),
                cfg,
                lora_cfg,
                &targets,
                &loader,
                &format!("model.layers.{i}"),
            )?;
            layers.push(layer);
        }

        let rope = build_rope_cache(cfg, base_vb.dtype(), base_vb.device())?;

        Ok(Self {
            embed_tokens,
            layers,
            norm,
            lm_head,
            rope,
            tied: cfg.tie_word_embeddings,
        })
    }

    /// Whether `lm_head` is tied to the input embeddings.
    #[must_use]
    pub fn tied_embeddings(&self) -> bool {
        self.tied
    }

    /// Forward pass — returns per-token logits `[B, T, vocab_size]`.
    ///
    /// `input_ids` is a `[B, T]` `u32` tensor of token IDs.
    ///
    /// # Errors
    ///
    /// Forwards any candle tensor error.
    pub fn forward(&self, input_ids: &Tensor) -> Result<Tensor> {
        let (_b, seq_len) = input_ids.dims2()?;
        let mut x = self.embed_tokens.forward(input_ids)?;
        let mask = build_causal_mask(seq_len, x.device())?;
        for layer in &self.layers {
            x = layer.forward(&x, &self.rope, &mask)?;
        }
        // Why: see DecoderLayer::forward — use `forward_diff` to bypass
        // the no-bwd fast-path RmsNorm.
        let x = self.norm.forward_diff(&x)?;
        // Why: training loss needs logits at every position (next-token
        // shift happens in the loss), so don't slice the last token like
        // the upstream inference forward does.
        let logits = self.lm_head.forward(&x)?;
        logits.to_dtype(DType::F32)
    }

    /// Forward returning *only* the final-position logits `[B, vocab]`,
    /// matching the upstream inference convention.
    ///
    /// Provided for tests that want to compare against the upstream
    /// model's per-step output shape.
    ///
    /// # Errors
    ///
    /// Forwards any candle tensor error.
    pub fn forward_last(&self, input_ids: &Tensor) -> Result<Tensor> {
        let logits = self.forward(input_ids)?;
        let (_b, seq_len, _v) = logits.dims3()?;
        logits.i((.., seq_len - 1, ..))?.contiguous()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};
    use candle_nn::{Init, VarMap};

    fn tiny_llama_config(num_kv_heads: usize) -> Config {
        // Why: 2-layer, 32-dim, 4-head Llama; runs in milliseconds on CPU.
        // `num_kv_heads = 4` is full attention; `2` exercises the GQA
        // kv-repeat path.
        Config {
            hidden_size: 32,
            intermediate_size: 64,
            vocab_size: 128,
            num_hidden_layers: 2,
            num_attention_heads: 4,
            num_key_value_heads: num_kv_heads,
            use_flash_attn: false,
            rms_norm_eps: 1e-5,
            rope_theta: 10_000.0,
            bos_token_id: None,
            eos_token_id: None,
            rope_scaling: None,
            max_position_embeddings: 32,
            tie_word_embeddings: false,
        }
    }

    fn tiny_lora_config(targets: &[&str]) -> LoraConfig {
        LoraConfig {
            rank: 4,
            alpha: 8.0,
            dropout: 0.0,
            target_modules: targets.iter().map(|s| (*s).to_string()).collect(),
        }
    }

    /// Populate a fresh varmap with random base weights matching `cfg`,
    /// then build a `TrainableLlama` against it. Returns `(model, varmap)`.
    fn build_model(
        cfg: &Config,
        lora_cfg: &LoraConfig,
        tied: bool,
    ) -> (TrainableLlama, VarMap, Device) {
        let device = Device::Cpu;
        let mut cfg = cfg.clone();
        cfg.tie_word_embeddings = tied;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        // Why: pre-register every base var with Kaiming-uniform init via
        // `vb.get_with_hints` paths inside the loaders. VarMap auto-creates
        // any var that `vb.get()` requests, so we just call `load`.
        let lora_vb =
            VarBuilder::from_varmap(&varmap, DType::F32, &device).push_prefix("base_model.model");
        let model = TrainableLlama::load(vb, lora_vb, &cfg, lora_cfg).expect("load");
        (model, varmap, device)
    }

    #[test]
    fn forward_shape_correct() {
        let cfg = tiny_llama_config(4);
        let lora_cfg = tiny_lora_config(&["q_proj", "k_proj", "v_proj", "o_proj"]);
        let (model, _vm, device) = build_model(&cfg, &lora_cfg, false);

        let input = Tensor::from_vec(vec![1u32, 2, 3, 4, 5, 6], (2, 3), &device).expect("input");
        let logits = model.forward(&input).expect("forward");
        assert_eq!(logits.dims(), &[2, 3, cfg.vocab_size]);

        let last = model.forward_last(&input).expect("forward_last");
        assert_eq!(last.dims(), &[2, cfg.vocab_size]);
    }

    #[test]
    fn lora_target_modules_only_creates_lora_at_targets() {
        let cfg = tiny_llama_config(4);
        // Why: only q_proj is in targets — k/v/o must NOT have lora vars.
        let lora_cfg = tiny_lora_config(&["q_proj"]);
        let (_model, varmap, _device) = build_model(&cfg, &lora_cfg, false);

        let guard = varmap.data().lock().expect("varmap mutex");
        let lora_keys: Vec<&String> = guard
            .keys()
            .filter(|k| k.contains(".lora_A.weight") || k.contains(".lora_B.weight"))
            .collect();

        // 2 layers × 1 target × 2 vars (A + B) = 4 lora vars
        assert_eq!(lora_keys.len(), 4, "got: {lora_keys:?}");
        for k in &lora_keys {
            assert!(
                k.contains(".q_proj.lora_"),
                "non-q_proj lora var leaked: {k}"
            );
            assert!(
                k.starts_with("base_model.model.model.layers."),
                "non-PEFT prefix: {k}"
            );
        }
    }

    #[test]
    fn lora_target_modules_includes_mlp_when_requested() {
        let cfg = tiny_llama_config(4);
        let lora_cfg = tiny_lora_config(&["gate_proj", "up_proj", "down_proj"]);
        let (_model, varmap, _device) = build_model(&cfg, &lora_cfg, false);

        let guard = varmap.data().lock().expect("varmap mutex");
        let lora_keys: Vec<String> = guard
            .keys()
            .filter(|k| k.contains(".lora_A.weight") || k.contains(".lora_B.weight"))
            .cloned()
            .collect();

        // 2 layers × 3 mlp targets × 2 vars = 12
        assert_eq!(lora_keys.len(), 12, "got: {lora_keys:?}");
        for k in &lora_keys {
            assert!(k.contains(".mlp."), "expected mlp.* lora var, got: {k}");
        }
    }

    #[test]
    fn zero_init_lora_b_means_initial_logits_match_base_only() {
        let cfg = tiny_llama_config(4);
        let lora_cfg = tiny_lora_config(&["q_proj", "k_proj", "v_proj", "o_proj"]);

        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let lora_vb =
            VarBuilder::from_varmap(&varmap, DType::F32, &device).push_prefix("base_model.model");
        let with_lora = TrainableLlama::load(vb, lora_vb, &cfg, &lora_cfg).expect("load with-lora");

        // Why: rebuild against the SAME varmap with an empty target set so
        // the base weights are identical, but no LoraLinear wraps any
        // projection. The two forwards must match exactly because LoRA B
        // init to zero zeroes out the delta.
        let no_lora_cfg = tiny_lora_config(&[]);
        let vb2 = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let lora_vb2 =
            VarBuilder::from_varmap(&varmap, DType::F32, &device).push_prefix("base_model.model");
        let no_lora =
            TrainableLlama::load(vb2, lora_vb2, &cfg, &no_lora_cfg).expect("load no-lora");

        let input = Tensor::from_vec(vec![1u32, 2, 3, 4], (1, 4), &device).expect("input");
        let a = with_lora.forward(&input).expect("with-lora forward");
        let b = no_lora.forward(&input).expect("no-lora forward");

        let diff = (&a - &b).expect("diff");
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
            "LoRA-B zero init must give identical logits; got max-abs delta {max_abs}",
        );
    }

    #[test]
    fn gqa_kv_repetition_correct() {
        // Why: num_attention_heads = 4, num_key_value_heads = 2 → repeat
        // factor = 2. The forward must produce a sensible logits tensor
        // (right shape, no NaNs) under GQA.
        let cfg = tiny_llama_config(2);
        let lora_cfg = tiny_lora_config(&["q_proj", "k_proj", "v_proj", "o_proj"]);
        let (model, _vm, device) = build_model(&cfg, &lora_cfg, false);

        let input = Tensor::from_vec(vec![5u32, 6, 7, 8], (1, 4), &device).expect("input");
        let logits = model.forward(&input).expect("gqa forward");
        assert_eq!(logits.dims(), &[1, 4, cfg.vocab_size]);

        // Why: smoke-check no NaNs in the output (would indicate a botched
        // repeat_kv or shape mismatch in the attention path).
        let vals: Vec<f32> = logits
            .flatten_all()
            .expect("flatten")
            .to_vec1()
            .expect("vec1");
        assert!(
            vals.iter().all(|v| v.is_finite()),
            "GQA forward produced non-finite logits",
        );
    }

    #[test]
    fn tied_word_embeddings_honored_and_reflected() {
        let cfg = tiny_llama_config(4);
        let lora_cfg = tiny_lora_config(&["q_proj"]);

        // Untied: lm_head.weight is a separately-registered var.
        let (untied_model, untied_vm, _device) = build_model(&cfg, &lora_cfg, false);
        assert!(!untied_model.tied_embeddings());
        let untied_guard = untied_vm.data().lock().expect("vm");
        assert!(
            untied_guard.contains_key("lm_head.weight"),
            "untied path must register lm_head.weight",
        );
        drop(untied_guard);

        // Tied: lm_head shares embed_tokens.weight, no separate var.
        let (tied_model, tied_vm, _device) = build_model(&cfg, &lora_cfg, true);
        assert!(tied_model.tied_embeddings());
        let tied_guard = tied_vm.data().lock().expect("vm");
        assert!(
            !tied_guard.contains_key("lm_head.weight"),
            "tied path must NOT register a separate lm_head.weight",
        );
        assert!(
            tied_guard.contains_key("model.embed_tokens.weight"),
            "embed_tokens.weight always registered",
        );
    }

    #[test]
    fn forward_runs_with_llama3_rope_scaling() {
        // Why: exercise the Llama3 rope_scaling code path so the smooth
        // interpolation isn't silently broken by a refactor.
        let mut cfg = tiny_llama_config(4);
        cfg.rope_scaling = Some(Llama3RopeConfig {
            factor: 8.0,
            low_freq_factor: 1.0,
            high_freq_factor: 4.0,
            original_max_position_embeddings: 16,
            rope_type: Llama3RopeType::Llama3,
        });

        let lora_cfg = tiny_lora_config(&["q_proj"]);
        let (model, _vm, device) = build_model(&cfg, &lora_cfg, false);

        let input = Tensor::from_vec(vec![1u32, 2, 3], (1, 3), &device).expect("input");
        let logits = model.forward(&input).expect("forward");
        assert_eq!(logits.dims(), &[1, 3, cfg.vocab_size]);
    }

    #[test]
    fn unused_init_import_is_silenced() {
        // Why: keep `Init` referenced in tests so a future refactor that
        // hand-initializes vars (e.g. to make a determinism test) doesn't
        // have to re-add the import.
        let _ = Init::Const(0.0);
    }

    // -----------------------------------------------------------------
    // FullFineTune mode tests (Wave 8b)
    // -----------------------------------------------------------------

    /// Build a Llama in `FullFineTune` mode, returning the train VarMap
    /// that received every base weight as a fresh `Var`.
    ///
    /// `base_map` is provided by the caller so two builds can share a
    /// source-of-truth set of base weights (used by the
    /// `shape_matches_lora_mode` test).
    fn build_full_finetune(
        base_map: &VarMap,
        cfg: &Config,
    ) -> (TrainableLlama, VarMap, VarMap, Device) {
        let device = Device::Cpu;
        let lora_cfg = LoraConfig {
            rank: 0,
            alpha: 0.0,
            dropout: 0.0,
            target_modules: vec![],
        };
        let lora_map = VarMap::new();
        let train_map = VarMap::new();
        let base_vb = VarBuilder::from_varmap(base_map, DType::F32, &device);
        let lora_vb =
            VarBuilder::from_varmap(&lora_map, DType::F32, &device).push_prefix("base_model.model");

        let model = TrainableLlama::load_with_mode(
            base_vb,
            lora_vb,
            Some(&train_map),
            cfg,
            &lora_cfg,
            TrainMode::FullFineTune,
        )
        .expect("full-FT model loads");

        (model, lora_map, train_map, device)
    }

    /// Expected absolute key list (every trainable `.weight`) for the
    /// `tiny_llama_config()` model. Mirrors the safetensors naming
    /// convention so `VarMap::save` output round-trips through the
    /// inference-side loader.
    ///
    /// Llama has NO biases on q/k/v/o projections (unlike Qwen2).
    /// `tie_word_embeddings` controls whether `lm_head.weight` appears
    /// separately (untied) or is omitted entirely (tied).
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
        if !cfg.tie_word_embeddings {
            keys.push("lm_head.weight".to_string());
        }
        keys
    }

    #[test]
    fn llama_full_finetune_loads_base_into_varmap() {
        let cfg = tiny_llama_config(4);
        let base_map = VarMap::new();
        let (_model, _lora, train_map, _dev) = build_full_finetune(&base_map, &cfg);

        let guard = train_map
            .data()
            .lock()
            .expect("train varmap mutex poisoned by another thread");
        let keys: HashSet<String> = guard.keys().cloned().collect();
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
    fn llama_full_finetune_forward_shape_matches_lora_mode() {
        let device = Device::Cpu;
        let cfg = tiny_llama_config(4);

        // Build the LoraOnly reference with NO LoRA targets so it's a
        // base-only forward — values will match FullFineTune at step 0
        // because the FFT path copies the same base weights into Vars
        // without modifying them.
        let no_lora_cfg = LoraConfig {
            rank: 4,
            alpha: 8.0,
            dropout: 0.0,
            target_modules: vec![],
        };
        let base_map = VarMap::new();
        let lora_map_ref = VarMap::new();
        let base_vb_ref = VarBuilder::from_varmap(&base_map, DType::F32, &device);
        let lora_vb_ref = VarBuilder::from_varmap(&lora_map_ref, DType::F32, &device)
            .push_prefix("base_model.model");
        let lora_only =
            TrainableLlama::load(base_vb_ref, lora_vb_ref, &cfg, &no_lora_cfg).expect("lora load");

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
    fn llama_full_finetune_every_linear_has_var() {
        let cfg = tiny_llama_config(4);
        let base_map = VarMap::new();
        let (_model, _lora, train_map, _dev) = build_full_finetune(&base_map, &cfg);

        let guard = train_map
            .data()
            .lock()
            .expect("train varmap mutex poisoned by another thread");
        let actual: HashSet<String> = guard.keys().cloned().collect();
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
        let expected_set: HashSet<&String> = expected.iter().collect();
        for key in &actual {
            assert!(
                expected_set.contains(key),
                "FullFineTune train varmap has unexpected key {key}; expected one of: {expected:?}",
            );
        }
    }

    #[test]
    fn llama_full_finetune_does_not_double_register_tied_embedding() {
        // Why: when `tie_word_embeddings` is true, lm_head shares the
        // embedding weight matrix. In FFT mode we must register that
        // tensor exactly once (under `model.embed_tokens.weight`); a
        // separate `lm_head.weight` entry would be a correctness bug —
        // optimizer steps would apply gradients to two independent Vars
        // and tying would silently break.
        let mut cfg = tiny_llama_config(4);
        cfg.tie_word_embeddings = true;

        let base_map = VarMap::new();
        let (model, _lora, train_map, _dev) = build_full_finetune(&base_map, &cfg);
        assert!(model.tied_embeddings(), "model reports tied");

        let guard = train_map
            .data()
            .lock()
            .expect("train varmap mutex poisoned by another thread");
        let keys: HashSet<String> = guard.keys().cloned().collect();
        drop(guard);

        assert!(
            keys.contains("model.embed_tokens.weight"),
            "tied FFT must register embed_tokens.weight; have: {keys:?}",
        );
        assert!(
            !keys.contains("lm_head.weight"),
            "tied FFT must NOT register a separate lm_head.weight; have: {keys:?}",
        );
    }
}
