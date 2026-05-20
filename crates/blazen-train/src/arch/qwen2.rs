// Why: VarBuilder is a small handle; passing by value matches the upstream
// candle-transformers builder ergonomics and avoids reference plumbing through
// every nested layer.
#![allow(clippy::needless_pass_by_value)]

//! Trainable Qwen2 wrapper for LoRA fine-tuning.
//!
//! Mirrors `candle_transformers::models::qwen2::Model`'s forward pass but
//! substitutes [`crate::lora::LoraLinear`] at every site listed in
//! [`crate::config::LoraConfig::target_modules`]. Non-target projections
//! are loaded as plain frozen [`Linear`]s from the base safetensors; LoRA
//! A/B params are registered against a separate [`VarBuilder`] backed by
//! the caller's [`candle_nn::VarMap`] so only those slots receive gradients.
//!
//! No KV cache: training drives full-sequence forwards.
//!
//! Param-name layout (CRITICAL for PEFT round-trip): the LoRA `VarBuilder`
//! is pushed under `base_model.model.<module_path>` so the registered keys
//! match `crates/blazen-llm-candle/src/lora.rs::build_layers` after its
//! `base_model.model.` strip — adapters trained here load back via that
//! inference-side loader unchanged.

use std::collections::HashSet;
use std::sync::Arc;

use candle_core::{DType, Device, IndexOp, Module, Result, Tensor};
use candle_nn::{Embedding, Linear, RmsNorm, VarBuilder, VarMap};

use crate::arch::{BaseLoader, TrainMode, build_embedding, build_rms_norm};
use crate::config::LoraConfig;
use crate::lora::LoraLinear;

pub use candle_transformers::models::qwen2::Config;

/// A Linear projection that is either frozen (base only) or LoRA-augmented.
pub enum MaybeLora {
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

impl MaybeLora {
    /// Wrap `base` with LoRA if `module_name` is in `targets`, otherwise
    /// keep it frozen. `lora_vb` should already be scoped to the parent
    /// module path (so the registered LoRA params land at
    /// `<parent>.lora_A.weight` / `<parent>.lora_B.weight`).
    // Why: VarBuilder is a cheap Arc-cloning handle; candle's idiomatic call
    // sites move freshly `pp`-scoped builders in, mirroring `candle_nn::linear`.
    #[allow(clippy::needless_pass_by_value)]
    fn build(
        module_name: &str,
        targets: &HashSet<String>,
        base: Linear,
        in_dim: usize,
        out_dim: usize,
        lora_cfg: &LoraConfig,
        lora_vb: VarBuilder,
    ) -> Result<Self> {
        if targets.contains(module_name) {
            let layer = LoraLinear::wrap(
                base,
                in_dim,
                out_dim,
                lora_cfg.rank,
                lora_cfg.alpha,
                lora_vb,
            )?;
            Ok(Self::Lora(layer))
        } else {
            Ok(Self::Frozen(base))
        }
    }

    #[must_use]
    pub fn is_lora(&self) -> bool {
        matches!(self, Self::Lora(_))
    }
}

#[derive(Debug, Clone)]
struct RotaryEmbedding {
    sin: Tensor,
    cos: Tensor,
}

impl RotaryEmbedding {
    fn new(dtype: DType, cfg: &Config, dev: &Device) -> Result<Self> {
        let dim = cfg.hidden_size / cfg.num_attention_heads;
        let max_seq_len = cfg.max_position_embeddings;
        #[allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)]
        let inv_freq: Vec<f32> = (0..dim)
            .step_by(2)
            .map(|i| 1f32 / cfg.rope_theta.powf(i as f64 / dim as f64) as f32)
            .collect();
        let inv_freq_len = inv_freq.len();
        let inv_freq = Tensor::from_vec(inv_freq, (1, inv_freq_len), dev)?.to_dtype(dtype)?;
        #[allow(clippy::cast_possible_truncation)]
        let t = Tensor::arange(0u32, max_seq_len as u32, dev)?
            .to_dtype(dtype)?
            .reshape((max_seq_len, 1))?;
        let freqs = t.matmul(&inv_freq)?;
        Ok(Self {
            sin: freqs.sin()?,
            cos: freqs.cos()?,
        })
    }

    fn apply_rotary_emb_qkv(&self, q: &Tensor, k: &Tensor) -> Result<(Tensor, Tensor)> {
        let (_b, _h, seq_len, _d) = q.dims4()?;
        let cos = self.cos.narrow(0, 0, seq_len)?;
        let sin = self.sin.narrow(0, 0, seq_len)?;
        // Why: `candle_nn::rotary_emb::rope` is a no-bwd CustomOp3 and
        // would sever the autograd graph at attention. `rope_slow` is the
        // primitive-op equivalent that backpropagates.
        let q_embed = candle_nn::rotary_emb::rope_slow(&q.contiguous()?, &cos, &sin)?;
        let k_embed = candle_nn::rotary_emb::rope_slow(&k.contiguous()?, &cos, &sin)?;
        Ok((q_embed, k_embed))
    }
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
    hidden_size: usize,
    rotary_emb: Arc<RotaryEmbedding>,
}

impl Attention {
    // Why: every argument is load-bearing — rotary table, arch cfg, LoRA
    // cfg + target set, two VarBuilders (frozen vs trainable scopes),
    // mode-aware loader, and the absolute key prefix for FFT varmap
    // insertion. Bundling into a single struct would just move the line
    // count without clarifying anything.
    #[allow(clippy::too_many_arguments)]
    fn new(
        rotary_emb: Arc<RotaryEmbedding>,
        cfg: &Config,
        lora_cfg: &LoraConfig,
        targets: &HashSet<String>,
        base_vb: VarBuilder,
        lora_vb: VarBuilder,
        loader: &BaseLoader<'_>,
        abs_path: &str,
    ) -> Result<Self> {
        let hidden_sz = cfg.hidden_size;
        let num_heads = cfg.num_attention_heads;
        let num_kv_heads = cfg.num_key_value_heads;
        let num_kv_groups = num_heads / num_kv_heads;
        let head_dim = hidden_sz / num_heads;

        let q_dim = num_heads * head_dim;
        let kv_dim = num_kv_heads * head_dim;

        let q_base = loader.linear(
            hidden_sz,
            q_dim,
            base_vb.pp("q_proj"),
            &format!("{abs_path}.q_proj"),
        )?;
        let k_base = loader.linear(
            hidden_sz,
            kv_dim,
            base_vb.pp("k_proj"),
            &format!("{abs_path}.k_proj"),
        )?;
        let v_base = loader.linear(
            hidden_sz,
            kv_dim,
            base_vb.pp("v_proj"),
            &format!("{abs_path}.v_proj"),
        )?;
        let o_base = loader.linear_no_bias(
            q_dim,
            hidden_sz,
            base_vb.pp("o_proj"),
            &format!("{abs_path}.o_proj"),
        )?;

        let q_proj = MaybeLora::build(
            "q_proj",
            targets,
            q_base,
            hidden_sz,
            q_dim,
            lora_cfg,
            lora_vb.pp("q_proj"),
        )?;
        let k_proj = MaybeLora::build(
            "k_proj",
            targets,
            k_base,
            hidden_sz,
            kv_dim,
            lora_cfg,
            lora_vb.pp("k_proj"),
        )?;
        let v_proj = MaybeLora::build(
            "v_proj",
            targets,
            v_base,
            hidden_sz,
            kv_dim,
            lora_cfg,
            lora_vb.pp("v_proj"),
        )?;
        let o_proj = MaybeLora::build(
            "o_proj",
            targets,
            o_base,
            q_dim,
            hidden_sz,
            lora_cfg,
            lora_vb.pp("o_proj"),
        )?;

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            num_heads,
            num_kv_heads,
            num_kv_groups,
            head_dim,
            hidden_size: hidden_sz,
            rotary_emb,
        })
    }

    fn forward(&self, xs: &Tensor, attention_mask: Option<&Tensor>) -> Result<Tensor> {
        let (b_sz, q_len, _) = xs.dims3()?;

        let q = self.q_proj.forward(xs)?;
        let k = self.k_proj.forward(xs)?;
        let v = self.v_proj.forward(xs)?;

        let q = q
            .reshape((b_sz, q_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        let k = k
            .reshape((b_sz, q_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;
        let v = v
            .reshape((b_sz, q_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;

        let (q, k) = self.rotary_emb.apply_rotary_emb_qkv(&q, &k)?;

        let k = candle_transformers::utils::repeat_kv(k, self.num_kv_groups)?.contiguous()?;
        let v = candle_transformers::utils::repeat_kv(v, self.num_kv_groups)?.contiguous()?;

        #[allow(clippy::cast_precision_loss)]
        let scale = 1f64 / f64::sqrt(self.head_dim as f64);
        let attn_weights = (q.matmul(&k.transpose(2, 3)?)? * scale)?;
        let attn_weights = match attention_mask {
            None => attn_weights,
            Some(mask) => attn_weights.broadcast_add(mask)?,
        };
        // Why: candle's `softmax_last_dim` is a CustomOp1 marked no-bwd, so it
        // severs the autograd graph at attention. Training requires the
        // general `softmax` (built from primitive ops) which propagates.
        let attn_weights = candle_nn::ops::softmax(&attn_weights, candle_core::D::Minus1)?;
        let attn_output = attn_weights.matmul(&v)?;

        attn_output
            .transpose(1, 2)?
            .reshape((b_sz, q_len, self.hidden_size))?
            .apply(&self.o_proj)
    }
}

struct Mlp {
    gate_proj: MaybeLora,
    up_proj: MaybeLora,
    down_proj: MaybeLora,
    act_fn: candle_nn::Activation,
}

impl Mlp {
    fn new(
        cfg: &Config,
        lora_cfg: &LoraConfig,
        targets: &HashSet<String>,
        base_vb: VarBuilder,
        lora_vb: VarBuilder,
        loader: &BaseLoader<'_>,
        abs_path: &str,
    ) -> Result<Self> {
        let hidden_sz = cfg.hidden_size;
        let intermediate_sz = cfg.intermediate_size;

        let gate_base = loader.linear_no_bias(
            hidden_sz,
            intermediate_sz,
            base_vb.pp("gate_proj"),
            &format!("{abs_path}.gate_proj"),
        )?;
        let up_base = loader.linear_no_bias(
            hidden_sz,
            intermediate_sz,
            base_vb.pp("up_proj"),
            &format!("{abs_path}.up_proj"),
        )?;
        let down_base = loader.linear_no_bias(
            intermediate_sz,
            hidden_sz,
            base_vb.pp("down_proj"),
            &format!("{abs_path}.down_proj"),
        )?;

        let gate_proj = MaybeLora::build(
            "gate_proj",
            targets,
            gate_base,
            hidden_sz,
            intermediate_sz,
            lora_cfg,
            lora_vb.pp("gate_proj"),
        )?;
        let up_proj = MaybeLora::build(
            "up_proj",
            targets,
            up_base,
            hidden_sz,
            intermediate_sz,
            lora_cfg,
            lora_vb.pp("up_proj"),
        )?;
        let down_proj = MaybeLora::build(
            "down_proj",
            targets,
            down_base,
            intermediate_sz,
            hidden_sz,
            lora_cfg,
            lora_vb.pp("down_proj"),
        )?;

        Ok(Self {
            gate_proj,
            up_proj,
            down_proj,
            act_fn: cfg.hidden_act,
        })
    }
}

impl Module for Mlp {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let lhs = xs.apply(&self.gate_proj)?.apply(&self.act_fn)?;
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
    // Why: see Attention::new — same argument set composes the per-layer
    // building blocks (attn + mlp + 2 layernorms) and there's no natural
    // smaller grouping.
    #[allow(clippy::too_many_arguments)]
    fn new(
        rotary_emb: Arc<RotaryEmbedding>,
        cfg: &Config,
        lora_cfg: &LoraConfig,
        targets: &HashSet<String>,
        base_vb: VarBuilder,
        lora_vb: VarBuilder,
        loader: &BaseLoader<'_>,
        abs_path: &str,
    ) -> Result<Self> {
        let self_attn = Attention::new(
            rotary_emb,
            cfg,
            lora_cfg,
            targets,
            base_vb.pp("self_attn"),
            lora_vb.pp("self_attn"),
            loader,
            &format!("{abs_path}.self_attn"),
        )?;
        let mlp = Mlp::new(
            cfg,
            lora_cfg,
            targets,
            base_vb.pp("mlp"),
            lora_vb.pp("mlp"),
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

/// Trainable Qwen2 causal-LM. Holds frozen base weights plus a set of
/// LoRA A/B matrices at the modules named in [`LoraConfig::target_modules`].
pub struct TrainableQwen2 {
    embed_tokens: Embedding,
    layers: Vec<DecoderLayer>,
    norm: RmsNorm,
    lm_head: Linear,
    device: Device,
    dtype: DType,
}

impl TrainableQwen2 {
    /// Build the trainable Qwen2 wrapper.
    ///
    /// `base_vb` reads the frozen base weights (typically from a
    /// safetensors mmap via `VarBuilder::from_mmaped_safetensors`).
    /// `lora_vb` is the [`VarBuilder::from_varmap`]-style builder whose
    /// underlying [`candle_nn::VarMap`] will receive the trainable LoRA
    /// params at PEFT-canonical keys
    /// (`base_model.model.<module_path>.lora_A.weight` etc.).
    /// `lora_cfg.target_modules` selects which projections become LoRA.
    ///
    /// # Errors
    ///
    /// Propagates any `candle` error from weight loading or tensor
    /// construction.
    pub fn load(
        base_vb: VarBuilder,
        lora_vb: VarBuilder,
        cfg: &Config,
        lora_cfg: &LoraConfig,
    ) -> Result<Self> {
        Self::load_with_mode(base_vb, lora_vb, None, cfg, lora_cfg, TrainMode::LoraOnly)
    }

    /// Build the trainable Qwen2 wrapper with an explicit [`TrainMode`].
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
    /// # Errors
    ///
    /// Propagates any `candle` error from weight loading or tensor
    /// construction.
    ///
    /// # Panics
    ///
    /// Panics if `mode == TrainMode::FullFineTune` and `train_varmap` is
    /// `None`. This is a programming error, not a runtime input failure.
    pub fn load_with_mode(
        base_vb: VarBuilder,
        lora_vb: VarBuilder,
        train_varmap: Option<&VarMap>,
        cfg: &Config,
        lora_cfg: &LoraConfig,
        mode: TrainMode,
    ) -> Result<Self> {
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

        let device = base_vb.device().clone();
        let dtype = base_vb.dtype();

        let vb_m = base_vb.pp("model");
        let embed_tokens = build_embedding(
            cfg.vocab_size,
            cfg.hidden_size,
            vb_m.pp("embed_tokens"),
            &loader,
            "model.embed_tokens",
        )?;

        let rotary_emb = Arc::new(RotaryEmbedding::new(dtype, cfg, &device)?);

        // Why: the LoRA varmap must mirror PEFT's `base_model.model.` wrapping
        // so trained tensors round-trip through `LoadedAdapter::from_dir`.
        // In FullFineTune the LoRA varmap goes unused but we keep the
        // pp chain to avoid divergent control flow.
        let lora_vb_m = lora_vb.pp("base_model").pp("model").pp("model");

        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        let vb_l = vb_m.pp("layers");
        let lora_vb_l = lora_vb_m.pp("layers");
        for layer_idx in 0..cfg.num_hidden_layers {
            let layer = DecoderLayer::new(
                rotary_emb.clone(),
                cfg,
                lora_cfg,
                &targets,
                vb_l.pp(layer_idx),
                lora_vb_l.pp(layer_idx),
                &loader,
                &format!("model.layers.{layer_idx}"),
            )?;
            layers.push(layer);
        }

        let norm = build_rms_norm(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            vb_m.pp("norm"),
            &loader,
            "model.norm",
        )?;

        let lm_head = if cfg.tie_word_embeddings {
            Linear::new(embed_tokens.embeddings().clone(), None)
        } else {
            loader.linear_no_bias(
                cfg.hidden_size,
                cfg.vocab_size,
                base_vb.pp("lm_head"),
                "lm_head",
            )?
        };

        Ok(Self {
            embed_tokens,
            layers,
            norm,
            lm_head,
            device,
            dtype,
        })
    }

    /// Full-sequence forward for training. Returns logits of shape
    /// `(batch, seq_len, vocab_size)`.
    ///
    /// # Errors
    ///
    /// Propagates any `candle` tensor error.
    pub fn forward(&self, input_ids: &Tensor) -> Result<Tensor> {
        let (b_size, seq_len) = input_ids.dims2()?;
        let attention_mask = if seq_len <= 1 {
            None
        } else {
            Some(self.causal_mask(b_size, seq_len)?)
        };

        let mut xs = self.embed_tokens.forward(input_ids)?;
        for layer in &self.layers {
            xs = layer.forward(&xs, attention_mask.as_ref())?;
        }
        // Why: see DecoderLayer::forward — use `forward_diff` to bypass
        // the no-bwd fast-path RmsNorm.
        self.norm.forward_diff(&xs)?.apply(&self.lm_head)
    }

    fn causal_mask(&self, b_size: usize, tgt_len: usize) -> Result<Tensor> {
        let mask: Vec<f32> = (0..tgt_len)
            .flat_map(|i| (0..tgt_len).map(move |j| if i < j { f32::NEG_INFINITY } else { 0.0 }))
            .collect();
        let mask = Tensor::from_slice(&mask, (tgt_len, tgt_len), &self.device)?;
        mask.expand((b_size, 1, tgt_len, tgt_len))?
            .to_dtype(self.dtype)
    }

    /// Number of decoder layers (test/debug helper).
    #[must_use]
    pub fn num_layers(&self) -> usize {
        self.layers.len()
    }

    /// Reference to a per-layer attention block (test/debug helper).
    #[must_use]
    pub fn attention(&self, layer_idx: usize) -> Option<AttentionView<'_>> {
        self.layers.get(layer_idx).map(|l| AttentionView {
            q_proj: &l.self_attn.q_proj,
            k_proj: &l.self_attn.k_proj,
            v_proj: &l.self_attn.v_proj,
            o_proj: &l.self_attn.o_proj,
        })
    }

    /// Forward path that takes a hidden-state seed instead of token ids.
    /// Test helper for verifying base-only vs LoRA-on-zero-init parity
    /// without having to keep two embedding tables in sync.
    ///
    /// # Errors
    ///
    /// Propagates any `candle` tensor error.
    pub fn forward_from_hidden(&self, hidden: &Tensor) -> Result<Tensor> {
        let (b_size, seq_len, _) = hidden.dims3()?;
        let attention_mask = if seq_len <= 1 {
            None
        } else {
            Some(self.causal_mask(b_size, seq_len)?)
        };
        let mut xs = hidden.clone();
        for layer in &self.layers {
            xs = layer.forward(&xs, attention_mask.as_ref())?;
        }
        // Why: see DecoderLayer::forward — use `forward_diff` to bypass
        // the no-bwd fast-path RmsNorm.
        self.norm.forward_diff(&xs)?.apply(&self.lm_head)
    }

    /// Drop the [`IndexOp`]-imported logits at the last token only.
    /// Surfaced for consumers that want greedy-style scoring without
    /// holding the whole sequence in memory.
    ///
    /// # Errors
    ///
    /// Propagates any `candle` tensor error.
    pub fn last_token_logits(&self, input_ids: &Tensor) -> Result<Tensor> {
        let logits = self.forward(input_ids)?;
        let (_, seq_len, _) = logits.dims3()?;
        logits.i((.., seq_len - 1, ..))
    }
}

/// Read-only handle to one layer's four attention projections, used by
/// the unit tests to assert which became [`MaybeLora::Lora`] vs
/// [`MaybeLora::Frozen`].
pub struct AttentionView<'a> {
    pub q_proj: &'a MaybeLora,
    pub k_proj: &'a MaybeLora,
    pub v_proj: &'a MaybeLora,
    pub o_proj: &'a MaybeLora,
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_nn::{Activation, VarMap};

    fn tiny_qwen2_config() -> Config {
        Config {
            vocab_size: 128,
            hidden_size: 32,
            intermediate_size: 64,
            num_hidden_layers: 2,
            num_attention_heads: 4,
            num_key_value_heads: 2,
            max_position_embeddings: 64,
            sliding_window: 64,
            max_window_layers: 2,
            tie_word_embeddings: false,
            rope_theta: 10000.0,
            rms_norm_eps: 1e-6,
            use_sliding_window: false,
            hidden_act: Activation::Silu,
        }
    }

    fn build_with_targets(targets: &[&str]) -> (TrainableQwen2, VarMap, VarMap, Config) {
        let device = Device::Cpu;
        let cfg = tiny_qwen2_config();
        let lora_cfg = LoraConfig {
            rank: 4,
            alpha: 8.0,
            dropout: 0.0,
            target_modules: targets.iter().map(|s| (*s).to_string()).collect(),
        };

        let base_map = VarMap::new();
        let lora_map = VarMap::new();
        let base_vb = VarBuilder::from_varmap(&base_map, DType::F32, &device);
        let lora_vb = VarBuilder::from_varmap(&lora_map, DType::F32, &device);

        let model = TrainableQwen2::load(base_vb, lora_vb, &cfg, &lora_cfg).expect("model loads");
        (model, base_map, lora_map, cfg)
    }

    #[test]
    fn forward_shape_correct() {
        let (model, _base, _lora, cfg) = build_with_targets(&["q_proj", "v_proj"]);
        let device = Device::Cpu;

        let batch = 1;
        let seq_len = 16;
        let input_ids = Tensor::zeros((batch, seq_len), DType::U32, &device).expect("ids");

        let logits = model.forward(&input_ids).expect("forward");
        assert_eq!(logits.dims(), &[batch, seq_len, cfg.vocab_size]);
    }

    #[test]
    fn lora_target_modules_only_creates_lora_at_targets() {
        let (model, _base, _lora, _cfg) = build_with_targets(&["q_proj"]);
        let view = model.attention(0).expect("layer 0 exists");
        assert!(view.q_proj.is_lora(), "q_proj should be Lora");
        assert!(!view.k_proj.is_lora(), "k_proj should be Frozen");
        assert!(!view.v_proj.is_lora(), "v_proj should be Frozen");
        assert!(!view.o_proj.is_lora(), "o_proj should be Frozen");
    }

    #[test]
    fn lora_param_names_are_peft_canonical() {
        let (_model, _base, lora_map, _cfg) = build_with_targets(&["q_proj", "v_proj"]);
        let guard = lora_map
            .data()
            .lock()
            .expect("varmap mutex poisoned by another thread");
        let keys: Vec<&String> = guard.keys().collect();

        let want_keys = [
            "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight",
            "base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight",
            "base_model.model.model.layers.0.self_attn.v_proj.lora_A.weight",
            "base_model.model.model.layers.0.self_attn.v_proj.lora_B.weight",
        ];
        for want in want_keys {
            let want_owned = want.to_string();
            assert!(
                keys.contains(&&want_owned),
                "missing PEFT key {want}; have {keys:?}"
            );
        }

        for k in &keys {
            assert!(
                k.starts_with("base_model.model.model."),
                "non-PEFT-prefixed LoRA key: {k}"
            );
            assert!(
                k.ends_with(".lora_A.weight") || k.ends_with(".lora_B.weight"),
                "non-LoRA suffix in LoRA varmap: {k}"
            );
        }
    }

    #[test]
    fn zero_init_lora_b_means_initial_logits_match_base_only() {
        let device = Device::Cpu;
        let cfg = tiny_qwen2_config();

        // Why: build the no-LoRA reference first, then build a LoRA-on
        // model that shares the *same* base VarMap so the frozen weights
        // are bit-identical and any output delta is attributable to LoRA.
        let base_map = VarMap::new();
        let base_vb_ref = VarBuilder::from_varmap(&base_map, DType::F32, &device);
        let no_lora_cfg = LoraConfig {
            rank: 4,
            alpha: 8.0,
            dropout: 0.0,
            target_modules: vec![],
        };
        let reference_lora_map = VarMap::new();
        let reference_lora_vb = VarBuilder::from_varmap(&reference_lora_map, DType::F32, &device);
        let reference = TrainableQwen2::load(base_vb_ref, reference_lora_vb, &cfg, &no_lora_cfg)
            .expect("reference loads");

        let lora_cfg = LoraConfig {
            rank: 4,
            alpha: 8.0,
            dropout: 0.0,
            target_modules: vec![
                "q_proj".into(),
                "k_proj".into(),
                "v_proj".into(),
                "o_proj".into(),
            ],
        };
        let base_vb_lora = VarBuilder::from_varmap(&base_map, DType::F32, &device);
        let lora_map = VarMap::new();
        let lora_vb = VarBuilder::from_varmap(&lora_map, DType::F32, &device);
        let with_lora =
            TrainableQwen2::load(base_vb_lora, lora_vb, &cfg, &lora_cfg).expect("lora loads");

        let batch = 1;
        let seq_len = 8;
        let input_ids = Tensor::from_vec(
            (0u32..u32::try_from(seq_len).unwrap()).collect::<Vec<u32>>(),
            (batch, seq_len),
            &device,
        )
        .expect("ids");

        let out_ref = reference.forward(&input_ids).expect("ref forward");
        let out_lora = with_lora.forward(&input_ids).expect("lora forward");

        let diff = (&out_lora - &out_ref).expect("diff");
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
            "B-init-to-zero should make LoRA forward match base; got max-abs delta {max_abs}",
        );
    }

    #[test]
    fn last_token_logits_returns_final_position() {
        let (model, _base, _lora, cfg) = build_with_targets(&["q_proj"]);
        let device = Device::Cpu;

        let batch = 2;
        let seq_len = 4;
        let input_ids = Tensor::zeros((batch, seq_len), DType::U32, &device).expect("ids");

        let last = model.last_token_logits(&input_ids).expect("last");
        assert_eq!(last.dims(), &[batch, cfg.vocab_size]);
    }

    /// Build a Qwen2 in `FullFineTune` mode, returning the train VarMap
    /// that received every base weight as a fresh `Var`.
    ///
    /// `base_map` is provided by the caller so two builds can share a
    /// source-of-truth set of base weights (used by the
    /// `shape_matches_lora_mode` test).
    fn build_full_finetune(
        base_map: &VarMap,
        cfg: &Config,
    ) -> (TrainableQwen2, VarMap, VarMap, Device) {
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
        let lora_vb = VarBuilder::from_varmap(&lora_map, DType::F32, &device);

        let model = TrainableQwen2::load_with_mode(
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

    /// Expected absolute key list (every trainable `.weight` / `.bias`)
    /// for the `tiny_qwen2_config()` model. Mirrors the safetensors
    /// naming convention so `VarMap::save` output round-trips through
    /// the inference-side loader.
    fn expected_full_finetune_keys(cfg: &Config) -> Vec<String> {
        let mut keys = Vec::new();
        keys.push("model.embed_tokens.weight".to_string());
        for li in 0..cfg.num_hidden_layers {
            for proj in ["q_proj", "k_proj", "v_proj"] {
                keys.push(format!("model.layers.{li}.self_attn.{proj}.weight"));
                keys.push(format!("model.layers.{li}.self_attn.{proj}.bias"));
            }
            keys.push(format!("model.layers.{li}.self_attn.o_proj.weight"));
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
    fn qwen2_full_finetune_loads_base_into_varmap() {
        let cfg = tiny_qwen2_config();
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
    fn qwen2_full_finetune_forward_shape_matches_lora_mode() {
        let device = Device::Cpu;
        let cfg = tiny_qwen2_config();

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
        let lora_vb_ref = VarBuilder::from_varmap(&lora_map_ref, DType::F32, &device);
        let lora_only =
            TrainableQwen2::load(base_vb_ref, lora_vb_ref, &cfg, &no_lora_cfg).expect("lora load");

        // Reuse the same base_map for FullFineTune so the source weights
        // are bit-identical between the two models.
        let (full_ft, _lora_map, _train_map, _dev) = build_full_finetune(&base_map, &cfg);

        let batch = 1;
        let seq_len = 8;
        let input_ids = Tensor::from_vec(
            (0u32..u32::try_from(seq_len).unwrap()).collect::<Vec<u32>>(),
            (batch, seq_len),
            &device,
        )
        .expect("ids");

        let out_lora = lora_only.forward(&input_ids).expect("lora forward");
        let out_ft = full_ft.forward(&input_ids).expect("ft forward");

        // Why: shapes must match exactly — same arch wraps the same cfg.
        assert_eq!(out_ft.dims(), out_lora.dims());
        assert_eq!(out_ft.dims(), &[batch, seq_len, cfg.vocab_size]);

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
    fn qwen2_full_finetune_every_linear_has_var() {
        let cfg = tiny_qwen2_config();
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
}
