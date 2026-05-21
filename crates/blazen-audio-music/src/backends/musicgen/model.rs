//! MusicGen decoder + T5 text-encoder + EnCodec wrapper.
//!
//! Adapted from the partial reference in
//! [huggingface/candle](https://github.com/huggingface/candle)
//! `candle-examples/examples/musicgen/musicgen_model.rs` (blob SHA
//! `7fbe8b5306eb4a543f1d400b982125d20536ed49`, MIT licensed). The upstream
//! example loads the model graph but leaves `prepare_decoder_attention_mask`
//! as `todo!()` and never runs the autoregressive generation loop --
//! [`super::generation`] supplies the loop and this module supplies the
//! patched decoder.
//!
//! Key changes vs upstream:
//!
//! 1. `prepare_decoder_attention_mask` is now a real causal mask builder
//!    (pattern lifted from
//!    `candle_transformers::models::parler_tts::Model::prepare_causal_mask`).
//! 2. The decoder forward takes an explicit `encoder_hidden_states` argument
//!    so cross-attention against the T5 prompt embeddings actually runs.
//!    Upstream silently passes `None`.
//! 3. The decoder layer's cross-attention now receives an `encoder_attention_mask`
//!    of `None` (full attention over the encoder sequence) rather than
//!    reusing the causal self-attention mask, which is shape-incompatible.
//! 4. The sinusoidal embedding table is built directly on the target device.
//!
//! The candle `VarBuilder` API is consumed by value in this module
//! (matching the upstream `candle-transformers::models::parler_tts`
//! convention) -- the value is internally `Arc<Inner>`-shaped so by-value
//! consumption is cheap. We `#[allow]` the corresponding pedantic lint
//! locally where it would otherwise fire dozens of times.

#![allow(clippy::needless_pass_by_value)]

use candle_core::{D, DType, Device, IndexOp, Result, Tensor};
use candle_nn::{
    Activation, Embedding, LayerNorm, Linear, Module, VarBuilder, embedding, layer_norm,
    linear_no_bias,
};
use candle_transformers::models::{encodec, t5};

/// Decoder LM hyper-parameters for a MusicGen checkpoint.
///
/// Mirrors `transformers/src/transformers/models/musicgen/configuration_musicgen.py`.
#[derive(Debug, Clone, PartialEq)]
pub struct Config {
    /// Number of EnCodec codebook entries (per codebook). MusicGen always
    /// trains with 2048.
    pub vocab_size: usize,
    /// Maximum context length for the sinusoidal positional embedding
    /// table.
    pub max_position_embeddings: usize,
    /// Number of decoder transformer layers.
    pub num_hidden_layers: usize,
    /// Inner dim of the per-layer feed-forward block.
    pub ffn_dim: usize,
    /// Multi-head attention head count.
    pub num_attention_heads: usize,
    /// FFN activation (always GELU for the published checkpoints).
    pub activation_function: Activation,
    /// Decoder hidden / residual width.
    pub hidden_size: usize,
    /// Whether to scale embeddings by `sqrt(hidden_size)` (always false
    /// for the published checkpoints).
    pub scale_embedding: bool,
    /// Number of parallel EnCodec codebooks the decoder predicts.
    pub num_codebooks: usize,
    /// Token id used to start each codebook stream (== `vocab_size` because
    /// the embedding table is sized `vocab_size + 1`).
    pub pad_token_id: usize,
    /// Token id used as the BOS / delay-fill sentinel.
    pub bos_token_id: usize,
}

impl Config {
    /// Decoder config for `facebook/musicgen-small` (~300M params).
    #[must_use]
    pub fn musicgen_small() -> Self {
        Self {
            vocab_size: 2048,
            max_position_embeddings: 2048,
            num_hidden_layers: 24,
            ffn_dim: 4096,
            num_attention_heads: 16,
            activation_function: Activation::Gelu,
            hidden_size: 1024,
            scale_embedding: false,
            num_codebooks: 4,
            pad_token_id: 2048,
            bos_token_id: 2048,
        }
    }

    /// Decoder config for `facebook/musicgen-medium` (~1.5B params).
    #[must_use]
    pub fn musicgen_medium() -> Self {
        Self {
            vocab_size: 2048,
            max_position_embeddings: 2048,
            num_hidden_layers: 48,
            ffn_dim: 6144,
            num_attention_heads: 24,
            activation_function: Activation::Gelu,
            hidden_size: 1536,
            scale_embedding: false,
            num_codebooks: 4,
            pad_token_id: 2048,
            bos_token_id: 2048,
        }
    }

    /// Decoder config for `facebook/musicgen-large` (~3.3B params).
    #[must_use]
    pub fn musicgen_large() -> Self {
        Self {
            vocab_size: 2048,
            max_position_embeddings: 2048,
            num_hidden_layers: 48,
            ffn_dim: 8192,
            num_attention_heads: 32,
            activation_function: Activation::Gelu,
            hidden_size: 2048,
            scale_embedding: false,
            num_codebooks: 4,
            pad_token_id: 2048,
            bos_token_id: 2048,
        }
    }
}

/// Build the sinusoidal positional embedding table on `device`.
///
/// Matches the table generation in the upstream HF
/// `MusicgenSinusoidalPositionalEmbedding`.
fn get_embedding(num_embeddings: usize, embedding_dim: usize, device: &Device) -> Result<Tensor> {
    let half_dim = embedding_dim / 2;
    #[allow(clippy::cast_precision_loss)]
    let emb = f64::ln(10000.) / (half_dim - 1) as f64;
    #[allow(clippy::cast_precision_loss)]
    let xs: Vec<_> = (0..num_embeddings).map(|v| v as f32).collect();
    let xs = Tensor::from_vec(xs, (num_embeddings, 1), device)?;
    #[allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)]
    let ys: Vec<_> = (0..half_dim)
        .map(|v| f64::exp(v as f64 * -emb) as f32)
        .collect();
    let ys = Tensor::from_vec(ys, (1, half_dim), device)?;
    let shape = (num_embeddings, half_dim);
    let emb = (xs.broadcast_as(shape)? * ys.broadcast_as(shape)?)?;
    let emb =
        Tensor::cat(&[&emb.cos()?, &emb.sin()?], 1)?.reshape((num_embeddings, 2 * half_dim))?;
    let emb = if embedding_dim % 2 == 1 {
        let zeros = Tensor::zeros((num_embeddings, 1), DType::F32, device)?;
        Tensor::cat(&[&emb, &zeros], 1)?
    } else {
        emb
    };
    Ok(emb)
}

#[derive(Debug)]
struct SinusoidalPositionalEmbedding {
    embedding_dim: usize,
    weights: Tensor,
    device: Device,
}

impl SinusoidalPositionalEmbedding {
    fn load(vb: &VarBuilder, cfg: &Config) -> Result<Self> {
        let device = vb.device().clone();
        let weights = get_embedding(cfg.max_position_embeddings, cfg.hidden_size, &device)?;
        Ok(Self {
            embedding_dim: cfg.hidden_size,
            weights,
            device,
        })
    }

    /// Return positional slice `[pos_offset .. pos_offset+seq_len]`.
    fn slice(&mut self, pos_offset: usize, seq_len: usize) -> Result<Tensor> {
        let needed = pos_offset + seq_len;
        if needed > self.weights.dim(0)? {
            self.weights = get_embedding(
                needed.max(self.weights.dim(0)? * 2),
                self.embedding_dim,
                &self.device,
            )?;
        }
        self.weights.narrow(0, pos_offset, seq_len)
    }
}

#[derive(Debug)]
struct Attention {
    scaling: f64,
    num_heads: usize,
    head_dim: usize,
    k_proj: Linear,
    v_proj: Linear,
    q_proj: Linear,
    out_proj: Linear,
}

impl Attention {
    fn load(vb: VarBuilder, cfg: &Config) -> Result<Self> {
        let h = cfg.hidden_size;
        let num_heads = cfg.num_attention_heads;
        let head_dim = h / num_heads;
        let k_proj = linear_no_bias(h, h, vb.pp("k_proj"))?;
        let v_proj = linear_no_bias(h, h, vb.pp("v_proj"))?;
        let q_proj = linear_no_bias(h, h, vb.pp("q_proj"))?;
        let out_proj = linear_no_bias(h, h, vb.pp("out_proj"))?;
        Ok(Self {
            #[allow(clippy::cast_precision_loss)]
            scaling: 1. / (head_dim as f64).sqrt(),
            num_heads,
            head_dim,
            k_proj,
            v_proj,
            q_proj,
            out_proj,
        })
    }

    fn forward(
        &self,
        xs: &Tensor,
        kv_states: Option<&Tensor>,
        attention_mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        let (b_sz, tgt_len, _) = xs.dims3()?;
        let query_states = (self.q_proj.forward(xs)? * self.scaling)?;

        let kv_states = kv_states.unwrap_or(xs);
        let key_states = self.k_proj.forward(kv_states)?;
        let value_states = self.v_proj.forward(kv_states)?;
        let kv_len = key_states.dim(1)?;

        let q_shape = (b_sz, tgt_len, self.num_heads, self.head_dim);
        let kv_shape = (b_sz, kv_len, self.num_heads, self.head_dim);
        let query_states = query_states
            .reshape(q_shape)?
            .transpose(1, 2)?
            .contiguous()?;
        let key_states = key_states
            .reshape(kv_shape)?
            .transpose(1, 2)?
            .contiguous()?;
        let value_states = value_states
            .reshape(kv_shape)?
            .transpose(1, 2)?
            .contiguous()?;

        let attn_weights = query_states.matmul(&key_states.transpose(2, 3)?)?;
        let attn_weights = match attention_mask {
            None => attn_weights,
            Some(mask) => attn_weights.broadcast_add(mask)?,
        };
        let attn_weights = candle_nn::ops::softmax(&attn_weights, D::Minus1)?;
        let attn_output = attn_weights
            .matmul(&value_states)?
            .transpose(1, 2)?
            .reshape((b_sz, tgt_len, self.num_heads * self.head_dim))?;
        self.out_proj.forward(&attn_output)
    }
}

#[derive(Debug)]
struct DecoderLayer {
    self_attn: Attention,
    self_attn_layer_norm: LayerNorm,
    encoder_attn: Attention,
    encoder_attn_layer_norm: LayerNorm,
    fc1: Linear,
    fc2: Linear,
    final_layer_norm: LayerNorm,
    activation_fn: Activation,
}

impl DecoderLayer {
    fn load(vb: VarBuilder, cfg: &Config) -> Result<Self> {
        let h = cfg.hidden_size;
        let self_attn = Attention::load(vb.pp("self_attn"), cfg)?;
        let self_attn_layer_norm = layer_norm(h, 1e-5, vb.pp("self_attn_layer_norm"))?;
        let encoder_attn = Attention::load(vb.pp("encoder_attn"), cfg)?;
        let encoder_attn_layer_norm = layer_norm(h, 1e-5, vb.pp("encoder_attn_layer_norm"))?;
        let fc1 = linear_no_bias(h, cfg.ffn_dim, vb.pp("fc1"))?;
        let fc2 = linear_no_bias(cfg.ffn_dim, h, vb.pp("fc2"))?;
        let final_layer_norm = layer_norm(h, 1e-5, vb.pp("final_layer_norm"))?;
        Ok(Self {
            self_attn,
            self_attn_layer_norm,
            encoder_attn,
            encoder_attn_layer_norm,
            fc1,
            fc2,
            final_layer_norm,
            activation_fn: cfg.activation_function,
        })
    }

    fn forward(
        &self,
        xs: &Tensor,
        attention_mask: Option<&Tensor>,
        encoder_hidden_states: &Tensor,
    ) -> Result<Tensor> {
        // Self-attention block.
        let residual = xs.clone();
        let xs = self.self_attn_layer_norm.forward(xs)?;
        let xs = self.self_attn.forward(&xs, None, attention_mask)?;
        let xs = (xs + residual)?;

        // Cross-attention block over the T5 encoder outputs.
        let residual = xs.clone();
        let xs = self.encoder_attn_layer_norm.forward(&xs)?;
        let xs = self
            .encoder_attn
            .forward(&xs, Some(encoder_hidden_states), None)?;
        let xs = (xs + residual)?;

        // Feed-forward block.
        let residual = xs.clone();
        let xs = self.final_layer_norm.forward(&xs)?;
        let xs = self.fc1.forward(&xs)?;
        let xs = self.activation_fn.forward(&xs)?;
        let xs = self.fc2.forward(&xs)?;
        let xs = (xs + residual)?;
        Ok(xs)
    }
}

#[derive(Debug)]
struct Decoder {
    embed_tokens: Vec<Embedding>,
    embed_positions: SinusoidalPositionalEmbedding,
    layers: Vec<DecoderLayer>,
    layer_norm: LayerNorm,
    embed_scale: f64,
    num_codebooks: usize,
    d_model: usize,
}

impl Decoder {
    fn load(vb: VarBuilder, cfg: &Config) -> Result<Self> {
        let h = cfg.hidden_size;
        #[allow(clippy::cast_precision_loss)]
        let embed_scale = if cfg.scale_embedding {
            (h as f64).sqrt()
        } else {
            1.
        };
        // The embedding table is sized vocab_size + 1 so the pad/BOS token
        // (which equals `vocab_size`) is representable.
        let embed_dim = cfg.vocab_size + 1;
        let embed_tokens = (0..cfg.num_codebooks)
            .map(|i| embedding(embed_dim, h, vb.pp(format!("embed_tokens.{i}"))))
            .collect::<Result<Vec<_>>>()?;
        let embed_positions = SinusoidalPositionalEmbedding::load(&vb, cfg)?;
        let layers = (0..cfg.num_hidden_layers)
            .map(|i| DecoderLayer::load(vb.pp(format!("layers.{i}")), cfg))
            .collect::<Result<Vec<_>>>()?;
        let layer_norm = layer_norm(h, 1e-5, vb.pp("layer_norm"))?;
        Ok(Self {
            embed_tokens,
            embed_positions,
            layers,
            layer_norm,
            embed_scale,
            num_codebooks: cfg.num_codebooks,
            d_model: cfg.hidden_size,
        })
    }

    /// Build a `[1, 1, q_len, kv_len]` additive causal mask of `0` and
    /// `-inf` for self-attention. Pattern lifted from parler_tts.
    fn causal_mask(q_len: usize, kv_len: usize, device: &Device) -> Result<Tensor> {
        let mask: Vec<f32> = (0..q_len)
            .flat_map(|i| {
                (0..kv_len).map(move |j| {
                    if i + kv_len < j + q_len {
                        f32::NEG_INFINITY
                    } else {
                        0.
                    }
                })
            })
            .collect();
        let mask = Tensor::from_slice(&mask, (q_len, kv_len), device)?;
        mask.reshape((1, 1, q_len, kv_len))
    }

    /// Forward through the decoder.
    ///
    /// `input_ids`: `[B, num_codebooks, S]` (one token per codebook per step).
    /// `encoder_hidden_states`: `[B, T_enc, hidden_size]` from the T5 encoder
    ///   (post enc-to-dec projection if applicable).
    /// `pos_offset`: KV-cache offset -- absolute position of the first
    ///   token in `input_ids`. (We do *not* keep a KV cache yet, so the
    ///   caller passes the full prefix every step and `pos_offset` is `0`
    ///   here; the field stays in the signature for future KV-cache work.)
    fn forward(
        &mut self,
        input_ids: &Tensor,
        encoder_hidden_states: &Tensor,
        pos_offset: usize,
    ) -> Result<Tensor> {
        let device = input_ids.device();
        let (b_sz, num_codebooks, seq_len) = input_ids.dims3()?;
        debug_assert_eq!(num_codebooks, self.num_codebooks);

        let mut inputs_embeds = Tensor::zeros((b_sz, seq_len, self.d_model), DType::F32, device)?;
        for (idx, codebook) in self.embed_tokens.iter().enumerate() {
            let inp = input_ids.i((.., idx, ..))?;
            inputs_embeds = (inputs_embeds + codebook.forward(&inp)?)?;
        }
        let inputs_embeds = (inputs_embeds * self.embed_scale)?;
        let positions = self.embed_positions.slice(pos_offset, seq_len)?;
        let mut xs = inputs_embeds.broadcast_add(&positions)?;
        let causal = Self::causal_mask(seq_len, seq_len, device)?;
        for layer in &self.layers {
            xs = layer.forward(&xs, Some(&causal), encoder_hidden_states)?;
        }
        self.layer_norm.forward(&xs)
    }
}

/// Decoder + per-codebook LM heads (the autoregressive head).
#[derive(Debug)]
pub struct MusicgenForCausalLM {
    decoder: Decoder,
    lm_heads: Vec<Linear>,
    num_codebooks: usize,
    vocab_size: usize,
}

impl MusicgenForCausalLM {
    /// Load the decoder + LM heads from a `VarBuilder` rooted at the
    /// `decoder.*` prefix of the MusicGen safetensors file.
    ///
    /// # Errors
    ///
    /// Returns any candle error raised while constructing the underlying
    /// `Linear` / `Embedding` / `LayerNorm` layers (typically a missing
    /// tensor name in the safetensors file).
    pub fn load(vb: VarBuilder, cfg: &Config) -> Result<Self> {
        let h = cfg.hidden_size;
        let decoder = Decoder::load(vb.pp("model.decoder"), cfg)?;
        let lm_heads = (0..cfg.num_codebooks)
            .map(|i| linear_no_bias(h, cfg.vocab_size, vb.pp(format!("lm_heads.{i}"))))
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            decoder,
            lm_heads,
            num_codebooks: cfg.num_codebooks,
            vocab_size: cfg.vocab_size,
        })
    }

    /// Run the decoder + LM heads. Returns one logits tensor per codebook
    /// of shape `[B, S, vocab_size]`.
    ///
    /// # Errors
    ///
    /// Returns the first candle error raised by the decoder or any of
    /// the per-codebook LM heads.
    pub fn forward(
        &mut self,
        input_ids: &Tensor,
        encoder_hidden_states: &Tensor,
        pos_offset: usize,
    ) -> Result<Vec<Tensor>> {
        let hidden_states = self
            .decoder
            .forward(input_ids, encoder_hidden_states, pos_offset)?;
        self.lm_heads
            .iter()
            .map(|h| h.forward(&hidden_states))
            .collect()
    }

    /// Decoder hidden width.
    #[must_use]
    pub fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    /// Number of parallel codebooks.
    #[must_use]
    pub fn num_codebooks(&self) -> usize {
        self.num_codebooks
    }
}

/// Conditional generation root: T5 text encoder + EnCodec codec + the
/// MusicGen LM decoder.
#[derive(Debug)]
pub struct MusicgenForConditionalGeneration {
    /// T5 prompt encoder.
    pub text_encoder: t5::T5EncoderModel,
    /// EnCodec audio codec (re-used during weight load + for decoding).
    pub audio_encoder: encodec::Model,
    /// Autoregressive LM head.
    pub decoder: MusicgenForCausalLM,
    /// Combined config snapshot.
    pub cfg: GenConfig,
    /// Optional projection from `t5.d_model` to `decoder.hidden_size`
    /// when the two widths differ.
    pub enc_to_dec_proj: Option<Linear>,
}

/// Combined config for the three component models.
#[derive(Debug, Clone, PartialEq)]
pub struct GenConfig {
    /// MusicGen decoder hyper-parameters.
    pub musicgen: Config,
    /// T5 prompt-encoder hyper-parameters.
    pub t5: t5::Config,
    /// EnCodec audio-codec hyper-parameters.
    pub encodec: encodec::Config,
}

impl GenConfig {
    /// Combined config for `facebook/musicgen-small`.
    #[must_use]
    pub fn small() -> Self {
        Self {
            musicgen: Config::musicgen_small(),
            t5: t5::Config::musicgen_small(),
            encodec: musicgen_encodec_config(),
        }
    }

    /// Combined config for `facebook/musicgen-medium`.
    #[must_use]
    pub fn medium() -> Self {
        let mut t5 = t5::Config::musicgen_small();
        // musicgen-medium uses T5-base (d_model=768, num_layers=12). Same
        // as musicgen-small from the candle helper's perspective, so we
        // reuse it. The actual upstream "medium" T5 is also a base-class
        // T5 here -- per HF config.json the text encoder is identical;
        // only the decoder grows.
        t5.d_ff = 3072;
        Self {
            musicgen: Config::musicgen_medium(),
            t5,
            encodec: musicgen_encodec_config(),
        }
    }

    /// Combined config for `facebook/musicgen-large`.
    #[must_use]
    pub fn large() -> Self {
        Self {
            musicgen: Config::musicgen_large(),
            t5: t5::Config::musicgen_small(),
            encodec: musicgen_encodec_config(),
        }
    }
}

/// EnCodec config baked into every `facebook/musicgen-*` checkpoint
/// (32 kHz mono, 4 codebooks of 2048 entries each at 2.2 kbps).
fn musicgen_encodec_config() -> encodec::Config {
    encodec::Config {
        audio_channels: 1,
        chunk_length_s: None,
        codebook_dim: Some(128),
        codebook_size: 2048,
        compress: 2,
        dilation_growth_rate: 2,
        hidden_size: 128,
        kernel_size: 7,
        last_kernel_size: 7,
        norm_type: encodec::NormType::WeightNorm,
        normalize: false,
        num_filters: 64,
        num_lstm_layers: 2,
        num_residual_layers: 1,
        overlap: None,
        // Upstream comment: "This should be Reflect and not Replicate but
        // Reflect does not work yet" in candle 0.10.2.
        pad_mode: encodec::PadMode::Replicate,
        residual_kernel_size: 3,
        sampling_rate: 32_000,
        target_bandwidths: vec![2.2],
        trim_right_ratio: 1.0,
        upsampling_ratios: vec![8, 5, 4, 4],
        use_causal_conv: false,
        use_conv_shortcut: false,
    }
}

impl MusicgenForConditionalGeneration {
    /// Borrow the combined config.
    #[must_use]
    pub fn config(&self) -> &GenConfig {
        &self.cfg
    }

    /// Load every component model from the same `VarBuilder` rooted at the
    /// safetensors file.
    ///
    /// # Errors
    ///
    /// Returns any candle error raised by the T5 encoder, EnCodec, or
    /// MusicGen decoder loaders (typically a missing tensor name).
    pub fn load(vb: VarBuilder, cfg: GenConfig) -> Result<Self> {
        let text_encoder = t5::T5EncoderModel::load(vb.pp("text_encoder"), &cfg.t5)?;
        let audio_encoder = encodec::Model::new(&cfg.encodec, vb.pp("audio_encoder"))?;
        let decoder = MusicgenForCausalLM::load(vb.pp("decoder"), &cfg.musicgen)?;
        let enc_to_dec_proj = if cfg.t5.d_model == cfg.musicgen.hidden_size {
            None
        } else {
            // The HF checkpoint stores this as `enc_to_dec_proj.{weight,bias}`
            // when the encoder/decoder widths differ.
            Some(candle_nn::linear(
                cfg.t5.d_model,
                cfg.musicgen.hidden_size,
                vb.pp("enc_to_dec_proj"),
            )?)
        };
        Ok(Self {
            text_encoder,
            audio_encoder,
            decoder,
            cfg,
            enc_to_dec_proj,
        })
    }
}
