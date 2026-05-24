//! `BiCodec` neural audio codec (Wave S.2).
//!
//! Sub-wave plan:
#![allow(
    dead_code,
    reason = "Top-level BiCodec surface (S.2.1.g) is wired up to consume the \
              S.2.1.{a..f} sub-modules but is not itself consumed until the \
              S.2.4 pipeline orchestration wave (which adds the public \
              re-export in `spark/mod.rs`). The unit tests in this module \
              exercise the configured surface in the meantime."
)]
//!
//! - **S.2.1.a**: shared low-level primitives —
//!   [`primitives::Snake1d`], [`primitives::WeightNormConv1d`],
//!   [`primitives::WeightNormConvTranspose1d`],
//!   [`primitives::AdaLayerNorm`], [`primitives::ResidualUnit`],
//!   [`primitives::repeat_interleave_dim2`].
//! - **S.2.1.b**: vocos backbone.
//! - **S.2.1.c**: sampler.
//! - **S.2.1.d**: quantizer.
//! - **S.2.1.e**: speaker.
//! - **S.2.1.f**: decoder.
//! - **S.2.1.g** (this commit): top-level [`BiCodec`] wiring —
//!   [`BiCodecConfig`], [`BiCodec`] struct, [`BiCodec::from_safetensors`]
//!   loader, [`BiCodec::tokenize`] / [`BiCodec::detokenize`] pipeline.
//!
//! # Top-level pipeline
//!
//! Mirrors `sparktts/models/bicodec.py::BiCodec`. The Python class wires
//! six sub-modules (`encoder`, `decoder`, `quantizer`, `speaker_encoder`,
//! `prenet`, `postnet`) into two inference entry points
//! (`BiCodec.tokenize` and `BiCodec.detokenize`) plus a training
//! `forward`. Only the inference paths are ported here.
//!
//! ## Upstream attribute → Rust field mapping
//!
//! | upstream `self.…` | this struct |
//! |---|---|
//! | `encoder`         | [`BiCodec::encoder`]         (feature encoder) |
//! | `quantizer`       | [`BiCodec::quantizer`]       (semantic codebook) |
//! | `speaker_encoder` | [`BiCodec::speaker_encoder`] (global tokens / d-vector) |
//! | `prenet`          | [`BiCodec::prenet`]          (d-vector-conditioned `Decoder`) |
//! | `postnet`         | [`BiCodec::postnet`]         (auxiliary, NOT called at inference) |
//! | `decoder`         | [`BiCodec::wave_generator`]  (DAC vocoder — renamed for clarity) |
//!
//! **State-dict key** for the vocoder is `decoder.…` (upstream's
//! attribute name), even though the Rust field is `wave_generator` for
//! readability. The [`BiCodec::from_safetensors`] loader passes
//! `vb.pp("decoder")` to [`super::decoder::WaveGenerator::load`].
//!
//! There is **no** separate `BiCodec.decoder` Decoder member upstream —
//! `BiCodec.decoder` IS the `WaveGenerator`. The plan's "decoder member
//! distinct from prenet/postnet/`wave_generator`" possibility is ruled out
//! by reading `bicodec.py::load_from_checkpoint` (line 86:
//! `decoder = WaveGenerator(**config["decoder"])`).
//!
//! ## `detokenize` recipe (verbatim from `bicodec.py::detokenize`)
//!
//! ```python
//! z_q      = self.quantizer.detokenize(semantic_tokens)    # (B, 1024, T)
//! d_vector = self.speaker_encoder.detokenize(global_tokens) # (B, 1024)
//! x = self.prenet(z_q, d_vector)                            # (B, 1024, T)
//! x = x + d_vector.unsqueeze(-1)                            # broadcast-add
//! wav_recon = self.decoder(x)                               # (B, 1, T * 320)
//! ```
//!
//! The broadcast-add lives in `BiCodec.detokenize` itself (NOT inside
//! the prenet), and `postnet` is not called at all. We mirror that
//! exactly.

#[allow(
    dead_code,
    reason = "Shared primitives are consumed by the S.2.1.{b..g} BiCodec \
              sub-waves (vocos, sampler, quantizer, speaker, decoder, \
              top-level). The unit tests in primitives.rs exercise the \
              public surface in the meantime."
)]
pub(super) mod primitives;

#[allow(
    dead_code,
    reason = "Vocos backbone (S.2.1.b) is consumed by the Encoder, prenet, \
              and postnet trunks in the S.2.1.{d..g} sub-waves. The unit \
              tests in vocos.rs exercise the public surface in the \
              meantime."
)]
pub(super) mod vocos;

#[allow(
    dead_code,
    reason = "Sampler + Encoder (S.2.1.c) are consumed by the top-level \
              BiCodec wiring in S.2.1.g. The unit tests in sampler.rs \
              exercise the public surface in the meantime."
)]
pub(super) mod sampler;

#[allow(
    dead_code,
    reason = "Quantizers (S.2.1.d) — FactorizedVectorQuantize for the \
              semantic stream, FSQ + ResidualFSQ for the global stream — \
              are consumed by the SpeakerEncoder (S.2.1.e) and the \
              top-level BiCodec wiring (S.2.1.g). The unit tests in \
              quantizer.rs exercise the public surface in the meantime."
)]
pub(super) mod quantizer;

#[allow(
    dead_code,
    reason = "SpeakerEncoder tower (S.2.1.e) — ECAPA-TDNN-GLOB-c512 + \
              PerceiverResampler + ResidualFSQ — is consumed by the \
              top-level BiCodec wiring in S.2.1.g. The unit tests in \
              speaker.rs exercise the public surface in the meantime."
)]
pub(super) mod speaker;

#[allow(
    dead_code,
    reason = "Feature Decoder + DAC WaveGenerator (S.2.1.f) — the prenet, \
              postnet, and vocoder — are consumed by the top-level \
              BiCodec wiring in S.2.1.g. The unit tests in decoder.rs \
              exercise the public surface in the meantime."
)]
pub(super) mod decoder;

// `BiCodec`, `BiCodecConfig`, and `BiCodecError` are all `pub(super)` —
// the `bicodec` module itself is `pub(super) mod bicodec;` in
// `spark/mod.rs`, so a literal `pub` on the items inside would resolve
// to the same effective visibility outside the crate but would (a)
// confuse readers about the intended export surface and (b) trip the
// `private_interfaces` lint by trying to re-export the also-`pub(super)`
// `SpeakerEncoderConfig`. The S.2.4 orchestration wave will add the
// public `pub use bicodec::{BiCodec, BiCodecConfig, BiCodecError};` at
// `spark/mod.rs` (or wherever the public Spark-TTS surface lives).

use std::path::{Path, PathBuf};

use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;

use self::speaker::SpeakerEncoderConfig;

// ---------------------------------------------------------------------------
// BiCodecConfig
// ---------------------------------------------------------------------------

/// Configuration for [`BiCodec::from_safetensors`].
///
/// Field names mirror upstream `BiCodec/config.yaml` keys grouped by
/// sub-module. The [`BiCodecConfig::spark_tts_05b`] constructor returns
/// the exact values shipped with `SparkAudio/Spark-TTS-0.5B` (verified
/// against `https://huggingface.co/SparkAudio/Spark-TTS-0.5B/raw/main/
/// BiCodec/config.yaml`).
#[derive(Debug, Clone)]
pub(super) struct BiCodecConfig {
    // ---- Semantic stream (Encoder + FactorizedVQ) ----
    /// Wav2vec feature dim feeding the encoder (`encoder.input_channels`
    /// upstream). Spark-TTS: `1024`.
    pub semantic_input_dim: usize,
    /// `VocosBackbone` hidden width (`encoder.vocos_dim`). Spark-TTS:
    /// `384`.
    pub semantic_vocos_dim: usize,
    /// `VocosBackbone` feed-forward width (`encoder.vocos_intermediate_dim`).
    /// Spark-TTS: `2048`.
    pub semantic_vocos_intermediate_dim: usize,
    /// `VocosBackbone` trunk depth (`encoder.vocos_num_layers`). Spark-TTS:
    /// `12`.
    pub semantic_vocos_num_layers: usize,
    /// Encoder output channel count (`encoder.out_channels`). Spark-TTS:
    /// `1024` — back to wav2vec dim, where the quantizer's `in_project`
    /// then drops to `codebook_dim`.
    pub semantic_out_channels: usize,
    /// Per-stage downsample ratios for the encoder (`encoder.sample_ratios`).
    /// Spark-TTS: `[1, 1]` — both stages are length-preserving (modulo
    /// the inner `3 * x` factor inside `SamplingBlock`).
    pub semantic_sample_ratios: Vec<usize>,
    /// `FactorizedVectorQuantize` codebook size (`quantizer.codebook_size`).
    /// Spark-TTS: `8192`.
    pub semantic_codebook_size: usize,
    /// `FactorizedVectorQuantize` codebook entry dim
    /// (`quantizer.codebook_dim`). Spark-TTS: `8`.
    pub semantic_codebook_dim: usize,

    // ---- Global stream (SpeakerEncoder) ----
    /// `SpeakerEncoder` config (ECAPA + Perceiver + `ResidualFsq`).
    /// Field stays module-private because [`SpeakerEncoderConfig`] is
    /// itself a `pub(super)` type from the [`speaker`] sub-module —
    /// exposing it via a `pub`/`pub(super)` field would trip the
    /// `private_interfaces` lint. Construct a `BiCodecConfig` via
    /// [`BiCodecConfig::spark_tts_05b`] which fills it in with the
    /// upstream default.
    speaker: SpeakerEncoderConfig,

    // ---- Prenet (Decoder conditioned on d-vector) ----
    /// `prenet.input_channels`. Spark-TTS: `1024`.
    pub prenet_input_channels: usize,
    /// `prenet.vocos_dim`. Spark-TTS: `384`.
    pub prenet_vocos_dim: usize,
    /// `prenet.vocos_intermediate_dim`. Spark-TTS: `2048`.
    pub prenet_vocos_intermediate_dim: usize,
    /// `prenet.vocos_num_layers`. Spark-TTS: `12`.
    pub prenet_vocos_num_layers: usize,
    /// `prenet.out_channels`. Spark-TTS: `1024`.
    pub prenet_out_channels: usize,
    /// `prenet.sample_ratios`. Spark-TTS: `[1, 1]`.
    pub prenet_sample_ratios: Vec<usize>,
    /// `prenet.condition_dim` — d-vector dim feeding the trunk's
    /// `AdaLN`. Spark-TTS: `1024`.
    pub prenet_condition_dim: usize,

    // ---- Postnet (Decoder, UNCONDITIONED; loaded for state-dict parity, NOT called at inference) ----
    /// `postnet.input_channels`. Spark-TTS: `1024`.
    pub postnet_input_channels: usize,
    /// `postnet.vocos_dim`. Spark-TTS: `384`.
    pub postnet_vocos_dim: usize,
    /// `postnet.vocos_intermediate_dim`. Spark-TTS: `2048`.
    pub postnet_vocos_intermediate_dim: usize,
    /// `postnet.vocos_num_layers`. Spark-TTS: `6` (NOT 12 — postnet is
    /// shallower than the prenet upstream).
    pub postnet_vocos_num_layers: usize,
    /// `postnet.out_channels`. Spark-TTS: `1024`.
    pub postnet_out_channels: usize,
    /// `postnet.sample_ratios`. Spark-TTS defaults to `[1, 1]` (the
    /// yaml omits this and falls back to upstream's
    /// `Decoder.__init__` default).
    pub postnet_sample_ratios: Vec<usize>,

    // ---- WaveGenerator (DAC vocoder; upstream `decoder.…` state-dict key) ----
    /// `decoder.input_channel`. Spark-TTS: `1024`.
    pub vocoder_input_channels: usize,
    /// `decoder.channels`. Spark-TTS: `1536`.
    pub vocoder_channels: usize,
    /// `decoder.rates`. Spark-TTS: `[8, 5, 4, 2]` — total upsample
    /// `8 * 5 * 4 * 2 = 320` (50 Hz semantic tokens → 16 kHz waveform).
    pub vocoder_rates: Vec<usize>,
    /// `decoder.kernel_sizes`. Spark-TTS: `[16, 11, 8, 4]`.
    pub vocoder_kernel_sizes: Vec<usize>,
}

impl BiCodecConfig {
    /// Defaults matching `SparkAudio/Spark-TTS-0.5B/BiCodec/config.yaml`
    /// verbatim.
    #[must_use]
    pub(super) fn spark_tts_05b() -> Self {
        Self {
            // ---- semantic stream ----
            semantic_input_dim: 1024,
            semantic_vocos_dim: 384,
            semantic_vocos_intermediate_dim: 2048,
            semantic_vocos_num_layers: 12,
            semantic_out_channels: 1024,
            semantic_sample_ratios: vec![1, 1],
            semantic_codebook_size: 8192,
            semantic_codebook_dim: 8,

            // ---- global stream ----
            speaker: SpeakerEncoderConfig::default(),

            // ---- prenet (d-vector conditioned) ----
            prenet_input_channels: 1024,
            prenet_vocos_dim: 384,
            prenet_vocos_intermediate_dim: 2048,
            prenet_vocos_num_layers: 12,
            prenet_out_channels: 1024,
            prenet_sample_ratios: vec![1, 1],
            prenet_condition_dim: 1024,

            // ---- postnet (auxiliary; loaded but not called at inference) ----
            postnet_input_channels: 1024,
            postnet_vocos_dim: 384,
            postnet_vocos_intermediate_dim: 2048,
            postnet_vocos_num_layers: 6,
            postnet_out_channels: 1024,
            postnet_sample_ratios: vec![1, 1],

            // ---- WaveGenerator (DAC vocoder; loaded under `decoder.…`) ----
            vocoder_input_channels: 1024,
            vocoder_channels: 1536,
            vocoder_rates: vec![8, 5, 4, 2],
            vocoder_kernel_sizes: vec![16, 11, 8, 4],
        }
    }
}

// ---------------------------------------------------------------------------
// BiCodecError
// ---------------------------------------------------------------------------

/// Typed errors for [`BiCodec::from_safetensors`], [`BiCodec::tokenize`],
/// and [`BiCodec::detokenize`].
#[derive(thiserror::Error, Debug)]
pub(super) enum BiCodecError {
    /// Safetensors discovery / mmap / `VarBuilder` construction failed.
    #[error("safetensors load failed: {0}")]
    Load(String),
    /// Input tensor shapes don't match the configured dims.
    #[error("invalid input shape: {0}")]
    InvalidShape(String),
    /// Sub-module forward pass returned an error.
    #[error("forward pass failed: {0}")]
    Forward(String),
}

impl From<candle_core::Error> for BiCodecError {
    fn from(e: candle_core::Error) -> Self {
        Self::Forward(e.to_string())
    }
}

// ---------------------------------------------------------------------------
// BiCodec
// ---------------------------------------------------------------------------

/// Top-level `BiCodec` neural audio codec.
///
/// See the [module docstring](self) for the upstream attribute mapping
/// and `detokenize` recipe. The two entry points are
/// [`BiCodec::tokenize`] (audio features → semantic + global token
/// streams) and [`BiCodec::detokenize`] (token streams → reconstructed
/// waveform).
pub(super) struct BiCodec {
    /// Semantic feature encoder. Upstream: `self.encoder`.
    encoder: self::sampler::Encoder,
    /// Semantic-stream codebook. Upstream: `self.quantizer`.
    quantizer: self::quantizer::FactorizedVectorQuantize,
    /// Global-stream speaker encoder. Upstream: `self.speaker_encoder`.
    speaker_encoder: self::speaker::SpeakerEncoder,
    /// d-vector-conditioned refinement decoder. Upstream: `self.prenet`.
    prenet: self::decoder::Decoder,
    /// Auxiliary postnet — loaded for state-dict completeness but NOT
    /// called by [`BiCodec::detokenize`]. Upstream: `self.postnet`.
    /// Wrapped in `Option` because some downstream Spark-TTS forks ship
    /// checkpoints without it; the standard `SparkAudio/Spark-TTS-0.5B`
    /// release does include it.
    #[allow(
        dead_code,
        reason = "Postnet is part of the BiCodec training graph (auxiliary \
                  feature-prediction loss). Inference paths do not consume \
                  it, but we still load it to keep state-dict parity with \
                  upstream — otherwise `load_state_dict(..., strict=False)` \
                  would log spurious `Unexpected tensor: postnet.…` warnings \
                  in any Python interop test."
    )]
    postnet: Option<self::decoder::Decoder>,
    /// DAC vocoder. Upstream attribute name is `self.decoder` (and the
    /// state-dict key is `decoder.…`); renamed here for readability —
    /// see module docs for the rationale.
    wave_generator: self::decoder::WaveGenerator,
    /// Saved-by-value config — handy for downstream pipeline code that
    /// needs to know rates / token dims.
    config: BiCodecConfig,
    /// Device the weights live on.
    device: Device,
}

impl BiCodec {
    /// Load a `BiCodec` from a Spark-TTS bundle directory or a direct
    /// path to its `model.safetensors`.
    ///
    /// `model_path` may be either:
    /// - the `BiCodec` bundle directory (containing `model.safetensors`
    ///   directly), or
    /// - the bundle's parent directory (containing a `BiCodec/`
    ///   subdirectory), or
    /// - the safetensors file itself.
    ///
    /// # Errors
    ///
    /// Returns [`BiCodecError::Load`] if the safetensors file cannot be
    /// resolved, mmapped, or parsed; returns [`BiCodecError::Forward`]
    /// (via the `From<candle_core::Error>` impl) if any sub-module loader
    /// fails (e.g. missing key, shape mismatch).
    pub(super) fn from_safetensors(
        model_path: &Path,
        config: BiCodecConfig,
        device: &Device,
    ) -> Result<Self, BiCodecError> {
        let st_path = resolve_safetensors_path(model_path)?;
        let dtype = DType::F32;
        // SAFETY: safetensors mmap is sound provided the file is not
        // mutated underneath us, which holds for read-only model
        // distributions. The `unsafe_code` lint is denied workspace-wide;
        // we narrowly allow it here for the single mmap call (same
        // pattern other candle backends in this crate use).
        #[allow(
            unsafe_code,
            reason = "VarBuilder::from_mmaped_safetensors is an unsafe API \
                      because mmapped contents can race with on-disk mutation. \
                      Model checkpoints are read-only on disk, so this is the \
                      canonical safe usage pattern."
        )]
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[&st_path], dtype, device).map_err(|e| {
                BiCodecError::Load(format!(
                    "VarBuilder::from_mmaped_safetensors({}): {e}",
                    st_path.display()
                ))
            })?
        };
        Self::from_var_builder(vb, config, device.clone())
    }

    /// Internal constructor — shared between [`BiCodec::from_safetensors`]
    /// and the synthetic-weights test path (which builds a `VarBuilder`
    /// from a `VarMap`).
    ///
    /// # Errors
    ///
    /// Returns [`BiCodecError::Forward`] if any sub-module load fails
    /// (the `candle_core::Error` is mapped through the `From` impl).
    #[allow(
        clippy::needless_pass_by_value,
        reason = "VarBuilder is the canonical consume-by-value handle in \
                  candle — every sub-module loader takes it by value via \
                  `vb.pp(prefix)`. Same convention used by every other \
                  *::load function in this module's siblings."
    )]
    fn from_var_builder(
        vb: VarBuilder<'_>,
        config: BiCodecConfig,
        device: Device,
    ) -> Result<Self, BiCodecError> {
        let encoder = self::sampler::Encoder::load(
            vb.pp("encoder"),
            config.semantic_input_dim,
            config.semantic_vocos_dim,
            config.semantic_vocos_intermediate_dim,
            config.semantic_vocos_num_layers,
            config.semantic_out_channels,
            &config.semantic_sample_ratios,
        )?;
        let quantizer = self::quantizer::FactorizedVectorQuantize::load(
            vb.pp("quantizer"),
            config.semantic_out_channels,
            config.semantic_codebook_size,
            config.semantic_codebook_dim,
        )?;
        let speaker_encoder =
            self::speaker::SpeakerEncoder::load(vb.pp("speaker_encoder"), config.speaker.clone())?;
        let prenet = self::decoder::Decoder::load(
            vb.pp("prenet"),
            config.prenet_input_channels,
            config.prenet_vocos_dim,
            config.prenet_vocos_intermediate_dim,
            config.prenet_vocos_num_layers,
            config.prenet_out_channels,
            &config.prenet_sample_ratios,
            Some(config.prenet_condition_dim),
        )?;
        // Postnet is tolerated-absent: some Spark-TTS forks ship
        // inference-only checkpoints that strip the postnet. We try to
        // load it, and if anything goes wrong (missing keys, shape
        // mismatch) we silently set it to `None`. The standard
        // `SparkAudio/Spark-TTS-0.5B` release includes it; this branch
        // exists for forward-compat with stripped-checkpoint forks.
        let postnet = self::decoder::Decoder::load(
            vb.pp("postnet"),
            config.postnet_input_channels,
            config.postnet_vocos_dim,
            config.postnet_vocos_intermediate_dim,
            config.postnet_vocos_num_layers,
            config.postnet_out_channels,
            &config.postnet_sample_ratios,
            None,
        )
        .ok();
        // `wave_generator` is loaded under the state-dict prefix
        // `decoder.…` — upstream's `BiCodec.decoder` attribute IS the
        // `WaveGenerator`. See module docs.
        let wave_generator = self::decoder::WaveGenerator::load(
            vb.pp("decoder"),
            config.vocoder_input_channels,
            config.vocoder_channels,
            &config.vocoder_rates,
            &config.vocoder_kernel_sizes,
        )?;

        Ok(Self {
            encoder,
            quantizer,
            speaker_encoder,
            prenet,
            postnet,
            wave_generator,
            config,
            device,
        })
    }

    /// Encode wav2vec features + mel-spectrogram into the two `BiCodec`
    /// token streams.
    ///
    /// - `feat`: `(B, semantic_input_dim, T_feat)` — wav2vec hidden
    ///   states at 50 Hz (channels-first; upstream's
    ///   `BiCodec.tokenize` transposes a channels-last input
    ///   internally — callers here pass channels-first directly).
    /// - `mels`: `(B, mel_input_dim, T_mel)` — log-mel spectrogram
    ///   ready for the speaker encoder.
    ///
    /// Returns `(semantic_tokens, global_tokens)`:
    /// - `semantic_tokens`: `(B, T_feat)` `u32` indices into the
    ///   semantic codebook.
    /// - `global_tokens`: `(B, token_num, fsq_num_quantizers)` `u32`
    ///   indices into the global `ResidualFsq` codebook.
    ///
    /// # Errors
    ///
    /// Returns [`BiCodecError::InvalidShape`] if `feat` or `mels` have
    /// the wrong rank / channel dim. Propagates [`BiCodecError::Forward`]
    /// (via `From<candle_core::Error>`) from any sub-module.
    pub(super) fn tokenize(
        &self,
        feat: &Tensor,
        mels: &Tensor,
    ) -> Result<(Tensor, Tensor), BiCodecError> {
        validate_three_dim(feat, "feat", self.config.semantic_input_dim)?;
        validate_three_dim(mels, "mels", self.config.speaker.input_dim)?;

        // Semantic stream.
        let z_semantic = self.encoder.forward(feat)?;
        let semantic_tokens = self.quantizer.tokenize(&z_semantic)?;

        // Global stream.
        let global_tokens = self.speaker_encoder.tokenize(mels)?;

        Ok((semantic_tokens, global_tokens))
    }

    /// Decode the two `BiCodec` token streams back to a 16 kHz waveform.
    ///
    /// - `semantic_tokens`: `(B, T_feat)` `u32` indices.
    /// - `global_tokens`: `(B, token_num, fsq_num_quantizers)` `u32`
    ///   indices.
    ///
    /// Returns `(B, 1, T_feat * prod(vocoder_rates))` `f32` waveform in
    /// `[-1, 1]` (tanh-bounded by the [`super::decoder::WaveGenerator`]).
    ///
    /// Mirrors `bicodec.py::BiCodec.detokenize` verbatim (no postnet
    /// call, broadcast-add lives in this function — NOT inside the
    /// prenet).
    ///
    /// # Errors
    ///
    /// Propagates [`BiCodecError::Forward`] from any sub-module.
    pub(super) fn detokenize(
        &self,
        semantic_tokens: &Tensor,
        global_tokens: &Tensor,
    ) -> Result<Tensor, BiCodecError> {
        // 1. Semantic codebook lookup → (B, semantic_out_channels=1024, T).
        let z_q = self.quantizer.detokenize(semantic_tokens)?;

        // 2. d-vector from global tokens → (B, out_dim=1024).
        let d_vector = self.speaker_encoder.detokenize(global_tokens)?;

        // 3. Prenet refinement (AdaLN-conditioned on d-vector) →
        //    (B, prenet_out_channels=1024, T).
        let refined = self.prenet.forward(&z_q, Some(&d_vector))?;

        // 4. Broadcast-add d_vector onto the refined feature.
        //    Upstream `bicodec.py::detokenize`:
        //      x = x + d_vector.unsqueeze(-1)
        //    `d_vector` is (B, 1024) → (B, 1024, 1), then broadcast.
        let d_vec_unsq = d_vector.unsqueeze(2)?;
        let conditioned = refined.broadcast_add(&d_vec_unsq)?;

        // 5. WaveGenerator: (B, 1024, T) → (B, 1, T * 320), tanh-bounded.
        let wav = self.wave_generator.forward(&conditioned)?;
        Ok(wav)
    }

    /// Borrow the device the weights live on.
    #[must_use]
    pub(super) fn device(&self) -> &Device {
        &self.device
    }

    /// Borrow the loaded config.
    #[must_use]
    pub(super) fn config(&self) -> &BiCodecConfig {
        &self.config
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Resolve `model_path` to an actual `model.safetensors` file path.
///
/// Accepts: the safetensors file itself, the `BiCodec` directory
/// containing it, or the `BiCodec`-parent directory containing a
/// `BiCodec/` subdirectory.
fn resolve_safetensors_path(model_path: &Path) -> Result<PathBuf, BiCodecError> {
    if model_path.is_file() {
        return Ok(model_path.to_path_buf());
    }
    if model_path.is_dir() {
        // Try `<dir>/model.safetensors` first.
        let direct = model_path.join("model.safetensors");
        if direct.is_file() {
            return Ok(direct);
        }
        // Then `<dir>/BiCodec/model.safetensors`.
        let nested = model_path.join("BiCodec").join("model.safetensors");
        if nested.is_file() {
            return Ok(nested);
        }
    }
    Err(BiCodecError::Load(format!(
        "could not locate BiCodec/model.safetensors under `{}` (checked: \
         the path itself, `<dir>/model.safetensors`, `<dir>/BiCodec/model.safetensors`)",
        model_path.display()
    )))
}

/// Validate that `t` is rank-3 with channel dim matching `expected_channels`.
fn validate_three_dim(
    t: &Tensor,
    name: &str,
    expected_channels: usize,
) -> Result<(), BiCodecError> {
    let dims = t.dims();
    if dims.len() != 3 {
        return Err(BiCodecError::InvalidShape(format!(
            "{name}: expected rank-3 (B, C, T) tensor, got rank-{} with dims {:?}",
            dims.len(),
            dims
        )));
    }
    if dims[1] != expected_channels {
        return Err(BiCodecError::InvalidShape(format!(
            "{name}: expected channel dim {expected_channels}, got {} (full dims {:?})",
            dims[1], dims
        )));
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bicodec_config_default_spark_tts_05b_matches_upstream() {
        let cfg = BiCodecConfig::spark_tts_05b();

        // Semantic stream — from SparkAudio/Spark-TTS-0.5B/BiCodec/config.yaml.
        assert_eq!(cfg.semantic_input_dim, 1024);
        assert_eq!(cfg.semantic_vocos_dim, 384);
        assert_eq!(cfg.semantic_vocos_intermediate_dim, 2048);
        assert_eq!(cfg.semantic_vocos_num_layers, 12);
        assert_eq!(cfg.semantic_out_channels, 1024);
        assert_eq!(cfg.semantic_sample_ratios, vec![1, 1]);
        assert_eq!(cfg.semantic_codebook_size, 8192);
        assert_eq!(cfg.semantic_codebook_dim, 8);

        // Global stream.
        assert_eq!(cfg.speaker.input_dim, 128);
        assert_eq!(cfg.speaker.out_dim, 1024);
        assert_eq!(cfg.speaker.latent_dim, 128);
        assert_eq!(cfg.speaker.token_num, 32);
        assert_eq!(cfg.speaker.fsq_levels, vec![4, 4, 4, 4, 4, 4]);
        assert_eq!(cfg.speaker.fsq_num_quantizers, 1);

        // Prenet.
        assert_eq!(cfg.prenet_input_channels, 1024);
        assert_eq!(cfg.prenet_vocos_dim, 384);
        assert_eq!(cfg.prenet_vocos_intermediate_dim, 2048);
        assert_eq!(cfg.prenet_vocos_num_layers, 12);
        assert_eq!(cfg.prenet_out_channels, 1024);
        assert_eq!(cfg.prenet_sample_ratios, vec![1, 1]);
        assert_eq!(cfg.prenet_condition_dim, 1024);

        // Postnet — shallower trunk (6 layers), no condition.
        assert_eq!(cfg.postnet_vocos_num_layers, 6);
        assert_eq!(cfg.postnet_input_channels, 1024);
        assert_eq!(cfg.postnet_out_channels, 1024);

        // WaveGenerator — total upsample = 8 * 5 * 4 * 2 = 320.
        assert_eq!(cfg.vocoder_input_channels, 1024);
        assert_eq!(cfg.vocoder_channels, 1536);
        assert_eq!(cfg.vocoder_rates, vec![8, 5, 4, 2]);
        assert_eq!(cfg.vocoder_kernel_sizes, vec![16, 11, 8, 4]);
        let total_upsample: usize = cfg.vocoder_rates.iter().product();
        assert_eq!(total_upsample, 320);
    }

    #[test]
    fn bicodec_error_from_candle_error_is_forward_variant() {
        // Synthesize a candle_core::Error and check the From impl routes
        // it to BiCodecError::Forward (NOT Load — Load is reserved for
        // safetensors discovery / mmap failures).
        let candle_err = candle_core::Error::Msg("synthetic forward failure".into());
        let bi_err: BiCodecError = candle_err.into();
        match bi_err {
            BiCodecError::Forward(msg) => {
                assert!(msg.contains("synthetic forward failure"), "msg was: {msg}");
            }
            other => panic!("expected BiCodecError::Forward, got {other:?}"),
        }
    }

    #[test]
    fn bicodec_load_returns_helpful_error_for_missing_safetensors() {
        // Point at a path that definitely doesn't exist. Resolver should
        // fail with BiCodecError::Load and a message that mentions the
        // path it actually tried. (We can't use `Result::expect_err`
        // because `BiCodec` doesn't derive `Debug` — its sub-modules
        // don't either — so we hand-match the Result.)
        let bogus = Path::new("/nonexistent/blazen-bicodec-test-path");
        let cfg = BiCodecConfig::spark_tts_05b();
        match BiCodec::from_safetensors(bogus, cfg, &Device::Cpu) {
            Ok(_) => panic!("loading from a nonexistent path must fail"),
            Err(BiCodecError::Load(msg)) => {
                assert!(
                    msg.contains("/nonexistent/blazen-bicodec-test-path"),
                    "Load error message did not mention the bogus path; got: {msg}"
                );
            }
            Err(other) => panic!("expected BiCodecError::Load, got {other:?}"),
        }
    }

    #[test]
    fn validate_three_dim_rejects_rank_mismatch() {
        let dev = Device::Cpu;
        let t = Tensor::zeros((4, 8), DType::F32, &dev).unwrap();
        let err = validate_three_dim(&t, "feat", 8).expect_err("rank-2 input must be rejected");
        match err {
            BiCodecError::InvalidShape(msg) => {
                assert!(msg.contains("rank-2"), "msg was: {msg}");
                assert!(msg.contains("feat"), "msg was: {msg}");
            }
            other => panic!("expected InvalidShape, got {other:?}"),
        }
    }

    #[test]
    fn validate_three_dim_rejects_channel_mismatch() {
        let dev = Device::Cpu;
        let t = Tensor::zeros((2, 7, 16), DType::F32, &dev).unwrap();
        let err =
            validate_three_dim(&t, "mels", 128).expect_err("wrong channel dim must be rejected");
        match err {
            BiCodecError::InvalidShape(msg) => {
                assert!(
                    msg.contains("128"),
                    "expected message mentions 128; got: {msg}"
                );
                assert!(msg.contains('7'), "expected message mentions 7; got: {msg}");
            }
            other => panic!("expected InvalidShape, got {other:?}"),
        }
    }
}
