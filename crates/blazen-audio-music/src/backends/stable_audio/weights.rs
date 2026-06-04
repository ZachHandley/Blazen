//! Safetensors loader + PyTorch → candle key-remapping for the Stable
//! Audio Open Small native candle port.
//!
//! Stability AI's `stabilityai/stable-audio-open-small` ships a single
//! 1.68 GB `model.safetensors` with PyTorch-state-dict key layout (e.g.
//! `model.diffusion_model.transformer.layers.0.attention.q.weight`,
//! `pretransform.model.encoder.blocks.0.layers.0.residual_layers.0.0.weight`,
//! `conditioner.conditioners.prompt.model.encoder.block.0.layer.0.SelfAttention.q.weight`).
//!
//! The Rust components built in Wave 3.2 expect a [`VarBuilder`] rooted
//! at three submodules — `oobleck`, `dit`, `conditioner` — whose internal
//! paths mirror the candle module tree declared by [`OobleckVAE::new`],
//! [`DiT::new`], and [`Conditioner::new`]. This module does the
//! rename-only translation from PyTorch keys to those candle paths and
//! materialises a single root [`VarBuilder`] that the pipeline can
//! `pp("oobleck")` / `pp("dit")` / `pp("conditioner")` into.
//!
//! Only renames are performed here. If parity tests in a later wave
//! surface shape mismatches (most likely candidates: PyTorch `Conv1d`
//! weights stored as `(C_out, C_in, K)` that need to land as candle
//! `Linear` `(C_out, C_in)` when `K == 1`), the fix belongs in this
//! file as a tensor-transform pass alongside the existing rename map.
//!
//! [`OobleckVAE::new`]: super::oobleck::OobleckVAE::new
//! [`DiT::new`]: super::dit::DiT::new
//! [`Conditioner::new`]: super::conditioner::Conditioner::new

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use candle_core::safetensors::MmapedSafetensors;
use candle_core::{DType, Device, Result, Tensor};
use candle_nn::VarBuilder;

/// Loaded + remapped Stable Audio weights, ready to hand to the
/// pipeline constructor.
///
/// The internal [`VarBuilder`] is rooted at the top of the model — call
/// `weights.var_builder().pp("oobleck")` / `pp("dit")` / `pp("conditioner")`
/// to descend into each component.
pub struct StableAudioWeights {
    var_builder: VarBuilder<'static>,
}

impl StableAudioWeights {
    /// Load weights from a local `model.safetensors` file.
    ///
    /// The mmap is dropped before this function returns; every tensor
    /// is fully materialised in `device` memory under its candle-side
    /// name. This is intentional: holding the mmap alive across an
    /// async runtime would complicate the lifetime story for negligible
    /// memory savings (the Small variant is 1.68 GB regardless).
    ///
    /// # Errors
    ///
    /// Propagates I/O and candle deserialization errors.
    pub fn load(safetensors_path: &Path, device: &Device, dtype: DType) -> Result<Self> {
        // SAFETY: candle's safetensors mmap loader requires `unsafe`
        // because the file must not change underneath the mapping. The
        // caller is expected to pass a path inside an immutable cache
        // (hf-hub cache or a CI fixture); same convention as the
        // existing MusicGen loader.
        #[allow(unsafe_code)]
        let mmap = unsafe { MmapedSafetensors::new(safetensors_path)? };

        let mut remapped: HashMap<String, Tensor> = HashMap::new();
        let mut unmapped: u32 = 0;
        for (pytorch_key, _view) in mmap.tensors() {
            let candle_key = remap_pytorch_key(&pytorch_key);
            if candle_key == pytorch_key {
                unmapped = unmapped.saturating_add(1);
                tracing::debug!(key = %pytorch_key, "stable-audio weight key did not match any remap rule; passing through");
            }
            let tensor = mmap.load(&pytorch_key, device)?.to_dtype(dtype)?;
            remapped.insert(candle_key, tensor);
        }
        tracing::debug!(
            total = remapped.len(),
            unmapped,
            "stable-audio weights loaded",
        );

        let var_builder = VarBuilder::from_tensors(remapped, dtype, device);
        Ok(Self { var_builder })
    }

    /// Load weights from the HuggingFace Hub, caching them under the
    /// default `~/.cache/huggingface/hub/` location.
    ///
    /// The repo `stabilityai/stable-audio-open-small` is gated — users
    /// must accept the Stability AI Community License once and configure
    /// an HF token before this call will succeed.
    ///
    /// # Errors
    ///
    /// Propagates hf-hub fetch errors and any error from [`Self::load`].
    pub fn load_from_hf(device: &Device, dtype: DType) -> Result<Self> {
        let path: PathBuf = hf_hub::api::sync::ApiBuilder::new()
            .build()
            .map_err(|e| candle_core::Error::Msg(format!("hf-hub init failed: {e}")))?
            .model("stabilityai/stable-audio-open-small".to_string())
            .get("model.safetensors")
            .map_err(|e| {
                candle_core::Error::Msg(format!(
                    "hf-hub fetch failed for stabilityai/stable-audio-open-small: {e}"
                ))
            })?;
        Self::load(&path, device, dtype)
    }

    /// Borrow the root [`VarBuilder`].
    #[must_use]
    pub fn var_builder(&self) -> &VarBuilder<'static> {
        &self.var_builder
    }
}

/// Returns the ordered PyTorch → candle prefix-rename rules.
///
/// Order matters: the longest / most specific patterns must come first
/// because the rules are applied as `replace`-on-first-match. Both
/// prefix renames (e.g. `pretransform.model.encoder.` → `oobleck.encoder.`)
/// and tail-renames (e.g. `attention.q.weight` → `attention.to_qkv.weight`
/// is **not** done — see the note in [`remap_pytorch_key`] for why we
/// keep `to_q` / `to_kv` separate) live in this table.
///
/// The candle-side prefixes target the namespaces declared by the
/// component constructors:
/// - `oobleck.encoder.*`, `oobleck.decoder.*` — see [`OobleckVAE::new`].
/// - `dit.transformer.layers.{i}.*`, `dit.preprocess_conv.*`,
///   `dit.postprocess_conv.*`, `dit.timestep_features.*`,
///   `dit.to_timestep_embed.*`, `dit.to_cond_embed.*` — see [`DiT::new`].
/// - `conditioner.text_encoder.*` — see [`Conditioner::new`].
///
/// [`OobleckVAE::new`]: super::oobleck::OobleckVAE::new
/// [`DiT::new`]: super::dit::DiT::new
/// [`Conditioner::new`]: super::conditioner::Conditioner::new
fn key_remap_rules() -> Vec<(&'static str, &'static str)> {
    vec![
        // ---- Oobleck VAE (pretransform.model.* → oobleck.*) ----
        ("pretransform.model.encoder.", "oobleck.encoder."),
        ("pretransform.model.decoder.", "oobleck.decoder."),
        ("pretransform.model.", "oobleck."),
        // ---- DiT (model.diffusion_model.* → dit.*) ----
        // The candle DiT keeps the upstream `transformer.layers.{i}`
        // structure verbatim, so a single top-level rename suffices.
        ("model.diffusion_model.", "dit."),
        // ---- Conditioner T5 ----
        // Upstream: conditioner.conditioners.prompt.model.<...>
        // candle:   conditioner.text_encoder.<...>
        // (the conditioner crate uses `vb.pp("text_encoder")` for T5.)
        (
            "conditioner.conditioners.prompt.model.",
            "conditioner.text_encoder.",
        ),
        // Optional t5_proj weight (1.0 variant) — keep under conditioner root.
        (
            "conditioner.conditioners.prompt.t5_proj.",
            "conditioner.t5_proj.",
        ),
    ]
}

/// Apply [`key_remap_rules`] to a single PyTorch state-dict key.
///
/// Keys that match no rule are returned unchanged. The candle DiT
/// implementation already mirrors the upstream attention naming
/// (`to_qkv` / `to_kv` / `to_q` / `to_out` / `ff.0.proj` / `ff.2`), so
/// no per-tensor attention-suffix renaming is required at this layer.
#[must_use]
pub fn remap_pytorch_key(pytorch_key: &str) -> String {
    for (from, to) in key_remap_rules() {
        if let Some(rest) = pytorch_key.strip_prefix(from) {
            let mut out = String::with_capacity(to.len() + rest.len());
            out.push_str(to);
            out.push_str(rest);
            return out;
        }
    }
    pytorch_key.to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn remap_oobleck_encoder() {
        let pt = "pretransform.model.encoder.blocks.0.layers.0.residual_layers.0.0.weight";
        assert_eq!(
            remap_pytorch_key(pt),
            "oobleck.encoder.blocks.0.layers.0.residual_layers.0.0.weight",
        );
    }

    #[test]
    fn remap_oobleck_decoder() {
        let pt = "pretransform.model.decoder.blocks.0.layers.0.weight";
        assert_eq!(
            remap_pytorch_key(pt),
            "oobleck.decoder.blocks.0.layers.0.weight",
        );
    }

    #[test]
    fn remap_dit_attention() {
        // The candle DiT mirrors upstream attention names (`to_qkv`,
        // `to_kv`, `to_q`, `to_out`) so only the top-level prefix
        // changes here. Same for the per-block AdaLN and FF paths.
        let pt = "model.diffusion_model.transformer.layers.5.self_attn.to_qkv.weight";
        assert_eq!(
            remap_pytorch_key(pt),
            "dit.transformer.layers.5.self_attn.to_qkv.weight",
        );

        let adaln = "model.diffusion_model.transformer.layers.3.to_scale_shift_gate.1.weight";
        assert_eq!(
            remap_pytorch_key(adaln),
            "dit.transformer.layers.3.to_scale_shift_gate.1.weight",
        );
    }

    #[test]
    fn remap_dit_top_level() {
        assert_eq!(
            remap_pytorch_key("model.diffusion_model.preprocess_conv.weight"),
            "dit.preprocess_conv.weight",
        );
        assert_eq!(
            remap_pytorch_key("model.diffusion_model.to_timestep_embed.0.weight"),
            "dit.to_timestep_embed.0.weight",
        );
        assert_eq!(
            remap_pytorch_key("model.diffusion_model.timestep_features.weight"),
            "dit.timestep_features.weight",
        );
    }

    #[test]
    fn remap_conditioner_t5() {
        let pt =
            "conditioner.conditioners.prompt.model.encoder.block.0.layer.0.SelfAttention.q.weight";
        assert_eq!(
            remap_pytorch_key(pt),
            "conditioner.text_encoder.encoder.block.0.layer.0.SelfAttention.q.weight",
        );

        let shared = "conditioner.conditioners.prompt.model.shared.weight";
        assert_eq!(
            remap_pytorch_key(shared),
            "conditioner.text_encoder.shared.weight",
        );
    }

    #[test]
    fn unknown_key_passes_through() {
        // No rule matches → returned unchanged. The `load` path logs
        // this case at debug level so unexpected upstream additions
        // surface in the trace stream without breaking parity tests.
        let pt = "some.future.unknown.key.weight";
        assert_eq!(remap_pytorch_key(pt), pt);
    }

    #[test]
    fn pretransform_root_rename_for_non_enc_dec_keys() {
        // The third Oobleck rule (`pretransform.model.` → `oobleck.`)
        // is the catch-all for any future `pretransform.model.foo` key
        // that isn't under encoder/decoder. The more specific rules
        // must win when applicable — verified by the encoder/decoder
        // tests above.
        let pt = "pretransform.model.something_else.weight";
        assert_eq!(remap_pytorch_key(pt), "oobleck.something_else.weight");
    }

    #[test]
    #[ignore = "requires BLAZEN_TEST_STABLE_AUDIO=1 and downloads 1.68 GB of gated weights from HF Hub"]
    fn load_from_hf_smoke() {
        // Gated by `BLAZEN_TEST_STABLE_AUDIO=1` (mirrors the bark / f5 download
        // smokes) because the repo `stabilityai/stable-audio-open-small` is
        // gated and pulls 1.68 GB. The beastpc-e2e Rust music step does not set
        // this var — it exercises the real download via the Python GPU smokes —
        // so this stays a no-op skip on tokenless / non-gated machines.
        if std::env::var("BLAZEN_TEST_STABLE_AUDIO").ok().as_deref() != Some("1") {
            eprintln!("skipping: BLAZEN_TEST_STABLE_AUDIO != 1");
            return;
        }
        let device = Device::Cpu;
        let weights = StableAudioWeights::load_from_hf(&device, DType::F32)
            .expect("hf-hub load + remap should succeed when HF_TOKEN is set");
        // The root VarBuilder should at minimum contain the DiT input
        // projection — a sanity check that the rename pass reached the
        // expected candle namespace.
        assert!(
            weights
                .var_builder()
                .contains_tensor("dit.transformer.project_in.weight")
        );
    }
}
