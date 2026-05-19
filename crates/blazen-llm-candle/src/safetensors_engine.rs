//! Non-quantized safetensors loader for the candle backend.
//!
//! Parallels [`crate::provider`]'s GGUF engine but loads dequantized
//! weights through `candle_nn::VarBuilder::from_mmaped_safetensors`
//! into one of the supported `candle_transformers` model graphs
//! ([`Llama`], [`qwen2::ModelForCausalLM`], [`mistral::Model`]).
//!
//! Architecture dispatch is driven by `config.json`'s `model_type`:
//! `"llama"`, `"qwen2"`, `"mistral"`. Every variant funnels through the
//! same `VarBuilder` so the architecture-agnostic
//! [`crate::lora_backend::LoraMergingBackend`] just works — PEFT keys
//! for `q_proj` / `k_proj` / `v_proj` / `o_proj` are spelled the same
//! across all three model families.
//!
//! The whole module is gated behind the `engine` feature: without it
//! [`crate::lib`] does not re-export this module, matching the GGUF
//! engine's gating pattern.

#![cfg(feature = "engine")]

use std::path::PathBuf;

use candle_core::{DType, Device, IndexOp, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::generation::LogitsProcessor;
use candle_transformers::models::llama::{Cache, Config, Llama, LlamaConfig, LlamaEosToks};
use candle_transformers::models::{mistral, qwen2};
use serde::Deserialize;
use tokenizers::Tokenizer;

use crate::CandleLlmOptions;
use crate::provider::CandleLlmError;

/// Subset of `config.json` needed for architecture dispatch.
///
/// Only the `model_type` field is mandatory; everything else is
/// re-parsed into the per-architecture `Config` once the architecture
/// is confirmed.
#[derive(Debug, Deserialize)]
struct ArchProbe {
    #[serde(default)]
    model_type: Option<String>,
}

/// Supported `model_type` values, in the form the per-arch `Config`
/// expects to deserialize from the same `config.json` bytes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Arch {
    Llama,
    Qwen2,
    Mistral,
}

impl Arch {
    fn from_model_type(model_type: &str) -> Option<Self> {
        match model_type {
            "llama" => Some(Self::Llama),
            "qwen2" => Some(Self::Qwen2),
            "mistral" => Some(Self::Mistral),
            _ => None,
        }
    }
}

/// Tolerant Qwen2 config wrapper.
///
/// Why: real Qwen2 / Qwen2.5 configs may omit `use_sliding_window`
/// (defaults to false), set `sliding_window` to null, or skip
/// `max_window_layers` / `tie_word_embeddings` / `hidden_act`. Upstream
/// `qwen2::Config` requires every field, so we deserialize into a
/// permissive shim and project to the strict struct.
#[derive(Debug, Deserialize)]
struct Qwen2ConfigShim {
    vocab_size: usize,
    hidden_size: usize,
    intermediate_size: usize,
    num_hidden_layers: usize,
    num_attention_heads: usize,
    num_key_value_heads: usize,
    max_position_embeddings: usize,
    #[serde(default)]
    sliding_window: Option<usize>,
    #[serde(default)]
    max_window_layers: Option<usize>,
    #[serde(default)]
    tie_word_embeddings: Option<bool>,
    rope_theta: f64,
    rms_norm_eps: f64,
    #[serde(default)]
    use_sliding_window: Option<bool>,
    #[serde(default)]
    hidden_act: Option<candle_nn::Activation>,
}

impl Qwen2ConfigShim {
    fn into_config(self) -> qwen2::Config {
        let sliding_window = self.sliding_window.unwrap_or(self.max_position_embeddings);
        qwen2::Config {
            vocab_size: self.vocab_size,
            hidden_size: self.hidden_size,
            intermediate_size: self.intermediate_size,
            num_hidden_layers: self.num_hidden_layers,
            num_attention_heads: self.num_attention_heads,
            num_key_value_heads: self.num_key_value_heads,
            max_position_embeddings: self.max_position_embeddings,
            sliding_window,
            max_window_layers: self.max_window_layers.unwrap_or(self.num_hidden_layers),
            tie_word_embeddings: self.tie_word_embeddings.unwrap_or(false),
            rope_theta: self.rope_theta,
            rms_norm_eps: self.rms_norm_eps,
            use_sliding_window: self.use_sliding_window.unwrap_or(false),
            hidden_act: self.hidden_act.unwrap_or(candle_nn::Activation::Silu),
        }
    }
}

/// Generic "EOS token(s)" representation extracted from `config.json`.
/// Mirrors HF's `eos_token_id` convention which is sometimes a single
/// integer, sometimes a list.
#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum EosTokenSpec {
    Single(u32),
    Multiple(Vec<u32>),
}

/// Probe just the EOS id(s) out of `config.json`. Used to feed the
/// stop-token check for Qwen2 / Mistral (whose `Config` structs don't
/// carry an `eos_token_id` field).
#[derive(Debug, Deserialize, Default)]
struct EosProbe {
    #[serde(default)]
    eos_token_id: Option<EosTokenSpec>,
}

/// Resolve the candle [`Device`] from the user's device string.
///
/// Matches the GGUF engine's resolver byte-for-byte so the two paths
/// stay behavior-compatible.
fn resolve_device(device_str: Option<&str>) -> Result<Device, CandleLlmError> {
    match device_str.unwrap_or("cpu") {
        "cpu" => Ok(Device::Cpu),
        s if s.starts_with("cuda") => {
            let ordinal = s
                .strip_prefix("cuda:")
                .and_then(|n| n.parse::<usize>().ok())
                .unwrap_or(0);
            Device::new_cuda(ordinal)
                .map_err(|e| CandleLlmError::ModelLoad(format!("CUDA device error: {e}")))
        }
        "metal" => Device::new_metal(0)
            .map_err(|e| CandleLlmError::ModelLoad(format!("Metal device error: {e}"))),
        other => Err(CandleLlmError::InvalidOptions(format!(
            "unsupported device: {other}"
        ))),
    }
}

/// Build the HF Hub repo handle from a repo id + optional revision.
fn open_repo(
    repo_id: &str,
    revision: Option<&str>,
) -> Result<hf_hub::api::tokio::ApiRepo, CandleLlmError> {
    let api = hf_hub::api::tokio::ApiBuilder::new()
        .with_progress(true)
        .build()
        .map_err(|e| CandleLlmError::ModelLoad(format!("HF API error: {e}")))?;

    let repo = if let Some(rev) = revision {
        api.repo(hf_hub::Repo::with_revision(
            repo_id.to_string(),
            hf_hub::RepoType::Model,
            rev.to_string(),
        ))
    } else {
        api.model(repo_id.to_string())
    };
    Ok(repo)
}

/// Categorisation of a model repo's weight layout discovered via the
/// HF Hub `info` endpoint.
#[derive(Debug, Clone)]
pub(crate) struct RepoLayout {
    /// Names (relative) of any `*.gguf` files in the repo.
    pub(crate) gguf: Vec<String>,
    /// Names of safetensors shards. `model.safetensors` (single file)
    /// or `model-NNNNN-of-MMMMM.safetensors` (sharded). Always sorted
    /// so multi-file repos download in deterministic order.
    pub(crate) safetensors: Vec<String>,
}

impl RepoLayout {
    pub(crate) fn has_gguf(&self) -> bool {
        !self.gguf.is_empty()
    }
    pub(crate) fn has_safetensors(&self) -> bool {
        !self.safetensors.is_empty()
    }
}

/// Cheap HEAD-level metadata probe — no file content is downloaded.
///
/// Filters `siblings` by suffix and shape; rejects adapter weights
/// (`adapter_model.safetensors`) which would otherwise masquerade as
/// the base model.
pub(crate) async fn probe_repo_layout(
    repo_id: &str,
    revision: Option<&str>,
) -> Result<RepoLayout, CandleLlmError> {
    let repo = open_repo(repo_id, revision)?;
    let info = repo
        .info()
        .await
        .map_err(|e| CandleLlmError::ModelLoad(format!("HF repo info failed: {e}")))?;

    let mut gguf: Vec<String> = Vec::new();
    let mut safetensors: Vec<String> = Vec::new();

    for sib in info.siblings {
        let name = sib.rfilename;
        if std::path::Path::new(&name)
            .extension()
            .is_some_and(|ext| ext.eq_ignore_ascii_case("gguf"))
        {
            gguf.push(name);
            continue;
        }
        // Why: PEFT adapter weights also use the `.safetensors` suffix
        // — excluding them keeps the base-model detector honest.
        if name == "adapter_model.safetensors" || name.ends_with("/adapter_model.safetensors") {
            continue;
        }
        if name == "model.safetensors" {
            safetensors.push(name);
            continue;
        }
        if is_sharded_safetensors(&name) {
            safetensors.push(name);
        }
    }

    safetensors.sort();
    gguf.sort();
    Ok(RepoLayout { gguf, safetensors })
}

/// Match the canonical HF sharded-safetensors naming convention:
/// `model-NNNNN-of-MMMMM.safetensors`.
fn is_sharded_safetensors(name: &str) -> bool {
    let Some(rest) = name.strip_prefix("model-") else {
        return false;
    };
    let Some(rest) = rest.strip_suffix(".safetensors") else {
        return false;
    };
    let Some((shard, total)) = rest.split_once("-of-") else {
        return false;
    };
    !shard.is_empty()
        && !total.is_empty()
        && shard.bytes().all(|b| b.is_ascii_digit())
        && total.bytes().all(|b| b.is_ascii_digit())
}

/// Download `config.json`, `tokenizer.json`, and every safetensors
/// file listed in `layout.safetensors`, returning the local cache
/// paths.
async fn download_safetensors_assets(
    repo_id: &str,
    revision: Option<&str>,
    layout: &RepoLayout,
) -> Result<(PathBuf, PathBuf, Vec<PathBuf>), CandleLlmError> {
    let repo = open_repo(repo_id, revision)?;

    let config_path = repo
        .get("config.json")
        .await
        .map_err(|e| CandleLlmError::ModelLoad(format!("config.json download failed: {e}")))?;

    let tokenizer_path = repo
        .get("tokenizer.json")
        .await
        .map_err(|e| CandleLlmError::ModelLoad(format!("tokenizer.json download failed: {e}")))?;

    let mut weight_paths: Vec<PathBuf> = Vec::with_capacity(layout.safetensors.len());
    for name in &layout.safetensors {
        let path = repo.get(name).await.map_err(|e| {
            CandleLlmError::ModelLoad(format!("safetensors download failed ({name}): {e}"))
        })?;
        weight_paths.push(path);
    }

    Ok((config_path, tokenizer_path, weight_paths))
}

/// Per-architecture config + max context window, derived from
/// `config.json` after architecture detection.
#[derive(Debug)]
enum ParsedConfig {
    Llama(Config),
    Qwen2(qwen2::Config),
    Mistral(mistral::Config),
}

impl ParsedConfig {
    fn arch(&self) -> Arch {
        match self {
            Self::Llama(_) => Arch::Llama,
            Self::Qwen2(_) => Arch::Qwen2,
            Self::Mistral(_) => Arch::Mistral,
        }
    }

    fn max_position_embeddings(&self) -> usize {
        match self {
            Self::Llama(c) => c.max_position_embeddings,
            Self::Qwen2(c) => c.max_position_embeddings,
            Self::Mistral(c) => c.max_position_embeddings,
        }
    }
}

/// Parse `config.json` and confirm the architecture is one this
/// loader supports. Returns the parsed per-arch `Config`.
fn parse_arch_config(config_path: &std::path::Path) -> Result<ParsedConfig, CandleLlmError> {
    let bytes = std::fs::read(config_path).map_err(|e| {
        CandleLlmError::ModelLoad(format!("cannot read {}: {e}", config_path.display()))
    })?;

    let probe: ArchProbe = serde_json::from_slice(&bytes)
        .map_err(|e| CandleLlmError::ModelLoad(format!("config.json arch probe failed: {e}")))?;

    let arch_str = probe.model_type.as_deref().unwrap_or("");
    let Some(arch) = Arch::from_model_type(arch_str) else {
        return Err(CandleLlmError::Unsupported(format!(
            "candle safetensors path: model_type '{arch_str}' not supported \
             (wired: llama, qwen2, mistral)"
        )));
    };

    match arch {
        Arch::Llama => {
            let llama_cfg: LlamaConfig = serde_json::from_slice(&bytes)
                .map_err(|e| CandleLlmError::ModelLoad(format!("LlamaConfig parse failed: {e}")))?;
            Ok(ParsedConfig::Llama(llama_cfg.into_config(false)))
        }
        Arch::Qwen2 => {
            let shim: Qwen2ConfigShim = serde_json::from_slice(&bytes).map_err(|e| {
                CandleLlmError::ModelLoad(format!("qwen2 Config parse failed: {e}"))
            })?;
            Ok(ParsedConfig::Qwen2(shim.into_config()))
        }
        Arch::Mistral => {
            let mistral_cfg: mistral::Config = serde_json::from_slice(&bytes).map_err(|e| {
                CandleLlmError::ModelLoad(format!("mistral Config parse failed: {e}"))
            })?;
            Ok(ParsedConfig::Mistral(mistral_cfg))
        }
    }
}

/// Pull the first EOS token id out of a `config.json`, if any.
fn parse_eos_token(config_path: &std::path::Path) -> Option<u32> {
    let bytes = std::fs::read(config_path).ok()?;
    let probe: EosProbe = serde_json::from_slice(&bytes).ok()?;
    match probe.eos_token_id? {
        EosTokenSpec::Single(id) => Some(id),
        EosTokenSpec::Multiple(ids) => ids.first().copied(),
    }
}

/// Pick the model dtype from the user's quantization hint. The
/// safetensors path is non-quantized so only floating-point options
/// make sense; everything else falls back to `F16` (the canonical
/// Llama2 weights dtype).
fn resolve_dtype(opts: &CandleLlmOptions, device: &Device) -> DType {
    match opts.quantization.as_deref().map(str::to_ascii_lowercase) {
        Some(ref q) if q == "f32" => DType::F32,
        Some(ref q) if q == "bf16" => DType::BF16,
        Some(ref q) if q == "f16" => DType::F16,
        // Why: CPU lacks fast f16/bf16 kernels for some ops in candle
        // 0.10.2; default to f32 there to avoid silent slowdowns.
        _ if matches!(device, Device::Cpu) => DType::F32,
        _ => DType::F16,
    }
}

/// Per-architecture loaded model. Each variant owns its own forward
/// state — Llama keeps its KV cache in an external [`Cache`], while
/// Qwen2 / Mistral keep theirs inside the model struct and expose
/// `clear_kv_cache(&mut self)`.
pub enum LoadedModel {
    Llama(Box<Llama>, Box<Cache>),
    Qwen2(Box<qwen2::ModelForCausalLM>),
    Mistral(Box<mistral::Model>),
}

/// Loaded non-quantized candle engine.
pub struct SafetensorsEngine {
    /// The dequantized forward graph for one of the supported
    /// architectures. Held by-value so the LoRA-merging `VarBuilder`
    /// can substitute any per-`Linear` weight at load time.
    pub model: LoadedModel,
    /// HF tokenizer loaded from `tokenizer.json`.
    pub tokenizer: Tokenizer,
    /// Device the weights live on.
    pub device: Device,
    /// Original llama-only config field, retained for the
    /// llama variant only — kept so existing callers can read e.g.
    /// `num_hidden_layers` without re-parsing JSON. `None` for
    /// non-llama architectures.
    pub config: Option<Config>,
    /// User-supplied or default context window in tokens.
    pub context_length: usize,
    /// dtype used to construct the cache + model. Cached so per-call
    /// resets stay byte-identical to the initial load.
    dtype: DType,
    /// EOS token id resolved at load time (config first, tokenizer fallback).
    eos_token_id: Option<u32>,
    /// Detected architecture, used by [`format_prompt`] and other
    /// arch-aware helpers.
    arch: Arch,
}

impl SafetensorsEngine {
    /// Load a Llama2 safetensors model from `HuggingFace` Hub,
    /// optionally merging a list of parsed PEFT `LoRA` adapters into
    /// the resulting `Llama` graph. When `adapters` is non-empty the
    /// `VarBuilder` is built through a
    /// [`crate::lora_backend::LoraMergingBackend`] so every targeted
    /// `Linear` weight has its `(B@A) * scale` delta added at load
    /// time; with an empty slice this is equivalent to the plain
    /// safetensors path Wave B introduced.
    ///
    /// # Errors
    /// See [`CandleLlmError`] variants — most failures are
    /// `ModelLoad` (download / parse) or `Unsupported`
    /// (non-`llama` `model_type`).
    #[allow(clippy::too_many_lines)] // single happy-path load function; per-arch dispatch lives inside.
    pub(crate) async fn load_with_adapters(
        opts: &CandleLlmOptions,
        adapters: &[crate::lora::LoadedAdapter],
    ) -> Result<Self, CandleLlmError> {
        let repo_id = opts.model_id.as_deref().ok_or_else(|| {
            CandleLlmError::InvalidOptions("model_id is required for engine init".into())
        })?;
        let revision = opts.revision.as_deref();

        let device = resolve_device(opts.device.as_deref())?;

        let layout = probe_repo_layout(repo_id, revision).await?;
        if !layout.has_safetensors() {
            return Err(CandleLlmError::ModelLoad(format!(
                "no safetensors weights found in repo '{repo_id}'"
            )));
        }

        let (config_path, tokenizer_path, weight_paths) =
            download_safetensors_assets(repo_id, revision, &layout).await?;

        let tokenizer = Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| CandleLlmError::ModelLoad(format!("failed to load tokenizer: {e}")))?;

        let parsed = parse_arch_config(&config_path)?;
        let arch = parsed.arch();
        let max_position_embeddings = parsed.max_position_embeddings();
        let dtype = resolve_dtype(opts, &device);
        let config_eos = parse_eos_token(&config_path);

        // Why: the blocking closure needs to take adapters by value, so
        // shallow-clone each layer table (tensors are Arc-backed inside
        // candle, so this only copies the per-layer wrappers).
        let adapters_owned: Vec<crate::lora::LoadedAdapter> = adapters
            .iter()
            .map(|a| crate::lora::LoadedAdapter {
                id: a.id.clone(),
                source_dir: a.source_dir.clone(),
                scale: a.scale,
                layers: a.layers.clone(),
            })
            .collect();

        let device_for_blocking = device.clone();
        let (model, llama_config) = tokio::task::spawn_blocking(move || {
            // Safety: `MmapedSafetensors::multi` is `unsafe` because
            // the mmap'd files must not be mutated while it is alive.
            // The HF cache files are append-only / immutable once
            // written, so this is sound.
            #[allow(unsafe_code)]
            let mmap = unsafe { candle_core::safetensors::MmapedSafetensors::multi(&weight_paths) }
                .map_err(|e| CandleLlmError::ModelLoad(format!("safetensors mmap failed: {e}")))?;

            let vb: VarBuilder<'static> = if adapters_owned.is_empty() {
                VarBuilder::from_backend(Box::new(mmap), dtype, device_for_blocking.clone())
            } else {
                let backend = crate::lora_backend::LoraMergingBackend::new(
                    mmap,
                    &adapters_owned,
                    &device_for_blocking,
                    dtype,
                )
                .map_err(|e| {
                    CandleLlmError::ModelLoad(format!("LoRA merging backend init failed: {e}"))
                })?;
                VarBuilder::from_backend(Box::new(backend), dtype, device_for_blocking.clone())
            };

            match parsed {
                ParsedConfig::Llama(cfg) => {
                    let model = Llama::load(vb, &cfg).map_err(|e| {
                        CandleLlmError::ModelLoad(format!("Llama::load failed: {e}"))
                    })?;
                    let cache =
                        Cache::new(true, dtype, &cfg, &device_for_blocking).map_err(|e| {
                            CandleLlmError::ModelLoad(format!("Cache::new failed: {e}"))
                        })?;
                    Ok::<_, CandleLlmError>((
                        LoadedModel::Llama(Box::new(model), Box::new(cache)),
                        Some(cfg),
                    ))
                }
                ParsedConfig::Qwen2(cfg) => {
                    let model = qwen2::ModelForCausalLM::new(&cfg, vb).map_err(|e| {
                        CandleLlmError::ModelLoad(format!("qwen2::Model::new failed: {e}"))
                    })?;
                    Ok((LoadedModel::Qwen2(Box::new(model)), None))
                }
                ParsedConfig::Mistral(cfg) => {
                    let model = mistral::Model::new(&cfg, vb).map_err(|e| {
                        CandleLlmError::ModelLoad(format!("mistral::Model::new failed: {e}"))
                    })?;
                    Ok((LoadedModel::Mistral(Box::new(model)), None))
                }
            }
        })
        .await
        .map_err(|e| CandleLlmError::ModelLoad(format!("blocking task failed: {e}")))??;

        let context_length = opts
            .context_length
            .unwrap_or(max_position_embeddings.min(4096));

        // Why: prefer config-supplied EOS, fall back to tokenizer lookups
        // covering Llama/Qwen2/Mistral conventions.
        let eos_token_id = config_eos.or_else(|| {
            tokenizer
                .token_to_id("</s>")
                .or_else(|| tokenizer.token_to_id("<|im_end|>"))
                .or_else(|| tokenizer.token_to_id("<|end_of_text|>"))
                .or_else(|| tokenizer.token_to_id("<|endoftext|>"))
                .or_else(|| tokenizer.token_to_id("<|eot_id|>"))
        });

        tracing::info!(
            repo = repo_id,
            context_length,
            device = ?opts.device,
            dtype = ?dtype,
            adapters = adapters.len(),
            arch = ?arch,
            "candle safetensors LLM engine loaded"
        );

        Ok(Self {
            model,
            tokenizer,
            device,
            config: llama_config,
            context_length,
            dtype,
            eos_token_id,
            arch,
        })
    }

    /// Reset the KV / mask caches so the next inference starts from
    /// a clean rotary position. Cheap (the rotary tables stay
    /// allocated). Dispatches on the loaded variant: Llama rebuilds
    /// its external [`Cache`]; Qwen2 / Mistral call their internal
    /// `clear_kv_cache`.
    fn reset_cache(&mut self) -> Result<(), CandleLlmError> {
        match &mut self.model {
            LoadedModel::Llama(_, cache) => {
                let cfg = self.config.as_ref().ok_or_else(|| {
                    CandleLlmError::Inference(
                        "llama variant missing cached Config for KV reset".into(),
                    )
                })?;
                **cache = Cache::new(true, self.dtype, cfg, &self.device)
                    .map_err(|e| CandleLlmError::Inference(format!("cache reset failed: {e}")))?;
            }
            LoadedModel::Qwen2(m) => m.clear_kv_cache(),
            LoadedModel::Mistral(m) => m.clear_kv_cache(),
        }
        Ok(())
    }

    /// Resolve a usable EOS token id, preferring the (possibly
    /// llama-specific) config field over the cached lookup from load.
    fn eos_token(&self) -> Option<u32> {
        if let Some(cfg) = self.config.as_ref()
            && let Some(eos) = cfg.eos_token_id.as_ref()
        {
            match eos {
                LlamaEosToks::Single(id) => return Some(*id),
                LlamaEosToks::Multiple(ids) => {
                    if let Some(first) = ids.first() {
                        return Some(*first);
                    }
                }
            }
        }
        self.eos_token_id
    }
}

/// Run the loaded model's forward pass for one decode step, returning
/// a 1-D `[vocab]` tensor of logits ready for `LogitsProcessor::sample`.
///
/// `index_pos` is the position of the first token in `input` inside the
/// running sequence (0 for prompt prefill, then `prompt_len + step` for
/// each generated token). Llama uses it as its rotary `index_pos`,
/// Qwen2 / Mistral use it as their `seqlen_offset`.
fn forward_step(
    model: &mut LoadedModel,
    input: &Tensor,
    index_pos: usize,
) -> Result<Tensor, CandleLlmError> {
    match model {
        LoadedModel::Llama(m, cache) => {
            let logits = m
                .forward(input, index_pos, cache)
                .map_err(|e| CandleLlmError::Inference(format!("forward pass failed: {e}")))?;
            // Why: Llama::forward already last-token-slices and returns
            // `[b, vocab]`; squeeze the batch dim to a `[vocab]` 1-D.
            logits
                .squeeze(0)
                .map_err(|e| CandleLlmError::Inference(format!("squeeze failed: {e}")))
        }
        LoadedModel::Qwen2(m) => {
            let logits = m
                .forward(input, index_pos)
                .map_err(|e| CandleLlmError::Inference(format!("forward pass failed: {e}")))?;
            squeeze_b1_vocab(&logits)
        }
        LoadedModel::Mistral(m) => {
            let logits = m
                .forward(input, index_pos)
                .map_err(|e| CandleLlmError::Inference(format!("forward pass failed: {e}")))?;
            squeeze_b1_vocab(&logits)
        }
    }
}

/// Squeeze a `[b, 1, vocab]` logits tensor down to a `[vocab]` 1-D so
/// it's directly samplable. Qwen2's `ModelForCausalLM::forward` and
/// Mistral's `Model::forward` both already perform the last-token
/// slice, so the seq dim is always length 1 here.
fn squeeze_b1_vocab(logits: &Tensor) -> Result<Tensor, CandleLlmError> {
    logits
        .i((0, 0))
        .map_err(|e| CandleLlmError::Inference(format!("logits squeeze failed: {e}")))
}

/// Format chat messages into an architecture-appropriate prompt.
///
/// Dispatches to the per-arch helper based on the engine's detected
/// architecture so the caller doesn't have to duplicate the match.
pub(crate) fn format_prompt(engine: &SafetensorsEngine, messages: &[(String, String)]) -> String {
    match engine.arch {
        Arch::Llama => format_prompt_llama(messages),
        Arch::Qwen2 => format_prompt_qwen2(messages),
        Arch::Mistral => format_prompt_mistral(messages),
    }
}

/// Llama2-style prompt template: `<s>[INST] <<SYS>>...<</SYS>> ... [/INST]`.
///
/// Why a separate formatter from the GGUF engine: the GGUF engine's
/// `<|system|>` / `<|end|>` tags are tuned for Phi-class models. Llama2
/// chat uses `[INST] ... [/INST]` with system prompts wrapped in
/// `<<SYS>>`. Mixing the two formats causes the model to ignore the
/// system message.
pub(crate) fn format_prompt_llama(messages: &[(String, String)]) -> String {
    let mut prompt = String::new();
    let mut system: Option<&str> = None;
    let mut first_user = true;
    for (role, content) in messages {
        match role.as_str() {
            "system" => {
                system = Some(content.as_str());
            }
            "user" => {
                if first_user {
                    prompt.push_str("<s>[INST] ");
                    if let Some(sys) = system {
                        prompt.push_str("<<SYS>>\n");
                        prompt.push_str(sys);
                        prompt.push_str("\n<</SYS>>\n\n");
                    }
                    prompt.push_str(content);
                    prompt.push_str(" [/INST] ");
                    first_user = false;
                } else {
                    prompt.push_str("<s>[INST] ");
                    prompt.push_str(content);
                    prompt.push_str(" [/INST] ");
                }
            }
            "assistant" => {
                prompt.push_str(content);
                prompt.push_str(" </s>");
            }
            _ => {
                prompt.push_str(content);
                prompt.push(' ');
            }
        }
    }
    prompt
}

/// Qwen2 `ChatML` template: `<|im_start|>role\n...content...<|im_end|>\n`,
/// terminated with an open assistant turn so the model continues.
pub(crate) fn format_prompt_qwen2(messages: &[(String, String)]) -> String {
    let mut prompt = String::new();
    for (role, content) in messages {
        let role_tag = match role.as_str() {
            "system" | "user" | "assistant" => role.as_str(),
            _ => "user",
        };
        prompt.push_str("<|im_start|>");
        prompt.push_str(role_tag);
        prompt.push('\n');
        prompt.push_str(content);
        prompt.push_str("<|im_end|>\n");
    }
    prompt.push_str("<|im_start|>assistant\n");
    prompt
}

/// Mistral instruction template: `<s>[INST] ...[/INST]`, with any
/// system message folded into the first user turn (Mistral has no
/// dedicated system role).
pub(crate) fn format_prompt_mistral(messages: &[(String, String)]) -> String {
    let mut prompt = String::new();
    let mut system: Option<&str> = None;
    let mut first_user = true;
    for (role, content) in messages {
        match role.as_str() {
            "system" => {
                system = Some(content.as_str());
            }
            "user" => {
                if first_user {
                    prompt.push_str("<s>[INST] ");
                    if let Some(sys) = system {
                        prompt.push_str(sys);
                        prompt.push_str("\n\n");
                    }
                    prompt.push_str(content);
                    prompt.push_str(" [/INST] ");
                    first_user = false;
                } else {
                    prompt.push_str("[INST] ");
                    prompt.push_str(content);
                    prompt.push_str(" [/INST] ");
                }
            }
            "assistant" => {
                prompt.push_str(content);
                prompt.push_str("</s> ");
            }
            _ => {
                prompt.push_str(content);
                prompt.push(' ');
            }
        }
    }
    prompt
}

/// Run a non-streaming autoregressive generation on the loaded
/// safetensors model. Mirrors [`crate::provider`]'s GGUF
/// `engine::generate` signature so the provider dispatch enum can
/// call either.
pub(crate) fn infer_engine(
    engine: &mut SafetensorsEngine,
    prompt: &str,
    max_tokens: usize,
    temperature: Option<f64>,
    top_p: Option<f64>,
) -> Result<(String, usize), CandleLlmError> {
    engine.reset_cache()?;

    let encoding = engine
        .tokenizer
        .encode(prompt, true)
        .map_err(|e| CandleLlmError::Inference(format!("tokenization failed: {e}")))?;

    let prompt_tokens = encoding.get_ids().to_vec();
    let max_prompt = engine.context_length.saturating_sub(max_tokens);
    let prompt_tokens = if prompt_tokens.len() > max_prompt {
        tracing::warn!(
            prompt_len = prompt_tokens.len(),
            max_prompt,
            "truncating prompt to fit context window"
        );
        prompt_tokens[prompt_tokens.len() - max_prompt..].to_vec()
    } else {
        prompt_tokens
    };

    let eos_token = engine.eos_token();
    let temp = temperature.unwrap_or(0.7);
    let mut logits_processor = LogitsProcessor::new(42, Some(temp), top_p);

    let input = Tensor::new(prompt_tokens.as_slice(), &engine.device)
        .map_err(|e| CandleLlmError::Inference(format!("tensor creation failed: {e}")))?
        .unsqueeze(0)
        .map_err(|e| CandleLlmError::Inference(format!("unsqueeze failed: {e}")))?;

    let logits = forward_step(&mut engine.model, &input, 0)?;

    let mut next_token = logits_processor
        .sample(&logits)
        .map_err(|e| CandleLlmError::Inference(format!("sampling failed: {e}")))?;

    let mut generated_tokens: Vec<u32> = Vec::with_capacity(max_tokens);
    let prompt_len = prompt_tokens.len();

    for index_pos in prompt_len..(prompt_len + max_tokens) {
        if eos_token.is_some_and(|eos| next_token == eos) {
            break;
        }
        generated_tokens.push(next_token);

        let input = Tensor::new(&[next_token], &engine.device)
            .map_err(|e| CandleLlmError::Inference(format!("tensor creation failed: {e}")))?
            .unsqueeze(0)
            .map_err(|e| CandleLlmError::Inference(format!("unsqueeze failed: {e}")))?;

        let logits = forward_step(&mut engine.model, &input, index_pos)?;

        next_token = logits_processor
            .sample(&logits)
            .map_err(|e| CandleLlmError::Inference(format!("sampling failed: {e}")))?;
    }

    let text = engine
        .tokenizer
        .decode(&generated_tokens, true)
        .map_err(|e| CandleLlmError::Inference(format!("detokenization failed: {e}")))?;
    Ok((text, generated_tokens.len()))
}

/// Streaming variant: feeds each decoded token chunk into `on_token`.
pub(crate) fn infer_stream_engine<F>(
    engine: &mut SafetensorsEngine,
    prompt: &str,
    max_tokens: usize,
    temperature: Option<f64>,
    top_p: Option<f64>,
    mut on_token: F,
) -> Result<usize, CandleLlmError>
where
    F: FnMut(String),
{
    engine.reset_cache()?;

    let encoding = engine
        .tokenizer
        .encode(prompt, true)
        .map_err(|e| CandleLlmError::Inference(format!("tokenization failed: {e}")))?;
    let prompt_tokens = encoding.get_ids().to_vec();
    let max_prompt = engine.context_length.saturating_sub(max_tokens);
    let prompt_tokens = if prompt_tokens.len() > max_prompt {
        prompt_tokens[prompt_tokens.len() - max_prompt..].to_vec()
    } else {
        prompt_tokens
    };

    let eos_token = engine.eos_token();
    let temp = temperature.unwrap_or(0.7);
    let mut logits_processor = LogitsProcessor::new(42, Some(temp), top_p);

    let input = Tensor::new(prompt_tokens.as_slice(), &engine.device)
        .map_err(|e| CandleLlmError::Inference(format!("tensor creation failed: {e}")))?
        .unsqueeze(0)
        .map_err(|e| CandleLlmError::Inference(format!("unsqueeze failed: {e}")))?;
    let logits = forward_step(&mut engine.model, &input, 0)?;
    let mut next_token = logits_processor
        .sample(&logits)
        .map_err(|e| CandleLlmError::Inference(format!("sampling failed: {e}")))?;

    let mut token_count = 0usize;
    let prompt_len = prompt_tokens.len();
    for index_pos in prompt_len..(prompt_len + max_tokens) {
        if eos_token.is_some_and(|eos| next_token == eos) {
            break;
        }
        let text = engine
            .tokenizer
            .decode(&[next_token], true)
            .map_err(|e| CandleLlmError::Inference(format!("detokenization failed: {e}")))?;
        if !text.is_empty() {
            on_token(text);
        }
        token_count += 1;

        let input = Tensor::new(&[next_token], &engine.device)
            .map_err(|e| CandleLlmError::Inference(format!("tensor creation failed: {e}")))?
            .unsqueeze(0)
            .map_err(|e| CandleLlmError::Inference(format!("unsqueeze failed: {e}")))?;
        let logits = forward_step(&mut engine.model, &input, index_pos)?;
        next_token = logits_processor
            .sample(&logits)
            .map_err(|e| CandleLlmError::Inference(format!("sampling failed: {e}")))?;
    }

    Ok(token_count)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detects_single_safetensors() {
        assert!(!is_sharded_safetensors("model.safetensors"));
        assert!(is_sharded_safetensors("model-00001-of-00002.safetensors"));
        assert!(is_sharded_safetensors("model-00002-of-00002.safetensors"));
    }

    #[test]
    fn rejects_non_canonical_safetensors() {
        assert!(!is_sharded_safetensors("adapter_model.safetensors"));
        assert!(!is_sharded_safetensors("model-abc.safetensors"));
        assert!(!is_sharded_safetensors("model.safetensors.tmp"));
        assert!(!is_sharded_safetensors("model--of-.safetensors"));
    }

    fn scratch_dir() -> std::path::PathBuf {
        let dir = std::env::var("HOME")
            .map(|h| std::path::PathBuf::from(h).join(".cache/blazen-candle-arches"))
            .expect("HOME env required");
        std::fs::create_dir_all(&dir).expect("mkdir cache");
        dir
    }

    #[test]
    fn unknown_arch_returns_unsupported_variant() {
        let cfg_path = scratch_dir().join("phi_probe_config.json");
        let body = serde_json::json!({
            "model_type": "phi3",
            "hidden_size": 1024,
        });
        std::fs::write(&cfg_path, serde_json::to_vec(&body).unwrap()).expect("write cfg");

        let err = parse_arch_config(&cfg_path).expect_err("should reject phi3");
        assert!(
            matches!(err, CandleLlmError::Unsupported(_)),
            "expected Unsupported, got {err:?}"
        );
        assert!(
            err.to_string().contains("phi3"),
            "error should mention model_type: {err}"
        );
    }

    #[test]
    fn llama_config_parses() {
        let cfg_path = scratch_dir().join("llama_probe_config.json");
        let body = serde_json::json!({
            "model_type": "llama",
            "hidden_size": 4096,
            "intermediate_size": 11008,
            "vocab_size": 32000,
            "num_hidden_layers": 32,
            "num_attention_heads": 32,
            "num_key_value_heads": 32,
            "rms_norm_eps": 1.0e-5,
            "rope_theta": 10000.0,
            "max_position_embeddings": 4096,
        });
        std::fs::write(&cfg_path, serde_json::to_vec(&body).unwrap()).expect("write cfg");

        let parsed = parse_arch_config(&cfg_path).expect("llama parse");
        let ParsedConfig::Llama(cfg) = parsed else {
            panic!("expected Llama variant");
        };
        assert_eq!(cfg.hidden_size, 4096);
        assert_eq!(cfg.num_hidden_layers, 32);
    }

    #[test]
    fn qwen2_config_parses_with_tolerant_shim() {
        let cfg_path = scratch_dir().join("qwen2_probe_config.json");
        // Why: real Qwen2.5 configs frequently set use_sliding_window=false
        // and omit max_window_layers / tie_word_embeddings entirely.
        let body = serde_json::json!({
            "model_type": "qwen2",
            "vocab_size": 151_936,
            "hidden_size": 1536,
            "intermediate_size": 8960,
            "num_hidden_layers": 28,
            "num_attention_heads": 12,
            "num_key_value_heads": 2,
            "max_position_embeddings": 32768,
            "rope_theta": 1_000_000.0,
            "rms_norm_eps": 1.0e-6,
            "use_sliding_window": false,
            "hidden_act": "silu",
        });
        std::fs::write(&cfg_path, serde_json::to_vec(&body).unwrap()).expect("write cfg");

        let parsed = parse_arch_config(&cfg_path).expect("qwen2 parse");
        let ParsedConfig::Qwen2(cfg) = parsed else {
            panic!("expected Qwen2 variant");
        };
        assert_eq!(cfg.hidden_size, 1536);
        assert_eq!(cfg.num_hidden_layers, 28);
        assert!(!cfg.use_sliding_window);
    }

    #[test]
    fn mistral_config_parses() {
        let cfg_path = scratch_dir().join("mistral_probe_config.json");
        let body = serde_json::json!({
            "model_type": "mistral",
            "vocab_size": 32000,
            "hidden_size": 4096,
            "intermediate_size": 14336,
            "num_hidden_layers": 32,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "hidden_act": "silu",
            "max_position_embeddings": 32768,
            "rms_norm_eps": 1.0e-5,
            "rope_theta": 10000.0,
            "sliding_window": 4096,
        });
        std::fs::write(&cfg_path, serde_json::to_vec(&body).unwrap()).expect("write cfg");

        let parsed = parse_arch_config(&cfg_path).expect("mistral parse");
        let ParsedConfig::Mistral(cfg) = parsed else {
            panic!("expected Mistral variant");
        };
        assert_eq!(cfg.hidden_size, 4096);
        assert_eq!(cfg.sliding_window, Some(4096));
    }

    #[test]
    fn format_prompt_llama_uses_llama2_template() {
        let msgs = vec![
            ("system".to_string(), "You are helpful.".to_string()),
            ("user".to_string(), "Hi!".to_string()),
        ];
        let prompt = format_prompt_llama(&msgs);
        assert!(prompt.contains("[INST]"));
        assert!(prompt.contains("<<SYS>>"));
        assert!(prompt.contains("You are helpful."));
        assert!(prompt.contains("Hi!"));
        assert!(prompt.contains("[/INST]"));
    }

    #[test]
    fn format_prompt_qwen2_uses_chatml() {
        let msgs = vec![
            ("system".to_string(), "Be terse.".to_string()),
            ("user".to_string(), "Hello".to_string()),
        ];
        let prompt = format_prompt_qwen2(&msgs);
        assert!(prompt.contains("<|im_start|>system\nBe terse.<|im_end|>"));
        assert!(prompt.contains("<|im_start|>user\nHello<|im_end|>"));
        assert!(prompt.ends_with("<|im_start|>assistant\n"));
    }

    #[test]
    fn format_prompt_mistral_folds_system_into_first_user() {
        let msgs = vec![
            ("system".to_string(), "Be friendly.".to_string()),
            ("user".to_string(), "Hi".to_string()),
            ("assistant".to_string(), "Hello!".to_string()),
            ("user".to_string(), "How are you?".to_string()),
        ];
        let prompt = format_prompt_mistral(&msgs);
        assert!(prompt.starts_with("<s>[INST] Be friendly.\n\nHi [/INST] "));
        assert!(prompt.contains("Hello!</s>"));
        assert!(prompt.contains("[INST] How are you? [/INST]"));
    }
}
