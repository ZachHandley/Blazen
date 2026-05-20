//! Auto-detect the right local-inference backend for a Hugging Face repo.
//!
//! Casual users say "give me this repo id, you pick the engine"; power users
//! still construct providers directly via [`crate::ModelManager::register`].
//!
//! The detection is metadata-only: a single `GET /api/models/{repo}` against
//! the Hub returns the sibling file list and (for most repos) per-file sizes,
//! so we can pick a backend and produce a memory estimate without touching the
//! weights. Actual download/load happens later when the manager calls
//! [`blazen_llm::LocalModel::load`].
//!
//! The selection rules are documented on [`choose_backend`]. They are pure and
//! unit-testable; only [`detect_layout`] and
//! [`crate::ModelManager::load_from_hf`] reach the network.

use std::path::PathBuf;

use blazen_llm::{BlazenError, Pool};

#[cfg(feature = "hf-loader")]
use serde::Deserialize;

/// Local inference backend identifier returned by [`choose_backend`] and
/// [`crate::ModelManager::load_from_hf`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackendHint {
    /// mistral.rs — broad architecture coverage, handles both safetensors and
    /// GGUF, supports vision/multimodal models.
    Mistralrs,
    /// candle — pure-Rust, supports safetensors and GGUF for the subset of
    /// architectures candle ships.
    Candle,
    /// llama.cpp — GGUF only, best CPU performance and lowest memory.
    Llamacpp,
}

impl BackendHint {
    /// Lower-case stable string form (`"mistralrs"`, `"candle"`, `"llamacpp"`).
    #[must_use]
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Mistralrs => "mistralrs",
            Self::Candle => "candle",
            Self::Llamacpp => "llamacpp",
        }
    }
}

impl std::str::FromStr for BackendHint {
    type Err = BlazenError;

    /// Case-insensitive parse of [`Self::as_str`], also accepting
    /// `"mistral.rs"`, `"mistral_rs"`, `"llama.cpp"`, `"llama_cpp"`.
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_ascii_lowercase().as_str() {
            "mistralrs" | "mistral.rs" | "mistral_rs" => Ok(Self::Mistralrs),
            "candle" => Ok(Self::Candle),
            "llamacpp" | "llama.cpp" | "llama_cpp" => Ok(Self::Llamacpp),
            other => Err(BlazenError::validation(format!(
                "unknown backend '{other}': expected one of mistralrs, candle, llamacpp"
            ))),
        }
    }
}

/// User-supplied knobs for [`crate::ModelManager::load_from_hf`] and
/// [`detect_layout`].
#[derive(Debug, Clone, Default)]
pub struct HfLoadOptions {
    /// Force a specific backend; skips inference of the chosen engine but
    /// still runs [`detect_layout`] for memory sizing.
    pub backend_hint: Option<BackendHint>,
    /// Git revision (branch, tag, or commit sha). Defaults to the repo's
    /// default branch.
    pub revision: Option<String>,
    /// Hugging Face access token. When `None`, falls back to the `HF_TOKEN`
    /// environment variable, then to anonymous access.
    pub hf_token: Option<String>,
    /// Override the on-disk cache directory used by `hf-hub`. Defaults to
    /// `$HF_HOME` or `~/.cache/huggingface/` per upstream conventions.
    pub cache_dir: Option<PathBuf>,
    /// Device specifier string forwarded to the chosen provider (`"cpu"`,
    /// `"cuda:0"`, `"metal"`, …). When `None`, the provider's own default
    /// applies (currently `"cpu"`).
    pub device: Option<String>,
    /// Explicit GGUF filename for repos that ship multiple quantizations.
    /// Required when [`choose_backend`] would otherwise pick Llamacpp from a
    /// repo with several `*.gguf` siblings.
    pub gguf_file: Option<String>,
    /// Override the manager's memory budgeting estimate. When `None`,
    /// [`crate::ModelManager::load_from_hf`] sums the chosen backend's weight
    /// files from repo metadata, rounded up to the nearest MB.
    pub memory_estimate_bytes: Option<u64>,
    /// Pool that the registered model targets. When `None`, defaults to
    /// [`Pool::Cpu`].
    pub pool: Option<Pool>,
}

/// Snapshot of a Hugging Face repo's relevant files, returned by
/// [`detect_layout`] and consumed by [`choose_backend`].
#[derive(Debug, Clone, Default)]
pub struct DetectedLayout {
    /// All `*.gguf` siblings, in repo declaration order.
    pub gguf_files: Vec<String>,
    /// All `*.safetensors` siblings, in repo declaration order.
    pub safetensors_files: Vec<String>,
    /// Per-weight-file size in bytes when the Hub exposes them. Keys are
    /// the same `rfilename`s as in `gguf_files` / `safetensors_files`;
    /// entries with `None` mean the Hub omitted size for that file.
    pub file_sizes: Vec<(String, Option<u64>)>,
    /// Sum of weight-file sizes in bytes when the Hub exposes them; `None`
    /// when the API response omits sizes (rare; mostly affects datasets).
    pub total_weight_bytes: Option<u64>,
    /// Whether `config.json` is present (signals a real model repo as
    /// opposed to a PEFT-only adapter repo).
    pub config_json_present: bool,
    /// Whether `adapter_config.json` is present. Adapter-only repos must be
    /// mounted via [`crate::ModelManager::load_adapter`] on a base model;
    /// [`crate::ModelManager::load_from_hf`] rejects them with a hint.
    pub adapter_config_json_present: bool,
}

/// Pure backend selection given a probed [`DetectedLayout`].
///
/// Rules (first match wins):
///
/// 1. `hint` is `Some` → return it (caller wins, no auto-detection).
/// 2. `gguf_override` is `Some` OR (`gguf_files` non-empty AND
///    `safetensors_files` empty) → [`BackendHint::Llamacpp`].
/// 3. Both `gguf_files` AND `safetensors_files` non-empty →
///    [`BackendHint::Mistralrs`] (mistral.rs handles both formats and adds
///    vision support).
/// 4. Only `safetensors_files` non-empty → [`BackendHint::Mistralrs`]
///    (broader architecture coverage than candle; users wanting candle pass
///    `hint = Candle`).
/// 5. PEFT-only (`adapter_config_json_present` AND no weight files) →
///    [`BlazenError::Validation`] suggesting [`crate::ModelManager::load_adapter`].
/// 6. Otherwise → [`BlazenError::Validation`] listing what was found.
///
/// # Errors
/// See rules 5 and 6.
pub fn choose_backend(
    layout: &DetectedLayout,
    hint: Option<BackendHint>,
    gguf_override: Option<&str>,
) -> Result<BackendHint, BlazenError> {
    if let Some(h) = hint {
        return Ok(h);
    }

    let has_gguf = !layout.gguf_files.is_empty();
    let has_st = !layout.safetensors_files.is_empty();

    if gguf_override.is_some() || (has_gguf && !has_st) {
        return Ok(BackendHint::Llamacpp);
    }
    if has_gguf && has_st {
        return Ok(BackendHint::Mistralrs);
    }
    if has_st {
        return Ok(BackendHint::Mistralrs);
    }

    if layout.adapter_config_json_present {
        return Err(BlazenError::validation(
            "repo contains only PEFT adapter files (adapter_config.json) and no \
             base weights; mount it on a registered base model via \
             ModelManager::load_adapter instead of load_from_hf",
        ));
    }

    Err(BlazenError::validation(format!(
        "repo contains no recognisable model weights \
         (no *.gguf, no *.safetensors); \
         config.json present: {}, adapter_config.json present: {}",
        layout.config_json_present, layout.adapter_config_json_present,
    )))
}

/// Probe a Hugging Face model repo for the files relevant to backend
/// selection.
///
/// This is a metadata-only request — no weights are downloaded. The
/// returned [`DetectedLayout`] feeds [`choose_backend`].
///
/// # Errors
/// - [`BlazenError::Validation`] when `repo` is empty or the Hub returns a
///   client error (404, 401, …).
/// - [`BlazenError::Internal`] when the Hub is unreachable or returns a
///   response shape we cannot parse.
#[cfg(feature = "hf-loader")]
pub async fn detect_layout(
    repo: &str,
    options: &HfLoadOptions,
) -> Result<DetectedLayout, BlazenError> {
    if repo.trim().is_empty() {
        return Err(BlazenError::validation("repo id must not be empty"));
    }

    let token = options
        .hf_token
        .clone()
        .or_else(|| std::env::var("HF_TOKEN").ok());

    let mut builder = hf_hub::api::tokio::ApiBuilder::new().with_progress(false);
    if let Some(cache) = options.cache_dir.clone() {
        builder = builder.with_cache_dir(cache);
    }
    if let Some(tok) = token {
        builder = builder.with_token(Some(tok));
    }

    let api = builder
        .build()
        .map_err(|e| BlazenError::internal(format!("hf-hub build: {e}")))?;

    let repo_handle = if let Some(rev) = options.revision.as_deref() {
        api.repo(hf_hub::Repo::with_revision(
            repo.to_string(),
            hf_hub::RepoType::Model,
            rev.to_string(),
        ))
    } else {
        api.model(repo.to_string())
    };

    // Why: hf-hub's `RepoInfo` exposes only `rfilename` per sibling, dropping
    // the `size` field present on the upstream JSON. We hit `info_request()`
    // directly (which returns the prepared reqwest::RequestBuilder) and parse
    // a richer shape ourselves so the manager can compute a memory estimate
    // without per-file HEAD probes (HF's tree endpoint returns 405 for those).
    let resp = repo_handle
        .info_request()
        .send()
        .await
        .map_err(|e| BlazenError::internal(format!("hf-hub request for '{repo}': {e}")))?;

    let status = resp.status();
    let body = resp
        .text()
        .await
        .map_err(|e| BlazenError::internal(format!("hf-hub read body for '{repo}': {e}")))?;

    if !status.is_success() {
        // Why: client errors (404 unknown repo, 401 gated repo) are user input
        // problems; surface them as validation rather than internal.
        if status.is_client_error() {
            return Err(BlazenError::validation(format!(
                "hf-hub returned {status} for repo '{repo}': {body}"
            )));
        }
        return Err(BlazenError::internal(format!(
            "hf-hub returned {status} for repo '{repo}': {body}"
        )));
    }

    let parsed: HubModelInfo = serde_json::from_str(&body).map_err(|e| {
        BlazenError::internal(format!(
            "failed to parse hf-hub /api/models response for '{repo}': {e}"
        ))
    })?;

    Ok(layout_from_siblings(&parsed.siblings))
}

#[cfg(feature = "hf-loader")]
fn layout_from_siblings(siblings: &[HubSibling]) -> DetectedLayout {
    let mut layout = DetectedLayout::default();
    let mut total: u64 = 0;
    let mut any_size = false;

    for s in siblings {
        let name = &s.rfilename;
        if name.eq_ignore_ascii_case("config.json") {
            layout.config_json_present = true;
        }
        if name.eq_ignore_ascii_case("adapter_config.json") {
            layout.adapter_config_json_present = true;
        }
        let ext = std::path::Path::new(name)
            .extension()
            .and_then(|e| e.to_str())
            .map(str::to_ascii_lowercase);
        match ext.as_deref() {
            Some("gguf") => {
                layout.gguf_files.push(name.clone());
                layout.file_sizes.push((name.clone(), s.size));
                if let Some(sz) = s.size {
                    total = total.saturating_add(sz);
                    any_size = true;
                }
            }
            Some("safetensors") => {
                layout.safetensors_files.push(name.clone());
                layout.file_sizes.push((name.clone(), s.size));
                if let Some(sz) = s.size {
                    total = total.saturating_add(sz);
                    any_size = true;
                }
            }
            _ => {}
        }
    }

    layout.total_weight_bytes = if any_size { Some(total) } else { None };
    layout
}

/// Round `bytes` up to the nearest 1 MiB. Used to stabilise the memory
/// estimate the manager records — sub-MB jitter would churn LRU decisions.
#[allow(dead_code)] // Why: only consumed by load_from_hf (hf-loader feature) + unit tests.
#[must_use]
pub(crate) fn round_up_to_mb(bytes: u64) -> u64 {
    const MIB: u64 = 1024 * 1024;
    if bytes == 0 {
        return 0;
    }
    bytes.div_ceil(MIB).saturating_mul(MIB)
}

/// Sum the bytes of the files the chosen backend will actually load.
#[allow(dead_code)] // Why: only consumed by load_from_hf (hf-loader feature) + unit tests.
#[must_use]
pub(crate) fn estimate_backend_bytes(
    backend: BackendHint,
    layout: &DetectedLayout,
    siblings_sizes: &[(String, Option<u64>)],
    gguf_override: Option<&str>,
) -> Option<u64> {
    let mut total: u64 = 0;
    let mut any = false;
    let want_gguf: Vec<&str> = match backend {
        BackendHint::Llamacpp => {
            if let Some(name) = gguf_override {
                vec![name]
            } else if let Some(first) = layout.gguf_files.first() {
                vec![first.as_str()]
            } else {
                return None;
            }
        }
        BackendHint::Mistralrs | BackendHint::Candle => {
            if !layout.safetensors_files.is_empty() {
                layout
                    .safetensors_files
                    .iter()
                    .map(String::as_str)
                    .collect()
            } else if let Some(first) = layout.gguf_files.first() {
                vec![first.as_str()]
            } else {
                return None;
            }
        }
    };

    for name in want_gguf {
        if let Some((_, Some(sz))) = siblings_sizes.iter().find(|(n, _)| n.as_str() == name) {
            total = total.saturating_add(*sz);
            any = true;
        }
    }

    if any {
        Some(round_up_to_mb(total))
    } else {
        None
    }
}

#[cfg(feature = "hf-loader")]
#[derive(Debug, Deserialize)]
struct HubModelInfo {
    #[serde(default)]
    siblings: Vec<HubSibling>,
}

#[cfg(feature = "hf-loader")]
#[derive(Debug, Deserialize)]
pub(crate) struct HubSibling {
    pub(crate) rfilename: String,
    // Why: the Hub returns `size` for normal model repos but omits it for some
    // dataset-style listings; serde_json default keeps the field optional so we
    // do not 500 on those responses.
    #[serde(default)]
    pub(crate) size: Option<u64>,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn layout(gguf: &[&str], st: &[&str]) -> DetectedLayout {
        let gguf_v: Vec<String> = gguf.iter().map(|s| (*s).to_string()).collect();
        let st_v: Vec<String> = st.iter().map(|s| (*s).to_string()).collect();
        let mut file_sizes: Vec<(String, Option<u64>)> = Vec::new();
        for n in gguf_v.iter().chain(st_v.iter()) {
            file_sizes.push((n.clone(), None));
        }
        DetectedLayout {
            gguf_files: gguf_v,
            safetensors_files: st_v,
            file_sizes,
            total_weight_bytes: None,
            config_json_present: !st.is_empty() || !gguf.is_empty(),
            adapter_config_json_present: false,
        }
    }

    #[test]
    fn backend_hint_overrides_detection() {
        let l = layout(&[], &["model-00001-of-00002.safetensors"]);
        let got = choose_backend(&l, Some(BackendHint::Candle), None).unwrap();
        assert_eq!(got, BackendHint::Candle);

        let l_gguf = layout(&["q4_k_m.gguf"], &[]);
        let got = choose_backend(&l_gguf, Some(BackendHint::Mistralrs), None).unwrap();
        assert_eq!(got, BackendHint::Mistralrs);
    }

    #[test]
    fn gguf_only_repo_chooses_llamacpp() {
        let l = layout(&["llama-3.2-1b-q4_k_m.gguf"], &[]);
        assert_eq!(
            choose_backend(&l, None, None).unwrap(),
            BackendHint::Llamacpp,
        );
    }

    #[test]
    fn safetensors_only_repo_chooses_mistralrs() {
        let l = layout(
            &[],
            &[
                "model-00001-of-00002.safetensors",
                "model-00002-of-00002.safetensors",
            ],
        );
        assert_eq!(
            choose_backend(&l, None, None).unwrap(),
            BackendHint::Mistralrs,
        );
    }

    #[test]
    fn both_formats_repo_chooses_mistralrs() {
        let l = layout(&["q4_k_m.gguf"], &["model.safetensors"]);
        assert_eq!(
            choose_backend(&l, None, None).unwrap(),
            BackendHint::Mistralrs,
        );
    }

    #[test]
    fn gguf_override_forces_llamacpp_even_with_safetensors_present() {
        let l = layout(&["q4_k_m.gguf"], &["model.safetensors"]);
        assert_eq!(
            choose_backend(&l, None, Some("q4_k_m.gguf")).unwrap(),
            BackendHint::Llamacpp,
        );
    }

    #[test]
    fn peft_only_repo_returns_validation_error() {
        let l = DetectedLayout {
            adapter_config_json_present: true,
            ..DetectedLayout::default()
        };
        let err = choose_backend(&l, None, None).expect_err("PEFT-only must fail");
        let msg = err.to_string();
        assert!(
            msg.contains("load_adapter"),
            "expected suggestion to use load_adapter, got: {msg}"
        );
        assert!(
            msg.contains("PEFT") || msg.contains("adapter"),
            "expected 'PEFT' or 'adapter' in error, got: {msg}"
        );
    }

    #[test]
    fn empty_repo_returns_validation_error_with_file_list() {
        let l = DetectedLayout::default();
        let err = choose_backend(&l, None, None).expect_err("empty repo must fail");
        let msg = err.to_string();
        assert!(
            msg.contains("no recognisable model weights"),
            "expected weight diagnostic, got: {msg}"
        );
        assert!(
            msg.contains("*.gguf") && msg.contains("*.safetensors"),
            "expected both formats named in error, got: {msg}"
        );
    }

    #[test]
    fn backend_hint_as_str_and_from_str_roundtrip() {
        use std::str::FromStr;
        for h in [
            BackendHint::Mistralrs,
            BackendHint::Candle,
            BackendHint::Llamacpp,
        ] {
            assert_eq!(BackendHint::from_str(h.as_str()).unwrap(), h);
        }
        assert_eq!(
            BackendHint::from_str("MISTRALRS").unwrap(),
            BackendHint::Mistralrs,
        );
        assert_eq!(
            BackendHint::from_str("Llama.cpp").unwrap(),
            BackendHint::Llamacpp,
        );
        BackendHint::from_str("not-a-backend")
            .expect_err("unknown backend must be a validation error");
    }

    #[test]
    fn round_up_to_mb_is_stable() {
        const MIB: u64 = 1024 * 1024;
        assert_eq!(round_up_to_mb(0), 0);
        assert_eq!(round_up_to_mb(1), MIB);
        assert_eq!(round_up_to_mb(MIB), MIB);
        assert_eq!(round_up_to_mb(MIB + 1), 2 * MIB);
        assert_eq!(round_up_to_mb(10 * MIB), 10 * MIB);
    }

    #[test]
    fn estimate_backend_bytes_llamacpp_picks_first_gguf_when_no_override() {
        let l = layout(&["a-q4.gguf", "b-q8.gguf"], &[]);
        let sizes = vec![
            ("a-q4.gguf".to_string(), Some(4 * 1024 * 1024 + 7)),
            ("b-q8.gguf".to_string(), Some(8 * 1024 * 1024)),
        ];
        let got = estimate_backend_bytes(BackendHint::Llamacpp, &l, &sizes, None);
        // 4 MiB + 7 bytes rounds up to 5 MiB.
        assert_eq!(got, Some(5 * 1024 * 1024));
    }

    #[test]
    fn estimate_backend_bytes_llamacpp_uses_override_when_given() {
        let l = layout(&["a-q4.gguf", "b-q8.gguf"], &[]);
        let sizes = vec![
            ("a-q4.gguf".to_string(), Some(1024 * 1024)),
            ("b-q8.gguf".to_string(), Some(3 * 1024 * 1024)),
        ];
        let got = estimate_backend_bytes(BackendHint::Llamacpp, &l, &sizes, Some("b-q8.gguf"));
        assert_eq!(got, Some(3 * 1024 * 1024));
    }

    #[test]
    fn estimate_backend_bytes_mistralrs_sums_all_safetensors() {
        let l = layout(
            &[],
            &[
                "model-00001-of-00002.safetensors",
                "model-00002-of-00002.safetensors",
            ],
        );
        let sizes = vec![
            (
                "model-00001-of-00002.safetensors".to_string(),
                Some(2 * 1024 * 1024),
            ),
            (
                "model-00002-of-00002.safetensors".to_string(),
                Some(2 * 1024 * 1024),
            ),
        ];
        let got = estimate_backend_bytes(BackendHint::Mistralrs, &l, &sizes, None);
        assert_eq!(got, Some(4 * 1024 * 1024));
    }

    #[test]
    fn estimate_backend_bytes_returns_none_when_sizes_missing() {
        let l = layout(&["a.gguf"], &[]);
        let sizes = vec![("a.gguf".to_string(), None)];
        assert!(estimate_backend_bytes(BackendHint::Llamacpp, &l, &sizes, None).is_none());
    }
}
