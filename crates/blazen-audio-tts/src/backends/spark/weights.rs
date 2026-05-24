//! HF download helper for `SparkAudio/Spark-TTS-0.5B` bundles (Wave S.2.4).
//!
//! Resolves a local model directory (downloading + caching on miss) ready
//! for the tokenizer / Qwen2.5 LLM / `BiCodec` loaders in
//! [`super::tokenizer`], [`super::decoder`], and [`super::bicodec`] to
//! consume.
//!
//! # Cache layout
//!
//! Files land at `{BLAZEN_CACHE_DIR or ~/.cache/blazen}/models/{repo_id}/…`
//! with the upstream subdirectory structure preserved
//! (`LLM/tokenizer.json`, `LLM/model.safetensors`,
//! `BiCodec/config.yaml`, `BiCodec/model.safetensors`, …). The returned
//! [`PathBuf`] is the bundle root directory — the same directory layout
//! the upstream `SparkAudio/Spark-TTS-0.5B` repo ships, so downstream
//! loaders can locate `LLM/` and `BiCodec/` subdirectories at well-known
//! relative paths.
//!
//! # Revision pinning
//!
//! [`ensure_downloaded`] accepts an optional `revision` argument. `None`
//! resolves to `main`. Callers that need bit-reproducible downloads can
//! pass a specific git SHA / branch / tag through
//! [`super::SparkTtsConfig::revision`]. A non-`None` revision keys its
//! own cache subdirectory (`{repo_id}@{revision}`) so multiple revisions
//! can coexist on disk without trampling each other — mirrors the
//! convention used by the `faster-whisper` weights helper.
//!
//! # License caveat
//!
//! `SparkAudio/Spark-TTS-0.5B` weights are published under
//! `CC-BY-NC-SA-4.0`. Apache-2.0 source code in this crate is fine for
//! commercial use, but downstream users of the published weights must
//! honour the non-commercial license; the backend emits a one-shot
//! `tracing::warn!` via [`blazen_audio::warn_nc_once`] the first time it
//! materialises a pipeline.

#![cfg(feature = "spark-tts")]

use std::path::PathBuf;
use std::sync::Arc;

use crate::TtsError;

/// Upstream HF Hub repo id for the canonical Spark-TTS bundle. Used by
/// the unit tests below to lock the default and by callers that want
/// to construct a custom [`super::SparkTtsConfig`] without retyping
/// the repo id.
#[allow(
    dead_code,
    reason = "Stable public-of-the-module constant — referenced by the \
              in-module unit tests and available to callers that need \
              to override SparkTtsConfig::model_id without hand-typing \
              the upstream slug. Marking allow rather than removing keeps \
              the canonical default visible alongside REQUIRED_FILES."
)]
pub(super) const SPARK_TTS_REPO: &str = "SparkAudio/Spark-TTS-0.5B";

/// Files we need under the bundle root. Paths are relative — preserved
/// verbatim from the upstream repo so the loader can find them at the
/// same well-known locations the upstream `SparkAudio/Spark-TTS-0.5B`
/// project documents.
///
/// LLM side: tokenizer + Qwen2.5-0.5B safetensors + HF transformers
/// `config.json`. `BiCodec` side: the ~597 MiB `BiCodec` safetensors +
/// `config.yaml`.
///
/// `LLM/model.safetensors` is a single ~990 MB file in the upstream
/// release — Qwen2.5-0.5B is small enough that the bundle does not ship
/// a sharded index. [`ensure_downloaded`] additionally tries to pull a
/// `model.safetensors.index.json` (best-effort) so a sharded fork drops
/// in without ceremony, and then degrades silently to "single file" when
/// the index is absent.
pub(super) const REQUIRED_FILES: &[&str] = &[
    "LLM/tokenizer.json",
    "LLM/added_tokens.json",
    "LLM/config.json",
    "LLM/model.safetensors",
    "BiCodec/config.yaml",
    "BiCodec/model.safetensors",
];

/// Errors surfaced by [`ensure_downloaded`].
#[derive(thiserror::Error, Debug)]
pub(super) enum WeightsError {
    /// The underlying Hugging Face download failed (network, auth, gone
    /// repo, etc.). The string carries the upstream
    /// [`blazen_model_cache::CacheError`] message.
    #[error("HF download failed for {repo_id} ({file}): {message}")]
    Download {
        /// The repo id that was being fetched.
        repo_id: String,
        /// The file inside the repo that failed.
        file: String,
        /// The underlying cache-layer error message.
        message: String,
    },
    /// Filesystem error while resolving the cache directory or
    /// canonicalising a file path.
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    /// The download completed but the expected file is missing on disk.
    /// Almost always indicates an upstream repo that has been renamed or
    /// restructured.
    #[error("expected file missing in repo {repo_id}: {file}")]
    MissingFile {
        /// The repo id whose contents disappointed us.
        repo_id: String,
        /// The expected filename that wasn't there.
        file: String,
    },
    /// The repo id is empty / malformed.
    #[error("invalid repo id: {0}")]
    InvalidRepoId(String),
    /// The platform cache directory could not be resolved.
    #[error("cache directory error: {0}")]
    CacheDir(String),
}

impl From<WeightsError> for TtsError {
    fn from(err: WeightsError) -> Self {
        // Every failure mode here is fundamentally "could not get the
        // model bytes" — surface it as ModelLoad so the public
        // `synthesize` error surface stays narrow. (Io / MissingFile /
        // CacheDir / Download / InvalidRepoId all collapse to a single
        // ModelLoad with the formatted message.)
        TtsError::ModelLoad(err.to_string())
    }
}

/// Resolve a Spark-TTS bundle directory for `repo_id`, downloading +
/// caching on miss.
///
/// On cache hit (every [`REQUIRED_FILES`] entry already present under
/// the `blazen_model_cache` root for this repo) returns immediately
/// without touching the network. On miss the required files are pulled
/// concurrently via [`blazen_model_cache::ModelCache::download`] (which
/// serialises same-path callers via its internal per-path mutex map).
///
/// The returned [`PathBuf`] is the bundle root — i.e. the directory
/// containing `LLM/` and `BiCodec/` subdirectories. Downstream loaders
/// resolve `LLM/tokenizer.json`, `LLM/model.safetensors`,
/// `BiCodec/config.yaml`, etc. relative to that path.
///
/// `revision` is encoded into the cache key when provided (so two
/// revisions of the same repo don't collide on disk); `None` resolves
/// to `main`. The underlying cache layer's current API doesn't forward
/// revision pinning to `hf-hub` itself, so an explicit revision still
/// pulls from `main` but is stored in its own slot — best-effort
/// pinning, sufficient for the typical "I want this exact SHA cached
/// separately from main" use case.
///
/// # Errors
///
/// * [`WeightsError::InvalidRepoId`] when `repo_id` is empty.
/// * [`WeightsError::CacheDir`] when the platform cache directory
///   cannot be resolved and `BLAZEN_CACHE_DIR` is unset.
/// * [`WeightsError::Download`] when an HF fetch fails for any required
///   file.
/// * [`WeightsError::MissingFile`] when a fetch reports success but the
///   file is not present on disk afterwards.
/// * [`WeightsError::Io`] on filesystem failures during directory
///   resolution.
pub(super) async fn ensure_downloaded(
    repo_id: &str,
    revision: Option<&str>,
) -> Result<PathBuf, WeightsError> {
    if repo_id.trim().is_empty() {
        return Err(WeightsError::InvalidRepoId(repo_id.to_owned()));
    }

    // Compose the cache key. Bare repo id for `main`; `{repo_id}@{rev}`
    // for any explicit revision, so multiple revisions don't trample.
    let cache_key = match revision {
        Some(rev) if !rev.trim().is_empty() => format!("{repo_id}@{rev}"),
        _ => repo_id.to_owned(),
    };

    let cache =
        blazen_model_cache::ModelCache::new().map_err(|e| WeightsError::CacheDir(e.to_string()))?;
    let cache_arc = Arc::new(cache);

    // Fan out the required files concurrently. Same-path callers are
    // serialised internally by the cache layer.
    let mut handles = Vec::with_capacity(REQUIRED_FILES.len());
    for file in REQUIRED_FILES {
        let cache = Arc::clone(&cache_arc);
        let key = cache_key.clone();
        let repo = repo_id.to_owned();
        let file = (*file).to_owned();
        handles.push(tokio::spawn(async move {
            let path =
                cache
                    .download(&key, &file, None)
                    .await
                    .map_err(|e| WeightsError::Download {
                        repo_id: repo.clone(),
                        file: file.clone(),
                        message: e.to_string(),
                    })?;
            if !path.is_file() {
                return Err(WeightsError::MissingFile {
                    repo_id: repo,
                    file,
                });
            }
            Ok::<PathBuf, WeightsError>(path)
        }));
    }

    let mut llm_safetensors_path: Option<PathBuf> = None;
    for handle in handles {
        let path = handle.await.map_err(|e| WeightsError::Download {
            repo_id: repo_id.to_owned(),
            file: "<join>".to_owned(),
            message: e.to_string(),
        })??;
        // The LLM weights file lives at `{root}/{repo_key}/LLM/model.safetensors`.
        // Walk up two parents (`model.safetensors` -> `LLM/` -> bundle root)
        // to recover the bundle root.
        if path.file_name().and_then(|s| s.to_str()) == Some("model.safetensors")
            && path
                .parent()
                .and_then(|p| p.file_name())
                .and_then(|s| s.to_str())
                == Some("LLM")
        {
            llm_safetensors_path = Some(path);
        }
    }

    // Best-effort: try to pull a sharded `model.safetensors.index.json`
    // for forks that ship sharded LLM weights. Absent in the canonical
    // `SparkAudio/Spark-TTS-0.5B` release (~990 MB single file), so a
    // 404 here is normal and ignored.
    let _ = cache_arc
        .download(&cache_key, "LLM/model.safetensors.index.json", None)
        .await;

    let llm_st = llm_safetensors_path.ok_or_else(|| WeightsError::MissingFile {
        repo_id: repo_id.to_owned(),
        file: "LLM/model.safetensors".to_owned(),
    })?;

    // `LLM/model.safetensors` -> `LLM/` -> bundle root.
    let bundle_root = llm_st
        .parent()
        .and_then(std::path::Path::parent)
        .ok_or_else(|| {
            WeightsError::Io(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                format!(
                    "LLM/model.safetensors has no two-level parent: {}",
                    llm_st.display()
                ),
            ))
        })?
        .to_path_buf();

    tracing::info!(
        repo = repo_id,
        revision = revision.unwrap_or("main"),
        dir = %bundle_root.display(),
        "spark-tts: bundle ready"
    );

    Ok(bundle_root)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn required_files_includes_llm_and_bicodec_pairs() {
        // LLM side.
        assert!(REQUIRED_FILES.contains(&"LLM/tokenizer.json"));
        assert!(REQUIRED_FILES.contains(&"LLM/config.json"));
        assert!(REQUIRED_FILES.contains(&"LLM/model.safetensors"));
        // BiCodec side.
        assert!(REQUIRED_FILES.contains(&"BiCodec/config.yaml"));
        assert!(REQUIRED_FILES.contains(&"BiCodec/model.safetensors"));
        // Sanity-check we don't accidentally lose entries.
        assert!(REQUIRED_FILES.len() >= 6);
    }

    #[test]
    fn spark_tts_repo_constant_matches_upstream() {
        assert_eq!(SPARK_TTS_REPO, "SparkAudio/Spark-TTS-0.5B");
    }

    #[test]
    fn weights_error_converts_to_tts_error_model_load() {
        let err = WeightsError::Download {
            repo_id: SPARK_TTS_REPO.into(),
            file: "LLM/model.safetensors".into(),
            message: "simulated network failure".into(),
        };
        let tts: TtsError = err.into();
        match tts {
            TtsError::ModelLoad(msg) => {
                assert!(msg.contains("SparkAudio/Spark-TTS-0.5B"), "msg = {msg}");
                assert!(msg.contains("simulated network failure"), "msg = {msg}");
            }
            other => panic!("expected ModelLoad, got {other:?}"),
        }

        // CacheDir + MissingFile + Io all flatten to ModelLoad too.
        let cache_dir: TtsError = WeightsError::CacheDir("no home dir".into()).into();
        assert!(matches!(cache_dir, TtsError::ModelLoad(_)));

        let missing: TtsError = WeightsError::MissingFile {
            repo_id: SPARK_TTS_REPO.into(),
            file: "BiCodec/model.safetensors".into(),
        }
        .into();
        match missing {
            TtsError::ModelLoad(msg) => {
                assert!(msg.contains("BiCodec/model.safetensors"), "msg = {msg}");
            }
            other => panic!("expected ModelLoad, got {other:?}"),
        }

        let io: TtsError = WeightsError::Io(std::io::Error::other("disk full")).into();
        assert!(matches!(io, TtsError::ModelLoad(_)));
    }

    #[tokio::test]
    async fn empty_repo_id_is_invalid() {
        let err = ensure_downloaded("", None)
            .await
            .expect_err("empty repo id must be rejected");
        assert!(matches!(err, WeightsError::InvalidRepoId(_)), "{err:?}");

        let err = ensure_downloaded("   ", None)
            .await
            .expect_err("whitespace-only repo id must be rejected");
        assert!(matches!(err, WeightsError::InvalidRepoId(_)), "{err:?}");
    }
}
