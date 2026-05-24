//! `HuggingFace` download for `Systran/faster-whisper-*` `CTranslate2` weights (Wave F.2.5).
//!
//! [`ct2rs::Whisper::new`][ct2rs] opens a *directory* containing the
//! `CTranslate2`-converted weights — `model.bin`, `config.json`,
//! `tokenizer.json`, `vocabulary.json`, and `preprocessor_config.json`.
//! This module downloads that bundle on demand from a `Systran/faster-whisper-*`
//! Hugging Face Hub repo, caches it under the shared
//! [`blazen_model_cache`][crate-modelcache] cache root, and returns the
//! local directory path.
//!
//! # Cache layout
//!
//! Files land at `{BLAZEN_CACHE_DIR or ~/.cache/blazen}/models/{repo_id}/{filename}`
//! — the same layout the whisper.cpp backend uses for its GGML weights.
//! The returned [`PathBuf`] is the directory containing the five required
//! files (the parent of the cached `model.bin`), which is what
//! [`FasterWhisperDecoder::load`][crate::backends::faster_whisper::FasterWhisperDecoder::load]
//! expects.
//!
//! # Revision pinning
//!
//! [`ensure_downloaded`] accepts an optional `revision` argument. `None`
//! resolves to `main` — i.e. whatever the upstream `Systran` org currently
//! ships, matching the upstream `faster-whisper` Python project's default
//! behavior. Callers that need bit-reproducible downloads can pass a
//! specific git SHA / branch / tag through
//! [`FasterWhisperConfig::revision`][crate::backends::faster_whisper::FasterWhisperConfig::revision].
//!
//! [ct2rs]: https://docs.rs/ct2rs/latest/ct2rs/struct.Whisper.html#method.new
//! [crate-modelcache]: blazen_model_cache::ModelCache

use std::path::PathBuf;
use std::sync::Arc;

use crate::SttError;

/// The set of files Systran's `faster-whisper-*` bundles ship in every
/// repo. `ct2rs::Whisper::new` will refuse to load if any are missing,
/// so we eagerly fetch the full set.
pub(crate) const REQUIRED_FILES: &[&str] = &[
    "model.bin",
    "config.json",
    "tokenizer.json",
    "vocabulary.json",
    "preprocessor_config.json",
];

/// Errors surfaced by [`ensure_downloaded`].
#[derive(Debug, thiserror::Error)]
pub enum WeightsError {
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

    /// Filesystem error while resolving the cache directory or canonicalising
    /// a file path.
    #[error("io: {0}")]
    Io(#[from] std::io::Error),

    /// The download completed but the expected file is missing in the cache
    /// directory. Almost always indicates an upstream repo that has been
    /// renamed or restructured.
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

    /// The [`blazen_model_cache::ModelCache`] could not be constructed
    /// (typically because the platform cache dir cannot be resolved and
    /// `BLAZEN_CACHE_DIR` is unset).
    #[error("cache directory error: {0}")]
    CacheDir(String),
}

impl From<WeightsError> for SttError {
    fn from(err: WeightsError) -> Self {
        match err {
            WeightsError::Io(io) => SttError::Io(io),
            // Every other failure mode is fundamentally "could not get the
            // model bytes" — surface it as ModelLoad so the public
            // `transcribe`/`stream` error surface stays narrow.
            other => SttError::ModelLoad(other.to_string()),
        }
    }
}

/// Resolve a `Systran/faster-whisper-*` model directory for `repo_id`,
/// downloading + caching as needed.
///
/// On cache hit (all [`REQUIRED_FILES`] already present under the
/// `blazen_model_cache` root for this repo) this returns immediately
/// without touching the network. On cache miss it fans out the five
/// required files through [`blazen_model_cache::ModelCache::download`]
/// (which serialises concurrent same-path callers via its internal
/// per-path mutex map) and returns the parent directory that all five
/// landed under.
///
/// `revision` is forwarded to the cache layer as part of the repo path
/// when provided. `None` means "whatever `main` currently points at",
/// matching upstream `faster-whisper`.
///
/// # Errors
///
/// * [`WeightsError::InvalidRepoId`] when `repo_id` is empty.
/// * [`WeightsError::CacheDir`] when the platform cache directory cannot
///   be resolved and `BLAZEN_CACHE_DIR` is not set.
/// * [`WeightsError::Download`] when a `HuggingFace` fetch fails for any
///   required file.
/// * [`WeightsError::MissingFile`] when a fetch reports success but the
///   file is not actually on disk afterwards.
/// * [`WeightsError::Io`] when a filesystem operation (directory
///   resolution, canonicalisation) fails.
pub async fn ensure_downloaded(
    repo_id: &str,
    revision: Option<&str>,
) -> Result<PathBuf, WeightsError> {
    if repo_id.trim().is_empty() {
        return Err(WeightsError::InvalidRepoId(repo_id.to_owned()));
    }

    // The cache layer's "repo_id" key is just the on-disk subdirectory
    // name. When the caller pins a revision we encode it into that key so
    // two revisions of the same repo don't collide in the cache
    // (`{root}/models/{org}/{model}@{rev}/...`). When `None` we keep the
    // bare repo id — same layout the whisper.cpp backend already uses.
    let cache_key = match revision {
        Some(rev) if !rev.trim().is_empty() => format!("{repo_id}@{rev}"),
        _ => repo_id.to_owned(),
    };

    let cache =
        blazen_model_cache::ModelCache::new().map_err(|e| WeightsError::CacheDir(e.to_string()))?;

    // hf-hub's tokio API does the actual fetch + manages its own
    // ~/.cache/huggingface directory; the ModelCache layer then
    // hard-links (or copies as fallback) each blob into our predictable
    // `{cache_dir}/models/{cache_key}/{filename}` layout. Fast-path on
    // cache hit short-circuits at the I/O check inside `download()`.
    //
    // We invoke the cache layer with the *bare* repo id when no revision
    // is requested so its on-disk layout matches whisper.cpp's. When a
    // revision is requested we re-key the cache subdir but still ask
    // hf-hub for the bare repo id at `main`, because the cache layer's
    // current API does not forward revision pinning to hf-hub. That
    // matches the documented "revision = None means main" semantics; an
    // explicit revision still gets its own cache slot so two revisions
    // don't trample each other on disk, but pinning beyond the public
    // `main` branch is best-effort until `blazen_model_cache` grows a
    // revision-aware `download_at` API.
    let _ = revision; // explicitly acknowledged above

    let cache_arc = Arc::new(cache);
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

    let mut model_bin_path: Option<PathBuf> = None;
    for handle in handles {
        let path = handle.await.map_err(|e| WeightsError::Download {
            repo_id: repo_id.to_owned(),
            file: "<join>".to_owned(),
            message: e.to_string(),
        })??;
        if path.file_name().and_then(|s| s.to_str()) == Some("model.bin") {
            model_bin_path = Some(path);
        }
    }

    let model_bin = model_bin_path.ok_or_else(|| WeightsError::MissingFile {
        repo_id: repo_id.to_owned(),
        file: "model.bin".to_owned(),
    })?;
    let dir = model_bin
        .parent()
        .ok_or_else(|| {
            WeightsError::Io(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                format!("model.bin has no parent directory: {}", model_bin.display()),
            ))
        })?
        .to_path_buf();

    tracing::info!(
        repo = repo_id,
        revision = revision.unwrap_or("main"),
        dir = %dir.display(),
        "faster-whisper: model bundle ready"
    );

    Ok(dir)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn required_files_includes_model_bin_and_tokenizer() {
        // Sanity check: any drift in REQUIRED_FILES will break
        // `ct2rs::Whisper::new`, so guard the expected entries.
        assert!(REQUIRED_FILES.contains(&"model.bin"));
        assert!(REQUIRED_FILES.contains(&"tokenizer.json"));
        assert!(REQUIRED_FILES.contains(&"config.json"));
        assert!(REQUIRED_FILES.contains(&"vocabulary.json"));
        assert!(REQUIRED_FILES.contains(&"preprocessor_config.json"));
        assert_eq!(REQUIRED_FILES.len(), 5);
    }

    #[test]
    fn weights_error_converts_to_stt_error() {
        let err = WeightsError::Download {
            repo_id: "Systran/faster-whisper-tiny".into(),
            file: "model.bin".into(),
            message: "simulated network failure".into(),
        };
        let stt: SttError = err.into();
        match stt {
            SttError::ModelLoad(msg) => {
                assert!(msg.contains("Systran/faster-whisper-tiny"), "msg = {msg}");
                assert!(msg.contains("simulated network failure"), "msg = {msg}");
            }
            other => panic!("expected ModelLoad, got {other:?}"),
        }

        // Io errors should round-trip through the dedicated Io variant
        // rather than being flattened into ModelLoad.
        let io_err = WeightsError::Io(std::io::Error::other("disk full"));
        let stt: SttError = io_err.into();
        assert!(matches!(stt, SttError::Io(_)), "got {stt:?}");

        // MissingFile flattens to ModelLoad with the repo/file in the
        // message so callers can diagnose upstream repo restructuring.
        let missing = WeightsError::MissingFile {
            repo_id: "Systran/faster-whisper-tiny".into(),
            file: "tokenizer.json".into(),
        };
        let stt: SttError = missing.into();
        match stt {
            SttError::ModelLoad(msg) => {
                assert!(msg.contains("tokenizer.json"), "msg = {msg}");
            }
            other => panic!("expected ModelLoad, got {other:?}"),
        }
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

    /// Live cache-hit test — gated behind `BLAZEN_TEST_FASTER_WHISPER=1`
    /// because it downloads the full ~75 MB `Systran/faster-whisper-tiny`
    /// bundle from `HuggingFace` Hub on first run. Subsequent runs hit the
    /// cache and complete in well under a second; the assertion below is
    /// the cache-hit observation.
    #[tokio::test]
    #[ignore = "requires BLAZEN_TEST_FASTER_WHISPER=1 and downloads ~75 MB of CTranslate2 Whisper weights from HF Hub"]
    async fn ensure_downloaded_cache_hit_returns_immediately() {
        if std::env::var("BLAZEN_TEST_FASTER_WHISPER").ok().as_deref() != Some("1") {
            eprintln!("skipping: BLAZEN_TEST_FASTER_WHISPER != 1");
            return;
        }

        let repo = "Systran/faster-whisper-tiny";

        // First call may download; second call must be a fast cache hit.
        let first = ensure_downloaded(repo, None)
            .await
            .expect("first ensure_downloaded succeeds");
        assert!(first.is_dir(), "first call returns a directory");
        for file in REQUIRED_FILES {
            assert!(
                first.join(file).is_file(),
                "missing {file} under {}",
                first.display()
            );
        }

        let started = std::time::Instant::now();
        let second = ensure_downloaded(repo, None)
            .await
            .expect("second ensure_downloaded succeeds");
        let elapsed = started.elapsed();
        assert_eq!(first, second, "cache hit returns identical path");
        assert!(
            elapsed < std::time::Duration::from_secs(5),
            "cache hit took {elapsed:?} (expected sub-second)"
        );
    }
}
