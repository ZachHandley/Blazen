//! Configuration options for the fastembed local embedding backend.

use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// Options for constructing a [`FastEmbedModel`](crate::FastEmbedModel).
///
/// All fields are optional; defaults produce a working model using
/// `BAAI/bge-small-en-v1.5` on CPU with fastembed's built-in cache.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct FastEmbedOptions {
    /// Fastembed model variant name (e.g. `"BGESmallENV15"`).
    ///
    /// Parsed via `fastembed::EmbeddingModel::from_str`. When `None`, defaults
    /// to `fastembed::EmbeddingModel::BGESmallENV15`.
    pub model_name: Option<String>,
    /// Model cache directory. When `None`, fastembed uses its built-in cache
    /// (controlled by `FASTEMBED_CACHE_DIR` / `HF_HOME` env vars).
    pub cache_dir: Option<PathBuf>,
    /// Maximum batch size for embedding. When `None`, fastembed uses its
    /// default (256).
    pub max_batch_size: Option<usize>,
    /// Whether to display download progress when fetching models from
    /// `HuggingFace`. Defaults to `true`.
    pub show_download_progress: Option<bool>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_options_fields_are_none() {
        let opts = FastEmbedOptions::default();
        assert!(opts.model_name.is_none());
        assert!(opts.cache_dir.is_none());
        assert!(opts.max_batch_size.is_none());
        assert!(opts.show_download_progress.is_none());
    }
}
