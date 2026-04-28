//! Options for [`TractEmbedModel`](crate::provider::TractEmbedModel).
//!
//! Mirrors `FastEmbedOptions` so users can swap between the fastembed and tract
//! backends without changing their configuration surface. Model selection is
//! done via a case-insensitive string that matches the variants of fastembed's
//! [`EmbeddingModel`] enum (e.g. `"BGESmallENV15"`).

use serde::{Deserialize, Serialize};

/// Construction options for a [`TractEmbedModel`](crate::provider::TractEmbedModel).
///
/// This mirrors the shape of `FastEmbedOptions` on purpose: the tract backend
/// is a drop-in replacement that loads the same fastembed model catalog via
/// `tract_onnx` instead of `onnxruntime`, so callers can flip backends without
/// touching their model configuration.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TractOptions {
    /// Model name matching a variant of fastembed's `EmbeddingModel` enum
    /// (debug spelling, e.g. `"BGESmallENV15"`). Case-insensitive.
    ///
    /// When `None`, resolves to the fastembed default (`BGESmallENV15`).
    pub model_name: Option<String>,

    /// Override for the model cache directory. If `None`, uses the default
    /// from `blazen_model_cache`.
    pub cache_dir: Option<std::path::PathBuf>,

    /// Optional maximum batch size override for inference. Tokens are still
    /// padded per-batch; this only caps how many inputs we feed into one
    /// `tract_onnx` forward pass.
    pub max_batch_size: Option<usize>,

    /// Whether to print a progress bar while downloading model files from the
    /// Hugging Face hub. Defaults to `false` when unset.
    pub show_download_progress: Option<bool>,
}

/// Pooling strategy applied to the final hidden-state tensor to collapse the
/// token dimension into a single sentence embedding.
///
/// Mirrors fastembed's internal `Pooling` enum so the resulting vectors are
/// byte-identical (up to floating-point noise) between the two backends.
#[derive(Debug, Clone, Copy)]
pub(crate) enum Pooling {
    /// Token-wise mean, weighted by the attention mask (ignore pad tokens).
    Mean,
    /// Take the first token's hidden state (the `[CLS]` token for BERT-family
    /// models).
    Cls,
}

/// Static description of one entry in the fastembed model registry.
///
/// All fields are `&'static` because the registry is a compile-time constant
/// — there is no scenario in which we need to mutate these values at runtime,
/// and keeping them as static slices avoids any per-lookup allocation.
pub(crate) struct ModelInfo {
    /// Registry key. Matches the debug spelling of fastembed's
    /// `EmbeddingModel` variant (e.g. `"BGESmallENV15"`). This is what users
    /// put in `TractOptions::model_name`.
    pub name: &'static str,

    /// Hugging Face repo id (e.g. `"Xenova/bge-small-en-v1.5"`).
    pub model_code: &'static str,

    /// Path to the primary ONNX file within the HF repo
    /// (e.g. `"onnx/model.onnx"`). Only used by the native provider; on
    /// wasm32 the caller passes a fully-resolved URL to `create()`, so this
    /// field is unused on that target.
    #[cfg_attr(target_arch = "wasm32", allow(dead_code))]
    pub model_file: &'static str,

    /// Sibling files that must be downloaded alongside `model_file`. Typically
    /// `onnx/model.onnx_data` for models too large to fit in a single
    /// protobuf, or sparse tensor value files. Only used by the native
    /// provider; see [`Self::model_file`] for why this is dead on wasm32.
    #[cfg_attr(target_arch = "wasm32", allow(dead_code))]
    pub additional_files: &'static [&'static str],

    /// Output embedding dimensionality.
    pub dim: usize,

    /// Pooling strategy applied to collapse token hidden states into a single
    /// sentence vector.
    pub pooling: Pooling,
}

/// Full fastembed text-embedding registry, copied verbatim (in field order and
/// content) from `fastembed-5.13.1/src/models/text_embedding.rs`. Keep this in
/// sync when bumping the fastembed dependency.
///
/// Ordering here follows the order of the `EmbeddingModel` enum rather than
/// the `init_models_map` vector (which has `ParaphraseMLMiniLML12V2Q` before
/// `ParaphraseMLMiniLML12V2`); the enum order is what users see in the
/// fastembed docs.
pub(crate) const MODELS: &[ModelInfo] = &[
    ModelInfo {
        name: "AllMiniLML6V2",
        model_code: "Qdrant/all-MiniLM-L6-v2-onnx",
        model_file: "model.onnx",
        additional_files: &[],
        dim: 384,
        pooling: Pooling::Mean,
    },
    ModelInfo {
        name: "AllMiniLML6V2Q",
        model_code: "Xenova/all-MiniLM-L6-v2",
        model_file: "onnx/model_quantized.onnx",
        additional_files: &[],
        dim: 384,
        pooling: Pooling::Mean,
    },
    ModelInfo {
        name: "AllMiniLML12V2",
        model_code: "Xenova/all-MiniLM-L12-v2",
        model_file: "onnx/model.onnx",
        additional_files: &[],
        dim: 384,
        pooling: Pooling::Mean,
    },
    ModelInfo {
        name: "AllMiniLML12V2Q",
        model_code: "Xenova/all-MiniLM-L12-v2",
        model_file: "onnx/model_quantized.onnx",
        additional_files: &[],
        dim: 384,
        pooling: Pooling::Mean,
    },
    ModelInfo {
        name: "AllMpnetBaseV2",
        model_code: "Xenova/all-mpnet-base-v2",
        model_file: "onnx/model.onnx",
        additional_files: &[],
        dim: 768,
        pooling: Pooling::Mean,
    },
    ModelInfo {
        name: "BGEBaseENV15",
        model_code: "Xenova/bge-base-en-v1.5",
        model_file: "onnx/model.onnx",
        additional_files: &[],
        dim: 768,
        pooling: Pooling::Cls,
    },
    ModelInfo {
        name: "BGEBaseENV15Q",
        model_code: "Qdrant/bge-base-en-v1.5-onnx-Q",
        model_file: "model_optimized.onnx",
        additional_files: &[],
        dim: 768,
        pooling: Pooling::Cls,
    },
    ModelInfo {
        name: "BGELargeENV15",
        model_code: "Xenova/bge-large-en-v1.5",
        model_file: "onnx/model.onnx",
        additional_files: &[],
        dim: 1024,
        pooling: Pooling::Cls,
    },
    ModelInfo {
        name: "BGELargeENV15Q",
        model_code: "Qdrant/bge-large-en-v1.5-onnx-Q",
        model_file: "model_optimized.onnx",
        additional_files: &[],
        dim: 1024,
        pooling: Pooling::Cls,
    },
    ModelInfo {
        name: "BGESmallENV15",
        model_code: "Xenova/bge-small-en-v1.5",
        model_file: "onnx/model.onnx",
        additional_files: &[],
        dim: 384,
        pooling: Pooling::Cls,
    },
    ModelInfo {
        name: "BGESmallENV15Q",
        model_code: "Qdrant/bge-small-en-v1.5-onnx-Q",
        model_file: "model_optimized.onnx",
        additional_files: &[],
        dim: 384,
        pooling: Pooling::Cls,
    },
    ModelInfo {
        name: "NomicEmbedTextV1",
        model_code: "nomic-ai/nomic-embed-text-v1",
        model_file: "onnx/model.onnx",
        additional_files: &[],
        dim: 768,
        pooling: Pooling::Mean,
    },
    ModelInfo {
        name: "NomicEmbedTextV15",
        model_code: "nomic-ai/nomic-embed-text-v1.5",
        model_file: "onnx/model.onnx",
        additional_files: &[],
        dim: 768,
        pooling: Pooling::Mean,
    },
    ModelInfo {
        name: "NomicEmbedTextV15Q",
        model_code: "nomic-ai/nomic-embed-text-v1.5",
        model_file: "onnx/model_quantized.onnx",
        additional_files: &[],
        dim: 768,
        pooling: Pooling::Mean,
    },
    ModelInfo {
        name: "ParaphraseMLMiniLML12V2",
        model_code: "Xenova/paraphrase-multilingual-MiniLM-L12-v2",
        model_file: "onnx/model.onnx",
        additional_files: &[],
        dim: 384,
        pooling: Pooling::Mean,
    },
    ModelInfo {
        name: "ParaphraseMLMiniLML12V2Q",
        model_code: "Qdrant/paraphrase-multilingual-MiniLM-L12-v2-onnx-Q",
        model_file: "model_optimized.onnx",
        additional_files: &[],
        dim: 384,
        pooling: Pooling::Mean,
    },
    ModelInfo {
        name: "ParaphraseMLMpnetBaseV2",
        model_code: "Xenova/paraphrase-multilingual-mpnet-base-v2",
        model_file: "onnx/model.onnx",
        additional_files: &[],
        dim: 768,
        pooling: Pooling::Mean,
    },
    ModelInfo {
        name: "BGESmallZHV15",
        model_code: "Xenova/bge-small-zh-v1.5",
        model_file: "onnx/model.onnx",
        additional_files: &[],
        dim: 512,
        pooling: Pooling::Cls,
    },
    ModelInfo {
        name: "BGELargeZHV15",
        model_code: "Xenova/bge-large-zh-v1.5",
        model_file: "onnx/model.onnx",
        additional_files: &[],
        dim: 1024,
        pooling: Pooling::Cls,
    },
    ModelInfo {
        name: "BGEM3",
        model_code: "BAAI/bge-m3",
        model_file: "onnx/model.onnx",
        additional_files: &["onnx/model.onnx_data", "onnx/Constant_7_attr__value"],
        dim: 1024,
        pooling: Pooling::Cls,
    },
    ModelInfo {
        name: "ModernBertEmbedLarge",
        model_code: "lightonai/modernbert-embed-large",
        model_file: "onnx/model.onnx",
        additional_files: &[],
        dim: 1024,
        pooling: Pooling::Mean,
    },
    ModelInfo {
        name: "MultilingualE5Small",
        model_code: "intfloat/multilingual-e5-small",
        model_file: "onnx/model.onnx",
        additional_files: &[],
        dim: 384,
        pooling: Pooling::Mean,
    },
    ModelInfo {
        name: "MultilingualE5Base",
        model_code: "intfloat/multilingual-e5-base",
        model_file: "onnx/model.onnx",
        additional_files: &[],
        dim: 768,
        pooling: Pooling::Mean,
    },
    ModelInfo {
        name: "MultilingualE5Large",
        model_code: "Qdrant/multilingual-e5-large-onnx",
        model_file: "model.onnx",
        additional_files: &["model.onnx_data"],
        dim: 1024,
        pooling: Pooling::Mean,
    },
    ModelInfo {
        name: "MxbaiEmbedLargeV1",
        model_code: "mixedbread-ai/mxbai-embed-large-v1",
        model_file: "onnx/model.onnx",
        additional_files: &[],
        dim: 1024,
        pooling: Pooling::Cls,
    },
    ModelInfo {
        name: "MxbaiEmbedLargeV1Q",
        model_code: "mixedbread-ai/mxbai-embed-large-v1",
        model_file: "onnx/model_quantized.onnx",
        additional_files: &[],
        dim: 1024,
        pooling: Pooling::Cls,
    },
    ModelInfo {
        name: "GTEBaseENV15",
        model_code: "Alibaba-NLP/gte-base-en-v1.5",
        model_file: "onnx/model.onnx",
        additional_files: &[],
        dim: 768,
        pooling: Pooling::Cls,
    },
    ModelInfo {
        name: "GTEBaseENV15Q",
        model_code: "Alibaba-NLP/gte-base-en-v1.5",
        model_file: "onnx/model_quantized.onnx",
        additional_files: &[],
        dim: 768,
        pooling: Pooling::Cls,
    },
    ModelInfo {
        name: "GTELargeENV15",
        model_code: "Alibaba-NLP/gte-large-en-v1.5",
        model_file: "onnx/model.onnx",
        additional_files: &[],
        dim: 1024,
        pooling: Pooling::Cls,
    },
    ModelInfo {
        name: "GTELargeENV15Q",
        model_code: "Alibaba-NLP/gte-large-en-v1.5",
        model_file: "onnx/model_quantized.onnx",
        additional_files: &[],
        dim: 1024,
        pooling: Pooling::Cls,
    },
    ModelInfo {
        name: "ClipVitB32",
        model_code: "Qdrant/clip-ViT-B-32-text",
        model_file: "model.onnx",
        additional_files: &[],
        dim: 512,
        pooling: Pooling::Mean,
    },
    ModelInfo {
        name: "JinaEmbeddingsV2BaseCode",
        model_code: "jinaai/jina-embeddings-v2-base-code",
        model_file: "onnx/model.onnx",
        additional_files: &[],
        dim: 768,
        pooling: Pooling::Mean,
    },
    ModelInfo {
        name: "JinaEmbeddingsV2BaseEN",
        model_code: "jinaai/jina-embeddings-v2-base-en",
        model_file: "model.onnx",
        additional_files: &[],
        dim: 768,
        pooling: Pooling::Mean,
    },
    ModelInfo {
        name: "EmbeddingGemma300M",
        model_code: "onnx-community/embeddinggemma-300m-ONNX",
        model_file: "onnx/model.onnx",
        additional_files: &["onnx/model.onnx_data"],
        dim: 768,
        pooling: Pooling::Mean,
    },
    ModelInfo {
        name: "SnowflakeArcticEmbedXS",
        model_code: "snowflake/snowflake-arctic-embed-xs",
        model_file: "onnx/model.onnx",
        additional_files: &[],
        dim: 384,
        pooling: Pooling::Cls,
    },
    ModelInfo {
        name: "SnowflakeArcticEmbedXSQ",
        model_code: "snowflake/snowflake-arctic-embed-xs",
        model_file: "onnx/model_quantized.onnx",
        additional_files: &[],
        dim: 384,
        pooling: Pooling::Cls,
    },
    ModelInfo {
        name: "SnowflakeArcticEmbedS",
        model_code: "snowflake/snowflake-arctic-embed-s",
        model_file: "onnx/model.onnx",
        additional_files: &[],
        dim: 384,
        pooling: Pooling::Cls,
    },
    ModelInfo {
        name: "SnowflakeArcticEmbedSQ",
        model_code: "snowflake/snowflake-arctic-embed-s",
        model_file: "onnx/model_quantized.onnx",
        additional_files: &[],
        dim: 384,
        pooling: Pooling::Cls,
    },
    ModelInfo {
        name: "SnowflakeArcticEmbedM",
        model_code: "Snowflake/snowflake-arctic-embed-m",
        model_file: "onnx/model.onnx",
        additional_files: &[],
        dim: 768,
        pooling: Pooling::Cls,
    },
    ModelInfo {
        name: "SnowflakeArcticEmbedMQ",
        model_code: "Snowflake/snowflake-arctic-embed-m",
        model_file: "onnx/model_quantized.onnx",
        additional_files: &[],
        dim: 768,
        pooling: Pooling::Cls,
    },
    ModelInfo {
        name: "SnowflakeArcticEmbedMLong",
        model_code: "snowflake/snowflake-arctic-embed-m-long",
        model_file: "onnx/model.onnx",
        additional_files: &[],
        dim: 768,
        pooling: Pooling::Cls,
    },
    ModelInfo {
        name: "SnowflakeArcticEmbedMLongQ",
        model_code: "snowflake/snowflake-arctic-embed-m-long",
        model_file: "onnx/model_quantized.onnx",
        additional_files: &[],
        dim: 768,
        pooling: Pooling::Cls,
    },
    ModelInfo {
        name: "SnowflakeArcticEmbedL",
        model_code: "snowflake/snowflake-arctic-embed-l",
        model_file: "onnx/model.onnx",
        additional_files: &[],
        dim: 1024,
        pooling: Pooling::Cls,
    },
    ModelInfo {
        name: "SnowflakeArcticEmbedLQ",
        model_code: "snowflake/snowflake-arctic-embed-l",
        model_file: "onnx/model_quantized.onnx",
        additional_files: &[],
        dim: 1024,
        pooling: Pooling::Cls,
    },
];

/// Default model name used when [`TractOptions::model_name`] is `None`.
///
/// Matches fastembed's `EmbeddingModel::default()`, which picks
/// `BGESmallENV15` (the fast, small, default English model).
pub(crate) const DEFAULT_MODEL_NAME: &str = "BGESmallENV15";

/// Look up a model entry by name, case-insensitively.
///
/// When `name` is `None`, resolves to [`DEFAULT_MODEL_NAME`]
/// (`BGESmallENV15`). Returns `None` when the provided name does not match any
/// registry entry.
pub(crate) fn lookup(name: Option<&str>) -> Option<&'static ModelInfo> {
    let needle = name.unwrap_or(DEFAULT_MODEL_NAME);
    MODELS.iter().find(|m| m.name.eq_ignore_ascii_case(needle))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_resolves_to_bge_small_en_v15() {
        let info = lookup(None).expect("default must resolve");
        assert_eq!(info.name, "BGESmallENV15");
        assert_eq!(info.dim, 384);
    }

    #[test]
    fn lookup_is_case_insensitive() {
        assert!(lookup(Some("bgesmallenv15")).is_some());
        assert!(lookup(Some("BGESMALLENV15")).is_some());
        assert!(lookup(Some("BgEsMaLlEnV15")).is_some());
    }

    #[test]
    fn unknown_model_returns_none() {
        assert!(lookup(Some("DoesNotExist")).is_none());
    }

    #[test]
    fn registry_has_unique_names() {
        let mut names: Vec<&str> = MODELS.iter().map(|m| m.name).collect();
        names.sort_unstable();
        let mut deduped = names.clone();
        deduped.dedup();
        assert_eq!(names, deduped, "duplicate model name in MODELS registry");
    }

    #[test]
    fn registry_has_expected_size() {
        // 44 entries lifted from fastembed 5.13.1's text_embedding registry.
        assert_eq!(MODELS.len(), 44);
    }

    #[test]
    fn bgem3_has_additional_files() {
        let info = lookup(Some("BGEM3")).unwrap();
        assert_eq!(
            info.additional_files,
            &["onnx/model.onnx_data", "onnx/Constant_7_attr__value"]
        );
    }
}
