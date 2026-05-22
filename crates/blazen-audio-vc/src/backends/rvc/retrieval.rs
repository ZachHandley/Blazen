//! kNN feature retrieval index for the RVC (Retrieval-based Voice
//! Conversion) backend — the "R" in RVC.
//!
//! At training time the speaker's content features (extracted by the
//! `HuBERT`/`ContentVec` encoder — 768-dim for RVC v2, 256-dim for RVC v1)
//! are indexed in an HNSW graph. At inference time, each query frame
//! retrieves its top-`k` nearest neighbors from the index (cosine
//! distance) and the speaker timbre is biased toward the index by blending
//! the query with the mean of those neighbors:
//!
//! ```text
//! output_frame = (1 - blend) * query_frame + blend * mean(top_k)
//! ```
//!
//! `blend` is in `[0, 1]` (typical RVC default 0.5–0.75) and `top_k` is
//! conventionally 8.
//!
//! ## Implementation notes
//!
//! - Backed by [`instant_distance`] (pure-Rust HNSW, MIT) — no native deps.
//! - Features are L2-normalized at build time so cosine distance reduces
//!   to `1 - dot(a, b)`, which is a cheap fused operation. The
//!   un-normalized originals are cached alongside the index as the
//!   `value`s of an [`instant_distance::HnswMap`] so blending preserves
//!   the true feature magnitude.
//! - Serialization is bincode; the HNSW graph round-trips via serde.

use std::fs::File;
use std::io::{BufReader, BufWriter};
use std::path::Path;

use candle_core::Tensor;
use instant_distance::{Builder, HnswMap, Point, Search};
use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Error type for the kNN retrieval index.
#[derive(Debug, Error)]
pub enum RetrievalError {
    /// Index construction failed (bad input shape, dtype, etc.).
    #[error("retrieval index build failed: {0}")]
    Build(String),

    /// I/O error during [`FeatureIndex::load`] or [`FeatureIndex::save`].
    #[error(transparent)]
    Io(#[from] std::io::Error),

    /// bincode (de)serialization error.
    #[error("retrieval index (de)serialization failed: {0}")]
    Serde(String),

    /// Underlying candle error.
    #[error(transparent)]
    Candle(#[from] candle_core::Error),

    /// The index has no entries — cannot build or query.
    #[error("retrieval index is empty")]
    EmptyIndex,
}

/// A single L2-normalized content feature stored as an HNSW point.
#[derive(Clone, Debug, Serialize, Deserialize)]
struct ContentFeature(Vec<f32>);

impl Point for ContentFeature {
    /// Cosine distance: since both vectors are L2-normalized at build
    /// time, this is `1 - dot(a, b)`.
    fn distance(&self, other: &Self) -> f32 {
        debug_assert_eq!(
            self.0.len(),
            other.0.len(),
            "ContentFeature dim mismatch in distance(): {} vs {}",
            self.0.len(),
            other.0.len(),
        );
        let dot: f32 = self.0.iter().zip(other.0.iter()).map(|(a, b)| a * b).sum();
        // Clamp into [0, 2] — small fp drift can push (1 - dot) very
        // slightly negative for near-identical vectors, which would
        // misorder the HNSW heap.
        (1.0 - dot).max(0.0)
    }
}

/// HNSW kNN index over RVC content features.
///
/// Construct with [`FeatureIndex::build`], query with
/// [`FeatureIndex::retrieve`], and persist with [`FeatureIndex::save`] /
/// [`FeatureIndex::load`].
#[derive(Serialize, Deserialize)]
pub struct FeatureIndex {
    /// HNSW graph keyed by L2-normalized features; the `value` for each
    /// point is the original (un-normalized) feature, used for blending.
    index: HnswMap<ContentFeature, Vec<f32>>,
    /// Hidden dimensionality of stored features (256 for RVC v1, 768 for
    /// RVC v2). Used to validate query shape.
    hidden_dim: usize,
}

impl FeatureIndex {
    /// Build an index from a `(n_frames, hidden_dim)` tensor of
    /// training-time content features.
    ///
    /// # Errors
    ///
    /// Returns [`RetrievalError::EmptyIndex`] when `n_frames == 0`,
    /// [`RetrievalError::Build`] on shape/dtype errors, or
    /// [`RetrievalError::Candle`] on tensor-layout failures.
    pub fn build(features: &Tensor) -> Result<Self, RetrievalError> {
        let dims = features.dims();
        if dims.len() != 2 {
            return Err(RetrievalError::Build(format!(
                "expected (n_frames, hidden_dim) 2-D tensor, got rank {} shape {:?}",
                dims.len(),
                dims,
            )));
        }
        let (n_frames, hidden_dim) = (dims[0], dims[1]);
        if n_frames == 0 {
            return Err(RetrievalError::EmptyIndex);
        }
        if hidden_dim == 0 {
            return Err(RetrievalError::Build("hidden_dim must be > 0".to_string()));
        }

        let cpu_features = features
            .to_device(&candle_core::Device::Cpu)?
            .to_dtype(candle_core::DType::F32)?;
        let rows: Vec<Vec<f32>> = cpu_features.to_vec2::<f32>()?;

        let mut points = Vec::with_capacity(n_frames);
        let mut values = Vec::with_capacity(n_frames);
        for row in rows {
            if row.len() != hidden_dim {
                return Err(RetrievalError::Build(format!(
                    "feature row length {} != hidden_dim {}",
                    row.len(),
                    hidden_dim,
                )));
            }
            let normalized = l2_normalize(&row);
            points.push(ContentFeature(normalized));
            values.push(row);
        }

        let index = Builder::default().build(points, values);
        Ok(Self { index, hidden_dim })
    }

    /// Load a previously-saved index from `path`. The file must have been
    /// produced by [`FeatureIndex::save`].
    ///
    /// # Errors
    ///
    /// Returns [`RetrievalError::Io`] when the file cannot be opened and
    /// [`RetrievalError::Serde`] on a bincode decode failure.
    pub fn load(path: &Path) -> Result<Self, RetrievalError> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        let this: Self =
            bincode::deserialize_from(reader).map_err(|e| RetrievalError::Serde(e.to_string()))?;
        Ok(this)
    }

    /// Persist the index to `path` in bincode format.
    ///
    /// # Errors
    ///
    /// Returns [`RetrievalError::Io`] when the file cannot be created and
    /// [`RetrievalError::Serde`] on a bincode encode failure.
    pub fn save(&self, path: &Path) -> Result<(), RetrievalError> {
        let file = File::create(path)?;
        let writer = BufWriter::new(file);
        bincode::serialize_into(writer, self).map_err(|e| RetrievalError::Serde(e.to_string()))?;
        Ok(())
    }

    /// Hidden dimensionality (256 or 768 for stock RVC checkpoints).
    #[must_use]
    pub fn hidden_dim(&self) -> usize {
        self.hidden_dim
    }

    /// Number of indexed frames.
    #[must_use]
    pub fn len(&self) -> usize {
        self.index.values.len()
    }

    /// `true` when the index contains no frames.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.index.values.is_empty()
    }

    /// For each query frame, find the top-`k` nearest neighbors in the
    /// index (cosine distance) and return a blended feature:
    ///
    /// ```text
    /// output_frame = (1 - blend) * query_frame + blend * mean(top_k)
    /// ```
    ///
    /// `blend` is clamped into `[0, 1]`; `top_k` is clamped into
    /// `[1, len()]`.
    ///
    /// The input is shaped `(1, n_frames, hidden_dim)`; the output has
    /// the same shape and device.
    ///
    /// # Errors
    ///
    /// Returns [`RetrievalError::EmptyIndex`] when the index is empty,
    /// [`RetrievalError::Build`] on shape/dim mismatch, and
    /// [`RetrievalError::Candle`] on tensor-layout failures.
    pub fn retrieve(
        &self,
        queries: &Tensor,
        top_k: usize,
        blend: f32,
    ) -> Result<Tensor, RetrievalError> {
        if self.is_empty() {
            return Err(RetrievalError::EmptyIndex);
        }

        let dims = queries.dims();
        if dims.len() != 3 || dims[0] != 1 {
            return Err(RetrievalError::Build(format!(
                "expected (1, n_frames, hidden_dim) query tensor, got shape {dims:?}",
            )));
        }
        let (n_frames, hidden_dim) = (dims[1], dims[2]);
        if hidden_dim != self.hidden_dim {
            return Err(RetrievalError::Build(format!(
                "query hidden_dim {} != index hidden_dim {}",
                hidden_dim, self.hidden_dim,
            )));
        }

        let device = queries.device().clone();
        let dtype = queries.dtype();

        // n_frames == 0 is a valid (no-op) input — return an empty-frame
        // tensor with the same dtype/device as the input.
        if n_frames == 0 {
            return Ok(queries.clone());
        }

        let cpu_queries = queries
            .to_device(&candle_core::Device::Cpu)?
            .to_dtype(candle_core::DType::F32)?
            .squeeze(0)?;
        let query_rows: Vec<Vec<f32>> = cpu_queries.to_vec2::<f32>()?;

        let blend = blend.clamp(0.0, 1.0);
        let k = top_k.clamp(1, self.len());

        let mut search = Search::default();
        let mut out: Vec<f32> = Vec::with_capacity(n_frames * hidden_dim);

        for query in &query_rows {
            let normalized = ContentFeature(l2_normalize(query));

            // `instant_distance` returns results in nearest-first order;
            // take the first `k` of them.
            let mut acc = vec![0.0_f32; hidden_dim];
            let mut count: usize = 0;
            for item in self.index.search(&normalized, &mut search).take(k) {
                for (a, v) in acc.iter_mut().zip(item.value.iter()) {
                    *a += *v;
                }
                count += 1;
            }
            if count == 0 {
                // Defensive — non-empty index should always return at
                // least one neighbor, but if the HNSW happens to yield
                // nothing, fall back to the query itself.
                out.extend_from_slice(query);
                continue;
            }
            // count is bounded by top_k (clamped to <= len() and small in
            // practice — typically 8); the f32 cast is exact for any
            // realistic value here.
            #[allow(clippy::cast_precision_loss)]
            let inv = 1.0 / count as f32;
            for v in &mut acc {
                *v *= inv;
            }

            for (q, m) in query.iter().zip(acc.iter()) {
                out.push((1.0 - blend) * *q + blend * *m);
            }
        }

        let stacked = Tensor::from_vec(out, (1, n_frames, hidden_dim), &device)?;
        let stacked = stacked.to_dtype(dtype)?;
        Ok(stacked)
    }
}

/// L2-normalize a feature vector. Zero (or near-zero-norm) vectors are
/// returned unchanged — distance from anything to a zero vector is then
/// effectively 1.0 (cosine to zero is undefined; we treat it as maximal).
fn l2_normalize(v: &[f32]) -> Vec<f32> {
    let norm_sq: f32 = v.iter().map(|x| x * x).sum();
    let norm = norm_sq.sqrt();
    if norm <= f32::EPSILON {
        return v.to_vec();
    }
    let inv = 1.0 / norm;
    v.iter().map(|x| *x * inv).collect()
}

#[cfg(test)]
#[allow(clippy::cast_precision_loss)]
mod tests {
    use super::*;
    use candle_core::Device;

    fn make_features(n: usize, dim: usize, seed: u64) -> Tensor {
        // Simple linear-congruential pseudo-random values, no external
        // rand dep required.
        let mut state = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15);
        let mut data = Vec::with_capacity(n * dim);
        for _ in 0..(n * dim) {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1);
            // Convert the high 24 bits into a float in roughly [-1, 1].
            let bits = (state >> 40) as u32;
            let val = ((bits & 0xFF_FFFF) as f32 / (0xFF_FFFF as f32 / 2.0)) - 1.0;
            data.push(val);
        }
        Tensor::from_vec(data, (n, dim), &Device::Cpu).expect("build training tensor")
    }

    #[test]
    fn build_and_retrieve_roundtrip() {
        let dim = 16;
        let train = make_features(100, dim, 0xC0FF_EE00);
        let index = FeatureIndex::build(&train).expect("build index");
        assert_eq!(index.len(), 100);
        assert_eq!(index.hidden_dim(), dim);

        let query = make_features(5, dim, 0xBEEF_BEEF)
            .unsqueeze(0)
            .expect("add batch dim");
        let top_k = 8;

        // Sanity check on shape with a mid-range blend.
        let blended = index
            .retrieve(&query, top_k, 0.5)
            .expect("retrieve blend=0.5");
        assert_eq!(blended.dims(), &[1, 5, dim]);

        // With blend = 1.0, output should equal the per-frame mean of the
        // top_k neighbors and be distinct from the query.
        let only_index = index
            .retrieve(&query, top_k, 1.0)
            .expect("retrieve blend=1.0");
        assert_eq!(only_index.dims(), &[1, 5, dim]);

        let query_rows = query.squeeze(0).unwrap().to_vec2::<f32>().unwrap();
        let out_rows = only_index.squeeze(0).unwrap().to_vec2::<f32>().unwrap();

        // Manually compute the expected mean for the first query row and
        // compare against the retrieved output to confirm we're returning
        // the neighbor mean (not the query) at blend=1.0.
        let train_rows = train.to_vec2::<f32>().unwrap();
        let q0 = &query_rows[0];
        let q0_norm = l2_normalize(q0);
        // Brute-force the top_k by cosine distance.
        let mut scored: Vec<(f32, usize)> = train_rows
            .iter()
            .enumerate()
            .map(|(i, row)| {
                let n = l2_normalize(row);
                let dot: f32 = q0_norm.iter().zip(n.iter()).map(|(a, b)| a * b).sum();
                ((1.0 - dot).max(0.0), i)
            })
            .collect();
        scored.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        let mut expected_mean = vec![0.0_f32; dim];
        for &(_, idx) in scored.iter().take(top_k) {
            for (acc, v) in expected_mean.iter_mut().zip(train_rows[idx].iter()) {
                *acc += *v;
            }
        }
        let inv = 1.0 / top_k as f32;
        for v in &mut expected_mean {
            *v *= inv;
        }

        // HNSW is approximate; allow some slack but the mean should be
        // far closer to the brute-force mean than to the original query.
        let dist_to_mean: f32 = out_rows[0]
            .iter()
            .zip(expected_mean.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f32>()
            .sqrt();
        let dist_to_query: f32 = out_rows[0]
            .iter()
            .zip(q0.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f32>()
            .sqrt();
        assert!(
            dist_to_mean < dist_to_query,
            "blend=1.0 output should be closer to neighbor mean than to query \
             (got dist_to_mean={dist_to_mean}, dist_to_query={dist_to_query})",
        );
    }

    #[test]
    fn save_load_roundtrip() {
        let dim = 16;
        let train = make_features(100, dim, 0x5A5A_5A5A);
        let index = FeatureIndex::build(&train).expect("build index");

        // tempfile is in workspace dev-deps; prefer it over hand-rolling a
        // ~/.cache directory so we don't leak files on test failure.
        let dir = tempfile::tempdir().expect("create tempdir");
        let path = dir.path().join("rvc-index.bin");
        index.save(&path).expect("save index");
        let loaded = FeatureIndex::load(&path).expect("load index");

        assert_eq!(loaded.len(), index.len());
        assert_eq!(loaded.hidden_dim(), index.hidden_dim());

        let query = make_features(5, dim, 0x1234_5678)
            .unsqueeze(0)
            .expect("add batch dim");
        let top_k = 8;
        let blend = 0.6;

        let before = index
            .retrieve(&query, top_k, blend)
            .expect("retrieve in-memory");
        let after = loaded
            .retrieve(&query, top_k, blend)
            .expect("retrieve loaded");

        let before_v = before.squeeze(0).unwrap().to_vec2::<f32>().unwrap();
        let after_v = after.squeeze(0).unwrap().to_vec2::<f32>().unwrap();
        assert_eq!(before_v.len(), after_v.len());
        for (a, b) in before_v.iter().zip(after_v.iter()) {
            assert_eq!(a, b, "round-trip should be byte-identical");
        }
    }

    #[test]
    fn empty_index_errors() {
        let empty =
            Tensor::from_vec(Vec::<f32>::new(), (0, 16), &Device::Cpu).expect("build empty tensor");
        match FeatureIndex::build(&empty) {
            Err(RetrievalError::EmptyIndex) => {}
            Err(other) => panic!("expected EmptyIndex, got {other:?}"),
            Ok(_) => panic!("expected EmptyIndex, got Ok(FeatureIndex)"),
        }
    }
}
