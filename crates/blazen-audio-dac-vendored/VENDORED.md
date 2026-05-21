# Vendored: `candle_transformers::models::dac`

This crate is a vendored, patched fork of the
[`candle_transformers`][upstream] `dac` module — the Descript Audio
Codec wrapper that ships under `candle-transformers/src/models/dac.rs`.

| Field             | Value                                            |
|-------------------|--------------------------------------------------|
| Upstream repo     | https://github.com/huggingface/candle            |
| Pinned version    | `candle-transformers = "0.10.2"`                 |
| Upstream license  | MIT OR Apache-2.0                                |
| Workspace license | MPL-2.0 (this fork, redistributed under MIT's permissive terms) |
| Reference impl    | https://github.com/descriptinc/descript-audio-codec (`dac/nn/quantize.py`) |

## Why we vendor

Upstream `candle_transformers::models::dac::Model` exposes a single
public verb — [`decode_codes`][decode_codes] — and keeps the residual
vector quantiser's internals private:

```rust
// upstream candle-transformers v0.10.2
pub struct VectorQuantizer {
    in_proj: Conv1d,    // <-- private
    out_proj: Conv1d,
    codebook: candle_nn::Embedding,   // <-- private
}
```

Without `in_proj` and `codebook` an external crate cannot run
nearest-neighbour quantisation against a loaded checkpoint. The
result is that `blazen-audio-codec`'s `DacBackend::encode_pcm` was
forced to return `CodecError::NotYetImplemented`, breaking encode +
decode round-trips and blocking DAC as a tokenizer for downstream
audio-token language models. Vendoring + patching is the minimum
change needed to ship a working encode path without waiting on an
upstream release.

## Patches applied

| Subsystem | Upstream | Patched fork |
|-----------|----------|--------------|
| `VectorQuantizer.in_proj` visibility | `Conv1d` (private) | `pub Conv1d` |
| `VectorQuantizer.codebook` visibility | `candle_nn::Embedding` (private) | `pub candle_nn::Embedding` |
| `ResidualVectorQuantizer.quantizers` visibility | `Vec<VectorQuantizer>` (private) | `pub Vec<VectorQuantizer>` |
| `VectorQuantizer::forward` | *(missing)* | **new** — nearest-neighbour `[B, D, T] → ([B, D, T], [B, T])` quantisation, mirrors `descript-audio-codec/dac/nn/quantize.py::VectorQuantize.forward` (factorised codes + L2-normed cosine-similarity argmin) |
| `ResidualVectorQuantizer::forward_codes` | *(missing)* | **new** — residual-quantises `[B, D, T]` against all codebooks in turn, returns `[B, n_codebooks, T]` `u32` indices |
| `Model::encode_to_codes` | *(missing)* | **new** — `[B, 1, T] → [B, n_codebooks, T']` end-to-end encode path (encoder + RVQ.forward_codes) |

The remainder of the source is a verbatim port of
`candle-transformers/src/models/dac.rs` at the pinned version; please
keep the diff minimal when re-syncing.

## Implementation notes

The new `VectorQuantizer::forward` implements the encode-side of the
Descript reference quantiser:

1. `z_e = in_proj(z)` — project `[B, latent_dim, T]` down to the
   compact codebook space `[B, cb_dim, T]`.
2. Flatten to `[B*T, cb_dim]` and L2-normalise along the feature
   axis. Apply the same normalisation to the `[N, cb_dim]` codebook
   table.
3. Compute the full `‖a‖² − 2·a·bᵀ + ‖b‖²` Euclidean-distance matrix
   `[B*T, N]` against the codebook. (`‖·‖²` are ~1 after L2-norm but
   the explicit expansion matches the Python reference verbatim and
   is numerically more stable than `1 − cos` for near-orthogonal
   pairs.)
4. `argmax(−dist)` along the codebook axis → `u32` indices `[B, T]`.
5. Re-embed via the codebook + `out_proj` to recover the quantised
   `[B, latent_dim, T]` latent that the next residual stage will
   subtract.

`ResidualVectorQuantizer::forward_codes` then drives `forward` across
each codebook in turn, subtracting the per-step contribution from the
running residual (the standard SoundStream-style RVQ schedule —
inference-mode only, so no quantiser dropout). The returned
`[B, n_codebooks, T]` `Tensor` round-trips exactly through the
existing `from_codes` / `decode_codes` decode path.

## How to re-sync from upstream

When `candle-transformers` ships a new release:

1. `cargo update -p candle-transformers` in the workspace and let
   cargo refetch the new version into
   `~/.cargo/registry/src/.../candle-transformers-<NEW>/src/models/dac.rs`.
2. Diff that file against `src/lib.rs` in this crate (the verbatim
   parts should match line-for-line modulo the patches above).
3. Re-apply the visibility tweaks and the three `new` methods. The
   `forward` body should not need changes unless upstream rewrites
   the quantiser graph (very unlikely — it has been stable since
   the original DAC release).
4. Update the **Pinned version** row at the top of this file.
5. Run `cargo clippy -p blazen-audio-dac-vendored --all-features
   --tests -- -D warnings` and the
   `blazen-audio-codec --features live-models,dac` round-trip test.

[upstream]: https://github.com/huggingface/candle/tree/main/candle-transformers
[decode_codes]: https://docs.rs/candle-transformers/0.10.2/candle_transformers/models/dac/struct.Model.html#method.decode_codes
