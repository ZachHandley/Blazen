//! Shared candle primitives for Diffusion Transformer (DiT) backends.
//!
//! Wave 0 of the audio capability completion push extracted these
//! reusable building blocks out of the Stable Audio Open Small backend
//! in `blazen-audio-music` so other audio + 3D ports (F5-TTS,
//! AudioLDM, future video DiTs) can share them without copying code.
//!
//! # What lives here
//!
//! - [`dit`] — multi-head attention (self + cross) and the SwiGLU
//!   feed-forward block.
//! - [`adaln`] — AdaLN-Zero modulation (the six-chunk
//!   `(scale_*, shift_*, gate_*)` projection used to drive a DiT block
//!   from a global conditioning vector) and the broadcasting
//!   [`modulate`](adaln::modulate) helper.
//! - [`rope`] — Rotary Position Embedding helpers
//!   ([`precompute_rope_freqs`](rope::precompute_rope_freqs) and the
//!   partial-rotary [`apply_rope`](rope::apply_rope)) plus a
//!   [`FourierFeatures`](rope::FourierFeatures) block for embedding
//!   continuous scalars.
//!
//! # What lives in the consuming backend
//!
//! - Backend-specific configs (e.g. `DiTConfig` for Stable Audio),
//!   conditioner wiring, numeric / timestep embedding heads, output
//!   projections sized to specific latent shapes, and the actual DiT
//!   block recipe (e.g. self-attn → cross-attn → SwiGLU FFN with
//!   `sigmoid(1 - gate)` blending for Stable Audio).
//!
//! These primitives are intentionally feature-free; consuming crates
//! gate their own backends as needed.

// The candle-error / shape-mismatch failure modes of every function in
// this crate are documented inline in the function bodies and through
// the candle docs; repeating an `# Errors` section on every wrapper
// would be pure noise. Same for `# Panics` — the `assert!` calls that
// would trip these lints all guard against compile-time-constant
// shape invariants (e.g. `embed_dim == num_heads * head_dim`) that
// any reasonable caller respects.
// `clippy::doc_markdown` fires on every math abbreviation
// (`RoPE`, `SiLU`, `SwiGLU`, `B`, `T`, `D`, `MLP`, ...) — these are
// terms-of-art that look wrong in backticks; suppress the lint at the
// crate level rather than papering over every doc line.
#![allow(
    clippy::missing_errors_doc,
    clippy::missing_panics_doc,
    clippy::doc_markdown
)]

pub mod adaln;
pub mod dit;
pub mod rope;
