//! `BiCodec` neural audio codec (Wave S.2).
//!
//! Sub-wave plan:
//!
//! - **S.2.1.a** (this commit): shared low-level primitives —
//!   [`primitives::Snake1d`], [`primitives::WeightNormConv1d`],
//!   [`primitives::WeightNormConvTranspose1d`],
//!   [`primitives::AdaLayerNorm`], [`primitives::ResidualUnit`],
//!   [`primitives::repeat_interleave_dim2`].
//! - **S.2.1.b**: vocos backbone.
//! - **S.2.1.c**: sampler.
//! - **S.2.1.d**: quantizer.
//! - **S.2.1.e**: speaker.
//! - **S.2.1.f**: decoder.
//! - **S.2.1.g**: top-level `BiCodec` wiring.

#[allow(
    dead_code,
    reason = "Shared primitives are consumed by the S.2.1.{b..g} BiCodec \
              sub-waves (vocos, sampler, quantizer, speaker, decoder, \
              top-level). The unit tests in primitives.rs exercise the \
              public surface in the meantime."
)]
pub(super) mod primitives;
