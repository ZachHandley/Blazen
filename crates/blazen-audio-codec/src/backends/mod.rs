//! Concrete codec backend implementations.
//!
//! Each backend is feature-gated. Compiling without any backend feature
//! produces a usable crate that exposes only the [`crate::CodecBackend`]
//! trait surface — useful for downstream crates that just want to consume
//! `Arc<dyn CodecBackend>` without dragging in candle.
//!
//! | Module | Feature | Status |
//! |---|---|---|
//! | [`encodec`] | `encodec` | Functional — Meta's `facebook/encodec_24khz` neural codec via `candle-transformers`. |
//! | [`dac`] | `dac` | Functional decode — Descript Audio Codec (`descript/dac_44khz`) via `candle-transformers`. Encode short-circuits until candle exposes a public RVQ encode path. |
//! | [`snac`] | `snac` | Functional — Multi-Scale Neural Audio Codec (`hubertsiuzdak/snac_24khz`, 3 codebooks @ vq_strides [4, 2, 1], 24 kHz) via `candle-transformers`. Both encode and decode are wired. |

#[cfg(feature = "encodec")]
pub mod encodec;

#[cfg(feature = "dac")]
pub mod dac;

#[cfg(feature = "snac")]
pub mod snac;

// When the `dac` feature is off, keep a tiny compile-time-only stub module
// so doc links and downstream `use blazen_audio_codec::backends::dac` paths
// stay resolvable. The stub does NOT implement `CodecBackend` (it would
// require the candle stack); attempts to use it surface as a clear
// missing-feature error at the call site.
#[cfg(not(feature = "dac"))]
pub mod dac {
    //! DAC backend slot — disabled at build time.
    //!
    //! Rebuild with `--features dac` (or `--features dac,cuda` /
    //! `--features dac,metal`) to enable the real Descript Audio Codec
    //! wrapper.

    /// Compile-time marker that the DAC backend is not built in. The real
    /// type lives at the same path when the `dac` feature is enabled.
    #[derive(Debug, Default, Clone, Copy)]
    pub struct DacBackendDisabled;
}

// Same compile-time stub pattern for SNAC when its feature is off.
#[cfg(not(feature = "snac"))]
pub mod snac {
    //! SNAC backend slot — disabled at build time.
    //!
    //! Rebuild with `--features snac` (or `--features snac,cuda` /
    //! `--features snac,metal`) to enable the real Multi-Scale Neural
    //! Audio Codec wrapper.

    /// Compile-time marker that the SNAC backend is not built in. The real
    /// type lives at the same path when the `snac` feature is enabled.
    #[derive(Debug, Default, Clone, Copy)]
    pub struct SnacBackendDisabled;
}
