//! Typed compute result types -- re-exported from `blazen_llm::compute`.
//!
//! Each result type derives [`tsify_next::Tsify`] upstream with
//! `into_wasm_abi` + `from_wasm_abi`, so the `pub use` is sufficient to
//! surface them as TypeScript interfaces in `blazen_wasm_sdk.d.ts`.

pub use blazen_llm::compute::{
    AudioResult, ImageResult, ThreeDResult, TranscriptionResult, TranscriptionSegment, VideoResult,
    VoiceHandle,
};
