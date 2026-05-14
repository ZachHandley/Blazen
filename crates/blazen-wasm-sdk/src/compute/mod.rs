//! Compute request/response types re-exported from `blazen-llm`.
//!
//! These types derive [`tsify_next::Tsify`] upstream (via the `tsify` feature,
//! which `blazen-wasm-sdk` enables on `blazen-llm`) and surface as TypeScript
//! interfaces in the generated `blazen_wasm_sdk.d.ts`. No hand-written
//! `wasm-bindgen` wrapper classes are needed -- `Tsify` already handles the
//! ABI translation through `serde-wasm-bindgen` under the hood.

pub mod requests;
pub mod results;

pub use requests::*;
pub use results::*;
