//! Node bindings for [`blazen_model_cache`].
//!
//! Exposes [`JsModelCache`] (`ModelCache` on the JS side) for downloading and
//! caching ML models from `HuggingFace` Hub.

pub mod cache;
pub mod error;

pub use cache::JsModelCache;
pub use error::cache_error_to_napi;
