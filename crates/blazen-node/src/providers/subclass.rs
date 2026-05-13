//! Bridge wrapping a JavaScript subclass of `CompletionModel` as a Rust
//! [`CompletionModel`].
//!
//! ## Status
//!
//! This module exists to define [`JsSubclassCompletionModel`], a future
//! hook for letting JavaScript users write `class MyLLM extends
//! CompletionModel { async complete(...) { ... } }` and pass instances of
//! `MyLLM` to `runAgent` and other Rust-driven entry points.
//!
//! Today the type is a stub: it carries the model id but its
//! [`CompletionModel::complete`] and [`CompletionModel::stream`]
//! implementations return [`BlazenError::Unsupported`]. The reason is
//! mechanical, not philosophical — wiring the subclass path end-to-end
//! requires:
//!
//! 1. `JsCompletionModel` (or `runAgent`'s argument) to retain a handle
//!    to the JavaScript instance so we can extract its `complete` /
//!    `stream` methods as [`ThreadsafeFunction`]s.
//! 2. `runAgent` to accept an `Object` parameter (not `&JsCompletionModel`)
//!    so napi can hand us the JS handle.
//!
//! Neither of those changes is in scope for this pass. Users who need
//! a custom Rust-driven completion model should call
//! [`crate::providers::JsCompletionModel::custom`] which accepts a JS
//! host object directly and wraps it via [`NodeHostDispatch`].
//!
//! Once the upstream wiring lands, the [`from_js_instance`] constructor
//! is the place to extract a per-method [`ThreadsafeFunction`] and the
//! trait impls below are the place to invoke them.
//!
//! [`NodeHostDispatch`]: crate::providers::custom::NodeHostDispatch
//! [`from_js_instance`]: JsSubclassCompletionModel::from_js_instance

use std::pin::Pin;

use async_trait::async_trait;
use futures_util::Stream;

use blazen_llm::CompletionModel;
use blazen_llm::error::BlazenError;
use blazen_llm::types::{CompletionRequest, CompletionResponse, StreamChunk};

/// A Rust [`CompletionModel`] backed by a JavaScript subclass of
/// `CompletionModel`.
///
/// See the [module docs](self) for the current implementation status.
pub(crate) struct JsSubclassCompletionModel {
    /// The `modelId` reported via the [`CompletionModel::model_id`]
    /// accessor. Pulled from the JS-side `config.modelId` when the
    /// instance was constructed.
    model_id: String,
}

impl std::fmt::Debug for JsSubclassCompletionModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("JsSubclassCompletionModel")
            .field("model_id", &self.model_id)
            .finish()
    }
}

impl JsSubclassCompletionModel {
    /// Build a [`JsSubclassCompletionModel`] from a JavaScript class
    /// instance.
    ///
    /// Currently this only records the `model_id`. The full wiring (see
    /// the module docs) requires upstream changes to `runAgent`'s
    /// signature.
    #[allow(dead_code)]
    pub(crate) fn new(model_id: String) -> Self {
        Self { model_id }
    }
}

#[async_trait]
impl CompletionModel for JsSubclassCompletionModel {
    fn model_id(&self) -> &str {
        &self.model_id
    }

    async fn complete(
        &self,
        _request: CompletionRequest,
    ) -> Result<CompletionResponse, BlazenError> {
        Err(BlazenError::unsupported(
            "complete() is not yet implemented for JS subclass CompletionModels. \
             Use CompletionModel.custom(hostObject) factory instead.",
        ))
    }

    async fn stream(
        &self,
        _request: CompletionRequest,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<StreamChunk, BlazenError>> + Send>>, BlazenError>
    {
        Err(BlazenError::unsupported(
            "stream() is not yet implemented for JS subclass CompletionModels. \
             Use CompletionModel.custom(hostObject) factory instead.",
        ))
    }
}
