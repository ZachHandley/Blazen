//! Node binding for [`blazen_telemetry::TracingCompletionModel`].
//!
//! Exposes a `withTracing(name)` decorator on [`JsCompletionModel`] that
//! wraps the inner [`CompletionModel`] in a tracing span emitter. The
//! underlying Rust wrapper is generic over `M: CompletionModel` and
//! demands a `&'static str` provider name; we adapt this to the dynamic
//! NAPI shape via a small forwarding wrapper around `Arc<dyn CompletionModel>`
//! and by leaking the JS-provided name into a static string.

use std::pin::Pin;
use std::result::Result as StdResult;
use std::sync::Arc;

use async_trait::async_trait;
use futures_util::Stream;
use napi::bindgen_prelude::*;
use napi_derive::napi;

use blazen_llm::{
    BlazenError, CompletionModel, CompletionRequest, CompletionResponse, StreamChunk,
};
use blazen_telemetry::TracingCompletionModel;

use crate::providers::completion_model::JsCompletionModel;

/// Adapter that lets `TracingCompletionModel` wrap an `Arc<dyn CompletionModel>`.
///
/// The underlying generic parameter `M` requires a `Sized + CompletionModel`,
/// which `dyn CompletionModel` is not. This forwards every trait method
/// through the trait-object pointer.
struct ArcCompletionModel(Arc<dyn CompletionModel>);

#[async_trait]
impl CompletionModel for ArcCompletionModel {
    fn model_id(&self) -> &str {
        self.0.model_id()
    }

    async fn complete(
        &self,
        request: CompletionRequest,
    ) -> StdResult<CompletionResponse, BlazenError> {
        self.0.complete(request).await
    }

    async fn stream(
        &self,
        request: CompletionRequest,
    ) -> StdResult<
        Pin<Box<dyn Stream<Item = StdResult<StreamChunk, BlazenError>> + Send>>,
        BlazenError,
    > {
        self.0.stream(request).await
    }

    fn provider_config(&self) -> Option<&blazen_llm::ProviderConfig> {
        self.0.provider_config()
    }
}

#[napi]
#[allow(clippy::must_use_candidate, clippy::missing_errors_doc)]
impl JsCompletionModel {
    /// Wrap this model in a [`TracingCompletionModel`] that emits a
    /// structured `tracing` span around every `complete` and `stream`
    /// call.
    ///
    /// `name` is recorded on the span as the `provider` field. It is
    /// leaked into a `&'static str` because the underlying span macro
    /// captures it by reference for the process lifetime; this is
    /// intentional and bounded by the small set of distinct provider
    /// names a typical application uses.
    ///
    /// ```javascript
    /// const traced = CompletionModel.openai({ apiKey }).withTracing("openai");
    /// ```
    #[napi(js_name = "withTracing")]
    pub fn with_tracing(&self, name: String) -> Result<JsCompletionModel> {
        let inner = self.inner.as_ref().ok_or_else(|| {
            napi::Error::from_reason(
                "withTracing() is not supported on subclassed CompletionModel instances",
            )
        })?;
        let static_name: &'static str = Box::leak(name.into_boxed_str());
        let adapter = ArcCompletionModel(Arc::clone(inner));
        let traced = TracingCompletionModel::new(adapter, static_name);
        Ok(JsCompletionModel {
            inner: Some(Arc::new(traced)),
            local_model: self.local_model.clone(),
            config: None,
        })
    }
}
