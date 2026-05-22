//! Node binding for [`blazen_telemetry::TracingModel`].
//!
//! Exposes a `withTracing(name)` decorator on [`JsModel`] that
//! wraps the inner [`Model`] in a tracing span emitter. The
//! underlying Rust wrapper is generic over `M: Model` and
//! demands a `&'static str` provider name; we adapt this to the dynamic
//! NAPI shape via a small forwarding wrapper around `Arc<dyn Model>`
//! and by leaking the JS-provided name into a static string.

use std::pin::Pin;
use std::result::Result as StdResult;
use std::sync::Arc;

use async_trait::async_trait;
use futures_util::Stream;
use napi::bindgen_prelude::*;
use napi_derive::napi;

use blazen_llm::{BlazenError, Model, ModelRequest, ModelResponse, StreamChunk};
use blazen_telemetry::TracingModel;

use crate::providers::model::JsModel;

/// Adapter that lets `TracingModel` wrap an `Arc<dyn Model>`.
///
/// The underlying generic parameter `M` requires a `Sized + Model`,
/// which `dyn Model` is not. This forwards every trait method
/// through the trait-object pointer.
struct ArcModel(Arc<dyn Model>);

#[async_trait]
impl Model for ArcModel {
    fn model_id(&self) -> &str {
        self.0.model_id()
    }

    async fn complete(&self, request: ModelRequest) -> StdResult<ModelResponse, BlazenError> {
        self.0.complete(request).await
    }

    async fn stream(
        &self,
        request: ModelRequest,
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
impl JsModel {
    /// Wrap this model in a [`TracingModel`] that emits a
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
    /// const traced = Model.openai({ apiKey }).withTracing("openai");
    /// ```
    #[napi(js_name = "withTracing")]
    pub fn with_tracing(&self, name: String) -> Result<JsModel> {
        let inner = self.inner.as_ref().ok_or_else(|| {
            napi::Error::from_reason("withTracing() is not supported on subclassed Model instances")
        })?;
        let static_name: &'static str = Box::leak(name.into_boxed_str());
        let adapter = ArcModel(Arc::clone(inner));
        let traced = TracingModel::new(adapter, static_name);
        Ok(JsModel {
            inner: Some(Arc::new(traced)),
            local_model: self.local_model.clone(),
            config: None,
        })
    }
}
