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
use blazen_telemetry::{TracingConfig, TracingModel};

use crate::providers::model::JsModel;

// ---------------------------------------------------------------------------
// JsTracingConfig
// ---------------------------------------------------------------------------

/// Runtime configuration for the tracing wrapper installed by
/// [`JsModel::with_tracing`](JsModel::with_tracing).
///
/// Defaults are privacy-safe: token counts, model id, provider, and finish
/// reason are always recorded; the raw prompt + completion message text is
/// captured only when `captureMessages` is `true`.
///
/// ```javascript
/// const traced = Model.openai({ apiKey }).withTracingConfig({ captureMessages: true });
/// ```
#[napi(object, js_name = "TracingConfig")]
pub struct JsTracingConfig {
    /// Capture raw prompt + completion message text as span attributes
    /// (`llm.input_messages` / `llm.output_messages`). Defaults to `false`.
    /// Leave off for privacy-sensitive deployments.
    pub capture_messages: Option<bool>,
}

impl From<JsTracingConfig> for TracingConfig {
    fn from(c: JsTracingConfig) -> Self {
        TracingConfig::default().with_message_capture(c.capture_messages.unwrap_or(false))
    }
}

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
    /// `name` is recorded on the span as the `provider` field plus the
    /// `OpenInference` / `gen_ai.*` aliases (`gen_ai.system`, etc.). It is
    /// leaked into a `&'static str` because the underlying span macro
    /// captures it by reference for the process lifetime; this is
    /// intentional and bounded by the small set of distinct provider
    /// names a typical application uses.
    ///
    /// `captureMessages` (default `false`) opts into recording the raw
    /// prompt + completion text as `llm.input_messages` /
    /// `llm.output_messages` for Phoenix eval-grade ingest. Leave off for
    /// privacy-sensitive deployments.
    ///
    /// ```javascript
    /// const traced = Model.openai({ apiKey }).withTracing("openai");
    /// const evalGrade = Model.openai({ apiKey }).withTracing("openai", true);
    /// ```
    #[napi(js_name = "withTracing")]
    pub fn with_tracing(&self, name: String, capture_messages: Option<bool>) -> Result<JsModel> {
        let inner = self.inner.as_ref().ok_or_else(|| {
            napi::Error::from_reason("withTracing() is not supported on subclassed Model instances")
        })?;
        let static_name: &'static str = Box::leak(name.into_boxed_str());
        let adapter = ArcModel(Arc::clone(inner));
        let config =
            TracingConfig::default().with_message_capture(capture_messages.unwrap_or(false));
        let traced = TracingModel::new(adapter, static_name, config);
        Ok(JsModel {
            inner: Some(Arc::new(traced)),
            local_model: self.local_model.clone(),
            config: None,
        })
    }

    /// Wrap this model in a [`TracingModel`] using an explicit
    /// [`TracingConfig`](JsTracingConfig) object.
    ///
    /// Equivalent to [`with_tracing`](Self::with_tracing) but takes the
    /// structured config record (`{ captureMessages?: boolean }`) instead of
    /// a positional boolean. `name` is leaked into a `&'static str` for the
    /// span macro, exactly as in [`with_tracing`](Self::with_tracing).
    ///
    /// ```javascript
    /// const traced = Model.openai({ apiKey })
    ///   .withTracingConfig("openai", { captureMessages: true });
    /// ```
    #[napi(js_name = "withTracingConfig")]
    pub fn with_tracing_config(&self, name: String, config: JsTracingConfig) -> Result<JsModel> {
        let inner = self.inner.as_ref().ok_or_else(|| {
            napi::Error::from_reason(
                "withTracingConfig() is not supported on subclassed Model instances",
            )
        })?;
        let static_name: &'static str = Box::leak(name.into_boxed_str());
        let adapter = ArcModel(Arc::clone(inner));
        let traced = TracingModel::new(adapter, static_name, config.into());
        Ok(JsModel {
            inner: Some(Arc::new(traced)),
            local_model: self.local_model.clone(),
            config: None,
        })
    }
}
