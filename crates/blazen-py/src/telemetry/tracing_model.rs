//! Python binding for `blazen_telemetry::TracingCompletionModel`.
//!
//! Wraps a [`PyCompletionModel`] in a tracing-instrumented decorator that
//! emits structured `tracing` spans for every LLM call.

use std::pin::Pin;
use std::sync::Arc;

use async_trait::async_trait;
use pyo3::prelude::*;
use pyo3_stub_gen::derive::gen_stub_pyfunction;
use tokio_stream::Stream;

use blazen_llm::{
    BlazenError, CompletionModel, CompletionRequest, CompletionResponse, ProviderConfig,
    StreamChunk,
};
use blazen_telemetry::TracingCompletionModel;

use crate::providers::PyCompletionModel;
use crate::providers::completion_model::arc_from_bound;

// ---------------------------------------------------------------------------
// Arc<dyn CompletionModel> -> CompletionModel adapter
// ---------------------------------------------------------------------------

/// Newtype that lets `Arc<dyn CompletionModel>` satisfy the
/// `M: CompletionModel` bound on `TracingCompletionModel<M>`.
struct ArcModel(Arc<dyn CompletionModel>);

#[async_trait]
impl CompletionModel for ArcModel {
    fn model_id(&self) -> &str {
        self.0.model_id()
    }

    async fn complete(
        &self,
        request: CompletionRequest,
    ) -> Result<CompletionResponse, BlazenError> {
        self.0.complete(request).await
    }

    async fn stream(
        &self,
        request: CompletionRequest,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<StreamChunk, BlazenError>> + Send>>, BlazenError>
    {
        self.0.stream(request).await
    }

    fn provider_config(&self) -> Option<&ProviderConfig> {
        self.0.provider_config()
    }
}

// ---------------------------------------------------------------------------
// wrap_with_tracing
// ---------------------------------------------------------------------------

/// Wrap a `CompletionModel` with a tracing decorator.
///
/// Each call to `complete()` or `stream()` is instrumented with an
/// `info_span!` carrying provider name, model id, token usage, duration,
/// and finish reason.
///
/// Args:
///     model: The CompletionModel to wrap.
///     provider_name: Static label included in every emitted span.
///         Note: the underlying Rust type requires a `'static` lifetime,
///         so the provided string is leaked for the lifetime of the
///         process. Use a small, stable set of labels.
///
/// Returns:
///     A new CompletionModel that emits tracing spans on every call.
#[gen_stub_pyfunction]
#[pyfunction]
pub fn wrap_with_tracing(
    model: Bound<'_, PyCompletionModel>,
    provider_name: String,
) -> PyCompletionModel {
    let local_model = model.borrow().local_model.clone();
    let inner = arc_from_bound(&model);
    let leaked: &'static str = Box::leak(provider_name.into_boxed_str());
    let wrapped = TracingCompletionModel::new(ArcModel(inner), leaked);
    PyCompletionModel {
        inner: Some(Arc::new(wrapped)),
        local_model,
        config: None,
    }
}
