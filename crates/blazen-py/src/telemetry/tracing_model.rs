//! Python binding for `blazen_telemetry::TracingModel`.
//!
//! Wraps a [`PyModel`] in a tracing-instrumented decorator that
//! emits structured `tracing` spans for every LLM call.

use std::pin::Pin;
use std::sync::Arc;

use async_trait::async_trait;
use pyo3::prelude::*;
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pyfunction, gen_stub_pymethods};
use tokio_stream::Stream;

use blazen_llm::{BlazenError, Model, ModelRequest, ModelResponse, ProviderConfig, StreamChunk};
use blazen_telemetry::{TracingConfig, TracingModel};

use crate::providers::PyModel;
use crate::providers::model::arc_from_bound;

// ---------------------------------------------------------------------------
// Arc<dyn Model> -> Model adapter
// ---------------------------------------------------------------------------

/// Newtype that lets `Arc<dyn Model>` satisfy the
/// `M: Model` bound on `TracingModel<M>`.
struct ArcModel(Arc<dyn Model>);

#[async_trait]
impl Model for ArcModel {
    fn model_id(&self) -> &str {
        self.0.model_id()
    }

    async fn complete(&self, request: ModelRequest) -> Result<ModelResponse, BlazenError> {
        self.0.complete(request).await
    }

    async fn stream(
        &self,
        request: ModelRequest,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<StreamChunk, BlazenError>> + Send>>, BlazenError>
    {
        self.0.stream(request).await
    }

    fn provider_config(&self) -> Option<&ProviderConfig> {
        self.0.provider_config()
    }
}

// ---------------------------------------------------------------------------
// PyTracingConfig
// ---------------------------------------------------------------------------

/// Runtime configuration for the tracing decorator built by
/// :func:`wrap_with_tracing`.
///
/// Defaults are privacy-safe: token counts, model id, provider, and finish
/// reason are always recorded, but raw prompt + completion message text are
/// NOT recorded unless ``capture_messages`` is enabled.
///
/// Example:
/// ```text
///  >>> cfg = TracingConfig(capture_messages=True)
///  >>> traced = wrap_with_tracing(model, "openai", config=cfg)
/// ```
#[gen_stub_pyclass]
#[pyclass(name = "TracingConfig", from_py_object)]
#[derive(Clone)]
pub struct PyTracingConfig {
    pub(crate) inner: TracingConfig,
}

#[gen_stub_pymethods]
#[pymethods]
impl PyTracingConfig {
    /// Create a new tracing configuration.
    ///
    /// Args:
    ///     capture_messages: If True, the raw prompt messages and completion
    ///         text are serialized to JSON and recorded on each span as
    ///         ``llm.input_messages`` / ``llm.output_messages``. Defaults to
    ///         False (privacy-safe). Stream calls never capture messages even
    ///         with this flag enabled.
    #[new]
    #[pyo3(signature = (capture_messages=false))]
    fn new(capture_messages: bool) -> Self {
        Self {
            inner: TracingConfig::default().with_message_capture(capture_messages),
        }
    }

    /// Whether raw prompt + completion message text is captured on spans.
    #[getter]
    fn capture_messages(&self) -> bool {
        self.inner.capture_messages()
    }

    #[setter]
    fn set_capture_messages(&mut self, capture: bool) {
        self.inner = self.inner.with_message_capture(capture);
    }

    /// Return a copy with message capture enabled or disabled.
    fn with_message_capture(&self, capture: bool) -> Self {
        Self {
            inner: self.inner.with_message_capture(capture),
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "TracingConfig(capture_messages={})",
            if self.inner.capture_messages() {
                "True"
            } else {
                "False"
            }
        )
    }
}

// ---------------------------------------------------------------------------
// wrap_with_tracing
// ---------------------------------------------------------------------------

/// Wrap a `Model` with a tracing decorator.
///
/// Each call to `complete()` or `stream()` is instrumented with an
/// `info_span!` carrying provider name, model id, token usage, duration,
/// finish reason, and OpenInference + ``gen_ai.*`` aliases for eval-grade
/// ingest by Phoenix Arize and PostHog AI.
///
/// Args:
///     model: The Model to wrap.
///     provider_name: Static label included in every emitted span.
///         Note: the underlying Rust type requires a `'static` lifetime,
///         so the provided string is leaked for the lifetime of the
///         process. Use a small, stable set of labels.
///     capture_messages: If True, the raw prompt messages and completion
///         text are serialized to JSON and recorded on the span as
///         ``llm.input_messages`` / ``llm.output_messages``. Defaults to
///         False — privacy-safe. Phoenix's eval-grade surfaces
///         (faithfulness, RAG hit rate, schema-compliance) need this
///         enabled. Ignored when ``config`` is supplied.
///     config: Optional :class:`TracingConfig`. When provided, it takes
///         precedence over ``capture_messages``.
///
/// Returns:
///     A new Model that emits tracing spans on every call.
#[gen_stub_pyfunction]
#[pyfunction]
#[pyo3(signature = (model, provider_name, capture_messages=false, config=None))]
pub fn wrap_with_tracing(
    model: Bound<'_, PyModel>,
    provider_name: String,
    capture_messages: bool,
    config: Option<PyTracingConfig>,
) -> PyModel {
    let local_model = model.borrow().local_model.clone();
    let inner = arc_from_bound(&model);
    let leaked: &'static str = Box::leak(provider_name.into_boxed_str());
    let config = config.map_or_else(
        || TracingConfig::default().with_message_capture(capture_messages),
        |c| c.inner,
    );
    let wrapped = TracingModel::new(ArcModel(inner), leaked, config);
    PyModel {
        inner: Some(Arc::new(wrapped)),
        local_model,
        config: None,
    }
}
