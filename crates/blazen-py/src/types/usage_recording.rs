//! Python wrappers for `blazen_llm::usage_recording`: the `UsageEmitter` ABC
//! and the `UsageRecording*` decorators that emit usage events on every
//! provider call.

use std::pin::Pin;
use std::sync::Arc;

use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pymethods};
use tokio_stream::Stream;
use uuid::Uuid;

use blazen_events::UsageEvent;
use blazen_llm::usage_recording::{
    NoopUsageEmitter, UsageEmitter, UsageRecordingCompletionModel, UsageRecordingEmbeddingModel,
};
use blazen_llm::{BlazenError, ChatMessage, CompletionModel, EmbeddingModel, StreamChunk};

use crate::error::BlazenPyError;
use crate::events::PyUsageEvent;
use crate::providers::completion_model::{
    PyCompletionModel, PyCompletionOptions, arc_from_bound, build_request,
};
use crate::types::{PyChatMessage, PyCompletionResponse, PyEmbeddingModel, PyEmbeddingResponse};

// ---------------------------------------------------------------------------
// PyUsageEmitter -- subclassable ABC
// ---------------------------------------------------------------------------

/// Subclassable ABC mirroring `blazen_llm::usage_recording::UsageEmitter`.
///
/// Implementations sink emitted `UsageEvent`s somewhere observable: a
/// workflow's broadcast stream, a tokio channel, a tracing span field, etc.
///
/// Override ``emit`` in a subclass to handle each event. The default
/// implementation raises ``NotImplementedError``.
///
/// Example:
///     >>> class CountingEmitter(UsageEmitter):
///     ...     def __init__(self):
///     ...         self.count = 0
///     ...     def emit(self, event: UsageEvent) -> None:
///     ...         self.count += 1
#[gen_stub_pyclass]
#[pyclass(name = "UsageEmitter", subclass)]
pub struct PyUsageEmitter;

#[gen_stub_pymethods]
#[pymethods]
impl PyUsageEmitter {
    #[new]
    fn new() -> Self {
        Self
    }

    /// Subclass-friendly `__init__` no-op so `super().__init__()` chains
    /// don't fall through to `object.__init__` and raise ``TypeError``.
    fn __init__(&self) {}

    /// Sink a single `UsageEvent`. Subclasses override this method.
    fn emit(&self, _event: &Bound<'_, PyAny>) -> PyResult<()> {
        Err(pyo3::exceptions::PyNotImplementedError::new_err(
            "subclass must override emit()",
        ))
    }
}

// ---------------------------------------------------------------------------
// PyNoopUsageEmitter -- the default no-op sink
// ---------------------------------------------------------------------------

/// A no-op `UsageEmitter` that drops every event.
///
/// Useful as a default when no downstream observer is wired up.
#[gen_stub_pyclass]
#[pyclass(name = "NoopUsageEmitter", extends = PyUsageEmitter, from_py_object)]
#[derive(Clone)]
pub struct PyNoopUsageEmitter;

#[gen_stub_pymethods]
#[pymethods]
impl PyNoopUsageEmitter {
    #[new]
    fn new() -> (PyNoopUsageEmitter, PyUsageEmitter) {
        (PyNoopUsageEmitter, PyUsageEmitter)
    }

    /// No-op emit; the event is discarded.
    fn emit(&self, _event: &Bound<'_, PyAny>) {}

    fn __repr__(&self) -> &'static str {
        "NoopUsageEmitter()"
    }
}

// ---------------------------------------------------------------------------
// PyHostUsageEmitter -- Rust trait adapter over a Python subclass
// ---------------------------------------------------------------------------

/// Adapter that implements [`UsageEmitter`] by calling `emit(event)` on a
/// Python subclass instance.
#[derive(Debug)]
pub(crate) struct PyHostUsageEmitter {
    py_obj: Py<PyAny>,
}

impl PyHostUsageEmitter {
    pub fn new(py_obj: Py<PyAny>) -> Self {
        Self { py_obj }
    }
}

impl UsageEmitter for PyHostUsageEmitter {
    fn emit(&self, event: UsageEvent) {
        // `UsageEmitter::emit` is sync. We acquire the GIL, build a
        // `PyUsageEvent` wrapper, and call the subclass's `emit`. Errors
        // are logged via tracing -- propagating a panic out of an
        // emitter would tear down the whole completion call.
        Python::attach(|py| {
            let py_event = Py::new(py, PyUsageEvent::from(event));
            let py_event = match py_event {
                Ok(e) => e,
                Err(e) => {
                    tracing::warn!("PyHostUsageEmitter: failed to construct PyUsageEvent: {e}");
                    return;
                }
            };
            if let Err(e) = self.py_obj.call_method1(py, "emit", (py_event,)) {
                tracing::warn!("PyHostUsageEmitter: subclass emit() raised: {e}");
            }
        });
    }
}

/// Extract an `Arc<dyn UsageEmitter>` from a Python object: either a
/// concrete `NoopUsageEmitter`, or a subclass of `UsageEmitter` (wrapped
/// in a `PyHostUsageEmitter` adapter).
fn extract_emitter(obj: &Bound<'_, PyAny>) -> PyResult<Arc<dyn UsageEmitter>> {
    if obj.extract::<PyRef<'_, PyNoopUsageEmitter>>().is_ok() {
        return Ok(Arc::new(NoopUsageEmitter));
    }
    if obj.is_instance_of::<PyUsageEmitter>() {
        return Ok(Arc::new(PyHostUsageEmitter::new(obj.clone().unbind())));
    }
    Err(PyTypeError::new_err(
        "expected NoopUsageEmitter or UsageEmitter subclass",
    ))
}

// ---------------------------------------------------------------------------
// PyUsageRecordingCompletionModel
// ---------------------------------------------------------------------------

/// A `CompletionModel` decorator that emits a `UsageEvent` after each
/// successful `complete` / `stream` call.
///
/// Mirrors `blazen_llm::usage_recording::UsageRecordingCompletionModel`.
///
/// Example:
///     >>> base = CompletionModel.openai()
///     >>> emitter = NoopUsageEmitter()
///     >>> model = UsageRecordingCompletionModel(base, emitter, "openai")
#[gen_stub_pyclass]
#[pyclass(name = "UsageRecordingCompletionModel")]
pub struct PyUsageRecordingCompletionModel {
    inner: Arc<dyn CompletionModel>,
}

#[gen_stub_pymethods]
#[pymethods]
impl PyUsageRecordingCompletionModel {
    /// Wrap a `CompletionModel` with a usage-recording layer.
    ///
    /// Args:
    ///     model: The `CompletionModel` to wrap.
    ///     emitter: A `UsageEmitter` that receives each emitted event.
    ///     provider_label: A string used as the `provider` field on each
    ///         emitted `UsageEvent` (e.g. `"openai"`).
    ///     run_id: Optional UUID string identifying the workflow run. If
    ///         omitted, a random UUID is generated.
    #[new]
    #[pyo3(signature = (model, emitter, provider_label, run_id=None))]
    fn new(
        model: Bound<'_, PyCompletionModel>,
        emitter: &Bound<'_, PyAny>,
        provider_label: String,
        run_id: Option<String>,
    ) -> PyResult<Self> {
        let inner_model = arc_from_bound(&model);
        let emitter_arc = extract_emitter(emitter)?;
        let run_id_uuid = match run_id {
            Some(s) => Uuid::parse_str(&s)
                .map_err(|e| PyValueError::new_err(format!("invalid run_id UUID: {e}")))?,
            None => Uuid::new_v4(),
        };
        let wrapped = UsageRecordingCompletionModel::from_arc(
            inner_model,
            emitter_arc,
            provider_label,
            run_id_uuid,
        );
        Ok(Self {
            inner: Arc::new(wrapped),
        })
    }

    /// Convert this decorator into a `CompletionModel` so it can be passed to
    /// APIs that expect a `CompletionModel`.
    fn as_model(&self) -> PyCompletionModel {
        PyCompletionModel {
            inner: Some(self.inner.clone()),
            local_model: None,
            config: None,
        }
    }

    /// The model id reported by the underlying provider.
    #[getter]
    fn model_id(&self) -> String {
        self.inner.model_id().to_owned()
    }

    /// Perform a chat completion, emitting a `UsageEvent` on success.
    #[pyo3(signature = (messages, options=None))]
    #[gen_stub(override_return_type(type_repr = "typing.Coroutine[typing.Any, typing.Any, CompletionResponse]", imports = ("typing",)))]
    fn complete<'py>(
        &self,
        py: Python<'py>,
        messages: Vec<PyRef<'py, PyChatMessage>>,
        options: Option<PyRef<'py, PyCompletionOptions>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let inner = self.inner.clone();
        let chat_messages: Vec<ChatMessage> = messages.iter().map(|m| m.inner.clone()).collect();
        let request = build_request(py, chat_messages, options.as_deref())?;
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let response = inner.complete(request).await.map_err(BlazenPyError::from)?;
            Ok(PyCompletionResponse { inner: response })
        })
    }

    fn __repr__(&self) -> String {
        format!(
            "UsageRecordingCompletionModel(model_id='{}')",
            self.inner.model_id()
        )
    }
}

// Type alias for the streaming chunk type that `CompletionModel::stream` yields.
#[allow(dead_code)]
type PinnedChunkStream = Pin<Box<dyn Stream<Item = Result<StreamChunk, BlazenError>> + Send>>;

// ---------------------------------------------------------------------------
// PyUsageRecordingEmbeddingModel
// ---------------------------------------------------------------------------

/// An `EmbeddingModel` decorator that emits a `UsageEvent` after each
/// successful `embed` call.
///
/// Mirrors `blazen_llm::usage_recording::UsageRecordingEmbeddingModel`.
///
/// Example:
///     >>> base = EmbeddingModel.openai()
///     >>> model = UsageRecordingEmbeddingModel(base, NoopUsageEmitter(), "openai")
#[gen_stub_pyclass]
#[pyclass(name = "UsageRecordingEmbeddingModel")]
pub struct PyUsageRecordingEmbeddingModel {
    inner: Arc<dyn EmbeddingModel>,
}

#[gen_stub_pymethods]
#[pymethods]
impl PyUsageRecordingEmbeddingModel {
    /// Wrap an `EmbeddingModel` with a usage-recording layer.
    ///
    /// Args:
    ///     model: The `EmbeddingModel` to wrap.
    ///     emitter: A `UsageEmitter` that receives each emitted event.
    ///     provider_label: A string used as the `provider` field on each
    ///         emitted `UsageEvent`.
    ///     run_id: Optional UUID string identifying the workflow run. If
    ///         omitted, a random UUID is generated.
    #[new]
    #[pyo3(signature = (model, emitter, provider_label, run_id=None))]
    fn new(
        model: Bound<'_, PyEmbeddingModel>,
        emitter: &Bound<'_, PyAny>,
        provider_label: String,
        run_id: Option<String>,
    ) -> PyResult<Self> {
        let inner_model = model
            .borrow()
            .inner
            .as_ref()
            .ok_or_else(|| {
                PyValueError::new_err(
                    "UsageRecordingEmbeddingModel: source EmbeddingModel has no inner provider",
                )
            })?
            .clone();
        let emitter_arc = extract_emitter(emitter)?;
        let run_id_uuid = match run_id {
            Some(s) => Uuid::parse_str(&s)
                .map_err(|e| PyValueError::new_err(format!("invalid run_id UUID: {e}")))?,
            None => Uuid::new_v4(),
        };
        let wrapped = UsageRecordingEmbeddingModel::from_arc(
            inner_model,
            emitter_arc,
            provider_label,
            run_id_uuid,
        );
        Ok(Self {
            inner: Arc::new(wrapped),
        })
    }

    /// The model id reported by the underlying provider.
    #[getter]
    fn model_id(&self) -> String {
        self.inner.model_id().to_owned()
    }

    /// Output dimensionality.
    #[getter]
    fn dimensions(&self) -> usize {
        self.inner.dimensions()
    }

    /// Embed a list of texts, emitting a `UsageEvent` on success.
    #[gen_stub(override_return_type(type_repr = "typing.Coroutine[typing.Any, typing.Any, EmbeddingResponse]", imports = ("typing",)))]
    fn embed<'py>(&self, py: Python<'py>, texts: Vec<String>) -> PyResult<Bound<'py, PyAny>> {
        let inner = self.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let response = inner.embed(&texts).await.map_err(BlazenPyError::from)?;
            Ok(PyEmbeddingResponse { inner: response })
        })
    }

    fn __repr__(&self) -> String {
        format!(
            "UsageRecordingEmbeddingModel(model_id='{}', dimensions={})",
            self.inner.model_id(),
            self.inner.dimensions(),
        )
    }
}
