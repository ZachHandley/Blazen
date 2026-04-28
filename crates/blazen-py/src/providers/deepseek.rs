//! Python wrapper for the DeepSeek provider.

use std::sync::Arc;

use pyo3::prelude::*;
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pymethods};
use tokio::sync::Mutex;

use crate::error::blazen_error_to_pyerr;
use crate::providers::completion_model::{
    LazyStreamState, PendingStream, PyCompletionOptions, PyLazyCompletionStream, build_request,
};
use crate::providers::options::PyProviderOptions;
use crate::types::{PyChatMessage, PyCompletionResponse};
use blazen_llm::ChatMessage;
use blazen_llm::providers::deepseek::DeepSeekProvider;
use blazen_llm::traits::CompletionModel;

/// A DeepSeek provider.
///
/// Standalone class form of `CompletionModel.deepseek(...)`.
#[gen_stub_pyclass]
#[pyclass(name = "DeepSeekProvider", from_py_object)]
#[derive(Clone)]
pub struct PyDeepSeekProvider {
    inner: Arc<DeepSeekProvider>,
}

#[gen_stub_pymethods]
#[pymethods]
impl PyDeepSeekProvider {
    /// Create a new DeepSeek provider.
    #[new]
    #[pyo3(signature = (*, options=None))]
    fn new(options: Option<PyRef<'_, PyProviderOptions>>) -> PyResult<Self> {
        let opts = options.map(|o| o.inner.clone()).unwrap_or_default();
        Ok(Self {
            inner: Arc::new(DeepSeekProvider::from_options(opts).map_err(blazen_error_to_pyerr)?),
        })
    }

    #[getter]
    fn model_id(&self) -> &str {
        CompletionModel::model_id(self.inner.as_ref())
    }

    #[gen_stub(override_return_type(type_repr = "typing.Coroutine[typing.Any, typing.Any, CompletionResponse]", imports = ("typing",)))]
    #[pyo3(signature = (messages, options=None))]
    fn complete<'py>(
        &self,
        py: Python<'py>,
        messages: Vec<PyRef<'py, PyChatMessage>>,
        options: Option<PyRef<'py, PyCompletionOptions>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let rust_messages: Vec<ChatMessage> = messages.iter().map(|m| m.inner.clone()).collect();
        let request = build_request(py, rust_messages, options.as_deref())?;
        let inner = self.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let response = CompletionModel::complete(inner.as_ref(), request)
                .await
                .map_err(blazen_error_to_pyerr)?;
            Ok(PyCompletionResponse { inner: response })
        })
    }

    #[pyo3(signature = (messages, options=None))]
    fn stream<'py>(
        &self,
        py: Python<'py>,
        messages: Vec<PyRef<'py, PyChatMessage>>,
        options: Option<PyRef<'py, PyCompletionOptions>>,
    ) -> PyResult<Bound<'py, PyLazyCompletionStream>> {
        let rust_messages: Vec<ChatMessage> = messages.iter().map(|m| m.inner.clone()).collect();
        let request = build_request(py, rust_messages, options.as_deref())?;
        let inner: Arc<dyn CompletionModel> = self.inner.clone();
        let stream = PyLazyCompletionStream {
            state: Arc::new(Mutex::new(LazyStreamState::NotStarted(Box::new(
                PendingStream {
                    model: inner,
                    request: Some(request),
                },
            )))),
        };
        Bound::new(py, stream)
    }

    fn __repr__(&self) -> String {
        format!(
            "DeepSeekProvider(model_id='{}')",
            CompletionModel::model_id(self.inner.as_ref())
        )
    }
}
