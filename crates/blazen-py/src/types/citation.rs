//! Python wrapper for `Citation` (web/document citation backing model output).

use pyo3::prelude::*;
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pymethods};

use blazen_llm::types::Citation;

/// A web/document citation backing a model statement (Perplexity, Gemini
/// grounding, Anthropic web search, etc.).
///
/// Example:
///     >>> for c in response.citations:
///     ...     print(c.url, c.title, c.snippet)
#[gen_stub_pyclass]
#[pyclass(name = "Citation", frozen, from_py_object)]
#[derive(Clone)]
pub struct PyCitation {
    pub(crate) inner: Citation,
}

#[gen_stub_pymethods]
#[pymethods]
impl PyCitation {
    /// Construct a citation explicitly. Most users get these from
    /// `CompletionResponse.citations` rather than building them.
    #[new]
    #[pyo3(signature = (
        *,
        url,
        title=None,
        snippet=None,
        start=None,
        end=None,
        document_id=None,
        metadata=None,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        url: String,
        title: Option<String>,
        snippet: Option<String>,
        start: Option<usize>,
        end: Option<usize>,
        document_id: Option<String>,
        metadata: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        let metadata_value = match metadata {
            Some(m) => crate::convert::py_to_json(m.py(), m)?,
            None => serde_json::Value::Null,
        };
        Ok(Self {
            inner: Citation {
                url,
                title,
                snippet,
                start,
                end,
                document_id,
                metadata: metadata_value,
            },
        })
    }

    /// The cited URL.
    #[getter]
    fn url(&self) -> &str {
        &self.inner.url
    }

    /// Optional title of the cited document.
    #[getter]
    fn title(&self) -> Option<&str> {
        self.inner.title.as_deref()
    }

    /// Optional excerpt from the cited document.
    #[getter]
    fn snippet(&self) -> Option<&str> {
        self.inner.snippet.as_deref()
    }

    /// Byte offset in the response text where this citation begins.
    #[getter]
    fn start(&self) -> Option<usize> {
        self.inner.start
    }

    /// Byte offset in the response text where this citation ends.
    #[getter]
    fn end(&self) -> Option<usize> {
        self.inner.end
    }

    /// Optional document identifier for retrieval-augmented citations.
    #[getter]
    fn document_id(&self) -> Option<&str> {
        self.inner.document_id.as_deref()
    }

    /// Provider-specific extra fields, preserved as a dict.
    #[getter]
    #[gen_stub(override_return_type(type_repr = "dict[str, typing.Any]", imports = ("typing",)))]
    fn metadata(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        crate::convert::json_to_py(py, &self.inner.metadata)
    }

    fn __repr__(&self) -> String {
        format!(
            "Citation(url={:?}, title={:?})",
            self.inner.url, self.inner.title
        )
    }
}

impl From<Citation> for PyCitation {
    fn from(inner: Citation) -> Self {
        Self { inner }
    }
}

impl From<&Citation> for PyCitation {
    fn from(inner: &Citation) -> Self {
        Self {
            inner: inner.clone(),
        }
    }
}
