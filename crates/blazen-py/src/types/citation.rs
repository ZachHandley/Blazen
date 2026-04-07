//! Python wrapper for `Citation`.

use pyo3::prelude::*;

use blazen_llm::Citation;

/// A web/document citation backing a model statement.
///
/// Populated by Perplexity (`citations` array), Gemini (`groundingMetadata`),
/// and any provider that returns retrieval-augmented citations.
///
/// Example:
///     >>> for cite in response.citations:
///     ...     print(cite.url, cite.title)
#[pyclass(name = "Citation", frozen, from_py_object)]
#[derive(Clone)]
pub struct PyCitation {
    pub(crate) inner: Citation,
}

#[pymethods]
impl PyCitation {
    /// The cited URL.
    #[getter]
    fn url(&self) -> &str {
        &self.inner.url
    }

    /// Document or web-page title, if available.
    #[getter]
    fn title(&self) -> Option<&str> {
        self.inner.title.as_deref()
    }

    /// Excerpt/snippet from the source.
    #[getter]
    fn snippet(&self) -> Option<&str> {
        self.inner.snippet.as_deref()
    }

    /// Byte offset in the response text where this citation starts.
    #[getter]
    fn start(&self) -> Option<usize> {
        self.inner.start
    }

    /// Byte offset in the response text where this citation ends.
    #[getter]
    fn end(&self) -> Option<usize> {
        self.inner.end
    }

    /// Optional document id (for retrieval-augmented citations).
    #[getter]
    fn document_id(&self) -> Option<&str> {
        self.inner.document_id.as_deref()
    }

    /// Provider-specific extra fields, preserved as a Python dict.
    #[getter]
    fn metadata(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        crate::convert::json_to_py(py, &self.inner.metadata)
    }

    fn __repr__(&self) -> String {
        format!(
            "Citation(url={:?}, title={:?}, snippet={:?})",
            self.inner.url, self.inner.title, self.inner.snippet
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
