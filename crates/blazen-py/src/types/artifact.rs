//! Python wrapper for `Artifact` (typed inline content extracted from LLM output).

use pyo3::prelude::*;
use pyo3::types::PyDict;
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pymethods};

use blazen_llm::Artifact;

/// A typed artifact extracted from or returned by a model.
///
/// SVG / code blocks / markdown / mermaid / html / latex / json / custom payloads
/// can be returned inline as text by an LLM. The `Artifact` surface lifts them
/// into typed values that callers can dispatch on without re-parsing the
/// assistant content string.
///
/// Use the ``kind`` getter to discriminate, then read the variant-specific
/// fields. Fields not relevant to the variant return ``None``.
///
/// Example:
///     >>> for art in response.artifacts:
///     ...     if art.kind == "svg":
///     ...         render_svg(art.content)
///     ...     elif art.kind == "code_block":
///     ...         print(f"{art.language}: {art.content}")
#[gen_stub_pyclass]
#[pyclass(name = "Artifact", frozen, from_py_object)]
#[derive(Clone)]
pub struct PyArtifact {
    pub(crate) inner: Artifact,
}

#[gen_stub_pymethods]
#[pymethods]
impl PyArtifact {
    /// Discriminator string. One of: ``"svg"``, ``"code_block"``, ``"markdown"``,
    /// ``"mermaid"``, ``"html"``, ``"latex"``, ``"json"``, ``"custom"``.
    #[getter]
    fn kind(&self) -> &'static str {
        match &self.inner {
            Artifact::Svg { .. } => "svg",
            Artifact::CodeBlock { .. } => "code_block",
            Artifact::Markdown { .. } => "markdown",
            Artifact::Mermaid { .. } => "mermaid",
            Artifact::Html { .. } => "html",
            Artifact::Latex { .. } => "latex",
            Artifact::Json { .. } => "json",
            Artifact::Custom { .. } => "custom",
        }
    }

    /// The artifact's primary payload.
    ///
    /// For ``svg``, ``code_block``, ``markdown``, ``mermaid``, ``html``, ``latex``,
    /// and ``custom`` this is a string. For ``json`` it is a parsed Python value
    /// (dict / list / scalar).
    #[getter]
    fn content(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        match &self.inner {
            Artifact::Svg { content, .. }
            | Artifact::CodeBlock { content, .. }
            | Artifact::Markdown { content }
            | Artifact::Mermaid { content }
            | Artifact::Html { content }
            | Artifact::Latex { content }
            | Artifact::Custom { content, .. } => {
                Ok(content.clone().into_pyobject(py)?.into_any().unbind())
            }
            Artifact::Json { content } => crate::convert::json_to_py(py, content),
        }
    }

    /// Optional title for ``svg`` artifacts.
    #[getter]
    fn title(&self) -> Option<&str> {
        if let Artifact::Svg { title, .. } = &self.inner {
            title.as_deref()
        } else {
            None
        }
    }

    /// Language hint for ``code_block`` artifacts.
    #[getter]
    fn language(&self) -> Option<&str> {
        if let Artifact::CodeBlock { language, .. } = &self.inner {
            language.as_deref()
        } else {
            None
        }
    }

    /// Filename hint for ``code_block`` artifacts.
    #[getter]
    fn filename(&self) -> Option<&str> {
        if let Artifact::CodeBlock { filename, .. } = &self.inner {
            filename.as_deref()
        } else {
            None
        }
    }

    /// Provider-specific metadata for ``custom`` artifacts. Empty for other variants.
    #[getter]
    fn metadata(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        if let Artifact::Custom { metadata, .. } = &self.inner {
            crate::convert::json_to_py(py, metadata)
        } else {
            Ok(PyDict::new(py).into_any().unbind())
        }
    }

    /// Inner ``kind`` tag for ``custom`` artifacts. Distinct from the top-level
    /// ``kind`` getter (which returns ``"custom"`` for this variant).
    #[getter]
    fn custom_kind(&self) -> Option<&str> {
        if let Artifact::Custom { kind, .. } = &self.inner {
            Some(kind.as_str())
        } else {
            None
        }
    }

    // -- Static factories ----------------------------------------------------

    /// Build an SVG artifact.
    #[staticmethod]
    #[pyo3(signature = (content, title=None))]
    fn svg(content: String, title: Option<String>) -> Self {
        Self {
            inner: Artifact::Svg { content, title },
        }
    }

    /// Build a code block artifact.
    #[staticmethod]
    #[pyo3(signature = (content, language=None, filename=None))]
    fn code_block(content: String, language: Option<String>, filename: Option<String>) -> Self {
        Self {
            inner: Artifact::CodeBlock {
                language,
                content,
                filename,
            },
        }
    }

    /// Build a markdown artifact.
    #[staticmethod]
    fn markdown(content: String) -> Self {
        Self {
            inner: Artifact::Markdown { content },
        }
    }

    /// Build a mermaid diagram artifact.
    #[staticmethod]
    fn mermaid(content: String) -> Self {
        Self {
            inner: Artifact::Mermaid { content },
        }
    }

    /// Build an HTML artifact.
    #[staticmethod]
    fn html(content: String) -> Self {
        Self {
            inner: Artifact::Html { content },
        }
    }

    /// Build a LaTeX artifact.
    #[staticmethod]
    fn latex(content: String) -> Self {
        Self {
            inner: Artifact::Latex { content },
        }
    }

    /// Build a JSON artifact from a Python dict/list/value.
    #[staticmethod]
    fn json(content: &Bound<'_, PyAny>) -> PyResult<Self> {
        let py = content.py();
        let value = crate::convert::py_to_json(py, content)?;
        Ok(Self {
            inner: Artifact::Json { content: value },
        })
    }

    /// Build a custom artifact with an inner kind tag.
    #[staticmethod]
    #[pyo3(signature = (kind, content, metadata=None))]
    fn custom(
        kind: String,
        content: String,
        metadata: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        let metadata_value = match metadata {
            Some(m) => crate::convert::py_to_json(m.py(), m)?,
            None => serde_json::Value::Object(serde_json::Map::new()),
        };
        Ok(Self {
            inner: Artifact::Custom {
                kind,
                content,
                metadata: metadata_value,
            },
        })
    }

    fn __repr__(&self) -> String {
        format!("Artifact(kind={:?})", self.kind())
    }
}

impl From<Artifact> for PyArtifact {
    fn from(inner: Artifact) -> Self {
        Self { inner }
    }
}

impl From<&Artifact> for PyArtifact {
    fn from(inner: &Artifact) -> Self {
        Self {
            inner: inner.clone(),
        }
    }
}
