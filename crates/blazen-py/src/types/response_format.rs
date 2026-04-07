//! Python wrapper for `ResponseFormat`.

use pyo3::prelude::*;

use blazen_llm::ResponseFormat;

/// Typed response-format hint for structured output.
///
/// Use ``ResponseFormat.text()`` for free-form text, ``ResponseFormat.json_object()``
/// for any JSON object, or ``ResponseFormat.json_schema(name, schema)`` for
/// JSON-Schema-validated structured output.
///
/// Example:
///     >>> rf = ResponseFormat.json_schema("Person", {"type": "object", ...})
///     >>> options = CompletionOptions(response_format=rf.to_dict())
#[pyclass(name = "ResponseFormat", frozen, from_py_object)]
#[derive(Clone)]
pub struct PyResponseFormat {
    pub(crate) inner: ResponseFormat,
}

#[pymethods]
impl PyResponseFormat {
    /// Discriminator. One of ``"text"``, ``"json_object"``, ``"json_schema"``.
    #[getter]
    fn kind(&self) -> &'static str {
        match &self.inner {
            ResponseFormat::Text => "text",
            ResponseFormat::JsonObject => "json_object",
            ResponseFormat::JsonSchema { .. } => "json_schema",
        }
    }

    /// Schema name for ``json_schema`` variants. ``None`` otherwise.
    #[getter]
    fn schema_name(&self) -> Option<&str> {
        if let ResponseFormat::JsonSchema { name, .. } = &self.inner {
            Some(name.as_str())
        } else {
            None
        }
    }

    /// Schema body for ``json_schema`` variants, returned as a Python dict.
    /// ``None`` otherwise.
    #[getter]
    fn schema(&self, py: Python<'_>) -> PyResult<Option<Py<PyAny>>> {
        if let ResponseFormat::JsonSchema { schema, .. } = &self.inner {
            Ok(Some(crate::convert::json_to_py(py, schema)?))
        } else {
            Ok(None)
        }
    }

    /// Whether the schema is strict (``json_schema`` variants only). Defaults
    /// to ``False`` for non-schema variants.
    #[getter]
    fn strict(&self) -> bool {
        if let ResponseFormat::JsonSchema { strict, .. } = &self.inner {
            *strict
        } else {
            false
        }
    }

    /// Serialize this response format to a JSON-compatible dict suitable for
    /// passing to ``CompletionOptions.response_format``.
    fn to_dict(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let value = serde_json::to_value(&self.inner)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        crate::convert::json_to_py(py, &value)
    }

    // -- Static factories ----------------------------------------------------

    /// Build a free-text response format.
    #[staticmethod]
    fn text() -> Self {
        Self {
            inner: ResponseFormat::Text,
        }
    }

    /// Build a JSON-object response format (any valid JSON object).
    #[staticmethod]
    fn json_object() -> Self {
        Self {
            inner: ResponseFormat::JsonObject,
        }
    }

    /// Build a JSON-Schema response format.
    #[staticmethod]
    #[pyo3(signature = (name, schema, strict=true))]
    fn json_schema(name: String, schema: &Bound<'_, PyAny>, strict: bool) -> PyResult<Self> {
        let schema_value = crate::convert::py_to_json(schema.py(), schema)?;
        Ok(Self {
            inner: ResponseFormat::JsonSchema {
                name,
                schema: schema_value,
                strict,
            },
        })
    }

    fn __repr__(&self) -> String {
        format!("ResponseFormat(kind={:?})", self.kind())
    }
}

impl From<ResponseFormat> for PyResponseFormat {
    fn from(inner: ResponseFormat) -> Self {
        Self { inner }
    }
}

impl From<&ResponseFormat> for PyResponseFormat {
    fn from(inner: &ResponseFormat) -> Self {
        Self {
            inner: inner.clone(),
        }
    }
}
