//! Free-standing helper functions and types for provider parity.
//!
//! Mirrors functions exported by [`blazen_llm`] that are useful from
//! Python: API key resolution, environment variable lookups, context
//! window heuristics, inline-artifact extraction, and pricing
//! registration from a [`ModelInfo`].

use pyo3::prelude::*;
use pyo3::types::PyDict;
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pyfunction, gen_stub_pymethods};

use crate::types::PyArtifact;
use crate::types::pricing::PyModelPricing;
use blazen_llm::traits::{ModelCapabilities, ModelInfo};

// ---------------------------------------------------------------------------
// Inline artifact extraction
// ---------------------------------------------------------------------------

/// Scan a string of LLM-generated text for inline artifacts.
///
/// Detects fenced code blocks (` ```python ... ``` `), SVG runs, mermaid
/// diagrams, and similar inline formats. Returns artifacts in source
/// order.
///
/// Args:
///     content: The assistant content string to scan.
///
/// Returns:
///     A list of [`Artifact`] objects.
#[gen_stub_pyfunction]
#[pyfunction]
pub fn extract_inline_artifacts(content: &str) -> Vec<PyArtifact> {
    blazen_llm::extract_inline_artifacts(content)
        .into_iter()
        .map(|a| PyArtifact { inner: a })
        .collect()
}

// ---------------------------------------------------------------------------
// Provider env var lookup
// ---------------------------------------------------------------------------

/// Return the well-known environment variable name for ``provider``.
///
/// Returns ``None`` for unknown providers.
///
/// Example:
///     >>> env_var_for_provider("openai")
///     'OPENAI_API_KEY'
#[gen_stub_pyfunction]
#[pyfunction]
pub fn env_var_for_provider(provider: &str) -> Option<String> {
    blazen_llm::keys::env_var_for_provider(provider).map(str::to_owned)
}

/// Resolve an API key for ``provider``.
///
/// Resolution order:
/// 1. ``explicit`` if non-empty.
/// 2. The provider's env var (see ``PROVIDER_ENV_VARS``).
/// 3. Raises a Blazen auth error if neither is available.
///
/// Returns the resolved key, or raises an exception on failure.
#[gen_stub_pyfunction]
#[pyfunction]
#[pyo3(signature = (provider, *, explicit=None))]
pub fn resolve_api_key(provider: &str, explicit: Option<String>) -> PyResult<String> {
    blazen_llm::keys::resolve_api_key(provider, explicit)
        .map_err(crate::error::blazen_error_to_pyerr)
}

// ---------------------------------------------------------------------------
// Context window lookup
// ---------------------------------------------------------------------------

/// Best-effort context window size for a model id.
///
/// Falls back to 128 000 when the model string does not match any known
/// pattern.
#[gen_stub_pyfunction]
#[pyfunction]
pub fn get_context_window(model: &str) -> usize {
    blazen_llm::tokens::get_context_window(model)
}

// ---------------------------------------------------------------------------
// Provider HTTP error tail formatting
// ---------------------------------------------------------------------------

/// Format the tail string used in a `BlazenError::ProviderHttp` Display.
///
/// Returns either the structured ``detail`` or a 200-char prefix of
/// ``raw_body``, optionally suffixed with a ``(request-id=...)`` clause.
#[gen_stub_pyfunction]
#[pyfunction]
#[pyo3(signature = (raw_body, *, detail=None, request_id=None))]
pub fn format_provider_http_tail(
    raw_body: &str,
    detail: Option<String>,
    request_id: Option<String>,
) -> String {
    blazen_llm::providers::format_provider_http_tail(
        detail.as_deref(),
        raw_body,
        request_id.as_deref(),
    )
}

// ---------------------------------------------------------------------------
// PyModelInfo
// ---------------------------------------------------------------------------

/// Information about a model offered by a provider.
///
/// Mirrors [`blazen_llm::traits::ModelInfo`]. Used by
/// [`register_from_model_info`] to register pricing without going
/// through the lower-level ``register_pricing`` entry point.
#[gen_stub_pyclass]
#[pyclass(name = "ModelInfo", from_py_object)]
#[derive(Clone)]
pub struct PyModelInfo {
    pub(crate) inner: ModelInfo,
}

#[gen_stub_pymethods]
#[pymethods]
impl PyModelInfo {
    /// Create a new ModelInfo.
    ///
    /// Args:
    ///     id: Model identifier used in API requests.
    ///     provider: Provider name.
    ///     name: Human-readable display name (defaults to ``id``).
    ///     context_length: Maximum context window in tokens.
    ///     pricing: Optional [`ModelPricing`] entry.
    ///     capabilities: Optional capability flags as a dict
    ///         (``{"chat": True, "streaming": True, ...}``).
    #[new]
    #[pyo3(signature = (
        *,
        id,
        provider,
        name=None,
        context_length=None,
        pricing=None,
        capabilities=None,
    ))]
    fn new(
        py: Python<'_>,
        id: String,
        provider: String,
        name: Option<String>,
        context_length: Option<u64>,
        pricing: Option<PyRef<'_, PyModelPricing>>,
        capabilities: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Self> {
        let caps = match capabilities {
            Some(d) => parse_capabilities(py, d)?,
            None => ModelCapabilities::default(),
        };
        Ok(Self {
            inner: ModelInfo {
                id,
                name,
                provider,
                context_length,
                pricing: pricing.map(|p| p.inner.clone()),
                capabilities: caps,
            },
        })
    }

    #[getter]
    fn id(&self) -> &str {
        &self.inner.id
    }

    #[getter]
    fn name(&self) -> Option<String> {
        self.inner.name.clone()
    }

    #[getter]
    fn provider(&self) -> &str {
        &self.inner.provider
    }

    #[getter]
    fn context_length(&self) -> Option<u64> {
        self.inner.context_length
    }

    #[getter]
    fn pricing(&self) -> Option<PyModelPricing> {
        self.inner
            .pricing
            .clone()
            .map(|p| PyModelPricing { inner: p })
    }

    fn __repr__(&self) -> String {
        format!(
            "ModelInfo(id={:?}, provider={:?}, context_length={:?})",
            self.inner.id, self.inner.provider, self.inner.context_length
        )
    }
}

fn parse_capabilities(_py: Python<'_>, dict: &Bound<'_, PyDict>) -> PyResult<ModelCapabilities> {
    let mut caps = ModelCapabilities::default();
    let read = |key: &str| -> PyResult<Option<bool>> {
        match dict.get_item(key)? {
            Some(v) => Ok(Some(v.extract::<bool>()?)),
            None => Ok(None),
        }
    };
    if let Some(v) = read("chat")? {
        caps.chat = v;
    }
    if let Some(v) = read("streaming")? {
        caps.streaming = v;
    }
    if let Some(v) = read("tool_use")? {
        caps.tool_use = v;
    }
    if let Some(v) = read("structured_output")? {
        caps.structured_output = v;
    }
    if let Some(v) = read("vision")? {
        caps.vision = v;
    }
    if let Some(v) = read("image_generation")? {
        caps.image_generation = v;
    }
    if let Some(v) = read("embeddings")? {
        caps.embeddings = v;
    }
    if let Some(v) = read("video_generation")? {
        caps.video_generation = v;
    }
    if let Some(v) = read("text_to_speech")? {
        caps.text_to_speech = v;
    }
    if let Some(v) = read("speech_to_text")? {
        caps.speech_to_text = v;
    }
    if let Some(v) = read("audio_generation")? {
        caps.audio_generation = v;
    }
    if let Some(v) = read("three_d_generation")? {
        caps.three_d_generation = v;
    }
    Ok(caps)
}

// ---------------------------------------------------------------------------
// register_from_model_info
// ---------------------------------------------------------------------------

/// Register pricing in the global pricing registry from a [`ModelInfo`].
///
/// No-op when ``info.pricing`` is ``None`` or contains neither input
/// nor output cost.
#[gen_stub_pyfunction]
#[pyfunction]
pub fn register_from_model_info(info: &PyModelInfo) {
    blazen_llm::pricing::register_from_model_info(&info.inner);
}

// ---------------------------------------------------------------------------
// PROVIDER_ENV_VARS helper
// ---------------------------------------------------------------------------

/// Build the `PROVIDER_ENV_VARS` constant as a list of ``(provider, env_var)``
/// tuples, ready to attach to the module via ``m.add(...)``.
#[must_use]
pub fn provider_env_vars() -> Vec<(String, String)> {
    blazen_llm::keys::PROVIDER_ENV_VARS
        .iter()
        .map(|(p, v)| ((*p).to_owned(), (*v).to_owned()))
        .collect()
}
