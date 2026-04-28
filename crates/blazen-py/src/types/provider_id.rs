//! Python wrapper for [`blazen_llm::types::ProviderId`].

use pyo3::prelude::*;
use pyo3_stub_gen::derive::{gen_stub_pyclass_enum, gen_stub_pymethods};

use blazen_llm::types::ProviderId;

/// Identifies the underlying provider for `LlmPayload.provider_raw(...)`
/// targeting and other provider-specific routing decisions.
///
/// Mirrors [`blazen_llm::types::ProviderId`].
#[gen_stub_pyclass_enum]
#[pyclass(name = "ProviderId", eq, eq_int, from_py_object)]
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum PyProviderId {
    OpenAi,
    OpenAiCompat,
    Azure,
    Anthropic,
    Gemini,
    Responses,
    Fal,
}

#[gen_stub_pymethods]
#[pymethods]
impl PyProviderId {
    /// The canonical wire name (lowercase).
    #[getter]
    #[allow(clippy::trivially_copy_pass_by_ref)]
    fn name_str(&self) -> &'static str {
        match self {
            Self::OpenAi => "openai",
            Self::OpenAiCompat => "openai_compat",
            Self::Azure => "azure",
            Self::Anthropic => "anthropic",
            Self::Gemini => "gemini",
            Self::Responses => "responses",
            Self::Fal => "fal",
        }
    }

    /// Parse a provider id from its canonical wire name.
    #[staticmethod]
    fn from_str(value: &str) -> PyResult<Self> {
        match value {
            "openai" => Ok(Self::OpenAi),
            "openai_compat" => Ok(Self::OpenAiCompat),
            "azure" => Ok(Self::Azure),
            "anthropic" => Ok(Self::Anthropic),
            "gemini" => Ok(Self::Gemini),
            "responses" => Ok(Self::Responses),
            "fal" => Ok(Self::Fal),
            other => Err(crate::error::BlazenPyError::InvalidArgument(format!(
                "unknown provider id: '{other}'"
            ))
            .into()),
        }
    }

    #[allow(clippy::trivially_copy_pass_by_ref)]
    fn __repr__(&self) -> String {
        format!("ProviderId.{}", self.name_str())
    }
}

impl From<PyProviderId> for ProviderId {
    fn from(p: PyProviderId) -> Self {
        match p {
            PyProviderId::OpenAi => Self::OpenAi,
            PyProviderId::OpenAiCompat => Self::OpenAiCompat,
            PyProviderId::Azure => Self::Azure,
            PyProviderId::Anthropic => Self::Anthropic,
            PyProviderId::Gemini => Self::Gemini,
            PyProviderId::Responses => Self::Responses,
            PyProviderId::Fal => Self::Fal,
        }
    }
}

impl From<ProviderId> for PyProviderId {
    fn from(p: ProviderId) -> Self {
        match p {
            ProviderId::OpenAi => Self::OpenAi,
            ProviderId::OpenAiCompat => Self::OpenAiCompat,
            ProviderId::Azure => Self::Azure,
            ProviderId::Anthropic => Self::Anthropic,
            ProviderId::Gemini => Self::Gemini,
            ProviderId::Responses => Self::Responses,
            ProviderId::Fal => Self::Fal,
        }
    }
}
