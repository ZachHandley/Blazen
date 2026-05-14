//! Python wrapper for [`blazen_llm::ApiProtocol`].
//!
//! Selects how a [`CustomProvider`] talks to its backend for completion
//! calls. Two variants today:
//!
//! - ``ApiProtocol.openai(config)`` -- the framework speaks the OpenAI
//!   Chat Completions wire format against the supplied
//!   [`OpenAiCompatConfig`].
//! - ``ApiProtocol.custom()`` -- the framework dispatches every
//!   completion method through the host language (Phase C wires this).
//!
//! Phase A exposes the constructor surface; the value is plumbed
//! through the bindings without yet being consumed by a Python-side
//! `CustomProvider` constructor (that arrives in Phase B).

use pyo3::prelude::*;
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pymethods};

use crate::providers::openai_compat::PyOpenAiCompatConfig;

/// Internal discriminator for [`PyApiProtocol`]. Kept private --- callers
/// inspect [`PyApiProtocol::kind`] and [`PyApiProtocol::config`] from
/// Python rather than matching on the enum directly.
#[derive(Clone)]
pub(crate) enum ApiProtocolKind {
    /// OpenAI Chat Completions wire format with the embedded config.
    OpenAi(PyOpenAiCompatConfig),
    /// User-defined protocol -- framework dispatches through the host.
    Custom,
}

/// Selects how a [`CustomProvider`] talks to its backend for completion
/// calls.
///
/// Construct via the classmethod factories:
///
/// ```text
/// >>> from blazen import ApiProtocol, OpenAiCompatConfig
/// >>> cfg = OpenAiCompatConfig(
/// ...     provider_name="ollama",
/// ...     base_url="http://localhost:11434/v1",
/// ...     api_key="",
/// ...     default_model="llama3.1",
/// ... )
/// >>> proto = ApiProtocol.openai(cfg)
/// >>> proto.kind
/// 'openai'
/// >>> dispatch_proto = ApiProtocol.custom()
/// >>> dispatch_proto.kind
/// 'custom'
/// ```
#[gen_stub_pyclass]
#[pyclass(name = "ApiProtocol", frozen, from_py_object)]
#[derive(Clone)]
pub struct PyApiProtocol {
    pub(crate) kind: ApiProtocolKind,
}

#[gen_stub_pymethods]
#[pymethods]
impl PyApiProtocol {
    /// Build an OpenAI-protocol selector wrapping the supplied config.
    #[classmethod]
    #[pyo3(signature = (config))]
    fn openai(_cls: &Bound<'_, pyo3::types::PyType>, config: PyOpenAiCompatConfig) -> Self {
        Self {
            kind: ApiProtocolKind::OpenAi(config),
        }
    }

    /// Build a host-dispatch (user-defined) protocol selector.
    #[classmethod]
    fn custom(_cls: &Bound<'_, pyo3::types::PyType>) -> Self {
        Self {
            kind: ApiProtocolKind::Custom,
        }
    }

    /// String discriminator: ``"openai"`` or ``"custom"``.
    #[getter]
    fn kind(&self) -> &'static str {
        match &self.kind {
            ApiProtocolKind::OpenAi(_) => "openai",
            ApiProtocolKind::Custom => "custom",
        }
    }

    /// The embedded [`OpenAiCompatConfig`] for the ``openai`` variant,
    /// or ``None`` for the ``custom`` variant.
    #[getter]
    fn config(&self) -> Option<PyOpenAiCompatConfig> {
        match &self.kind {
            ApiProtocolKind::OpenAi(cfg) => Some(cfg.clone()),
            ApiProtocolKind::Custom => None,
        }
    }

    fn __repr__(&self) -> String {
        match &self.kind {
            ApiProtocolKind::OpenAi(cfg) => format!(
                "ApiProtocol.openai(provider_name={:?}, base_url={:?})",
                cfg.inner.provider_name, cfg.inner.base_url
            ),
            ApiProtocolKind::Custom => "ApiProtocol.custom()".to_owned(),
        }
    }
}
