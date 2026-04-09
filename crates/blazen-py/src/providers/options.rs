//! Typed option wrappers for provider factories.

use pyo3::prelude::*;
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pyclass_enum, gen_stub_pymethods};

use blazen_llm::types::provider_options::{
    AzureOptions, BedrockOptions, FalLlmEndpointKind, FalOptions, ProviderOptions,
};

// ---------------------------------------------------------------------------
// ProviderOptions
// ---------------------------------------------------------------------------

/// Base options shared by every provider.
#[gen_stub_pyclass]
#[pyclass(name = "ProviderOptions", from_py_object)]
#[derive(Clone, Default)]
pub struct PyProviderOptions {
    pub(crate) inner: ProviderOptions,
}

#[gen_stub_pymethods]
#[pymethods]
impl PyProviderOptions {
    /// Create a new ProviderOptions.
    ///
    /// Args:
    ///     api_key: Override the provider's API key.
    ///     model: Override the default model identifier.
    ///     base_url: Override the provider's base URL.
    #[new]
    #[pyo3(signature = (*, api_key=None, model=None, base_url=None))]
    fn new(api_key: Option<String>, model: Option<String>, base_url: Option<String>) -> Self {
        Self {
            inner: ProviderOptions {
                api_key,
                model,
                base_url,
            },
        }
    }

    #[getter]
    fn api_key(&self) -> Option<String> {
        self.inner.api_key.clone()
    }
    #[setter]
    fn set_api_key(&mut self, value: Option<String>) {
        self.inner.api_key = value;
    }

    #[getter]
    fn model(&self) -> Option<String> {
        self.inner.model.clone()
    }
    #[setter]
    fn set_model(&mut self, value: Option<String>) {
        self.inner.model = value;
    }

    #[getter]
    fn base_url(&self) -> Option<String> {
        self.inner.base_url.clone()
    }
    #[setter]
    fn set_base_url(&mut self, value: Option<String>) {
        self.inner.base_url = value;
    }

    fn __repr__(&self) -> String {
        format!(
            "ProviderOptions(model={:?}, base_url={:?})",
            self.inner.model, self.inner.base_url
        )
    }
}

// ---------------------------------------------------------------------------
// FalLlmEndpointKind
// ---------------------------------------------------------------------------

/// Simplified fal.ai endpoint family.
#[gen_stub_pyclass_enum]
#[pyclass(name = "FalLlmEndpointKind", eq, eq_int, from_py_object)]
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum PyFalLlmEndpointKind {
    OpenAiChat,
    OpenAiResponses,
    OpenAiEmbeddings,
    OpenRouter,
    AnyLlm,
}

impl From<PyFalLlmEndpointKind> for FalLlmEndpointKind {
    fn from(kind: PyFalLlmEndpointKind) -> Self {
        match kind {
            PyFalLlmEndpointKind::OpenAiChat => Self::OpenAiChat,
            PyFalLlmEndpointKind::OpenAiResponses => Self::OpenAiResponses,
            PyFalLlmEndpointKind::OpenAiEmbeddings => Self::OpenAiEmbeddings,
            PyFalLlmEndpointKind::OpenRouter => Self::OpenRouter,
            PyFalLlmEndpointKind::AnyLlm => Self::AnyLlm,
        }
    }
}

// ---------------------------------------------------------------------------
// FalOptions
// ---------------------------------------------------------------------------

/// Options specific to the fal.ai provider.
#[gen_stub_pyclass]
#[pyclass(name = "FalOptions", from_py_object)]
#[derive(Clone, Default)]
pub struct PyFalOptions {
    pub(crate) inner: FalOptions,
}

#[gen_stub_pymethods]
#[pymethods]
impl PyFalOptions {
    /// Create a new FalOptions.
    ///
    /// Args:
    ///     api_key: Override the provider's API key.
    ///     model: Override the default model identifier.
    ///     base_url: Override the provider's base URL.
    ///     endpoint: The fal endpoint family (defaults to OpenAiChat).
    ///     enterprise: Promote the endpoint to its enterprise / SOC2-eligible variant.
    ///     auto_route_modality: Auto-route to vision/audio/video variant when content has media.
    #[new]
    #[pyo3(signature = (*, api_key=None, model=None, base_url=None, endpoint=None, enterprise=false, auto_route_modality=true))]
    fn new(
        api_key: Option<String>,
        model: Option<String>,
        base_url: Option<String>,
        endpoint: Option<PyFalLlmEndpointKind>,
        enterprise: bool,
        auto_route_modality: bool,
    ) -> Self {
        Self {
            inner: FalOptions {
                base: ProviderOptions {
                    api_key,
                    model,
                    base_url,
                },
                endpoint: endpoint.map(Into::into),
                enterprise,
                auto_route_modality,
            },
        }
    }

    #[getter]
    fn api_key(&self) -> Option<String> {
        self.inner.base.api_key.clone()
    }
    #[setter]
    fn set_api_key(&mut self, value: Option<String>) {
        self.inner.base.api_key = value;
    }

    #[getter]
    fn model(&self) -> Option<String> {
        self.inner.base.model.clone()
    }
    #[setter]
    fn set_model(&mut self, value: Option<String>) {
        self.inner.base.model = value;
    }

    #[getter]
    fn base_url(&self) -> Option<String> {
        self.inner.base.base_url.clone()
    }
    #[setter]
    fn set_base_url(&mut self, value: Option<String>) {
        self.inner.base.base_url = value;
    }

    #[getter]
    fn enterprise(&self) -> bool {
        self.inner.enterprise
    }
    #[setter]
    fn set_enterprise(&mut self, value: bool) {
        self.inner.enterprise = value;
    }

    #[getter]
    fn auto_route_modality(&self) -> bool {
        self.inner.auto_route_modality
    }
    #[setter]
    fn set_auto_route_modality(&mut self, value: bool) {
        self.inner.auto_route_modality = value;
    }

    fn __repr__(&self) -> String {
        format!(
            "FalOptions(model={:?}, enterprise={})",
            self.inner.base.model, self.inner.enterprise
        )
    }
}

// ---------------------------------------------------------------------------
// AzureOptions
// ---------------------------------------------------------------------------

/// Options specific to Azure OpenAI.
#[gen_stub_pyclass]
#[pyclass(name = "AzureOptions", from_py_object)]
#[derive(Clone)]
pub struct PyAzureOptions {
    pub(crate) inner: AzureOptions,
}

#[gen_stub_pymethods]
#[pymethods]
impl PyAzureOptions {
    /// Create a new AzureOptions.
    ///
    /// Args:
    ///     resource_name: Azure resource name (required).
    ///     deployment_name: Azure deployment name (required).
    ///     api_key: Override the provider's API key.
    ///     model: Override the default model identifier.
    ///     base_url: Override the provider's base URL.
    ///     api_version: API version override (e.g. "2024-02-15-preview").
    #[new]
    #[pyo3(signature = (resource_name, deployment_name, *, api_key=None, model=None, base_url=None, api_version=None))]
    fn new(
        resource_name: String,
        deployment_name: String,
        api_key: Option<String>,
        model: Option<String>,
        base_url: Option<String>,
        api_version: Option<String>,
    ) -> Self {
        Self {
            inner: AzureOptions {
                base: ProviderOptions {
                    api_key,
                    model,
                    base_url,
                },
                resource_name,
                deployment_name,
                api_version,
            },
        }
    }

    #[getter]
    fn api_key(&self) -> Option<String> {
        self.inner.base.api_key.clone()
    }
    #[setter]
    fn set_api_key(&mut self, value: Option<String>) {
        self.inner.base.api_key = value;
    }

    #[getter]
    fn model(&self) -> Option<String> {
        self.inner.base.model.clone()
    }
    #[setter]
    fn set_model(&mut self, value: Option<String>) {
        self.inner.base.model = value;
    }

    #[getter]
    fn base_url(&self) -> Option<String> {
        self.inner.base.base_url.clone()
    }
    #[setter]
    fn set_base_url(&mut self, value: Option<String>) {
        self.inner.base.base_url = value;
    }

    #[getter]
    fn resource_name(&self) -> String {
        self.inner.resource_name.clone()
    }
    #[setter]
    fn set_resource_name(&mut self, value: String) {
        self.inner.resource_name = value;
    }

    #[getter]
    fn deployment_name(&self) -> String {
        self.inner.deployment_name.clone()
    }
    #[setter]
    fn set_deployment_name(&mut self, value: String) {
        self.inner.deployment_name = value;
    }

    #[getter]
    fn api_version(&self) -> Option<String> {
        self.inner.api_version.clone()
    }
    #[setter]
    fn set_api_version(&mut self, value: Option<String>) {
        self.inner.api_version = value;
    }

    fn __repr__(&self) -> String {
        format!(
            "AzureOptions(resource_name={:?}, deployment_name={:?}, api_version={:?})",
            self.inner.resource_name, self.inner.deployment_name, self.inner.api_version
        )
    }
}

// ---------------------------------------------------------------------------
// BedrockOptions
// ---------------------------------------------------------------------------

/// Options specific to AWS Bedrock.
#[gen_stub_pyclass]
#[pyclass(name = "BedrockOptions", from_py_object)]
#[derive(Clone)]
pub struct PyBedrockOptions {
    pub(crate) inner: BedrockOptions,
}

#[gen_stub_pymethods]
#[pymethods]
impl PyBedrockOptions {
    /// Create a new BedrockOptions.
    ///
    /// Args:
    ///     region: AWS region (required, e.g. "us-east-1").
    ///     api_key: Override the provider's API key.
    ///     model: Override the default model identifier.
    ///     base_url: Override the provider's base URL.
    #[new]
    #[pyo3(signature = (region, *, api_key=None, model=None, base_url=None))]
    fn new(
        region: String,
        api_key: Option<String>,
        model: Option<String>,
        base_url: Option<String>,
    ) -> Self {
        Self {
            inner: BedrockOptions {
                base: ProviderOptions {
                    api_key,
                    model,
                    base_url,
                },
                region,
            },
        }
    }

    #[getter]
    fn api_key(&self) -> Option<String> {
        self.inner.base.api_key.clone()
    }
    #[setter]
    fn set_api_key(&mut self, value: Option<String>) {
        self.inner.base.api_key = value;
    }

    #[getter]
    fn model(&self) -> Option<String> {
        self.inner.base.model.clone()
    }
    #[setter]
    fn set_model(&mut self, value: Option<String>) {
        self.inner.base.model = value;
    }

    #[getter]
    fn base_url(&self) -> Option<String> {
        self.inner.base.base_url.clone()
    }
    #[setter]
    fn set_base_url(&mut self, value: Option<String>) {
        self.inner.base.base_url = value;
    }

    #[getter]
    fn region(&self) -> String {
        self.inner.region.clone()
    }
    #[setter]
    fn set_region(&mut self, value: String) {
        self.inner.region = value;
    }

    fn __repr__(&self) -> String {
        format!("BedrockOptions(region={:?})", self.inner.region)
    }
}
