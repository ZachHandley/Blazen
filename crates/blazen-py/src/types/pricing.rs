//! Python bindings for model pricing and the pricing registry.

use pyo3::prelude::*;
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pyfunction, gen_stub_pymethods};

use blazen_llm::traits::ModelPricing;

/// Pricing information for a model.
///
/// Example:
///     >>> pricing = ModelPricing(input_per_million=1.0, output_per_million=2.0)
///     >>> pricing.input_per_million
///     1.0
#[gen_stub_pyclass]
#[pyclass(name = "ModelPricing", from_py_object)]
#[derive(Clone)]
pub struct PyModelPricing {
    pub(crate) inner: ModelPricing,
}

#[gen_stub_pymethods]
#[pymethods]
impl PyModelPricing {
    #[new]
    #[pyo3(signature = (*, input_per_million=None, output_per_million=None, per_image=None, per_second=None))]
    fn new(
        input_per_million: Option<f64>,
        output_per_million: Option<f64>,
        per_image: Option<f64>,
        per_second: Option<f64>,
    ) -> Self {
        Self {
            inner: ModelPricing {
                input_per_million,
                output_per_million,
                per_image,
                per_second,
            },
        }
    }

    /// Cost per million input tokens in USD.
    #[getter]
    fn input_per_million(&self) -> Option<f64> {
        self.inner.input_per_million
    }

    /// Cost per million output tokens in USD.
    #[getter]
    fn output_per_million(&self) -> Option<f64> {
        self.inner.output_per_million
    }

    /// Cost per image in USD.
    #[getter]
    fn per_image(&self) -> Option<f64> {
        self.inner.per_image
    }

    /// Cost per second of compute in USD.
    #[getter]
    fn per_second(&self) -> Option<f64> {
        self.inner.per_second
    }

    fn __repr__(&self) -> String {
        format!(
            "ModelPricing(input_per_million={:?}, output_per_million={:?}, per_image={:?}, per_second={:?})",
            self.inner.input_per_million,
            self.inner.output_per_million,
            self.inner.per_image,
            self.inner.per_second,
        )
    }
}

/// Register pricing for a model ID.
///
/// Args:
///     model_id: The model identifier (e.g. "my-custom-model").
///     pricing: A ``ModelPricing`` object with at least ``input_per_million``
///         and ``output_per_million`` set.
///
/// Example:
///     >>> register_pricing("my-model", ModelPricing(input_per_million=1.0, output_per_million=2.0))
#[gen_stub_pyfunction]
#[pyfunction]
pub fn register_pricing(model_id: &str, pricing: &PyModelPricing) {
    if let (Some(input), Some(output)) = (
        pricing.inner.input_per_million,
        pricing.inner.output_per_million,
    ) {
        blazen_llm::register_pricing(
            model_id,
            blazen_llm::PricingEntry {
                input_per_million: input,
                output_per_million: output,
            },
        );
    }
}

/// Look up pricing for a model ID.
///
/// Returns ``None`` if the model is unknown.
///
/// Args:
///     model_id: The model identifier.
///
/// Returns:
///     Optional ``ModelPricing`` with input and output costs.
///
/// Example:
///     >>> pricing = lookup_pricing("gpt-4o")
///     >>> if pricing:
///     ...     print(pricing.input_per_million)
#[gen_stub_pyfunction]
#[pyfunction]
#[must_use]
pub fn lookup_pricing(model_id: &str) -> Option<PyModelPricing> {
    blazen_llm::lookup_pricing(model_id).map(|e| PyModelPricing {
        inner: ModelPricing {
            input_per_million: Some(e.input_per_million),
            output_per_million: Some(e.output_per_million),
            per_image: None,
            per_second: None,
        },
    })
}
