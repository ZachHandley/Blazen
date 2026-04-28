//! Python wrapper for [`blazen_llm::PricingEntry`].

use pyo3::prelude::*;
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pymethods};

use blazen_llm::PricingEntry;

/// A pricing entry for the global pricing registry.
///
/// Mirrors [`blazen_llm::PricingEntry`]. Distinct from [`ModelPricing`] which
/// is the richer per-model record (input/output/per_image/per_second);
/// `PricingEntry` is the registry's canonical input/output-only shape used by
/// `register_pricing(...)` and `lookup_pricing(...)`.
#[gen_stub_pyclass]
#[pyclass(name = "PricingEntry", frozen, from_py_object)]
#[derive(Clone, Copy)]
pub struct PyPricingEntry {
    pub(crate) inner: PricingEntry,
}

#[gen_stub_pymethods]
#[pymethods]
impl PyPricingEntry {
    /// Construct a pricing entry.
    #[new]
    #[pyo3(signature = (*, input_per_million, output_per_million))]
    fn new(input_per_million: f64, output_per_million: f64) -> Self {
        Self {
            inner: PricingEntry {
                input_per_million,
                output_per_million,
            },
        }
    }

    /// USD per million input (prompt) tokens.
    #[getter]
    fn input_per_million(&self) -> f64 {
        self.inner.input_per_million
    }

    /// USD per million output (completion) tokens.
    #[getter]
    fn output_per_million(&self) -> f64 {
        self.inner.output_per_million
    }

    fn __repr__(&self) -> String {
        format!(
            "PricingEntry(input_per_million={}, output_per_million={})",
            self.inner.input_per_million, self.inner.output_per_million
        )
    }
}

impl From<PricingEntry> for PyPricingEntry {
    fn from(inner: PricingEntry) -> Self {
        Self { inner }
    }
}

impl From<PyPricingEntry> for PricingEntry {
    fn from(p: PyPricingEntry) -> Self {
        p.inner
    }
}
