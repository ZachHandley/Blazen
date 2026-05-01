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
    #[pyo3(signature = (*, input_per_million, output_per_million, per_image=None, per_second=None))]
    fn new(
        input_per_million: f64,
        output_per_million: f64,
        per_image: Option<f64>,
        per_second: Option<f64>,
    ) -> Self {
        Self {
            inner: PricingEntry {
                input_per_million,
                output_per_million,
                per_image,
                per_second,
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

    /// USD per image (for image-generation / vision-input pricing).
    #[getter]
    fn per_image(&self) -> Option<f64> {
        self.inner.per_image
    }

    /// USD per second of compute (for time-billed providers).
    #[getter]
    fn per_second(&self) -> Option<f64> {
        self.inner.per_second
    }

    fn __repr__(&self) -> String {
        format!(
            "PricingEntry(input_per_million={}, output_per_million={}, per_image={:?}, per_second={:?})",
            self.inner.input_per_million,
            self.inner.output_per_million,
            self.inner.per_image,
            self.inner.per_second,
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
