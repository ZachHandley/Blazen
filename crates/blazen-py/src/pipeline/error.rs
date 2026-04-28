//! Python exception type for pipeline errors.

use pyo3::prelude::*;

use crate::error::BlazenException;

pyo3::create_exception!(blazen, PipelineException, BlazenException);

/// Register the pipeline exception type on the module.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("PipelineError", m.py().get_type::<PipelineException>())?;
    Ok(())
}

/// Convert a [`blazen_pipeline::PipelineError`] to a [`PyErr`].
pub fn pipeline_err(err: blazen_pipeline::PipelineError) -> PyErr {
    PipelineException::new_err(err.to_string())
}
