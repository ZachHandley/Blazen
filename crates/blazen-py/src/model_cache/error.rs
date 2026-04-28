//! Python exception type for model-cache errors.

use pyo3::prelude::*;

use crate::error::BlazenException;
use blazen_model_cache::CacheError;

pyo3::create_exception!(blazen, CacheException, BlazenException);

/// Register the cache exception type on the module.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("CacheError", m.py().get_type::<CacheException>())?;
    Ok(())
}

/// Convert a [`blazen_model_cache::CacheError`] to a [`PyErr`].
pub fn cache_err(err: CacheError) -> PyErr {
    CacheException::new_err(err.to_string())
}
