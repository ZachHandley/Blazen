//! Python exception type for persistence errors.

use pyo3::prelude::*;

use crate::error::BlazenException;

pyo3::create_exception!(blazen, PersistException, BlazenException);

/// Register the persistence exception type on the module.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("PersistError", m.py().get_type::<PersistException>())?;
    Ok(())
}

/// Convert a [`blazen_persist::PersistError`] to a [`PyErr`].
pub fn persist_err(err: blazen_persist::PersistError) -> PyErr {
    PersistException::new_err(err.to_string())
}
