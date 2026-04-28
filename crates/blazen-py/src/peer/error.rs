//! Python exception type for peer errors.

use pyo3::prelude::*;

use crate::error::BlazenException;

pyo3::create_exception!(blazen, PeerException, BlazenException);

/// Register the peer exception type on the module.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("PeerError", m.py().get_type::<PeerException>())?;
    Ok(())
}

/// Convert a [`blazen_peer::PeerError`] to a [`PyErr`].
pub fn peer_err(err: blazen_peer::PeerError) -> PyErr {
    PeerException::new_err(err.to_string())
}
