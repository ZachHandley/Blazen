//! Stub generator for the Blazen Python bindings.
//!
//! Collects type metadata registered by `#[gen_stub_pyclass]` /
//! `#[gen_stub_pymethods]` annotations across the entire `blazen-py` crate
//! and writes `blazen.pyi` next to `pyproject.toml`.
//!
//! Run with:
//!
//! ```sh
//! cargo run --example stub_gen -p blazen-py
//! ```
//!
//! This is an `example` rather than a `bin` so that
//! `cargo build --workspace --all-features` (which enables
//! `extension-module`) does NOT try to link it against libpython.
//! Examples are only built when explicitly requested.

fn main() {
    let stub = blazen::stub_info().expect("failed to gather stub info");
    stub.generate().expect("failed to generate blazen.pyi");
}
