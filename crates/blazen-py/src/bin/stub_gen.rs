//! Stub generator binary for blazen Python bindings.
//!
//! Collects type metadata registered by `#[gen_stub_pyclass]` /
//! `#[gen_stub_pymethods]` annotations and writes `blazen.pyi`.
//!
//! Wired up in Phase 4 after annotations are added to all pyclass types.

fn main() {
    // TODO: wire up pyo3_stub_gen::StubInfo once annotations are added
    eprintln!("stub_gen: not yet wired up — annotations needed first");
}
