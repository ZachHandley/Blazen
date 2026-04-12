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

use std::fs;

fn main() {
    let stub = blazen::stub_info().expect("failed to gather stub info");
    stub.generate().expect("failed to generate blazen.pyi");

    // Post-process: convert Coroutine return types to async def syntax.
    //
    // pyo3-stub-gen 0.22.0 only marks methods as `async` when the Rust
    // function uses the `async fn` keyword. Our `future_into_py` methods
    // are synchronous Rust functions, so we annotate them with
    // `#[gen_stub(override_return_type(type_repr = "typing.Coroutine[...]"))]`
    // and then rewrite the stubs here for readability:
    //
    //   def foo(...) -> typing.Coroutine[typing.Any, typing.Any, T]:
    //     →  async def foo(...) -> T:
    let pyi_path = stub
        .pyproject_dir
        .expect("pyproject_dir not set")
        .join("blazen.pyi");
    let content = fs::read_to_string(&pyi_path).expect("failed to read blazen.pyi");
    let processed = rewrite_coroutine_to_async(&content);
    fs::write(&pyi_path, processed).expect("failed to write blazen.pyi");
}

/// Rewrite `def method(...) -> typing.Coroutine[typing.Any, typing.Any, T]:`
/// into `async def method(...) -> T:`.
fn rewrite_coroutine_to_async(content: &str) -> String {
    let mut result = String::with_capacity(content.len());
    for line in content.lines() {
        if let Some(rewritten) = try_rewrite_line(line) {
            result.push_str(&rewritten);
        } else {
            result.push_str(line);
        }
        result.push('\n');
    }
    result
}

fn try_rewrite_line(line: &str) -> Option<String> {
    // Match lines like:
    //   "    def foo(...) -> typing.Coroutine[typing.Any, typing.Any, SomeType]:"
    //   "    def foo(...) -> typing.Coroutine[typing.Any, typing.Any, None]:"
    let trimmed = line.trim_start();

    // Must start with "def " (not already "async def")
    if !trimmed.starts_with("def ") {
        return None;
    }

    // Find the return type annotation
    let arrow = line.rfind(") -> ")?;
    let after_arrow = &line[arrow + 5..]; // skip ") -> "

    // Check for the Coroutine pattern
    let prefix = "typing.Coroutine[typing.Any, typing.Any, ";
    if !after_arrow.starts_with(prefix) {
        return None;
    }

    // Extract the inner type (everything between the last comma+space and the trailing "]:")
    let inner_start = arrow + 5 + prefix.len();
    let rest = &line[inner_start..];
    let end = rest.rfind("]:")?;
    let inner_type = &rest[..end];

    // Reconstruct: replace "def " with "async def " and use the inner type
    let indent = &line[..line.len() - trimmed.len()];
    let sig = &trimmed[4..]; // skip "def "
    let paren_end = sig.find(") -> ")?;
    let method_sig = &sig[..paren_end + 1]; // "method_name(...)"

    Some(format!("{indent}async def {method_sig} -> {inner_type}:"))
}
