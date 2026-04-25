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
//!
//! The body is gated on `not(feature = "extension-module")` so that
//! `cargo test --no-run --workspace --all-features` (which is what
//! `cargo nextest run --workspace --all-features` invokes under the hood)
//! does not try to link an unusable example binary against a libpython
//! whose symbols have been stripped by `pyo3/extension-module`.
//! When that feature is on, the example is a no-op stub.

#[cfg(feature = "extension-module")]
fn main() {
    eprintln!(
        "stub_gen is disabled when the `extension-module` feature is on; \
         re-run without that feature: `cargo run --example stub_gen -p blazen-py`"
    );
    std::process::exit(2);
}

#[cfg(not(feature = "extension-module"))]
use std::fs;

#[cfg(not(feature = "extension-module"))]
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
    let processed = inject_exception_stubs(&processed);
    fs::write(&pyi_path, processed).expect("failed to write blazen.pyi");
}

/// Inject stub declarations for the 10 Blazen exception classes.
///
/// `pyo3::create_exception!` produces runtime-only Python exception classes
/// that are invisible to `pyo3-stub-gen`. Without this injection step, the
/// generated `blazen.pyi` would not declare `BlazenError`, `AuthError`,
/// `ProviderError`, etc., which breaks IDE autocompletion and type checking
/// for any code that does `except ProviderError as e: e.status`.
///
/// This function:
/// 1. Inserts a hand-written block of class declarations AFTER the
///    `__all__ = [...]` list, mirroring the hierarchy from
///    `crates/blazen-py/src/error.rs`.
/// 2. Adds the exception class names into `__all__` alphabetically.
///
/// `ProviderError` attributes (`provider`, `status`, etc.) are set on the
/// exception instance via `setattr` in the Rust mapper, but declared in
/// the stub with their typed signatures so IDE completion and type
/// checkers understand them.
#[cfg(not(feature = "extension-module"))]
fn inject_exception_stubs(content: &str) -> String {
    const EXCEPTION_STUBS: &str = r#"

# --- Exception hierarchy ---------------------------------------------------
# create_exception!-produced types are invisible to pyo3-stub-gen, so they
# are hand-declared here. Inheritance mirrors src/error.rs:
#   BlazenError <- builtins.Exception
#   AuthError / RateLimitError / TimeoutError / ValidationError /
#   ContentPolicyError / ProviderError / UnsupportedError /
#   ComputeError / MediaError  <- BlazenError

class BlazenError(builtins.Exception):
    """Base class for all Blazen runtime errors."""
    ...

class AuthError(BlazenError):
    """Authentication failed (invalid / missing API key)."""
    ...

class RateLimitError(BlazenError):
    """Provider rate-limited the request."""
    ...

class TimeoutError(BlazenError):
    """The operation timed out."""
    ...

class ValidationError(BlazenError):
    """Invalid input rejected before the provider round-trip."""
    ...

class ContentPolicyError(BlazenError):
    """Provider rejected the request for policy reasons."""
    ...

class ProviderError(BlazenError):
    """Provider-side error. For HTTP failures, structured attributes are
    populated via `setattr` on the exception instance:
    - `provider`: str (e.g. "fal", "openrouter")
    - `status`: int | None (HTTP status, None for non-HTTP provider errors)
    - `endpoint`: str | None (request URL)
    - `request_id`: str | None (x-fal-request-id / x-request-id if present)
    - `detail`: str | None (parsed from JSON error body)
    - `raw_body`: str | None (response body, capped at 4 KiB)
    - `retry_after_ms`: int | None (parsed Retry-After header)
    """
    provider: str
    status: typing.Optional[int]
    endpoint: typing.Optional[str]
    request_id: typing.Optional[str]
    detail: typing.Optional[str]
    raw_body: typing.Optional[str]
    retry_after_ms: typing.Optional[int]

class UnsupportedError(BlazenError):
    """Requested capability is not supported by this provider / backend."""
    ...

class ComputeError(BlazenError):
    """Compute job error (cancelled, quota exceeded, etc)."""
    ...

class MediaError(BlazenError):
    """Media handling error (invalid input, size exceeded, etc)."""
    ...
"#;

    // Insert exception names into __all__ alphabetically. Existing __all__
    // entries come from pyo3-stub-gen and are already sorted.
    const EXTRA_ALL_NAMES: &[&str] = &[
        "AuthError",
        "BlazenError",
        "ComputeError",
        "ContentPolicyError",
        "MediaError",
        "ProviderError",
        "RateLimitError",
        "TimeoutError",
        "UnsupportedError",
        "ValidationError",
    ];

    let with_all = inject_into_all(content, EXTRA_ALL_NAMES);

    // Append the class-stub block at the end of the file.
    format!("{}{}", with_all.trim_end(), EXCEPTION_STUBS)
}

/// Insert `extras` into the existing `__all__ = [...]` block, keeping the
/// list alphabetically sorted. Each name is wrapped in double quotes and
/// matched case-sensitively.
#[cfg(not(feature = "extension-module"))]
fn inject_into_all(content: &str, extras: &[&str]) -> String {
    let Some(start) = content.find("__all__ = [") else {
        return content.to_string();
    };
    let Some(end_rel) = content[start..].find("]") else {
        return content.to_string();
    };
    let end = start + end_rel;

    let block = &content[start..end];
    // Extract existing quoted names.
    let mut names: Vec<String> = Vec::new();
    for piece in block.split(',') {
        let trimmed = piece.trim().trim_matches(|c: char| c == '[' || c == ']');
        let trimmed = trimmed
            .trim_start_matches("__all__")
            .trim_start_matches(" = [");
        let trimmed = trimmed.trim();
        if trimmed.starts_with('"') && trimmed.len() >= 2 {
            let inner = &trimmed[1..trimmed.len() - 1];
            if !inner.is_empty() {
                names.push(inner.to_string());
            }
        }
    }

    for extra in extras {
        if !names.iter().any(|n| n == extra) {
            names.push((*extra).to_string());
        }
    }
    names.sort();

    // Rebuild the __all__ block with 4-space indentation matching the
    // existing format.
    let mut rebuilt = String::from("__all__ = [\n");
    for n in &names {
        rebuilt.push_str(&format!("    \"{n}\",\n"));
    }
    rebuilt.push(']');

    let mut result = String::with_capacity(content.len() + 256);
    result.push_str(&content[..start]);
    result.push_str(&rebuilt);
    result.push_str(&content[end + 1..]);
    result
}

/// Rewrite `def method(...) -> typing.Coroutine[typing.Any, typing.Any, T]:`
/// into `async def method(...) -> T:`.
#[cfg(not(feature = "extension-module"))]
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

#[cfg(not(feature = "extension-module"))]
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
