"""Pytest configuration for Blazen Python binding E2E tests."""

import pytest

# Skip the entire test suite if the blazen native module is not importable
# (e.g. the wheel was not built with `maturin develop`).
try:
    import blazen  # noqa: F401
except ImportError:
    pytest.skip(
        "blazen native module not available (run `maturin develop -m crates/blazen-py/Cargo.toml`)",
        allow_module_level=True,
    )
