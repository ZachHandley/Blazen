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

from _mock_llm_server import MockServer, start_mock_server  # noqa: E402


@pytest.fixture
def mock_llm_server() -> MockServer:
    """Function-scoped local mock LLM server with a fresh free port.

    Function scope guarantees that parallel ``pytest -n auto`` workers never
    share a port or controller state. The server is torn down cleanly after
    each test.
    """
    server = start_mock_server()
    try:
        yield server
    finally:
        server.shutdown()
