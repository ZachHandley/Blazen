"""Unit tests for the structured ProviderError exception class.

These tests exercise the Python-side `ProviderError` class directly —
they do NOT require API keys and do NOT make network calls. They verify
the class hierarchy, that `setattr`-style attribute access works for
users who *construct* a ProviderError themselves (e.g. in test fixtures
or custom providers), and that raise/catch patterns work.

Integration coverage — asserting that a real `BlazenError::ProviderHttp`
from Rust arrives with `.status`, `.provider`, etc. populated — lives in
`tests/python/test_fal_smoke.py` under an API-keyed test that
deliberately triggers a 4xx.
"""

import pytest

import blazen
from blazen import BlazenError, ProviderError


def test_provider_error_is_registered():
    """ProviderError is exposed as a module-level class."""
    assert hasattr(blazen, "ProviderError")
    assert ProviderError is blazen.ProviderError


def test_provider_error_inherits_blazen_error():
    """ProviderError is a subclass of BlazenError (and Exception)."""
    assert issubclass(ProviderError, BlazenError)
    assert issubclass(ProviderError, Exception)


def test_provider_error_raisable_and_catchable_as_blazen_error():
    """ProviderError can be caught by a `except BlazenError:` block."""
    with pytest.raises(BlazenError):
        raise ProviderError("fal: boom")


def test_provider_error_direct_message_construction():
    """User code can construct and raise ProviderError with a message."""
    with pytest.raises(ProviderError) as exc_info:
        raise ProviderError("fal: upstream down (status=503)")
    assert "fal" in str(exc_info.value)


def test_provider_error_setattr_attributes_roundtrip():
    """Structured attributes can be set and read back.

    Mirrors the shape the Rust mapper uses: `setattr` on the exception
    instance. Verifies Python callers can inspect `e.status`, `e.provider`,
    etc. after catching.
    """
    err = ProviderError("fal: overloaded")
    err.provider = "fal"
    err.status = 503
    err.endpoint = "https://fal.ai/queue/xyz/result"
    err.request_id = "abc-123"
    err.detail = "service temporarily unavailable"
    err.raw_body = '{"detail":"service temporarily unavailable"}'
    err.retry_after_ms = 5000

    with pytest.raises(ProviderError) as exc_info:
        raise err

    caught = exc_info.value
    assert caught.provider == "fal"
    assert caught.status == 503
    assert caught.endpoint == "https://fal.ai/queue/xyz/result"
    assert caught.request_id == "abc-123"
    assert caught.detail == "service temporarily unavailable"
    assert caught.raw_body == '{"detail":"service temporarily unavailable"}'
    assert caught.retry_after_ms == 5000


def test_provider_error_retry_branching_pattern():
    """Canonical Python usage: inspect `.status` to decide retry."""
    err = ProviderError("openrouter: upstream 5xx")
    err.provider = "openrouter"
    err.status = 503
    err.retry_after_ms = 2000

    should_retry = False
    try:
        raise err
    except ProviderError as e:
        if e.status is not None and (e.status >= 500 or e.status == 429):
            should_retry = True

    assert should_retry, "5xx should trigger retry branch"
