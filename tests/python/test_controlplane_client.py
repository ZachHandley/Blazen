"""Smoke tests for the orchestrator-side ``ControlPlaneClient``.

These tests verify import + basic argument handling. End-to-end
RPC behavior against a live server is exercised by the Rust
integration suite in ``crates/blazen-controlplane/tests/``; here we
just confirm the Python surface marshals arguments and raises
sensible errors when the connection cannot be made.
"""

from __future__ import annotations

import pytest

blazen = pytest.importorskip("blazen")
required = ["ControlPlaneClient", "ResourceHint", "ControlPlaneError"]
missing = [name for name in required if not hasattr(blazen, name)]
if missing:
    pytest.skip(
        f"blazen was built without the distributed feature (missing: {', '.join(missing)})",
        allow_module_level=True,
    )

from blazen import ControlPlaneClient, ResourceHint  # noqa: E402


@pytest.mark.asyncio
async def test_connect_rejects_invalid_endpoint() -> None:
    # Empty / malformed URIs must be rejected eagerly so callers don't
    # block forever on a doomed connection attempt.
    with pytest.raises(Exception):  # noqa: B017
        await ControlPlaneClient.connect("not a uri")


@pytest.mark.asyncio
async def test_connect_to_unreachable_endpoint_raises() -> None:
    # Port 1 is reserved; nothing listens there. The connect must fail
    # within the tonic transport layer and surface as
    # ``ControlPlaneError`` (which subclasses ``BlazenError``).
    with pytest.raises(Exception):  # noqa: B017
        await ControlPlaneClient.connect("http://127.0.0.1:1")


@pytest.mark.asyncio
async def test_connect_with_mtls_rejects_missing_pem() -> None:
    with pytest.raises(Exception):  # noqa: B017
        await ControlPlaneClient.connect(
            "https://127.0.0.1:7445",
            mtls=("/missing/cert.pem", "/missing/key.pem", "/missing/ca.pem"),
        )


def test_resource_hint_round_trips_through_submit_call_site() -> None:
    # ResourceHint is the only typed builder argument to
    # ControlPlaneClient.submit_workflow — verify it constructs and
    # exposes its fields so callers can pass it through.
    hint = ResourceHint(vram_mb=8192, cpu_cores=4.0, expected_seconds=15)
    assert hint.vram_mb == 8192
    assert hint.cpu_cores == pytest.approx(4.0)
    assert hint.expected_seconds == 15
