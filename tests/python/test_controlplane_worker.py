"""Smoke tests for the control-plane worker Python bindings.

These tests verify that the worker-side classes import, the builder
chain on ``ControlPlaneWorkerConfig`` composes, and the abstract
``AssignmentHandler`` base behaves sensibly. They deliberately stop
short of spinning up an in-process control-plane server — that surface
is exercised by the Rust integration suite in
``crates/blazen-controlplane/tests/``.
"""

from __future__ import annotations

import pytest

# Skip the whole module when the distributed build wasn't compiled in.
blazen = pytest.importorskip("blazen")
required = [
    "WorkerCapability",
    "AdmissionMode",
    "ResourceHint",
    "RunStatus",
    "ControlPlaneWorkerConfig",
    "AssignmentHandler",
    "AssignmentContext",
    "ControlPlaneWorker",
    "ControlPlaneClient",
    "RunEventStream",
    "ControlPlaneError",
]
missing = [name for name in required if not hasattr(blazen, name)]
if missing:
    pytest.skip(
        f"blazen was built without the distributed feature (missing: {', '.join(missing)})",
        allow_module_level=True,
    )

from blazen import (  # noqa: E402
    AdmissionMode,
    AssignmentHandler,
    ControlPlaneWorker,
    ControlPlaneWorkerConfig,
    ResourceHint,
    RunStatus,
    WorkerCapability,
)


# ---------------------------------------------------------------------------
# Value-type smoke tests
# ---------------------------------------------------------------------------


def test_worker_capability_roundtrip() -> None:
    cap = WorkerCapability("workflow:hello", 7)
    assert cap.kind == "workflow:hello"
    assert cap.version == 7
    assert "workflow:hello" in repr(cap)


def test_worker_capability_default_version() -> None:
    cap = WorkerCapability("step:fetch")
    assert cap.version == 1


def test_admission_mode_variants() -> None:
    fixed = AdmissionMode.fixed(max_in_flight=4)
    assert fixed.kind == "Fixed"
    assert fixed.max_in_flight == 4
    assert fixed.max_vram_mb is None

    vram = AdmissionMode.vram_budget(max_vram_mb=24576)
    assert vram.kind == "VramBudget"
    assert vram.max_vram_mb == 24576
    assert vram.max_in_flight is None

    reactive = AdmissionMode.reactive()
    assert reactive.kind == "Reactive"
    assert reactive.max_in_flight is None
    assert reactive.max_vram_mb is None


def test_resource_hint_optional_fields() -> None:
    hint = ResourceHint()
    assert hint.vram_mb is None
    assert hint.cpu_cores is None
    assert hint.expected_seconds is None

    hint = ResourceHint(vram_mb=12_000, cpu_cores=8.0, expected_seconds=30)
    assert hint.vram_mb == 12_000
    assert hint.cpu_cores == pytest.approx(8.0)
    assert hint.expected_seconds == 30


def test_run_status_enum_members_present() -> None:
    # Enum members must exist so callers can compare against the strings
    # returned in the `status` field of snapshot dicts.
    for name in ("Pending", "Running", "Completed", "Failed", "Cancelled"):
        assert hasattr(RunStatus, name), f"RunStatus is missing {name}"


# ---------------------------------------------------------------------------
# WorkerConfig builder chain
# ---------------------------------------------------------------------------


def test_worker_config_builder_chain_composes() -> None:
    cfg = (
        ControlPlaneWorkerConfig("http://localhost:7445", "node-a")
        .with_capability(WorkerCapability("workflow:hello"))
        .with_capability(WorkerCapability("step:fetch", 2))
        .with_tag("region", "us-west")
        .with_tag("gpu", "h100")
        .with_admission(AdmissionMode.fixed(max_in_flight=2))
        .with_heartbeat_interval_ms(2500)
    )
    assert cfg.endpoint == "http://localhost:7445"
    assert cfg.node_id == "node-a"
    # Repr should reflect the chained additions so subscribers can sanity-check
    # the config from the Python side without exposing internal getters.
    assert "node-a" in repr(cfg)


def test_worker_config_with_mtls_errors_on_missing_pem() -> None:
    cfg = ControlPlaneWorkerConfig("https://localhost:7445", "node-tls")
    with pytest.raises(Exception):  # noqa: B017 — ControlPlaneError
        cfg.with_mtls("/nonexistent/cert.pem", "/nonexistent/key.pem", "/nonexistent/ca.pem")


# ---------------------------------------------------------------------------
# AssignmentHandler abstract base
# ---------------------------------------------------------------------------


def test_assignment_handler_default_handle_raises_not_implemented() -> None:
    handler = AssignmentHandler()
    with pytest.raises(NotImplementedError):
        handler.handle({}, None)


def test_assignment_handler_default_lifecycle_hooks_are_noops() -> None:
    handler = AssignmentHandler()
    # These must not raise — the defaults are no-ops so a Python author
    # only has to override `handle` to get a working worker.
    handler.on_cancel("00000000-0000-0000-0000-000000000000")
    handler.on_drain(False)
    handler.on_drain(True)


def test_assignment_handler_default_evaluate_offer_claims() -> None:
    handler = AssignmentHandler()
    assert handler.evaluate_offer({}) == "claim"


def test_assignment_handler_subclass_overrides_hooks() -> None:
    class DeclineEverything(AssignmentHandler):
        def __init__(self) -> None:
            super().__init__()
            self.cancel_calls: list[str] = []

        def evaluate_offer(self, offer: dict) -> str:
            _ = offer
            return "decline"

        def on_cancel(self, run_id: str) -> None:
            self.cancel_calls.append(run_id)

    handler = DeclineEverything()
    assert handler.evaluate_offer({"assignment": {}}) == "decline"
    handler.on_cancel("run-1")
    assert handler.cancel_calls == ["run-1"]


# ---------------------------------------------------------------------------
# ControlPlaneWorker connect-and-validate
# ---------------------------------------------------------------------------


def test_control_plane_worker_connect_rejects_bad_endpoint() -> None:
    cfg = ControlPlaneWorkerConfig("not a uri", "node-bad")
    with pytest.raises(Exception):  # noqa: B017 — ControlPlaneError
        ControlPlaneWorker.connect(cfg)


def test_control_plane_worker_connect_accepts_valid_uri() -> None:
    cfg = ControlPlaneWorkerConfig("http://localhost:7445", "node-good")
    worker = ControlPlaneWorker.connect(cfg)
    # No connection happens yet — `run()` is responsible for opening
    # the bidi stream. Calling `shutdown()` on the un-started worker
    # must be safe and idempotent.
    worker.shutdown()
    worker.shutdown()
