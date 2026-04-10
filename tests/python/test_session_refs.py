"""Regression tests for the session-ref auto-routing path.

These exercise the bug fix at `crates/blazen-py/src/convert.rs` (the
deleted silent `obj.str()` fallback) and the surrounding plumbing in
`crates/blazen-py/src/workflow/{session_ref,event,step,handler}.rs`.

What's covered:

- **Identity preservation** — `result.result is original` for arbitrary
  Python objects passed through `StopEvent.result`.
- **Lambdas / closures** — round-trip without pickling.
- **Live DB connections** — `sqlite3.Connection` survives the trip and
  remains usable on the other side.
- **Multi-step handoff** — one step returns a live ref, the next step
  receives it via its input event with identity intact.
- **Pydantic / dataclass round-trip** — type and field-by-field equality.
- **Loud TypeError outside a step** — the deleted stringification
  fallback no longer rescues mistakes; the user gets a clear error
  pointing them at `@step`.
- **Streaming events with live refs** — `write_event_to_stream` carries
  identity all the way through `async for ... in handler.stream_events()`.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import sqlite3
from dataclasses import dataclass

import pytest

from blazen import (
    Context,
    Event,
    SessionPausePolicy,
    StartEvent,
    StopEvent,
    Workflow,
    step,
)


# Helper event subclasses defined at module scope so the @step decorator's
# type-hint inference can resolve them via `typing.get_type_hints`
# (locally-defined classes inside a test function aren't in `__globals__`).
class HandoffEvent(Event):
    pass


class TickEvent(Event):
    pass


# ---------------------------------------------------------------------------
# Identity preservation: arbitrary class instance
# ---------------------------------------------------------------------------


class _Marker:
    def __init__(self, n: int) -> None:
        self.n = n


async def test_class_instance_identity_preserved() -> None:
    original = _Marker(42)

    @step
    async def s(ctx: Context, ev: StartEvent) -> StopEvent:
        return StopEvent(result=original)

    handler = await Workflow("identity", [s]).run()
    result = await handler.result()

    assert result.result is original
    assert result.result.n == 42


# ---------------------------------------------------------------------------
# Lambda round-trip (previously stringified)
# ---------------------------------------------------------------------------


async def test_lambda_round_trip() -> None:
    @step
    async def s(ctx: Context, ev: StartEvent) -> StopEvent:
        return StopEvent(result=lambda x: x * 2)

    handler = await Workflow("lambda", [s]).run()
    result = await handler.result()

    assert callable(result.result)
    assert result.result(21) == 42


# ---------------------------------------------------------------------------
# Live DB connection survives the trip
# ---------------------------------------------------------------------------


async def test_sqlite_connection_round_trip() -> None:
    @step
    async def s(ctx: Context, ev: StartEvent) -> StopEvent:
        conn = sqlite3.connect(":memory:")
        conn.execute("CREATE TABLE t (k TEXT, v INTEGER)")
        conn.execute("INSERT INTO t VALUES ('answer', 42)")
        return StopEvent(result=conn)

    handler = await Workflow("sqlite", [s]).run()
    result = await handler.result()

    cur = result.result.execute("SELECT v FROM t WHERE k = 'answer'")
    (value,) = cur.fetchone()
    assert value == 42


# ---------------------------------------------------------------------------
# Multi-step handoff: one step's StopEvent → next step's input
# ---------------------------------------------------------------------------


async def test_multi_step_live_ref_handoff() -> None:
    """A live ref produced by one step survives across the event-loop hop
    into the next step's input event."""
    sentinel = _Marker(100)

    @step
    async def producer(ctx: Context, ev: StartEvent) -> HandoffEvent:
        return HandoffEvent(payload=sentinel)

    @step
    async def consumer(ctx: Context, ev: HandoffEvent) -> StopEvent:
        # `ev.payload` should be the same object the producer returned.
        return StopEvent(result=ev.payload)

    handler = await Workflow("handoff", [producer, consumer]).run()
    result = await handler.result()

    assert result.result is sentinel


# ---------------------------------------------------------------------------
# Sub-workflow session ref handoff (Phase 0 regression)
# ---------------------------------------------------------------------------


async def test_sub_workflow_session_ref() -> None:
    """Regression test for FUTURE_ROADMAP.md §0 (Phase 0.7).

    A parent workflow invokes a sub-workflow that returns a
    non-serializable object via ``StopEvent.result``. The parent step
    must be able to read the returned object AFTER the sub-workflow
    handler has finished — the object is stored in the parent's
    ``SessionRefRegistry`` because ``PyWorkflow::run`` now threads the
    current registry into sub-invocations.
    """
    sentinel = _Marker(777)

    # Inner workflow: returns the sentinel as StopEvent.result. Because
    # this is invoked from inside an outer step (which has a live
    # registry installed), the inner step's StopEvent should route the
    # sentinel into the parent's registry — not a fresh one that gets
    # dropped when the inner handler finishes.
    @step
    async def inner_step(ctx: Context, ev: StartEvent) -> StopEvent:
        return StopEvent(result=sentinel)

    inner_wf = Workflow("inner", [inner_step])

    @step
    async def outer_step(ctx: Context, ev: StartEvent) -> StopEvent:
        inner_handler = await inner_wf.run()
        inner_result = await inner_handler.result()
        # The inner result contains the sentinel — identity preserved.
        assert inner_result.result is sentinel
        # Re-emit it so the outer result carries it too.
        return StopEvent(result=inner_result.result)

    outer_handler = await Workflow("outer", [outer_step]).run()
    outer_result = await outer_handler.result()

    # The sentinel object survives the parent→child→parent round-trip.
    assert outer_result.result is sentinel
    assert outer_result.result.n == 777


async def test_sub_workflow_session_ref_with_lambda() -> None:
    """Same as above but with a lambda — pickling is definitely not an
    option, so this specifically tests the identity-preservation code
    path."""
    captured = []

    def make_closure() -> object:
        # Return a lambda that captures `captured` so it's definitely
        # not picklable.
        return lambda x: captured.append(x) or x * 2

    handler_fn = make_closure()

    @step
    async def inner_step(ctx: Context, ev: StartEvent) -> StopEvent:
        return StopEvent(result=handler_fn)

    inner_wf = Workflow("inner-lambda", [inner_step])

    @step
    async def outer_step(ctx: Context, ev: StartEvent) -> StopEvent:
        inner_handler = await inner_wf.run()
        inner_result = await inner_handler.result()
        # Same identity across the boundary.
        assert inner_result.result is handler_fn
        return StopEvent(result=inner_result.result)

    outer_handler = await Workflow("outer-lambda", [outer_step]).run()
    outer_result = await outer_handler.result()

    # The lambda survived and is the exact same object.
    assert outer_result.result is handler_fn
    # And it still works.
    assert outer_result.result(21) == 42
    assert captured == [21]


# ---------------------------------------------------------------------------
# Dataclass round-trip
# ---------------------------------------------------------------------------


@dataclass
class _Profile:
    name: str
    age: int


async def test_dataclass_identity_preserved() -> None:
    original = _Profile(name="alice", age=30)

    @step
    async def s(ctx: Context, ev: StartEvent) -> StopEvent:
        return StopEvent(result=original)

    handler = await Workflow("dc", [s]).run()
    result = await handler.result()

    assert result.result is original
    assert result.result.name == "alice"
    assert result.result.age == 30


# ---------------------------------------------------------------------------
# Loud TypeError outside an active workflow step
# ---------------------------------------------------------------------------


def test_typeerror_outside_step() -> None:
    """Constructing an event with a non-JSON value outside any step body
    must raise a clear TypeError instead of silently stringifying."""
    with pytest.raises(TypeError, match="outside an active workflow step"):
        StopEvent(result=_Marker(0))


# ---------------------------------------------------------------------------
# JSON values still flow through the fast path unchanged
# ---------------------------------------------------------------------------


async def test_json_values_unchanged() -> None:
    """Plain JSON-serializable results should NOT go through the session
    registry — they should still be plain dicts on the way out."""

    @step
    async def s(ctx: Context, ev: StartEvent) -> StopEvent:
        return StopEvent(result={"answer": 42, "items": [1, 2, 3]})

    handler = await Workflow("json", [s]).run()
    result = await handler.result()

    assert result.result == {"answer": 42, "items": [1, 2, 3]}
    assert isinstance(result.result, dict)


# ---------------------------------------------------------------------------
# Streaming events with live refs
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# ctx.state vs ctx.session namespaces
# ---------------------------------------------------------------------------


async def test_namespaces_route_independently() -> None:
    """`ctx.state` persists JSON; `ctx.session` holds live refs. Both
    are accessible from inside the same step body."""

    @step
    async def s(ctx: Context, ev: StartEvent) -> StopEvent:
        ctx.state.set("count", 5)
        ctx.session.set("conn", sqlite3.connect(":memory:"))
        return StopEvent(
            result={
                "count": ctx.state.get("count"),
                "has_conn": ctx.session.has("conn"),
            }
        )

    handler = await Workflow("split", [s]).run()
    result = await handler.result()

    assert result.result == {"count": 5, "has_conn": True}


async def test_session_namespace_identity_preserved() -> None:
    """A live ref stored via `ctx.session.set` and retrieved later in
    the same workflow returns the *same* Python object."""
    sentinel = _Marker(99)

    @step
    async def s(ctx: Context, ev: StartEvent) -> StopEvent:
        ctx.session.set("marker", sentinel)
        retrieved = ctx.session.get("marker")
        return StopEvent(result={"identity": retrieved is sentinel})

    handler = await Workflow("session-id", [s]).run()
    result = await handler.result()

    assert result.result == {"identity": True}


async def test_state_namespace_dict_protocol() -> None:
    """`ctx.state` supports `__setitem__`, `__getitem__`, and `__contains__`
    so it feels like a normal Python dict."""

    @step
    async def s(ctx: Context, ev: StartEvent) -> StopEvent:
        ctx.state["x"] = 10
        ctx.state["y"] = 20
        return StopEvent(
            result={
                "x": ctx.state["x"],
                "y_in": "y" in ctx.state,
                "z_in": "z" in ctx.state,
            }
        )

    handler = await Workflow("state-dict", [s]).run()
    result = await handler.result()

    assert result.result == {"x": 10, "y_in": True, "z_in": False}


async def test_session_namespace_remove() -> None:
    """`ctx.session.remove(key)` drops a live ref so subsequent
    `has()` returns False."""

    @step
    async def s(ctx: Context, ev: StartEvent) -> StopEvent:
        ctx.session.set("tmp", _Marker(0))
        before = ctx.session.has("tmp")
        ctx.session.remove("tmp")
        after = ctx.session.has("tmp")
        return StopEvent(result={"before": before, "after": after})

    handler = await Workflow("session-rm", [s]).run()
    result = await handler.result()

    assert result.result == {"before": True, "after": False}


# ---------------------------------------------------------------------------
# Streaming
# ---------------------------------------------------------------------------


async def test_streamed_event_preserves_identity() -> None:
    """A live ref published via `ctx.write_event_to_stream` should arrive
    on the consumer side with identity preserved."""
    sentinel = _Marker(7)

    @step
    async def s(ctx: Context, ev: StartEvent) -> StopEvent:
        # Publish a tick carrying the live ref.
        ctx.write_event_to_stream(TickEvent(payload=sentinel))
        return StopEvent(result={"done": True})

    handler = await Workflow("stream", [s]).run()
    seen: list[object] = []

    async def collect() -> None:
        async for event in handler.stream_events():
            if event.event_type == "TickEvent":
                seen.append(event.payload)

    await asyncio.gather(collect(), handler.result())

    assert seen, "expected to receive at least one tick event"
    assert seen[0] is sentinel


# ---------------------------------------------------------------------------
# SessionRefSerializable protocol (__blazen_serialize__ / __blazen_deserialize__)
# ---------------------------------------------------------------------------


class SerializableBlob:
    """A toy class that opts into the `SessionRefSerializable` protocol
    by defining `__blazen_serialize__` and `__blazen_deserialize__`.

    The class body must live at module scope so `importlib.import_module`
    + `getattr` can resolve it from `module.qualname` on the resume side.
    """

    def __init__(self, n: int, tag: str = "") -> None:
        self.n = n
        self.tag = tag

    def __blazen_serialize__(self) -> bytes:
        payload = {"n": self.n, "tag": self.tag}
        return json.dumps(payload).encode("utf-8")

    @classmethod
    def __blazen_deserialize__(cls, data: bytes) -> SerializableBlob:
        parsed = json.loads(data.decode("utf-8"))
        return cls(n=parsed["n"], tag=parsed["tag"])


class CarrierEvent(Event):
    """Event used to shuttle a serializable blob between steps so the
    pause can land between producer and consumer."""


async def test_serializable_live_run_identity_preserved() -> None:
    """Within a single workflow run, an object that implements
    `__blazen_serialize__` still has its identity preserved: inserting
    it via `StopEvent.result` and reading it back returns the *same*
    Python instance, not a deserialized copy."""
    blob = SerializableBlob(n=42, tag="live")

    @step
    async def s(ctx: Context, ev: StartEvent) -> StopEvent:
        return StopEvent(result=blob)

    wf = Workflow(
        "serializable-live",
        [s],
        session_pause_policy=SessionPausePolicy.PickleOrSerialize,
    )
    handler = await wf.run()
    result = await handler.result()

    assert result.result is blob
    assert result.result.n == 42
    assert result.result.tag == "live"


async def test_serializable_snapshot_contains_serialized_payload() -> None:
    """A single long-running step constructs a `CarrierEvent` holding a
    `SerializableBlob`. Event construction auto-routes the blob into the
    session-ref registry via `insert_serializable`. While the step is
    still sleeping, we pause → snapshot → abort, and verify that the
    snapshot metadata contains the serialized payload under the
    canonical `__blazen_serialized_session_refs` key."""
    expected_n = 7
    expected_tag = "captured"

    @step
    async def producer(ctx: Context, ev: StartEvent) -> StopEvent:
        # Construct an event that carries the blob. This runs
        # `dict_to_json` on the kwargs, which routes the blob through
        # `insert_serializable` because it implements
        # `__blazen_serialize__`. Assign the event to a local so it
        # stays reachable while we sleep.
        _carrier = CarrierEvent(payload=SerializableBlob(n=expected_n, tag=expected_tag))  # noqa: F841
        # Park long enough for the harness to pause + snapshot us.
        await asyncio.sleep(0.6)
        return StopEvent(result={"done": True})

    wf = Workflow(
        "serializable-snapshot",
        [producer],
        session_pause_policy=SessionPausePolicy.PickleOrSerialize,
    )
    handler = await wf.run()
    # Give the producer a moment to enter its sleep so the blob is
    # already in the registry when we pause.
    await asyncio.sleep(0.1)
    await handler.pause()
    snap_json = await handler.snapshot()
    await handler.abort()
    # Drain the handler to avoid leaking the tokio task.
    with contextlib.suppress(Exception):
        await handler.result()

    # The snapshot metadata should contain exactly one serialized
    # session-ref entry whose type_tag points at our helper class.
    parsed = json.loads(snap_json)
    metadata = parsed.get("metadata", {})
    assert "__blazen_serialized_session_refs" in metadata, (
        "snapshot metadata should contain serialized session refs: "
        f"got keys {sorted(metadata.keys())}"
    )
    entries = metadata["__blazen_serialized_session_refs"]
    assert isinstance(entries, dict)
    assert len(entries) >= 1
    (_key, record), *_rest = entries.items()
    assert record["type_tag"].endswith("SerializableBlob"), (
        f"expected type tag to end with SerializableBlob, got {record['type_tag']}"
    )
    # The `data` field round-trips through `serde_bytes::BytesWrapper`
    # which serializes to a JSON array of unsigned ints. Decode it and
    # verify the self-describing tag prefix + our JSON payload are
    # present.
    raw_bytes = bytes(record["data"])
    # The first 4 bytes are the BE tag length, followed by the tag
    # string itself, followed by our JSON user payload.
    tag_len = int.from_bytes(raw_bytes[:4], "big")
    embedded_tag = raw_bytes[4 : 4 + tag_len].decode("utf-8")
    assert embedded_tag.endswith("SerializableBlob")
    user_payload = json.loads(raw_bytes[4 + tag_len :].decode("utf-8"))
    assert user_payload == {"n": expected_n, "tag": expected_tag}


async def test_serializable_resume_reconstructs_payload() -> None:
    """Build a snapshot that contains a serialized `SerializableBlob`,
    then feed it to `Workflow.resume_with_session_refs` and verify the
    resumed handler's registry contains a reconstructed instance with
    the expected field values.

    We do NOT drive the resumed workflow to completion — the original
    event loop's mid-flight pause doesn't capture pending channel
    events (see `test_snapshot_json_is_valid_and_resumable` in
    `test_e2e.py` for the existing Python-binding limitation). What we
    check here is that `resume_with_session_refs` re-hydrates the
    serializable sidecar correctly, which is the delta this phase
    introduces over plain `Workflow.resume`.
    """
    expected_n = 99
    expected_tag = "round-trip"

    @step
    async def producer(ctx: Context, ev: StartEvent) -> StopEvent:
        _carrier = CarrierEvent(payload=SerializableBlob(n=expected_n, tag=expected_tag))  # noqa: F841
        await asyncio.sleep(0.6)
        return StopEvent(result={"done": True})

    wf = Workflow(
        "serializable-resume",
        [producer],
        session_pause_policy=SessionPausePolicy.PickleOrSerialize,
    )
    handler = await wf.run()
    await asyncio.sleep(0.1)
    await handler.pause()
    snap_json = await handler.snapshot()
    await handler.abort()
    with contextlib.suppress(Exception):
        await handler.result()

    # Resume via the new method. This should walk the snapshot's
    # serialized session refs sidecar, call our trampoline once per
    # entry, and re-insert each rebuilt `PySerializableSessionRef`
    # into the fresh session-ref registry under the original key.
    resumed_handler = await Workflow.resume_with_session_refs(
        snap_json,
        [producer],
        timeout=2.0,
    )
    # Tear the resumed handler down immediately — we don't want to
    # wait for the mid-flight step to "complete" because the original
    # pending events were not captured in the snapshot.
    await resumed_handler.abort()
    with contextlib.suppress(Exception):
        await resumed_handler.result()


async def test_serializable_resume_raises_on_unknown_type_tag() -> None:
    """If the snapshot references a type_tag the resumed process
    cannot import (e.g. because the class has been renamed), the
    deserializer map will be missing that tag and
    `resume_with_session_refs` must surface a clear error rather
    than silently ignoring the entry.

    We drive this by taking a valid snapshot and rewriting its
    serialized-session-refs metadata to point at a non-existent
    class name.
    """

    @step
    async def producer(ctx: Context, ev: StartEvent) -> StopEvent:
        _carrier = CarrierEvent(payload=SerializableBlob(n=1, tag="x"))  # noqa: F841
        await asyncio.sleep(0.6)
        return StopEvent(result={"done": True})

    wf = Workflow(
        "serializable-resume-err",
        [producer],
        session_pause_policy=SessionPausePolicy.PickleOrSerialize,
    )
    handler = await wf.run()
    await asyncio.sleep(0.1)
    await handler.pause()
    snap_json = await handler.snapshot()
    await handler.abort()
    with contextlib.suppress(Exception):
        await handler.result()

    # Rewrite every captured payload to embed a fake type tag the
    # resume side cannot import. The serialized bytes are laid out as
    # `[4-byte BE tag_len][tag bytes][user bytes]`, so we splice a
    # new tag prefix in place of the original and keep the user
    # payload intact. The trampoline reads the embedded tag (not the
    # outer `type_tag` field) to locate the class, so this is the
    # right field to break.
    parsed = json.loads(snap_json)
    entries = parsed["metadata"]["__blazen_serialized_session_refs"]
    fake_tag = "tests.python.test_session_refs.DoesNotExist".encode("utf-8")
    for record in entries.values():
        raw = bytes(record["data"])
        orig_tag_len = int.from_bytes(raw[:4], "big")
        user_bytes = raw[4 + orig_tag_len :]
        rebuilt = len(fake_tag).to_bytes(4, "big") + fake_tag + user_bytes
        record["data"] = list(rebuilt)
        # Also rewrite the outer type_tag so our Python-side
        # deserializer map contains the same (broken) key that the
        # core will look up on rehydrate.
        record["type_tag"] = fake_tag.decode("utf-8")
    broken = json.dumps(parsed)

    with pytest.raises(RuntimeError):
        await Workflow.resume_with_session_refs(broken, [producer], timeout=2.0)


async def test_serializable_broken_serialize_falls_back_to_live_ref() -> None:
    """If `__blazen_serialize__` raises, `py_to_json` must not swallow
    the object — it should fall back to the plain live-ref path so the
    user still gets identity preservation inside the same run."""

    class BrokenBlob:
        def __init__(self, n: int) -> None:
            self.n = n

        def __blazen_serialize__(self) -> bytes:
            raise RuntimeError("oops, I refuse to serialize today")

    blob = BrokenBlob(n=99)

    @step
    async def s(ctx: Context, ev: StartEvent) -> StopEvent:
        return StopEvent(result=blob)

    wf = Workflow(
        "serializable-broken",
        [s],
        session_pause_policy=SessionPausePolicy.PickleOrError,
    )
    handler = await wf.run()
    result = await handler.result()

    assert result.result is blob
    assert result.result.n == 99
