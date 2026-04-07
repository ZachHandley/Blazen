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
import sqlite3
from dataclasses import dataclass

import pytest

from blazen import Context, Event, StartEvent, StopEvent, Workflow, step


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
