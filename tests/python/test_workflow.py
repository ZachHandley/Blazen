"""E2E tests for the Blazen Python bindings.

These tests exercise the full workflow lifecycle through the PyO3 bindings:
event creation, step decoration, workflow execution, context sharing,
streaming, and fan-out.
"""

import asyncio

import pytest

from blazen import (
    Context,
    Event,
    StartEvent,
    StopEvent,
    Workflow,
    step,
)


# =========================================================================
# Event creation
# =========================================================================


def test_event_creation():
    ev = Event("AnalyzeEvent", text="hello", score=0.9)
    assert ev.event_type == "AnalyzeEvent"
    assert ev.text == "hello"
    assert ev.score == 0.9


def test_event_to_dict():
    ev = Event("TestEvent", x=1, y="two")
    d = ev.to_dict()
    assert d["x"] == 1
    assert d["y"] == "two"


def test_start_event():
    ev = StartEvent(message="hi")
    assert ev.event_type == "blazen::StartEvent"
    assert ev.message == "hi"


def test_stop_event():
    ev = StopEvent(result={"answer": 42})
    assert ev.event_type == "blazen::StopEvent"
    assert ev.result == {"answer": 42}


def test_event_attribute_error():
    ev = Event("Foo", x=1)
    with pytest.raises(AttributeError):
        _ = ev.nonexistent_field


# =========================================================================
# Single step echo (async)
# =========================================================================


@pytest.mark.asyncio
async def test_single_step_echo():
    @step
    async def echo(ctx: Context, ev: Event):
        return StopEvent(result=ev.to_dict())

    wf = Workflow("echo", [echo])
    handler = await wf.run(message="hello")
    result = await handler.result()

    assert result.event_type == "blazen::StopEvent"
    assert result.result["message"] == "hello"


# =========================================================================
# Single step echo (sync)
# =========================================================================


@pytest.mark.asyncio
async def test_single_step_echo_sync():
    @step
    def echo(ctx: Context, ev: Event):
        return StopEvent(result=ev.to_dict())

    wf = Workflow("echo-sync", [echo])
    handler = await wf.run(message="hello")
    result = await handler.result()

    assert result.event_type == "blazen::StopEvent"
    assert result.result["message"] == "hello"


# =========================================================================
# Multi-step pipeline
# =========================================================================


@pytest.mark.asyncio
async def test_multi_step_pipeline():
    @step
    async def analyze(ctx: Context, ev: Event):
        return Event("AnalyzeDone", text=ev.message, length=len(ev.message))

    @step(accepts=["AnalyzeDone"])
    async def finalize(ctx: Context, ev: Event):
        return StopEvent(result={"text": ev.text, "length": ev.length})

    wf = Workflow("pipeline", [analyze, finalize])
    handler = await wf.run(message="hello world")
    result = await handler.result()

    assert result.result["text"] == "hello world"
    assert result.result["length"] == 11


# =========================================================================
# Context set / get
# =========================================================================


@pytest.mark.asyncio
async def test_context_set_get():
    @step
    async def setter(ctx: Context, ev: Event):
        ctx.set("counter", 42)
        return Event("NextEvent")

    @step(accepts=["NextEvent"])
    async def getter(ctx: Context, ev: Event):
        val = ctx.get("counter")
        return StopEvent(result={"counter": val})

    wf = Workflow("ctx-test", [setter, getter])
    handler = await wf.run()
    result = await handler.result()

    assert result.result["counter"] == 42


# =========================================================================
# Context run_id
# =========================================================================


@pytest.mark.asyncio
async def test_context_run_id():
    run_id_holder = {}

    @step
    async def capture_id(ctx: Context, ev: Event):
        rid = ctx.run_id()
        run_id_holder["id"] = rid
        return StopEvent(result={"run_id": rid})

    wf = Workflow("run-id-test", [capture_id])
    handler = await wf.run()
    result = await handler.result()

    rid = result.result["run_id"]
    assert isinstance(rid, str)
    assert len(rid) > 0
    # UUID format: 8-4-4-4-12
    assert rid.count("-") == 4


# =========================================================================
# Streaming
# =========================================================================


@pytest.mark.asyncio
async def test_streaming():
    @step
    async def producer(ctx: Context, ev: Event):
        for i in range(3):
            ctx.write_event_to_stream(
                Event("Progress", step=i)
            )
        return StopEvent(result={"done": True})

    wf = Workflow("stream-test", [producer])
    handler = await wf.run()

    collected = []
    async for event in handler.stream_events():
        collected.append(event)

    result = await handler.result()
    assert result.result["done"] is True

    # We should have received at least some progress events.
    progress = [e for e in collected if e.event_type == "Progress"]
    assert len(progress) > 0


# =========================================================================
# Step returns list (fan-out)
# =========================================================================


@pytest.mark.asyncio
async def test_step_returns_list():
    @step
    async def fan_out(ctx: Context, ev: Event):
        return [
            Event("BranchA", value="a"),
            Event("BranchB", value="b"),
        ]

    @step(accepts=["BranchA"])
    async def handle_a(ctx: Context, ev: Event):
        return StopEvent(result={"branch": "a"})

    @step(accepts=["BranchB"])
    async def handle_b(ctx: Context, ev: Event):
        return StopEvent(result={"branch": "b"})

    wf = Workflow("fan-out", [fan_out, handle_a, handle_b])
    handler = await wf.run()
    result = await handler.result()

    # One of the two branches wins.
    assert result.result["branch"] in ("a", "b")


# =========================================================================
# Step returns None (side-effect + ctx.send_event continuation)
# =========================================================================


@pytest.mark.asyncio
async def test_step_returns_none():
    @step
    async def side_effect(ctx: Context, ev: Event):
        ctx.set("processed", True)
        ctx.send_event(Event("Continue"))
        return None

    @step(accepts=["Continue"])
    async def finisher(ctx: Context, ev: Event):
        processed = ctx.get("processed")
        return StopEvent(result={"processed": processed})

    wf = Workflow("none-test", [side_effect, finisher])
    handler = await wf.run()
    result = await handler.result()

    assert result.result["processed"] is True


# =========================================================================
# Context bytes roundtrip
# =========================================================================


@pytest.mark.asyncio
async def test_context_bytes_roundtrip():
    @step
    def setter(ctx: Context, ev: Event):
        ctx.set("k", b"\x00\x01\x02")
        return Event("NextEvent")

    @step(accepts=["NextEvent"])
    def getter(ctx: Context, ev: Event):
        val = ctx.get("k")
        assert isinstance(val, bytes)
        return StopEvent(result={"data": list(val)})

    wf = Workflow("bytes-roundtrip", [setter, getter])
    handler = await wf.run()
    result = await handler.result()

    assert result.result["data"] == [0, 1, 2]


# =========================================================================
# Context complex JSON roundtrip
# =========================================================================


@pytest.mark.asyncio
async def test_context_complex_json():
    @step
    def setter(ctx: Context, ev: Event):
        ctx.set("k", {"nested": [1, 2.5, None, True, "str"]})
        return Event("NextEvent")

    @step(accepts=["NextEvent"])
    def getter(ctx: Context, ev: Event):
        val = ctx.get("k")
        return StopEvent(result=val)

    wf = Workflow("complex-json", [setter, getter])
    handler = await wf.run()
    result = await handler.result()

    assert result.result == {"nested": [1, 2.5, None, True, "str"]}


# =========================================================================
# Context pickle roundtrip
# =========================================================================


@pytest.mark.asyncio
async def test_context_pickle_roundtrip():
    from dataclasses import dataclass

    @dataclass
    class Payload:
        name: str
        score: float

    @step
    def setter(ctx: Context, ev: Event):
        ctx.set("k", Payload(name="test", score=3.14))
        return Event("NextEvent")

    @step(accepts=["NextEvent"])
    def getter(ctx: Context, ev: Event):
        val = ctx.get("k")
        return StopEvent(result={"name": val.name, "score": val.score})

    wf = Workflow("pickle-roundtrip", [setter, getter])
    handler = await wf.run()
    result = await handler.result()

    assert result.result["name"] == "test"
    assert abs(result.result["score"] - 3.14) < 1e-9


# =========================================================================
# Context set_bytes / get_bytes
# =========================================================================


@pytest.mark.asyncio
async def test_context_set_bytes_get_bytes():
    @step
    def setter(ctx: Context, ev: Event):
        ctx.set_bytes("k", b"\xff\xfe")
        return Event("NextEvent")

    @step(accepts=["NextEvent"])
    def getter(ctx: Context, ev: Event):
        val = ctx.get_bytes("k")
        return StopEvent(result={"data": list(val)})

    wf = Workflow("set-bytes-get-bytes", [setter, getter])
    handler = await wf.run()
    result = await handler.result()

    assert result.result["data"] == [0xFF, 0xFE]


# =========================================================================
# Context overwrite
# =========================================================================


@pytest.mark.asyncio
async def test_context_overwrite():
    @step
    def setter(ctx: Context, ev: Event):
        ctx.set("k", 1)
        ctx.set("k", "hello")
        return Event("NextEvent")

    @step(accepts=["NextEvent"])
    def getter(ctx: Context, ev: Event):
        val = ctx.get("k")
        return StopEvent(result={"value": val})

    wf = Workflow("overwrite", [setter, getter])
    handler = await wf.run()
    result = await handler.result()

    assert result.result["value"] == "hello"


# =========================================================================
# Context buffer not mangled (bytearray vs array.array)
# =========================================================================


@pytest.mark.asyncio
async def test_context_buffer_not_mangled():
    import array

    @step
    def setter(ctx: Context, ev: Event):
        # bytearray should be stored as raw bytes (tier 1)
        ctx.set("ba", bytearray(b"\xaa\xbb"))
        # array.array is NOT bytes/bytearray, so it should pickle (tier 3)
        ctx.set("arr", array.array("i", [1, 2, 3]))
        return Event("NextEvent")

    @step(accepts=["NextEvent"])
    def getter(ctx: Context, ev: Event):
        ba = ctx.get("ba")
        arr = ctx.get("arr")
        return StopEvent(
            result={
                "ba_is_bytes": isinstance(ba, bytes),
                "ba_data": list(ba),
                "arr_type": type(arr).__name__,
                "arr_data": arr.tolist(),
            }
        )

    wf = Workflow("buffer-not-mangled", [setter, getter])
    handler = await wf.run()
    result = await handler.result()

    assert result.result["ba_is_bytes"] is True
    assert result.result["ba_data"] == [0xAA, 0xBB]
    assert result.result["arr_type"] == "array"
    assert result.result["arr_data"] == [1, 2, 3]
