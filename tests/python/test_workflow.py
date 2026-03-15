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
