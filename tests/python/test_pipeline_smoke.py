"""Smoke tests for the Blazen Pipeline Python bindings.

Exercises the PyO3 bindings of `blazen-pipeline`: builder validation,
sequential and parallel stages, input mappers, and basic snapshot
reflection. Mirrors the patterns used in `test_workflow.py`.
"""

import asyncio

import pytest

from blazen import (
    Context,
    Event,
    JoinStrategy,
    ParallelStage,
    Pipeline,
    PipelineBuilder,
    PipelineError,
    PipelineEvent,
    PipelineHandler,
    PipelineResult,
    PipelineSnapshot,
    Stage,
    StageResult,
    StartEvent,
    StopEvent,
    Workflow,
    step,
)


# =========================================================================
# Two-stage sequential pipeline
# =========================================================================


@pytest.mark.asyncio
async def test_two_stage_sequential_pipeline():
    @step
    async def first_step(ctx: Context, ev: StartEvent) -> StopEvent:
        data = ev.to_dict()
        return StopEvent(result={"first": int(data.get("input", 0)) + 1})

    @step
    async def second_step(ctx: Context, ev: StartEvent) -> StopEvent:
        data = ev.to_dict()
        return StopEvent(result={"second": int(data.get("first", 0)) * 2})

    wf1 = Workflow("first-wf", [first_step])
    wf2 = Workflow("second-wf", [second_step])

    stage1 = Stage(name="ingest", workflow=wf1)
    stage2 = Stage(name="enrich", workflow=wf2)

    pipeline = (
        PipelineBuilder("etl")
        .stage(stage1)
        .stage(stage2)
        .build()
    )
    assert isinstance(pipeline, Pipeline)

    handler = await pipeline.start(input=10)
    assert isinstance(handler, PipelineHandler)

    result = await handler.result()
    assert isinstance(result, PipelineResult)
    assert result.pipeline_name == "etl"

    stage_results = result.stage_results
    assert len(stage_results) == 2
    for sr in stage_results:
        assert isinstance(sr, StageResult)
        assert sr.skipped is False

    assert stage_results[0].name == "ingest"
    assert stage_results[1].name == "enrich"
    assert result.final_output == {"second": 22}


# =========================================================================
# Builder validation: empty pipeline
# =========================================================================


def test_pipeline_builder_validation_empty_stages():
    builder = PipelineBuilder("empty")
    with pytest.raises(PipelineError):
        builder.build()


# =========================================================================
# PipelineSnapshot class surface
# =========================================================================


def test_pipeline_snapshot_class_surface():
    assert hasattr(PipelineSnapshot, "from_json")
    assert callable(PipelineSnapshot.from_json)

    with pytest.raises(Exception):
        PipelineSnapshot.from_json("not valid json")


# =========================================================================
# Parallel stage with WaitAll
# =========================================================================


@pytest.mark.asyncio
async def test_parallel_stage():
    @step
    async def left(ctx: Context, ev: StartEvent) -> StopEvent:
        return StopEvent(result={"side": "left"})

    @step
    async def right(ctx: Context, ev: StartEvent) -> StopEvent:
        return StopEvent(result={"side": "right"})

    wf_left = Workflow("left-wf", [left])
    wf_right = Workflow("right-wf", [right])

    branch_left = Stage(name="left-branch", workflow=wf_left)
    branch_right = Stage(name="right-branch", workflow=wf_right)

    parallel = ParallelStage(
        name="fanout",
        branches=[branch_left, branch_right],
        join_strategy=JoinStrategy.WaitAll,
    )

    pipeline = PipelineBuilder("parallel-pipe").parallel(parallel).build()

    handler = await pipeline.start()
    result = await handler.result()

    assert isinstance(result, PipelineResult)
    assert len(result.stage_results) == 1

    final = result.final_output
    assert isinstance(final, dict)
    assert set(final.keys()) == {"left-branch", "right-branch"}
    assert final["left-branch"] == {"side": "left"}
    assert final["right-branch"] == {"side": "right"}


# =========================================================================
# Input mapper transforms state for the next stage
# =========================================================================


@pytest.mark.asyncio
async def test_input_mapper():
    @step
    async def producer(ctx: Context, ev: StartEvent) -> StopEvent:
        data = ev.to_dict()
        return StopEvent(result={"first": int(data.get("input", 0)) + 1})

    @step
    async def consumer(ctx: Context, ev: StartEvent) -> StopEvent:
        data = ev.to_dict()
        return StopEvent(result={"transformed": data.get("transformed")})

    wf_producer = Workflow("producer-wf", [producer])
    wf_consumer = Workflow("consumer-wf", [consumer])

    stage1 = Stage(name="produce", workflow=wf_producer)
    stage2 = Stage(
        name="consume",
        workflow=wf_consumer,
        input_mapper=lambda state: {
            "transformed": state.last_result()["first"] * 100
        },
    )

    pipeline = (
        PipelineBuilder("mapper-pipe")
        .stage(stage1)
        .stage(stage2)
        .build()
    )

    handler = await pipeline.start(input=4)
    result = await handler.result()

    assert result.final_output == {"transformed": 500}
    assert len(result.stage_results) == 2
    assert result.stage_results[0].output == {"first": 5}
    assert result.stage_results[1].output == {"transformed": 500}


# =========================================================================
# Streaming events from a running pipeline
# =========================================================================


@pytest.mark.asyncio
async def test_pipeline_stream_events():
    """Verify pipeline.start() exposes stream_events() yielding PipelineEvent items."""

    # First stage emits a couple of intermediate progress events before stopping.
    @step
    async def emitter(ctx: Context, ev: StartEvent) -> StopEvent:
        for i in range(3):
            ctx.write_event_to_stream(Event("Progress", step=i))
        return StopEvent(result={"emitted": 3})

    # Second stage just forwards a derived value so the pipeline has 2 stages.
    @step
    async def forwarder(ctx: Context, ev: StartEvent) -> StopEvent:
        data = ev.to_dict()
        return StopEvent(result={"forwarded": int(data.get("emitted", 0)) * 2})

    wf_emit = Workflow("emit-wf", [emitter])
    wf_fwd = Workflow("fwd-wf", [forwarder])

    pipeline = (
        PipelineBuilder("stream-pipe")
        .stage(Stage(name="emit", workflow=wf_emit))
        .stage(Stage(name="forward", workflow=wf_fwd))
        .build()
    )

    handler = await pipeline.start()
    assert isinstance(handler, PipelineHandler)

    # Drain the stream concurrently with awaiting the final result so the
    # pre-stream buffer doesn't backpressure the pipeline.
    events: list[PipelineEvent] = []

    async def collect() -> None:
        async for event in handler.stream_events():
            events.append(event)

    collector = asyncio.create_task(collect())
    result = await handler.result()
    await collector

    assert isinstance(result, PipelineResult)
    assert result.final_output == {"forwarded": 6}

    # The stream should have surfaced multiple events tagged with stage
    # provenance. We expect at least the 3 Progress events from `emit`.
    assert len(events) >= 2
    for ev in events:
        assert isinstance(ev, PipelineEvent)
        assert ev.stage_name in {"emit", "forward"}

    progress = [
        ev for ev in events
        if ev.stage_name == "emit" and ev.event.event_type == "Progress"
    ]
    assert len(progress) >= 1


# =========================================================================
# Pause mid-pipeline, resume from snapshot
# =========================================================================


@pytest.mark.asyncio
async def test_pipeline_pause_resume():
    """Pause a running pipeline mid-stage-2, then resume from the snapshot."""

    # Stage 1 is instant. Stage 2 sleeps so we have a deterministic window
    # to issue a pause while it is still running.
    @step
    async def fast_step(ctx: Context, ev: StartEvent) -> StopEvent:
        data = ev.to_dict()
        return StopEvent(result={"seed": int(data.get("input", 0)) + 1})

    @step
    async def slow_step(ctx: Context, ev: StartEvent) -> StopEvent:
        data = ev.to_dict()
        await asyncio.sleep(2.0)
        return StopEvent(result={"final": int(data.get("seed", 0)) * 10})

    def build_pipeline() -> Pipeline:
        wf_fast = Workflow("fast-wf", [fast_step])
        wf_slow = Workflow("slow-wf", [slow_step])
        return (
            PipelineBuilder("pause-pipe")
            .stage(Stage(name="fast", workflow=wf_fast))
            .stage(Stage(name="slow", workflow=wf_slow))
            .build()
        )

    # Reference run: end-to-end without pausing, used as the oracle.
    reference = build_pipeline()
    ref_handler = await reference.start(input=4)
    ref_result = await ref_handler.result()
    assert ref_result.final_output == {"final": 50}

    # Pause/resume run: build a fresh pipeline, start it, let stage-1
    # complete and stage-2 begin, then pause and snapshot.
    pipeline = build_pipeline()
    handler = await pipeline.start(input=4)
    # Give stage-1 (instant) time to finish and stage-2 (sleeping) to be
    # in flight when we issue pause.
    await asyncio.sleep(0.3)
    snapshot = await handler.pause()
    assert isinstance(snapshot, PipelineSnapshot)
    assert snapshot.pipeline_name == "pause-pipe"
    # Stage-1 should already be in completed_stages.
    assert len(snapshot.completed_stages) >= 1
    assert snapshot.completed_stages[0].name == "fast"

    # Snapshot JSON should round-trip through PipelineSnapshot.from_json.
    snap_json = snapshot.to_json()
    restored = PipelineSnapshot.from_json(snap_json)
    assert restored.pipeline_name == "pause-pipe"

    # Resume on a fresh pipeline with the same shape.
    resume_pipeline = build_pipeline()
    resumed_handler = await resume_pipeline.resume(snapshot)
    assert isinstance(resumed_handler, PipelineHandler)
    resumed_result = await resumed_handler.result()

    assert isinstance(resumed_result, PipelineResult)
    assert resumed_result.final_output == ref_result.final_output
    assert len(resumed_result.stage_results) == len(ref_result.stage_results)


# Reference imports kept to assert the symbols are exported even if not
# directly exercised above.
_ = (Event,)
