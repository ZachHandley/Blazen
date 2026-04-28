"""Smoke tests for the Blazen Pipeline Python bindings.

Exercises the PyO3 bindings of `blazen-pipeline`: builder validation,
sequential and parallel stages, input mappers, and basic snapshot
reflection. Mirrors the patterns used in `test_workflow.py`.
"""

import pytest

from blazen import (
    Context,
    Event,
    JoinStrategy,
    ParallelStage,
    Pipeline,
    PipelineBuilder,
    PipelineError,
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


# Reference imports kept to assert the symbols are exported even if not
# directly exercised above.
_ = (Event,)
