pub const TEMPLATE: &str = r#"# Blazen Pipeline — Python Usage Guide

## Installation

```bash
pip install blazen --index-url https://forge.blackleafdigital.com/api/packages/BlackLeafDigital/pypi/simple/
```

## Core Concepts

- **Pipeline** — An ordered sequence of stages, each wrapping a Workflow.
- **Stage** — A sequential step running a single workflow with optional input mapping and conditions.
- **ParallelStage** — Multiple workflow branches running concurrently.
- **PipelineState** — Shared key/value state that flows between stages.
- **Persist callback** — Called after each stage with a serializable snapshot.

## Quick Start — Two-Stage Sequential Pipeline

```python
import asyncio
from blazen import (
    Workflow, step, Event, StartEvent, StopEvent, Context,
    Pipeline, Stage,
)

# Stage 1: Extract keywords
@step
async def extract(ctx: Context, ev: StartEvent) -> StopEvent:
    text = ev.to_dict().get("text", "")
    keywords = [w for w in text.split() if len(w) > 4]
    return StopEvent(result={"keywords": keywords})

# Stage 2: Summarise
@step(accepts=["blazen::StartEvent"])
async def summarise(ctx: Context, ev: StartEvent) -> StopEvent:
    keywords = ev.to_dict().get("keywords", [])
    return StopEvent(result={"summary": f"Found {len(keywords)} keywords"})

async def main():
    extract_wf = Workflow("extract", [extract])
    summarise_wf = Workflow("summarise", [summarise])

    pipeline = (
        Pipeline("text-pipeline")
        .stage(Stage("extract", extract_wf))
        .stage(Stage("summarise", summarise_wf))
        .build()
    )

    result = await pipeline.run(text="Blazen provides powerful workflow orchestration")
    print("Output:", result.final_output)

    for stage in result.stage_results:
        print(f"  Stage '{stage.name}': {stage.output} ({stage.duration_ms}ms)")

asyncio.run(main())
```

## Parallel Stages

Run multiple workflows concurrently and collect all results:

```python
from blazen import Pipeline, Stage, ParallelStage

pipeline = (
    Pipeline("analyze-pipeline")
    .stage(Stage("preprocess", preprocess_wf))
    .parallel(ParallelStage(
        "analyze",
        branches=[
            Stage("sentiment", sentiment_wf),
            Stage("entities", entities_wf),
        ],
        join_strategy="wait_all",  # or "first_completes"
    ))
    .build()
)
```

## Conditional Stages & Input Mapping

```python
from blazen import Stage

stage = Stage(
    "optional-step",
    optional_wf,
    input_mapper=lambda state: {"data": state.stage_result("extract")},
    condition=lambda state: bool(
        state.stage_result("extract", {}).get("keywords")
    ),
)
```

## Persistence Callback

Save a snapshot after each stage for crash recovery:

```python
import json

async def persist_snapshot(snapshot_json: str):
    with open("pipeline_snapshot.json", "w") as f:
        f.write(snapshot_json)
    print(f"Snapshot saved ({len(snapshot_json)} bytes)")

pipeline = (
    Pipeline("durable-pipeline")
    .stage(stage_a)
    .stage(stage_b)
    .on_persist_json(persist_snapshot)
    .build()
)
```

## Streaming Events

```python
async def main():
    handler = await pipeline.run_with_handler(text="hello world")

    async for event in handler.stream_events():
        print(f"[{event.stage_name}] {event.event.to_dict()}")

    result = await handler.result()
```

## Pause & Resume

```python
import json

handler = await pipeline.run_with_handler(text="hello")
handler.pause()
snapshot = await handler.snapshot()

with open("paused.json", "w") as f:
    f.write(snapshot)

# Later: resume
with open("paused.json") as f:
    snapshot = f.read()

handler = await rebuilt_pipeline.resume(snapshot)
await handler.resume_in_place()
result = await handler.result()
```
"#;
