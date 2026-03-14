pub const TEMPLATE: &str = r#"# Blazen Pipeline — TypeScript / Node.js Usage Guide

## Installation

```bash
npm install blazen --registry https://forge.blackleafdigital.com/api/packages/BlackLeafDigital/npm/
```

## Core Concepts

- **Pipeline** — An ordered sequence of stages, each wrapping a Workflow.
- **Stage** — A sequential step running a single workflow with optional input mapping and conditions.
- **ParallelStage** — Multiple workflow branches running concurrently.
- **Persist callback** — Called after each stage with a serializable snapshot.

## Quick Start — Two-Stage Sequential Pipeline

```typescript
import { Workflow, Event, Context, Pipeline, Stage } from "blazen";

// Stage 1: Extract keywords
const extractWf = new Workflow("extract");
extractWf.addStep(
  "extract",
  ["blazen::StartEvent"],
  async (event: Event, ctx: Context): Promise<Event> => {
    const text: string = event.text ?? "";
    const keywords = text.split(" ").filter((w: string) => w.length > 4);
    return { type: "blazen::StopEvent", result: { keywords } };
  }
);

// Stage 2: Summarise
const summariseWf = new Workflow("summarise");
summariseWf.addStep(
  "summarise",
  ["blazen::StartEvent"],
  async (event: Event, ctx: Context): Promise<Event> => {
    const keywords: string[] = event.keywords ?? [];
    return {
      type: "blazen::StopEvent",
      result: { summary: `Found ${keywords.length} keywords` },
    };
  }
);

const pipeline = new Pipeline("text-pipeline")
  .stage(new Stage("extract", extractWf))
  .stage(new Stage("summarise", summariseWf))
  .build();

const result = await pipeline.run({
  text: "Blazen provides powerful workflow orchestration for Node applications",
});

console.log("Output:", result.finalOutput);
for (const stage of result.stageResults) {
  console.log(`  Stage '${stage.name}': ${JSON.stringify(stage.output)} (${stage.durationMs}ms)`);
}
```

## Parallel Stages

Run multiple workflows concurrently and collect all results:

```typescript
const pipeline = new Pipeline("analyze-pipeline")
  .stage(new Stage("preprocess", preprocessWf))
  .parallel(new ParallelStage("analyze", {
    branches: [
      new Stage("sentiment", sentimentWf),
      new Stage("entities", entitiesWf),
    ],
    joinStrategy: "waitAll", // or "firstCompletes"
  }))
  .build();
```

## Conditional Stages & Input Mapping

```typescript
new Stage("optional-step", optionalWf, {
  inputMapper: (state) => ({
    data: state.stageResult("extract"),
  }),
  condition: (state) => {
    const result = state.stageResult("extract");
    return result?.keywords?.length > 0;
  },
});
```

## Persistence Callback

Save a snapshot after each stage for crash recovery:

```typescript
import { writeFileSync } from "fs";

const pipeline = new Pipeline("durable-pipeline")
  .stage(stageA)
  .stage(stageB)
  .onPersistJson(async (json: string) => {
    writeFileSync("pipeline_snapshot.json", json);
    console.log(`Snapshot saved (${json.length} bytes)`);
  })
  .build();
```

## Streaming Events

```typescript
const handler = await pipeline.runWithHandler({ text: "hello world" });

handler.streamEvents((event) => {
  console.log(`[${event.stageName}]`, event.event);
});

const result = await handler.result();
```

## Pause & Resume

```typescript
import { readFileSync, writeFileSync } from "fs";

const handler = await pipeline.runWithHandler({ text: "hello" });
const snapshot = await handler.pause();
writeFileSync("paused.json", snapshot);

// Later: resume
const saved = readFileSync("paused.json", "utf-8");
const resumed = await rebuiltPipeline.resume(saved);
const result = await resumed.result();
```
"#;
