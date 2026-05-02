/**
 * Smoke tests for the Blazen Pipeline Node bindings.
 *
 * These tests verify the BINDING SURFACE only — they construct the various
 * pipeline classes and assert their shape. They do NOT call `pipeline.start()`
 * or any other async method, because the napi-rs tokio runtime keeps handles
 * alive past Promise resolution and would block Node's test runner from
 * exiting. End-to-end pipeline execution must be exercised from a harness
 * that runs the test in a subprocess with a hard kill timer.
 *
 * Build the native binding first:
 *   cd crates/blazen-node && npm install && npm run build
 */

import test from "ava";

import {
  Pipeline,
  PipelineBuilder,
  Stage,
  ParallelStage,
  JoinStrategy,
  PipelineSnapshot,
  Workflow,
} from "../../crates/blazen-node/index.js";

// =========================================================================
// Class exports
// =========================================================================

test("pipeline class exports · exports Pipeline, PipelineBuilder, Stage, ParallelStage, JoinStrategy classes", (t) => {
  t.is(typeof Pipeline, "function", "Pipeline should be a class");
  t.is(
    typeof PipelineBuilder,
    "function",
    "PipelineBuilder should be a class",
  );
  t.is(typeof Stage, "function", "Stage should be a class");
  t.is(
    typeof ParallelStage,
    "function",
    "ParallelStage should be a class",
  );
  // JoinStrategy is a napi-rs enum — exposed as an object with string members.
  t.truthy(
    typeof JoinStrategy === "object" || typeof JoinStrategy === "function",
    "JoinStrategy should be an enum object",
  );
  t.is(
    typeof PipelineSnapshot,
    "function",
    "PipelineSnapshot should be a class",
  );
});

// =========================================================================
// PipelineBuilder
// =========================================================================

test("PipelineBuilder · constructor accepts a name", (t) => {
  const b = new PipelineBuilder("test");
  t.truthy(b, "PipelineBuilder constructor should return a value");
});

test("PipelineBuilder · build() rejects empty stages", (t) => {
  t.throws(
    () => new PipelineBuilder("empty").build(),
    { message: /stage|empty|at least/i },
    "build() should throw when no stages have been added",
  );
});

// =========================================================================
// Stage
// =========================================================================

test("Stage · constructor builds from a Workflow", (t) => {
  const wf = new Workflow("wf");
  wf.addStep("noop", ["blazen::StartEvent"], (_event, _ctx) => {
    return { type: "blazen::StopEvent", result: {} };
  });

  const s = new Stage("name", wf);
  t.is(s.name, "name", "Stage.name should match constructor arg");
});

// =========================================================================
// ParallelStage
// =========================================================================

test("ParallelStage · constructor accepts branches and join strategy", (t) => {
  const wfA = new Workflow("wf-a");
  wfA.addStep("a", ["blazen::StartEvent"], (_event, _ctx) => {
    return { type: "blazen::StopEvent", result: { branch: "a" } };
  });

  const wfB = new Workflow("wf-b");
  wfB.addStep("b", ["blazen::StartEvent"], (_event, _ctx) => {
    return { type: "blazen::StopEvent", result: { branch: "b" } };
  });

  const s1 = new Stage("a", wfA);
  const s2 = new Stage("b", wfB);

  const parallel = new ParallelStage("p", [s1, s2], JoinStrategy.WaitAll);
  t.is(
    parallel.name,
    "p",
    "ParallelStage.name should match constructor arg",
  );
});

// =========================================================================
// JoinStrategy enum
// =========================================================================

test("JoinStrategy · enum has WaitAll and FirstCompletes", (t) => {
  t.truthy(
    "WaitAll" in JoinStrategy,
    "JoinStrategy should expose a WaitAll member",
  );
  t.truthy(
    "FirstCompletes" in JoinStrategy,
    "JoinStrategy should expose a FirstCompletes member",
  );
});

// =========================================================================
// PipelineSnapshot
// =========================================================================

test("PipelineSnapshot · fromJson rejects invalid JSON", (t) => {
  t.is(
    typeof PipelineSnapshot.fromJson,
    "function",
    "PipelineSnapshot.fromJson should be a static method",
  );
  t.throws(
    () => PipelineSnapshot.fromJson("not json"),
    { message: /./ },
    "fromJson should throw on invalid JSON",
  );
});

// =========================================================================
// Pipeline construction via PipelineBuilder
// =========================================================================

test("Pipeline · class is constructed via PipelineBuilder.build()", (t) => {
  const wf = new Workflow("only-stage");
  wf.addStep("only", ["blazen::StartEvent"], (_event, _ctx) => {
    return { type: "blazen::StopEvent", result: {} };
  });

  const stage = new Stage("only", wf);
  const pipeline = new PipelineBuilder("one-stage").stage(stage).build();

  t.is(
    typeof pipeline,
    "object",
    "PipelineBuilder.build() should return an object",
  );
  t.truthy(pipeline instanceof Pipeline, "build() should yield a Pipeline");
  t.is(
    typeof pipeline.start,
    "function",
    "Pipeline should expose a .start() method",
  );
});
