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

import { describe, it } from "node:test";
import assert from "node:assert/strict";

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

describe("pipeline class exports", () => {
  it("exports Pipeline, PipelineBuilder, Stage, ParallelStage, JoinStrategy classes", () => {
    assert.strictEqual(typeof Pipeline, "function", "Pipeline should be a class");
    assert.strictEqual(
      typeof PipelineBuilder,
      "function",
      "PipelineBuilder should be a class",
    );
    assert.strictEqual(typeof Stage, "function", "Stage should be a class");
    assert.strictEqual(
      typeof ParallelStage,
      "function",
      "ParallelStage should be a class",
    );
    // JoinStrategy is a napi-rs enum — exposed as an object with string members.
    assert.ok(
      typeof JoinStrategy === "object" || typeof JoinStrategy === "function",
      "JoinStrategy should be an enum object",
    );
    assert.strictEqual(
      typeof PipelineSnapshot,
      "function",
      "PipelineSnapshot should be a class",
    );
  });
});

// =========================================================================
// PipelineBuilder
// =========================================================================

describe("PipelineBuilder", () => {
  it("constructor accepts a name", () => {
    const b = new PipelineBuilder("test");
    assert.ok(b, "PipelineBuilder constructor should return a value");
  });

  it("build() rejects empty stages", () => {
    assert.throws(
      () => new PipelineBuilder("empty").build(),
      /stage|empty|at least/i,
      "build() should throw when no stages have been added",
    );
  });
});

// =========================================================================
// Stage
// =========================================================================

describe("Stage", () => {
  it("constructor builds from a Workflow", () => {
    const wf = new Workflow("wf");
    wf.addStep("noop", ["blazen::StartEvent"], (_event, _ctx) => {
      return { type: "blazen::StopEvent", result: {} };
    });

    const s = new Stage("name", wf);
    assert.strictEqual(s.name, "name", "Stage.name should match constructor arg");
  });
});

// =========================================================================
// ParallelStage
// =========================================================================

describe("ParallelStage", () => {
  it("constructor accepts branches and join strategy", () => {
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
    assert.strictEqual(
      parallel.name,
      "p",
      "ParallelStage.name should match constructor arg",
    );
  });
});

// =========================================================================
// JoinStrategy enum
// =========================================================================

describe("JoinStrategy", () => {
  it("enum has WaitAll and FirstCompletes", () => {
    assert.ok(
      "WaitAll" in JoinStrategy,
      "JoinStrategy should expose a WaitAll member",
    );
    assert.ok(
      "FirstCompletes" in JoinStrategy,
      "JoinStrategy should expose a FirstCompletes member",
    );
  });
});

// =========================================================================
// PipelineSnapshot
// =========================================================================

describe("PipelineSnapshot", () => {
  it("fromJson rejects invalid JSON", () => {
    assert.strictEqual(
      typeof PipelineSnapshot.fromJson,
      "function",
      "PipelineSnapshot.fromJson should be a static method",
    );
    assert.throws(
      () => PipelineSnapshot.fromJson("not json"),
      /./,
      "fromJson should throw on invalid JSON",
    );
  });
});

// =========================================================================
// Pipeline construction via PipelineBuilder
// =========================================================================

describe("Pipeline", () => {
  it("class is constructed via PipelineBuilder.build()", () => {
    const wf = new Workflow("only-stage");
    wf.addStep("only", ["blazen::StartEvent"], (_event, _ctx) => {
      return { type: "blazen::StopEvent", result: {} };
    });

    const stage = new Stage("only", wf);
    const pipeline = new PipelineBuilder("one-stage").stage(stage).build();

    assert.strictEqual(
      typeof pipeline,
      "object",
      "PipelineBuilder.build() should return an object",
    );
    assert.ok(pipeline instanceof Pipeline, "build() should yield a Pipeline");
    assert.strictEqual(
      typeof pipeline.start,
      "function",
      "Pipeline should expose a .start() method",
    );
  });
});
