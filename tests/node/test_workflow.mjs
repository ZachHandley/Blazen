/**
 * E2E tests for the Blazen Node.js bindings.
 *
 * Uses the built-in node:test runner (Node >= 18).
 * Build the native binding first:
 *   cd crates/blazen-node && npm install && npm run build
 */

import { describe, it } from "node:test";
import assert from "node:assert/strict";

import { Workflow, Context, version } from "../../crates/blazen-node/index.js";

// =========================================================================
// Version
// =========================================================================

describe("version", () => {
  it("returns a string", () => {
    const v = version();
    assert.equal(typeof v, "string");
    assert.ok(v.length > 0);
  });
});

// =========================================================================
// Single step echo
// =========================================================================

describe("single step echo", () => {
  it("echoes the input back as the result", async () => {
    const wf = new Workflow("echo");
    wf.addStep("echo", ["blazen::StartEvent"], async (event, ctx) => {
      return { type: "blazen::StopEvent", result: { message: event.message } };
    });

    const result = await wf.run({ message: "hello e2e" });
    assert.equal(result.type, "blazen::StopEvent");
    assert.equal(result.data.message, "hello e2e");
  });
});

// =========================================================================
// Multi-step pipeline
// =========================================================================

describe("multi-step pipeline", () => {
  it("chains two steps via custom event type", async () => {
    const wf = new Workflow("pipeline");

    wf.addStep("analyze", ["blazen::StartEvent"], async (event, ctx) => {
      return {
        type: "AnalyzeEvent",
        text: event.message,
        length: event.message.length,
      };
    });

    wf.addStep("finalize", ["AnalyzeEvent"], async (event, ctx) => {
      return {
        type: "blazen::StopEvent",
        result: { text: event.text, length: event.length },
      };
    });

    const result = await wf.run({ message: "hello world" });
    assert.equal(result.data.text, "hello world");
    assert.equal(result.data.length, 11);
  });
});

// =========================================================================
// Context sharing
// =========================================================================

describe("context sharing", () => {
  it("shares state between steps via context", async () => {
    const wf = new Workflow("ctx-test");

    wf.addStep("setter", ["blazen::StartEvent"], async (event, ctx) => {
      await ctx.set("counter", 42);
      return { type: "NextEvent" };
    });

    wf.addStep("getter", ["NextEvent"], async (event, ctx) => {
      const val = await ctx.get("counter");
      return { type: "blazen::StopEvent", result: { counter: val } };
    });

    const result = await wf.run({});
    assert.equal(result.data.counter, 42);
  });
});

// =========================================================================
// Context run ID
// =========================================================================

describe("context runId", () => {
  it("returns a UUID string", async () => {
    const wf = new Workflow("run-id");

    wf.addStep("capture", ["blazen::StartEvent"], async (event, ctx) => {
      const rid = await ctx.runId();
      return { type: "blazen::StopEvent", result: { run_id: rid } };
    });

    const result = await wf.run({});
    const rid = result.data.run_id;
    assert.equal(typeof rid, "string");
    assert.ok(rid.length > 0);
    // UUID format: 8-4-4-4-12
    assert.equal(rid.split("-").length, 5);
  });
});

// =========================================================================
// Streaming
// =========================================================================

describe("streaming", () => {
  it("receives intermediate events via runStreaming", async () => {
    const wf = new Workflow("stream-test");

    wf.addStep("producer", ["blazen::StartEvent"], async (event, ctx) => {
      for (let i = 0; i < 3; i++) {
        await ctx.writeEventToStream({ type: "Progress", step: i });
      }
      return { type: "blazen::StopEvent", result: { done: true } };
    });

    const collected = [];
    const result = await wf.runStreaming({}, (event) => {
      collected.push(event);
    });

    assert.equal(result.data.done, true);

    const progress = collected.filter((e) => e.type === "Progress");
    assert.ok(
      progress.length > 0,
      `expected at least one Progress event, got ${progress.length}`
    );
  });
});

// =========================================================================
// Step returns array (fan-out)
// =========================================================================

describe("fan-out", () => {
  it("handles step returning an array of events", async () => {
    const wf = new Workflow("fan-out");

    wf.addStep("fan", ["blazen::StartEvent"], async (event, ctx) => {
      return [
        { type: "BranchA", value: "a" },
        { type: "BranchB", value: "b" },
      ];
    });

    wf.addStep("handle_a", ["BranchA"], async (event, ctx) => {
      return { type: "blazen::StopEvent", result: { branch: "a" } };
    });

    wf.addStep("handle_b", ["BranchB"], async (event, ctx) => {
      return { type: "blazen::StopEvent", result: { branch: "b" } };
    });

    const result = await wf.run({});
    assert.ok(
      result.data.branch === "a" || result.data.branch === "b",
      `unexpected branch: ${result.data.branch}`
    );
  });
});

// =========================================================================
// Step returns null (side-effect + sendEvent continuation)
// =========================================================================

describe("step returns null", () => {
  it("continues via ctx.sendEvent when step returns null", async () => {
    const wf = new Workflow("null-test");

    wf.addStep("side_effect", ["blazen::StartEvent"], async (event, ctx) => {
      await ctx.set("processed", true);
      await ctx.sendEvent({ type: "Continue" });
      return null;
    });

    wf.addStep("finisher", ["Continue"], async (event, ctx) => {
      const processed = await ctx.get("processed");
      return { type: "blazen::StopEvent", result: { processed } };
    });

    const result = await wf.run({});
    assert.equal(result.data.processed, true);
  });
});

// =========================================================================
// Timeout
// =========================================================================

describe("timeout", () => {
  it("times out when step takes too long", async () => {
    const wf = new Workflow("timeout-test");
    wf.setTimeout(0.1); // 100ms

    wf.addStep("slow", ["blazen::StartEvent"], async (event, ctx) => {
      await new Promise((r) => setTimeout(r, 10000));
      return { type: "blazen::StopEvent", result: {} };
    });

    await assert.rejects(() => wf.run({}), {
      message: /timeout|timed out/i,
    });
  });
});
