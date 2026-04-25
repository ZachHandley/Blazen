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
    assert.strictEqual(typeof v, "string");
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
    assert.strictEqual(result.type, "blazen::StopEvent");
    assert.strictEqual(result.data.message, "hello e2e");
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
    assert.strictEqual(result.data.text, "hello world");
    assert.strictEqual(result.data.length, 11);
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
    assert.strictEqual(result.data.counter, 42);
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
    assert.strictEqual(typeof rid, "string");
    assert.ok(rid.length > 0);
    // UUID format: 8-4-4-4-12
    assert.strictEqual(rid.split("-").length, 5);
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

    assert.strictEqual(result.data.done, true);

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
    assert.strictEqual(result.data.processed, true);
  });
});

// =========================================================================
// setBytes / getBytes roundtrip
// =========================================================================

describe("setBytes / getBytes roundtrip", () => {
  it("stores and retrieves binary data as Buffer", async () => {
    const wf = new Workflow("bytes-roundtrip");

    wf.addStep("bytes", ["blazen::StartEvent"], async (event, ctx) => {
      await ctx.setBytes("bin", Buffer.from([0xde, 0xad, 0xbe, 0xef]));
      const buf = await ctx.getBytes("bin");
      return {
        type: "blazen::StopEvent",
        result: { bytes: Array.from(buf) },
      };
    });

    const result = await wf.run({});
    assert.deepEqual(result.data.bytes, [0xde, 0xad, 0xbe, 0xef]);
  });
});

// =========================================================================
// get() on bytes key returns array
// =========================================================================

describe("get() on bytes key returns array", () => {
  it("returns a JSON array of numbers instead of null", async () => {
    const wf = new Workflow("bytes-get");

    wf.addStep("bytes", ["blazen::StartEvent"], async (event, ctx) => {
      await ctx.setBytes("bin", Buffer.from([0xde, 0xad, 0xbe, 0xef]));
      const val = await ctx.get("bin");
      return {
        type: "blazen::StopEvent",
        result: { val },
      };
    });

    const result = await wf.run({});
    assert.notEqual(result.data.val, null);
    assert.deepEqual(result.data.val, [222, 173, 190, 239]);
  });
});

// =========================================================================
// Complex JSON roundtrip
// =========================================================================

describe("complex JSON roundtrip", () => {
  it("stores and retrieves nested objects, arrays, and mixed types", async () => {
    const wf = new Workflow("complex-json");
    const payload = {
      nested: { arr: [1, 2.5, null, true, "str"], obj: { a: 1 } },
    };

    wf.addStep("complex", ["blazen::StartEvent"], async (event, ctx) => {
      await ctx.set("complex", payload);
      const val = await ctx.get("complex");
      return { type: "blazen::StopEvent", result: { val } };
    });

    const result = await wf.run({});
    assert.deepEqual(result.data.val, payload);
  });
});

// =========================================================================
// Overwrite behavior
// =========================================================================

describe("overwrite behavior", () => {
  it("overwrites a key with a value of a different type", async () => {
    const wf = new Workflow("overwrite");

    wf.addStep("overwrite", ["blazen::StartEvent"], async (event, ctx) => {
      await ctx.set("k", 1);
      await ctx.set("k", "hello");
      const val = await ctx.get("k");
      return { type: "blazen::StopEvent", result: { val } };
    });

    const result = await wf.run({});
    assert.strictEqual(result.data.val, "hello");
  });
});

// =========================================================================
// Null / undefined handling
// =========================================================================

describe("null / undefined handling", () => {
  it("stores null and returns null for missing keys", async () => {
    const wf = new Workflow("null-handling");

    wf.addStep("nulls", ["blazen::StartEvent"], async (event, ctx) => {
      await ctx.set("k", null);
      const stored = await ctx.get("k");
      const missing = await ctx.get("no_such_key");
      return {
        type: "blazen::StopEvent",
        result: { stored, missing },
      };
    });

    const result = await wf.run({});
    assert.strictEqual(result.data.stored, null);
    assert.strictEqual(result.data.missing, null);
  });

  it("returns default when key is missing or null", async () => {
    const wf = new Workflow("get-default");

    wf.addStep("defaults", ["blazen::StartEvent"], async (event, ctx) => {
      await ctx.set("present", 5);
      await ctx.set("nullish", null);
      const buf = Buffer.from([1, 2]);
      return {
        type: "blazen::StopEvent",
        result: {
          missingDefault: await ctx.get("missing", "fb"),
          missingNumber: await ctx.get("missing", 0),
          presentIgnoresDefault: await ctx.get("present", 99),
          nullishUsesDefault: await ctx.get("nullish", "fb"),
          missingNoDefault: await ctx.get("missing"),
          missingBytesDefault: Array.from(
            await ctx.getBytes("missing", buf),
          ),
          stateMissingDefault: await ctx.state.get("missing", "sd"),
          sessionMissingDefault: await ctx.session.get("missing", "ssd"),
        },
      };
    });

    const result = await wf.run({});
    assert.strictEqual(result.data.missingDefault, "fb");
    assert.strictEqual(result.data.missingNumber, 0);
    assert.strictEqual(result.data.presentIgnoresDefault, 5);
    assert.strictEqual(result.data.nullishUsesDefault, "fb");
    assert.strictEqual(result.data.missingNoDefault, null);
    assert.deepStrictEqual(result.data.missingBytesDefault, [1, 2]);
    assert.strictEqual(result.data.stateMissingDefault, "sd");
    assert.strictEqual(result.data.sessionMissingDefault, "ssd");
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
