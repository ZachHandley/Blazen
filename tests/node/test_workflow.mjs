/**
 * E2E tests for the Blazen Node.js bindings.
 *
 * Uses the ava test runner.
 * Build the native binding first:
 *   cd crates/blazen-node && npm install && npm run build
 */

import test from "ava";

import { Workflow, Context, version } from "../../crates/blazen-node/index.js";

// =========================================================================
// Version
// =========================================================================

test("version · returns a string", (t) => {
  const v = version();
  t.is(typeof v, "string");
  t.truthy(v.length > 0);
});

// =========================================================================
// Single step echo
// =========================================================================

test("single step echo · echoes the input back as the result", async (t) => {
  const wf = new Workflow("echo");
  wf.addStep("echo", ["blazen::StartEvent"], async (event, ctx) => {
    return { type: "blazen::StopEvent", result: { message: event.message } };
  });

  const result = await wf.run({ message: "hello e2e" });
  t.is(result.type, "blazen::StopEvent");
  t.is(result.data.message, "hello e2e");
});

// =========================================================================
// Multi-step pipeline
// =========================================================================

test("multi-step pipeline · chains two steps via custom event type", async (t) => {
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
  t.is(result.data.text, "hello world");
  t.is(result.data.length, 11);
});

// =========================================================================
// Context sharing
// =========================================================================

test("context sharing · shares state between steps via context", async (t) => {
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
  t.is(result.data.counter, 42);
});

// =========================================================================
// Context run ID
// =========================================================================

test("context runId · returns a UUID string", async (t) => {
  const wf = new Workflow("run-id");

  wf.addStep("capture", ["blazen::StartEvent"], async (event, ctx) => {
    const rid = await ctx.runId();
    return { type: "blazen::StopEvent", result: { run_id: rid } };
  });

  const result = await wf.run({});
  const rid = result.data.run_id;
  t.is(typeof rid, "string");
  t.truthy(rid.length > 0);
  // UUID format: 8-4-4-4-12
  t.is(rid.split("-").length, 5);
});

// =========================================================================
// Streaming
// =========================================================================

test("streaming · receives intermediate events via runStreaming", async (t) => {
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

  t.is(result.data.done, true);

  const progress = collected.filter((e) => e.type === "Progress");
  t.truthy(
    progress.length > 0,
    `expected at least one Progress event, got ${progress.length}`
  );
});

// =========================================================================
// Step returns array (fan-out)
// =========================================================================

test("fan-out · handles step returning an array of events", async (t) => {
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
  t.truthy(
    result.data.branch === "a" || result.data.branch === "b",
    `unexpected branch: ${result.data.branch}`
  );
});

// =========================================================================
// Step returns null (side-effect + sendEvent continuation)
// =========================================================================

test("step returns null · continues via ctx.sendEvent when step returns null", async (t) => {
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
  t.is(result.data.processed, true);
});

// =========================================================================
// setBytes / getBytes roundtrip
// =========================================================================

test("setBytes / getBytes roundtrip · stores and retrieves binary data as Buffer", async (t) => {
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
  t.deepEqual(result.data.bytes, [0xde, 0xad, 0xbe, 0xef]);
});

// =========================================================================
// get() on bytes key returns array
// =========================================================================

test("get() on bytes key returns array · returns a JSON array of numbers instead of null", async (t) => {
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
  t.not(result.data.val, null);
  t.deepEqual(result.data.val, [222, 173, 190, 239]);
});

// =========================================================================
// Complex JSON roundtrip
// =========================================================================

test("complex JSON roundtrip · stores and retrieves nested objects, arrays, and mixed types", async (t) => {
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
  t.deepEqual(result.data.val, payload);
});

// =========================================================================
// Overwrite behavior
// =========================================================================

test("overwrite behavior · overwrites a key with a value of a different type", async (t) => {
  const wf = new Workflow("overwrite");

  wf.addStep("overwrite", ["blazen::StartEvent"], async (event, ctx) => {
    await ctx.set("k", 1);
    await ctx.set("k", "hello");
    const val = await ctx.get("k");
    return { type: "blazen::StopEvent", result: { val } };
  });

  const result = await wf.run({});
  t.is(result.data.val, "hello");
});

// =========================================================================
// Null / undefined handling
// =========================================================================

test("null / undefined handling · stores null and returns null for missing keys", async (t) => {
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
  t.is(result.data.stored, null);
  t.is(result.data.missing, null);
});

test("null / undefined handling · returns default when key is missing or null", async (t) => {
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
  t.is(result.data.missingDefault, "fb");
  t.is(result.data.missingNumber, 0);
  t.is(result.data.presentIgnoresDefault, 5);
  t.is(result.data.nullishUsesDefault, "fb");
  t.is(result.data.missingNoDefault, null);
  t.deepEqual(result.data.missingBytesDefault, [1, 2]);
  t.is(result.data.stateMissingDefault, "sd");
  t.is(result.data.sessionMissingDefault, "ssd");
});

// =========================================================================
// Timeout
// =========================================================================

test("timeout · times out when step takes too long", async (t) => {
  const wf = new Workflow("timeout-test");
  wf.setTimeout(0.1); // 100ms

  wf.addStep("slow", ["blazen::StartEvent"], async (event, ctx) => {
    await new Promise((r) => setTimeout(r, 10000));
    return { type: "blazen::StopEvent", result: {} };
  });

  await t.throwsAsync(() => wf.run({}), {
    message: /timeout|timed out/i,
  });
});
