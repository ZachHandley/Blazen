/**
 * Regression tests for the Node state/session namespace API.
 *
 * These mirror the Python `test_session_refs.py` parity surface, with
 * the important caveat that the Node bindings do NOT currently
 * preserve JS class-instance identity through event payloads or the
 * session namespace — values are routed through `serde_json::Value`
 * because napi-rs's `Reference<T>` is not `Send` (its `Drop` must run
 * on the v8 main thread), so a cross-thread live-ref registry would
 * need a different threading model. The session namespace therefore
 * focuses on its *functional* distinction from state: session values
 * are excluded from snapshots, state values are not.
 */

import { describe, it } from "node:test";
import assert from "node:assert/strict";

import { Workflow } from "../../crates/blazen-node/index.js";

// =========================================================================
// ctx.state namespace
// =========================================================================

describe("ctx.state namespace", () => {
  it("stores and retrieves JSON values", async () => {
    const wf = new Workflow("state-json");
    wf.addStep("s", ["blazen::StartEvent"], async (event, ctx) => {
      await ctx.state.set("counter", 5);
      await ctx.state.set("name", "alice");
      await ctx.state.set("nested", { a: 1, b: [2, 3] });

      return {
        type: "blazen::StopEvent",
        result: {
          counter: await ctx.state.get("counter"),
          name: await ctx.state.get("name"),
          nested: await ctx.state.get("nested"),
        },
      };
    });

    const result = await wf.run({});
    assert.strictEqual(result.data.counter, 5);
    assert.strictEqual(result.data.name, "alice");
    assert.deepEqual(result.data.nested, { a: 1, b: [2, 3] });
  });

  it("stores and retrieves Buffer bytes", async () => {
    const wf = new Workflow("state-bytes");
    wf.addStep("s", ["blazen::StartEvent"], async (event, ctx) => {
      await ctx.state.setBytes("blob", Buffer.from([0xde, 0xad, 0xbe, 0xef]));
      const blob = await ctx.state.getBytes("blob");
      return {
        type: "blazen::StopEvent",
        result: { bytes: Array.from(blob) },
      };
    });

    const result = await wf.run({});
    assert.deepEqual(result.data.bytes, [0xde, 0xad, 0xbe, 0xef]);
  });

  it("returns null for missing keys", async () => {
    const wf = new Workflow("state-missing");
    wf.addStep("s", ["blazen::StartEvent"], async (event, ctx) => {
      return {
        type: "blazen::StopEvent",
        result: { missing: await ctx.state.get("nope") },
      };
    });

    const result = await wf.run({});
    assert.strictEqual(result.data.missing, null);
  });
});

// =========================================================================
// ctx.session namespace
// =========================================================================

describe("ctx.session namespace", () => {
  it("stores and retrieves values", async () => {
    const wf = new Workflow("session-basic");
    wf.addStep("s", ["blazen::StartEvent"], async (event, ctx) => {
      await ctx.session.set("tmp", { tag: "ephemeral", n: 42 });
      const value = await ctx.session.get("tmp");
      const has = await ctx.session.has("tmp");
      return { type: "blazen::StopEvent", result: { value, has } };
    });

    const result = await wf.run({});
    assert.deepEqual(result.data.value, { tag: "ephemeral", n: 42 });
    assert.strictEqual(result.data.has, true);
  });

  it("returns null for missing keys", async () => {
    const wf = new Workflow("session-missing");
    wf.addStep("s", ["blazen::StartEvent"], async (event, ctx) => {
      return {
        type: "blazen::StopEvent",
        result: {
          value: await ctx.session.get("nope"),
          has: await ctx.session.has("nope"),
        },
      };
    });

    const result = await wf.run({});
    assert.strictEqual(result.data.value, null);
    assert.strictEqual(result.data.has, false);
  });

  it("remove() drops the key", async () => {
    const wf = new Workflow("session-remove");
    wf.addStep("s", ["blazen::StartEvent"], async (event, ctx) => {
      await ctx.session.set("tmp", 1);
      const before = await ctx.session.has("tmp");
      await ctx.session.remove("tmp");
      const after = await ctx.session.has("tmp");
      return { type: "blazen::StopEvent", result: { before, after } };
    });

    const result = await wf.run({});
    assert.strictEqual(result.data.before, true);
    assert.strictEqual(result.data.after, false);
  });
});

// =========================================================================
// state vs session independence
// =========================================================================

describe("state vs session independence", () => {
  it("state and session use separate keyspaces", async () => {
    const wf = new Workflow("independence");
    wf.addStep("s", ["blazen::StartEvent"], async (event, ctx) => {
      // Same key in both namespaces should not collide.
      await ctx.state.set("k", "state-value");
      await ctx.session.set("k", "session-value");
      return {
        type: "blazen::StopEvent",
        result: {
          fromState: await ctx.state.get("k"),
          fromSession: await ctx.session.get("k"),
        },
      };
    });

    const result = await wf.run({});
    assert.strictEqual(result.data.fromState, "state-value");
    assert.strictEqual(result.data.fromSession, "session-value");
  });

  it("session values survive across steps within a run", async () => {
    const wf = new Workflow("cross-step");
    wf.addStep("producer", ["blazen::StartEvent"], async (event, ctx) => {
      await ctx.session.set("shared", { step: "producer", value: 100 });
      return { type: "HandoffEvent" };
    });
    wf.addStep("consumer", ["HandoffEvent"], async (event, ctx) => {
      const shared = await ctx.session.get("shared");
      return { type: "blazen::StopEvent", result: { shared } };
    });

    const result = await wf.run({});
    assert.deepEqual(result.data.shared, { step: "producer", value: 100 });
  });
});
