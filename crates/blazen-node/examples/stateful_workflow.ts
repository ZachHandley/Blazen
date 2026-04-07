/**
 * Stateful Workflow Example
 *
 * Demonstrates the two explicit context namespaces on the Node bindings:
 *   - ctx.state    -- persistable values (survives pause()/resume())
 *   - ctx.session  -- in-process-only values (excluded from snapshots)
 *
 * ---------------------------------------------------------------------------
 * IMPORTANT: Node identity caveat
 * ---------------------------------------------------------------------------
 * On the Node binding, ctx.session values are routed through
 * `serde_json::Value` -- JS object identity is NOT preserved.
 * `await ctx.session.get(key)` returns a plain object equal to the value
 * you passed in, not the same object reference. This is a napi-rs
 * threading limitation: `Reference<T>` is `!Send` because its `Drop`
 * must run on the v8 main thread, so live JS handles cannot be moved
 * into the workflow's Tokio executor.
 *
 * For true identity preservation of live JS objects across steps, use
 * the Python (PyO3) or WASM bindings instead.
 *
 * The session namespace is still functionally distinct from state:
 * session values are excluded from snapshots while state values are not.
 * ---------------------------------------------------------------------------
 *
 * Run with: npx tsx stateful_workflow.ts
 */

import { Workflow } from "blazen";
import type { Context, JsWorkflowResult } from "blazen";

const wf = new Workflow("stateful-example");

// ---------------------------------------------------------------------------
// Step 1: setup -- write to both namespaces.
//
// state: persistable scalar that survives pause()/resume() snapshots.
// session: in-process-only object, NOT snapshotted. Note that the value
// will round-trip through JSON when read back, so identity is lost.
// ---------------------------------------------------------------------------
wf.addStep("setup", ["blazen::StartEvent"], async (event: Record<string, any>, ctx: Context) => {
  // Persistable state -- survives pause/resume.
  await ctx.state.set("row_count_expected", 3);

  // In-process-only values -- excluded from snapshots.
  // Note: identity is NOT preserved; values round-trip through JSON.
  await ctx.session.set("request", { id: "req-abc123", attempt: 1 });

  console.log("[setup] state stored: row_count_expected=3");
  console.log("[setup] session stored: request={id: 'req-abc123', attempt: 1}");
  return { type: "QueryEvent" };
});

// ---------------------------------------------------------------------------
// Step 2: query -- read from both namespaces and prove they are independent.
//
// Demonstrates that state and session are separate keyspaces: writing the
// same key "k" to both namespaces yields two distinct values.
// ---------------------------------------------------------------------------
wf.addStep("query", ["QueryEvent"], async (event: Record<string, any>, ctx: Context) => {
  const expectedCount: number = await ctx.state.get("row_count_expected");
  const request: Record<string, any> = await ctx.session.get("request");
  const hasRequest: boolean = await ctx.session.has("request");

  console.log(`[query] expected_count=${expectedCount}`);
  console.log(`[query] request=${JSON.stringify(request)}, has=${hasRequest}`);

  // Demonstrate state vs session independence: same key in both namespaces.
  await ctx.state.set("k", "state-value");
  await ctx.session.set("k", "session-value");

  return {
    type: "blazen::StopEvent",
    result: {
      expectedCount,
      request,
      hasRequest,
      fromState: await ctx.state.get("k"),
      fromSession: await ctx.session.get("k"),
    },
  };
});

const result: JsWorkflowResult = await wf.run({});

console.log("\nWorkflow finished!");
console.log("  type:", result.type);
console.log("  data:", JSON.stringify(result.data, null, 2));
