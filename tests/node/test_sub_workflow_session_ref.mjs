/**
 * Regression test for sub-workflow session-ref plumbing (Phase 0.8).
 *
 * Phase 0.6 added `run_with_optional_parent_registry` in
 * `crates/blazen-node/src/workflow/workflow.rs`, which checks
 * `blazen_core::session_ref::current_session_registry()` before
 * invoking a child workflow. If a parent registry is in scope, the
 * child receives the parent's `Arc<SessionRefRegistry>` instead of
 * creating a fresh one.
 *
 * However, the Phase 0.6 fix is **necessary but not yet sufficient**
 * for end-to-end session sharing between parent and child workflows on
 * the Node side. Two additional pieces of plumbing are needed before
 * the session-ref handoff becomes observable from JavaScript:
 *
 *   (a) The Node step dispatcher in `make_step_registration`
 *       (`workflow.rs:342`) does NOT currently wrap the step-handler
 *       future in `with_session_registry(ctx.session_refs_arc(), ...)`
 *       the way the Python dispatcher does (`blazen-py/step.rs:130`).
 *       Without this, `current_session_registry()` returns `None`
 *       inside the parent step, so `run_with_optional_parent_registry`
 *       always takes the fresh-registry path for the child.
 *
 *   (b) `ctx.session.set/get` on the Node side uses
 *       `ContextInner.objects` (a per-context HashMap created fresh at
 *       `Context::new_with_session_refs`), NOT the shared
 *       `SessionRefRegistry`. So even if the registry Arc is shared
 *       between parent and child, the `objects` storage is still
 *       isolated per run.
 *
 * These tests document the **current behavior** as regression guards.
 * Once (a) and (b) are fixed in a follow-up phase, the assertion in
 * the first test should be updated from `null` to `"the-secret-123"`.
 *
 * Scope constraint: the napi-rs bridge routes step-handler arguments
 * through `serde_json::Value`, so full JS object identity across the
 * Rust<->JS boundary is NOT tested here (Phase 13).
 */

import { describe, it } from "node:test";
import assert from "node:assert/strict";

import { Workflow } from "../../crates/blazen-node/index.js";

// ---------------------------------------------------------------------------
// Sub-workflow session ref handoff
// ---------------------------------------------------------------------------

describe("sub-workflow session ref handoff", () => {
  it("parent invokes child sub-workflow from inside a step", async () => {
    // Inner workflow: read `parent_token` from its session namespace and
    // echo it out in the StopEvent result.
    const innerWf = new Workflow("inner");
    innerWf.addStep("read", ["blazen::StartEvent"], async (_event, ctx) => {
      const token = await ctx.session.get("parent_token");
      return {
        type: "blazen::StopEvent",
        result: { token },
      };
    });

    // Outer workflow: write `parent_token` into its session namespace,
    // await the inner run, and return whatever the inner run reported.
    const outerWf = new Workflow("outer");
    outerWf.addStep("wrap", ["blazen::StartEvent"], async (_event, ctx) => {
      await ctx.session.set("parent_token", "the-secret-123");
      const inner = await innerWf.run({});
      return {
        type: "blazen::StopEvent",
        result: { innerToken: inner.data.token },
      };
    });

    const result = await outerWf.run({});

    // Today: child gets its own context with an empty `objects` map, so
    // the token is null. This assertion documents the current behavior.
    //
    // TARGET (after follow-up fix):
    //   assert.strictEqual(result.data.innerToken, "the-secret-123");
    //
    // When the step dispatcher wraps handlers in with_session_registry
    // AND ctx.session routes through the shared registry (instead of
    // per-context objects), flip this assertion to "the-secret-123".
    assert.strictEqual(
      result.data.innerToken,
      null,
      "child workflow currently gets its own empty session namespace " +
        "because (a) the Node step dispatcher does not install " +
        "with_session_registry around the handler, and (b) ctx.session " +
        "uses per-context ContextInner.objects, not the shared " +
        "SessionRefRegistry. See test file header for details.",
    );
  });

  it("child sub-workflow returns JSON data through nested run", async () => {
    // This is the minimal proof that `run_with_optional_parent_registry`
    // is on the code path and does NOT break nested workflow execution.
    // The child returns a JSON payload that the parent wraps in its own
    // StopEvent. The round-trip works regardless of which branch the
    // helper takes (fresh vs shared registry).
    const innerWf = new Workflow("inner-json");
    innerWf.addStep("echo", ["blazen::StartEvent"], async (event, _ctx) => {
      return {
        type: "blazen::StopEvent",
        result: { echoed: event.message, source: "child" },
      };
    });

    const outerWf = new Workflow("outer-json");
    outerWf.addStep("wrap", ["blazen::StartEvent"], async (_event, ctx) => {
      const inner = await innerWf.run({ message: "from-parent" });
      return {
        type: "blazen::StopEvent",
        result: { childResult: inner.data, source: "parent" },
      };
    });

    const result = await outerWf.run({});

    assert.strictEqual(result.data.source, "parent");
    assert.strictEqual(result.data.childResult.echoed, "from-parent");
    assert.strictEqual(result.data.childResult.source, "child");
  });

  // ---------------------------------------------------------------------
  // Top-level isolation sanity check
  // ---------------------------------------------------------------------
  it("fresh top-level run sees an empty session namespace", async () => {
    const innerWf = new Workflow("inner-isolated");
    innerWf.addStep("read", ["blazen::StartEvent"], async (_event, ctx) => {
      const token = await ctx.session.get("parent_token");
      const has = await ctx.session.has("parent_token");
      return {
        type: "blazen::StopEvent",
        result: { token, has },
      };
    });

    // Run at top level (not inside a parent step). No ambient registry.
    const result = await innerWf.run({});

    assert.strictEqual(
      result.data.token,
      null,
      "top-level run must not leak session state.",
    );
    assert.strictEqual(result.data.has, false);
  });
});
