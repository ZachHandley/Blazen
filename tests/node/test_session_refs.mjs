/**
 * Regression tests for the Node state/session namespace API and for the
 * live-JS-object passthrough that preserves class-instance identity across
 * workflow step hops.
 *
 * These mirror the Python `test_session_refs.py` parity surface. JS object
 * identity IS preserved through event payloads: the object a step returns
 * is the same object the next matching step receives, with its prototype,
 * methods, and non-JSON fields intact. This is implemented via the unified
 * `SessionRefRegistry` + a `__blazen_native_ref__` marker — the live object
 * stays on the v8 heap (napi's `Reference<T>` is `!Send`), and only the
 * registry-key UUID crosses into Rust (see
 * `crates/blazen-node/src/workflow/native_passthrough.rs`).
 *
 * The session namespace also has a *functional* distinction from state:
 * session values are excluded from snapshots, state values are not.
 */

import test from "ava";

import { Workflow } from "../../crates/blazen-node/index.js";

// =========================================================================
// Live-JS-object identity passthrough across step hops
// =========================================================================

test("identity passthrough · class instance survives a step hop with identity, prototype, and mutations", async (t) => {
  class Sentinel {
    constructor(v) {
      this.v = v;
      this.touched = [];
    }
    greet() {
      return `hi ${this.v}`;
    }
  }

  let firstObj = null;
  const wf = new Workflow("identity");

  wf.addStep("make", ["blazen::StartEvent"], async (event) => {
    const s = new Sentinel(event.who);
    s.touched.push("make");
    firstObj = s;
    return { type: "MidEvent", payload: s };
  });

  wf.addStep("consume", ["MidEvent"], async (event) => {
    // The nested live object is the SAME instance the previous step made.
    event.payload.touched.push("consume");
    return {
      type: "blazen::StopEvent",
      result: {
        sameInstance: event.payload === firstObj,
        isSentinel: event.payload instanceof Sentinel,
        greet: event.payload.greet(),
        touched: event.payload.touched,
      },
    };
  });

  const result = await wf.run({ who: "world" });
  t.is(result.data.sameInstance, true);
  t.is(result.data.isSentinel, true);
  t.is(result.data.greet, "hi world");
  t.deepEqual(result.data.touched, ["make", "consume"]);
  // The mutation made in the consuming step is visible on the original.
  t.deepEqual(firstObj.touched, ["make", "consume"]);
});

test("identity passthrough · fan-out preserves identity for each event, including own-property methods", async (t) => {
  const wf = new Workflow("fan");
  wf.addStep("fan", ["blazen::StartEvent"], async () => {
    // `live.method` is an OWN property (a function) — not JSON-serializable.
    // The live object must still survive even though its JSON projection
    // cannot represent the function.
    const a = { type: "T", tag: "a", live: { method: () => 1 } };
    const b = { type: "T", tag: "b", live: { method: () => 2 } };
    return [a, b];
  });

  const seen = [];
  wf.addStep("collect", ["T"], async (event) => {
    seen.push(event.live.method());
    if (seen.length === 2) {
      return {
        type: "blazen::StopEvent",
        result: { seen: seen.slice().sort((x, y) => x - y) },
      };
    }
    return null;
  });

  const result = await wf.run({});
  t.deepEqual(result.data.seen, [1, 2]);
});

test("identity passthrough · internal native-ref marker never leaks into events or results", async (t) => {
  const wf = new Workflow("leak");
  let midKeys = null;
  let midHadMarker = null;

  wf.addStep("a", ["blazen::StartEvent"], async (event) => {
    return { type: "Mid", n: event.n + 1 };
  });
  wf.addStep("b", ["Mid"], async (event) => {
    midKeys = Object.keys(event);
    midHadMarker = Object.prototype.hasOwnProperty.call(
      event,
      "__blazen_native_ref__",
    );
    return { type: "blazen::StopEvent", result: { n: event.n } };
  });

  const result = await wf.run({ n: 1 });
  t.is(result.data.n, 2);
  t.is(midHadMarker, false);
  t.deepEqual(midKeys.sort(), ["n", "type"]);
  // The marker must not surface on the result payload either.
  t.is(
    Object.prototype.hasOwnProperty.call(result.data, "__blazen_native_ref__"),
    false,
  );
});

// =========================================================================
// ctx.state namespace
// =========================================================================

test("ctx.state namespace · stores and retrieves JSON values", async (t) => {
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
  t.is(result.data.counter, 5);
  t.is(result.data.name, "alice");
  t.deepEqual(result.data.nested, { a: 1, b: [2, 3] });
});

test("ctx.state namespace · stores and retrieves Buffer bytes", async (t) => {
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
  t.deepEqual(result.data.bytes, [0xde, 0xad, 0xbe, 0xef]);
});

test("ctx.state namespace · returns null for missing keys", async (t) => {
  const wf = new Workflow("state-missing");
  wf.addStep("s", ["blazen::StartEvent"], async (event, ctx) => {
    return {
      type: "blazen::StopEvent",
      result: { missing: await ctx.state.get("nope") },
    };
  });

  const result = await wf.run({});
  t.is(result.data.missing, null);
});

// =========================================================================
// ctx.session namespace
// =========================================================================

test("ctx.session namespace · stores and retrieves values", async (t) => {
  const wf = new Workflow("session-basic");
  wf.addStep("s", ["blazen::StartEvent"], async (event, ctx) => {
    await ctx.session.set("tmp", { tag: "ephemeral", n: 42 });
    const value = await ctx.session.get("tmp");
    const has = await ctx.session.has("tmp");
    return { type: "blazen::StopEvent", result: { value, has } };
  });

  const result = await wf.run({});
  t.deepEqual(result.data.value, { tag: "ephemeral", n: 42 });
  t.is(result.data.has, true);
});

test("ctx.session namespace · returns null for missing keys", async (t) => {
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
  t.is(result.data.value, null);
  t.is(result.data.has, false);
});

test("ctx.session namespace · remove() drops the key", async (t) => {
  const wf = new Workflow("session-remove");
  wf.addStep("s", ["blazen::StartEvent"], async (event, ctx) => {
    await ctx.session.set("tmp", 1);
    const before = await ctx.session.has("tmp");
    await ctx.session.remove("tmp");
    const after = await ctx.session.has("tmp");
    return { type: "blazen::StopEvent", result: { before, after } };
  });

  const result = await wf.run({});
  t.is(result.data.before, true);
  t.is(result.data.after, false);
});

// =========================================================================
// state vs session independence
// =========================================================================

test("state vs session independence · state and session use separate keyspaces", async (t) => {
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
  t.is(result.data.fromState, "state-value");
  t.is(result.data.fromSession, "session-value");
});

test("state vs session independence · session values survive across steps within a run", async (t) => {
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
  t.deepEqual(result.data.shared, { step: "producer", value: 100 });
});
