/**
 * Smoke tests for the Blazen control-plane Node.js bindings.
 *
 * Covers:
 *   1. Importing the new classes / types succeeds.
 *   2. `ControlPlaneWorkerConfig` constructor + every builder method
 *      returns the same instance so chaining works.
 *   3. `ControlPlaneClient.connect` rejects cleanly against an
 *      unreachable endpoint.
 *
 * Full end-to-end coverage (worker + server + assignment dispatch) lives
 * in the Rust integration tests under
 * `crates/blazen-controlplane/tests/`; this file just verifies the FFI
 * surface is wired correctly.
 *
 * Uses the ava test runner. Build the native binding first:
 *   pnpm --filter blazen run build
 */

import test from "ava";

import {
  AssignmentContext,
  ControlPlaneClient,
  ControlPlaneWorker,
  ControlPlaneWorkerConfig,
} from "../../crates/blazen-node/index.js";

// ==========================================================================
// 1. Class imports
// ==========================================================================

test("control plane · all expected classes are exported", (t) => {
  t.is(typeof ControlPlaneClient, "function");
  t.is(typeof ControlPlaneWorker, "function");
  t.is(typeof ControlPlaneWorkerConfig, "function");
  t.is(typeof AssignmentContext, "function");
});

// ==========================================================================
// 2. ControlPlaneWorkerConfig builder
// ==========================================================================

test("ControlPlaneWorkerConfig · constructor accepts endpoint + nodeId", (t) => {
  const cfg = new ControlPlaneWorkerConfig("http://cp.local:7445", "node-test");
  t.truthy(cfg);
  t.true(cfg instanceof ControlPlaneWorkerConfig);
});

test("ControlPlaneWorkerConfig · builder methods return `this` for chaining", (t) => {
  const cfg = new ControlPlaneWorkerConfig("http://cp.local:7445", "node-test");

  const withCap = cfg.withCapability({ kind: "workflow:summarize", version: 1 });
  t.is(withCap, cfg, "withCapability returns the same instance");

  const withTag = cfg.withTag("region", "us-west");
  t.is(withTag, cfg, "withTag returns the same instance");

  const withAdm = cfg.withAdmission({ type: "Fixed", maxInFlight: 4 });
  t.is(withAdm, cfg, "withAdmission returns the same instance");

  const withHb = cfg.withHeartbeatIntervalMs(2_500);
  t.is(withHb, cfg, "withHeartbeatIntervalMs returns the same instance");
});

test("ControlPlaneWorkerConfig · supports all three admission modes", (t) => {
  // Fixed
  const fixed = new ControlPlaneWorkerConfig("http://cp:7445", "n1");
  t.notThrows(() => fixed.withAdmission({ type: "Fixed", maxInFlight: 8 }));

  // Reactive
  const reactive = new ControlPlaneWorkerConfig("http://cp:7445", "n2");
  t.notThrows(() => reactive.withAdmission({ type: "Reactive" }));

  // VramBudget — totalMb is a BigInt on the napi side.
  const vram = new ControlPlaneWorkerConfig("http://cp:7445", "n3");
  t.notThrows(() =>
    vram.withAdmission({ type: "VramBudget", totalMb: 24_000n })
  );
});

test("ControlPlaneWorkerConfig · admission mode requires payload for Fixed/VramBudget", (t) => {
  const fixed = new ControlPlaneWorkerConfig("http://cp:7445", "n1");
  t.throws(() => fixed.withAdmission({ type: "Fixed" }), {
    message: /maxInFlight/,
  });

  const vram = new ControlPlaneWorkerConfig("http://cp:7445", "n2");
  t.throws(() => vram.withAdmission({ type: "VramBudget" }), {
    message: /totalMb/,
  });
});

test("ControlPlaneWorkerConfig · rejects invalid endpoint URI on `connect`", (t) => {
  const cfg = new ControlPlaneWorkerConfig("not a uri", "node-test");
  t.throws(() => ControlPlaneWorker.connect(cfg), {
    message: /ControlPlaneTransportError/,
  });
});

// ==========================================================================
// 3. ControlPlaneClient.connect against unreachable endpoint
// ==========================================================================

test("ControlPlaneClient.connect · rejects cleanly against unreachable endpoint", async (t) => {
  // Port 1 (tcpmux) is reserved and won't accept TCP on any normal host;
  // pick a target the IANA reserved-ports range guarantees no listener for.
  await t.throwsAsync(
    ControlPlaneClient.connect("http://127.0.0.1:1"),
    {
      message: /ControlPlane(Transport|Tls)Error/,
    },
  );
});

test("ControlPlaneClient.connect · surfaces an error for malformed URIs", async (t) => {
  await t.throwsAsync(
    ControlPlaneClient.connect("not a uri"),
    {
      message: /ControlPlane(Transport|Tls)Error/,
    },
  );
});
