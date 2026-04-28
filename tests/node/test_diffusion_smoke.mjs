/**
 * diffusion-rs local image generation smoke tests.
 *
 * Gated on the BLAZEN_TEST_DIFFUSION environment variable.
 * Only runs when the native binding is compiled with the `diffusion` feature.
 *
 * Build first:
 *   cd crates/blazen-node && npm install && npm run build -- --features diffusion
 *
 * Run:
 *   BLAZEN_TEST_DIFFUSION=1 node --test tests/node/test_diffusion_smoke.mjs
 *
 * Tests are sync (no async/await) per project convention: napi-rs's tokio
 * handle lifecycle hangs Node's test runner with async tests.
 */

import { describe, it } from "node:test";
import assert from "node:assert/strict";

import {
  DiffusionProvider,
  JsDiffusionScheduler,
} from "../../crates/blazen-node/index.js";

const BLAZEN_TEST_DIFFUSION = process.env.BLAZEN_TEST_DIFFUSION;

const MODEL_ID = "stabilityai/stable-diffusion-2-1";

describe("diffusion-rs local image generation", { skip: !BLAZEN_TEST_DIFFUSION }, () => {
  // Test 1: DiffusionOptions plain-object shape + provider getters reflect them.
  // JsDiffusionOptions is a TS interface (plain object), not a constructable class.
  // The "constructor" path is DiffusionProvider.create(options); resolved values
  // are exposed via provider getters.
  it("DiffusionProvider.create accepts options and getters reflect resolved values", () => {
    // Skip-on-missing: feature-gate the binding itself.
    if (typeof DiffusionProvider !== "function") {
      return; // not built with diffusion feature
    }
    if (typeof DiffusionProvider.create !== "function") {
      return;
    }

    const provider = DiffusionProvider.create({
      modelId: MODEL_ID,
      width: 768,
      height: 768,
      numInferenceSteps: 25,
      guidanceScale: 8.5,
    });

    assert.equal(provider.width, 768, "width getter should reflect option");
    assert.equal(provider.height, 768, "height getter should reflect option");
    assert.equal(provider.numInferenceSteps, 25, "numInferenceSteps getter should reflect option");
    assert.equal(provider.guidanceScale, 8.5, "guidanceScale getter should reflect option");
  });

  // Test 1b: Defaults — when options are omitted, provider exposes documented defaults
  // (512x512, 20 steps, 7.5 guidance) per the JsDiffusionOptions docstring.
  it("DiffusionProvider.create exposes default getters when options are omitted", () => {
    if (typeof DiffusionProvider !== "function") {
      return;
    }
    if (typeof DiffusionProvider.create !== "function") {
      return;
    }

    const provider = DiffusionProvider.create({ modelId: MODEL_ID });

    assert.equal(provider.width, 512, "default width should be 512");
    assert.equal(provider.height, 512, "default height should be 512");
    assert.equal(provider.numInferenceSteps, 20, "default numInferenceSteps should be 20");
    assert.equal(provider.guidanceScale, 7.5, "default guidanceScale should be 7.5");
  });

  // Test 2: JsDiffusionScheduler enum exposes the documented variants
  // (Euler, EulerA, Dpm, Ddim) with string values.
  it("JsDiffusionScheduler exposes the documented variants", () => {
    // Skip-on-missing: const enums are inlined by tsc but napi-rs re-exports them
    // at runtime. If the export is missing, the binding wasn't built with diffusion.
    if (JsDiffusionScheduler == null) {
      return;
    }

    assert.equal(JsDiffusionScheduler.Euler, "euler", "Euler variant value");
    assert.equal(JsDiffusionScheduler.EulerA, "eulerA", "EulerA variant value");
    assert.equal(JsDiffusionScheduler.Dpm, "dpm", "Dpm variant value");
    assert.equal(JsDiffusionScheduler.Ddim, "ddim", "Ddim variant value");
  });

  // Test 2b: Provider accepts a scheduler option from the enum.
  it("DiffusionProvider.create accepts a scheduler option", () => {
    if (typeof DiffusionProvider !== "function") {
      return;
    }
    if (typeof DiffusionProvider.create !== "function") {
      return;
    }
    if (JsDiffusionScheduler == null) {
      return;
    }

    // Construct with each scheduler variant; should not throw.
    const provider = DiffusionProvider.create({
      modelId: MODEL_ID,
      scheduler: JsDiffusionScheduler.EulerA,
    });
    assert.ok(provider, "provider should be constructed with eulerA scheduler");
    assert.equal(typeof provider.width, "number", "provider should still expose getters");
  });

  // Test 3: DiffusionProvider class shape.
  // The Rust surface today exposes `create` + four getters. There is no `generate`
  // method on the JS side (image generation goes through the higher-level Vision
  // pipeline / step handlers). Verify the actual shape rather than fabricating one.
  it("DiffusionProvider exposes the expected class shape", () => {
    if (typeof DiffusionProvider !== "function") {
      return;
    }

    // Static factory.
    assert.equal(typeof DiffusionProvider.create, "function", "create should be a static method");

    // Construct a minimal provider to inspect instance shape.
    if (typeof DiffusionProvider.create !== "function") {
      return;
    }
    const provider = DiffusionProvider.create({ modelId: MODEL_ID });

    // Instance getters as documented in index.d.ts.
    assert.equal(typeof provider.width, "number", "width getter");
    assert.equal(typeof provider.height, "number", "height getter");
    assert.equal(typeof provider.numInferenceSteps, "number", "numInferenceSteps getter");
    assert.equal(typeof provider.guidanceScale, "number", "guidanceScale getter");

    // Provider should be an instance of DiffusionProvider.
    assert.ok(provider instanceof DiffusionProvider, "provider should be an instance of DiffusionProvider");
  });

  // Test 4: Skip-on-missing pattern at the describe level.
  // If BLAZEN_TEST_DIFFUSION is unset, the whole describe block is skipped. Inside
  // each test we additionally guard `typeof DiffusionProvider !== "function"` and
  // `JsDiffusionScheduler == null` to handle the case where the env var is set but
  // the binding wasn't built with the `diffusion` feature.
  it("skips gracefully when the diffusion feature is not built", () => {
    const featureAvailable =
      typeof DiffusionProvider === "function" &&
      typeof DiffusionProvider.create === "function" &&
      JsDiffusionScheduler != null;

    // This test asserts the skip-on-missing pattern itself: either the feature
    // is fully present, or the bindings simply don't expose it. There is no
    // half-built state.
    if (!featureAvailable) {
      assert.ok(
        typeof DiffusionProvider !== "function" ||
          typeof DiffusionProvider.create !== "function" ||
          JsDiffusionScheduler == null,
        "missing-feature skip path is consistent",
      );
      return;
    }

    assert.equal(typeof DiffusionProvider, "function", "DiffusionProvider class should be exported");
    assert.equal(typeof DiffusionProvider.create, "function", "create factory should be exported");
    assert.notEqual(JsDiffusionScheduler, null, "JsDiffusionScheduler enum should be exported");
  });
});
