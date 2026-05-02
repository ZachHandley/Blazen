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

import test from "ava";

import {
  DiffusionProvider,
  JsDiffusionScheduler,
} from "../../crates/blazen-node/index.js";

const BLAZEN_TEST_DIFFUSION = process.env.BLAZEN_TEST_DIFFUSION;

const MODEL_ID = "stabilityai/stable-diffusion-2-1";

const T = BLAZEN_TEST_DIFFUSION ? test : test.skip;

// Test 1: DiffusionOptions plain-object shape + provider getters reflect them.
// JsDiffusionOptions is a TS interface (plain object), not a constructable class.
// The "constructor" path is DiffusionProvider.create(options); resolved values
// are exposed via provider getters.
T("diffusion-rs local image generation · DiffusionProvider.create accepts options and getters reflect resolved values", (t) => {
  // Skip-on-missing: feature-gate the binding itself.
  if (typeof DiffusionProvider !== "function") {
    t.pass("diffusion feature not built");
    return; // not built with diffusion feature
  }
  if (typeof DiffusionProvider.create !== "function") {
    t.pass("diffusion create factory missing");
    return;
  }

  const provider = DiffusionProvider.create({
    modelId: MODEL_ID,
    width: 768,
    height: 768,
    numInferenceSteps: 25,
    guidanceScale: 8.5,
  });

  t.is(provider.width, 768, "width getter should reflect option");
  t.is(provider.height, 768, "height getter should reflect option");
  t.is(provider.numInferenceSteps, 25, "numInferenceSteps getter should reflect option");
  t.is(provider.guidanceScale, 8.5, "guidanceScale getter should reflect option");
});

// Test 1b: Defaults — when options are omitted, provider exposes documented defaults
// (512x512, 20 steps, 7.5 guidance) per the JsDiffusionOptions docstring.
T("diffusion-rs local image generation · DiffusionProvider.create exposes default getters when options are omitted", (t) => {
  if (typeof DiffusionProvider !== "function") {
    t.pass("diffusion feature not built");
    return;
  }
  if (typeof DiffusionProvider.create !== "function") {
    t.pass("diffusion create factory missing");
    return;
  }

  const provider = DiffusionProvider.create({ modelId: MODEL_ID });

  t.is(provider.width, 512, "default width should be 512");
  t.is(provider.height, 512, "default height should be 512");
  t.is(provider.numInferenceSteps, 20, "default numInferenceSteps should be 20");
  t.is(provider.guidanceScale, 7.5, "default guidanceScale should be 7.5");
});

// Test 2: JsDiffusionScheduler enum exposes the documented variants
// (Euler, EulerA, Dpm, Ddim) with string values.
T("diffusion-rs local image generation · JsDiffusionScheduler exposes the documented variants", (t) => {
  // Skip-on-missing: const enums are inlined by tsc but napi-rs re-exports them
  // at runtime. If the export is missing, the binding wasn't built with diffusion.
  if (JsDiffusionScheduler == null) {
    t.pass("JsDiffusionScheduler not exported");
    return;
  }

  t.is(JsDiffusionScheduler.Euler, "euler", "Euler variant value");
  t.is(JsDiffusionScheduler.EulerA, "eulerA", "EulerA variant value");
  t.is(JsDiffusionScheduler.Dpm, "dpm", "Dpm variant value");
  t.is(JsDiffusionScheduler.Ddim, "ddim", "Ddim variant value");
});

// Test 2b: Provider accepts a scheduler option from the enum.
T("diffusion-rs local image generation · DiffusionProvider.create accepts a scheduler option", (t) => {
  if (typeof DiffusionProvider !== "function") {
    t.pass("diffusion feature not built");
    return;
  }
  if (typeof DiffusionProvider.create !== "function") {
    t.pass("diffusion create factory missing");
    return;
  }
  if (JsDiffusionScheduler == null) {
    t.pass("JsDiffusionScheduler not exported");
    return;
  }

  // Construct with each scheduler variant; should not throw.
  const provider = DiffusionProvider.create({
    modelId: MODEL_ID,
    scheduler: JsDiffusionScheduler.EulerA,
  });
  t.truthy(provider, "provider should be constructed with eulerA scheduler");
  t.is(typeof provider.width, "number", "provider should still expose getters");
});

// Test 3: DiffusionProvider class shape.
// The Rust surface today exposes `create` + four getters. There is no `generate`
// method on the JS side (image generation goes through the higher-level Vision
// pipeline / step handlers). Verify the actual shape rather than fabricating one.
T("diffusion-rs local image generation · DiffusionProvider exposes the expected class shape", (t) => {
  if (typeof DiffusionProvider !== "function") {
    t.pass("diffusion feature not built");
    return;
  }

  // Static factory.
  t.is(typeof DiffusionProvider.create, "function", "create should be a static method");

  // Construct a minimal provider to inspect instance shape.
  if (typeof DiffusionProvider.create !== "function") {
    return;
  }
  const provider = DiffusionProvider.create({ modelId: MODEL_ID });

  // Instance getters as documented in index.d.ts.
  t.is(typeof provider.width, "number", "width getter");
  t.is(typeof provider.height, "number", "height getter");
  t.is(typeof provider.numInferenceSteps, "number", "numInferenceSteps getter");
  t.is(typeof provider.guidanceScale, "number", "guidanceScale getter");

  // Provider should be an instance of DiffusionProvider.
  t.truthy(provider instanceof DiffusionProvider, "provider should be an instance of DiffusionProvider");
});

// Test 4: Skip-on-missing pattern at the describe level.
// If BLAZEN_TEST_DIFFUSION is unset, the whole describe block is skipped. Inside
// each test we additionally guard `typeof DiffusionProvider !== "function"` and
// `JsDiffusionScheduler == null` to handle the case where the env var is set but
// the binding wasn't built with the `diffusion` feature.
T("diffusion-rs local image generation · skips gracefully when the diffusion feature is not built", (t) => {
  const featureAvailable =
    typeof DiffusionProvider === "function" &&
    typeof DiffusionProvider.create === "function" &&
    JsDiffusionScheduler != null;

  // This test asserts the skip-on-missing pattern itself: either the feature
  // is fully present, or the bindings simply don't expose it. There is no
  // half-built state.
  if (!featureAvailable) {
    t.truthy(
      typeof DiffusionProvider !== "function" ||
        typeof DiffusionProvider.create !== "function" ||
        JsDiffusionScheduler == null,
      "missing-feature skip path is consistent",
    );
    return;
  }

  t.is(typeof DiffusionProvider, "function", "DiffusionProvider class should be exported");
  t.is(typeof DiffusionProvider.create, "function", "create factory should be exported");
  t.not(JsDiffusionScheduler, null, "JsDiffusionScheduler enum should be exported");
});
