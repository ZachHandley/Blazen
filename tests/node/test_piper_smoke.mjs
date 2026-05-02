// Piper TTS local synthesis smoke tests.
//
// Gated on an environment variable to avoid surprises in CI:
//
//     BLAZEN_TEST_PIPER=1                       # run the tests
//     BLAZEN_PIPER_MODEL_ID=en_US-amy-medium    # optional voice id
//
// Only runs when the native binding is compiled with the `piper` feature.
// When the feature is absent, `PiperProvider.create` should still return a
// usable handle whose `engineAvailable` getter is `false`; synthesis would
// then error at runtime. We exercise the typed surface only here.
//
// Build with the feature enabled first:
//     cd crates/blazen-node && npm install && npm run build -- --features piper
//
// Run:
//     BLAZEN_TEST_PIPER=1 node --test tests/node/test_piper_smoke.mjs

import test from "ava";

import { PiperProvider } from "../../crates/blazen-node/index.js";

const BLAZEN_TEST_PIPER = process.env.BLAZEN_TEST_PIPER;
const MODEL_ID = process.env.BLAZEN_PIPER_MODEL_ID || "en_US-amy-medium";

// Skip the entire suite when the binding does not export PiperProvider.
const hasPiperExport = typeof PiperProvider === "function";

const T = BLAZEN_TEST_PIPER && hasPiperExport ? test : test.skip;

T("piper local TTS bindings · PiperProvider.create accepts an empty options object", (t) => {
  const provider = PiperProvider.create({});
  t.truthy(provider, "expected a PiperProvider instance");
  // engineAvailable is a boolean regardless of feature compilation.
  t.is(typeof provider.engineAvailable, "boolean");
  // modelId is null when not configured.
  t.is(provider.modelId, null);
});

T("piper local TTS bindings · PiperProvider.create accepts JsPiperOptions fields", (t) => {
  const provider = PiperProvider.create({
    modelId: MODEL_ID,
    speakerId: 0,
    sampleRate: 22050,
    cacheDir: "/tmp/blazen-piper-cache",
  });
  t.truthy(provider, "expected a PiperProvider instance");
  t.is(provider.modelId, MODEL_ID);
});

T("piper local TTS bindings · PiperProvider.create with no arguments returns a default provider", (t) => {
  // The create() signature accepts an optional options bag.
  const provider = PiperProvider.create();
  t.truthy(provider, "expected a PiperProvider instance");
  t.is(provider.modelId, null);
  t.is(typeof provider.engineAvailable, "boolean");
});

T("piper local TTS bindings · PiperProvider exposes the documented class shape", (t) => {
  const provider = PiperProvider.create({ modelId: MODEL_ID });

  // Static factory.
  t.is(typeof PiperProvider.create, "function");

  // Instance getters.
  const descriptors = Object.getOwnPropertyDescriptors(
    Object.getPrototypeOf(provider),
  );
  t.truthy(descriptors.modelId, "modelId getter must exist on prototype");
  t.truthy(
    descriptors.engineAvailable,
    "engineAvailable getter must exist on prototype",
  );
});

T("piper local TTS bindings · engineAvailable signals whether the piper feature was compiled in", (t) => {
  const provider = PiperProvider.create({ modelId: MODEL_ID });
  const available = provider.engineAvailable;
  t.is(typeof available, "boolean");
  // We do not assert true/false: this depends on build-time feature flags.
  // The getter simply must not throw and must return a boolean.
});

T("piper local TTS bindings · modelId round-trips the configured voice identifier", (t) => {
  const provider = PiperProvider.create({ modelId: "en_GB-alan-low" });
  t.is(provider.modelId, "en_GB-alan-low");
});
