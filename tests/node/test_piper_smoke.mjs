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

import { describe, it } from "node:test";
import assert from "node:assert/strict";

import { PiperProvider } from "../../crates/blazen-node/index.js";

const BLAZEN_TEST_PIPER = process.env.BLAZEN_TEST_PIPER;
const MODEL_ID = process.env.BLAZEN_PIPER_MODEL_ID || "en_US-amy-medium";

// Skip the entire suite when the binding does not export PiperProvider.
const hasPiperExport = typeof PiperProvider === "function";

describe("piper local TTS bindings", { skip: !BLAZEN_TEST_PIPER || !hasPiperExport }, () => {
  it("PiperProvider.create accepts an empty options object", () => {
    const provider = PiperProvider.create({});
    assert.ok(provider, "expected a PiperProvider instance");
    // engineAvailable is a boolean regardless of feature compilation.
    assert.equal(typeof provider.engineAvailable, "boolean");
    // modelId is null when not configured.
    assert.equal(provider.modelId, null);
  });

  it("PiperProvider.create accepts JsPiperOptions fields", () => {
    const provider = PiperProvider.create({
      modelId: MODEL_ID,
      speakerId: 0,
      sampleRate: 22050,
      cacheDir: "/tmp/blazen-piper-cache",
    });
    assert.ok(provider, "expected a PiperProvider instance");
    assert.equal(provider.modelId, MODEL_ID);
  });

  it("PiperProvider.create with no arguments returns a default provider", () => {
    // The create() signature accepts an optional options bag.
    const provider = PiperProvider.create();
    assert.ok(provider, "expected a PiperProvider instance");
    assert.equal(provider.modelId, null);
    assert.equal(typeof provider.engineAvailable, "boolean");
  });

  it("PiperProvider exposes the documented class shape", () => {
    const provider = PiperProvider.create({ modelId: MODEL_ID });

    // Static factory.
    assert.equal(typeof PiperProvider.create, "function");

    // Instance getters.
    const descriptors = Object.getOwnPropertyDescriptors(
      Object.getPrototypeOf(provider),
    );
    assert.ok(descriptors.modelId, "modelId getter must exist on prototype");
    assert.ok(
      descriptors.engineAvailable,
      "engineAvailable getter must exist on prototype",
    );
  });

  it("engineAvailable signals whether the piper feature was compiled in", () => {
    const provider = PiperProvider.create({ modelId: MODEL_ID });
    const available = provider.engineAvailable;
    assert.equal(typeof available, "boolean");
    // We do not assert true/false: this depends on build-time feature flags.
    // The getter simply must not throw and must return a boolean.
  });

  it("modelId round-trips the configured voice identifier", () => {
    const provider = PiperProvider.create({ modelId: "en_GB-alan-low" });
    assert.equal(provider.modelId, "en_GB-alan-low");
  });
});
