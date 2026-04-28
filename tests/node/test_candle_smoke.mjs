// Candle local LLM + embedding typed-binding smoke tests.
//
// Exercises the typed surface only, without performing inference. The
// classes are exposed by blazen-node whenever the binding is compiled
// with the `candle` feature; when the feature is absent the symbols
// are missing from the module exports and every test below silently
// passes through (early return) instead of failing.
//
// Build with the feature enabled first:
//     cd crates/blazen-node && npm install && npm run build -- --features candle
//
// Run:
//     node --test tests/node/test_candle_smoke.mjs
//
// All test bodies are synchronous: async tests hang the Node test runner
// because of the napi-rs tokio-handle lifecycle (project memory).

import { describe, test } from "node:test";
import assert from "node:assert/strict";

import {
  CandleLlmProvider,
  CandleEmbedProvider,
  CandleInferenceResult,
} from "../../crates/blazen-node/index.js";

// Feature-detection helpers. JsCandleLlmOptions / JsCandleEmbedOptions are
// TypeScript-only interfaces; at runtime they are plain object literals,
// so we feature-gate on the classes that consume them.
const hasCandleLlm = typeof CandleLlmProvider === "function";
const hasCandleEmbed = typeof CandleEmbedProvider === "function";
const hasCandleInferenceResult = typeof CandleInferenceResult === "function";

describe("candle local LLM + embed typed bindings", () => {
  test("CandleLlmOptions object literal is accepted by CandleLlmProvider.create", () => {
    if (!hasCandleLlm) return;

    // JsCandleLlmOptions is a structural interface: every field is optional.
    // create() is synchronous and returns a CandleLlmProvider directly.
    const provider = CandleLlmProvider.create({
      modelId: "meta-llama/Llama-3.2-1B",
      device: "cpu",
      quantization: "q4_k_m",
      revision: "main",
      contextLength: 2048,
      cacheDir: "/tmp/blazen-candle-cache",
    });
    assert.ok(provider, "expected a CandleLlmProvider instance");
    assert.equal(provider.modelId, "meta-llama/Llama-3.2-1B");
  });

  test("CandleLlmProvider.create with no arguments returns a default provider", () => {
    if (!hasCandleLlm) return;

    const provider = CandleLlmProvider.create();
    assert.ok(provider, "expected a CandleLlmProvider instance");
    assert.equal(typeof provider.modelId, "string");
  });

  test("CandleInferenceResult constructor + getters round-trip values", () => {
    if (!hasCandleInferenceResult) return;

    // Constructor: (content, promptTokens, completionTokens, totalTimeSecs).
    const result = new CandleInferenceResult("hello world", 12, 4, 0.42);
    assert.equal(result.content, "hello world");
    assert.equal(result.promptTokens, 12);
    assert.equal(result.completionTokens, 4);
    assert.equal(result.totalTimeSecs, 0.42);
  });

  test("CandleInferenceResult getters tolerate zero token counts", () => {
    if (!hasCandleInferenceResult) return;

    // The doc-comment on the constructor permits 0 for unknown counts.
    const result = new CandleInferenceResult("", 0, 0, 0.0);
    assert.equal(result.content, "");
    assert.equal(result.promptTokens, 0);
    assert.equal(result.completionTokens, 0);
    assert.equal(result.totalTimeSecs, 0.0);
  });

  test("CandleInferenceResult exposes documented getter shape", () => {
    if (!hasCandleInferenceResult) return;

    const result = new CandleInferenceResult("x", 1, 1, 0.01);
    const descriptors = Object.getOwnPropertyDescriptors(
      Object.getPrototypeOf(result),
    );
    assert.ok(descriptors.content, "content getter must exist on prototype");
    assert.ok(
      descriptors.promptTokens,
      "promptTokens getter must exist on prototype",
    );
    assert.ok(
      descriptors.completionTokens,
      "completionTokens getter must exist on prototype",
    );
    assert.ok(
      descriptors.totalTimeSecs,
      "totalTimeSecs getter must exist on prototype",
    );
  });

  test("CandleLlmProvider exposes the documented class shape", () => {
    if (!hasCandleLlm) return;

    // Static factory.
    assert.equal(typeof CandleLlmProvider.create, "function");

    const provider = CandleLlmProvider.create({
      modelId: "meta-llama/Llama-3.2-1B",
    });

    // Instance methods on the prototype (typeof on the bound method).
    assert.equal(typeof provider.complete, "function");
    assert.equal(typeof provider.completeWithOptions, "function");
    assert.equal(typeof provider.stream, "function");
    assert.equal(typeof provider.load, "function");
    assert.equal(typeof provider.unload, "function");
    assert.equal(typeof provider.isLoaded, "function");
    assert.equal(typeof provider.vramBytes, "function");

    // Instance getter on the prototype.
    const descriptors = Object.getOwnPropertyDescriptors(
      Object.getPrototypeOf(provider),
    );
    assert.ok(descriptors.modelId, "modelId getter must exist on prototype");
  });

  test("CandleEmbedOptions object literal shape is accepted by CandleEmbedProvider.create", () => {
    if (!hasCandleEmbed) return;

    // JsCandleEmbedOptions is a TS-only interface; the runtime contract
    // is that CandleEmbedProvider.create accepts a plain options bag.
    // create() is async (returns a Promise) so we only assert the call
    // is well-formed and yields a thenable. We do not await it: the test
    // body must remain synchronous (async hangs the Node runner).
    const pending = CandleEmbedProvider.create({
      modelId: "BAAI/bge-small-en-v1.5",
      device: "cpu",
      revision: "main",
      cacheDir: "/tmp/blazen-candle-cache",
    });
    assert.ok(pending, "expected create() to return a value");
    assert.equal(typeof pending.then, "function", "create() must return a Promise");
    // Attach a noop catch handler so an unfulfilled rejection (e.g. no
    // network) does not leak as an unhandled promise rejection.
    pending.then(
      () => {},
      () => {},
    );
  });

  test("CandleEmbedProvider (a.k.a. CandleEmbedModel) exposes the documented class shape", () => {
    if (!hasCandleEmbed) return;

    // Static factory.
    assert.equal(typeof CandleEmbedProvider.create, "function");

    // Instance methods + getters live on the prototype itself, regardless
    // of whether we have an instance handy. Inspect the prototype directly
    // so we never need to await the async create().
    const proto = CandleEmbedProvider.prototype;
    assert.equal(typeof proto.embed, "function");
    assert.equal(typeof proto.load, "function");
    assert.equal(typeof proto.unload, "function");
    assert.equal(typeof proto.isLoaded, "function");

    const descriptors = Object.getOwnPropertyDescriptors(proto);
    assert.ok(descriptors.modelId, "modelId getter must exist on prototype");
    assert.ok(
      descriptors.dimensions,
      "dimensions getter must exist on prototype",
    );
  });
});
