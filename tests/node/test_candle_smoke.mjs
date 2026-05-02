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

import test from "ava";

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

test("candle local LLM + embed typed bindings · CandleLlmOptions object literal is accepted by CandleLlmProvider.create", (t) => {
  if (!hasCandleLlm) {
    t.pass("candle llm feature not built");
    return;
  }

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
  t.truthy(provider, "expected a CandleLlmProvider instance");
  t.is(provider.modelId, "meta-llama/Llama-3.2-1B");
});

test("candle local LLM + embed typed bindings · CandleLlmProvider.create with no arguments returns a default provider", (t) => {
  if (!hasCandleLlm) {
    t.pass("candle llm feature not built");
    return;
  }

  const provider = CandleLlmProvider.create();
  t.truthy(provider, "expected a CandleLlmProvider instance");
  t.is(typeof provider.modelId, "string");
});

test("candle local LLM + embed typed bindings · CandleInferenceResult constructor + getters round-trip values", (t) => {
  if (!hasCandleInferenceResult) {
    t.pass("candle inference result feature not built");
    return;
  }

  // Constructor: (content, promptTokens, completionTokens, totalTimeSecs).
  const result = new CandleInferenceResult("hello world", 12, 4, 0.42);
  t.is(result.content, "hello world");
  t.is(result.promptTokens, 12);
  t.is(result.completionTokens, 4);
  t.is(result.totalTimeSecs, 0.42);
});

test("candle local LLM + embed typed bindings · CandleInferenceResult getters tolerate zero token counts", (t) => {
  if (!hasCandleInferenceResult) {
    t.pass("candle inference result feature not built");
    return;
  }

  // The doc-comment on the constructor permits 0 for unknown counts.
  const result = new CandleInferenceResult("", 0, 0, 0.0);
  t.is(result.content, "");
  t.is(result.promptTokens, 0);
  t.is(result.completionTokens, 0);
  t.is(result.totalTimeSecs, 0.0);
});

test("candle local LLM + embed typed bindings · CandleInferenceResult exposes documented getter shape", (t) => {
  if (!hasCandleInferenceResult) {
    t.pass("candle inference result feature not built");
    return;
  }

  const result = new CandleInferenceResult("x", 1, 1, 0.01);
  const descriptors = Object.getOwnPropertyDescriptors(
    Object.getPrototypeOf(result),
  );
  t.truthy(descriptors.content, "content getter must exist on prototype");
  t.truthy(
    descriptors.promptTokens,
    "promptTokens getter must exist on prototype",
  );
  t.truthy(
    descriptors.completionTokens,
    "completionTokens getter must exist on prototype",
  );
  t.truthy(
    descriptors.totalTimeSecs,
    "totalTimeSecs getter must exist on prototype",
  );
});

test("candle local LLM + embed typed bindings · CandleLlmProvider exposes the documented class shape", (t) => {
  if (!hasCandleLlm) {
    t.pass("candle llm feature not built");
    return;
  }

  // Static factory.
  t.is(typeof CandleLlmProvider.create, "function");

  const provider = CandleLlmProvider.create({
    modelId: "meta-llama/Llama-3.2-1B",
  });

  // Instance methods on the prototype (typeof on the bound method).
  t.is(typeof provider.complete, "function");
  t.is(typeof provider.completeWithOptions, "function");
  t.is(typeof provider.stream, "function");
  t.is(typeof provider.load, "function");
  t.is(typeof provider.unload, "function");
  t.is(typeof provider.isLoaded, "function");
  t.is(typeof provider.vramBytes, "function");

  // Instance getter on the prototype.
  const descriptors = Object.getOwnPropertyDescriptors(
    Object.getPrototypeOf(provider),
  );
  t.truthy(descriptors.modelId, "modelId getter must exist on prototype");
});

test("candle local LLM + embed typed bindings · CandleEmbedOptions object literal shape is accepted by CandleEmbedProvider.create", (t) => {
  if (!hasCandleEmbed) {
    t.pass("candle embed feature not built");
    return;
  }

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
  t.truthy(pending, "expected create() to return a value");
  t.is(typeof pending.then, "function", "create() must return a Promise");
  // Attach a noop catch handler so an unfulfilled rejection (e.g. no
  // network) does not leak as an unhandled promise rejection.
  pending.then(
    () => {},
    () => {},
  );
});

test("candle local LLM + embed typed bindings · CandleEmbedProvider (a.k.a. CandleEmbedModel) exposes the documented class shape", (t) => {
  if (!hasCandleEmbed) {
    t.pass("candle embed feature not built");
    return;
  }

  // Static factory.
  t.is(typeof CandleEmbedProvider.create, "function");

  // Instance methods + getters live on the prototype itself, regardless
  // of whether we have an instance handy. Inspect the prototype directly
  // so we never need to await the async create().
  const proto = CandleEmbedProvider.prototype;
  t.is(typeof proto.embed, "function");
  t.is(typeof proto.load, "function");
  t.is(typeof proto.unload, "function");
  t.is(typeof proto.isLoaded, "function");

  const descriptors = Object.getOwnPropertyDescriptors(proto);
  t.truthy(descriptors.modelId, "modelId getter must exist on prototype");
  t.truthy(
    descriptors.dimensions,
    "dimensions getter must exist on prototype",
  );
});
