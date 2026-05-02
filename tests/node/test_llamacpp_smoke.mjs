// llama.cpp typed-bindings smoke tests.
//
// Sync-only test bodies — async tests hang the Node runner due to napi-rs
// tokio handle lifecycle (see project memory feedback_node_sync_tests.md).
//
// Gated on BLAZEN_TEST_LLAMACPP=1 for live-model tests; class-shape tests
// always run and skip gracefully when the binding is built without the
// llamacpp feature (i.e. the typed classes are absent from the module).
//
// Build with the feature enabled first:
//     cd crates/blazen-node && npm install && npm run build -- --features llamacpp
//
// Run shape-only tests:
//     node --test tests/node/test_llamacpp_smoke.mjs
//
// Run live tests (requires a real GGUF model):
//     BLAZEN_TEST_LLAMACPP=1 \
//     BLAZEN_LLAMACPP_MODEL_PATH=/path/to/model.gguf \
//     node --test tests/node/test_llamacpp_smoke.mjs

import test from "ava";

import * as blazen from "../../crates/blazen-node/index.js";

const LLAMACPP_ENABLED = process.env.BLAZEN_TEST_LLAMACPP === "1";
const LLAMACPP_MODEL_PATH = process.env.BLAZEN_LLAMACPP_MODEL_PATH;

// Whether the native binding was built with the `llamacpp` feature. We probe
// for `LlamaCppProvider` since it is the canonical entry point for the
// feature; if it is absent every other typed class will also be absent.
const HAS_LLAMACPP = typeof blazen.LlamaCppProvider === "function";

const LiveTest = LLAMACPP_ENABLED ? test : test.skip;

test("LlamaCpp typed bindings · LlamaCppChatRole enum exposes System/User/Assistant/Tool", (t) => {
  if (!HAS_LLAMACPP) {
    t.pass("llamacpp feature not built");
    return;
  }
  // napi-rs string-backed enums are exposed as a plain JS object whose
  // property values mirror the variant names.
  const role = blazen.LlamaCppChatRole;
  t.truthy(role, "LlamaCppChatRole should be exported");
  t.is(role.System, "System");
  t.is(role.User, "User");
  t.is(role.Assistant, "Assistant");
  t.is(role.Tool, "Tool");
});

test("LlamaCpp typed bindings · LlamaCppChatMessageInput constructor + getters", (t) => {
  if (!HAS_LLAMACPP) {
    t.pass("llamacpp feature not built");
    return;
  }
  t.is(typeof blazen.LlamaCppChatMessageInput, "function");

  const msg = new blazen.LlamaCppChatMessageInput(
    blazen.LlamaCppChatRole.User,
    "Hello",
  );
  t.is(msg.role, "User");
  t.is(msg.text, "Hello");

  const sys = new blazen.LlamaCppChatMessageInput(
    blazen.LlamaCppChatRole.System,
    "You are concise.",
  );
  t.is(sys.role, "System");
  t.is(sys.text, "You are concise.");
});

test("LlamaCpp typed bindings · LlamaCppInferenceChunk exposes delta + finishReason getters", (t) => {
  if (!HAS_LLAMACPP) {
    t.pass("llamacpp feature not built");
    return;
  }
  t.is(typeof blazen.LlamaCppInferenceChunk, "function");

  // Cannot construct directly from JS — only the provider's stream yields
  // these. Verify the prototype carries the expected getters.
  const proto = blazen.LlamaCppInferenceChunk.prototype;
  const deltaDesc = Object.getOwnPropertyDescriptor(proto, "delta");
  const finishDesc = Object.getOwnPropertyDescriptor(proto, "finishReason");
  t.truthy(deltaDesc, "delta getter must exist on prototype");
  t.truthy(finishDesc, "finishReason getter must exist on prototype");
  t.is(typeof deltaDesc.get, "function");
  t.is(typeof finishDesc.get, "function");
});

test("LlamaCpp typed bindings · LlamaCppInferenceChunkStream exposes async next()", (t) => {
  if (!HAS_LLAMACPP) {
    t.pass("llamacpp feature not built");
    return;
  }
  t.is(typeof blazen.LlamaCppInferenceChunkStream, "function");
  t.is(
    typeof blazen.LlamaCppInferenceChunkStream.prototype.next,
    "function",
  );
});

test("LlamaCpp typed bindings · LlamaCppInferenceResult exposes content/finishReason/model/usage", (t) => {
  if (!HAS_LLAMACPP) {
    t.pass("llamacpp feature not built");
    return;
  }
  t.is(typeof blazen.LlamaCppInferenceResult, "function");

  const proto = blazen.LlamaCppInferenceResult.prototype;
  for (const name of ["content", "finishReason", "model", "usage"]) {
    const desc = Object.getOwnPropertyDescriptor(proto, name);
    t.truthy(desc, `${name} getter must exist on prototype`);
    t.is(typeof desc.get, "function");
  }
});

test("LlamaCpp typed bindings · LlamaCppInferenceUsage exposes the four token-counting getters", (t) => {
  if (!HAS_LLAMACPP) {
    t.pass("llamacpp feature not built");
    return;
  }
  t.is(typeof blazen.LlamaCppInferenceUsage, "function");

  const proto = blazen.LlamaCppInferenceUsage.prototype;
  for (const name of [
    "promptTokens",
    "completionTokens",
    "totalTokens",
    "totalTimeSec",
  ]) {
    const desc = Object.getOwnPropertyDescriptor(proto, name);
    t.truthy(desc, `${name} getter must exist on prototype`);
    t.is(typeof desc.get, "function");
  }
});

test("LlamaCpp typed bindings · LlamaCppProvider class shape (static create + instance methods)", (t) => {
  if (!HAS_LLAMACPP) {
    t.pass("llamacpp feature not built");
    return;
  }
  t.is(typeof blazen.LlamaCppProvider, "function");
  t.is(typeof blazen.LlamaCppProvider.create, "function");

  const proto = blazen.LlamaCppProvider.prototype;
  for (const method of [
    "complete",
    "completeWithOptions",
    "stream",
    "load",
    "unload",
    "isLoaded",
    "vramBytes",
  ]) {
    t.is(
      typeof proto[method],
      "function",
      `LlamaCppProvider.prototype.${method} must be a function`,
    );
  }

  // `modelId` is a getter, not a method.
  const modelIdDesc = Object.getOwnPropertyDescriptor(proto, "modelId");
  t.truthy(modelIdDesc, "modelId getter must exist on prototype");
  t.is(typeof modelIdDesc.get, "function");
});

test("LlamaCpp typed bindings · LlamaCppOptions is an object-literal interface (no runtime class)", (t) => {
  if (!HAS_LLAMACPP) {
    t.pass("llamacpp feature not built");
    return;
  }
  // `JsLlamaCppOptions` is a TypeScript interface — at runtime it is just a
  // plain object passed to `LlamaCppProvider.create`. Confirm there is no
  // accidental class export shadowing it.
  t.is(blazen.LlamaCppOptions, undefined);
  t.is(blazen.JsLlamaCppOptions, undefined);
});

test("LlamaCpp typed bindings · llama.cpp error classes form a ProviderError hierarchy", (t) => {
  if (!HAS_LLAMACPP) {
    t.pass("llamacpp feature not built");
    return;
  }
  // The typed error tree is declared in `index.d.ts` but is only present in
  // `index.js` once the runtime error-wiring lands. Skip gracefully when
  // the classes are absent so this smoke test can run on partial builds.
  if (typeof blazen.LlamaCppError !== "function") {
    t.pass("llamacpp error classes not yet wired");
    return;
  }

  t.is(typeof blazen.LlamaCppInvalidOptionsError, "function");
  t.is(typeof blazen.LlamaCppModelLoadError, "function");
  t.is(typeof blazen.LlamaCppInferenceError, "function");
  t.is(typeof blazen.LlamaCppEngineNotAvailableError, "function");

  // Each subclass extends LlamaCppError.
  for (const sub of [
    blazen.LlamaCppInvalidOptionsError,
    blazen.LlamaCppModelLoadError,
    blazen.LlamaCppInferenceError,
    blazen.LlamaCppEngineNotAvailableError,
  ]) {
    t.truthy(
      sub.prototype instanceof blazen.LlamaCppError,
      `${sub.name} must extend LlamaCppError`,
    );
  }

  // And LlamaCppError itself extends ProviderError when that base is
  // present (only true for the standard typed-error tree).
  if (typeof blazen.ProviderError === "function") {
    t.truthy(
      blazen.LlamaCppError.prototype instanceof blazen.ProviderError,
      "LlamaCppError must extend ProviderError",
    );
  }
});

// Live-model checks. Per the project's sync-only Node-test rule we cannot
// actually await `LlamaCppProvider.create(...)` here, so we restrict ourselves
// to verifying that the env-gated configuration is reachable and that the
// constructor entry point is callable. Real end-to-end inference is exercised
// by the Python smoke tests.
LiveTest("LlamaCpp provider live (env-gated) · BLAZEN_LLAMACPP_MODEL_PATH is set when live tests are enabled", (t) => {
  if (!HAS_LLAMACPP) {
    t.pass("llamacpp feature not built");
    return;
  }
  t.truthy(
    LLAMACPP_MODEL_PATH && LLAMACPP_MODEL_PATH.length > 0,
    "BLAZEN_LLAMACPP_MODEL_PATH must be set when BLAZEN_TEST_LLAMACPP=1",
  );
});

LiveTest("LlamaCpp provider live (env-gated) · LlamaCppProvider.create returns a thenable when invoked", (t) => {
  if (!HAS_LLAMACPP || !LLAMACPP_MODEL_PATH) {
    t.pass("llamacpp feature or model path not available");
    return;
  }
  // We cannot await in a sync test body. Just confirm `create` returns a
  // Promise-shaped value for the configured model path; we attach a
  // best-effort no-op handler so an eventual rejection does not surface as
  // an unhandled-rejection warning.
  const result = blazen.LlamaCppProvider.create({
    modelPath: LLAMACPP_MODEL_PATH,
  });
  t.truthy(result, "create() must return a value");
  t.is(typeof result.then, "function", "create() must return a Promise");
  if (typeof result.catch === "function") {
    result.catch(() => {});
  }
});
