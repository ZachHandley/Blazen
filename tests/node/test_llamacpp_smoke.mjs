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

import { test, describe } from "node:test";
import assert from "node:assert/strict";

import * as blazen from "../../crates/blazen-node/index.js";

const LLAMACPP_ENABLED = process.env.BLAZEN_TEST_LLAMACPP === "1";
const LLAMACPP_MODEL_PATH = process.env.BLAZEN_LLAMACPP_MODEL_PATH;

// Whether the native binding was built with the `llamacpp` feature. We probe
// for `LlamaCppProvider` since it is the canonical entry point for the
// feature; if it is absent every other typed class will also be absent.
const HAS_LLAMACPP = typeof blazen.LlamaCppProvider === "function";

describe("LlamaCpp typed bindings", () => {
  test("LlamaCppChatRole enum exposes System/User/Assistant/Tool", () => {
    if (!HAS_LLAMACPP) return;
    // napi-rs string-backed enums are exposed as a plain JS object whose
    // property values mirror the variant names.
    const role = blazen.LlamaCppChatRole;
    assert.ok(role, "LlamaCppChatRole should be exported");
    assert.equal(role.System, "System");
    assert.equal(role.User, "User");
    assert.equal(role.Assistant, "Assistant");
    assert.equal(role.Tool, "Tool");
  });

  test("LlamaCppChatMessageInput constructor + getters", () => {
    if (!HAS_LLAMACPP) return;
    assert.equal(typeof blazen.LlamaCppChatMessageInput, "function");

    const msg = new blazen.LlamaCppChatMessageInput(
      blazen.LlamaCppChatRole.User,
      "Hello",
    );
    assert.equal(msg.role, "User");
    assert.equal(msg.text, "Hello");

    const sys = new blazen.LlamaCppChatMessageInput(
      blazen.LlamaCppChatRole.System,
      "You are concise.",
    );
    assert.equal(sys.role, "System");
    assert.equal(sys.text, "You are concise.");
  });

  test("LlamaCppInferenceChunk exposes delta + finishReason getters", () => {
    if (!HAS_LLAMACPP) return;
    assert.equal(typeof blazen.LlamaCppInferenceChunk, "function");

    // Cannot construct directly from JS — only the provider's stream yields
    // these. Verify the prototype carries the expected getters.
    const proto = blazen.LlamaCppInferenceChunk.prototype;
    const deltaDesc = Object.getOwnPropertyDescriptor(proto, "delta");
    const finishDesc = Object.getOwnPropertyDescriptor(proto, "finishReason");
    assert.ok(deltaDesc, "delta getter must exist on prototype");
    assert.ok(finishDesc, "finishReason getter must exist on prototype");
    assert.equal(typeof deltaDesc.get, "function");
    assert.equal(typeof finishDesc.get, "function");
  });

  test("LlamaCppInferenceChunkStream exposes async next()", () => {
    if (!HAS_LLAMACPP) return;
    assert.equal(typeof blazen.LlamaCppInferenceChunkStream, "function");
    assert.equal(
      typeof blazen.LlamaCppInferenceChunkStream.prototype.next,
      "function",
    );
  });

  test("LlamaCppInferenceResult exposes content/finishReason/model/usage", () => {
    if (!HAS_LLAMACPP) return;
    assert.equal(typeof blazen.LlamaCppInferenceResult, "function");

    const proto = blazen.LlamaCppInferenceResult.prototype;
    for (const name of ["content", "finishReason", "model", "usage"]) {
      const desc = Object.getOwnPropertyDescriptor(proto, name);
      assert.ok(desc, `${name} getter must exist on prototype`);
      assert.equal(typeof desc.get, "function");
    }
  });

  test("LlamaCppInferenceUsage exposes the four token-counting getters", () => {
    if (!HAS_LLAMACPP) return;
    assert.equal(typeof blazen.LlamaCppInferenceUsage, "function");

    const proto = blazen.LlamaCppInferenceUsage.prototype;
    for (const name of [
      "promptTokens",
      "completionTokens",
      "totalTokens",
      "totalTimeSec",
    ]) {
      const desc = Object.getOwnPropertyDescriptor(proto, name);
      assert.ok(desc, `${name} getter must exist on prototype`);
      assert.equal(typeof desc.get, "function");
    }
  });

  test("LlamaCppProvider class shape (static create + instance methods)", () => {
    if (!HAS_LLAMACPP) return;
    assert.equal(typeof blazen.LlamaCppProvider, "function");
    assert.equal(typeof blazen.LlamaCppProvider.create, "function");

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
      assert.equal(
        typeof proto[method],
        "function",
        `LlamaCppProvider.prototype.${method} must be a function`,
      );
    }

    // `modelId` is a getter, not a method.
    const modelIdDesc = Object.getOwnPropertyDescriptor(proto, "modelId");
    assert.ok(modelIdDesc, "modelId getter must exist on prototype");
    assert.equal(typeof modelIdDesc.get, "function");
  });

  test("LlamaCppOptions is an object-literal interface (no runtime class)", () => {
    if (!HAS_LLAMACPP) return;
    // `JsLlamaCppOptions` is a TypeScript interface — at runtime it is just a
    // plain object passed to `LlamaCppProvider.create`. Confirm there is no
    // accidental class export shadowing it.
    assert.equal(blazen.LlamaCppOptions, undefined);
    assert.equal(blazen.JsLlamaCppOptions, undefined);
  });

  test("llama.cpp error classes form a ProviderError hierarchy", () => {
    if (!HAS_LLAMACPP) return;
    // The typed error tree is declared in `index.d.ts` but is only present in
    // `index.js` once the runtime error-wiring lands. Skip gracefully when
    // the classes are absent so this smoke test can run on partial builds.
    if (typeof blazen.LlamaCppError !== "function") return;

    assert.equal(typeof blazen.LlamaCppInvalidOptionsError, "function");
    assert.equal(typeof blazen.LlamaCppModelLoadError, "function");
    assert.equal(typeof blazen.LlamaCppInferenceError, "function");
    assert.equal(typeof blazen.LlamaCppEngineNotAvailableError, "function");

    // Each subclass extends LlamaCppError.
    for (const sub of [
      blazen.LlamaCppInvalidOptionsError,
      blazen.LlamaCppModelLoadError,
      blazen.LlamaCppInferenceError,
      blazen.LlamaCppEngineNotAvailableError,
    ]) {
      assert.ok(
        sub.prototype instanceof blazen.LlamaCppError,
        `${sub.name} must extend LlamaCppError`,
      );
    }

    // And LlamaCppError itself extends ProviderError when that base is
    // present (only true for the standard typed-error tree).
    if (typeof blazen.ProviderError === "function") {
      assert.ok(
        blazen.LlamaCppError.prototype instanceof blazen.ProviderError,
        "LlamaCppError must extend ProviderError",
      );
    }
  });
});

// Live-model checks. Per the project's sync-only Node-test rule we cannot
// actually await `LlamaCppProvider.create(...)` here, so we restrict ourselves
// to verifying that the env-gated configuration is reachable and that the
// constructor entry point is callable. Real end-to-end inference is exercised
// by the Python smoke tests.
describe("LlamaCpp provider live (env-gated)", { skip: !LLAMACPP_ENABLED }, () => {
  test("BLAZEN_LLAMACPP_MODEL_PATH is set when live tests are enabled", () => {
    if (!HAS_LLAMACPP) return;
    assert.ok(
      LLAMACPP_MODEL_PATH && LLAMACPP_MODEL_PATH.length > 0,
      "BLAZEN_LLAMACPP_MODEL_PATH must be set when BLAZEN_TEST_LLAMACPP=1",
    );
  });

  test("LlamaCppProvider.create returns a thenable when invoked", () => {
    if (!HAS_LLAMACPP || !LLAMACPP_MODEL_PATH) return;
    // We cannot await in a sync test body. Just confirm `create` returns a
    // Promise-shaped value for the configured model path; we attach a
    // best-effort no-op handler so an eventual rejection does not surface as
    // an unhandled-rejection warning.
    const result = blazen.LlamaCppProvider.create({
      modelPath: LLAMACPP_MODEL_PATH,
    });
    assert.ok(result, "create() must return a value");
    assert.equal(typeof result.then, "function", "create() must return a Promise");
    if (typeof result.catch === "function") {
      result.catch(() => {});
    }
  });
});
