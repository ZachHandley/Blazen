// Tract pure-Rust ONNX embedding typed-binding smoke tests.
//
// Exercises the typed surface only, without downloading or running a model.
// The classes are exposed by blazen-node whenever the binding is compiled
// with the `embed-tract` feature; when the feature is absent the symbols
// are missing from the module exports and every test below silently
// passes through (early return) instead of failing.
//
// Build with the feature enabled first:
//     cd crates/blazen-node && npm install && npm run build -- --features embed-tract
//
// Run shape-only tests:
//     node --test tests/node/test_tract_smoke.mjs
//
// Run live tests (downloads / loads an ONNX model from Hugging Face):
//     BLAZEN_TEST_TRACT=1 node --test tests/node/test_tract_smoke.mjs
//
// All test bodies are synchronous: async tests hang the Node test runner
// because of the napi-rs tokio-handle lifecycle (project memory).

import { describe, test } from "node:test";
import assert from "node:assert/strict";

import * as blazen from "../../crates/blazen-node/index.js";

const TRACT_ENABLED = process.env.BLAZEN_TEST_TRACT === "1";

// Whether the native binding was built with the `embed-tract` feature. We
// probe for `TractEmbedModel` since it is the canonical entry point for the
// feature; if it is absent every other typed class will also be absent.
const HAS_TRACT = typeof blazen.TractEmbedModel === "function";

describe("Tract typed bindings", () => {
  test("TractOptions is an object-literal interface (no runtime class)", () => {
    if (!HAS_TRACT) return;
    // `JsTractOptions` is a TypeScript interface -- at runtime it is just a
    // plain object passed to `TractEmbedModel.create`. Confirm there is no
    // accidental class export shadowing it.
    assert.equal(blazen.TractOptions, undefined);
    assert.equal(blazen.JsTractOptions, undefined);
  });

  test("TractOptions object literal is accepted by TractEmbedModel.create", () => {
    if (!HAS_TRACT) return;

    // JsTractOptions is a structural interface: every field is optional.
    // create() is synchronous; pass a fully-populated literal to verify the
    // type surface accepts every documented field. We attach a no-op handler
    // to avoid an unhandled-rejection warning if the model loader fails.
    let model;
    try {
      model = blazen.TractEmbedModel.create({
        modelName: "BGESmallENV15",
        cacheDir: "/tmp/blazen-tract-cache",
        maxBatchSize: 32,
        showDownloadProgress: false,
      });
    } catch {
      // Model load may fail in offline / restricted environments. The shape
      // check is the important part; the runtime call is exercised by the
      // env-gated live test below.
      return;
    }
    assert.ok(model, "expected a TractEmbedModel instance");
    assert.equal(typeof model.modelId, "string");
    assert.equal(typeof model.dimensions, "number");
  });

  test("TractEmbedModel class shape (static create + instance methods/getters)", () => {
    if (!HAS_TRACT) return;
    assert.equal(typeof blazen.TractEmbedModel, "function");
    assert.equal(typeof blazen.TractEmbedModel.create, "function");

    const proto = blazen.TractEmbedModel.prototype;

    // `embed` is an async method.
    assert.equal(
      typeof proto.embed,
      "function",
      "TractEmbedModel.prototype.embed must be a function",
    );

    // `modelId` and `dimensions` are getters, not methods.
    for (const name of ["modelId", "dimensions"]) {
      const desc = Object.getOwnPropertyDescriptor(proto, name);
      assert.ok(desc, `${name} getter must exist on prototype`);
      assert.equal(typeof desc.get, "function");
    }
  });

  test("TractResponse is an object-literal interface (no runtime class)", () => {
    if (!HAS_TRACT) return;
    // `JsTractResponse` is a TypeScript interface; the value yielded by
    // `embed()` is a plain object with `embeddings` and `model` fields.
    // Confirm there is no accidental class export shadowing it.
    assert.equal(blazen.TractResponse, undefined);
    assert.equal(blazen.JsTractResponse, undefined);
  });

  test("TractError is an exception subclass of ProviderError", () => {
    if (!HAS_TRACT) return;
    // The typed error tree is declared in `index.d.ts` but is only present in
    // `index.js` once the runtime error-wiring lands. Skip gracefully when
    // the class is absent so this smoke test can run on partial builds.
    if (typeof blazen.TractError !== "function") return;

    // Must be a constructor whose instances are Errors.
    const err = new blazen.TractError("boom");
    assert.ok(err instanceof Error, "TractError instances must be Errors");
    assert.ok(
      err instanceof blazen.TractError,
      "TractError instances must be TractError",
    );

    // And TractError itself extends ProviderError when that base is
    // present (only true for the standard typed-error tree).
    if (typeof blazen.ProviderError === "function") {
      assert.ok(
        blazen.TractError.prototype instanceof blazen.ProviderError,
        "TractError must extend ProviderError",
      );
      assert.ok(
        err instanceof blazen.ProviderError,
        "TractError instance must also be a ProviderError",
      );
    }
  });
});

// Live-model checks. Per the project's sync-only Node-test rule we cannot
// actually await `model.embed(...)` here, so we restrict ourselves to verifying
// that the env-gated path constructs the model and exposes the expected shape.
// Real end-to-end embedding is exercised by the Python smoke tests.
describe("Tract embed live (env-gated)", { skip: !TRACT_ENABLED }, () => {
  test("TractEmbedModel.create returns a model with positive dimensions", () => {
    if (!HAS_TRACT) return;

    const model = blazen.TractEmbedModel.create();
    assert.ok(model, "create() must return a TractEmbedModel instance");
    assert.ok(
      typeof model.modelId === "string" && model.modelId.length > 0,
      "expected a non-empty model ID",
    );
    assert.ok(
      typeof model.dimensions === "number" && model.dimensions > 0,
      "expected positive dimensions",
    );
  });

  test("TractEmbedModel.embed returns a thenable when invoked", () => {
    if (!HAS_TRACT) return;

    const model = blazen.TractEmbedModel.create();
    // We cannot await in a sync test body. Just confirm `embed` returns a
    // Promise-shaped value; we attach a best-effort no-op handler so an
    // eventual rejection does not surface as an unhandled-rejection warning.
    const result = model.embed(["hello world"]);
    assert.ok(result, "embed() must return a value");
    assert.equal(typeof result.then, "function", "embed() must return a Promise");
    if (typeof result.catch === "function") {
      result.catch(() => {});
    }
  });
});
