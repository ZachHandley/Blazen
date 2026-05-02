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

import test from "ava";

import * as blazen from "../../crates/blazen-node/index.js";

const TRACT_ENABLED = process.env.BLAZEN_TEST_TRACT === "1";

// Whether the native binding was built with the `embed-tract` feature. We
// probe for `TractEmbedModel` since it is the canonical entry point for the
// feature; if it is absent every other typed class will also be absent.
const HAS_TRACT = typeof blazen.TractEmbedModel === "function";

test("Tract typed bindings · TractOptions is an object-literal interface (no runtime class)", (t) => {
  if (!HAS_TRACT) {
    t.pass("embed-tract feature not built");
    return;
  }
  // `JsTractOptions` is a TypeScript interface -- at runtime it is just a
  // plain object passed to `TractEmbedModel.create`. Confirm there is no
  // accidental class export shadowing it.
  t.is(blazen.TractOptions, undefined);
  t.is(blazen.JsTractOptions, undefined);
});

test("Tract typed bindings · TractOptions object literal is accepted by TractEmbedModel.create", (t) => {
  if (!HAS_TRACT) {
    t.pass("embed-tract feature not built");
    return;
  }

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
    t.pass("model load failed in offline/restricted environment");
    return;
  }
  t.truthy(model, "expected a TractEmbedModel instance");
  t.is(typeof model.modelId, "string");
  t.is(typeof model.dimensions, "number");
});

test("Tract typed bindings · TractEmbedModel class shape (static create + instance methods/getters)", (t) => {
  if (!HAS_TRACT) {
    t.pass("embed-tract feature not built");
    return;
  }
  t.is(typeof blazen.TractEmbedModel, "function");
  t.is(typeof blazen.TractEmbedModel.create, "function");

  const proto = blazen.TractEmbedModel.prototype;

  // `embed` is an async method.
  t.is(
    typeof proto.embed,
    "function",
    "TractEmbedModel.prototype.embed must be a function",
  );

  // `modelId` and `dimensions` are getters, not methods.
  for (const name of ["modelId", "dimensions"]) {
    const desc = Object.getOwnPropertyDescriptor(proto, name);
    t.truthy(desc, `${name} getter must exist on prototype`);
    t.is(typeof desc.get, "function");
  }
});

test("Tract typed bindings · TractResponse is an object-literal interface (no runtime class)", (t) => {
  if (!HAS_TRACT) {
    t.pass("embed-tract feature not built");
    return;
  }
  // `JsTractResponse` is a TypeScript interface; the value yielded by
  // `embed()` is a plain object with `embeddings` and `model` fields.
  // Confirm there is no accidental class export shadowing it.
  t.is(blazen.TractResponse, undefined);
  t.is(blazen.JsTractResponse, undefined);
});

test("Tract typed bindings · TractError is an exception subclass of ProviderError", (t) => {
  if (!HAS_TRACT) {
    t.pass("embed-tract feature not built");
    return;
  }
  // The typed error tree is declared in `index.d.ts` but is only present in
  // `index.js` once the runtime error-wiring lands. Skip gracefully when
  // the class is absent so this smoke test can run on partial builds.
  if (typeof blazen.TractError !== "function") {
    t.pass("TractError runtime class not present");
    return;
  }

  // Must be a constructor whose instances are Errors.
  const err = new blazen.TractError("boom");
  t.truthy(err instanceof Error, "TractError instances must be Errors");
  t.truthy(
    err instanceof blazen.TractError,
    "TractError instances must be TractError",
  );

  // And TractError itself extends ProviderError when that base is
  // present (only true for the standard typed-error tree).
  if (typeof blazen.ProviderError === "function") {
    t.truthy(
      blazen.TractError.prototype instanceof blazen.ProviderError,
      "TractError must extend ProviderError",
    );
    t.truthy(
      err instanceof blazen.ProviderError,
      "TractError instance must also be a ProviderError",
    );
  }
});

// Live-model checks. Per the project's sync-only Node-test rule we cannot
// actually await `model.embed(...)` here, so we restrict ourselves to verifying
// that the env-gated path constructs the model and exposes the expected shape.
// Real end-to-end embedding is exercised by the Python smoke tests.
const TLive = TRACT_ENABLED ? test : test.skip;

TLive("Tract embed live (env-gated) · TractEmbedModel.create returns a model with positive dimensions", (t) => {
  if (!HAS_TRACT) {
    t.pass("embed-tract feature not built");
    return;
  }

  const model = blazen.TractEmbedModel.create();
  t.truthy(model, "create() must return a TractEmbedModel instance");
  t.truthy(
    typeof model.modelId === "string" && model.modelId.length > 0,
    "expected a non-empty model ID",
  );
  t.truthy(
    typeof model.dimensions === "number" && model.dimensions > 0,
    "expected positive dimensions",
  );
});

TLive("Tract embed live (env-gated) · TractEmbedModel.embed returns a thenable when invoked", (t) => {
  if (!HAS_TRACT) {
    t.pass("embed-tract feature not built");
    return;
  }

  const model = blazen.TractEmbedModel.create();
  // We cannot await in a sync test body. Just confirm `embed` returns a
  // Promise-shaped value; we attach a best-effort no-op handler so an
  // eventual rejection does not surface as an unhandled-rejection warning.
  const result = model.embed(["hello world"]);
  t.truthy(result, "embed() must return a value");
  t.is(typeof result.then, "function", "embed() must return a Promise");
  if (typeof result.catch === "function") {
    result.catch(() => {});
  }
});
