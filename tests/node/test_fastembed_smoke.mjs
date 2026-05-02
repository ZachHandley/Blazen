// FastEmbed typed-bindings smoke tests.
//
// Sync-only test bodies -- async tests hang the Node runner due to napi-rs
// tokio handle lifecycle (see project memory feedback_node_sync_tests.md).
//
// Class-shape tests always run and skip gracefully when the native binding
// is built without the fastembed feature (i.e. the typed classes are absent
// from the module).
//
// Build with the feature enabled first:
//     cd crates/blazen-node && npm install && npm run build -- --features fastembed
//
// Run shape-only tests:
//     node --test tests/node/test_fastembed_smoke.mjs

import test from "ava";

import * as blazen from "../../crates/blazen-node/index.js";

// Whether the native binding was built with fastembed support. We probe for
// `FastEmbedModel` since it is the canonical entry point; if it is absent
// every fastembed-related symbol will also be absent.
const HAS_FASTEMBED = typeof blazen.FastEmbedModel === "function";

test("FastEmbed typed bindings · FastEmbedOptions is an object-literal interface (no runtime class)", (t) => {
  if (!HAS_FASTEMBED) {
    t.pass("fastembed feature not built");
    return;
  }
  // `JsFastEmbedOptions` is a TypeScript interface -- at runtime it is just
  // a plain object passed to `FastEmbedModel.create`. Confirm there is no
  // accidental class export shadowing it.
  t.is(blazen.FastEmbedOptions, undefined);
  t.is(blazen.JsFastEmbedOptions, undefined);
});

test("FastEmbed typed bindings · FastEmbedModel class shape (static create + instance methods + getters)", (t) => {
  if (!HAS_FASTEMBED) {
    t.pass("fastembed feature not built");
    return;
  }
  t.is(typeof blazen.FastEmbedModel, "function");
  t.is(typeof blazen.FastEmbedModel.create, "function");

  const proto = blazen.FastEmbedModel.prototype;
  // `embed` is an async instance method.
  t.is(
    typeof proto.embed,
    "function",
    "FastEmbedModel.prototype.embed must be a function",
  );

  // `modelId` and `dimensions` are getters, not methods.
  for (const name of ["modelId", "dimensions"]) {
    const desc = Object.getOwnPropertyDescriptor(proto, name);
    t.truthy(desc, `${name} getter must exist on prototype`);
    t.is(typeof desc.get, "function");
  }
});

test("FastEmbed typed bindings · JsFastEmbedModel is a TypeScript-only alias (no runtime value)", (t) => {
  if (!HAS_FASTEMBED) {
    t.pass("fastembed feature not built");
    return;
  }
  // `JsFastEmbedModel` is declared in `index.d.ts` as a type alias of
  // `FastEmbedModel`. The native binding only exposes the canonical
  // `FastEmbedModel` constructor; the alias carries no runtime value.
  t.is(blazen.JsFastEmbedModel, undefined);
});

test("FastEmbed typed bindings · FastEmbedResponse is an object-literal interface (no runtime class)", (t) => {
  if (!HAS_FASTEMBED) {
    t.pass("fastembed feature not built");
    return;
  }
  // `JsFastEmbedResponse` is a TypeScript interface -- at runtime it is a
  // plain object yielded by `FastEmbedModel.embed`. Confirm no runtime
  // class export shadows it.
  t.is(blazen.FastEmbedResponse, undefined);
  t.is(blazen.JsFastEmbedResponse, undefined);
});

test("FastEmbed typed bindings · fastembed error classes form a ProviderError hierarchy", (t) => {
  if (!HAS_FASTEMBED) {
    t.pass("fastembed feature not built");
    return;
  }
  // The typed error tree is declared in `index.d.ts` but is only present in
  // `index.js` once the runtime error-wiring lands. Skip gracefully when
  // the classes are absent so this smoke test can run on partial builds.
  if (typeof blazen.FastEmbedError !== "function") {
    t.pass("FastEmbedError runtime class not present");
    return;
  }

  t.is(typeof blazen.EmbedUnknownModelError, "function");
  t.is(typeof blazen.EmbedInitError, "function");
  t.is(typeof blazen.EmbedEmbedError, "function");
  t.is(typeof blazen.EmbedMutexPoisonedError, "function");
  t.is(typeof blazen.EmbedTaskPanickedError, "function");

  // Each subclass extends FastEmbedError.
  for (const sub of [
    blazen.EmbedUnknownModelError,
    blazen.EmbedInitError,
    blazen.EmbedEmbedError,
    blazen.EmbedMutexPoisonedError,
    blazen.EmbedTaskPanickedError,
  ]) {
    t.truthy(
      sub.prototype instanceof blazen.FastEmbedError,
      `${sub.name} must extend FastEmbedError`,
    );
  }

  // And FastEmbedError itself extends ProviderError when that base is
  // present (only true for the standard typed-error tree).
  if (typeof blazen.ProviderError === "function") {
    t.truthy(
      blazen.FastEmbedError.prototype instanceof blazen.ProviderError,
      "FastEmbedError must extend ProviderError",
    );
  }
});
