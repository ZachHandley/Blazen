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

import { test, describe } from "node:test";
import assert from "node:assert/strict";

import * as blazen from "../../crates/blazen-node/index.js";

// Whether the native binding was built with fastembed support. We probe for
// `FastEmbedModel` since it is the canonical entry point; if it is absent
// every fastembed-related symbol will also be absent.
const HAS_FASTEMBED = typeof blazen.FastEmbedModel === "function";

describe("FastEmbed typed bindings", () => {
  test("FastEmbedOptions is an object-literal interface (no runtime class)", () => {
    if (!HAS_FASTEMBED) return;
    // `JsFastEmbedOptions` is a TypeScript interface -- at runtime it is just
    // a plain object passed to `FastEmbedModel.create`. Confirm there is no
    // accidental class export shadowing it.
    assert.equal(blazen.FastEmbedOptions, undefined);
    assert.equal(blazen.JsFastEmbedOptions, undefined);
  });

  test("FastEmbedModel class shape (static create + instance methods + getters)", () => {
    if (!HAS_FASTEMBED) return;
    assert.equal(typeof blazen.FastEmbedModel, "function");
    assert.equal(typeof blazen.FastEmbedModel.create, "function");

    const proto = blazen.FastEmbedModel.prototype;
    // `embed` is an async instance method.
    assert.equal(
      typeof proto.embed,
      "function",
      "FastEmbedModel.prototype.embed must be a function",
    );

    // `modelId` and `dimensions` are getters, not methods.
    for (const name of ["modelId", "dimensions"]) {
      const desc = Object.getOwnPropertyDescriptor(proto, name);
      assert.ok(desc, `${name} getter must exist on prototype`);
      assert.equal(typeof desc.get, "function");
    }
  });

  test("JsFastEmbedModel is a TypeScript-only alias (no runtime value)", () => {
    if (!HAS_FASTEMBED) return;
    // `JsFastEmbedModel` is declared in `index.d.ts` as a type alias of
    // `FastEmbedModel`. The native binding only exposes the canonical
    // `FastEmbedModel` constructor; the alias carries no runtime value.
    assert.equal(blazen.JsFastEmbedModel, undefined);
  });

  test("FastEmbedResponse is an object-literal interface (no runtime class)", () => {
    if (!HAS_FASTEMBED) return;
    // `JsFastEmbedResponse` is a TypeScript interface -- at runtime it is a
    // plain object yielded by `FastEmbedModel.embed`. Confirm no runtime
    // class export shadows it.
    assert.equal(blazen.FastEmbedResponse, undefined);
    assert.equal(blazen.JsFastEmbedResponse, undefined);
  });

  test("fastembed error classes form a ProviderError hierarchy", () => {
    if (!HAS_FASTEMBED) return;
    // The typed error tree is declared in `index.d.ts` but is only present in
    // `index.js` once the runtime error-wiring lands. Skip gracefully when
    // the classes are absent so this smoke test can run on partial builds.
    if (typeof blazen.FastEmbedError !== "function") return;

    assert.equal(typeof blazen.EmbedUnknownModelError, "function");
    assert.equal(typeof blazen.EmbedInitError, "function");
    assert.equal(typeof blazen.EmbedEmbedError, "function");
    assert.equal(typeof blazen.EmbedMutexPoisonedError, "function");
    assert.equal(typeof blazen.EmbedTaskPanickedError, "function");

    // Each subclass extends FastEmbedError.
    for (const sub of [
      blazen.EmbedUnknownModelError,
      blazen.EmbedInitError,
      blazen.EmbedEmbedError,
      blazen.EmbedMutexPoisonedError,
      blazen.EmbedTaskPanickedError,
    ]) {
      assert.ok(
        sub.prototype instanceof blazen.FastEmbedError,
        `${sub.name} must extend FastEmbedError`,
      );
    }

    // And FastEmbedError itself extends ProviderError when that base is
    // present (only true for the standard typed-error tree).
    if (typeof blazen.ProviderError === "function") {
      assert.ok(
        blazen.FastEmbedError.prototype instanceof blazen.ProviderError,
        "FastEmbedError must extend ProviderError",
      );
    }
  });
});
