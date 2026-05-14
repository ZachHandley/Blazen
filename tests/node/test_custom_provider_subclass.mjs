/**
 * CustomProvider typed-subclass surface tests.
 *
 * Verifies the Phase B-Pivot rollout of `CustomProvider`:
 *
 *  - JS `class extends CustomProvider` instances dispatch overridden
 *    methods (textToSpeech, complete) through the Rust trait adapter.
 *  - Methods the subclass did NOT override raise the typed
 *    `UnsupportedError` (or carry an `[UnsupportedError]` tag the napi
 *    error-wrapping layer recognises).
 *  - The `ollama` static factory builds a usable `CustomProvider` with
 *    the expected `providerId`.
 *  - `BaseProvider.extract` parses a JSON-Schema-shaped completion when
 *    the subclass's `complete` returns a JSON string.
 *
 * No API keys / network calls required — every test stubs the backing
 * provider methods locally.
 *
 * Build the native binding first:
 *   pnpm --filter blazen run build
 */

import test from "ava";

import {
  BaseProvider,
  ChatMessage,
  CompletionModel,
  CustomProvider,
} from "../../crates/blazen-node/index.js";

// `UnsupportedError` is installed by the post-build error-classes shim
// rather than emitted by napi-rs, so it isn't picked up by ESM named
// imports of the CJS module. Pull it off the namespace where it
// actually lives (falls back to `null` if the shim hasn't run).
import * as blazen from "../../crates/blazen-node/index.js";
const UnsupportedError = blazen.UnsupportedError ?? null;

// ---------------------------------------------------------------------------
// 1. Subclass `textToSpeech` routes to override
// ---------------------------------------------------------------------------

test("CustomProvider subclass · textToSpeech override fires", async (t) => {
  let receivedText = null;

  class StubTts extends CustomProvider {
    constructor() {
      super({ providerId: "stub-tts" });
    }
    async textToSpeech(request) {
      receivedText = request.text;
      return {
        audio: [],
        timing: { totalMs: 0, queueMs: null, executionMs: null },
        metadata: { stub: true },
      };
    }
  }

  const provider = new StubTts();
  const result = await provider.textToSpeech({ text: "hello world" });

  t.is(receivedText, "hello world", "override should receive the request");
  t.deepEqual(result.audio, [], "stubbed result should round-trip back to JS");
  t.truthy(result.metadata, "stubbed metadata should be present");
});

// ---------------------------------------------------------------------------
// 2. Subclass `complete` routes to override
// ---------------------------------------------------------------------------
//
// `CustomProvider` does not expose `.complete()` directly to JS (the
// completion path is driven through `CompletionModel`). To verify the
// subclass override is wired into the trait adapter, wrap the subclass
// instance with `CompletionModel.custom(...)` and drive `.complete()`
// from there — the camelCase trait-method walk picks up the override.

test("CustomProvider subclass · complete override returns stubbed CompletionResponse", async (t) => {
  let completeCalls = 0;

  class StubLlm extends CustomProvider {
    constructor() {
      super({ providerId: "stub-llm" });
    }
    async complete(request) {
      completeCalls += 1;
      return {
        content: "hello from stub",
        toolCalls: [],
        model: "stub-llm",
        images: [],
        audio: [],
        videos: [],
        citations: [],
        artifacts: [],
        metadata: {},
      };
    }
  }

  const stub = new StubLlm();
  const model = CompletionModel.custom(stub, "stub-llm");

  const response = await model.complete([
    ChatMessage.user("ping"),
  ]);

  t.is(completeCalls, 1, "complete override should fire exactly once");
  t.is(response.content, "hello from stub");
  t.is(response.model, "stub-llm");
});

// ---------------------------------------------------------------------------
// 3. Unimplemented method rejects with Unsupported
// ---------------------------------------------------------------------------

test("CustomProvider · unimplemented method rejects with Unsupported", async (t) => {
  // NOTE: Use a direct `new CustomProvider(...)` (no subclass) here.
  //
  // A `class extends CustomProvider` instance currently traps the
  // `has_named_property` walk in `from_host_object` because every typed
  // napi method (e.g. `generateImage`) is exposed on
  // `CustomProvider.prototype` and therefore is reported as
  // "overridden" — calling the napi method then re-enters the adapter,
  // hanging the call. Validating the Unsupported surface against the
  // *direct* construction path keeps this test deterministic until that
  // binding issue is fixed; the subclass-routing semantics are still
  // covered by the `textToSpeech override fires` and `complete override
  // returns stubbed CompletionResponse` tests above.
  const provider = new CustomProvider({ providerId: "plain-custom" });
  const err = await t.throwsAsync(
    () => provider.generateImage({ prompt: "an unused capability" }),
  );
  t.truthy(err, "expected generateImage to reject");

  // The error layer either wraps as an `UnsupportedError` class or
  // preserves the `[UnsupportedError]` tag in the message — accept
  // either, since both are documented signals for callers.
  const isTypedClass =
    UnsupportedError !== null && err instanceof UnsupportedError;
  const carriesTag =
    /Unsupported/i.test(err.name ?? "") ||
    /Unsupported/i.test(err.message ?? "") ||
    /unsupported/i.test(err.message ?? "");
  t.true(
    isTypedClass || carriesTag,
    `expected an Unsupported-flavoured error, got ${err.name}: ${err.message}`,
  );
});

// ---------------------------------------------------------------------------
// 4. `ollama` factory constructs a provider with the expected providerId
// ---------------------------------------------------------------------------

test("CustomProvider · ollama factory constructs provider", (t) => {
  const provider = CustomProvider.ollama("llama3", "localhost", 11434);
  t.true(provider instanceof CustomProvider);
  t.is(provider.providerId, "ollama");
});

// ---------------------------------------------------------------------------
// 5. Prototype walk skips napi-installed methods (HalfImpl regression)
// ---------------------------------------------------------------------------
//
// Regression test for the `from_host_object` prototype-walk bug. Before
// the fix, `has_named_property` would report typed compute methods like
// `textToSpeech` as "overridden" on a subclass instance even when the
// subclass did NOT override them — because the `#[napi]` macro installs
// those methods on `CustomProvider.prototype`. Dispatching the phantom
// override then re-entered the napi binding and looped until the stack
// overflowed (RangeError).
//
// `HalfImpl` overrides ONLY `generateImage`. `textToSpeech` must report
// as Unsupported instead of recursing.

test("CustomProvider subclass · prototype walk skips napi-installed methods", async (t) => {
  class HalfImpl extends CustomProvider {
    constructor() {
      super({ providerId: "half-impl" });
    }
    async generateImage(_request) {
      return {
        images: [],
        timing: { totalMs: 0, queueMs: null, executionMs: null },
        metadata: { stub: true },
      };
    }
    // textToSpeech intentionally NOT overridden.
  }

  const provider = new HalfImpl();

  // generateImage should resolve via the user override.
  const imageResult = await provider.generateImage({ prompt: "a cat" });
  t.truthy(imageResult, "generateImage override should resolve");
  t.deepEqual(imageResult.images, [], "stubbed images round-trip back to JS");

  // textToSpeech should reject with Unsupported — NOT infinite-loop /
  // RangeError / stack overflow.
  const err = await t.throwsAsync(
    () => provider.textToSpeech({ text: "hello" }),
  );
  t.truthy(err, "expected textToSpeech to reject");

  const isTypedClass =
    UnsupportedError !== null && err instanceof UnsupportedError;
  const carriesTag =
    /Unsupported/i.test(err.name ?? "") ||
    /Unsupported/i.test(err.message ?? "") ||
    /not implemented/i.test(err.message ?? "") ||
    /does not override/i.test(err.message ?? "");
  t.true(
    isTypedClass || carriesTag,
    `expected an Unsupported-flavoured error, got ${err.name}: ${err.message}`,
  );
});

// ---------------------------------------------------------------------------
// 6. `extract` returns a parsed object via JSON Schema
// ---------------------------------------------------------------------------
//
// Stub `complete` on a CustomProvider subclass so it returns a JSON
// payload matching the supplied schema. Wrap via `CompletionModel.custom`
// so `BaseProvider.extract` has a concrete inner CompletionModel.

test("BaseProvider.extract · parses object via JSON Schema with subclass complete", async (t) => {
  class StubLlm extends CustomProvider {
    constructor() {
      super({ providerId: "stub-extract" });
    }
    async complete(_request) {
      return {
        content: '{"name":"Alice","age":30}',
        toolCalls: [],
        model: "stub-extract",
        images: [],
        audio: [],
        videos: [],
        citations: [],
        artifacts: [],
        metadata: {},
      };
    }
  }

  const stub = new StubLlm();
  const innerModel = CompletionModel.custom(stub, "stub-extract");
  const provider = new BaseProvider(innerModel);

  const schema = {
    type: "object",
    properties: {
      name: { type: "string" },
      age: { type: "integer" },
    },
    required: ["name", "age"],
  };

  const parsed = await provider.extract(schema, [
    ChatMessage.user("My name is Alice and I am 30."),
  ]);

  t.deepEqual(parsed, { name: "Alice", age: 30 });
});
