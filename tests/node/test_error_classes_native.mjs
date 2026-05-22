/**
 * Native Error-class regression tests.
 *
 * Verifies the napi 3.9.0 + `napi::register_error_class` migration:
 *
 *   1. The full class hierarchy is exported from `blazen`:
 *      BlazenError, ProviderError, AuthError, ..., LlamaCppError, ..., etc.
 *      All extend `Error` and form the documented subclass chain.
 *   2. A Rust mapper that emits ProviderError surfaces in JS as a real
 *      `instanceof ProviderError === true` instance with `instanceof
 *      BlazenError === true` and `instanceof Error === true`. Structured
 *      fields (`provider`, `status`, etc.) are present as own properties.
 *   3. The previously-deadly path -- a user toolHandler synchronously
 *      throwing a custom JS error subclass -- no longer SIGABRTs the
 *      host. With napi 3.9.0's `call_async_catch`, the original JS
 *      error instance round-trips through Rust and arrives back on the
 *      JS side as `instanceof MyError === true` with custom properties
 *      intact.
 *
 * No API keys / network calls required. Build the native binding first:
 *   pnpm --filter blazen run build
 */

import test from "ava";

import {
  ChatMessage,
  Model,
  CustomProvider,
  runAgent,
} from "../../crates/blazen-node/index.js";
import * as blazen from "../../crates/blazen-node/index.js";

// ---------------------------------------------------------------------------
// 1. Hierarchy: every documented class is exported and extends Error
// ---------------------------------------------------------------------------

const EXPECTED_CLASSES = [
  "BlazenError",
  "AuthError",
  "RateLimitError",
  "TimeoutError",
  "ValidationError",
  "ContentPolicyError",
  "UnsupportedError",
  "ComputeError",
  "MediaError",
  "ProviderError",
  "LlamaCppError",
  "CandleLlmError",
  "CandleEmbedError",
  "MistralRsError",
  "WhisperError",
  "PiperError",
  "DiffusionError",
  "FastEmbedError",
  "TractError",
  "PeerEncodeError",
  "PeerTransportError",
  "PersistError",
  "PromptError",
  "MemoryError",
  "CacheError",
];

for (const className of EXPECTED_CLASSES) {
  test(`Native error classes - ${className} is exported and extends Error`, (t) => {
    const Cls = blazen[className];
    t.is(typeof Cls, "function", `${className} should be exported as a class`);
    const instance = new Cls("test message");
    t.true(instance instanceof Error, `${className} instance should be an Error`);
    t.is(instance.name, className, `${className}.name should match`);
    t.is(instance.message, "test message", "message should round-trip");
  });
}

// ---------------------------------------------------------------------------
// 2. Subclass chains link up properly
// ---------------------------------------------------------------------------

test("Native error classes - ProviderError extends BlazenError extends Error", (t) => {
  const e = new blazen.ProviderError("rate limited");
  t.true(e instanceof blazen.ProviderError);
  t.true(e instanceof blazen.BlazenError);
  t.true(e instanceof Error);
});

test("Native error classes - per-backend subclass chains: LlamaCppError -> ProviderError -> BlazenError -> Error", (t) => {
  const e = new blazen.LlamaCppError("backend boom");
  t.true(e instanceof blazen.LlamaCppError);
  t.true(e instanceof blazen.ProviderError);
  t.true(e instanceof blazen.BlazenError);
  t.true(e instanceof Error);
});

test("Native error classes - MemoryNotFoundError -> MemoryError -> BlazenError -> Error", (t) => {
  const e = new blazen.MemoryNotFoundError("session not found");
  t.true(e instanceof blazen.MemoryNotFoundError);
  t.true(e instanceof blazen.MemoryError);
  t.true(e instanceof blazen.BlazenError);
  t.true(e instanceof Error);
});

test("Native error classes - PromptValidationError extends PromptError extends BlazenError", (t) => {
  const e = new blazen.PromptValidationError("schema mismatch");
  t.true(e instanceof blazen.PromptValidationError);
  t.true(e instanceof blazen.PromptError);
  t.true(e instanceof blazen.BlazenError);
});

// ---------------------------------------------------------------------------
// 3. Structured fields attached via `with_field` arrive as own properties
// ---------------------------------------------------------------------------

test("Native error classes - constructed instance accepts structured fields via props arg", (t) => {
  const e = new blazen.ProviderError("fal rejected", {
    provider: "fal",
    status: 503,
    endpoint: "https://fal.run/x",
    requestId: "abc-123",
    detail: "service unavailable",
    retryAfterMs: 5000,
  });
  t.is(e.provider, "fal");
  t.is(e.status, 503);
  t.is(e.endpoint, "https://fal.run/x");
  t.is(e.requestId, "abc-123");
  t.is(e.detail, "service unavailable");
  t.is(e.retryAfterMs, 5000);
});

// ---------------------------------------------------------------------------
// 4. Rust-emitted error: CustomProvider raises UnsupportedError for un-overridden methods
//
//    This exercises the full Rust->JS error path: the trait dispatcher
//    in `crates/blazen-node/src/providers/custom.rs` returns
//    `Err(BlazenError::Unsupported {...})`, the error mapper converts it
//    via `Error::with_class("UnsupportedError", ...)`, and the JS caller
//    sees a real `UnsupportedError` instance.
// ---------------------------------------------------------------------------

test("Native error classes - Rust-emitted UnsupportedError is the actual class on the JS side", async (t) => {
  // Direct `new CustomProvider(...)` (no subclass) — exercising the
  // un-overridden capability path that returns BlazenError::Unsupported.
  // (Subclass instances trap on the napi method-walk; see the analogous
  // test in test_custom_provider_subclass.mjs for the gnarly background.)
  const provider = new CustomProvider({ providerId: "plain-custom-stub" });
  await t.throwsAsync(
    () => provider.generateImage({ prompt: "an unused capability" }),
    { instanceOf: blazen.UnsupportedError },
  );
});

// ---------------------------------------------------------------------------
// 5. Caller errors: a user toolHandler that throws a custom JS Error
//    subclass synchronously is preserved across the Rust agent loop
//    boundary. Before napi 3.9.0's `call_async_catch`, this scenario
//    SIGABRTed the host.
// ---------------------------------------------------------------------------

class SignalDone extends Error {
  constructor(payload) {
    super("signal-done");
    this.name = "SignalDone";
    this.payload = payload;
  }
}

class ScriptedProvider extends CustomProvider {
  constructor() {
    super({ providerId: "scripted-stub" });
    this.calls = 0;
  }
  async complete(_request) {
    this.calls += 1;
    return {
      content: null,
      // snake_case to match `ModelResponse`'s serde naming
      tool_calls: [
        {
          id: "call-1",
          name: "trigger",
          arguments: { reason: "boom" },
        },
      ],
      model: "scripted-stub:test-model",
      finish_reason: "tool_use",
      usage: { prompt_tokens: 1, completion_tokens: 1, total_tokens: 2 },
    };
  }
}

test("Native error classes - toolHandler can throw a custom JS Error subclass without crashing the host", async (t) => {
  const provider = new ScriptedProvider();
  const model = Model.custom(provider, "scripted-stub:test-model");
  const messages = [ChatMessage.user("trigger the signal")];

  const toolHandler = (_name, args) => {
    // Synchronous throw inside an async-friendly handler: napi 3.9.0's
    // `call_async_catch` should capture this as an Err, the agent loop's
    // CallerError stash should preserve the original instance, and the
    // outer Promise.reject from runAgent should surface that exact
    // SignalDone instance.
    throw new SignalDone({ from: args.reason });
  };

  const toolDef = {
    name: "trigger",
    description: "Throws a SignalDone to test error round-trip",
    parameters: {
      type: "object",
      properties: { reason: { type: "string" } },
      required: ["reason"],
    },
  };
  const err = await t.throwsAsync(async () =>
    runAgent(model, messages, [toolDef], toolHandler, {}),
  );
  t.true(err instanceof SignalDone, "should be the original custom error instance");
  t.is(err.name, "SignalDone");
  t.deepEqual(err.payload, { from: "boom" }, "custom payload should round-trip");
  t.true(process.pid > 0, "host should still be alive");
});
