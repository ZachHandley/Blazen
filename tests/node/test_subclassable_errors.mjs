/**
 * Verifies that a custom `Error` subclass thrown from a JS tool handler
 * passed to `runAgent` is re-thrown verbatim with `instanceof` matching
 * and all custom properties preserved.
 *
 * Background
 * ----------
 * `error-classes.js` wraps the user's `toolHandler` arg via
 * `wrapToolHandlerForCallerErrors`, which envelope-formats thrown errors
 * as `{ __blazenOk: false, errorRef: uuid, ... }` and stashes the
 * ORIGINAL JS Error instance in a Map keyed by the UUID.
 *
 * On the way out, the napi side surfaces the failure with a
 * `__BLAZEN_CALLER_ERROR__` sentinel; `enrichError` parses the JSON
 * payload, looks up the stashed original, and re-throws it — preserving
 * `instanceof CustomError`, `.name`, and arbitrary custom fields.
 *
 * Strategy
 * --------
 * Two complementary tests:
 *
 *   1. End-to-end through `runAgent`: a scripted stub provider emits one
 *      tool call; the handler throws `SubmitEvaluationSignal`. We assert
 *      the rejection preserves the subclass identity and `.payload`.
 *
 *   2. Direct envelope round-trip: drives the
 *      `wrapToolHandlerForCallerErrors` + `enrichError` pair in isolation.
 *      This catches regressions even on builds where the native binding is
 *      stale or where the scripted-stub-driven path skips.
 *
 * No API keys / network calls required. Build the native binding first:
 *   pnpm --filter blazen run build
 */

import test from "ava";

import {
  ChatMessage,
  CompletionModel,
  CustomProvider,
  runAgent,
} from "../../crates/blazen-node/index.js";

// `error-classes.js` is a CJS module re-exported by index.js; pull the
// internal helpers off it via a dynamic import so the direct envelope
// test can exercise them without going through the native binding.
import * as errorClasses from "../../crates/blazen-node/error-classes.js";

// ---------------------------------------------------------------------------
// Custom Error subclass under test
// ---------------------------------------------------------------------------

class SubmitEvaluationSignal extends Error {
  constructor(payload) {
    super("submit-evaluation");
    this.name = "SubmitEvaluationSignal";
    this.payload = payload;
  }
}

// ---------------------------------------------------------------------------
// Scripted stub provider — a single tool-calling completion is enough.
// ---------------------------------------------------------------------------

class StubErrorProvider extends CustomProvider {
  constructor() {
    super({ providerId: "stub-error" });
    this.callIdx = 0;
  }

  async complete(_request) {
    this.callIdx += 1;
    // Always emit the same tool call; the handler will throw before the
    // agent loop gets a chance to ask for a second completion. If for any
    // reason it does, returning the same shape is harmless.
    // NOTE: snake_case `tool_calls` (not `toolCalls`) — the JS->Rust
    // bridge here uses raw serde from a `CompletionResponse` whose serde
    // representation matches the Rust field name. See
    // `crates/blazen-node/src/providers/custom.rs::dispatch`.
    return {
      content: "",
      tool_calls: [
        {
          id: "call-1",
          name: "submitEval",
          arguments: { score: 9.5 },
        },
      ],
      model: "stub-error",
      images: [],
      audio: [],
      videos: [],
      citations: [],
      artifacts: [],
      metadata: {},
    };
  }
}

// ---------------------------------------------------------------------------
// 1. End-to-end via runAgent
// ---------------------------------------------------------------------------

test("runAgent · throws subclassable error from tool handler preserving instanceof + payload", async (t) => {
  const stub = new StubErrorProvider();
  const model = CompletionModel.custom(stub, "stub-error");

  const toolHandler = async (toolName, args) => {
    if (toolName === "submitEval") {
      throw new SubmitEvaluationSignal({
        score: args.score,
        finished: true,
      });
    }
    return null;
  };

  const tools = [
    {
      name: "submitEval",
      description: "Submit an evaluation score.",
      parameters: {
        type: "object",
        properties: { score: { type: "number" } },
        required: ["score"],
      },
    },
  ];

  const err = await t.throwsAsync(() =>
    runAgent(
      model,
      [ChatMessage.user("Evaluate this.")],
      tools,
      toolHandler,
      { maxIterations: 3, noFinishTool: true },
    ),
  );

  t.truthy(err, "runAgent should reject when the tool handler throws");

  // Load-bearing: `instanceof` matches the original subclass. Without
  // the `enrichError` re-throw path, this would be a generic Error.
  t.true(
    err instanceof SubmitEvaluationSignal,
    `expected SubmitEvaluationSignal, got ${err?.constructor?.name}: ${err?.message}`,
  );
  t.true(err instanceof Error, "should still be an Error subclass");
  t.is(err.name, "SubmitEvaluationSignal", "name should round-trip");
  t.deepEqual(
    err.payload,
    { score: 9.5, finished: true },
    "custom .payload property should be preserved verbatim",
  );
});

// ---------------------------------------------------------------------------
// 2. Direct envelope round-trip — no napi dependency.
// ---------------------------------------------------------------------------
//
// Drives `wrapToolHandlerForCallerErrors` and `enrichError` together,
// simulating exactly what the napi layer does between them: format a
// sentinel-prefixed message and hand it to `enrichError`. This guards
// against regressions in either half of the envelope flow even when the
// runAgent integration is skipped or when changes to the native binding
// would otherwise mask a JS-side break.

test("error-classes · wrapToolHandlerForCallerErrors + enrichError round-trip preserves custom Error subclass", async (t) => {
  const {
    wrapToolHandlerForCallerErrors,
    enrichError,
    CALLER_ERROR_SENTINEL,
    callerErrorStash,
  } = errorClasses;

  t.is(typeof wrapToolHandlerForCallerErrors, "function");
  t.is(typeof enrichError, "function");
  t.is(typeof CALLER_ERROR_SENTINEL, "string");

  class MyErr extends Error {
    constructor() {
      super("boom");
      this.name = "MyErr";
      this.x = 1;
      this.context = { reason: "explosion", retryable: false };
    }
  }

  const wrapped = wrapToolHandlerForCallerErrors(async () => {
    throw new MyErr();
  });

  const envelope = await wrapped("toolName", {});

  // Envelope contract: `{__blazenOk: false, errorRef, errorName, errorMessage}`.
  t.is(envelope.__blazenOk, false, "envelope should be a failure envelope");
  t.is(typeof envelope.errorRef, "string", "errorRef should be a UUID string");
  t.true(envelope.errorRef.length > 0, "errorRef should be non-empty");
  t.is(envelope.errorName, "MyErr", "errorName should match");
  t.is(envelope.errorMessage, "boom", "errorMessage should match");

  // The stash should now contain our original error under the ref. This
  // is the bridge `enrichError` walks back across.
  t.true(
    callerErrorStash.has(envelope.errorRef),
    "stash should contain the errorRef key after envelope formation",
  );

  // Simulate the napi side formatting the sentinel-prefixed error.
  // Format mirrors `error.rs`:
  //   `__BLAZEN_CALLER_ERROR__ {json}\n[CallerError] msg`
  const sentinelPayload = {
    ref: envelope.errorRef,
    name: "MyErr",
    message: "boom",
  };
  const napiErr = new Error(
    `${CALLER_ERROR_SENTINEL} ${JSON.stringify(sentinelPayload)}\n[CallerError] boom`,
  );

  const enriched = enrichError(napiErr);

  // Load-bearing: the original MyErr instance is returned VERBATIM. Same
  // class, same custom fields, same `instanceof` semantics.
  t.true(enriched instanceof MyErr, "enriched value should be a MyErr instance");
  t.true(enriched instanceof Error, "should still be an Error subclass");
  t.is(enriched.name, "MyErr");
  t.is(enriched.message, "boom");
  t.is(enriched.x, 1, "custom scalar property should be preserved");
  t.deepEqual(
    enriched.context,
    { reason: "explosion", retryable: false },
    "custom nested-object property should be preserved by reference",
  );

  // After the lookup, the stash entry is consumed (idempotent cleanup).
  t.false(
    callerErrorStash.has(envelope.errorRef),
    "stash should be cleared after enrichError consumes the ref",
  );
});

// ---------------------------------------------------------------------------
// 3. Direct envelope fallback — unknown ref still surfaces name/message.
// ---------------------------------------------------------------------------
//
// If the stash entry is missing for any reason (e.g. cross-process
// boundary, GC, manual cleanup) `enrichError` still produces a generic
// `Error` with the embedded name/message. Document this contract.

test("error-classes · enrichError fallback when stash ref is missing surfaces name + message", (t) => {
  const { enrichError, CALLER_ERROR_SENTINEL } = errorClasses;

  const payload = {
    ref: "00000000-0000-0000-0000-000000000000", // not in the stash
    name: "GhostError",
    message: "vanished",
  };
  const napiErr = new Error(
    `${CALLER_ERROR_SENTINEL} ${JSON.stringify(payload)}\n[CallerError] vanished`,
  );

  const enriched = enrichError(napiErr);

  t.true(enriched instanceof Error);
  t.is(enriched.name, "GhostError", "fallback name should come from payload");
  t.is(enriched.message, "vanished", "fallback message should come from payload");
});
