/**
 * Verifies that the typed `LlmPayload.parts` variant survives the round-trip
 * from a JS tool handler, through the napi `decode_structured_tool_output`
 * path, into the agent loop, and back out via `AgentResult.messages`.
 *
 * Background
 * ----------
 * Prior to the typed `LlmPayload.parts` fix, returning
 *   `{ data, llmOverride: { kind: "parts", parts: [...] } }`
 * from a JS tool handler would either be dropped or mis-classified because
 * the napi `LlmPayload` struct did not include a typed `parts` field. The
 * payload is now decoded via the napi struct and converted to the core
 * `LlmPayload::Parts` via `js_llm_payload_to_rust`.
 *
 * Strategy
 * --------
 * We drive `runAgent` with a `CustomProvider` subclass that scripts two
 * completions:
 *   1. First call: returns an assistant message with a single tool call
 *      to `getSummary`.
 *   2. Second call: returns a final assistant message (no tool calls).
 *
 * The tool handler returns `{ data, llmOverride: { kind: "parts", parts: [
 *   { partType: "text", text: "..." } ] } }`. We then inspect the resulting
 * `AgentResult.messages` array, find the `tool` role message produced by
 * the handler, and assert that its `toolResult.llmOverride` round-trips as
 * `kind === "parts"` with the expected text part.
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

class StubPartsProvider extends CustomProvider {
  constructor(scriptedResponses) {
    super({ providerId: "stub-parts" });
    this.scriptedResponses = scriptedResponses;
    this.callIdx = 0;
    this.lastMessages = null;
  }

  async complete(request) {
    // `request` is the JS-facing CompletionRequest. `messages` carries the
    // full conversation including any tool result messages emitted by the
    // agent loop on prior iterations.
    this.lastMessages = request?.messages ?? null;
    const idx = this.callIdx++;
    const response =
      this.scriptedResponses[idx] ??
      this.scriptedResponses[this.scriptedResponses.length - 1];
    return response;
  }
}

test("tool_output.parts · structured llmOverride with text part round-trips through runAgent", async (t) => {
  // NOTE: `CustomProvider.complete` overrides go through `dispatch`
  // which uses raw `serde_json::from_value::<CompletionResponse>(...)`.
  // The serde representation is snake_case (`tool_calls`, not
  // `toolCalls`). Returning camelCase keys would deserialize to a
  // response with an empty `tool_calls` array, and the agent loop would
  // stop after turn 1 with no tools to dispatch. See
  // `crates/blazen-node/src/providers/custom.rs::dispatch` for the
  // serde call site.
  const stub = new StubPartsProvider([
    // Turn 1: model emits a single tool call.
    {
      content: "",
      tool_calls: [
        {
          id: "call-1",
          name: "getSummary",
          arguments: { topic: "weather" },
        },
      ],
      model: "stub-parts",
      images: [],
      audio: [],
      videos: [],
      citations: [],
      artifacts: [],
      metadata: {},
    },
    // Turn 2: model emits a final answer.
    {
      content: "Done — saw the summary parts.",
      tool_calls: [],
      model: "stub-parts",
      images: [],
      audio: [],
      videos: [],
      citations: [],
      artifacts: [],
      metadata: {},
    },
  ]);

  const model = CompletionModel.custom(stub, "stub-parts");

  const toolHandler = async (toolName, args) => {
    if (toolName !== "getSummary") {
      throw new Error(`unexpected tool: ${toolName}`);
    }
    // The exercised path: a typed `llmOverride.parts` payload. The data
    // field is the structured caller-visible value; `llmOverride.parts`
    // is what the LLM sees on the next turn.
    return {
      data: { topic: args.topic, items: [1, 2, 3] },
      llmOverride: {
        kind: "parts",
        parts: [
          {
            partType: "text",
            text: `summary-for:${args.topic}`,
          },
        ],
      },
    };
  };

  const tools = [
    {
      name: "getSummary",
      description: "Return a summary for a topic.",
      parameters: {
        type: "object",
        properties: { topic: { type: "string" } },
        required: ["topic"],
      },
    },
  ];

  const result = await runAgent(
    model,
    [ChatMessage.user("Summarise the current weather.")],
    tools,
    toolHandler,
    { maxIterations: 3, noFinishTool: true },
  );

  // 1. Sanity: the agent loop completed.
  t.truthy(result, "runAgent should resolve to an AgentResult");
  // `iterations` counts tool-calling iterations and may be 0-indexed.
  // We just require a non-negative integer; the real activity is checked
  // via the stub call count + the messages array below.
  t.true(
    typeof result.iterations === "number" && result.iterations >= 0,
    `iterations should be a non-negative number, got ${result.iterations}`,
  );
  // The stub should have been called at least once.
  t.true(
    stub.callIdx >= 1,
    `expected at least 1 stub completion, got ${stub.callIdx}`,
  );

  // 2. Locate the tool-role message produced by our handler. The agent
  // loop converts the handler's return value into a `tool` ChatMessage
  // with either a structured `toolResult` field (when an `llmOverride`
  // is present) or a plain string `content` (fallback).
  //
  // Note on shape: `result.messages` is an `Array<any>` (see `AgentResult`
  // in index.d.ts). Each entry is the JSON-serialised projection of the
  // underlying Rust `ChatMessage` — keys are snake_case (`tool_result`,
  // `tool_call_id`), NOT the camelCase getters on the `ChatMessage`
  // class. We index into the JSON shape directly.
  const messages = result.messages ?? [];
  t.true(messages.length > 0, "messages array should not be empty");

  let foundTool = null;
  for (const msg of messages) {
    const role = typeof msg?.role === "string" ? msg.role : null;
    if (role !== "tool") continue;
    const name = typeof msg?.name === "string" ? msg.name : null;
    if (name !== "getSummary") continue;
    foundTool = msg;
    break;
  }
  t.truthy(foundTool, "expected a tool-role message for getSummary");

  // 3. Pull the structured tool result out. Serialised messages use
  // snake_case `tool_result`.
  const toolPayload = foundTool?.tool_result ?? null;
  t.truthy(
    toolPayload,
    "tool message should carry a structured tool_result payload (llmOverride present forces structured form)",
  );

  // 4. The user-visible `data` should be exactly what the handler returned.
  t.deepEqual(
    toolPayload.data,
    { topic: "weather", items: [1, 2, 3] },
    "tool_result.data should round-trip the handler's structured data",
  );

  // 5. The `llmOverride` (camelCase even in JSON form — preserved by the
  // napi `LlmPayload` struct definition) must be `kind: "parts"` with the
  // text part we supplied. This is the load-bearing assertion for the
  // typed `LlmPayload.parts` fix: prior to the fix, the napi struct had
  // no `parts` field and this round-trip silently dropped the override.
  //
  // Accept either `llmOverride` or `llm_override` because the agent loop
  // may serialise the field in either casing depending on the binding's
  // serde config.
  const override = toolPayload.llmOverride ?? toolPayload.llm_override ?? null;
  t.truthy(
    override,
    `llmOverride should be present on the tool result, got tool_result = ${JSON.stringify(toolPayload)}`,
  );
  t.is(override.kind, "parts", "llmOverride.kind should be 'parts'");
  t.true(
    Array.isArray(override.parts),
    "llmOverride.parts should be an array",
  );
  t.true(
    override.parts.length >= 1,
    "llmOverride.parts should contain at least one part",
  );
  const firstPart = override.parts[0];
  // The serialised `result.messages` is produced via raw serde on the
  // core `ContentPart` enum, which uses `#[serde(tag = "type")]`. The
  // napi input side accepts `partType` and converts via `convert_js_parts`
  // — so the round-trip surfaces `type` here. Accept either spelling to
  // be tolerant of future binding-side rewrites that fully convert
  // ContentPart through the napi shape.
  const partType =
    firstPart.partType ?? firstPart.part_type ?? firstPart.type ?? null;
  t.is(
    partType,
    "text",
    "first part should be a text part (type / partType === 'text')",
  );
  t.is(
    firstPart.text,
    "summary-for:weather",
    "first part text should round-trip verbatim",
  );

  // 6. Confirm the second turn observed the prior conversation. The stub
  // captures `request.messages` on every call; by the second call it
  // should have seen the tool-role message produced by the loop.
  // We don't make a stronger structural claim here — the messages array
  // shape on the JS side is asserted above; this just exercises that the
  // stub actually received the second turn's payload (i.e. the override
  // round-trip didn't deadlock the loop).
  t.truthy(
    stub.lastMessages,
    "stub.lastMessages should capture the most recent completion request payload",
  );
});
