/**
 * Capabilities smoke tests: streaming, structured output, and agent tool calling.
 *
 * Gated on the OPENROUTER_API_KEY environment variable.
 * Only runs during release CI or manual invocation.
 */

import test from "ava";

import {
  CompletionModel,
  ChatMessage,
  runAgent,
} from "../../crates/blazen-node/index.js";

const OPENROUTER_API_KEY = process.env.OPENROUTER_API_KEY;
const T = OPENROUTER_API_KEY ? test : test.skip;

// -- Streaming ---------------------------------------------------------------

T("Streaming smoke tests · streams chunks for a basic prompt", async (t) => {
  const model = CompletionModel.openrouter({ apiKey: OPENROUTER_API_KEY });
  const chunks = [];

  await model.stream(
    [ChatMessage.user("Count from 1 to 5.")],
    (chunk) => {
      chunks.push(chunk);
    }
  );

  t.truthy(chunks.length > 0, "expected at least one streamed chunk");

  // At least some chunks should carry a delta with text content
  const hasTextDelta = chunks.some(
    (c) => typeof c.delta === "string" && c.delta.length > 0
  );
  t.truthy(hasTextDelta, "expected at least one chunk with a non-empty delta");
});

// -- Structured Output -------------------------------------------------------

T("Structured output smoke tests · returns JSON conforming to a schema", async (t) => {
  const model = CompletionModel.openrouter({ apiKey: OPENROUTER_API_KEY });

  const response = await model.completeWithOptions(
    [ChatMessage.user("What is 2 + 2? Respond in the required JSON format.")],
    {
      responseFormat: {
        type: "json_schema",
        json_schema: {
          name: "math",
          strict: true,
          schema: {
            type: "object",
            properties: {
              answer: { type: "integer" },
            },
            required: ["answer"],
          },
        },
      },
    }
  );

  t.truthy(response.content, "expected content in response");

  const parsed = JSON.parse(response.content);
  t.is(parsed.answer, 4, `expected answer=4, got ${parsed.answer}`);
});

// -- Agent Tool Calling ------------------------------------------------------

T("Agent tool calling smoke tests · uses a multiply tool and returns the correct result", async (t) => {
  const model = CompletionModel.openrouter({ apiKey: OPENROUTER_API_KEY });

  const tools = [
    {
      name: "multiply",
      description: "Multiply two numbers together and return the product.",
      parameters: {
        type: "object",
        properties: {
          a: { type: "number", description: "First number" },
          b: { type: "number", description: "Second number" },
        },
        required: ["a", "b"],
      },
    },
  ];

  const toolHandler = async (toolName, args) => {
    if (toolName === "multiply") {
      return JSON.stringify({ result: args.a * args.b });
    }
    throw new Error(`Unknown tool: ${toolName}`);
  };

  const result = await runAgent(
    model,
    [ChatMessage.user("What is 15 * 7? Use the multiply tool.")],
    tools,
    toolHandler,
    { maxIterations: 5 }
  );

  t.truthy(result.response, "expected response in agent result");
  t.truthy(result.response.content, "expected content in agent response");
  t.truthy(
    result.response.content.includes("105"),
    `expected '105' in: ${result.response.content}`
  );
  t.truthy(
    result.iterations >= 1,
    `expected at least 1 iteration, got ${result.iterations}`
  );
});
