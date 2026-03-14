/**
 * LLM smoke tests using OpenRouter.
 *
 * Gated on the OPENROUTER_API_KEY environment variable.
 * Only runs during release CI or manual invocation.
 */

import { describe, it } from "node:test";
import assert from "node:assert/strict";

import {
  Workflow,
  CompletionModel,
} from "../../crates/blazen-node/index.js";

const OPENROUTER_API_KEY = process.env.OPENROUTER_API_KEY;

describe("OpenRouter LLM smoke tests", { skip: !OPENROUTER_API_KEY }, () => {
  it("completes a basic prompt", async () => {
    const model = CompletionModel.openrouter(OPENROUTER_API_KEY);
    const response = await model.complete([
      { role: "user", content: "What is 2+2? Reply with just the number." },
    ]);

    assert.ok(response.content, "expected content in response");
    assert.ok(response.content.includes("4"), `expected '4' in: ${response.content}`);
    assert.ok(response.model, "expected model in response");
  });

  it("uses LLM inside a workflow step", async () => {
    const wf = new Workflow("llm-smoke");

    wf.addStep("ask", ["blazen::StartEvent"], async (event, ctx) => {
      const model = CompletionModel.openrouter(OPENROUTER_API_KEY);
      const response = await model.completeWithOptions(
        [
          { role: "system", content: "You are a math tutor. Reply with just the number." },
          { role: "user", content: event.question },
        ],
        { maxTokens: 10 }
      );
      return {
        type: "blazen::StopEvent",
        result: { answer: response.content },
      };
    });

    const result = await wf.run({ question: "What is 3+3?" });
    assert.ok(result.data.answer, "expected answer in result");
    assert.ok(result.data.answer.includes("6"), `expected '6' in: ${result.data.answer}`);
  });
});
