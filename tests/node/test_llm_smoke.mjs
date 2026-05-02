/**
 * LLM smoke tests using OpenRouter.
 *
 * Gated on the OPENROUTER_API_KEY environment variable.
 * Only runs during release CI or manual invocation.
 */

import test from "ava";

import {
  Workflow,
  CompletionModel,
  ChatMessage,
} from "../../crates/blazen-node/index.js";

const OPENROUTER_API_KEY = process.env.OPENROUTER_API_KEY;

const T = OPENROUTER_API_KEY ? test : test.skip;

T("OpenRouter LLM smoke tests · completes a basic prompt", async (t) => {
  const model = CompletionModel.openrouter({ apiKey: OPENROUTER_API_KEY });
  const response = await model.complete([
    ChatMessage.user("What is 2+2? Reply with just the number."),
  ]);

  t.truthy(response.content, "expected content in response");
  t.truthy(response.content.includes("4"), `expected '4' in: ${response.content}`);
  t.truthy(response.model, "expected model in response");
});

T("OpenRouter LLM smoke tests · uses LLM inside a workflow step", async (t) => {
  const wf = new Workflow("llm-smoke");

  wf.addStep("ask", ["blazen::StartEvent"], async (event, ctx) => {
    const model = CompletionModel.openrouter({ apiKey: OPENROUTER_API_KEY });
    const response = await model.completeWithOptions(
      [
        ChatMessage.system("You are a math tutor. Reply with just the number."),
        ChatMessage.user(event.question),
      ],
      { maxTokens: 32 }
    );
    return {
      type: "blazen::StopEvent",
      result: { answer: response.content },
    };
  });

  const result = await wf.run({ question: "What is 3+3?" });
  t.truthy(result.data.answer, "expected answer in result");
  t.truthy(result.data.answer.includes("6"), `expected '6' in: ${result.data.answer}`);
});
