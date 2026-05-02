/**
 * mistral.rs local LLM smoke tests.
 *
 * Gated on the BLAZEN_TEST_MISTRALRS environment variable.
 * Only runs when the native binding is compiled with the `mistralrs` feature.
 *
 * Build first:
 *   cd crates/blazen-node && npm install && npm run build -- --features mistralrs
 */

import test from "ava";

import { CompletionModel, ChatMessage } from "../../crates/blazen-node/index.js";

const BLAZEN_TEST_MISTRALRS = process.env.BLAZEN_TEST_MISTRALRS;

const MODEL_ID = "TheBloke/Mistral-7B-Instruct-v0.2-GGUF";

const T = BLAZEN_TEST_MISTRALRS ? test : test.skip;

T("mistral.rs local LLM · completes a prompt and returns non-empty content", async (t) => {
  // Feature gate: if mistralrs factory is not available, skip gracefully.
  if (typeof CompletionModel.mistralrs !== "function") {
    t.pass("mistralrs feature not built");
    return; // not built with mistralrs feature
  }

  const model = CompletionModel.mistralrs({ modelId: MODEL_ID });

  const response = await model.complete([
    ChatMessage.user("What is 2+2? Answer with just the number."),
  ]);

  t.truthy(response.content, "expected non-empty response content");
  t.truthy(response.content.length > 0, "content should not be empty");
});

T("mistral.rs local LLM · exposes model_id on the constructed model", async (t) => {
  if (typeof CompletionModel.mistralrs !== "function") {
    t.pass("mistralrs feature not built");
    return;
  }

  const model = CompletionModel.mistralrs({ modelId: MODEL_ID });

  t.truthy(model.modelId, "expected a non-empty model ID");
  t.truthy(model.modelId.length > 0, "modelId should not be empty");
});

T("mistral.rs local LLM · handles system + user message pairs", async (t) => {
  if (typeof CompletionModel.mistralrs !== "function") {
    t.pass("mistralrs feature not built");
    return;
  }

  const model = CompletionModel.mistralrs({ modelId: MODEL_ID });

  const response = await model.complete([
    ChatMessage.system("You are a helpful assistant. Be concise."),
    ChatMessage.user("What is the capital of France?"),
  ]);

  t.truthy(response.content, "expected non-empty response content");
  t.truthy(response.content.length > 0, "content should not be empty");
});
