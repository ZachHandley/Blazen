/**
 * mistral.rs local LLM smoke tests.
 *
 * Gated on the BLAZEN_TEST_MISTRALRS environment variable.
 * Only runs when the native binding is compiled with the `mistralrs` feature.
 *
 * Build first:
 *   cd crates/blazen-node && npm install && npm run build -- --features mistralrs
 */

import { describe, it } from "node:test";
import assert from "node:assert/strict";

import { CompletionModel, ChatMessage } from "../../crates/blazen-node/index.js";

const BLAZEN_TEST_MISTRALRS = process.env.BLAZEN_TEST_MISTRALRS;

const MODEL_ID = "TheBloke/Mistral-7B-Instruct-v0.2-GGUF";

describe("mistral.rs local LLM", { skip: !BLAZEN_TEST_MISTRALRS }, () => {
  it("completes a prompt and returns non-empty content", async () => {
    // Feature gate: if mistralrs factory is not available, skip gracefully.
    if (typeof CompletionModel.mistralrs !== "function") {
      return; // not built with mistralrs feature
    }

    const model = CompletionModel.mistralrs({ modelId: MODEL_ID });

    const response = await model.complete([
      ChatMessage.user("What is 2+2? Answer with just the number."),
    ]);

    assert.ok(response.content, "expected non-empty response content");
    assert.ok(response.content.length > 0, "content should not be empty");
  });

  it("exposes model_id on the constructed model", async () => {
    if (typeof CompletionModel.mistralrs !== "function") {
      return;
    }

    const model = CompletionModel.mistralrs({ modelId: MODEL_ID });

    assert.ok(model.modelId, "expected a non-empty model ID");
    assert.ok(model.modelId.length > 0, "modelId should not be empty");
  });

  it("handles system + user message pairs", async () => {
    if (typeof CompletionModel.mistralrs !== "function") {
      return;
    }

    const model = CompletionModel.mistralrs({ modelId: MODEL_ID });

    const response = await model.complete([
      ChatMessage.system("You are a helpful assistant. Be concise."),
      ChatMessage.user("What is the capital of France?"),
    ]);

    assert.ok(response.content, "expected non-empty response content");
    assert.ok(response.content.length > 0, "content should not be empty");
  });
});
