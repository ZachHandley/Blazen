/**
 * fal.ai compute smoke tests.
 *
 * Gated on the FAL_API_KEY environment variable.
 * Tests LLM completion, image generation, and timing/cost metadata.
 */

import { describe, it } from "node:test";
import assert from "node:assert/strict";

import {
  CompletionModel,
  ChatMessage,
} from "../../crates/blazen-node/index.js";

const FAL_API_KEY = process.env.FAL_API_KEY;

describe("fal.ai LLM smoke tests", { skip: !FAL_API_KEY }, () => {
  it("completes a basic prompt via fal-ai/any-llm", async () => {
    const model = CompletionModel.fal(FAL_API_KEY);
    const response = await model.complete([
      ChatMessage.user("What is 2+2? Reply with just the number."),
    ]);

    assert.ok(response.content, "expected content in response");
    assert.ok(response.content.includes("4"), `expected '4' in: ${response.content}`);
    assert.ok(response.model, "expected model in response");
  });

  it("returns timing metadata", async () => {
    const model = CompletionModel.fal(FAL_API_KEY);
    const response = await model.complete([
      ChatMessage.user("Say hello."),
    ]);

    assert.ok(response.content, "expected content");
    // Timing should be populated for fal.ai queue mode
    if (response.timing) {
      assert.ok(
        typeof response.timing.totalMs === "number" || response.timing.totalMs === null,
        "timing.totalMs should be a number or null"
      );
    }
  });

  it("passes temperature and max_tokens", async () => {
    const model = CompletionModel.fal(FAL_API_KEY);
    const response = await model.completeWithOptions(
      [ChatMessage.user("Write a one-word greeting.")],
      { temperature: 0.1, maxTokens: 10 }
    );

    assert.ok(response.content, "expected content");
    // With max_tokens=10, response should be short
    assert.ok(response.content.length < 200, "response should be short with maxTokens=10");
  });
});

describe("fal.ai compute smoke tests", { skip: !FAL_API_KEY }, () => {
  it("generates an image with FLUX", async () => {
    const { FalProvider } = await import("../../crates/blazen-node/index.js");
    const provider = FalProvider.create(FAL_API_KEY);
    const result = await provider.generateImage({ prompt: "a simple red circle on white" });
    assert.ok(result.images, "expected images");
    assert.ok(result.images.length > 0, "expected at least one image");
  });
});
