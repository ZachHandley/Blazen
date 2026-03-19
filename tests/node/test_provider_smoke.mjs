/**
 * Per-provider LLM smoke tests.
 *
 * Each provider is gated on its own API key environment variable.
 * Only runs during release CI or manual invocation.
 */

import { describe, it } from "node:test";
import assert from "node:assert/strict";

import {
  CompletionModel,
  ChatMessage,
} from "../../crates/blazen-node/index.js";

// ── OpenAI ──────────────────────────────────────────────────────────────────

const OPENAI_API_KEY = process.env.OPENAI_API_KEY;

describe("OpenAI smoke tests", { skip: !OPENAI_API_KEY }, () => {
  it("completes a basic prompt", async () => {
    const model = CompletionModel.openai(OPENAI_API_KEY);
    const response = await model.complete([
      ChatMessage.user("What is 2+2? Reply with just the number."),
    ]);

    assert.ok(response.content, "expected content in response");
    assert.ok(response.content.includes("4"), `expected '4' in: ${response.content}`);
    assert.ok(response.model, "expected model in response");
  });
});

// ── Anthropic ───────────────────────────────────────────────────────────────

const ANTHROPIC_API_KEY = process.env.ANTHROPIC_API_KEY;

describe("Anthropic smoke tests", { skip: !ANTHROPIC_API_KEY }, () => {
  it("completes a basic prompt", async () => {
    const model = CompletionModel.anthropic(ANTHROPIC_API_KEY);
    const response = await model.complete([
      ChatMessage.user("What is 2+2? Reply with just the number."),
    ]);

    assert.ok(response.content, "expected content in response");
    assert.ok(response.content.includes("4"), `expected '4' in: ${response.content}`);
    assert.ok(response.model, "expected model in response");
  });
});

// ── Gemini ──────────────────────────────────────────────────────────────────

const GEMINI_API_KEY = process.env.GEMINI_API_KEY;

describe("Gemini smoke tests", { skip: !GEMINI_API_KEY }, () => {
  it("completes a basic prompt", async () => {
    const model = CompletionModel.gemini(GEMINI_API_KEY);
    const response = await model.complete([
      ChatMessage.user("What is 2+2? Reply with just the number."),
    ]);

    assert.ok(response.content, "expected content in response");
    assert.ok(response.content.includes("4"), `expected '4' in: ${response.content}`);
    assert.ok(response.model, "expected model in response");
  });
});

// ── Groq ────────────────────────────────────────────────────────────────────

const GROQ_API_KEY = process.env.GROQ_API_KEY;

describe("Groq smoke tests", { skip: !GROQ_API_KEY }, () => {
  it("completes a basic prompt", async () => {
    const model = CompletionModel.groq(GROQ_API_KEY);
    const response = await model.complete([
      ChatMessage.user("What is 2+2? Reply with just the number."),
    ]);

    assert.ok(response.content, "expected content in response");
    assert.ok(response.content.includes("4"), `expected '4' in: ${response.content}`);
    assert.ok(response.model, "expected model in response");
  });
});

// ── Together ────────────────────────────────────────────────────────────────

const TOGETHER_API_KEY = process.env.TOGETHER_API_KEY;

describe("Together smoke tests", { skip: !TOGETHER_API_KEY }, () => {
  it("completes a basic prompt", async () => {
    const model = CompletionModel.together(TOGETHER_API_KEY);
    const response = await model.complete([
      ChatMessage.user("What is 2+2? Reply with just the number."),
    ]);

    assert.ok(response.content, "expected content in response");
    assert.ok(response.content.includes("4"), `expected '4' in: ${response.content}`);
    assert.ok(response.model, "expected model in response");
  });
});

// ── Mistral ─────────────────────────────────────────────────────────────────

const MISTRAL_API_KEY = process.env.MISTRAL_API_KEY;

describe("Mistral smoke tests", { skip: !MISTRAL_API_KEY }, () => {
  it("completes a basic prompt", async () => {
    const model = CompletionModel.mistral(MISTRAL_API_KEY);
    const response = await model.complete([
      ChatMessage.user("What is 2+2? Reply with just the number."),
    ]);

    assert.ok(response.content, "expected content in response");
    assert.ok(response.content.includes("4"), `expected '4' in: ${response.content}`);
    assert.ok(response.model, "expected model in response");
  });
});

// ── DeepSeek ────────────────────────────────────────────────────────────────

const DEEPSEEK_API_KEY = process.env.DEEPSEEK_API_KEY;

describe("DeepSeek smoke tests", { skip: !DEEPSEEK_API_KEY }, () => {
  it("completes a basic prompt", async () => {
    const model = CompletionModel.deepseek(DEEPSEEK_API_KEY);
    const response = await model.complete([
      ChatMessage.user("What is 2+2? Reply with just the number."),
    ]);

    assert.ok(response.content, "expected content in response");
    assert.ok(response.content.includes("4"), `expected '4' in: ${response.content}`);
    assert.ok(response.model, "expected model in response");
  });
});

// ── Fireworks ───────────────────────────────────────────────────────────────

const FIREWORKS_API_KEY = process.env.FIREWORKS_API_KEY;

describe("Fireworks smoke tests", { skip: !FIREWORKS_API_KEY }, () => {
  it("completes a basic prompt", async () => {
    const model = CompletionModel.fireworks(FIREWORKS_API_KEY);
    const response = await model.complete([
      ChatMessage.user("What is 2+2? Reply with just the number."),
    ]);

    assert.ok(response.content, "expected content in response");
    assert.ok(response.content.includes("4"), `expected '4' in: ${response.content}`);
    assert.ok(response.model, "expected model in response");
  });
});

// ── Perplexity ──────────────────────────────────────────────────────────────

const PERPLEXITY_API_KEY = process.env.PERPLEXITY_API_KEY;

describe("Perplexity smoke tests", { skip: !PERPLEXITY_API_KEY }, () => {
  it("completes a basic prompt", async () => {
    const model = CompletionModel.perplexity(PERPLEXITY_API_KEY);
    const response = await model.complete([
      ChatMessage.user("What is 2+2? Reply with just the number."),
    ]);

    assert.ok(response.content, "expected content in response");
    assert.ok(response.content.includes("4"), `expected '4' in: ${response.content}`);
    assert.ok(response.model, "expected model in response");
  });
});

// ── xAI ─────────────────────────────────────────────────────────────────────

const XAI_API_KEY = process.env.XAI_API_KEY;

describe("xAI smoke tests", { skip: !XAI_API_KEY }, () => {
  it("completes a basic prompt", async () => {
    const model = CompletionModel.xai(XAI_API_KEY);
    const response = await model.complete([
      ChatMessage.user("What is 2+2? Reply with just the number."),
    ]);

    assert.ok(response.content, "expected content in response");
    assert.ok(response.content.includes("4"), `expected '4' in: ${response.content}`);
    assert.ok(response.model, "expected model in response");
  });
});

// ── Cohere ──────────────────────────────────────────────────────────────────

const COHERE_API_KEY = process.env.COHERE_API_KEY;

describe("Cohere smoke tests", { skip: !COHERE_API_KEY }, () => {
  it("completes a basic prompt", async () => {
    const model = CompletionModel.cohere(COHERE_API_KEY);
    const response = await model.complete([
      ChatMessage.user("What is 2+2? Reply with just the number."),
    ]);

    assert.ok(response.content, "expected content in response");
    assert.ok(response.content.includes("4"), `expected '4' in: ${response.content}`);
    assert.ok(response.model, "expected model in response");
  });
});
