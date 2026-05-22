/**
 * Per-provider LLM smoke tests.
 *
 * Each provider is gated on its own API key environment variable.
 * Only runs during release CI or manual invocation.
 */

import test from "ava";

import {
  Model,
  ChatMessage,
} from "../../crates/blazen-node/index.js";

// ── OpenAI ──────────────────────────────────────────────────────────────────

const OPENAI_API_KEY = process.env.OPENAI_API_KEY;
const TOpenAI = OPENAI_API_KEY ? test : test.skip;

TOpenAI("OpenAI smoke tests · completes a basic prompt", async (t) => {
  const model = Model.openai({ apiKey: OPENAI_API_KEY });
  const response = await model.complete([
    ChatMessage.user("What is 2+2? Reply with just the number."),
  ]);

  t.truthy(response.content, "expected content in response");
  t.truthy(response.content.includes("4"), `expected '4' in: ${response.content}`);
  t.truthy(response.model, "expected model in response");
});

// ── Anthropic ───────────────────────────────────────────────────────────────

const ANTHROPIC_API_KEY = process.env.ANTHROPIC_API_KEY;
const TAnthropic = ANTHROPIC_API_KEY ? test : test.skip;

TAnthropic("Anthropic smoke tests · completes a basic prompt", async (t) => {
  const model = Model.anthropic({ apiKey: ANTHROPIC_API_KEY });
  const response = await model.complete([
    ChatMessage.user("What is 2+2? Reply with just the number."),
  ]);

  t.truthy(response.content, "expected content in response");
  t.truthy(response.content.includes("4"), `expected '4' in: ${response.content}`);
  t.truthy(response.model, "expected model in response");
});

// ── Gemini ──────────────────────────────────────────────────────────────────

const GEMINI_API_KEY = process.env.GEMINI_API_KEY;
const TGemini = GEMINI_API_KEY ? test : test.skip;

TGemini("Gemini smoke tests · completes a basic prompt", async (t) => {
  const model = Model.gemini({ apiKey: GEMINI_API_KEY });
  const response = await model.complete([
    ChatMessage.user("What is 2+2? Reply with just the number."),
  ]);

  t.truthy(response.content, "expected content in response");
  t.truthy(response.content.includes("4"), `expected '4' in: ${response.content}`);
  t.truthy(response.model, "expected model in response");
});

// ── Groq ────────────────────────────────────────────────────────────────────

const GROQ_API_KEY = process.env.GROQ_API_KEY;
const TGroq = GROQ_API_KEY ? test : test.skip;

TGroq("Groq smoke tests · completes a basic prompt", async (t) => {
  const model = Model.groq({ apiKey: GROQ_API_KEY });
  const response = await model.complete([
    ChatMessage.user("What is 2+2? Reply with just the number."),
  ]);

  t.truthy(response.content, "expected content in response");
  t.truthy(response.content.includes("4"), `expected '4' in: ${response.content}`);
  t.truthy(response.model, "expected model in response");
});

// ── Together ────────────────────────────────────────────────────────────────

const TOGETHER_API_KEY = process.env.TOGETHER_API_KEY;
const TTogether = TOGETHER_API_KEY ? test : test.skip;

TTogether("Together smoke tests · completes a basic prompt", async (t) => {
  const model = Model.together({ apiKey: TOGETHER_API_KEY });
  const response = await model.complete([
    ChatMessage.user("What is 2+2? Reply with just the number."),
  ]);

  t.truthy(response.content, "expected content in response");
  t.truthy(response.content.includes("4"), `expected '4' in: ${response.content}`);
  t.truthy(response.model, "expected model in response");
});

// ── Mistral ─────────────────────────────────────────────────────────────────

const MISTRAL_API_KEY = process.env.MISTRAL_API_KEY;
const TMistral = MISTRAL_API_KEY ? test : test.skip;

TMistral("Mistral smoke tests · completes a basic prompt", async (t) => {
  const model = Model.mistral({ apiKey: MISTRAL_API_KEY });
  const response = await model.complete([
    ChatMessage.user("What is 2+2? Reply with just the number."),
  ]);

  t.truthy(response.content, "expected content in response");
  t.truthy(response.content.includes("4"), `expected '4' in: ${response.content}`);
  t.truthy(response.model, "expected model in response");
});

// ── DeepSeek ────────────────────────────────────────────────────────────────

const DEEPSEEK_API_KEY = process.env.DEEPSEEK_API_KEY;
const TDeepSeek = DEEPSEEK_API_KEY ? test : test.skip;

TDeepSeek("DeepSeek smoke tests · completes a basic prompt", async (t) => {
  const model = Model.deepseek({ apiKey: DEEPSEEK_API_KEY });
  const response = await model.complete([
    ChatMessage.user("What is 2+2? Reply with just the number."),
  ]);

  t.truthy(response.content, "expected content in response");
  t.truthy(response.content.includes("4"), `expected '4' in: ${response.content}`);
  t.truthy(response.model, "expected model in response");
});

// ── Fireworks ───────────────────────────────────────────────────────────────

const FIREWORKS_API_KEY = process.env.FIREWORKS_API_KEY;
const TFireworks = FIREWORKS_API_KEY ? test : test.skip;

TFireworks("Fireworks smoke tests · completes a basic prompt", async (t) => {
  const model = Model.fireworks({ apiKey: FIREWORKS_API_KEY });
  const response = await model.complete([
    ChatMessage.user("What is 2+2? Reply with just the number."),
  ]);

  t.truthy(response.content, "expected content in response");
  t.truthy(response.content.includes("4"), `expected '4' in: ${response.content}`);
  t.truthy(response.model, "expected model in response");
});

// ── Perplexity ──────────────────────────────────────────────────────────────

const PERPLEXITY_API_KEY = process.env.PERPLEXITY_API_KEY;
const TPerplexity = PERPLEXITY_API_KEY ? test : test.skip;

TPerplexity("Perplexity smoke tests · completes a basic prompt", async (t) => {
  const model = Model.perplexity({ apiKey: PERPLEXITY_API_KEY });
  const response = await model.complete([
    ChatMessage.user("What is 2+2? Reply with just the number."),
  ]);

  t.truthy(response.content, "expected content in response");
  t.truthy(response.content.includes("4"), `expected '4' in: ${response.content}`);
  t.truthy(response.model, "expected model in response");
});

// ── xAI ─────────────────────────────────────────────────────────────────────

const XAI_API_KEY = process.env.XAI_API_KEY;
const TXai = XAI_API_KEY ? test : test.skip;

TXai("xAI smoke tests · completes a basic prompt", async (t) => {
  const model = Model.xai({ apiKey: XAI_API_KEY });
  const response = await model.complete([
    ChatMessage.user("What is 2+2? Reply with just the number."),
  ]);

  t.truthy(response.content, "expected content in response");
  t.truthy(response.content.includes("4"), `expected '4' in: ${response.content}`);
  t.truthy(response.model, "expected model in response");
});

// ── Cohere ──────────────────────────────────────────────────────────────────

const COHERE_API_KEY = process.env.COHERE_API_KEY;
const TCohere = COHERE_API_KEY ? test : test.skip;

TCohere("Cohere smoke tests · completes a basic prompt", async (t) => {
  const model = Model.cohere({ apiKey: COHERE_API_KEY });
  const response = await model.complete([
    ChatMessage.user("What is 2+2? Reply with just the number."),
  ]);

  t.truthy(response.content, "expected content in response");
  t.truthy(response.content.includes("4"), `expected '4' in: ${response.content}`);
  t.truthy(response.model, "expected model in response");
});
