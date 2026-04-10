/**
 * Using Blazen with OpenRouter as the LLM provider.
 *
 * OpenRouter (https://openrouter.ai) is a unified API gateway that proxies
 * requests to 200+ models from providers like Anthropic, OpenAI, Meta, Google,
 * Mistral, and many more -- all through a single API key.  This makes it easy
 * to swap models without changing provider credentials or endpoint configuration.
 *
 * OpenRouter supports 200+ models.  Change DEFAULT_MODEL below to try a
 * different one, e.g. "openai/gpt-4o", "meta-llama/llama-3-70b",
 * "google/gemini-2.0-flash", etc.
 *
 * This example builds a two-step workflow:
 *
 *     StartEvent  ->  generate_poem  ->  PoemEvent
 *     PoemEvent   ->  summarize      ->  StopEvent
 *
 * Both steps call OpenRouter via `CompletionModel.openrouter()` to demonstrate
 * real LLM completions inside a Blazen pipeline.
 *
 * Run with: OPENROUTER_API_KEY=sk-or-... npx tsx llm_openrouter.ts
 */

import { Workflow, CompletionModel } from "blazen";
import type { Context, JsWorkflowResult } from "blazen";

// Default model routed through OpenRouter.  You can swap this to any model
// OpenRouter supports, e.g. "openai/gpt-4o", "meta-llama/llama-3-70b",
// "google/gemini-2.0-flash", etc.
const DEFAULT_MODEL: string = "anthropic/claude-sonnet-4-6";

// ---------------------------------------------------------------------------
// 1. Load API key from environment
// ---------------------------------------------------------------------------
const apiKey: string | undefined = process.env.OPENROUTER_API_KEY;
if (!apiKey) {
  console.error(
    "ERROR: OPENROUTER_API_KEY environment variable is not set.\n" +
      "\n" +
      "Get a free key at https://openrouter.ai/keys and run:\n" +
      "  OPENROUTER_API_KEY=sk-or-... npx tsx llm_openrouter.ts"
  );
  process.exit(1);
}

// ---------------------------------------------------------------------------
// 2. Create the CompletionModel targeting OpenRouter
// ---------------------------------------------------------------------------
const llm: CompletionModel = CompletionModel.openrouter({ apiKey });
console.log(`Using model: ${DEFAULT_MODEL}`);
console.log();

// ---------------------------------------------------------------------------
// 3. Build the workflow
// ---------------------------------------------------------------------------
const wf: Workflow = new Workflow("poem-pipeline");

// ---------------------------------------------------------------------------
// Step 1: Generate a short poem about the given topic.
// ---------------------------------------------------------------------------
wf.addStep(
  "generate_poem",
  ["blazen::StartEvent"],
  async (event: Record<string, any>, ctx: Context) => {
    const topic: string = event.topic;

    console.log(`[generate_poem] Asking for a poem about: '${topic}'`);

    const response: Record<string, any> = await llm.completeWithOptions(
      [
        {
          role: "system",
          content:
            "You are a creative poet. Write short, vivid poems (4-8 lines).",
        },
        { role: "user", content: `Write a short poem about: ${topic}` },
      ],
      { model: DEFAULT_MODEL, maxTokens: 256 }
    );

    const poem: string = response.content;
    const usage: Record<string, any> = response.usage;

    console.log(
      `[generate_poem] Received poem (${usage.total_tokens ?? usage.totalTokens} tokens used)`
    );

    // Store the first call's usage in context so we can tally it later.
    await ctx.set("poem_usage", usage);

    return {
      type: "PoemEvent",
      poem: poem,
      model_used: response.model,
    };
  }
);

// ---------------------------------------------------------------------------
// Step 2: Summarize the poem in a single sentence.
// ---------------------------------------------------------------------------
wf.addStep(
  "summarize",
  ["PoemEvent"],
  async (event: Record<string, any>, ctx: Context) => {
    const poem: string = event.poem;

    console.log("[summarize] Summarizing the poem...");

    const response: Record<string, any> = await llm.completeWithOptions(
      [
        {
          role: "system",
          content:
            "You are a concise literary critic. Summarize poems in exactly " +
            "one sentence.",
        },
        {
          role: "user",
          content: `Summarize this poem in one sentence:\n\n${poem}`,
        },
      ],
      { model: DEFAULT_MODEL, maxTokens: 128 }
    );

    const summary: string = response.content;
    const summaryUsage: Record<string, any> = response.usage;

    console.log(
      `[summarize] Done (${summaryUsage.total_tokens ?? summaryUsage.totalTokens} tokens used)`
    );

    // Retrieve the first step's usage for the final tally.
    const poemUsage: Record<string, any> = await ctx.get("poem_usage");

    return {
      type: "blazen::StopEvent",
      result: {
        poem: poem,
        summary: summary,
        model: event.model_used,
        poem_usage: poemUsage,
        summary_usage: summaryUsage,
      },
    };
  }
);

// ---------------------------------------------------------------------------
// 4. Run the workflow
// ---------------------------------------------------------------------------
const result: JsWorkflowResult = await wf.run({
  topic: "the beauty of open-source software",
});
const output: Record<string, any> = result.data;

// ---------------------------------------------------------------------------
// 5. Print results and token usage
// ---------------------------------------------------------------------------
console.log();
console.log("=".repeat(60));
console.log("POEM");
console.log("=".repeat(60));
console.log(output.poem);

console.log();
console.log("=".repeat(60));
console.log("SUMMARY");
console.log("=".repeat(60));
console.log(output.summary);

console.log();
console.log("=".repeat(60));
console.log("TOKEN USAGE");
console.log("=".repeat(60));

const poemUsage: Record<string, any> = output.poem_usage;
const summaryUsage: Record<string, any> = output.summary_usage;
const promptTokensKey: string =
  "prompt_tokens" in poemUsage ? "prompt_tokens" : "promptTokens";
const completionTokensKey: string =
  "completion_tokens" in poemUsage ? "completion_tokens" : "completionTokens";
const totalTokensKey: string =
  "total_tokens" in poemUsage ? "total_tokens" : "totalTokens";

console.log(`  Model:              ${output.model}`);
console.log(
  `  Poem generation:    ${poemUsage[totalTokensKey]} tokens ` +
    `(prompt: ${poemUsage[promptTokensKey]}, ` +
    `completion: ${poemUsage[completionTokensKey]})`
);
console.log(
  `  Summarization:      ${summaryUsage[totalTokensKey]} tokens ` +
    `(prompt: ${summaryUsage[promptTokensKey]}, ` +
    `completion: ${summaryUsage[completionTokensKey]})`
);
const total: number =
  poemUsage[totalTokensKey] + summaryUsage[totalTokensKey];
console.log(`  Total:              ${total} tokens`);
