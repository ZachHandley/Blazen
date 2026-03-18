/**
 * OpenAI-powered content pipeline with Blazen.
 *
 * Demonstrates using Blazen's CompletionModel with OpenAI to build a real
 * 3-step content generation workflow:
 *
 *   1. generate_outline  -- GPT creates a blog post outline from a topic
 *   2. write_draft       -- GPT writes a short draft based on the outline
 *   3. review            -- GPT provides editorial feedback on the draft
 *
 * The example also tracks cumulative token usage across all LLM calls using
 * Blazen's context API (ctx.set / ctx.get).
 *
 * Run with: OPENAI_API_KEY=sk-... npx tsx llm_openai.ts
 */

import { Workflow, CompletionModel } from "blazen";
import type { Context, JsWorkflowResult } from "blazen";

// ---------------------------------------------------------------------------
// Usage tracking interface
// ---------------------------------------------------------------------------
interface UsageInfo {
  promptTokens: number;
  completionTokens: number;
  totalTokens: number;
}

// ---------------------------------------------------------------------------
// Helper: accumulate token usage in context
// ---------------------------------------------------------------------------
async function trackUsage(ctx: Context, usage: UsageInfo): Promise<void> {
  const prev: UsageInfo = (await ctx.get("total_usage")) || {
    promptTokens: 0,
    completionTokens: 0,
    totalTokens: 0,
  };
  await ctx.set("total_usage", {
    promptTokens: prev.promptTokens + (usage.promptTokens || 0),
    completionTokens: prev.completionTokens + (usage.completionTokens || 0),
    totalTokens: prev.totalTokens + (usage.totalTokens || 0),
  });
}

// ---------------------------------------------------------------------------
// Validate that the API key is available.
// ---------------------------------------------------------------------------
const apiKey: string | undefined = process.env.OPENAI_API_KEY;
if (!apiKey) {
  console.error(
    "ERROR: OPENAI_API_KEY environment variable is not set.\n" +
      "Run with: OPENAI_API_KEY=sk-... npx tsx llm_openai.ts"
  );
  process.exit(1);
}

// Create the OpenAI completion model (gpt-5.3-chat-latest is cheap for examples).
const model: CompletionModel = CompletionModel.openai(apiKey);

// Build the 3-step content pipeline.
const wf: Workflow = new Workflow("content-pipeline");

// ---------------------------------------------------------------------------
// Step 1: Generate an outline
// ---------------------------------------------------------------------------
wf.addStep(
  "generate_outline",
  ["blazen::StartEvent"],
  async (event: Record<string, any>, ctx: Context) => {
    const topic: string = event.topic;

    // Store the model reference key in shared context so downstream steps
    // can retrieve it. We store the topic too for convenience.
    await ctx.set("topic", topic);

    console.log(`[generate_outline] Requesting outline for: "${topic}"`);

    const response: Record<string, any> = await model.completeWithOptions(
      [
        {
          role: "system",
          content:
            "You are a professional content strategist. " +
            "When given a topic, produce a concise blog post outline " +
            "with 3-5 sections. Keep it brief -- just section titles " +
            "and one-line descriptions.",
        },
        {
          role: "user",
          content: `Create a blog post outline about: ${topic}`,
        },
      ],
      { temperature: 0.7, maxTokens: 512, model: "gpt-5.3-chat-latest" }
    );

    const outline: string = response.content;
    const usage: UsageInfo = response.usage;
    await trackUsage(ctx, usage);

    console.log(
      `[generate_outline] Outline received (${usage.totalTokens} tokens)`
    );
    console.log(`\n--- Outline ---\n${outline}\n`);

    return {
      type: "OutlineEvent",
      outline: outline,
      topic: topic,
    };
  }
);

// ---------------------------------------------------------------------------
// Step 2: Write a short draft
// ---------------------------------------------------------------------------
wf.addStep(
  "write_draft",
  ["OutlineEvent"],
  async (event: Record<string, any>, ctx: Context) => {
    console.log("[write_draft] Writing draft from outline...");

    const response: Record<string, any> = await model.completeWithOptions(
      [
        {
          role: "system",
          content:
            "You are a skilled blog writer. Given an outline, write a " +
            "short but engaging first draft (2-3 paragraphs). " +
            "Keep the tone approachable and informative.",
        },
        {
          role: "user",
          content:
            `Topic: ${event.topic}\n\nOutline:\n${event.outline}\n\n` +
            "Write a short draft based on this outline.",
        },
      ],
      { temperature: 0.8, maxTokens: 768, model: "gpt-5.3-chat-latest" }
    );

    const draft: string = response.content;
    const usage: UsageInfo = response.usage;
    await trackUsage(ctx, usage);

    console.log(
      `[write_draft] Draft received (${usage.totalTokens} tokens)`
    );
    console.log(`\n--- Draft ---\n${draft}\n`);

    return {
      type: "DraftEvent",
      draft: draft,
      topic: event.topic,
    };
  }
);

// ---------------------------------------------------------------------------
// Step 3: Editorial review
// ---------------------------------------------------------------------------
wf.addStep(
  "review",
  ["DraftEvent"],
  async (event: Record<string, any>, ctx: Context) => {
    console.log("[review] Reviewing draft...");

    const response: Record<string, any> = await model.completeWithOptions(
      [
        {
          role: "system",
          content:
            "You are a senior editor. Review the following blog draft " +
            "and provide 3-5 bullet points of constructive feedback. " +
            "Be specific and actionable.",
        },
        {
          role: "user",
          content:
            `Topic: ${event.topic}\n\nDraft:\n${event.draft}\n\n` +
            "Provide editorial feedback.",
        },
      ],
      { temperature: 0.5, maxTokens: 512, model: "gpt-5.3-chat-latest" }
    );

    const feedback: string = response.content;
    const usage: UsageInfo = response.usage;
    await trackUsage(ctx, usage);

    console.log(
      `[review] Feedback received (${usage.totalTokens} tokens)`
    );
    console.log(`\n--- Editorial Feedback ---\n${feedback}\n`);

    const totalUsage: UsageInfo = await ctx.get("total_usage");

    return {
      type: "blazen::StopEvent",
      result: {
        topic: event.topic,
        draft: event.draft,
        feedback: feedback,
        totalUsage: totalUsage,
      },
    };
  }
);

// ---------------------------------------------------------------------------
// Run the workflow
// ---------------------------------------------------------------------------
const topic: string =
  "Why Rust-powered Python libraries are the future of performance";

const result: JsWorkflowResult = await wf.run({ topic: topic });

// Print summary.
console.log("=".repeat(60));
console.log("CONTENT PIPELINE COMPLETE");
console.log("=".repeat(60));

const outputUsage: UsageInfo = result.data.totalUsage;
console.log(`Topic:              ${result.data.topic}`);
console.log(`Prompt tokens:      ${outputUsage.promptTokens}`);
console.log(`Completion tokens:  ${outputUsage.completionTokens}`);
console.log(`Total tokens:       ${outputUsage.totalTokens}`);
