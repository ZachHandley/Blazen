/**
 * Using Blazen with fal.ai as the LLM provider.
 *
 * fal.ai is FUNDAMENTALLY DIFFERENT from other LLM providers. It is a compute
 * platform that operates via an async job queue with polling, not direct HTTP
 * request/response like OpenAI or Anthropic.
 *
 * How fal.ai works under the hood:
 *
 *     1. SUBMIT  -- Your request is POST'd to https://queue.fal.run/<model>
 *                   and you receive a request_id back immediately.
 *     2. POLL    -- Blazen polls https://queue.fal.run/<model>/requests/<id>/status
 *                   every ~1 second, waiting for status "COMPLETED" or "FAILED".
 *     3. FETCH   -- Once complete, the result is GET'd from the request endpoint.
 *
 * This queue-based architecture means:
 *     - Latency is inherently higher than direct HTTP providers (extra round-trips).
 *     - There is no native SSE streaming -- Blazen simulates it by returning the
 *       complete response as a single chunk.
 *     - Tool calling is NOT supported (fal-ai/any-llm is text-in, text-out).
 *     - Image inputs are NOT supported (text-only; non-text content is dropped).
 *     - The LLM endpoint (fal-ai/any-llm) proxies through OpenRouter, so you
 *       can access many models (OpenAI, Anthropic, Meta, etc.) via a single key.
 *     - Authentication uses "Key <token>" format, not "Bearer <token>".
 *
 * Run with: FAL_KEY=... npx tsx llm_fal.ts
 */

import { Workflow, CompletionModel } from "blazen";
import type { Context, JsWorkflowResult } from "blazen";

// Module-level model instance. CompletionModel is a native Rust object and
// is NOT JSON-serializable, so it cannot be stored via ctx.set(). We create
// it once in main() and reference it from steps via this module variable.
let MODEL: CompletionModel | null = null;

// ---------------------------------------------------------------------------
// Step 1: Generate a response using fal.ai
//
// fal.ai uses the fal-ai/any-llm endpoint, which is powered by OpenRouter.
// Under the hood, Blazen submits to the fal.ai job queue, then polls until
// the result is ready. This means:
//   - Expect higher latency than direct providers (queue + poll overhead)
//   - No streaming support (Blazen simulates it with a single chunk)
//   - No tool calling support (tool_calls will always be an empty list)
//   - Text-only input (no image URLs or base64 images)
// ---------------------------------------------------------------------------

const wf = new Workflow("fal-demo");

wf.addStep("generate", ["blazen::StartEvent"], async (event: Record<string, any>, ctx: Context) => {
  const prompt: string = event.prompt;
  const model: CompletionModel = MODEL!;
  if (!model) throw new Error("MODEL must be set before running the workflow");

  console.log("[generate] Submitting to fal.ai queue...");
  console.log(`[generate] Prompt: "${prompt}"`);

  const start: number = performance.now();
  const response: Record<string, any> = await model.complete([
    { role: "system", content: "You are a concise, thoughtful assistant. Keep responses to 2-3 sentences." },
    { role: "user", content: prompt },
  ]);
  const elapsed: number = (performance.now() - start) / 1000;

  const content: string = response.content;
  const modelName: string = response.model;
  const usage: Record<string, any> | null = response.usage;
  const finishReason: string = response.finishReason;

  // toolCalls is always empty for fal.ai -- no tool calling support
  const toolCalls: any[] = response.toolCalls;
  if (toolCalls.length !== 0) throw new Error("fal.ai does not support tool calling");

  console.log(`[generate] Response received in ${elapsed.toFixed(2)}s (includes queue + poll time)`);
  console.log(`[generate] Model: ${modelName}`);
  console.log(`[generate] Finish reason: ${finishReason}`);
  console.log(`[generate] Content: ${content}`);

  // Store timing and response data in context for the next step.
  // ctx.set/ctx.get are async in the Node binding.
  await ctx.set("generate_content", content);
  await ctx.set("generate_elapsed", elapsed);
  await ctx.set("generate_usage", usage);

  return {
    type: "GenerateComplete",
    content: content,
  };
});

// ---------------------------------------------------------------------------
// Step 2: Analyze the response quality using fal.ai again
//
// This second call demonstrates that each fal.ai request goes through the
// same queue/poll cycle independently. The total workflow time is the sum
// of both queue round-trips.
// ---------------------------------------------------------------------------

wf.addStep("analyze", ["GenerateComplete"], async (event: Record<string, any>, ctx: Context) => {
  const content: string = event.content;
  const model: CompletionModel = MODEL!;
  if (!model) throw new Error("MODEL must be set before running the workflow");

  console.log("\n[analyze] Submitting analysis to fal.ai queue...");

  const start: number = performance.now();
  const response: Record<string, any> = await model.complete([
    {
      role: "system",
      content:
        "You are a writing quality analyst. " +
        "Rate the following text on clarity, conciseness, and helpfulness. " +
        "Give a score from 1-10 and a one-sentence explanation.",
    },
    { role: "user", content: `Analyze this response:\n\n${content}` },
  ]);
  const elapsed: number = (performance.now() - start) / 1000;

  const analysis: string = response.content;
  const analysisUsage: Record<string, any> | null = response.usage;

  console.log(`[analyze] Analysis received in ${elapsed.toFixed(2)}s`);
  console.log(`[analyze] Result: ${analysis}`);

  // Gather all timing and usage info.
  const generateElapsed: number = await ctx.get("generate_elapsed");
  const generateUsage: Record<string, any> | null = await ctx.get("generate_usage");

  return {
    type: "blazen::StopEvent",
    result: {
      original_prompt: await ctx.get("generate_content"),
      generated_response: content,
      analysis: analysis,
      timing: {
        generate_seconds: Math.round(generateElapsed * 100) / 100,
        analyze_seconds: Math.round(elapsed * 100) / 100,
        total_seconds: Math.round((generateElapsed + elapsed) * 100) / 100,
      },
      usage: {
        generate: generateUsage,
        analyze: analysisUsage,
      },
    },
  };
});

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

const falKey: string | undefined = process.env.FAL_KEY;
if (!falKey) {
  console.log("ERROR: Set the FAL_KEY environment variable.");
  console.log("  FAL_KEY=fal-... npx tsx llm_fal.ts");
  process.exit(1);
}

// Create the fal.ai model. By default this uses the fal-ai/any-llm
// endpoint, which proxies through OpenRouter to access many LLM providers.
// CompletionModel is a native Rust object (not JSON-serializable), so we
// store it as a module-level variable rather than in ctx.set().
MODEL = CompletionModel.fal(falKey);

// You can also specify a different fal.ai model endpoint:
//   MODEL = CompletionModel.fal(falKey, "fal-ai/any-llm");

console.log(`Using model: ${MODEL.modelId}`);
console.log("NOTE: fal.ai uses a queue-based architecture. Each call involves");
console.log("      submit -> poll -> fetch, so expect higher latency than");
console.log("      direct HTTP providers like OpenAI or Anthropic.\n");

// Run the workflow.
const result: JsWorkflowResult = await wf.run({ prompt: "What makes Rust's ownership system unique?" });
const output: Record<string, any> = result.data;

console.log("\n" + "=".repeat(60));
console.log("WORKFLOW COMPLETE");
console.log("=".repeat(60));
console.log(`\nGenerated response:\n  ${output.generated_response}`);
console.log(`\nAnalysis:\n  ${output.analysis}`);
console.log("\nTiming (includes fal.ai queue overhead):");
const timing: Record<string, number> = output.timing;
console.log(`  Generate step: ${timing.generate_seconds}s`);
console.log(`  Analyze step:  ${timing.analyze_seconds}s`);
console.log(`  Total:         ${timing.total_seconds}s`);

// Usage info may be null for fal.ai -- it depends on the underlying
// model and whether OpenRouter returns token counts.
const usage: Record<string, any> = output.usage;
if (usage.generate) {
  const u: Record<string, any> = usage.generate;
  console.log(
    `\nUsage (generate): ${u.prompt_tokens ?? "?"} prompt + ` +
      `${u.completion_tokens ?? "?"} completion = ` +
      `${u.total_tokens ?? "?"} total tokens`
  );
} else {
  console.log("\nUsage (generate): not available (fal.ai may not report token usage)");
}

if (usage.analyze) {
  const u: Record<string, any> = usage.analyze;
  console.log(
    `Usage (analyze):  ${u.prompt_tokens ?? "?"} prompt + ` +
      `${u.completion_tokens ?? "?"} completion = ` +
      `${u.total_tokens ?? "?"} total tokens`
  );
} else {
  console.log("Usage (analyze):  not available");
}
