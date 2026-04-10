/**
 * Anthropic (Claude) LLM integration with a Blazen Q&A + fact-check workflow.
 *
 * Demonstrates real Anthropic API calls through Blazen's CompletionModel
 * abstraction.  Two workflow steps form a question-answering pipeline:
 *
 *   1. ask_question  -- sends the user's question to Claude and gets an answer
 *   2. fact_check    -- sends the answer back to Claude for verification
 *
 * Key points:
 *   - CompletionModel.anthropic() creates an Anthropic-backed model.
 *   - model.completeWithOptions() is async and returns an object with content,
 *     model, usage, finishReason, and toolCalls.
 *   - Anthropic requires maxTokens (Blazen defaults to 4096, but you can set
 *     it explicitly).
 *   - ctx.set() / ctx.get() are async -- await is needed.
 *
 * Run with: ANTHROPIC_API_KEY=sk-ant-... npx tsx llm_anthropic.ts
 */

import { Workflow, CompletionModel } from "blazen";
import type { Context, JsWorkflowResult } from "blazen";

// ---------------------------------------------------------------------------
// Helper types for the LLM response shape.
// ---------------------------------------------------------------------------

interface Usage {
  promptTokens: number;
  completionTokens: number;
  totalTokens: number;
}

interface CompletionResponse {
  content: string;
  toolCalls: Array<{ id: string; name: string; arguments: string }>;
  usage: Usage | null;
  model: string;
  finishReason: string;
}

// ---------------------------------------------------------------------------
// Create the Anthropic-backed model (module-level so both steps share it).
// Using claude-haiku-3-5 -- fast and cheap, ideal for examples.
// ---------------------------------------------------------------------------
function getModel(): CompletionModel {
  const apiKey: string | undefined = process.env.ANTHROPIC_API_KEY;
  if (!apiKey) {
    console.error("ERROR: Set the ANTHROPIC_API_KEY environment variable.");
    console.error("  ANTHROPIC_API_KEY=sk-ant-... npx tsx llm_anthropic.ts");
    process.exit(1);
  }
  return CompletionModel.anthropic({ apiKey });
}

const MODEL: CompletionModel = getModel();

// ---------------------------------------------------------------------------
// Build the workflow.
// ---------------------------------------------------------------------------

const wf = new Workflow("qa-fact-check");

// ---------------------------------------------------------------------------
// Step 1: Ask Claude a question and get an answer.
// ---------------------------------------------------------------------------
wf.addStep(
  "ask_question",
  ["blazen::StartEvent"],
  async (event: Record<string, any>, ctx: Context): Promise<Record<string, any>> => {
    const question: string = event.question;
    console.log(`[ask_question] Question: ${question}`);

    const response: CompletionResponse = await MODEL.completeWithOptions(
      [
        { role: "system", content: "You are a helpful assistant. Give concise, factual answers in 2-3 sentences maximum." },
        { role: "user", content: question },
      ],
      { maxTokens: 256, model: "claude-haiku-4-5-20251001" },
    );

    const answer: string = response.content;
    const usage: Usage | null = response.usage;

    // Store in context for the final summary.
    await ctx.set("question", question);
    await ctx.set("answer", answer);
    await ctx.set("answer_model", response.model);
    await ctx.set("answer_usage", usage);

    console.log(`[ask_question] Answer received (${usage?.totalTokens ?? "?"} tokens used)`);

    return { type: "AnswerEvent", answer };
  },
);

// ---------------------------------------------------------------------------
// Step 2: Fact-check the answer with a second Claude call.
// ---------------------------------------------------------------------------
wf.addStep(
  "fact_check",
  ["AnswerEvent"],
  async (event: Record<string, any>, ctx: Context): Promise<Record<string, any>> => {
    const answer: string = event.answer;
    const question: string = await ctx.get("question");

    console.log("[fact_check] Verifying answer...");

    const response: CompletionResponse = await MODEL.completeWithOptions(
      [
        {
          role: "system",
          content:
            "You are a rigorous fact-checker. Given a question and a proposed answer, assess whether the answer is accurate. Reply with VERIFIED or DISPUTED followed by a brief explanation (1-2 sentences).",
        },
        {
          role: "user",
          content: `Question: ${question}\n\nProposed answer: ${answer}`,
        },
      ],
      { maxTokens: 256, model: "claude-haiku-4-5-20251001" },
    );

    const verdict: string = response.content;
    const verdictUsage: Usage | null = response.usage;

    console.log(`[fact_check] Verdict received (${verdictUsage?.totalTokens ?? "?"} tokens used)`);

    // Combine usage from both calls.
    const answerUsage: Usage | null = await ctx.get("answer_usage");
    const totalUsage: Record<string, number> = {
      promptTokens: (answerUsage?.promptTokens ?? 0) + (verdictUsage?.promptTokens ?? 0),
      completionTokens: (answerUsage?.completionTokens ?? 0) + (verdictUsage?.completionTokens ?? 0),
      totalTokens: (answerUsage?.totalTokens ?? 0) + (verdictUsage?.totalTokens ?? 0),
    };

    return {
      type: "blazen::StopEvent",
      result: {
        question,
        answer,
        factCheck: verdict,
        model: await ctx.get("answer_model"),
        totalUsage,
      },
    };
  },
);

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------
const result: JsWorkflowResult = await wf.run({
  question: "What is the speed of light in a vacuum, in meters per second?",
});

const data: Record<string, any> = result.data;

console.log();
console.log("=".repeat(60));
console.log("RESULTS");
console.log("=".repeat(60));
console.log(`Model:      ${data.model}`);
console.log(`Question:   ${data.question}`);
console.log(`Answer:     ${data.answer}`);
console.log(`Fact-check: ${data.factCheck}`);
console.log(`Tokens:     ${JSON.stringify(data.totalUsage)}`);
