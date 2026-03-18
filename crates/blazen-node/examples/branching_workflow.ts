/**
 * Branching (Fan-Out) Workflow Example
 *
 * Demonstrates a Blazen workflow that branches based on sentiment analysis,
 * routing positive and negative text to separate handler steps.
 *
 * Run with: npx tsx branching_workflow.ts
 */

import { Workflow } from "blazen";
import type { Context, JsWorkflowResult } from "blazen";

const wf = new Workflow("sentiment-branching");

wf.addStep("analyze", ["blazen::StartEvent"], async (event: Record<string, any>, ctx: Context) => {
  const text: string = event.data?.text ?? "";
  console.log(`[analyze] Received text: "${text}"`);
  const positiveWords: string[] = ["great", "awesome", "love", "happy", "fantastic", "good", "wonderful"];
  const isPositive: boolean = positiveWords.some((w: string) => text.toLowerCase().includes(w));
  if (isPositive) {
    console.log("[analyze] Detected POSITIVE sentiment -> routing to handle_positive");
    return [{ type: "Positive", text }];
  }
  console.log("[analyze] Detected NEGATIVE sentiment -> routing to handle_negative");
  return [{ type: "Negative", text }];
});

wf.addStep("handle_positive", ["Positive"], async (event: Record<string, any>, ctx: Context) => {
  console.log("[handle_positive] Processing positive text");
  return {
    type: "blazen::StopEvent",
    result: { sentiment: "positive", reply: "Glad to hear it!" },
  };
});

wf.addStep("handle_negative", ["Negative"], async (event: Record<string, any>, ctx: Context) => {
  console.log("[handle_negative] Processing negative text");
  return {
    type: "blazen::StopEvent",
    result: { sentiment: "negative", reply: "Sorry to hear that." },
  };
});

const input: Record<string, any> = { text: "I had a fantastic day at the park!" };
console.log("--- Running sentiment-branching workflow ---\n");
const result: JsWorkflowResult = await wf.run(input);
console.log(`\nBranch taken : ${result.data.sentiment}`);
console.log(`Reply        : ${result.data.reply}`);
