/**
 * Basic Multi-Step Workflow
 *
 * Demonstrates a three-step Blazen workflow that processes user input
 * through a pipeline: parsing, transforming, and producing a final greeting.
 *
 * Run with: npx tsx basic_workflow.ts
 */

import { Workflow } from "blazen";
import type { Context, JsWorkflowResult } from "blazen";

const wf = new Workflow("greeting-pipeline");

wf.addStep("parse_input", ["blazen::StartEvent"], async (event: Record<string, any>, ctx: Context) => {
  const name: string = event.name || "World";
  await ctx.set("received_at", new Date().toISOString());
  return {
    type: "GreetEvent",
    name: name,
  };
});

wf.addStep("transform", ["GreetEvent"], async (event: Record<string, any>, ctx: Context) => {
  const capitalized: string = event.name.charAt(0).toUpperCase() + event.name.slice(1);
  const greeting: string = `Hello, ${capitalized}!`;
  return {
    type: "FormattedEvent",
    greeting: greeting,
    originalName: event.name,
  };
});

wf.addStep("greet", ["FormattedEvent"], async (event: Record<string, any>, ctx: Context) => {
  const receivedAt: string = await ctx.get("received_at");
  return {
    type: "blazen::StopEvent",
    result: {
      greeting: event.greeting,
      originalName: event.originalName,
      receivedAt: receivedAt,
    },
  };
});

const result: JsWorkflowResult = await wf.run({ name: "blazen" });

console.log("Workflow finished!");
console.log("  type:", result.type);
console.log("  greeting:", result.data.greeting);
console.log("  originalName:", result.data.originalName);
console.log("  receivedAt:", result.data.receivedAt);
