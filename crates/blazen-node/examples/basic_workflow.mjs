/**
 * Basic Multi-Step Workflow
 *
 * Demonstrates a three-step Blazen workflow that processes user input
 * through a pipeline: parsing, transforming, and producing a final greeting.
 *
 * Concepts covered:
 *   - Creating a workflow with `new Workflow(name)`
 *   - Adding steps with `wf.addStep(name, eventTypes, handler)`
 *   - Using built-in events: `blazen::StartEvent` and `blazen::StopEvent`
 *   - Defining custom intermediate events (`GreetEvent`, `FormattedEvent`)
 *   - Sharing state between steps via `ctx.set()` and `ctx.get()` (async)
 *   - Running the workflow with `wf.run(input)` (returns a promise)
 *
 * Run with: node basic_workflow.mjs
 */

import { Workflow } from "blazen";

// Create a new workflow named "greeting-pipeline".
const wf = new Workflow("greeting-pipeline");

// ---------------------------------------------------------------------------
// Step 1: parse_input
//
// Listens for the built-in StartEvent, which wraps whatever object you pass
// to `wf.run()`. Extracts the user's name and stores a timestamp in the
// shared context so later steps can access it.
// ---------------------------------------------------------------------------
wf.addStep("parse_input", ["blazen::StartEvent"], async (event, ctx) => {
  const name = event.name || "World";

  // Store metadata in the shared context for downstream steps.
  await ctx.set("received_at", new Date().toISOString());

  // Return a custom event to pass data to the next step.
  return {
    type: "GreetEvent",
    name: name,
  };
});

// ---------------------------------------------------------------------------
// Step 2: transform
//
// Listens for our custom GreetEvent. Applies a simple transformation --
// capitalizing the name and building a greeting string.
// ---------------------------------------------------------------------------
wf.addStep("transform", ["GreetEvent"], async (event, ctx) => {
  const capitalized = event.name.charAt(0).toUpperCase() + event.name.slice(1);
  const greeting = `Hello, ${capitalized}!`;

  return {
    type: "FormattedEvent",
    greeting: greeting,
    originalName: event.name,
  };
});

// ---------------------------------------------------------------------------
// Step 3: greet
//
// Listens for FormattedEvent. Reads the timestamp from context and produces
// the final StopEvent, which ends the workflow and returns the result.
// ---------------------------------------------------------------------------
wf.addStep("greet", ["FormattedEvent"], async (event, ctx) => {
  const receivedAt = await ctx.get("received_at");

  // Returning a blazen::StopEvent completes the workflow.
  // The `result` field becomes `data` on the returned object.
  return {
    type: "blazen::StopEvent",
    result: {
      greeting: event.greeting,
      originalName: event.originalName,
      receivedAt: receivedAt,
    },
  };
});

// ---------------------------------------------------------------------------
// Run the workflow
// ---------------------------------------------------------------------------
const result = await wf.run({ name: "blazen" });

console.log("Workflow finished!");
console.log("  type:", result.type);
console.log("  greeting:", result.data.greeting);
console.log("  originalName:", result.data.originalName);
console.log("  receivedAt:", result.data.receivedAt);
