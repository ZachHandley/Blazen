/**
 * Branching (Fan-Out) Workflow Example
 *
 * Demonstrates conditional branching in a Blazen workflow. A single step can
 * return an ARRAY of event objects, causing the workflow to "fan out" into
 * multiple parallel branches. Each downstream step declares the event types
 * it listens for, so only the matching branch receives each event.
 *
 * Because multiple branches may race toward completion, the first
 * blazen::StopEvent to fire wins and the workflow resolves with that result.
 *
 * Scenario:
 *   1. The "analyze" step inspects a piece of text for sentiment.
 *   2. If the text is positive it emits a "Positive" event; if negative, a
 *      "Negative" event. (Only one event is emitted -- this is conditional
 *      routing, not a true parallel fan-out, but the mechanism is the same.)
 *   3. Dedicated handler steps for each sentiment produce the final result.
 *
 * Run with: node branching_workflow.mjs
 */

import { Workflow } from "blazen";

// --- Build the workflow --------------------------------------------------- //

const wf = new Workflow("sentiment-branching");

/**
 * Step 1 -- Analyze sentiment.
 *
 * Listens for the built-in StartEvent. Performs a naive keyword check and
 * returns an array of events. Returning an array is what triggers fan-out;
 * here we return a single-element array for conditional routing, but you
 * could return multiple elements to fan out into several branches at once.
 */
wf.addStep("analyze", ["blazen::StartEvent"], async (event, ctx) => {
  const text = event.data?.text ?? "";
  console.log(`[analyze] Received text: "${text}"`);

  // Naive sentiment check -- look for a handful of positive keywords.
  const positiveWords = ["great", "awesome", "love", "happy", "fantastic", "good", "wonderful"];
  const isPositive = positiveWords.some((w) => text.toLowerCase().includes(w));

  if (isPositive) {
    console.log("[analyze] Detected POSITIVE sentiment -> routing to handle_positive");
    return [{ type: "Positive", text }];
  }

  console.log("[analyze] Detected NEGATIVE sentiment -> routing to handle_negative");
  return [{ type: "Negative", text }];
});

/**
 * Step 2a -- Handle positive sentiment.
 *
 * Only fires when a "Positive" event is emitted.
 */
wf.addStep("handle_positive", ["Positive"], async (event, ctx) => {
  console.log("[handle_positive] Processing positive text");
  return {
    type: "blazen::StopEvent",
    result: { sentiment: "positive", reply: "Glad to hear it!" },
  };
});

/**
 * Step 2b -- Handle negative sentiment.
 *
 * Only fires when a "Negative" event is emitted.
 */
wf.addStep("handle_negative", ["Negative"], async (event, ctx) => {
  console.log("[handle_negative] Processing negative text");
  return {
    type: "blazen::StopEvent",
    result: { sentiment: "negative", reply: "Sorry to hear that." },
  };
});

// --- Run the workflow ----------------------------------------------------- //

const input = { text: "I had a fantastic day at the park!" };

console.log("--- Running sentiment-branching workflow ---");
console.log();

const result = await wf.run(input);

console.log();
console.log(`Branch taken : ${result.data.sentiment}`);
console.log(`Reply        : ${result.data.reply}`);
