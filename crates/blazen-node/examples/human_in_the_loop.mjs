/**
 * Human-in-the-Loop / Side-Effect Pattern
 *
 * This example demonstrates two key Blazen workflow patterns:
 *
 * 1. Side-effect steps — A step that performs work (stores state, calls APIs,
 *    etc.) but does not produce an output event. It returns `null` to signal
 *    "I handled this, but I have no direct output event." Instead, it uses
 *    `ctx.sendEvent()` to explicitly route the next event.
 *
 * 2. ctx.sendEvent() — An async method that injects an event into the
 *    workflow's internal router. This lets you decouple "when to continue"
 *    from the step's return value, which is the foundation for
 *    human-in-the-loop flows where an external signal (approval, review,
 *    user input) determines what happens next.
 *
 * Workflow:
 *   StartEvent
 *     -> "process_input"   : validates & stores intermediate state in context
 *     -> "review"          : simulates human review, sends approval event via ctx.sendEvent()
 *     -> "finalize"        : reads all accumulated context and produces the StopEvent
 *
 * Run with: node human_in_the_loop.mjs
 */

import { Workflow } from "blazen";

const wf = new Workflow("human-in-the-loop");

// Step 1: Process the incoming input and store intermediate state.
// This step receives the initial StartEvent, validates the payload,
// and writes the processed data into the workflow context for later steps.
wf.addStep(
  "process_input",
  ["blazen::StartEvent"],
  async (event, ctx) => {
    const proposal = event.data?.proposal ?? "default proposal";
    console.log(`[process_input] Received proposal: "${proposal}"`);

    // Store intermediate state — ctx.set() is async
    await ctx.set("proposal", proposal);
    await ctx.set("processed_at", new Date().toISOString());

    // Emit an event for the review step to pick up
    return { type: "ReadyForReview" };
  }
);

// Step 2: Simulate a human review / approval gate.
// In a real application this is where you would pause and wait for
// external input (a webhook, a UI button click, a Slack approval, etc.).
// Here we simulate the review inline.
//
// KEY PATTERN: This step returns `null` — it is a pure side-effect step.
// Instead of returning an event, it calls `ctx.sendEvent()` to explicitly
// route the next event. This decoupling is what enables human-in-the-loop:
// the "send" could happen minutes or hours later after real human input.
wf.addStep(
  "review",
  ["ReadyForReview"],
  async (event, ctx) => {
    const proposal = await ctx.get("proposal");
    console.log(`[review] Reviewing proposal: "${proposal}"`);

    // --- Simulate human review logic ---
    const approved = true;
    const reviewNote = "Looks good, approved by reviewer.";
    // -----------------------------------

    // Store the review outcome in context
    await ctx.set("approved", approved);
    await ctx.set("review_note", reviewNote);

    // Use sendEvent to route the workflow forward.
    // This is async — the event is delivered to matching steps internally.
    await ctx.sendEvent({ type: "ReviewComplete" });

    // Return null: "I handled this but produce no output event."
    return null;
  }
);

// Step 3: Finalize — gather everything from context and produce the result.
wf.addStep(
  "finalize",
  ["ReviewComplete"],
  async (event, ctx) => {
    const proposal = await ctx.get("proposal");
    const approved = await ctx.get("approved");
    const reviewNote = await ctx.get("review_note");
    const processedAt = await ctx.get("processed_at");

    console.log(`[finalize] Approved: ${approved} — "${reviewNote}"`);

    // Return a StopEvent to end the workflow and surface the result.
    return {
      type: "blazen::StopEvent",
      result: {
        proposal,
        approved,
        reviewNote,
        processedAt,
      },
    };
  }
);

// --- Run the workflow with top-level await (ESM) ---

console.log("Starting human-in-the-loop workflow...\n");

const result = await wf.run({ proposal: "Launch new feature X" });

console.log("\nWorkflow complete. Result:");
console.log(JSON.stringify(result.data, null, 2));
