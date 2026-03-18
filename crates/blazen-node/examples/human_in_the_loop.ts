/**
 * Human-in-the-Loop Workflow with Side-Effect Steps
 *
 * Demonstrates two key Blazen patterns:
 *
 * 1. **Side-effect steps** -- A step that performs work (stores state in
 *    context, logs, calls an external API, etc.) but does not produce an
 *    output event by returning `null`. Instead it uses `ctx.sendEvent()`
 *    to manually route the next event when ready.
 *
 * 2. **Human-in-the-loop simulation** -- A review step that pauses to
 *    inspect intermediate results before deciding to approve or reject.
 *    In a real application you would replace the simulated review with
 *    an actual human interaction (webhook, UI callback, message queue).
 *
 * Flow:
 *
 *    StartEvent  ->  process_submission  ->  (side-effect, no return)
 *                        |
 *                        +-- ctx.sendEvent(ReviewEvent) -->  review_submission
 *                                                                |
 *                                                                +-- ctx.sendEvent(ReviewedEvent)
 *                                                                        |
 *                                                                        v
 *                                                                    finalize  ->  StopEvent
 *
 * Key API notes:
 *    - `ctx.set()` / `ctx.get()` are async (return Promises).
 *    - `ctx.sendEvent()` is async -- it routes an event to the step
 *      whose `eventTypes` list matches.
 *    - Returning `null` from a step means "I handled this event but
 *      do not produce an output event through the normal return path."
 *
 * Run with: npx tsx human_in_the_loop.ts
 */

import { Workflow } from "blazen";
import type { Context, JsWorkflowResult } from "blazen";

const wf = new Workflow("human-review");

// ---------------------------------------------------------------------------
// Step 1: Process the incoming submission.
//
// Accepts the default StartEvent. Validates the data, stores
// intermediate state in the shared context, then uses ctx.sendEvent()
// to forward a ReviewEvent. Returns null (side-effect only).
// ---------------------------------------------------------------------------
wf.addStep(
  "process_submission",
  ["blazen::StartEvent"],
  async (event: Record<string, any>, ctx: Context): Promise<null> => {
    const title: string = event.title;
    const body: string = event.body;

    // Simulate processing -- normalise and store intermediate results.
    const processedTitle: string = title.trim().replace(/\w\S*/g, (txt: string) =>
      txt.charAt(0).toUpperCase() + txt.slice(1).toLowerCase()
    );
    const wordCount: number = body.split(/\s+/).filter(Boolean).length;

    // Persist intermediate state so downstream steps can read it.
    await ctx.set("processed_title", processedTitle);
    await ctx.set("word_count", wordCount);
    await ctx.set("original_body", body);

    console.log(`  [process] Title normalised to: '${processedTitle}'`);
    console.log(`  [process] Word count: ${wordCount}`);

    // Instead of returning an Event, we manually route a ReviewEvent.
    // This is the side-effect pattern: do work, call sendEvent, return null.
    await ctx.sendEvent({
      type: "ReviewEvent",
      title: processedTitle,
      word_count: wordCount,
    });
    return null;
  }
);

// ---------------------------------------------------------------------------
// Step 2: Simulate a human review.
//
// In production this could pause and wait for a webhook, poll a queue,
// or call out to a UI. Here we apply a simple automatic rule to
// demonstrate the pattern.
// ---------------------------------------------------------------------------
wf.addStep(
  "review_submission",
  ["ReviewEvent"],
  async (event: Record<string, any>, ctx: Context): Promise<null> => {
    const title: string = event.title;
    const wordCount: number = event.word_count;

    console.log(`  [review]  Reviewing: '${title}' (${wordCount} words)`);

    // Simulated human decision: approve if body has at least 3 words.
    const approved: boolean = wordCount >= 3;
    const reason: string = approved ? "Meets minimum length" : "Too short";

    // Store the review decision in context for the finalizer.
    await ctx.set("approved", approved);
    await ctx.set("review_reason", reason);

    console.log(`  [review]  Decision: ${approved ? "APPROVED" : "REJECTED"} -- ${reason}`);

    // Again, side-effect pattern: route manually and return null.
    await ctx.sendEvent({ type: "ReviewedEvent", approved });
    return null;
  }
);

// ---------------------------------------------------------------------------
// Step 3: Finalize the workflow.
//
// Gathers everything from context and returns a StopEvent with the
// full result.
// ---------------------------------------------------------------------------
wf.addStep(
  "finalize",
  ["ReviewedEvent"],
  async (event: Record<string, any>, ctx: Context): Promise<Record<string, any>> => {
    return {
      type: "blazen::StopEvent",
      result: {
        title: await ctx.get("processed_title"),
        word_count: await ctx.get("word_count"),
        approved: await ctx.get("approved"),
        review_reason: await ctx.get("review_reason"),
        original_body: await ctx.get("original_body"),
      },
    };
  }
);

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------
console.log("Running workflow (submission that passes review)...");
const result1: JsWorkflowResult = await wf.run({
  title: "  my first blazen workflow  ",
  body: "This is a perfectly reasonable submission with enough words.",
});
console.log("  Result:", result1.data, "\n");

console.log("Running workflow (submission that fails review)...");
const result2: JsWorkflowResult = await wf.run({
  title: "short",
  body: "Too few",
});
console.log("  Result:", result2.data);
