/**
 * Blazen Streaming Workflow Example
 *
 * This example demonstrates the difference between routing events and stream
 * events in Blazen:
 *
 *   Routing events  -- Returned from step handlers (or sent via ctx.sendEvent).
 *                      These are dispatched through the internal event router and
 *                      trigger other steps whose eventTypes list matches.
 *
 *   Stream events   -- Published via ctx.writeEventToStream(). These are
 *                      broadcast to external observers (the streaming callback)
 *                      but do NOT trigger any steps. They are purely for
 *                      real-time observation of workflow progress.
 *
 * The workflow below has two steps chained together:
 *
 *   StartEvent -> "fetch-data" -> DataReady -> "process-data" -> StopEvent
 *
 * Both steps publish Progress stream events while they work. The streaming
 * callback prints each one as it arrives.
 *
 * Run with: node streaming_workflow.mjs
 */

import { Workflow } from "blazen";

const wf = new Workflow("streaming-example");

// ---------------------------------------------------------------------------
// Step 1: Fetch data
//
// Listens for the implicit StartEvent. Simulates fetching data in chunks,
// publishing a Progress stream event after each chunk. When done, it emits
// a "DataReady" routing event that triggers the next step.
// ---------------------------------------------------------------------------
wf.addStep("fetch-data", ["blazen::StartEvent"], async (event, ctx) => {
  const items = ["alpha", "bravo", "charlie", "delta"];

  for (let i = 0; i < items.length; i++) {
    // Simulate async work (e.g. an HTTP request per item)
    await new Promise((resolve) => setTimeout(resolve, 50));

    // Publish a stream event -- observers see this in real-time,
    // but it does NOT trigger any other step.
    await ctx.writeEventToStream({
      type: "Progress",
      step: "fetch-data",
      message: `Fetched item ${i + 1}/${items.length}: ${items[i]}`,
    });
  }

  // Return a routing event to hand off to the next step.
  return { type: "DataReady", items };
});

// ---------------------------------------------------------------------------
// Step 2: Process data
//
// Listens for "DataReady". Processes each item and streams progress, then
// returns a StopEvent to finish the workflow.
// ---------------------------------------------------------------------------
wf.addStep("process-data", ["DataReady"], async (event, ctx) => {
  const results = [];

  for (let i = 0; i < event.items.length; i++) {
    const item = event.items[i];
    await new Promise((resolve) => setTimeout(resolve, 30));

    const processed = item.toUpperCase();
    results.push(processed);

    await ctx.writeEventToStream({
      type: "Progress",
      step: "process-data",
      message: `Processed item ${i + 1}/${event.items.length}: ${item} -> ${processed}`,
    });
  }

  // Return the final StopEvent with the collected results.
  return {
    type: "blazen::StopEvent",
    result: { processed: results, count: results.length },
  };
});

// ---------------------------------------------------------------------------
// Run the workflow with streaming
// ---------------------------------------------------------------------------
console.log("Starting streaming workflow...\n");

const streamedEvents = [];

const result = await wf.runStreaming({}, (event) => {
  // This callback fires for every ctx.writeEventToStream() call.
  streamedEvents.push(event);
  console.log(`  [stream] (${event.step}) ${event.message}`);
});

// Print summary
console.log("\n--- Workflow complete ---");
console.log(`Final result: ${JSON.stringify(result.data)}`);
console.log(`Total streamed events received: ${streamedEvents.length}`);
