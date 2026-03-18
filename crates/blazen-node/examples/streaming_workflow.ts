/**
 * Blazen Streaming Workflow Example
 * Run with: npx tsx streaming_workflow.ts
 */
import { Workflow } from "blazen";
import type { Context, JsWorkflowResult } from "blazen";

const wf = new Workflow("streaming-example");

wf.addStep("fetch-data", ["blazen::StartEvent"], async (event: Record<string, any>, ctx: Context) => {
  const items: string[] = ["alpha", "bravo", "charlie", "delta"];
  for (let i = 0; i < items.length; i++) {
    await new Promise<void>((resolve) => setTimeout(resolve, 50));
    await ctx.writeEventToStream({
      type: "Progress",
      step: "fetch-data",
      message: `Fetched item ${i + 1}/${items.length}: ${items[i]}`,
    });
  }
  return { type: "DataReady", items };
});

wf.addStep("process-data", ["DataReady"], async (event: Record<string, any>, ctx: Context) => {
  const results: string[] = [];
  for (let i = 0; i < event.items.length; i++) {
    const item: string = event.items[i];
    await new Promise<void>((resolve) => setTimeout(resolve, 30));
    const processed: string = item.toUpperCase();
    results.push(processed);
    await ctx.writeEventToStream({
      type: "Progress",
      step: "process-data",
      message: `Processed item ${i + 1}/${event.items.length}: ${item} -> ${processed}`,
    });
  }
  return {
    type: "blazen::StopEvent",
    result: { processed: results, count: results.length },
  };
});

console.log("Starting streaming workflow...\n");
const streamedEvents: Record<string, any>[] = [];
const result: JsWorkflowResult = await wf.runStreaming({}, (event: Record<string, any>) => {
  streamedEvents.push(event);
  console.log(`  [stream] (${event.step}) ${event.message}`);
});
console.log("\n--- Workflow complete ---");
console.log(`Final result: ${JSON.stringify(result.data)}`);
console.log(`Total streamed events received: ${streamedEvents.length}`);
