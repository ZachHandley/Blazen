"""Blazen Streaming Workflow Example

Demonstrates how to observe workflow progress in real-time using event
streaming.

Blazen has two kinds of event delivery:

  1. **Routing events** -- Events returned from steps (or sent via
     ``ctx.send_event()``) are placed on the internal routing queue.
     The engine matches them to steps whose ``accepts`` list includes
     the event type, driving the workflow forward.

  2. **Stream events** -- Events published via ``ctx.write_event_to_stream()``
     are broadcast to an external async iterator.  They do NOT enter the
     routing queue, so they never trigger other steps.  This makes them
     perfect for progress updates, logs, or partial results that a caller
     wants to observe without affecting workflow logic.

This example builds a small two-step pipeline that streams ``Progress``
events while it works, then prints them from an external consumer.

Run with: python streaming_workflow.py
"""

import asyncio

from blazen import Context, Event, StartEvent, StopEvent, Workflow, step


# ---------------------------------------------------------------------------
# Step 1: receive the StartEvent, simulate work, stream progress
# ---------------------------------------------------------------------------

@step
async def process_items(ctx: Context, ev: Event):
    """Process a batch of items, streaming progress after each one."""

    items = ev.items  # list[str] passed via StartEvent kwargs

    for i, item in enumerate(items):
        # Simulate a short async task (e.g. an API call, file I/O, etc.)
        await asyncio.sleep(0.05)

        # Publish a Progress event to the external stream.
        # NOTE: write_event_to_stream is synchronous -- no await needed.
        ctx.write_event_to_stream(
            Event(
                "Progress",
                current=i + 1,
                total=len(items),
                item=item,
            )
        )

    # Hand off to the next step via a routing event.
    return Event("ItemsProcessed", count=len(items))


# ---------------------------------------------------------------------------
# Step 2: summarize and terminate the workflow
# ---------------------------------------------------------------------------

@step(accepts=["ItemsProcessed"])
async def summarize(ctx: Context, ev: Event):
    """Produce the final result and stop the workflow."""

    # Stream one last event so the consumer knows we are wrapping up.
    ctx.write_event_to_stream(
        Event("Progress", current=ev.count, total=ev.count, item="(done)")
    )

    return StopEvent(result={"processed": ev.count, "status": "complete"})


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

async def main():
    # Build the workflow from the two steps.
    wf = Workflow("streaming-demo", [process_items, summarize])

    # Kick off the workflow.  Keyword arguments become the StartEvent payload.
    handler = await wf.run(
        items=["alpha", "bravo", "charlie", "delta", "echo"]
    )

    # Consume streamed events in real-time.
    # stream_events() returns an async iterator that yields every Event
    # published via ctx.write_event_to_stream().  It finishes automatically
    # once the workflow completes.
    print("--- streaming progress ---")
    async for event in handler.stream_events():
        print(
            f"  [{event.event_type}] "
            f"{event.current}/{event.total} - {event.item}"
        )

    # Retrieve the final result (the StopEvent payload).
    result = await handler.result()
    print("--- workflow result ---")
    print(f"  {result.to_dict()}")


if __name__ == "__main__":
    asyncio.run(main())
