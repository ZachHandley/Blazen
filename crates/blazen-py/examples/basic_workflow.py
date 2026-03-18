"""Basic 3-step Blazen workflow example.

Demonstrates the core Blazen Python API by building a simple greeting
pipeline with three steps:

    StartEvent -> parse_input -> GreetEvent
    GreetEvent -> transform   -> FormattedEvent
    FormattedEvent -> greet   -> StopEvent

Each step receives a Context (shared key/value store) and an Event,
then returns a new Event to route to the next step. The workflow
finishes when a StopEvent is returned.

Run with: python basic_workflow.py
"""

import asyncio

from blazen import Context, Event, StartEvent, StopEvent, Workflow, step


# Step 1: Parse the incoming start payload.
# By default, @step accepts StartEvent, so no `accepts` argument is needed.
@step
async def parse_input(ctx: Context, ev: Event) -> Event:
    name = ev.name
    greeting_style = ev.style

    # Store the original name in shared context for later steps.
    ctx.set("original_name", name)

    return Event("GreetEvent", name=name, style=greeting_style)


# Step 2: Transform the greeting into a formatted message.
# Accepts only GreetEvent -- it will not fire on StartEvent or other types.
@step(accepts=["GreetEvent"])
async def transform(ctx: Context, ev: Event) -> Event:
    name = ev.name
    style = ev.style

    if style == "formal":
        message = f"Good day, {name}. It is a pleasure to meet you."
    elif style == "casual":
        message = f"Hey {name}, what's up!"
    else:
        message = f"Hello, {name}!"

    return Event("FormattedEvent", message=message)


# Step 3: Finalize and stop the workflow.
# Returning a StopEvent ends execution and delivers the result.
@step(accepts=["FormattedEvent"])
async def greet(ctx: Context, ev: Event) -> StopEvent:
    # Read back from shared context to show ctx.get() usage.
    original_name = ctx.get("original_name")

    return StopEvent(result={
        "greeting": ev.message,
        "original_name": original_name,
    })


async def main() -> None:
    # Build the workflow from the three steps.
    wf = Workflow("greeting-pipeline", [parse_input, transform, greet])

    # Run the workflow. Keyword arguments become the StartEvent payload.
    handler = await wf.run(name="Alice", style="formal")

    # Await the final StopEvent.
    result = await handler.result()

    print("Workflow complete!")
    print(f"  Result: {result.to_dict()}")


if __name__ == "__main__":
    asyncio.run(main())
