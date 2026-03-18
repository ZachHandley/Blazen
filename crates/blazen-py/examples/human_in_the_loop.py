"""Human-in-the-loop workflow with side-effect steps.

Demonstrates two key Blazen patterns:

1. **Side-effect steps** -- A step that performs work (stores state in
   context, logs, calls an external API, etc.) but does not produce an
   output event by returning ``None``.  Instead it uses
   ``ctx.send_event()`` to manually route the next event when ready.

2. **Human-in-the-loop simulation** -- A review step that pauses to
   inspect intermediate results before deciding to approve or reject.
   In a real application you would replace the simulated review with
   an actual human interaction (webhook, UI callback, message queue).

Flow::

    StartEvent  ->  process_submission  ->  (side-effect, no return)
                        |
                        +-- ctx.send_event(ReviewEvent) -->  review_submission
                                                                |
                                                                +-- ctx.send_event(ApprovedEvent)
                                                                        |
                                                                        v
                                                                    finalize  ->  StopEvent

Key API notes:
    - ``ctx.set()`` / ``ctx.get()`` are synchronous (no await).
    - ``ctx.send_event()`` is synchronous -- it routes an event to the
      step whose ``accepts`` list matches.
    - Returning ``None`` from a step means "I handled this event but
      do not produce an output event through the normal return path."

Run with: python human_in_the_loop.py
"""

import asyncio

from blazen import Context, Event, StartEvent, StopEvent, Workflow, step


# ---------------------------------------------------------------------------
# Step 1: Process the incoming submission.
#
# Accepts the default StartEvent.  Validates the data, stores
# intermediate state in the shared context, then uses ctx.send_event()
# to forward a ReviewEvent.  Returns None (side-effect only).
# ---------------------------------------------------------------------------
@step
async def process_submission(ctx: Context, ev: Event) -> None:
    title = ev.title
    body = ev.body

    # Simulate processing -- normalise and store intermediate results.
    processed_title = title.strip().title()
    word_count = len(body.split())

    # Persist intermediate state so downstream steps can read it.
    ctx.set("processed_title", processed_title)
    ctx.set("word_count", word_count)
    ctx.set("original_body", body)

    print(f"  [process] Title normalised to: {processed_title!r}")
    print(f"  [process] Word count: {word_count}")

    # Instead of returning an Event, we manually route a ReviewEvent.
    # This is the side-effect pattern: do work, call send_event, return None.
    ctx.send_event(Event("ReviewEvent", title=processed_title, word_count=word_count))
    return None


# ---------------------------------------------------------------------------
# Step 2: Simulate a human review.
#
# In production this could pause and wait for a webhook, poll a queue,
# or call out to a UI.  Here we apply a simple automatic rule to
# demonstrate the pattern.
# ---------------------------------------------------------------------------
@step(accepts=["ReviewEvent"])
async def review_submission(ctx: Context, ev: Event) -> None:
    title = ev.title
    word_count = ev.word_count

    print(f"  [review]  Reviewing: {title!r} ({word_count} words)")

    # Simulated human decision: approve if body has at least 3 words.
    approved = word_count >= 3
    reason = "Meets minimum length" if approved else "Too short"

    # Store the review decision in context for the finalizer.
    ctx.set("approved", approved)
    ctx.set("review_reason", reason)

    print(f"  [review]  Decision: {'APPROVED' if approved else 'REJECTED'} -- {reason}")

    # Again, side-effect pattern: route manually and return None.
    ctx.send_event(Event("ReviewedEvent", approved=approved))
    return None


# ---------------------------------------------------------------------------
# Step 3: Finalize the workflow.
#
# Gathers everything from context and returns a StopEvent with the
# full result.
# ---------------------------------------------------------------------------
@step(accepts=["ReviewedEvent"])
async def finalize(ctx: Context, ev: Event) -> StopEvent:
    return StopEvent(result={
        "title": ctx.get("processed_title"),
        "word_count": ctx.get("word_count"),
        "approved": ctx.get("approved"),
        "review_reason": ctx.get("review_reason"),
        "original_body": ctx.get("original_body"),
    })


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
async def main() -> None:
    wf = Workflow("human-review", [process_submission, review_submission, finalize])

    print("Running workflow (submission that passes review)...")
    handler = await wf.run(
        title="  my first blazen workflow  ",
        body="This is a perfectly reasonable submission with enough words.",
    )
    result = await handler.result()
    print(f"  Result: {result.to_dict()}\n")

    print("Running workflow (submission that fails review)...")
    handler = await wf.run(
        title="short",
        body="Too few",
    )
    result = await handler.result()
    print(f"  Result: {result.to_dict()}")


if __name__ == "__main__":
    asyncio.run(main())
