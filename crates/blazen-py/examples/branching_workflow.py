"""Conditional branching (fan-out) in a Blazen workflow.

Demonstrates how a single step can dispatch multiple events to different
downstream steps, creating parallel branches that race to completion.

Key concepts:
  - **Fan-out**: Return a list of Events from a step to dispatch them all
    at once.  Each event is routed independently based on its event type.
  - **Routing**: Use ``accepts=["EventType"]`` on a step to subscribe it
    to a specific event type.  Steps without an explicit ``accepts`` only
    receive ``StartEvent``.
  - **First StopEvent wins**: When several branches race, the first
    ``StopEvent`` to arrive terminates the workflow and becomes the result.

Run with: python branching_workflow.py
"""

import asyncio

from blazen import Context, Event, StartEvent, StopEvent, Workflow, step

# A handful of positive and negative keywords used for our toy classifier.
POSITIVE_WORDS = {"great", "wonderful", "fantastic", "excellent", "happy", "love", "amazing"}
NEGATIVE_WORDS = {"terrible", "awful", "horrible", "bad", "hate", "worst", "sad"}


# ---------------------------------------------------------------------------
# Steps
# ---------------------------------------------------------------------------


@step  # Accepts StartEvent by default
async def analyze_sentiment(ctx: Context, ev: Event):
    """Examine the input text and route to the appropriate sentiment branch.

    Returns a *list* of one Event -- either PositiveSentiment or
    NegativeSentiment -- so the downstream handler for that type picks it up.
    Returning a list (even of length one) is the idiomatic fan-out pattern;
    you could return two or more events to trigger multiple branches in
    parallel.
    """
    text: str = ev.text.lower()

    # Count keyword hits for each sentiment.
    pos_hits = sum(1 for w in POSITIVE_WORDS if w in text)
    neg_hits = sum(1 for w in NEGATIVE_WORDS if w in text)

    # Store the original text in shared context so downstream steps can read it.
    ctx.set("original_text", ev.text)

    if pos_hits >= neg_hits:
        # Route to the positive branch.
        return [Event("PositiveSentiment", score=pos_hits)]
    else:
        # Route to the negative branch.
        return [Event("NegativeSentiment", score=neg_hits)]


@step(accepts=["PositiveSentiment"])
async def handle_positive(ctx: Context, ev: Event):
    """Handle text that was classified as positive."""
    original = ctx.get("original_text")
    return StopEvent(result={
        "sentiment": "positive",
        "confidence_hits": ev.score,
        "text": original,
    })


@step(accepts=["NegativeSentiment"])
async def handle_negative(ctx: Context, ev: Event):
    """Handle text that was classified as negative."""
    original = ctx.get("original_text")
    return StopEvent(result={
        "sentiment": "negative",
        "confidence_hits": ev.score,
        "text": original,
    })


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def main():
    # Build the workflow from all three steps.
    wf = Workflow("sentiment-branching", [analyze_sentiment, handle_positive, handle_negative])

    samples = [
        "I had a wonderful and fantastic day at the park!",
        "That was the worst experience, absolutely terrible service.",
        "The weather is okay today, nothing special.",
    ]

    for text in samples:
        # Kick off the workflow; keyword arguments become StartEvent attributes.
        handler = await wf.run(text=text)
        result = await handler.result()

        data = result.to_dict()
        branch = data.get("sentiment", "unknown")
        hits = data.get("confidence_hits", 0)

        print(f"Text:      {text}")
        print(f"Branch:    {branch}")
        print(f"Keyword hits: {hits}")
        print()


if __name__ == "__main__":
    asyncio.run(main())
