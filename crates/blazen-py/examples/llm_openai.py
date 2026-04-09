"""OpenAI-powered content pipeline with Blazen.

Demonstrates using Blazen's CompletionModel with OpenAI to build a real
3-step content generation workflow:

  1. generate_outline  -- GPT creates a blog post outline from a topic
  2. write_draft       -- GPT writes a short draft based on the outline
  3. review            -- GPT provides editorial feedback on the draft

The example also tracks cumulative token usage across all LLM calls using
Blazen's synchronous context API (ctx.set / ctx.get).

Run with: OPENAI_API_KEY=sk-... python llm_openai.py
"""

import asyncio
import os
import sys

from blazen import (
    ChatMessage,
    CompletionModel,
    Context,
    Event,
    ProviderOptions,
    StartEvent,
    StopEvent,
    Workflow,
    step,
)


# ---------------------------------------------------------------------------
# Helper: accumulate token usage in context
# ---------------------------------------------------------------------------
def track_usage(ctx: Context, usage: dict) -> None:
    """Add a single LLM response's token counts to the running totals."""
    prev = ctx.get("total_usage") or {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
    }
    ctx.set("total_usage", {
        "prompt_tokens": prev["prompt_tokens"] + (usage.get("prompt_tokens") or 0),
        "completion_tokens": prev["completion_tokens"] + (usage.get("completion_tokens") or 0),
        "total_tokens": prev["total_tokens"] + (usage.get("total_tokens") or 0),
    })


# ---------------------------------------------------------------------------
# Step 1: Generate an outline
# ---------------------------------------------------------------------------
# Module-level model reference. CompletionModel is a native PyO3 object and
# cannot be serialised to JSON, so we store it here instead of in context.
MODEL: CompletionModel | None = None


@step
async def generate_outline(ctx: Context, ev: Event) -> Event:
    """Ask GPT to produce a concise blog post outline for the given topic."""
    topic: str = ev.topic

    print(f"[generate_outline] Requesting outline for: {topic!r}")

    response = await MODEL.complete(
        [
            ChatMessage.system(
                "You are a professional content strategist. "
                "When given a topic, produce a concise blog post outline "
                "with 3-5 sections. Keep it brief -- just section titles "
                "and one-line descriptions."
            ),
            ChatMessage.user(f"Create a blog post outline about: {topic}"),
        ],
        temperature=0.7,
        max_tokens=512,
    )

    outline = response["content"]
    track_usage(ctx, response["usage"])

    print(f"[generate_outline] Outline received ({response['usage']['total_tokens']} tokens)")
    print(f"\n--- Outline ---\n{outline}\n")

    return Event("OutlineEvent", outline=outline, topic=topic)


# ---------------------------------------------------------------------------
# Step 2: Write a short draft
# ---------------------------------------------------------------------------
@step(accepts=["OutlineEvent"])
async def write_draft(ctx: Context, ev: Event) -> Event:
    """Ask GPT to write a short first draft based on the outline."""

    print("[write_draft] Writing draft from outline...")

    response = await MODEL.complete(
        [
            ChatMessage.system(
                "You are a skilled blog writer. Given an outline, write a "
                "short but engaging first draft (2-3 paragraphs). "
                "Keep the tone approachable and informative."
            ),
            ChatMessage.user(
                f"Topic: {ev.topic}\n\nOutline:\n{ev.outline}\n\n"
                "Write a short draft based on this outline."
            ),
        ],
        temperature=0.8,
        max_tokens=768,
    )

    draft = response["content"]
    track_usage(ctx, response["usage"])

    print(f"[write_draft] Draft received ({response['usage']['total_tokens']} tokens)")
    print(f"\n--- Draft ---\n{draft}\n")

    return Event("DraftEvent", draft=draft, topic=ev.topic)


# ---------------------------------------------------------------------------
# Step 3: Editorial review
# ---------------------------------------------------------------------------
@step(accepts=["DraftEvent"])
async def review(ctx: Context, ev: Event) -> StopEvent:
    """Ask GPT to provide brief editorial feedback on the draft."""
    print("[review] Reviewing draft...")

    response = await MODEL.complete(
        [
            ChatMessage.system(
                "You are a senior editor. Review the following blog draft "
                "and provide 3-5 bullet points of constructive feedback. "
                "Be specific and actionable."
            ),
            ChatMessage.user(
                f"Topic: {ev.topic}\n\nDraft:\n{ev.draft}\n\n"
                "Provide editorial feedback."
            ),
        ],
        temperature=0.5,
        max_tokens=512,
    )

    feedback = response["content"]
    track_usage(ctx, response["usage"])

    print(f"[review] Feedback received ({response['usage']['total_tokens']} tokens)")
    print(f"\n--- Editorial Feedback ---\n{feedback}\n")

    total_usage = ctx.get("total_usage")

    return StopEvent(result={
        "topic": ev.topic,
        "draft": ev.draft,
        "feedback": feedback,
        "total_usage": total_usage,
    })


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
async def main() -> None:
    # Validate that the API key is available.
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print(
            "ERROR: OPENAI_API_KEY environment variable is not set.\n"
            "Run with: OPENAI_API_KEY=sk-... python llm_openai.py",
            file=sys.stderr,
        )
        sys.exit(1)

    # Create the OpenAI completion model (gpt-5.3-chat-latest is cheap for examples).
    # Stored at module level because CompletionModel is a native object that
    # cannot be serialised into the JSON-based workflow context.
    global MODEL
    MODEL = CompletionModel.openai(options=ProviderOptions(api_key=api_key, model="gpt-5.3-chat-latest"))

    # Build the 3-step content pipeline.
    wf = Workflow("content-pipeline", [generate_outline, write_draft, review])

    topic = "Why Rust-powered Python libraries are the future of performance"
    handler = await wf.run(topic=topic)
    result = await handler.result()
    output = result.to_dict()

    # Print summary.
    print("=" * 60)
    print("CONTENT PIPELINE COMPLETE")
    print("=" * 60)

    usage = output["result"]["total_usage"]
    print(f"Topic:              {output['result']['topic']}")
    print(f"Prompt tokens:      {usage['prompt_tokens']}")
    print(f"Completion tokens:  {usage['completion_tokens']}")
    print(f"Total tokens:       {usage['total_tokens']}")


if __name__ == "__main__":
    asyncio.run(main())
