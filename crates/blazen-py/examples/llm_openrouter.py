"""Using Blazen with OpenRouter as the LLM provider.

OpenRouter (https://openrouter.ai) is a unified API gateway that proxies
requests to 200+ models from providers like Anthropic, OpenAI, Meta, Google,
Mistral, and many more -- all through a single API key.  This makes it easy
to swap models without changing provider credentials or endpoint configuration.

OpenRouter supports 200+ models.  Change DEFAULT_MODEL below to try a
different one, e.g. "openai/gpt-4o", "meta-llama/llama-3-70b",
"google/gemini-2.0-flash", etc.

This example builds a two-step workflow:

    StartEvent  ->  generate_poem  ->  PoemEvent
    PoemEvent   ->  summarize      ->  StopEvent

Both steps call OpenRouter via ``CompletionModel.openrouter()`` to demonstrate
real LLM completions inside a Blazen pipeline.

Run with: OPENROUTER_API_KEY=sk-or-... python llm_openrouter.py
"""

import asyncio
import os
import sys

from blazen import (
    ChatMessage,
    CompletionModel,
    Context,
    Event,
    StopEvent,
    Workflow,
    step,
)

# Default model routed through OpenRouter.  You can swap this to any model
# OpenRouter supports, e.g. "openai/gpt-4o", "meta-llama/llama-3-70b",
# "google/gemini-2.0-flash", etc.
DEFAULT_MODEL = "anthropic/claude-sonnet-4-6"

# Module-level model instance -- initialised in main() before the workflow runs.
llm: CompletionModel


# ---------------------------------------------------------------------------
# Step 1: Generate a short poem about the given topic.
# ---------------------------------------------------------------------------
@step
async def generate_poem(ctx: Context, ev: Event) -> Event:
    """Take a topic from the StartEvent and ask the LLM to write a short poem."""
    topic: str = ev.topic

    print(f"[generate_poem] Asking for a poem about: {topic!r}")

    response = await llm.complete(
        [
            ChatMessage.system(
                "You are a creative poet. Write short, vivid poems (4-8 lines)."
            ),
            ChatMessage.user(f"Write a short poem about: {topic}"),
        ],
        max_tokens=256,
    )

    poem = response["content"]
    usage = response["usage"]

    print(f"[generate_poem] Received poem ({usage['total_tokens']} tokens used)")

    # Store the first call's usage in context so we can tally it later.
    ctx.set("poem_usage", usage)

    return Event(
        "PoemEvent",
        poem=poem,
        model_used=response["model"],
    )


# ---------------------------------------------------------------------------
# Step 2: Summarize the poem in a single sentence.
# ---------------------------------------------------------------------------
@step(accepts=["PoemEvent"])
async def summarize(ctx: Context, ev: Event) -> StopEvent:
    """Take the generated poem and produce a one-sentence summary."""
    poem: str = ev.poem

    print("[summarize] Summarizing the poem...")

    response = await llm.complete(
        [
            ChatMessage.system(
                "You are a concise literary critic. Summarize poems in exactly "
                "one sentence."
            ),
            ChatMessage.user(f"Summarize this poem in one sentence:\n\n{poem}"),
        ],
        max_tokens=128,
    )

    summary = response["content"]
    summary_usage = response["usage"]

    print(f"[summarize] Done ({summary_usage['total_tokens']} tokens used)")

    # Retrieve the first step's usage for the final tally.
    poem_usage = ctx.get("poem_usage")

    return StopEvent(
        result={
            "poem": poem,
            "summary": summary,
            "model": ev.model_used,
            "poem_usage": poem_usage,
            "summary_usage": summary_usage,
        }
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
async def main() -> None:
    global llm

    # ------------------------------------------------------------------
    # 1. Load API key from environment
    # ------------------------------------------------------------------
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        print(
            "ERROR: OPENROUTER_API_KEY environment variable is not set.\n"
            "\n"
            "Get a free key at https://openrouter.ai/keys and run:\n"
            "  OPENROUTER_API_KEY=sk-or-... python llm_openrouter.py",
            file=sys.stderr,
        )
        sys.exit(1)

    # ------------------------------------------------------------------
    # 2. Create the CompletionModel targeting OpenRouter
    # ------------------------------------------------------------------
    llm = CompletionModel.openrouter(api_key, model=DEFAULT_MODEL)
    print(f"Using model: {llm.model_id}")
    print()

    # ------------------------------------------------------------------
    # 3. Build and run the workflow
    # ------------------------------------------------------------------
    wf = Workflow("poem-pipeline", [generate_poem, summarize])
    handler = await wf.run(topic="the beauty of open-source software")

    result = await handler.result()
    output = result.to_dict()

    # ------------------------------------------------------------------
    # 4. Print results and token usage
    # ------------------------------------------------------------------
    print()
    print("=" * 60)
    print("POEM")
    print("=" * 60)
    print(output["result"]["poem"])

    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(output["result"]["summary"])

    print()
    print("=" * 60)
    print("TOKEN USAGE")
    print("=" * 60)
    poem_usage = output["result"]["poem_usage"]
    summary_usage = output["result"]["summary_usage"]
    print(f"  Model:              {output['result']['model']}")
    print(f"  Poem generation:    {poem_usage['total_tokens']} tokens "
          f"(prompt: {poem_usage['prompt_tokens']}, "
          f"completion: {poem_usage['completion_tokens']})")
    print(f"  Summarization:      {summary_usage['total_tokens']} tokens "
          f"(prompt: {summary_usage['prompt_tokens']}, "
          f"completion: {summary_usage['completion_tokens']})")
    total = poem_usage["total_tokens"] + summary_usage["total_tokens"]
    print(f"  Total:              {total} tokens")


if __name__ == "__main__":
    asyncio.run(main())
