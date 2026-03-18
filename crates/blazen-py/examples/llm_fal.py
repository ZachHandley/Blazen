"""Using Blazen with fal.ai as the LLM provider.

fal.ai is FUNDAMENTALLY DIFFERENT from other LLM providers. It is a compute
platform that operates via an async job queue with polling, not direct HTTP
request/response like OpenAI or Anthropic.

How fal.ai works under the hood:

    1. SUBMIT  -- Your request is POST'd to https://queue.fal.run/<model>
                  and you receive a request_id back immediately.
    2. POLL    -- Blazen polls https://queue.fal.run/<model>/requests/<id>/status
                  every ~1 second, waiting for status "COMPLETED" or "FAILED".
    3. FETCH   -- Once complete, the result is GET'd from the request endpoint.

This queue-based architecture means:
    - Latency is inherently higher than direct HTTP providers (extra round-trips).
    - There is no native SSE streaming -- Blazen simulates it by returning the
      complete response as a single chunk.
    - Tool calling is NOT supported (fal-ai/any-llm is text-in, text-out).
    - Image inputs are NOT supported (text-only; non-text content is dropped).
    - The LLM endpoint (fal-ai/any-llm) proxies through OpenRouter, so you
      can access many models (OpenAI, Anthropic, Meta, etc.) via a single key.
    - Authentication uses "Key <token>" format, not "Bearer <token>".

Run with: FAL_KEY=... python llm_fal.py
"""

import asyncio
import os
import time

from blazen import (
    ChatMessage,
    CompletionModel,
    Context,
    Event,
    StopEvent,
    Workflow,
    step,
)

# Module-level model instance. CompletionModel is a native Rust object and
# is NOT JSON-serializable, so it cannot be stored via ctx.set(). We create
# it once in main() and reference it from steps via this module variable.
MODEL: CompletionModel | None = None


# ---------------------------------------------------------------------------
# Step 1: Generate a response using fal.ai
#
# fal.ai uses the fal-ai/any-llm endpoint, which is powered by OpenRouter.
# Under the hood, Blazen submits to the fal.ai job queue, then polls until
# the result is ready. This means:
#   - Expect higher latency than direct providers (queue + poll overhead)
#   - No streaming support (Blazen simulates it with a single chunk)
#   - No tool calling support (tool_calls will always be an empty list)
#   - Text-only input (no image URLs or base64 images)
# ---------------------------------------------------------------------------
@step
async def generate(ctx: Context, ev: Event) -> Event:
    """Submit a prompt to fal.ai and measure the queue-based round-trip."""
    prompt = ev.prompt
    model = MODEL
    assert model is not None, "MODEL must be set before running the workflow"

    print(f"[generate] Submitting to fal.ai queue...")
    print(f"[generate] Prompt: {prompt!r}")

    start = time.monotonic()
    response = await model.complete([
        ChatMessage.system(
            "You are a concise, thoughtful assistant. "
            "Keep responses to 2-3 sentences."
        ),
        ChatMessage.user(prompt),
    ])
    elapsed = time.monotonic() - start

    content = response["content"]
    model_name = response["model"]
    usage = response["usage"]
    finish_reason = response["finish_reason"]

    # tool_calls is always empty for fal.ai -- no tool calling support
    tool_calls = response["tool_calls"]
    assert len(tool_calls) == 0, "fal.ai does not support tool calling"

    print(f"[generate] Response received in {elapsed:.2f}s (includes queue + poll time)")
    print(f"[generate] Model: {model_name}")
    print(f"[generate] Finish reason: {finish_reason}")
    print(f"[generate] Content: {content}")

    # Store timing and response data in context for the next step.
    # ctx.set/ctx.get are SYNCHRONOUS -- no await needed.
    ctx.set("generate_content", content)
    ctx.set("generate_elapsed", elapsed)
    ctx.set("generate_usage", usage)

    return Event("GenerateComplete", content=content)


# ---------------------------------------------------------------------------
# Step 2: Analyze the response quality using fal.ai again
#
# This second call demonstrates that each fal.ai request goes through the
# same queue/poll cycle independently. The total workflow time is the sum
# of both queue round-trips.
# ---------------------------------------------------------------------------
@step(accepts=["GenerateComplete"])
async def analyze(ctx: Context, ev: Event) -> StopEvent:
    """Use fal.ai to analyze the quality of the generated response."""
    content = ev.content
    model = MODEL
    assert model is not None, "MODEL must be set before running the workflow"

    print(f"\n[analyze] Submitting analysis to fal.ai queue...")

    start = time.monotonic()
    response = await model.complete([
        ChatMessage.system(
            "You are a writing quality analyst. "
            "Rate the following text on clarity, conciseness, and helpfulness. "
            "Give a score from 1-10 and a one-sentence explanation."
        ),
        ChatMessage.user(f"Analyze this response:\n\n{content}"),
    ])
    elapsed = time.monotonic() - start

    analysis = response["content"]
    analysis_usage = response["usage"]

    print(f"[analyze] Analysis received in {elapsed:.2f}s")
    print(f"[analyze] Result: {analysis}")

    # Gather all timing and usage info.
    generate_elapsed = ctx.get("generate_elapsed")
    generate_usage = ctx.get("generate_usage")

    return StopEvent(result={
        "original_prompt": ctx.get("generate_content"),
        "generated_response": content,
        "analysis": analysis,
        "timing": {
            "generate_seconds": round(generate_elapsed, 2),
            "analyze_seconds": round(elapsed, 2),
            "total_seconds": round(generate_elapsed + elapsed, 2),
        },
        "usage": {
            "generate": generate_usage,
            "analyze": analysis_usage,
        },
    })


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
async def main() -> None:
    # Load the fal.ai API key from environment.
    fal_key = os.environ.get("FAL_KEY")
    if not fal_key:
        print("ERROR: Set the FAL_KEY environment variable.")
        print("  FAL_KEY=fal-... python llm_fal.py")
        return

    # Create the fal.ai model. By default this uses the fal-ai/any-llm
    # endpoint, which proxies through OpenRouter to access many LLM providers.
    # CompletionModel is a native Rust object (not JSON-serializable), so we
    # store it as a module-level variable rather than in ctx.set().
    global MODEL
    MODEL = CompletionModel.fal(fal_key)

    # You can also specify a different fal.ai model endpoint:
    #   MODEL = CompletionModel.fal(fal_key, model="fal-ai/any-llm")

    print(f"Using model: {MODEL.model_id}")
    print("NOTE: fal.ai uses a queue-based architecture. Each call involves")
    print("      submit -> poll -> fetch, so expect higher latency than")
    print("      direct HTTP providers like OpenAI or Anthropic.\n")

    # Build and run the workflow.
    wf = Workflow("fal-demo", [generate, analyze])
    handler = await wf.run(prompt="What makes Rust's ownership system unique?")

    result = await handler.result()
    output = result.to_dict()["result"]

    print("\n" + "=" * 60)
    print("WORKFLOW COMPLETE")
    print("=" * 60)
    print(f"\nGenerated response:\n  {output['generated_response']}")
    print(f"\nAnalysis:\n  {output['analysis']}")
    print(f"\nTiming (includes fal.ai queue overhead):")
    timing = output["timing"]
    print(f"  Generate step: {timing['generate_seconds']}s")
    print(f"  Analyze step:  {timing['analyze_seconds']}s")
    print(f"  Total:         {timing['total_seconds']}s")

    # Usage info may be None for fal.ai -- it depends on the underlying
    # model and whether OpenRouter returns token counts.
    usage = output["usage"]
    if usage["generate"]:
        u = usage["generate"]
        print(f"\nUsage (generate): {u.get('prompt_tokens', '?')} prompt + "
              f"{u.get('completion_tokens', '?')} completion = "
              f"{u.get('total_tokens', '?')} total tokens")
    else:
        print("\nUsage (generate): not available (fal.ai may not report token usage)")

    if usage["analyze"]:
        u = usage["analyze"]
        print(f"Usage (analyze):  {u.get('prompt_tokens', '?')} prompt + "
              f"{u.get('completion_tokens', '?')} completion = "
              f"{u.get('total_tokens', '?')} total tokens")
    else:
        print("Usage (analyze):  not available")


if __name__ == "__main__":
    asyncio.run(main())
