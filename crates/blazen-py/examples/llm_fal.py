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
    - Authentication uses "Key <token>" format, not "Bearer <token>".
    - The default LLM endpoint is fal's OpenAI-compatible chat-completions
      surface (``FalLlmEndpoint.OPENAI_CHAT``). Other variants include
      ``OPENAI_RESPONSES``, ``OPENROUTER``, and ``ANY_LLM``, each with an
      ``_ENTERPRISE`` cousin for SOC2-eligible traffic.

Beyond LLM completions, ``FalProvider`` exposes the full fal.ai compute API:
image / video / audio / 3D generation, transcription, background removal,
and text embeddings. The tail of this file demonstrates each of these.

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
    FalLlmEndpoint,
    FalOptions,
    FalProvider,
    StopEvent,
    ThreeDRequest,
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
# fal.ai's default LLM endpoint is the OpenAI-compatible chat-completions
# surface. Under the hood, Blazen submits to the fal.ai job queue, then
# polls until the result is ready. This means:
#   - Expect higher latency than direct providers (queue + poll overhead)
#   - No native streaming support (Blazen simulates it with a single chunk)
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
# Vision / audio / video extras
#
# fal.ai auto-routes multimodal chat requests to the matching vision, audio,
# or video-capable variant of the underlying router (controlled by
# ``FalOptions(auto_route_modality=True)``, the default). The helpers below
# demonstrate each modality using the same ``CompletionModel.fal`` instance.
# ---------------------------------------------------------------------------
async def demo_vision(model: CompletionModel) -> None:
    """Send a text + image-URL message through fal.ai (vision input)."""
    print("\n[vision] Describing an image via fal.ai...")
    response = await model.complete([
        ChatMessage.user_image_url(
            text="What is in this picture? Answer in one sentence.",
            url="https://upload.wikimedia.org/wikipedia/commons/thumb/4/47/PNG_transparency_demonstration_1.png/280px-PNG_transparency_demonstration_1.png",
        ),
    ])
    print(f"[vision] {response['content']}")


async def demo_audio(model: CompletionModel) -> None:
    """Send a text + audio-URL message through fal.ai (audio input)."""
    print("\n[audio] Transcribing/analysing a clip via fal.ai...")
    response = await model.complete([
        ChatMessage.user_audio(
            text="Summarise what you hear in one sentence.",
            url="https://storage.googleapis.com/falserverless/model_tests/whisper/dinner_conversation.mp3",
        ),
    ])
    print(f"[audio] {response['content']}")


async def demo_video(model: CompletionModel) -> None:
    """Send a text + video-URL message through fal.ai (video input)."""
    print("\n[video] Describing a video clip via fal.ai...")
    response = await model.complete([
        ChatMessage.user_video(
            text="What is happening in this video? Answer in one sentence.",
            url="https://storage.googleapis.com/falserverless/model_tests/video_models/robot.mp4",
        ),
    ])
    print(f"[video] {response['content']}")


# ---------------------------------------------------------------------------
# Non-LLM compute extras (embeddings, 3D, background removal)
#
# These live on ``FalProvider`` directly. ``FalProvider`` exposes the full
# fal.ai compute surface (image / video / audio / 3D / transcription /
# embeddings / background removal) in addition to the ``CompletionModel``
# interface.
# ---------------------------------------------------------------------------
async def demo_embeddings(provider: FalProvider) -> None:
    """Embed two short strings via fal's OpenAI-compatible embeddings endpoint."""
    print("\n[embeddings] Embedding via fal.ai...")
    em = provider.embedding_model()
    response = await em.embed(["hi", "hello world"])
    print(f"[embeddings] model={em.model_id} dims={em.dimensions} "
          f"n_vectors={len(response.embeddings)}")


async def demo_generate_3d(provider: FalProvider) -> None:
    """Generate a 3D model from a text prompt via fal.ai."""
    print("\n[3d] Generating a 3D model via fal.ai...")
    result = await provider.generate_3d(ThreeDRequest(
        prompt="a low-poly wooden treasure chest",
        format="glb",
    ))
    models = result.get("models", [])
    print(f"[3d] generated {len(models)} model(s); "
          f"elapsed={result.get('elapsed_ms', '?')}ms")


async def demo_remove_background(provider: FalProvider) -> None:
    """Run background removal on a sample image URL via fal.ai."""
    print("\n[bg-remove] Removing background via fal.ai...")
    result = await provider.remove_background(
        image_url="https://storage.googleapis.com/falserverless/model_tests/remove_bg/elephant.jpg",
    )
    image = result.get("image") or {}
    print(f"[bg-remove] got matted image url={image.get('url', '?')}")


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

    # Create the fal.ai model. By default this uses the OpenAiChat endpoint
    # (openrouter/router/openai/v1/chat/completions), which provides full
    # OpenAI chat-completions semantics: messages array, tool calls, structured
    # outputs, native streaming.
    # CompletionModel is a native Rust object (not JSON-serializable), so we
    # store it as a module-level variable rather than in ctx.set().
    global MODEL
    MODEL = CompletionModel.fal(fal_key)

    # You can also pin a specific model, endpoint, or the enterprise tier
    # by constructing a ``FalOptions`` and passing it as ``options=``. The
    # example below is built but not used -- uncomment the reassignment to
    # switch ``MODEL`` over to the SOC2-eligible variant.
    enterprise_opts = FalOptions(
        model="anthropic/claude-sonnet-4.5",
        endpoint=FalLlmEndpoint.OPENAI_CHAT,
        enterprise=True,
    )
    _ = enterprise_opts  # silence "unused" lint
    # MODEL = CompletionModel.fal(fal_key, options=enterprise_opts)

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
    # model and whether the router returns token counts.
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

    # -------------------------------------------------------------------
    # Optional multimodal / compute demos.
    #
    # Each demo block is wrapped in try/except so that a failure in one
    # (e.g. an unavailable sample URL or a restricted endpoint) does not
    # stop the rest of the example from running.
    # -------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("MULTIMODAL + COMPUTE DEMOS")
    print("=" * 60)

    for demo in (demo_vision, demo_audio, demo_video):
        try:
            await demo(MODEL)
        except Exception as exc:
            print(f"[{demo.__name__}] skipped: {exc}")

    provider = FalProvider(api_key=fal_key)
    for demo in (demo_embeddings, demo_generate_3d, demo_remove_background):
        try:
            await demo(provider)
        except Exception as exc:
            print(f"[{demo.__name__}] skipped: {exc}")


if __name__ == "__main__":
    asyncio.run(main())
