"""Local LLM inference with mistral.rs (no API key needed).

mistral.rs runs LLM inference entirely on-device using optimised Rust kernels.
It supports HuggingFace model IDs (automatic download + caching) and local GGUF
files.  No API key, no network access (after initial model download), and no
usage-based billing.

The first run downloads model weights from HuggingFace Hub and caches them
locally.  Subsequent runs reuse the cached weights automatically.

Unlike the cloud-based providers (OpenAI, Anthropic, etc.), mistral.rs is
ideal for local development, CI pipelines, air-gapped deployments, and
on-device inference where data must not leave the machine.

Usage:
    uv run python crates/blazen-py/examples/llm_mistralrs.py
"""

import asyncio

from blazen import ChatMessage, CompletionModel, CompletionOptions, MistralRsOptions


async def main() -> None:
    # ------------------------------------------------------------------
    # 1. Create a local completion model with mistral.rs
    # ------------------------------------------------------------------
    # Uses a small GGUF-quantised model suitable for quick demos.
    # Replace with any HuggingFace model ID or local GGUF path.
    opts = MistralRsOptions("TheBloke/Mistral-7B-Instruct-v0.2-GGUF")
    model = CompletionModel.mistralrs(options=opts)

    print(f"Model ID: {model.model_id}")
    print()

    # ------------------------------------------------------------------
    # 2. Simple completion
    # ------------------------------------------------------------------
    print("--- Simple Completion ---")
    response = await model.complete([
        ChatMessage.system(
            "You are a helpful assistant. Answer concisely in 1-2 sentences."
        ),
        ChatMessage.user("What is 2+2?"),
    ])

    print(f"Response: {response.content}")
    print(f"Model:    {response.model}")
    if response.usage:
        print(f"Tokens:   {response.usage}")
    print()

    # ------------------------------------------------------------------
    # 3. Completion with options (temperature, max_tokens)
    # ------------------------------------------------------------------
    print("--- Completion With Options ---")
    response = await model.complete(
        [
            ChatMessage.system("You are a creative storyteller."),
            ChatMessage.user("Tell me a very short story about a robot learning to cook."),
        ],
        CompletionOptions(temperature=0.9, max_tokens=256),
    )

    print(f"Response: {response.content}")
    print()

    # ------------------------------------------------------------------
    # 4. Streaming completion
    # ------------------------------------------------------------------
    print("--- Streaming ---")

    chunks_received = 0

    async for chunk in model.stream([
        ChatMessage.user("Explain why the sky is blue in 2-3 sentences."),
    ]):
        if chunk.delta:
            print(chunk.delta, end="", flush=True)
        chunks_received += 1

    print()
    print(f"\n(received {chunks_received} chunks)")
    print()

    print("Done.")


if __name__ == "__main__":
    asyncio.run(main())
