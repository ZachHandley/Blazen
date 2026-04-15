"""Local embeddings with the embed backend (no API key needed).

The embed backend runs entirely on-device using ONNX Runtime.  The default
model is BAAI/bge-small-en-v1.5 (384 dimensions, ~33 MB download on first
run).  Subsequent runs use the cached model automatically.

Unlike the cloud-based embedding providers (OpenAI, Cohere, etc.), the
local embed backend requires no API key, no network access after the
initial model download, and no usage-based billing.  This makes it ideal
for local development, CI pipelines, and air-gapped deployments.

Usage:
    uv run python crates/blazen-py/examples/llm_embed.py
"""

import asyncio

from blazen import EmbeddingModel, EmbedOptions


async def main() -> None:
    # ------------------------------------------------------------------
    # 1. Create a local embedding model with default settings
    # ------------------------------------------------------------------
    # Default model: BAAI/bge-small-en-v1.5 (384 dimensions)
    model = EmbeddingModel.local()

    print(f"Model:      {model.model_id}")
    print(f"Dimensions: {model.dimensions}")
    print()

    # ------------------------------------------------------------------
    # 2. Embed a batch of texts
    # ------------------------------------------------------------------
    texts = [
        "Hello, world!",
        "Blazen local embeddings are fast and free.",
        "ONNX Runtime provides cross-platform inference.",
    ]

    print(f"Embedding {len(texts)} texts...")
    response = await model.embed(texts)

    print(f"Received {len(response.embeddings)} vectors\n")
    for i, emb in enumerate(response.embeddings):
        preview = ", ".join(f"{v:.6f}" for v in emb[:5])
        print(f"  [{i}] length={len(emb)}  first 5: [{preview}]")

    # ------------------------------------------------------------------
    # 3. Create a model with explicit options
    # ------------------------------------------------------------------
    print()
    print("=" * 60)
    print("WITH EXPLICIT OPTIONS")
    print("=" * 60)

    opts = EmbedOptions(
        model_name="BGESmallENV15",
        show_download_progress=True,
    )
    custom_model = EmbeddingModel.local(options=opts)

    print(f"Model:      {custom_model.model_id}")
    print(f"Dimensions: {custom_model.dimensions}")

    response = await custom_model.embed(["Custom options work too!"])
    print(f"Embedded 1 text -> vector length={len(response.embeddings[0])}")


if __name__ == "__main__":
    asyncio.run(main())
