"""ModelManager + local-embedding smoke test.

Replaces the now-skipped ``test_fal_embeddings`` smoke test. fal does not host a
native text embedder; the only documented path
(``openrouter/router/openai/v1/embeddings``) is a BYOK OpenAI passthrough that
requires a working OpenAI key configured at fal/openrouter, so it can't be
covered reliably in CI.

This test runs entirely locally via ``TractEmbedModel`` (pure-Rust ONNX,
``BGEsmallENV15`` default, 384-d) so it has no network deps beyond a one-time
HuggingFace model download (cached after first run).

The ModelManager wiring uses Python's duck-typed adapter path: any object with
``load``/``unload`` (sync or async) is accepted by ``manager.register(...)``.
``TractEmbedModel`` doesn't implement Rust's ``LocalModel`` trait directly, so
we wrap it in a tiny adapter for ModelManager bookkeeping.
"""

import pytest

from blazen import ModelManager, TractEmbedModel, TractOptions


class _TractAdapter:
    """Duck-typed ``LocalModel`` for ``ModelManager``.

    ``TractEmbedModel`` loads its weights eagerly in its constructor, so the
    adapter just toggles a flag. ``ModelManager`` uses ``load``/``unload`` for
    lifecycle bookkeeping and LRU eviction; the actual inference (``embed``)
    is dispatched directly on ``TractEmbedModel``.
    """

    def __init__(self) -> None:
        self._loaded = False

    async def load(self) -> None:
        self._loaded = True

    async def unload(self) -> None:
        self._loaded = False

    async def is_loaded(self) -> bool:
        return self._loaded


@pytest.mark.asyncio
@pytest.mark.timeout(120)
async def test_model_manager_local_tract_embedder() -> None:
    em = TractEmbedModel(options=TractOptions())
    manager = ModelManager(budget_gb=1.0)

    adapter = _TractAdapter()
    await manager.register("local-bge", adapter, vram_estimate_bytes=200_000_000)

    assert not await manager.is_loaded("local-bge")
    await manager.ensure_loaded("local-bge")
    assert await manager.is_loaded("local-bge")

    response = await em.embed(["hello", "world"])
    assert len(response.embeddings) == 2
    assert len(response.embeddings[0]) == em.dimensions
    assert len(response.embeddings[1]) == em.dimensions
    assert em.dimensions > 0
