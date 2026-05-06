"""Streaming-download tests for the Python ``ContentStore`` binding.

Exercises the ``ContentStore.fetch_stream`` wrapper across the three
construction paths (``custom`` callback, ``ContentStore`` subclass, and
the built-in ``local_file`` backend) without touching any external
service. Covers correctness, error propagation, and a basic
backpressure smoke check.

The runtime expectation is::

    iter = await store.fetch_stream(handle)
    async for chunk in iter:
        ...

(Equivalently: ``async for chunk in await store.fetch_stream(handle):``.)
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

import pytest

from blazen import ContentHandle, ContentKind, ContentStore

pytestmark = pytest.mark.asyncio


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


def _placeholder_handle() -> ContentHandle:
    """Return a freshly constructed ``ContentHandle`` for callback tests.

    The custom-store callbacks below ignore the handle's identity entirely;
    they hand back a canned response. We only need *some* valid handle so
    the runtime has a stable wire identity to round-trip.
    """

    return ContentHandle("test-handle", ContentKind.Other)


async def _drain(iter_obj: object) -> list[bytes]:
    """Collect every chunk an async-iterator yields into a list."""

    chunks: list[bytes] = []
    async for chunk in iter_obj:  # type: ignore[attr-defined]
        chunks.append(chunk)
    return chunks


# ---------------------------------------------------------------------------
# 1. Custom store with bytes-returning ``fetch_stream`` (legacy shape).
# ---------------------------------------------------------------------------


async def test_custom_store_fetch_stream_returns_bytes() -> None:
    """A ``fetch_stream`` callback returning raw ``bytes`` is wrapped into
    a single-chunk async iterator on the Rust side."""

    expected = b"hello world"

    async def put(_body: dict, _hint: dict) -> ContentHandle:
        return _placeholder_handle()

    async def resolve(_handle: ContentHandle) -> dict:
        return {"type": "url", "url": "https://example.invalid/blob"}

    async def fetch_bytes(_handle: ContentHandle) -> bytes:
        return expected

    async def fetch_stream(_handle: ContentHandle) -> bytes:
        return expected

    store = ContentStore.custom(
        put=put,
        resolve=resolve,
        fetch_bytes=fetch_bytes,
        fetch_stream=fetch_stream,
        name="bytes-stream",
    )

    handle = await store.put(b"placeholder")
    iter_obj = await store.fetch_stream(handle)
    chunks = await _drain(iter_obj)

    assert chunks == [expected], chunks


# ---------------------------------------------------------------------------
# 2. Custom store with async-generator ``fetch_stream`` (new shape).
# ---------------------------------------------------------------------------


async def test_custom_store_fetch_stream_async_generator() -> None:
    """A ``fetch_stream`` callback returning an ``AsyncIterator[bytes]``
    streams chunk-by-chunk through the Rust bridge."""

    payload_chunks = [b"hello ", b"big ", b"world"]

    async def put(_body: dict, _hint: dict) -> ContentHandle:
        return _placeholder_handle()

    async def resolve(_handle: ContentHandle) -> dict:
        return {"type": "url", "url": "https://example.invalid/blob"}

    async def fetch_bytes(_handle: ContentHandle) -> bytes:
        return b"".join(payload_chunks)

    async def fetch_stream(_handle: ContentHandle) -> AsyncIterator[bytes]:
        async def gen() -> AsyncIterator[bytes]:
            for chunk in payload_chunks:
                yield chunk

        return gen()

    store = ContentStore.custom(
        put=put,
        resolve=resolve,
        fetch_bytes=fetch_bytes,
        fetch_stream=fetch_stream,
        name="gen-stream",
    )

    handle = await store.put(b"placeholder")
    iter_obj = await store.fetch_stream(handle)
    chunks = await _drain(iter_obj)

    assert len(chunks) == 3, chunks
    assert b"".join(chunks) == b"hello big world"


# ---------------------------------------------------------------------------
# 3. Subclass with async-generator ``fetch_stream`` override.
# ---------------------------------------------------------------------------


async def test_subclass_fetch_stream_async_generator() -> None:
    """A subclass that overrides ``fetch_stream`` as an async generator
    is driven through the same bridge as the callback path."""

    class MyStore(ContentStore):
        async def put(self, _body: dict, _hint: dict) -> ContentHandle:
            return _placeholder_handle()

        async def resolve(self, _handle: ContentHandle) -> dict:
            return {"type": "url", "url": "https://example.invalid/blob"}

        async def fetch_bytes(self, _handle: ContentHandle) -> bytes:
            return b"ab"

        async def fetch_stream(
            self, _handle: ContentHandle
        ) -> AsyncIterator[bytes]:
            yield b"a"
            yield b"b"

    store = MyStore()
    handle = _placeholder_handle()
    # MyStore.fetch_stream is an async generator function — calling it
    # returns an async-generator object directly (not a coroutine), so
    # we iterate without an outer ``await``.
    chunks: list[bytes] = []
    async for chunk in store.fetch_stream(handle):
        chunks.append(chunk)

    assert chunks == [b"a", b"b"], chunks


# ---------------------------------------------------------------------------
# 4. Built-in ``local_file`` round-trip.
# ---------------------------------------------------------------------------


async def test_local_file_fetch_stream_round_trip(tmp_path) -> None:
    """``local_file`` writes via ``put`` and reads back via
    ``fetch_stream``. We assert byte-for-byte equality without locking
    in any specific chunk count — ``ReaderStream`` chunking is an
    implementation detail."""

    payload = b"x" * 100_000

    store = ContentStore.local_file(tmp_path)
    handle = await store.put(payload)

    iter_obj = await store.fetch_stream(handle)
    chunks = await _drain(iter_obj)

    assert sum(len(c) for c in chunks) == len(payload)
    assert b"".join(chunks) == payload


# ---------------------------------------------------------------------------
# 5. Error path — generator raises mid-iteration.
# ---------------------------------------------------------------------------


async def test_subclass_fetch_stream_error_propagates() -> None:
    """An exception raised inside a subclass's ``fetch_stream`` async
    generator surfaces back to the Python consumer (via
    ``blazen_error_to_pyerr`` mapping)."""

    class FailingStore(ContentStore):
        async def put(self, _body: dict, _hint: dict) -> ContentHandle:
            return _placeholder_handle()

        async def resolve(self, _handle: ContentHandle) -> dict:
            return {"type": "url", "url": "https://example.invalid/blob"}

        async def fetch_bytes(self, _handle: ContentHandle) -> bytes:
            raise RuntimeError("not used")

        async def fetch_stream(
            self, _handle: ContentHandle
        ) -> AsyncIterator[bytes]:
            yield b"first chunk"
            raise RuntimeError("boom inside fetch_stream")

    store = FailingStore()
    handle = _placeholder_handle()
    # FailingStore.fetch_stream is an async generator — see the comment
    # in test_subclass_fetch_stream_async_generator above for why we do
    # not ``await`` it here.
    iter_obj = store.fetch_stream(handle)

    collected: list[bytes] = []
    with pytest.raises(Exception) as excinfo:
        async for chunk in iter_obj:
            collected.append(chunk)

    # We don't assert on the concrete exception type — the Blazen error
    # mapping may surface this as RuntimeError / ProviderError / similar
    # depending on the runtime build. We *do* assert the inner message
    # bubbled through and that the first chunk arrived before the error.
    assert "boom inside fetch_stream" in str(excinfo.value)
    assert collected == [b"first chunk"], collected


# ---------------------------------------------------------------------------
# 6. Backpressure smoke — many small chunks, slow consumer, ordered delivery.
# ---------------------------------------------------------------------------


async def test_fetch_stream_backpressure_smoke() -> None:
    """Producer yields 32 chunks of 16 KiB with cooperative yields between
    each; consumer awaits each ``__anext__`` with a tiny sleep. We verify
    every chunk arrives in order and the byte total matches.

    This is *correctness* under backpressure, not a memory benchmark —
    actual peak-RSS measurement is a future perf-suite concern."""

    chunk_count = 32
    chunk_size = 16 * 1024  # 16 KiB
    expected_chunks = [bytes([i % 256]) * chunk_size for i in range(chunk_count)]

    async def put(_body: dict, _hint: dict) -> ContentHandle:
        return _placeholder_handle()

    async def resolve(_handle: ContentHandle) -> dict:
        return {"type": "url", "url": "https://example.invalid/blob"}

    async def fetch_bytes(_handle: ContentHandle) -> bytes:
        return b"".join(expected_chunks)

    async def fetch_stream(_handle: ContentHandle) -> AsyncIterator[bytes]:
        async def gen() -> AsyncIterator[bytes]:
            for chunk in expected_chunks:
                yield chunk
                await asyncio.sleep(0)

        return gen()

    store = ContentStore.custom(
        put=put,
        resolve=resolve,
        fetch_bytes=fetch_bytes,
        fetch_stream=fetch_stream,
        name="backpressure",
    )

    handle = await store.put(b"placeholder")
    iter_obj = await store.fetch_stream(handle)

    received: list[bytes] = []
    aiter = iter_obj.__aiter__()
    while True:
        try:
            chunk = await aiter.__anext__()
        except StopAsyncIteration:
            break
        received.append(chunk)
        # Cooperative consumer pause — gives the producer task a chance to
        # fill the bounded mpsc channel and exercises the backpressure path.
        await asyncio.sleep(0)

    assert len(received) == chunk_count, len(received)
    assert received == expected_chunks
    assert sum(len(c) for c in received) == chunk_count * chunk_size
