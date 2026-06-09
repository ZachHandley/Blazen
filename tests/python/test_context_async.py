"""Async-surface + GIL-deadlock regression tests for `Context` bindings.

These tests cover two related concerns introduced together:

1. The `#[py_async]` proc-macro in `blazen-macros` emits a sync method
   and an `a<name>` async sibling from the single Rust `async fn`
   source. This file exercises every async sibling reachable from
   `Context`, its `state` / `session` namespaces, `SessionRefRegistry`,
   and the provider factory constructors.

2. `block_on_context` now releases the GIL via `Python::detach` for
   the duration of the blocking wait. The deadlock test below shoves a
   non-pickleable object whose `__del__` re-acquires the GIL through
   the registry's `insert_arc` path while another Python thread is
   pounding on `ctx.set` — the pre-fix behaviour was a multi-thread
   GIL-vs-tokio-worker deadlock that hung forever; the post-fix
   behaviour returns under the test timeout.
"""

from __future__ import annotations

import asyncio
import threading

import pytest

import blazen
from blazen import Context, Event, StartEvent, StopEvent, Workflow, step


# =========================================================================
# Async siblings on Context
# =========================================================================


@pytest.mark.asyncio
async def test_acontext_set_bytes_get_bytes_round_trip():
    captured: dict = {}

    @step
    async def s(ctx: Context, ev: Event):
        await ctx.aset_bytes("blob", b"\x00\x01\x02\x03")
        captured["round"] = await ctx.aget_bytes("blob")
        captured["miss"] = await ctx.aget_bytes("not-set", b"default")
        captured["rid"] = await ctx.arun_id()
        return StopEvent(result={})

    wf = Workflow("acontext-roundtrip", [s])
    await wf.run()

    assert captured["round"] == b"\x00\x01\x02\x03"
    assert captured["miss"] == b"default"
    assert isinstance(captured["rid"], str)
    assert captured["rid"].count("-") == 4


@pytest.mark.asyncio
async def test_acontext_set_aget_routes_through_tier_dispatch():
    captured: dict = {}

    @step
    async def s(ctx: Context, ev: Event):
        # Tier 2: JSON.
        await ctx.aset("score", 0.75)
        captured["score"] = await ctx.aget("score")
        # Tier 1: bytes (auto-routed through the same async surface).
        await ctx.aset("raw", b"\xde\xad")
        captured["raw"] = await ctx.aget("raw")
        # Missing key with default.
        captured["miss"] = await ctx.aget("nope", "fallback")
        return StopEvent(result={})

    wf = Workflow("acontext-tier", [s])
    await wf.run()

    assert captured["score"] == 0.75
    assert captured["raw"] == b"\xde\xad"
    assert captured["miss"] == "fallback"


@pytest.mark.asyncio
async def test_acontext_send_event_routes_to_next_step():
    seen: list[str] = []

    @step
    async def producer(ctx: Context, ev: Event):
        await ctx.asend_event(Event("Relay", payload="hello"))
        return None

    @step(accepts=["Relay"])
    async def consumer(ctx: Context, ev: Event):
        seen.append(ev.payload)
        return StopEvent(result={"payload": ev.payload})

    wf = Workflow("acontext-relay", [producer, consumer])
    result = await wf.run()
    assert result.result["payload"] == "hello"
    assert seen == ["hello"]


# =========================================================================
# Async siblings on the StateNamespace + SessionNamespace
# =========================================================================


@pytest.mark.asyncio
async def test_astate_namespace_round_trip():
    captured: dict = {}

    @step
    async def s(ctx: Context, ev: Event):
        await ctx.state.aset("counter", 7)
        captured["counter"] = await ctx.state.aget("counter")
        await ctx.state.aset_bytes("buf", b"abc")
        captured["buf"] = await ctx.state.aget_bytes("buf")
        return StopEvent(result={})

    wf = Workflow("astate-roundtrip", [s])
    await wf.run()

    assert captured["counter"] == 7
    assert captured["buf"] == b"abc"


@pytest.mark.asyncio
async def test_asession_namespace_identity_preserved():
    captured: dict = {}

    class LiveHandle:
        def __init__(self) -> None:
            self.tag = "live"

    @step
    async def s(ctx: Context, ev: Event):
        h = LiveHandle()
        await ctx.session.aset("conn", h)
        round_tripped = await ctx.session.aget("conn")
        captured["same_object"] = round_tripped is h
        captured["has"] = await ctx.session.ahas("conn")
        await ctx.session.aremove("conn")
        captured["has_after_remove"] = await ctx.session.ahas("conn")
        return StopEvent(result={})

    wf = Workflow("asession-id", [s])
    await wf.run()

    assert captured["same_object"] is True
    assert captured["has"] is True
    assert captured["has_after_remove"] is False


# =========================================================================
# GIL-deadlock regression: ctx.set under contention with __del__ side effects
# =========================================================================


@pytest.mark.asyncio
async def test_ctx_set_does_not_deadlock_under_thread_contention():
    """Synthesise the original deadlock: spawn a second Python thread that
    pounds on `ctx.set` while the main step also runs `ctx.set` with an
    object whose `__del__` re-acquires the GIL. Without the GIL-release
    fix in `block_on_context`, the registry's `insert_arc` future runs
    on a tokio worker that needs the GIL to drop the previous
    `Arc<Py<PyAny>>`, the worker blocks on the GIL the calling Python
    thread is holding, and both deadlock. With the fix this test
    returns; without it the test hits the asyncio timeout.
    """

    class HookedDelete:
        """Trips `__del__` (which re-enters Python) every time a stored
        instance is dropped — exactly the path the tokio worker has to
        take when the registry evicts an old entry."""

        def __del__(self) -> None:
            # Touch a Python builtin to force a real GIL acquire on drop.
            _ = list((1, 2, 3))

    stop_flag = threading.Event()

    def thread_body(ctx: Context) -> None:
        i = 0
        while not stop_flag.is_set() and i < 200:
            ctx.set(f"k{i % 4}", HookedDelete())
            i += 1

    @step
    async def s(ctx: Context, ev: Event):
        # Kick off a background thread before doing our own loop. Both
        # threads write to the same handful of keys, so each `set` evicts
        # the previous live-ref via a `Py<PyAny>` drop on a tokio worker.
        t = threading.Thread(target=thread_body, args=(ctx,))
        t.start()
        try:
            for i in range(200):
                ctx.set(f"k{i % 4}", HookedDelete())
        finally:
            stop_flag.set()
            t.join(timeout=5.0)
            assert not t.is_alive(), "background thread hung — deadlock regressed"
        return StopEvent(result={})

    wf = Workflow("ctx-set-no-deadlock", [s])
    # 10s is overkill if the fix is in place; pre-fix the test hangs
    # forever, so we let pytest-asyncio's per-test timeout catch it.
    await asyncio.wait_for(wf.run(), timeout=15.0)


# =========================================================================
# Provider factory async sibling — only validate compile-time symbol
# presence; loading the actual model is too heavy for a unit test.
# =========================================================================


def test_provider_factories_expose_aopen():
    """`#[py_async_factory]` must emit an `aopen` classmethod next to
    the sync `__new__` constructor on every provider it's applied to."""
    if hasattr(blazen, "WhisperCppProvider"):
        assert hasattr(blazen.WhisperCppProvider, "aopen"), (
            "WhisperCppProvider.aopen missing — #[py_async_factory] regressed"
        )
    if hasattr(blazen, "LlamaCppProvider"):
        assert hasattr(blazen.LlamaCppProvider, "aopen"), (
            "LlamaCppProvider.aopen missing — #[py_async_factory] regressed"
        )
    if hasattr(blazen, "CandleEmbedModel"):
        assert hasattr(blazen.CandleEmbedModel, "aopen"), (
            "CandleEmbedModel.aopen missing — #[py_async_factory] regressed"
        )
    if hasattr(blazen, "Transcription"):
        assert hasattr(blazen.Transcription, "awhispercpp"), (
            "Transcription.awhispercpp missing — #[py_async_static] regressed"
        )
