"""E2E tests for Blazen Python bindings -- extended coverage.

These tests exercise features NOT covered by test_workflow.py or
test_session_refs.py: pause/resume/snapshot, human-in-the-loop,
prompt templates, memory (local mode), error propagation, abort,
token counting, chat window, and retry/cache/fallback decorators.

All tests run WITHOUT API keys -- no network calls.
"""

import asyncio
import json
import os

import pytest

from blazen import (
    ChatMessage,
    ChatWindow,
    CompletionModel,
    CompletionOptions,
    Context,
    Event,
    InMemoryBackend,
    Memory,
    PromptRegistry,
    PromptTemplate,
    StartEvent,
    StopEvent,
    Workflow,
    count_message_tokens,
    estimate_tokens,
    step,
)


# =========================================================================
# Pause / Resume / Snapshot
# =========================================================================


@pytest.mark.asyncio
async def test_pause_and_resume_in_place():
    """Pause a running workflow and resume it in place."""

    @step
    async def slow_step(ctx: Context, ev: Event):
        ctx.set("started", True)
        await asyncio.sleep(0.3)
        return StopEvent(result={"done": True})

    wf = Workflow("pause-resume", [slow_step])
    handler = await wf.run()

    # Let the step start and write context
    await asyncio.sleep(0.05)

    await handler.pause()
    await handler.resume_in_place()
    result = await handler.result()

    assert result.event_type == "blazen::StopEvent"
    assert result.result["done"] is True


@pytest.mark.asyncio
async def test_snapshot_captures_state():
    """Snapshot contains workflow name and context state."""

    @step
    async def setter(ctx: Context, ev: Event):
        ctx.set("key", "value")
        await asyncio.sleep(0.3)
        return StopEvent(result={"done": True})

    wf = Workflow("snapshot-test", [setter])
    handler = await wf.run()

    await asyncio.sleep(0.05)
    await handler.pause()

    snap_json = await handler.snapshot()
    snap = json.loads(snap_json)

    assert "workflow_name" in snap
    assert snap["workflow_name"] == "snapshot-test"
    assert "context_state" in snap

    # Clean up -- resume and finish
    await handler.resume_in_place()
    await handler.result()


@pytest.mark.asyncio
async def test_snapshot_json_is_valid_and_resumable():
    """Snapshot JSON roundtrips through Workflow.resume without errors.

    Note: the in-place snapshot cannot capture pending channel events
    (events already sent but not yet delivered to a step). The Rust
    integration tests work around this by manually injecting pending
    events. From the binding layer, we verify the snapshot format is
    valid JSON and that Workflow.resume accepts it -- the mid-flight
    resume semantics are covered by the Rust integration tests.
    """

    @step
    async def persistent_step(ctx: Context, ev: Event):
        ctx.set("persisted", "value")
        await asyncio.sleep(0.5)
        return StopEvent(result={"done": True})

    wf = Workflow("snapshot-roundtrip", [persistent_step])
    handler = await wf.run()

    await asyncio.sleep(0.1)
    await handler.pause()
    snap_json = await handler.snapshot()

    # Verify the snapshot is valid JSON with the expected shape.
    snap = json.loads(snap_json)
    assert snap["workflow_name"] == "snapshot-roundtrip"
    assert "context_state" in snap
    assert snap["context_state"].get("persisted") is not None

    # Resume call must accept the snapshot JSON without raising.
    # (We don't wait for completion because the mid-flight pause
    # doesn't capture the in-flight step's channel events.)
    handler2 = await Workflow.resume(snap_json, [persistent_step])
    assert handler2 is not None

    # Clean up: abort both so the test doesn't hang.
    await handler.abort()
    await handler2.abort()


# =========================================================================
# Human-in-the-Loop
# =========================================================================


@pytest.mark.asyncio
async def test_human_in_the_loop():
    """Step emits InputRequestEvent, external code responds."""

    @step
    async def ask_step(ctx: Context, ev: Event):
        return Event(
            "blazen::InputRequestEvent",
            request_id="req-1",
            prompt="What is your name?",
            metadata={},
        )

    @step(accepts=["blazen::InputResponseEvent"])
    async def process_response(ctx: Context, ev: Event):
        return StopEvent(result={"response": ev.response})

    wf = Workflow("hitl-test", [ask_step, process_response])
    handler = await wf.run()

    # Give time for the InputRequestEvent to be emitted and auto-pause
    await asyncio.sleep(0.2)

    await handler.respond_to_input("req-1", {"name": "Alice"})
    result = await handler.result()

    assert result.event_type == "blazen::StopEvent"
    assert result.result["response"]["name"] == "Alice"


# =========================================================================
# Prompt Templates
# =========================================================================


def test_prompt_template_render():
    """PromptTemplate renders variables into a ChatMessage."""
    t = PromptTemplate("Hello {{name}}, welcome to {{place}}!", role="user")
    msg = t.render(name="Alice", place="Wonderland")
    assert msg.content == "Hello Alice, welcome to Wonderland!"
    assert msg.role == "user"


def test_prompt_template_variables():
    """PromptTemplate.variables returns sorted variable names."""
    t = PromptTemplate("{{b}} and {{a}} and {{b}}")
    assert t.variables == ["a", "b"]


def test_prompt_template_properties():
    """PromptTemplate exposes name, role, version, description."""
    t = PromptTemplate(
        "Hello {{name}}!",
        role="system",
        name="greet",
        description="A greeting",
        version="2.0",
    )
    assert t.name == "greet"
    assert t.role == "system"
    assert t.version == "2.0"
    assert t.description == "A greeting"
    assert t.template == "Hello {{name}}!"


def test_prompt_registry_crud():
    """PromptRegistry register, get, list, render."""
    reg = PromptRegistry()
    # Note: template variable is intentionally NOT called "name" because
    # PromptRegistry.render's first positional arg is also "name" (the
    # template name), and the Python binding collects variables via kwargs.
    t = PromptTemplate("Hello {{person}}!", name="greet")
    reg.register("greet", t)

    assert reg.list() == ["greet"]

    got = reg.get("greet")
    assert got is not None
    assert got.template == "Hello {{person}}!"

    msg = reg.render("greet", person="Bob")
    assert msg.content == "Hello Bob!"


def test_prompt_registry_from_file(tmp_path):
    """Load a PromptRegistry from a YAML file."""
    yaml_content = """\
prompts:
  - name: summarize
    role: system
    template: "Summarize the {{doc_type}} in {{style}} style."
    version: "1.0"
"""
    yaml_file = tmp_path / "prompts.yaml"
    yaml_file.write_text(yaml_content)

    reg = PromptRegistry.from_file(str(yaml_file))
    assert "summarize" in reg.list()

    msg = reg.render("summarize", doc_type="article", style="concise")
    assert "Summarize the article in concise style." == msg.content


# =========================================================================
# Memory (local mode -- no embedding model, no API key)
# =========================================================================


@pytest.mark.asyncio
async def test_memory_local_crud():
    """Memory.local CRUD: add, get, count, delete."""
    mem = Memory.local(InMemoryBackend())

    doc_id = await mem.add("doc1", "Paris is the capital of France")
    assert doc_id == "doc1"
    assert await mem.count() == 1

    entry = await mem.get("doc1")
    assert entry is not None
    assert entry["text"] == "Paris is the capital of France"

    deleted = await mem.delete("doc1")
    assert deleted is True
    assert await mem.count() == 0


@pytest.mark.asyncio
async def test_memory_local_search():
    """Memory.local search_local returns results with id, text, score."""
    mem = Memory.local(InMemoryBackend())

    await mem.add("d1", "Paris is the capital of France")
    await mem.add("d2", "Berlin is the capital of Germany")
    await mem.add("d3", "Tokyo is the capital of Japan")

    results = await mem.search_local("capital of France", limit=2)

    assert len(results) > 0
    assert len(results) <= 2

    r = results[0]
    assert hasattr(r, "id")
    assert hasattr(r, "text")
    assert hasattr(r, "score")
    assert isinstance(r.score, float)


@pytest.mark.asyncio
async def test_memory_local_add_many():
    """Memory.local add_many batch inserts."""
    mem = Memory.local(InMemoryBackend())

    ids = await mem.add_many([
        {"id": "a", "text": "First document"},
        {"id": "b", "text": "Second document"},
        {"id": "c", "text": "Third document"},
    ])

    assert len(ids) == 3
    assert await mem.count() == 3


@pytest.mark.asyncio
async def test_memory_local_metadata_filter():
    """Memory.local search_local with metadata filter."""
    mem = Memory.local(InMemoryBackend())

    await mem.add("d1", "Paris is the capital of France", metadata={"region": "europe"})
    await mem.add("d2", "Tokyo is the capital of Japan", metadata={"region": "asia"})

    # Search with filter -- only europe
    results = await mem.search_local("capital", limit=5, metadata_filter={"region": "europe"})
    for r in results:
        # All results should be from europe
        assert r.metadata.get("region") == "europe"


# =========================================================================
# Error Propagation
# =========================================================================


@pytest.mark.asyncio
async def test_step_error_propagates():
    """A step that raises should cause the workflow to fail."""

    @step
    async def bad_step(ctx: Context, ev: Event):
        msg = "intentional test failure"
        raise RuntimeError(msg)

    wf = Workflow("error-test", [bad_step])
    handler = await wf.run()

    with pytest.raises(Exception, match="intentional test failure"):
        await handler.result()


@pytest.mark.asyncio
async def test_timeout_python():
    """Workflow with short timeout fails when step is too slow."""

    @step
    async def slow(ctx: Context, ev: Event):
        await asyncio.sleep(10)
        return StopEvent(result={})

    wf = Workflow("timeout-test", [slow], timeout=0.1)
    handler = await wf.run()

    with pytest.raises(Exception, match="(?i)timeout|timed out"):
        await handler.result()


# =========================================================================
# Handler Abort
# =========================================================================


@pytest.mark.asyncio
async def test_handler_abort():
    """Aborting a workflow causes result() to raise."""

    @step
    async def long_step(ctx: Context, ev: Event):
        await asyncio.sleep(5)
        return StopEvent(result={})

    wf = Workflow("abort-test", [long_step])
    handler = await wf.run()

    await asyncio.sleep(0.05)
    await handler.abort()

    with pytest.raises(Exception):
        await handler.result()


# =========================================================================
# Token Counting / ChatWindow
# =========================================================================


def test_estimate_tokens():
    """estimate_tokens returns a positive int, scales with text length."""
    n = estimate_tokens("Hello, world!")
    assert isinstance(n, int)
    assert n > 0

    n_long = estimate_tokens("Hello, world! " * 100)
    assert n_long > n


def test_count_message_tokens():
    """count_message_tokens on a message list returns positive int."""
    msgs = [
        ChatMessage.system("You are helpful."),
        ChatMessage.user("Hi!"),
    ]
    n = count_message_tokens(msgs)
    assert isinstance(n, int)
    assert n > 0

    # More messages -> more tokens
    msgs.append(ChatMessage.user("Tell me a story about a brave knight."))
    n2 = count_message_tokens(msgs)
    assert n2 > n


def test_chat_window_basic():
    """ChatWindow: add, messages, token_count, remaining_tokens, len."""
    window = ChatWindow(max_tokens=4096)

    window.add(ChatMessage.system("You are helpful."))
    assert len(window) == 1

    window.add(ChatMessage.user("Hello!"))
    assert len(window) == 2

    assert window.token_count() > 0
    assert window.remaining_tokens() < 4096

    msgs = window.messages()
    assert len(msgs) == 2
    assert msgs[0].role == "system"
    assert msgs[1].role == "user"


def test_chat_window_eviction():
    """ChatWindow evicts oldest non-system messages when over budget."""
    window = ChatWindow(max_tokens=30)

    window.add(ChatMessage.system("Be helpful."))
    window.add(ChatMessage.user("First message"))
    window.add(ChatMessage.user("Second message"))
    window.add(ChatMessage.user("Third message that pushes over budget"))

    # Token count should be within budget (or close, system is never evicted)
    assert window.token_count() <= 30 or len(window) <= 2

    # System message must always survive
    msgs = window.messages()
    roles = [m.role for m in msgs]
    assert "system" in roles


def test_chat_window_clear():
    """ChatWindow.clear removes all messages."""
    window = ChatWindow(max_tokens=4096)
    # A fresh window still has a small base token count (assistant priming
    # overhead from the estimator, currently 3). Capture it so we can verify
    # clear() returns to the baseline.
    baseline = window.token_count()

    window.add(ChatMessage.user("msg1"))
    window.add(ChatMessage.user("msg2"))
    assert len(window) == 2
    assert window.token_count() > baseline

    window.clear()
    assert len(window) == 0
    assert window.token_count() == baseline


# =========================================================================
# Retry / Cache / Fallback (construction only, no API calls)
# =========================================================================


def test_with_retry_constructs():
    """CompletionModel.with_retry returns a CompletionModel."""
    model = CompletionModel.openai("fake-key")
    retried = model.with_retry(max_retries=3, initial_delay_ms=100, max_delay_ms=5000)
    assert retried is not None


def test_with_cache_constructs():
    """CompletionModel.with_cache returns a CompletionModel."""
    model = CompletionModel.openai("fake-key")
    cached = model.with_cache(ttl_seconds=60, max_entries=100)
    assert cached is not None


def test_with_fallback_constructs():
    """CompletionModel.with_fallback chains multiple models."""
    m1 = CompletionModel.openai("fake-key-1")
    m2 = CompletionModel.openai("fake-key-2")
    fallback = CompletionModel.with_fallback([m1, m2])
    assert fallback is not None
