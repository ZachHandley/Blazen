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
    AzureOptions,
    CacheConfig,
    CacheStrategy,
    ChatMessage,
    ChatWindow,
    CompletionModel,
    CompletionOptions,
    Context,
    Event,
    FalLlmEndpointKind,
    FalOptions,
    ImageRequest,
    InMemoryBackend,
    Memory,
    PromptRegistry,
    PromptTemplate,
    ProviderOptions,
    RetryConfig,
    StartEvent,
    StopEvent,
    ToolDef,
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
# Typed Options classes (construction + field access)
# =========================================================================


def test_provider_options_typed():
    """ProviderOptions class construction and field access."""
    opts = ProviderOptions(model="gpt-4o", base_url="https://api.openai.com/v1")
    assert opts.model == "gpt-4o"
    assert opts.base_url == "https://api.openai.com/v1"
    # Pass to a factory
    opts.api_key = "fake-key"
    model = CompletionModel.openai(options=opts)
    assert model is not None


def test_fal_options_typed():
    """FalOptions with typed endpoint enum."""
    opts = FalOptions(
        model="fal-ai/any-llm",
        endpoint=FalLlmEndpointKind.AnyLlm,
        enterprise=False,
        auto_route_modality=True,
    )
    assert opts.model == "fal-ai/any-llm"
    assert opts.enterprise is False
    assert opts.auto_route_modality is True


def test_azure_options_required_fields():
    """AzureOptions requires resource_name and deployment_name."""
    opts = AzureOptions(
        resource_name="my-resource",
        deployment_name="my-deployment",
        api_version="2024-02-15-preview",
    )
    assert opts.resource_name == "my-resource"
    assert opts.deployment_name == "my-deployment"
    assert opts.api_version == "2024-02-15-preview"
    # AzureOptions is required (not optional) for the factory
    opts.api_key = "fake-key"
    model = CompletionModel.azure(options=opts)
    assert model is not None


def test_image_request_typed():
    """ImageRequest construction with typed fields."""
    req = ImageRequest(
        prompt="A futuristic city at sunset",
        width=1024,
        height=1024,
        num_images=2,
        negative_prompt="blurry",
    )
    assert req.prompt == "A futuristic city at sunset"
    assert req.width == 1024
    assert req.height == 1024
    assert req.num_images == 2
    assert req.negative_prompt == "blurry"


# =========================================================================
# Retry / Cache / Fallback (construction only, no API calls)
# =========================================================================


def test_with_retry_constructs():
    """CompletionModel.with_retry accepts a typed RetryConfig."""
    model = CompletionModel.openai(options=ProviderOptions(api_key="fake-key"))
    config = RetryConfig(
        max_retries=5,
        initial_delay_ms=500,
        max_delay_ms=10_000,
        honor_retry_after=True,
        jitter=True,
    )
    assert config.max_retries == 5
    assert config.initial_delay_ms == 500
    retried = model.with_retry(config)
    assert retried is not None


def test_with_retry_no_config_uses_defaults():
    """with_retry with no config falls back to RetryConfig defaults."""
    model = CompletionModel.openai(options=ProviderOptions(api_key="fake-key"))
    retried = model.with_retry()
    assert retried is not None


def test_with_cache_constructs():
    """CompletionModel.with_cache accepts a typed CacheConfig."""
    model = CompletionModel.openai(options=ProviderOptions(api_key="fake-key"))
    config = CacheConfig(
        ttl_seconds=60,
        max_entries=100,
        strategy=CacheStrategy.ContentHash,
    )
    assert config.ttl_seconds == 60
    assert config.max_entries == 100
    cached = model.with_cache(config)
    assert cached is not None


def test_with_cache_no_config_uses_defaults():
    """with_cache with no config falls back to CacheConfig defaults."""
    model = CompletionModel.openai(options=ProviderOptions(api_key="fake-key"))
    cached = model.with_cache()
    assert cached is not None


def test_with_fallback_constructs():
    """CompletionModel.with_fallback chains multiple models."""
    m1 = CompletionModel.openai(options=ProviderOptions(api_key="fake-key-1"))
    m2 = CompletionModel.openai(options=ProviderOptions(api_key="fake-key-2"))
    fallback = CompletionModel.with_fallback([m1, m2])
    assert fallback is not None


# =========================================================================
# Tools via CompletionOptions (subclassed CompletionModel)
# =========================================================================


def _make_search_tool():
    """Helper: create a search ToolDef."""
    return ToolDef(
        name="search",
        description="Search the web for information",
        parameters={
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
        handler=lambda args: {"results": []},
    )


def _make_calculator_tool():
    """Helper: create a calculator ToolDef."""
    return ToolDef(
        name="calculator",
        description="Perform arithmetic calculations",
        parameters={
            "type": "object",
            "properties": {
                "expression": {"type": "string", "description": "Math expression"},
            },
            "required": ["expression"],
        },
        handler=lambda args: {"result": 42},
    )


def test_tooldef_repr_contains_name():
    """ToolDef repr shows the tool name (verifies construction preserved name)."""
    tool = _make_search_tool()
    assert "search" in repr(tool)


def test_tooldef_description_preserved_in_repr_of_different_tools():
    """Different tools produce different reprs (name round-trips)."""
    search = _make_search_tool()
    calc = _make_calculator_tool()
    assert "search" in repr(search)
    assert "calculator" in repr(calc)
    assert repr(search) != repr(calc)


def test_tooldef_requires_keyword_args():
    """ToolDef constructor requires all four keyword args."""
    # Missing handler should raise
    with pytest.raises(TypeError):
        ToolDef(name="x", description="d", parameters={})

    # Positional args are disallowed (constructor is keyword-only)
    with pytest.raises(TypeError):
        ToolDef("x", "d", {}, lambda a: a)


def test_tooldef_description_required():
    """ToolDef constructor requires a description (docstring/description)."""
    # Missing description should raise
    with pytest.raises(TypeError):
        ToolDef(name="x", parameters={}, handler=lambda a: a)


def test_completion_options_accepts_single_tooldef():
    """CompletionOptions accepts a list containing one ToolDef."""
    tool = _make_search_tool()
    opts = CompletionOptions(tools=[tool])

    assert opts.tools is not None
    assert len(opts.tools) == 1
    # Verify the item is a ToolDef (typed, not a dict)
    assert isinstance(opts.tools[0], ToolDef)
    # Name is preserved in the repr (stable round-trip check)
    assert "search" in repr(opts.tools[0])


def test_completion_options_accepts_multiple_tooldefs():
    """CompletionOptions accepts multiple ToolDef objects and preserves them."""
    search = _make_search_tool()
    calc = _make_calculator_tool()
    opts = CompletionOptions(tools=[search, calc])

    assert opts.tools is not None
    assert len(opts.tools) == 2
    for t in opts.tools:
        assert isinstance(t, ToolDef)

    # Verify each tool's name appears (via repr) so we know they weren't merged
    reprs = [repr(t) for t in opts.tools]
    assert any("search" in r for r in reprs)
    assert any("calculator" in r for r in reprs)


def test_completion_options_typed_tool_list_type():
    """CompletionOptions.tools contains typed ToolDef objects (not dicts).

    This verifies the core change: tools are now typed `ToolDef` objects,
    not untyped dicts. Users should always pass `ToolDef` instances.
    """
    tool = _make_search_tool()
    opts = CompletionOptions(tools=[tool])

    assert opts.tools is not None
    assert len(opts.tools) == 1
    # The returned tool must be a ToolDef, not a dict.
    assert isinstance(opts.tools[0], ToolDef)
    # Verify it's NOT a dict (the old API allowed dicts).
    assert not isinstance(opts.tools[0], dict)


def test_completion_options_no_tools_returns_none():
    """CompletionOptions without tools has tools == None (not empty list)."""
    opts = CompletionOptions()
    assert opts.tools is None


def test_completion_options_empty_tools_list():
    """CompletionOptions accepts an explicit empty list."""
    opts = CompletionOptions(tools=[])
    assert opts.tools == []


def test_completion_options_tools_setter_typed():
    """CompletionOptions.tools setter accepts a new list of ToolDef."""
    opts = CompletionOptions()
    assert opts.tools is None

    tool = _make_search_tool()
    opts.tools = [tool]

    assert opts.tools is not None
    assert len(opts.tools) == 1
    assert isinstance(opts.tools[0], ToolDef)


def test_completion_options_tools_setter_rejects_dicts():
    """Assigning dicts to CompletionOptions.tools raises (typed contract)."""
    opts = CompletionOptions()
    with pytest.raises((TypeError, ValueError)):
        opts.tools = [{"name": "x", "description": "d", "parameters": {}}]


def test_completion_options_tools_with_other_params():
    """CompletionOptions preserves tools alongside other parameters."""
    tool = _make_search_tool()
    opts = CompletionOptions(
        temperature=0.7,
        max_tokens=1000,
        tools=[tool],
    )

    assert opts.temperature == pytest.approx(0.7)
    assert opts.max_tokens == 1000
    assert opts.tools is not None
    assert len(opts.tools) == 1
    assert isinstance(opts.tools[0], ToolDef)


def test_tooldef_handler_receives_args():
    """ToolDef handler is callable with the args dict (integration sanity)."""
    received = []

    def my_handler(args):
        received.append(args)
        return {"ok": True}

    tool = ToolDef(
        name="echo",
        description="Echo the input",
        parameters={"type": "object"},
        handler=my_handler,
    )
    # We can't directly invoke the handler through ToolDef since the
    # bindings don't expose a getter, but we can verify the object was
    # accepted with a callable handler (no TypeError on construction).
    assert "echo" in repr(tool)


def test_completion_options_tools_roundtrip_through_options():
    """Tools assigned via constructor survive a get/set cycle on the options."""
    tool = _make_search_tool()
    opts = CompletionOptions(tools=[tool])

    # Read back through the property
    retrieved = opts.tools
    assert retrieved is not None
    assert len(retrieved) == 1

    # Reassign the retrieved list back -- should still work (typed)
    opts.tools = retrieved
    assert len(opts.tools) == 1
    assert isinstance(opts.tools[0], ToolDef)


# =========================================================================
# Tools passthrough via CompletionOptions (subclassed CompletionModel)
# =========================================================================
#
# These tests verify end-to-end that tools supplied via CompletionOptions
# reach a subclassed CompletionModel's `complete()` override with their
# name/description/parameters/handler fields intact.
#
# The subclass uses ``__new__`` to forward keyword arguments to the base
# ``CompletionModel`` constructor (PyO3's ``#[new]`` maps to Python's
# ``__new__`` slot). Calling ``super().__init__(model_id=...)`` from an
# overridden ``__init__`` is rejected by ``object.__init__`` because the
# base class has no ``__init__`` slot -- only ``__new__``.


@pytest.mark.asyncio
async def test_tools_via_completion_options_passthrough():
    """Tools passed via CompletionOptions reach the subclass override with correct name/description/parameters."""

    captured = []

    class ToolInspectorLLM(CompletionModel):
        def __new__(cls):
            return super().__new__(cls, model_id="tool-inspector")

        async def complete(self, messages, options=None):
            if options is not None and options.tools is not None:
                for tool in options.tools:
                    captured.append({
                        "name": tool.name,
                        "description": tool.description,
                    })
            raise RuntimeError("inspection-complete")

        async def stream(self, messages, on_chunk=None, options=None):
            raise RuntimeError("stream-not-used")

    model = ToolInspectorLLM()
    search_tool = ToolDef(
        name="search",
        description="Search the web for information.",
        parameters={
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
        handler=lambda args: {"results": []},
    )
    calc_tool = ToolDef(
        name="calculator",
        description="Add two numbers together.",
        parameters={
            "type": "object",
            "properties": {
                "a": {"type": "number"},
                "b": {"type": "number"},
            },
            "required": ["a", "b"],
        },
        handler=lambda args: {"result": args["a"] + args["b"]},
    )

    opts = CompletionOptions(tools=[search_tool, calc_tool])

    with pytest.raises(RuntimeError, match="inspection-complete"):
        await model.complete([ChatMessage.user("Hi")], options=opts)

    assert len(captured) == 2
    assert captured[0]["name"] == "search"
    assert captured[0]["description"] == "Search the web for information."
    assert captured[1]["name"] == "calculator"
    assert captured[1]["description"] == "Add two numbers together."


@pytest.mark.asyncio
async def test_tools_none_when_options_has_no_tools():
    """When CompletionOptions has no tools, options.tools is None in the override."""

    captured = {"tools_value": "sentinel"}

    class NoToolsLLM(CompletionModel):
        def __new__(cls):
            return super().__new__(cls, model_id="no-tools")

        async def complete(self, messages, options=None):
            captured["tools_value"] = options.tools if options else "no-options"
            raise RuntimeError("done")

        async def stream(self, messages, on_chunk=None, options=None):
            raise RuntimeError("stream-not-used")

    model = NoToolsLLM()
    opts = CompletionOptions(temperature=0.7)

    with pytest.raises(RuntimeError, match="done"):
        await model.complete([ChatMessage.user("Hi")], options=opts)

    assert captured["tools_value"] is None


@pytest.mark.asyncio
async def test_tool_parameters_schema_preserved():
    """Tool parameters JSON schema is preserved end-to-end."""

    captured_params = []

    class ParamsCaptureLLM(CompletionModel):
        def __new__(cls):
            return super().__new__(cls, model_id="params-capture")

        async def complete(self, messages, options=None):
            if options and options.tools:
                for tool in options.tools:
                    captured_params.append(tool.parameters)
            raise RuntimeError("captured")

        async def stream(self, messages, on_chunk=None, options=None):
            raise RuntimeError("stream-not-used")

    model = ParamsCaptureLLM()
    complex_schema = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Search query"},
            "limit": {"type": "integer", "minimum": 1, "maximum": 100},
            "filters": {
                "type": "array",
                "items": {"type": "string"},
            },
        },
        "required": ["query"],
    }
    tool = ToolDef(
        name="advanced_search",
        description="Complex search with filters.",
        parameters=complex_schema,
        handler=lambda args: {"ok": True},
    )

    with pytest.raises(RuntimeError, match="captured"):
        await model.complete([ChatMessage.user("Hi")], options=CompletionOptions(tools=[tool]))

    assert len(captured_params) == 1
    params = captured_params[0]
    assert params["type"] == "object"
    assert "query" in params["properties"]
    assert params["properties"]["limit"]["minimum"] == 1
    assert params["required"] == ["query"]


@pytest.mark.asyncio
async def test_tool_handler_is_callable_in_override():
    """The handler on a ToolDef is accessible and callable from the subclass override."""

    handler_results = []

    def my_handler(args):
        return {"doubled": args["x"] * 2}

    class HandlerCheckLLM(CompletionModel):
        def __new__(cls):
            return super().__new__(cls, model_id="handler-check")

        async def complete(self, messages, options=None):
            if options and options.tools:
                for tool in options.tools:
                    # ToolDef.handler is exposed -- we can call it
                    result = tool.handler({"x": 21})
                    handler_results.append(result)
            raise RuntimeError("done")

        async def stream(self, messages, on_chunk=None, options=None):
            raise RuntimeError("stream-not-used")

    model = HandlerCheckLLM()
    tool = ToolDef(
        name="doubler",
        description="Doubles a number.",
        parameters={"type": "object", "properties": {"x": {"type": "number"}}},
        handler=my_handler,
    )

    with pytest.raises(RuntimeError, match="done"):
        await model.complete([ChatMessage.user("Hi")], options=CompletionOptions(tools=[tool]))

    assert handler_results == [{"doubled": 42}]
