"""Capability functionality tests against a LOCAL mock HTTP server.

These are the per-PR "functionality" tier mirrors of the key-gated capability
smoke tests in ``test_capabilities_smoke.py``. They require NO API keys and
make NO external network calls -- every request lands on an in-process mock
bound to ``127.0.0.1`` on an OS-assigned port (see ``_mock_llm_server.py``).
They are deterministic and safe under ``pytest -n auto``.

The mocked provider is ``Model.openai(...)`` pointed at the mock via
``ProviderOptions(base_url=...)``. The OpenAI provider honours ``base_url``
(verified in ``crates/blazen-llm/src/providers/openai.rs`` /
``mod.rs::impl_simple_from_options`` -- the variant *without* ``no_base_url``),
so the request genuinely reaches the mock.
"""

import json

import pytest

from blazen import AgentResult, ChatMessage, Model, ModelOptions, ProviderOptions, ToolDef, run_agent


def _mock_model(server, model: str = "gpt-4o-mock") -> Model:
    return Model.openai(
        options=ProviderOptions(
            api_key="mock-key",
            base_url=server.base_url,
            model=model,
        )
    )


# ---------------------------------------------------------------------------
# Streaming (callback mode)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_streaming_completion_mocked(mock_llm_server):
    """Callback-mode streaming reconstructs exactly what the mock streamed."""
    mock_llm_server.controller.stream = True
    mock_llm_server.controller.stream_pieces = ["Hel", "lo", ", ", "world"]
    model = _mock_model(mock_llm_server)

    # The callback receives each chunk pythonized to a dict (the binding's
    # callback mode converts StreamChunk via ``pythonize``), so the incremental
    # text lives under the ``"delta"`` key.
    chunks: list[dict] = []

    def on_chunk(chunk: dict) -> None:
        chunks.append(chunk)

    await model.stream(
        [ChatMessage.user("Say hello.")],
        on_chunk,
        ModelOptions(max_tokens=64),
    )

    assert len(chunks) >= 1, "Expected at least one streamed chunk"

    reconstructed = "".join(c.get("delta") or "" for c in chunks)
    assert reconstructed == "Hello, world"

    # The mock recorded a streaming request carrying our model + prompt.
    body = mock_llm_server.controller.last_body
    assert body is not None
    assert body["model"] == "gpt-4o-mock"
    assert body["stream"] is True
    assert any(m.get("role") == "user" for m in body["messages"])


# ---------------------------------------------------------------------------
# Structured output (JSON schema)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_structured_output_mocked(mock_llm_server):
    """Structured-output content round-trips as a JSON string the caller parses."""
    mock_llm_server.controller.content = json.dumps({"answer": 4})
    model = _mock_model(mock_llm_server)

    response = await model.complete(
        [ChatMessage.user("What is 2+2? Return JSON.")],
        ModelOptions(
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "math",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {"answer": {"type": "integer"}},
                        "required": ["answer"],
                    },
                },
            },
            max_tokens=64,
        ),
    )

    assert response.content is not None
    parsed = json.loads(response.content)
    assert parsed["answer"] == 4

    # The provider forwarded the json_schema response_format to the wire.
    body = mock_llm_server.controller.last_body
    assert body is not None
    assert body["response_format"]["type"] == "json_schema"
    assert body["response_format"]["json_schema"]["name"] == "math"


# ---------------------------------------------------------------------------
# Agent tool calling (sync handler)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_agent_tool_calling_mocked(mock_llm_server):
    """Agent loop with a SYNC tool handler runs the stateful tool round-trip."""
    mock_llm_server.controller.tool_loop = True
    model = _mock_model(mock_llm_server)

    invoked: list[dict] = []

    def multiply(arguments: dict) -> dict:
        invoked.append(arguments)
        return {"result": arguments["a"] * arguments["b"]}

    tool = ToolDef(
        name="multiply",
        description="Multiply two numbers together.",
        parameters={
            "type": "object",
            "properties": {"a": {"type": "number"}, "b": {"type": "number"}},
            "required": ["a", "b"],
        },
        handler=multiply,
    )

    result = await run_agent(
        model,
        [ChatMessage.user("What is 15 * 7? Use the multiply tool.")],
        tools=[tool],
        max_iterations=5,
    )

    assert isinstance(result, AgentResult)
    assert result.iterations >= 1
    assert invoked == [{"a": 15, "b": 7}], "the sync tool handler was actually invoked"
    assert result.response.content is not None
    assert "105" in result.response.content
    # The mock observed the tool-result message fed back by the agent loop.
    assert mock_llm_server.controller.saw_tool_result is True


# ---------------------------------------------------------------------------
# Agent tool calling (async handler)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_agent_async_tool_mocked(mock_llm_server):
    """Agent loop with an ASYNC tool handler runs the stateful tool round-trip."""
    mock_llm_server.controller.tool_loop = True
    model = _mock_model(mock_llm_server)

    invoked: list[dict] = []

    async def multiply(arguments: dict) -> dict:
        invoked.append(arguments)
        return {"result": arguments["a"] * arguments["b"]}

    tool = ToolDef(
        name="multiply",
        description="Multiply two numbers together.",
        parameters={
            "type": "object",
            "properties": {"a": {"type": "number"}, "b": {"type": "number"}},
            "required": ["a", "b"],
        },
        handler=multiply,
    )

    result = await run_agent(
        model,
        [ChatMessage.user("What is 15 * 7? Use the multiply tool.")],
        tools=[tool],
        max_iterations=5,
    )

    assert isinstance(result, AgentResult)
    assert result.iterations >= 1
    assert invoked == [{"a": 15, "b": 7}], "the async tool handler was actually invoked"
    assert result.response.content is not None
    assert "105" in result.response.content
    assert mock_llm_server.controller.saw_tool_result is True
