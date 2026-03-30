"""Capability smoke tests: streaming, structured output, agent tool calling.

Gated on the OPENROUTER_API_KEY environment variable.
"""

import json
import os

import pytest

from blazen import AgentResult, ChatMessage, CompletionModel, CompletionOptions, ToolDef, run_agent

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")

skip_without_key = pytest.mark.skipif(
    not OPENROUTER_API_KEY,
    reason="OPENROUTER_API_KEY not set",
)


# ---------------------------------------------------------------------------
# Streaming
# ---------------------------------------------------------------------------


@skip_without_key
@pytest.mark.asyncio
async def test_streaming_completion():
    """Stream a completion and verify at least one chunk is received."""
    model = CompletionModel.openrouter(OPENROUTER_API_KEY)

    chunks: list[dict] = []

    def on_chunk(chunk: dict) -> None:
        chunks.append(chunk)

    await model.stream(
        [ChatMessage.user("Count from 1 to 5.")],
        on_chunk,
        CompletionOptions(max_tokens=64),
    )

    assert len(chunks) > 0, "Expected at least one streamed chunk"


# ---------------------------------------------------------------------------
# Structured output (JSON schema)
# ---------------------------------------------------------------------------


@skip_without_key
@pytest.mark.asyncio
async def test_structured_output():
    """Request structured JSON output and verify the schema is respected."""
    model = CompletionModel.openrouter(OPENROUTER_API_KEY)

    response = await model.complete(
        [ChatMessage.user("What is 2+2? Return JSON.")],
        CompletionOptions(
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "math",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {
                            "answer": {"type": "integer"},
                        },
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


# ---------------------------------------------------------------------------
# Agent tool calling (sync handler)
# ---------------------------------------------------------------------------


@skip_without_key
@pytest.mark.asyncio
async def test_agent_tool_calling():
    """Run the agent loop with a sync tool handler."""
    model = CompletionModel.openrouter(OPENROUTER_API_KEY)

    def multiply(arguments: dict) -> dict:
        return {"result": arguments["a"] * arguments["b"]}

    tool = ToolDef(
        name="multiply",
        description="Multiply two numbers together.",
        parameters={
            "type": "object",
            "properties": {
                "a": {"type": "number"},
                "b": {"type": "number"},
            },
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
    assert result.response.content is not None
    assert "105" in result.response.content
    assert result.iterations >= 1


# ---------------------------------------------------------------------------
# Agent tool calling (async handler)
# ---------------------------------------------------------------------------


@skip_without_key
@pytest.mark.asyncio
async def test_agent_async_tool():
    """Run the agent loop with an async tool handler."""
    model = CompletionModel.openrouter(OPENROUTER_API_KEY)

    async def multiply(arguments: dict) -> dict:
        return {"result": arguments["a"] * arguments["b"]}

    tool = ToolDef(
        name="multiply",
        description="Multiply two numbers together.",
        parameters={
            "type": "object",
            "properties": {
                "a": {"type": "number"},
                "b": {"type": "number"},
            },
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
    assert result.response.content is not None
    assert "105" in result.response.content
    assert result.iterations >= 1
