"""LLM smoke tests using OpenRouter.

These tests are gated on the OPENROUTER_API_KEY environment variable
and only run during release CI or manual invocation.
"""

import os

import pytest

from blazen import (
    ChatMessage,
    CompletionModel,
    Context,
    Event,
    StopEvent,
    Workflow,
    step,
)

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")

skip_without_key = pytest.mark.skipif(
    not OPENROUTER_API_KEY,
    reason="OPENROUTER_API_KEY not set",
)


@skip_without_key
@pytest.mark.asyncio
async def test_openrouter_completion():
    model = CompletionModel.openrouter(OPENROUTER_API_KEY)
    response = await model.complete(
        [ChatMessage.user("What is 2+2? Reply with just the number.")],
        max_tokens=10,
    )

    assert response["content"] is not None
    assert "4" in response["content"]
    assert response["model"] is not None


@skip_without_key
@pytest.mark.asyncio
async def test_openrouter_in_workflow():
    @step
    async def ask_llm(ctx: Context, ev: Event):
        model = CompletionModel.openrouter(OPENROUTER_API_KEY)
        response = await model.complete(
            [
                ChatMessage.system("You are a math tutor. Reply with just the number."),
                ChatMessage.user(ev.question),
            ],
            max_tokens=10,
        )
        return StopEvent(result={"answer": response["content"]})

    wf = Workflow("llm-smoke", [ask_llm])
    handler = await wf.run(question="What is 3+3?")
    result = await handler.result()

    assert result.result["answer"] is not None
    assert "6" in result.result["answer"]
