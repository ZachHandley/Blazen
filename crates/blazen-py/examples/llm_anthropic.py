"""Anthropic (Claude) LLM integration with a Blazen Q&A + fact-check workflow.

Demonstrates real Anthropic API calls through Blazen's CompletionModel
abstraction.  Two workflow steps form a question-answering pipeline:

  1. ask_question  -- sends the user's question to Claude and gets an answer
  2. fact_check    -- sends the answer back to Claude for verification

Key points:
  - CompletionModel.anthropic() creates an Anthropic-backed model.
  - model.complete() is async and returns a dict with content, model, usage,
    finish_reason, and tool_calls.
  - Anthropic requires max_tokens (Blazen defaults to 4096, but you can set
    it explicitly).
  - ctx.set() / ctx.get() are SYNCHRONOUS -- no await needed.

Run with: ANTHROPIC_API_KEY=sk-ant-... python llm_anthropic.py
"""

import asyncio
import os
import sys

from blazen import (
    ChatMessage,
    CompletionModel,
    Context,
    Event,
    ProviderOptions,
    StartEvent,
    StopEvent,
    Workflow,
    step,
)


# ---------------------------------------------------------------------------
# Create the Anthropic-backed model (module-level so both steps share it).
# Using claude-haiku-3-5 -- fast and cheap, ideal for examples.
# ---------------------------------------------------------------------------
def get_model() -> CompletionModel:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("ERROR: Set the ANTHROPIC_API_KEY environment variable.")
        print("  ANTHROPIC_API_KEY=sk-ant-... python llm_anthropic.py")
        sys.exit(1)
    return CompletionModel.anthropic(options=ProviderOptions(api_key=api_key, model="claude-haiku-4-5-20251001"))


MODEL = get_model()


# ---------------------------------------------------------------------------
# Step 1: Ask Claude a question and get an answer.
# ---------------------------------------------------------------------------
@step
async def ask_question(ctx: Context, ev: Event) -> Event:
    """Send the user's question to Claude with a concise-answer system prompt."""
    question: str = ev.question
    print(f"[ask_question] Question: {question}")

    response = await MODEL.complete(
        [
            ChatMessage.system(
                "You are a helpful assistant. Give concise, factual answers "
                "in 2-3 sentences maximum."
            ),
            ChatMessage.user(question),
        ],
        max_tokens=256,
    )

    answer: str = response["content"]
    usage: dict = response["usage"]

    # Store in context for the final summary.
    ctx.set("question", question)
    ctx.set("answer", answer)
    ctx.set("answer_model", response["model"])
    ctx.set("answer_usage", usage)

    print(f"[ask_question] Answer received ({usage['total_tokens']} tokens used)")

    return Event("AnswerEvent", answer=answer)


# ---------------------------------------------------------------------------
# Step 2: Fact-check the answer with a second Claude call.
# ---------------------------------------------------------------------------
@step(accepts=["AnswerEvent"])
async def fact_check(ctx: Context, ev: Event) -> StopEvent:
    """Ask Claude to verify the answer from the previous step."""
    answer: str = ev.answer
    question: str = ctx.get("question")

    print("[fact_check] Verifying answer...")

    response = await MODEL.complete(
        [
            ChatMessage.system(
                "You are a rigorous fact-checker. Given a question and a "
                "proposed answer, assess whether the answer is accurate. "
                "Reply with VERIFIED or DISPUTED followed by a brief "
                "explanation (1-2 sentences)."
            ),
            ChatMessage.user(
                f"Question: {question}\n\nProposed answer: {answer}"
            ),
        ],
        max_tokens=256,
    )

    verdict: str = response["content"]
    verdict_usage: dict = response["usage"]

    print(f"[fact_check] Verdict received ({verdict_usage['total_tokens']} tokens used)")

    # Combine usage from both calls.
    answer_usage: dict = ctx.get("answer_usage")
    total_usage = {
        "prompt_tokens": answer_usage["prompt_tokens"] + verdict_usage["prompt_tokens"],
        "completion_tokens": answer_usage["completion_tokens"] + verdict_usage["completion_tokens"],
        "total_tokens": answer_usage["total_tokens"] + verdict_usage["total_tokens"],
    }

    return StopEvent(result={
        "question": question,
        "answer": answer,
        "fact_check": verdict,
        "model": ctx.get("answer_model"),
        "total_usage": total_usage,
    })


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
async def main() -> None:
    wf = Workflow("qa-fact-check", [ask_question, fact_check])

    handler = await wf.run(
        question="What is the speed of light in a vacuum, in meters per second?"
    )
    result = await handler.result()
    data = result.to_dict()["result"]

    print()
    print("=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Model:      {data['model']}")
    print(f"Question:   {data['question']}")
    print(f"Answer:     {data['answer']}")
    print(f"Fact-check: {data['fact_check']}")
    print(f"Tokens:     {data['total_usage']}")


if __name__ == "__main__":
    asyncio.run(main())
