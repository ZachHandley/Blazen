"""Verify custom Python exception subclasses survive a ``run_agent`` round trip.

The newly-added ``pyerr_to_caller_error`` helper in
``crates/blazen-py/src/agent.rs`` captures the Python exception **instance**
(as ``Py<PyAny>``) when a tool handler raises, wraps it in a
``BlazenError::caller_error`` carrying that opaque source, and the
``blazen_error_to_pyerr`` ``CallerError`` arm in
``crates/blazen-py/src/error.rs`` downcasts that source and re-raises the
*original* exception via ``PyErr::from_value`` — preserving class identity
and custom attributes for ``except MyError`` pattern matching.

This file exercises the full round trip: a user-defined ``Exception``
subclass with custom attributes raised inside a tool handler must be caught
by ``except MyError`` on the caller side, and its attributes must survive.

No API keys / network access required.
"""

import pytest

from blazen import (
    ChatMessage,
    CompletionModel,
    ToolDef,
    run_agent,
)


# ---------------------------------------------------------------------------
# Custom Python exception with extra state
# ---------------------------------------------------------------------------


class SubmitEvaluationSignal(Exception):
    """A user-defined exception that carries a structured payload.

    Mirrors the pattern of an "exit signal" thrown from inside an agent tool
    handler: the caller wants to short-circuit ``run_agent`` and recover the
    payload via ``except SubmitEvaluationSignal``.
    """

    def __init__(self, payload):
        super().__init__("submit-evaluation")
        self.payload = payload


# ---------------------------------------------------------------------------
# Stub completion model that always emits a single tool call
# ---------------------------------------------------------------------------


class StubToolCallingModel(CompletionModel):
    """Subclassed ``CompletionModel`` whose ``complete`` always returns a single
    ``submit_eval`` tool call. The first invocation will trigger the tool
    handler — which raises ``SubmitEvaluationSignal`` — so the agent loop
    never reaches a second iteration.
    """

    def __init__(self):
        super().__init__(model_id="stub-error", context_length=4096)

    async def complete(self, messages, options=None):  # type: ignore[override]
        # The Rust ``CompletionResponse`` schema (see
        # ``crates/blazen-llm/src/types/completion.rs``) requires ``model``;
        # every other field is ``#[serde(default)]``. ``tool_calls`` accepts a
        # list of ``{id, name, arguments}`` dicts.
        return {
            "content": None,
            "model": "stub-error",
            "tool_calls": [
                {
                    "id": "call-1",
                    "name": "submit_eval",
                    "arguments": {"score": 9.5},
                }
            ],
            "finish_reason": "tool_calls",
            "metadata": {},
        }


# ---------------------------------------------------------------------------
# Main behaviour: subclass survives + custom attributes survive
# ---------------------------------------------------------------------------


async def test_run_agent_preserves_custom_exception_instance():
    """A custom ``Exception`` subclass raised in a tool handler is caught
    verbatim by ``except SubmitEvaluationSignal`` on the caller side, and
    its custom ``.payload`` attribute round-trips through the Rust bridge.
    """
    model = StubToolCallingModel()

    captured_args: dict[str, object] = {}

    def handler(arguments):
        captured_args["last"] = arguments
        raise SubmitEvaluationSignal(
            {"score": arguments["score"], "finished": True}
        )

    tool = ToolDef(
        name="submit_eval",
        description="Submit an evaluation result and signal completion.",
        parameters={
            "type": "object",
            "properties": {"score": {"type": "number"}},
            "required": ["score"],
        },
        handler=handler,
    )

    with pytest.raises(SubmitEvaluationSignal) as exc_info:
        await run_agent(
            model,
            [ChatMessage.user("Evaluate this.")],
            tools=[tool],
            max_iterations=3,
        )

    # The handler ran exactly once with the arguments the stub model produced.
    assert captured_args["last"] == {"score": 9.5}

    # Class identity survived the round trip through Rust.
    assert isinstance(exc_info.value, SubmitEvaluationSignal)

    # Custom attribute survived the round trip through Rust.
    assert exc_info.value.payload == {"score": 9.5, "finished": True}

    # Base message also survived.
    assert str(exc_info.value) == "submit-evaluation"


# ---------------------------------------------------------------------------
# Same behaviour for an async tool handler (separate dispatch path)
# ---------------------------------------------------------------------------


async def test_run_agent_preserves_custom_exception_from_async_handler():
    """Async tool handlers take a separate path in
    ``PyToolWrapper::execute`` (``self.is_async == true``), so verify the
    class + attribute round trip independently.
    """
    model = StubToolCallingModel()

    async def handler(arguments):
        raise SubmitEvaluationSignal(
            {"score": arguments["score"], "async": True}
        )

    tool = ToolDef(
        name="submit_eval",
        description="Submit an evaluation result and signal completion.",
        parameters={
            "type": "object",
            "properties": {"score": {"type": "number"}},
            "required": ["score"],
        },
        handler=handler,
    )

    with pytest.raises(SubmitEvaluationSignal) as exc_info:
        await run_agent(
            model,
            [ChatMessage.user("Evaluate this.")],
            tools=[tool],
            max_iterations=3,
        )

    assert isinstance(exc_info.value, SubmitEvaluationSignal)
    assert exc_info.value.payload == {"score": 9.5, "async": True}
    assert str(exc_info.value) == "submit-evaluation"
