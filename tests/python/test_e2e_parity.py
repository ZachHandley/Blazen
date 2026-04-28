"""End-to-end parity test exercising the full SDK shape.

This scenario is mirrored in the Node and WASM test suites to verify
cross-binding behavioral parity.

Covers:
- A 2-stage Pipeline built via PipelineBuilder.
- A custom CompletionModel subclass (sync override of complete) that
  returns a deterministic response without touching the network.
- A PromptTemplate rendered with the previous stage's output.
- A Memory store backed by InMemoryBackend, written to and queried.

No network access or API keys are required.
"""

import pytest

from blazen import (
    ChatMessage,
    CompletionModel,
    Context,
    Event,
    InMemoryBackend,
    Memory,
    Pipeline,
    PipelineBuilder,
    PipelineResult,
    PromptTemplate,
    Stage,
    StageResult,
    StartEvent,
    StopEvent,
    TemplateRole,
    Workflow,
    step,
)


# Deterministic mock provider used in stage 1.
class HelloWorldLLM(CompletionModel):
    """A CustomProvider subclass that always returns 'Hello World'."""

    def __new__(cls):
        return super().__new__(cls, model_id="hello-world-mock")

    async def complete(self, messages, options=None):
        # Sync logic wrapped in an async def -- no awaiting, no IO.
        return {
            "content": "Hello World",
            "model": "hello-world-mock",
            "tool_calls": [],
            "finish_reason": "stop",
        }


@pytest.mark.asyncio
async def test_e2e_parity_pipeline_with_memory_and_prompt_template():
    # Single shared instances used across stages.
    llm = HelloWorldLLM()
    backend = InMemoryBackend()
    memory = Memory.local(backend)
    template = PromptTemplate(
        "Echo says: {{phrase}}",
        role=TemplateRole.User,
        name="echo-template",
    )

    # Stage 1: ask the custom provider for a deterministic completion.
    @step
    async def call_llm(ctx: Context, ev: StartEvent) -> StopEvent:
        response = await llm.complete([ChatMessage.user("greet me")])
        # Subclass returns a dict, accessible by key.
        content = response["content"]
        return StopEvent(result={"llm_text": content})

    # Stage 2: render a PromptTemplate from stage 1 output and persist
    # the rendered text to Memory.
    @step
    async def render_and_store(ctx: Context, ev: StartEvent) -> StopEvent:
        data = ev.to_dict()
        phrase = data.get("llm_text", "")
        rendered = template.render(phrase=phrase)
        # Persist the rendered prompt content into memory.
        doc_id = await memory.add("rendered-doc", rendered.content)
        return StopEvent(
            result={
                "rendered": rendered.content,
                "doc_id": doc_id,
                "role": rendered.role,
            }
        )

    wf_llm = Workflow("llm-wf", [call_llm])
    wf_render = Workflow("render-wf", [render_and_store])

    stage_llm = Stage(name="generate", workflow=wf_llm)
    stage_render = Stage(name="render", workflow=wf_render)

    pipeline = (
        PipelineBuilder("e2e-parity")
        .stage(stage_llm)
        .stage(stage_render)
        .build()
    )
    assert isinstance(pipeline, Pipeline)

    # Run the pipeline.
    handler = await pipeline.start()
    result = await handler.result()

    # PipelineResult shape assertions.
    assert isinstance(result, PipelineResult)
    assert result.pipeline_name == "e2e-parity"

    stage_results = result.stage_results
    assert len(stage_results) == 2
    for sr in stage_results:
        assert isinstance(sr, StageResult)
        assert sr.skipped is False

    # Stage 1: custom provider produced "Hello World".
    assert stage_results[0].name == "generate"
    assert stage_results[0].output == {"llm_text": "Hello World"}

    # Stage 2: PromptTemplate rendered correctly with stage 1 result.
    assert stage_results[1].name == "render"
    final = stage_results[1].output
    assert final["rendered"] == "Echo says: Hello World"
    assert final["doc_id"] == "rendered-doc"
    assert final["role"] == "user"

    # final_output of the pipeline is the last stage's output.
    assert result.final_output["rendered"] == "Echo says: Hello World"

    # Memory backend was written to: count, get, search_local all succeed.
    count = await memory.count()
    assert count == 1

    entry = await memory.get("rendered-doc")
    assert entry is not None
    assert entry["text"] == "Echo says: Hello World"

    hits = await memory.search_local("Hello World", limit=5)
    assert isinstance(hits, list)
    assert len(hits) >= 1
    assert any(h.id == "rendered-doc" for h in hits)
    assert any(h.text == "Echo says: Hello World" for h in hits)


# Reference imports kept to assert the symbols are exported.
_ = (Event,)
