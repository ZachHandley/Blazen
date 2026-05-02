// End-to-end parity test for the Node.js bindings.
//
// Mirrors tests/python/test_e2e_parity.py: verifies that a 2-stage Pipeline
// composed of Stage + Workflow + CustomProvider + InMemoryBackend-backed
// Memory + PromptTemplate can be wired together via the public napi-rs
// surface without throwing at construction time, and that the resulting
// Pipeline exposes the expected method shape.
//
// Test bodies are SYNC ONLY -- per project memory async test bodies hang
// the Node test runner because the napi-rs tokio runtime keeps handles
// alive past Promise resolution. Pipeline.start() is therefore never
// invoked from this file; runtime behavior is exercised separately by
// subprocess-based harnesses.
//
// Build the native binding first:
//   pnpm --filter blazen run build

import test from "ava";

import {
  ChatMessage,
  CompletionModel,
  CustomProvider,
  EmbeddingModel,
  InMemoryBackend,
  Memory,
  Pipeline,
  PipelineBuilder,
  PromptTemplate,
  Stage,
  Workflow,
} from "../../crates/blazen-node/index.js";

// ===========================================================================
// Helpers -- minimal stub subclasses used purely for shape wiring.
// ===========================================================================

// A trivial CompletionModel subclass. complete() is never invoked from this
// file so the default "subclass must override" body is fine.
class StubLLM extends CompletionModel {
  constructor() {
    super({ modelId: "stub-llm", contextLength: 4096 });
  }
}

// A trivial EmbeddingModel subclass. We do NOT pass this into Memory because
// Memory rejects subclassed embedders without an inner provider (see
// test_subclass_providers.mjs). For the Memory branch we use Memory.local()
// which requires no embedder at all.
class StubEmbed extends EmbeddingModel {
  constructor() {
    super({ modelId: "stub-embed", dimensions: 16 });
  }
}

// ===========================================================================
// e2e parity (shape)
// ===========================================================================

test("e2e parity (shape) · Pipeline + CustomProvider + InMemoryBackend + PromptTemplate wire together", (t) => {
  // 1. PromptTemplate -- exercise the parity primitive.
  const template = new PromptTemplate(
    "Summarize the following text: {{text}}",
    { role: "user", name: "summarize" },
  );
  t.is(template.role, "user");
  t.is(template.name, "summarize");
  t.deepEqual(template.variables, ["text"]);

  const rendered = template.render({ text: "hello world" });
  t.truthy(rendered instanceof ChatMessage);
  t.is(rendered.role, "user");
  t.truthy(rendered.content && rendered.content.includes("hello world"));

  // 2. CustomProvider -- wrap a host object exposing a TTS-shaped method.
  // No method is invoked here; we only verify the wrapper constructs.
  const host = {
    async textToSpeech(_request) {
      return { audio: [], timing: {}, metadata: {} };
    },
  };
  const customProvider = new CustomProvider(host, {
    providerId: "stub-custom",
  });
  t.is(customProvider.providerId, "stub-custom");

  // 3. InMemoryBackend wrapped in a Memory (local-only mode -- no
  // embedder needed, and avoids the "subclassed EmbeddingModel" rejection
  // that Memory's full constructor enforces).
  const backend = new InMemoryBackend();
  const memory = Memory.local(backend);
  t.truthy(memory instanceof Memory);

  // Sanity: ensure StubEmbed/StubLLM also construct so their inclusion
  // in this parity test mirrors the Python file's surface, even though
  // we don't hand them to Memory.
  const embedder = new StubEmbed();
  const llm = new StubLLM();
  t.is(embedder.modelId, "stub-embed");
  t.is(llm.modelId, "stub-llm");

  // 4. Build two distinct workflows and wrap them as Stages.
  const wf1 = new Workflow("stage-1");
  wf1.addStep("ingest", ["blazen::StartEvent"], (_event, _ctx) => {
    // Hand off to stage 2 via StopEvent containing prompt text.
    return {
      type: "blazen::StopEvent",
      result: { prompt: "hello world" },
    };
  });

  const wf2 = new Workflow("stage-2");
  wf2.addStep("respond", ["blazen::StartEvent"], (_event, _ctx) => {
    return {
      type: "blazen::StopEvent",
      result: { reply: "ok" },
    };
  });

  const stage1 = new Stage("ingest", wf1);
  const stage2 = new Stage("respond", wf2);
  t.is(stage1.name, "ingest");
  t.is(stage2.name, "respond");

  // 5. Build the 2-stage Pipeline.
  const pipeline = new PipelineBuilder("e2e-parity")
    .stage(stage1)
    .stage(stage2)
    .build();

  t.truthy(pipeline instanceof Pipeline);
  t.is(typeof pipeline.start, "function");
  t.is(typeof pipeline.resume, "function");
});

test("e2e parity (shape) · PromptTemplate role override flows through to ChatMessage", (t) => {
  const sysTemplate = new PromptTemplate(
    "You are a helpful assistant named {{name}}.",
    { role: "system", name: "system-prompt" },
  );
  t.is(sysTemplate.role, "system");

  const msg = sysTemplate.render({ name: "Blazen" });
  t.is(msg.role, "system");
  t.truthy(msg.content && msg.content.includes("Blazen"));
});

test("e2e parity (shape) · PipelineBuilder rejects empty pipelines (parity with Python)", (t) => {
  // Mirrors the Python parity assertion that an empty pipeline cannot be
  // built -- the builder must require at least one stage.
  t.throws(
    () => new PipelineBuilder("empty").build(),
    { message: /stage|empty|at least/i },
  );
});

test("e2e parity (shape) · CustomProvider providerId defaults to 'custom' when omitted", (t) => {
  // Mirrors the Python parity expectation that an unspecified provider id
  // falls back to a stable default.
  const provider = new CustomProvider({});
  t.is(provider.providerId, "custom");
});
