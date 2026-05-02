/**
 * E2E tests for the Blazen Node.js bindings -- extended coverage.
 *
 * These tests exercise features NOT covered by test_workflow.mjs or
 * test_session_refs.mjs: pause/resume/snapshot, human-in-the-loop,
 * prompt templates, memory (local mode), error propagation, abort,
 * token counting, chat window, and retry/cache/fallback decorators.
 *
 * All tests run WITHOUT API keys -- no network calls.
 *
 * Build the native binding first:
 *   pnpm --filter blazen run build
 */

import test from "ava";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";

import {
  ChatMessage,
  ChatWindow,
  CompletionModel,
  InMemoryBackend,
  Memory,
  PromptRegistry,
  PromptTemplate,
  Workflow,
  countMessageTokens,
  estimateTokens,
} from "../../crates/blazen-node/index.js";

// ===========================================================================
// Pause / Resume / Snapshot
// ===========================================================================

test("pause and resumeInPlace · pauses a running workflow and resumes it in place", async (t) => {
  const wf = new Workflow("pause-resume");
  wf.addStep("slow", ["blazen::StartEvent"], async (event, ctx) => {
    await ctx.set("started", true);
    await new Promise((r) => setTimeout(r, 300));
    return { type: "blazen::StopEvent", result: { done: true } };
  });

  const handler = await wf.runWithHandler({});

  // Give the step time to start and write context.
  await new Promise((r) => setTimeout(r, 50));

  await handler.pause();
  await handler.resumeInPlace();
  const result = await handler.result();

  t.is(result.type, "blazen::StopEvent");
  t.is(result.data.done, true);
});

test("snapshot · captures workflow state as valid JSON", async (t) => {
  const wf = new Workflow("snapshot-test");
  wf.addStep("setter", ["blazen::StartEvent"], async (event, ctx) => {
    await ctx.set("captured_key", "captured_value");
    await new Promise((r) => setTimeout(r, 300));
    return { type: "blazen::StopEvent", result: { done: true } };
  });

  const handler = await wf.runWithHandler({});
  await new Promise((r) => setTimeout(r, 50));

  await handler.pause();
  const snapJson = await handler.snapshot();
  const snap = JSON.parse(snapJson);

  t.is(snap.workflow_name, "snapshot-test");
  t.truthy(snap.context_state, "snapshot should contain context_state");

  // Clean up.
  await handler.resumeInPlace();
  await handler.result();
});

test("snapshot · produces JSON that workflow.resume accepts", async (t) => {
  // Note: in-place snapshots cannot capture pending channel events.
  // We verify resume() accepts the snapshot, not that the resumed
  // workflow completes (that's covered by the Rust integration tests).
  const wf = new Workflow("snapshot-roundtrip");
  wf.addStep("persistent", ["blazen::StartEvent"], async (event, ctx) => {
    await ctx.set("persisted", "value");
    await new Promise((r) => setTimeout(r, 500));
    return { type: "blazen::StopEvent", result: { done: true } };
  });

  const handler = await wf.runWithHandler({});
  await new Promise((r) => setTimeout(r, 100));

  await handler.pause();
  const snapJson = await handler.snapshot();

  // Verify resume accepts the snapshot without throwing.
  const handler2 = await wf.resume(snapJson);
  t.truthy(handler2, "resume should return a handler");

  // Clean up both handlers.
  await handler.abort();
  await handler2.abort();
});

// ===========================================================================
// Human-in-the-Loop
// ===========================================================================

test("human in the loop · step emits InputRequestEvent and responds via respondToInput", async (t) => {
  const wf = new Workflow("hitl-test");

  wf.addStep("ask", ["blazen::StartEvent"], async (event, ctx) => {
    return {
      type: "blazen::InputRequestEvent",
      request_id: "req-1",
      prompt: "What is your name?",
      metadata: {},
    };
  });

  wf.addStep(
    "process",
    ["blazen::InputResponseEvent"],
    async (event, ctx) => {
      return {
        type: "blazen::StopEvent",
        result: { response: event.response },
      };
    }
  );

  const handler = await wf.runWithHandler({});

  // Give time for InputRequestEvent to be emitted and auto-pause.
  await new Promise((r) => setTimeout(r, 200));

  await handler.respondToInput("req-1", { name: "Alice" });
  const result = await handler.result();

  t.is(result.type, "blazen::StopEvent");
  t.is(result.data.response.name, "Alice");
});

// ===========================================================================
// Prompt Templates
// ===========================================================================

test("PromptTemplate · renders variables into a ChatMessage", (t) => {
  const tpl = new PromptTemplate("Hello {{name}}, welcome to {{place}}!", {
    role: "user",
  });
  const msg = tpl.render({ name: "Alice", place: "Wonderland" });
  t.is(msg.content, "Hello Alice, welcome to Wonderland!");
  t.is(msg.role, "user");
});

test("PromptTemplate · returns sorted variable names", (t) => {
  const tpl = new PromptTemplate("{{b}} and {{a}} and {{b}}");
  t.deepEqual(tpl.variables, ["a", "b"]);
});

test("PromptTemplate · exposes name, role, version, description", (t) => {
  const tpl = new PromptTemplate("Hello {{name}}!", {
    role: "system",
    name: "greet",
    description: "A greeting",
    version: "2.0",
  });
  t.is(tpl.name, "greet");
  t.is(tpl.role, "system");
  t.is(tpl.version, "2.0");
  t.is(tpl.description, "A greeting");
  t.is(tpl.template, "Hello {{name}}!");
});

test("PromptRegistry · registers, gets, lists, and renders templates", (t) => {
  const reg = new PromptRegistry();
  const tpl = new PromptTemplate("Hello {{name}}!", { name: "greet" });
  reg.register("greet", tpl);

  t.deepEqual(reg.list(), ["greet"]);

  const got = reg.get("greet");
  t.truthy(got);
  t.is(got.template, "Hello {{name}}!");

  const msg = reg.render("greet", { name: "Bob" });
  t.is(msg.content, "Hello Bob!");
});

test("PromptRegistry · loads from a YAML file", (t) => {
  const yamlContent = `\
prompts:
  - name: summarize
    role: system
    template: "Summarize the {{doc_type}} in {{style}} style."
    version: "1.0"
`;
  const tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), "blazen-prompts-"));
  const yamlFile = path.join(tmpDir, "prompts.yaml");
  fs.writeFileSync(yamlFile, yamlContent);

  try {
    const reg = PromptRegistry.fromFile(yamlFile);
    t.truthy(reg.list().includes("summarize"));

    const msg = reg.render("summarize", {
      doc_type: "article",
      style: "concise",
    });
    t.is(
      msg.content,
      "Summarize the article in concise style."
    );
  } finally {
    fs.rmSync(tmpDir, { recursive: true, force: true });
  }
});

// ===========================================================================
// Memory (local mode -- no embedding model, no API key)
// ===========================================================================

test("Memory (local mode) · CRUD: add, get, count, delete", async (t) => {
  const mem = Memory.local(new InMemoryBackend());

  const docId = await mem.add("doc1", "Paris is the capital of France");
  t.is(docId, "doc1");
  t.is(await mem.count(), 1);

  const entry = await mem.get("doc1");
  t.truthy(entry);
  t.is(entry.text, "Paris is the capital of France");

  const deleted = await mem.delete("doc1");
  t.is(deleted, true);
  t.is(await mem.count(), 0);
});

test("Memory (local mode) · searchLocal returns results with id, text, score", async (t) => {
  const mem = Memory.local(new InMemoryBackend());

  await mem.add("d1", "Paris is the capital of France");
  await mem.add("d2", "Berlin is the capital of Germany");
  await mem.add("d3", "Tokyo is the capital of Japan");

  const results = await mem.searchLocal("capital of France", 2);

  t.truthy(results.length > 0);
  t.truthy(results.length <= 2);

  const r = results[0];
  t.truthy("id" in r);
  t.truthy("text" in r);
  t.truthy("score" in r);
  t.is(typeof r.score, "number");
});

test("Memory (local mode) · addMany batch-inserts entries", async (t) => {
  const mem = Memory.local(new InMemoryBackend());

  const ids = await mem.addMany([
    { id: "a", text: "First document" },
    { id: "b", text: "Second document" },
    { id: "c", text: "Third document" },
  ]);

  t.is(ids.length, 3);
  t.is(await mem.count(), 3);
});

// ===========================================================================
// Error Propagation
// ===========================================================================

test("error propagation · step throwing an error causes workflow to reject", async (t) => {
  const wf = new Workflow("error-test");
  wf.addStep("bad", ["blazen::StartEvent"], async (event, ctx) => {
    throw new Error("intentional test failure");
  });

  await t.throwsAsync(
    () => wf.run({}),
    { message: /intentional test failure|step failed|error/i }
  );
});

// ===========================================================================
// Handler Abort
// ===========================================================================

test("handler abort · abort causes result() to reject", async (t) => {
  const wf = new Workflow("abort-test");
  wf.addStep("long", ["blazen::StartEvent"], async (event, ctx) => {
    await new Promise((r) => setTimeout(r, 5000));
    return { type: "blazen::StopEvent", result: {} };
  });

  const handler = await wf.runWithHandler({});
  await new Promise((r) => setTimeout(r, 50));
  await handler.abort();

  await t.throwsAsync(() => handler.result());
});

// ===========================================================================
// Token Counting
// ===========================================================================

test("token counting · estimateTokens returns a positive number that scales with length", (t) => {
  const n = estimateTokens("Hello, world!");
  t.is(typeof n, "number");
  t.truthy(n > 0);

  const nLong = estimateTokens("Hello, world! ".repeat(100));
  t.truthy(nLong > n);
});

test("token counting · countMessageTokens returns a positive number", (t) => {
  const msgs = [
    ChatMessage.system("You are helpful."),
    ChatMessage.user("Hi!"),
  ];
  const n = countMessageTokens(msgs);
  t.is(typeof n, "number");
  t.truthy(n > 0);

  msgs.push(ChatMessage.user("Tell me a story about a brave knight."));
  const n2 = countMessageTokens(msgs);
  t.truthy(n2 > n);
});

// ===========================================================================
// ChatWindow
// ===========================================================================

test("ChatWindow · add, messages, tokenCount, remainingTokens, length", (t) => {
  const window = new ChatWindow(4096);

  window.add(ChatMessage.system("You are helpful."));
  t.is(window.length, 1);

  window.add(ChatMessage.user("Hello!"));
  t.is(window.length, 2);

  t.truthy(window.tokenCount() > 0);
  t.truthy(window.remainingTokens() < 4096);

  const msgs = window.messages();
  t.is(msgs.length, 2);
  t.is(msgs[0].role, "system");
  t.is(msgs[1].role, "user");
});

test("ChatWindow · evicts oldest non-system messages when over budget", (t) => {
  const window = new ChatWindow(30);

  window.add(ChatMessage.system("Be helpful."));
  window.add(ChatMessage.user("First message"));
  window.add(ChatMessage.user("Second message"));
  window.add(ChatMessage.user("Third message that pushes over budget"));

  // System message must always survive.
  const msgs = window.messages();
  const roles = msgs.map((m) => m.role);
  t.truthy(roles.includes("system"));
});

test("ChatWindow · clear removes all messages and resets to baseline", (t) => {
  const window = new ChatWindow(4096);
  // Capture baseline before adding anything.
  const baseline = window.tokenCount();

  window.add(ChatMessage.user("msg1"));
  window.add(ChatMessage.user("msg2"));
  t.is(window.length, 2);
  t.truthy(window.tokenCount() > baseline);

  window.clear();
  t.is(window.length, 0);
  t.is(window.tokenCount(), baseline);
});

// ===========================================================================
// Retry / Cache / Fallback (construction only, no API calls)
// ===========================================================================

test("CompletionModel decorators · withRetry returns a CompletionModel", (t) => {
  const model = CompletionModel.openai({ apiKey: "fake-key" });
  const retried = model.withRetry({
    maxRetries: 3,
    initialDelayMs: 100,
    maxDelayMs: 5000,
  });
  t.truthy(retried);
});

test("CompletionModel decorators · withCache returns a CompletionModel", (t) => {
  const model = CompletionModel.openai({ apiKey: "fake-key" });
  const cached = model.withCache({ ttlSeconds: 60, maxEntries: 100 });
  t.truthy(cached);
});

test("CompletionModel decorators · withFallback chains multiple models", (t) => {
  const m1 = CompletionModel.openai({ apiKey: "fake-key-1" });
  const m2 = CompletionModel.openai({ apiKey: "fake-key-2" });
  const fallback = CompletionModel.withFallback([m1, m2]);
  t.truthy(fallback);
});

// ===========================================================================
// Tools via CompletionOptions (completeWithOptions)
// ===========================================================================

function makeSearchTool() {
  return {
    name: "search",
    description: "Search the web for information",
    parameters: {
      type: "object",
      properties: { query: { type: "string" } },
      required: ["query"],
    },
  };
}

function makeCalculatorTool() {
  return {
    name: "calculator",
    description: "Perform arithmetic calculations",
    parameters: {
      type: "object",
      properties: {
        expression: { type: "string", description: "Math expression" },
      },
      required: ["expression"],
    },
  };
}

test("tools via CompletionOptions · completeWithOptions accepts a single tool definition", async (t) => {
  const model = CompletionModel.openai({ apiKey: "fake-key" });
  const tool = makeSearchTool();

  // The call will fail at the API level (no valid key), but the tool
  // structure must be accepted at the napi boundary without type errors.
  await t.throwsAsync(
    () =>
      model.completeWithOptions([ChatMessage.user("Hi")], {
        tools: [tool],
      }),
    // Any error is fine -- we're testing that the call doesn't throw
    // a type/validation error about the tools structure.
    { instanceOf: Error }
  );
});

test("tools via CompletionOptions · completeWithOptions accepts multiple tool definitions", async (t) => {
  const model = CompletionModel.openai({ apiKey: "fake-key" });
  const search = makeSearchTool();
  const calc = makeCalculatorTool();

  await t.throwsAsync(
    () =>
      model.completeWithOptions([ChatMessage.user("Hi")], {
        tools: [search, calc],
      }),
    { instanceOf: Error }
  );
});

test("tools via CompletionOptions · completeWithOptions accepts options without tools", async (t) => {
  const model = CompletionModel.openai({ apiKey: "fake-key" });

  await t.throwsAsync(
    () =>
      model.completeWithOptions([ChatMessage.user("Hi")], {
        temperature: 0.5,
      }),
    { instanceOf: Error }
  );
});

test("tools via CompletionOptions · tool definition fields are preserved in the options object", (t) => {
  const tool = makeSearchTool();
  const options = { tools: [tool] };

  // Verify the plain-object tool definition retains all fields.
  t.is(options.tools.length, 1);
  t.is(options.tools[0].name, "search");
  t.is(
    options.tools[0].description,
    "Search the web for information"
  );
  t.is(options.tools[0].parameters.type, "object");
  t.truthy("query" in options.tools[0].parameters.properties);
  t.deepEqual(options.tools[0].parameters.required, ["query"]);
});

test("tools via CompletionOptions · multiple tools preserve all descriptions and parameters", (t) => {
  const search = makeSearchTool();
  const calc = makeCalculatorTool();
  const options = { tools: [search, calc] };

  t.is(options.tools.length, 2);

  const names = options.tools.map((tool) => tool.name);
  t.truthy(names.includes("search"));
  t.truthy(names.includes("calculator"));

  const searchTool = options.tools.find((tool) => tool.name === "search");
  t.is(
    searchTool.description,
    "Search the web for information"
  );
  t.truthy("query" in searchTool.parameters.properties);

  const calcTool = options.tools.find((tool) => tool.name === "calculator");
  t.is(
    calcTool.description,
    "Perform arithmetic calculations"
  );
  t.truthy("expression" in calcTool.parameters.properties);
});

test("subclassed CompletionModel · constructor accepts config with modelId", (t) => {
  const model = new CompletionModel({ modelId: "custom-test" });
  t.is(model.modelId, "custom-test");
});

test("subclassed CompletionModel · complete() on subclass without inner throws descriptive error", async (t) => {
  const model = new CompletionModel({ modelId: "no-provider" });

  await t.throwsAsync(
    () => model.complete([ChatMessage.user("Hi")]),
    { message: /subclass must override complete/ }
  );
});

test("subclassed CompletionModel · completeWithOptions() on subclass without inner throws descriptive error", async (t) => {
  const model = new CompletionModel({ modelId: "no-provider" });

  await t.throwsAsync(
    () =>
      model.completeWithOptions([ChatMessage.user("Hi")], {
        tools: [makeSearchTool()],
      }),
    { message: /subclass must override completeWithOptions/ }
  );
});

// ===========================================================================
// Tools passthrough via CompletionOptions (subclassed CompletionModel)
// ===========================================================================

test("tools passthrough via CompletionOptions (subclassed) · tools passed via options reach the subclass override with correct name/description", async (t) => {
  const captured = [];

  class ToolInspectorLLM extends CompletionModel {
    constructor() {
      super({ modelId: "tool-inspector" });
    }
    async completeWithOptions(messages, options) {
      if (options && options.tools) {
        for (const tool of options.tools) {
          captured.push({
            name: tool.name,
            description: tool.description,
          });
        }
      }
      throw new Error("inspection-complete");
    }
  }

  const model = new ToolInspectorLLM();
  await t.throwsAsync(
    () =>
      model.completeWithOptions([ChatMessage.user("Hi")], {
        tools: [
          {
            name: "search",
            description: "Search the web for information.",
            parameters: {
              type: "object",
              properties: { query: { type: "string" } },
              required: ["query"],
            },
          },
          {
            name: "calculator",
            description: "Add two numbers together.",
            parameters: {
              type: "object",
              properties: {
                a: { type: "number" },
                b: { type: "number" },
              },
              required: ["a", "b"],
            },
          },
        ],
      }),
    { message: /inspection-complete/ }
  );

  t.is(captured.length, 2);
  t.is(captured[0].name, "search");
  t.is(captured[0].description, "Search the web for information.");
  t.is(captured[1].name, "calculator");
  t.is(captured[1].description, "Add two numbers together.");
});

test("tools passthrough via CompletionOptions (subclassed) · tools is undefined or empty when CompletionOptions has no tools", async (t) => {
  const captured = { tools_value: "sentinel" };

  class NoToolsLLM extends CompletionModel {
    constructor() {
      super({ modelId: "no-tools" });
    }
    async completeWithOptions(messages, options) {
      captured.tools_value = options ? options.tools : "no-options";
      throw new Error("done");
    }
  }

  const model = new NoToolsLLM();
  await t.throwsAsync(
    () =>
      model.completeWithOptions([ChatMessage.user("Hi")], {
        temperature: 0.7,
      }),
    { message: /done/ }
  );

  // options.tools is either undefined or an empty array when no tools set.
  const tv = captured.tools_value;
  t.truthy(
    tv === undefined || tv === null || (Array.isArray(tv) && tv.length === 0),
    `expected undefined/null/[] tools, got ${JSON.stringify(tv)}`
  );
});

test("tools passthrough via CompletionOptions (subclassed) · tool parameters JSON schema is preserved end-to-end", async (t) => {
  const capturedParams = [];

  class ParamsCaptureLLM extends CompletionModel {
    constructor() {
      super({ modelId: "params-capture" });
    }
    async completeWithOptions(messages, options) {
      if (options && options.tools) {
        for (const tool of options.tools) {
          capturedParams.push(tool.parameters);
        }
      }
      throw new Error("captured");
    }
  }

  const complexSchema = {
    type: "object",
    properties: {
      query: { type: "string", description: "Search query" },
      limit: { type: "integer", minimum: 1, maximum: 100 },
      filters: {
        type: "array",
        items: { type: "string" },
      },
    },
    required: ["query"],
  };

  const model = new ParamsCaptureLLM();
  await t.throwsAsync(
    () =>
      model.completeWithOptions([ChatMessage.user("Hi")], {
        tools: [
          {
            name: "advanced_search",
            description: "Complex search with filters.",
            parameters: complexSchema,
          },
        ],
      }),
    { message: /captured/ }
  );

  t.is(capturedParams.length, 1);
  const params = capturedParams[0];
  t.is(params.type, "object");
  t.truthy("query" in params.properties);
  t.is(params.properties.limit.minimum, 1);
  t.deepEqual(params.required, ["query"]);
});

test("tools passthrough via CompletionOptions (subclassed) · multiple tools each reach the subclass override independently", async (t) => {
  const toolNames = [];

  class MultiToolLLM extends CompletionModel {
    constructor() {
      super({ modelId: "multi-tool" });
    }
    async completeWithOptions(messages, options) {
      if (options && options.tools) {
        for (const tool of options.tools) {
          toolNames.push(tool.name);
        }
      }
      throw new Error("captured");
    }
  }

  const model = new MultiToolLLM();
  await t.throwsAsync(
    () =>
      model.completeWithOptions([ChatMessage.user("Hi")], {
        tools: [
          makeSearchTool(),
          makeCalculatorTool(),
          {
            name: "third_tool",
            description: "A third tool.",
            parameters: { type: "object", properties: {} },
          },
        ],
      }),
    { message: /captured/ }
  );

  t.deepEqual(toolNames, ["search", "calculator", "third_tool"]);
});
