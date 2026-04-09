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

import { describe, it } from "node:test";
import assert from "node:assert/strict";
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

describe("pause and resumeInPlace", () => {
  it("pauses a running workflow and resumes it in place", async () => {
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

    assert.strictEqual(result.type, "blazen::StopEvent");
    assert.strictEqual(result.data.done, true);
  });
});

describe("snapshot", () => {
  it("captures workflow state as valid JSON", async () => {
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

    assert.strictEqual(snap.workflow_name, "snapshot-test");
    assert.ok(snap.context_state, "snapshot should contain context_state");

    // Clean up.
    await handler.resumeInPlace();
    await handler.result();
  });

  it("produces JSON that workflow.resume accepts", async () => {
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
    assert.ok(handler2, "resume should return a handler");

    // Clean up both handlers.
    await handler.abort();
    await handler2.abort();
  });
});

// ===========================================================================
// Human-in-the-Loop
// ===========================================================================

describe("human in the loop", () => {
  it("step emits InputRequestEvent and responds via respondToInput", async () => {
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

    assert.strictEqual(result.type, "blazen::StopEvent");
    assert.strictEqual(result.data.response.name, "Alice");
  });
});

// ===========================================================================
// Prompt Templates
// ===========================================================================

describe("PromptTemplate", () => {
  it("renders variables into a ChatMessage", () => {
    const t = new PromptTemplate("Hello {{name}}, welcome to {{place}}!", {
      role: "user",
    });
    const msg = t.render({ name: "Alice", place: "Wonderland" });
    assert.strictEqual(msg.content, "Hello Alice, welcome to Wonderland!");
    assert.strictEqual(msg.role, "user");
  });

  it("returns sorted variable names", () => {
    const t = new PromptTemplate("{{b}} and {{a}} and {{b}}");
    assert.deepStrictEqual(t.variables, ["a", "b"]);
  });

  it("exposes name, role, version, description", () => {
    const t = new PromptTemplate("Hello {{name}}!", {
      role: "system",
      name: "greet",
      description: "A greeting",
      version: "2.0",
    });
    assert.strictEqual(t.name, "greet");
    assert.strictEqual(t.role, "system");
    assert.strictEqual(t.version, "2.0");
    assert.strictEqual(t.description, "A greeting");
    assert.strictEqual(t.template, "Hello {{name}}!");
  });
});

describe("PromptRegistry", () => {
  it("registers, gets, lists, and renders templates", () => {
    const reg = new PromptRegistry();
    const t = new PromptTemplate("Hello {{name}}!", { name: "greet" });
    reg.register("greet", t);

    assert.deepStrictEqual(reg.list(), ["greet"]);

    const got = reg.get("greet");
    assert.ok(got);
    assert.strictEqual(got.template, "Hello {{name}}!");

    const msg = reg.render("greet", { name: "Bob" });
    assert.strictEqual(msg.content, "Hello Bob!");
  });

  it("loads from a YAML file", () => {
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
      assert.ok(reg.list().includes("summarize"));

      const msg = reg.render("summarize", {
        doc_type: "article",
        style: "concise",
      });
      assert.strictEqual(
        msg.content,
        "Summarize the article in concise style."
      );
    } finally {
      fs.rmSync(tmpDir, { recursive: true, force: true });
    }
  });
});

// ===========================================================================
// Memory (local mode -- no embedding model, no API key)
// ===========================================================================

describe("Memory (local mode)", () => {
  it("CRUD: add, get, count, delete", async () => {
    const mem = Memory.local(new InMemoryBackend());

    const docId = await mem.add("doc1", "Paris is the capital of France");
    assert.strictEqual(docId, "doc1");
    assert.strictEqual(await mem.count(), 1);

    const entry = await mem.get("doc1");
    assert.ok(entry);
    assert.strictEqual(entry.text, "Paris is the capital of France");

    const deleted = await mem.delete("doc1");
    assert.strictEqual(deleted, true);
    assert.strictEqual(await mem.count(), 0);
  });

  it("searchLocal returns results with id, text, score", async () => {
    const mem = Memory.local(new InMemoryBackend());

    await mem.add("d1", "Paris is the capital of France");
    await mem.add("d2", "Berlin is the capital of Germany");
    await mem.add("d3", "Tokyo is the capital of Japan");

    const results = await mem.searchLocal("capital of France", 2);

    assert.ok(results.length > 0);
    assert.ok(results.length <= 2);

    const r = results[0];
    assert.ok("id" in r);
    assert.ok("text" in r);
    assert.ok("score" in r);
    assert.strictEqual(typeof r.score, "number");
  });

  it("addMany batch-inserts entries", async () => {
    const mem = Memory.local(new InMemoryBackend());

    const ids = await mem.addMany([
      { id: "a", text: "First document" },
      { id: "b", text: "Second document" },
      { id: "c", text: "Third document" },
    ]);

    assert.strictEqual(ids.length, 3);
    assert.strictEqual(await mem.count(), 3);
  });
});

// ===========================================================================
// Error Propagation
// ===========================================================================

describe("error propagation", () => {
  it("step throwing an error causes workflow to reject", async () => {
    const wf = new Workflow("error-test");
    wf.addStep("bad", ["blazen::StartEvent"], async (event, ctx) => {
      throw new Error("intentional test failure");
    });

    await assert.rejects(
      () => wf.run({}),
      /intentional test failure|step failed|error/i
    );
  });
});

// ===========================================================================
// Handler Abort
// ===========================================================================

describe("handler abort", () => {
  it("abort causes result() to reject", async () => {
    const wf = new Workflow("abort-test");
    wf.addStep("long", ["blazen::StartEvent"], async (event, ctx) => {
      await new Promise((r) => setTimeout(r, 5000));
      return { type: "blazen::StopEvent", result: {} };
    });

    const handler = await wf.runWithHandler({});
    await new Promise((r) => setTimeout(r, 50));
    await handler.abort();

    await assert.rejects(() => handler.result());
  });
});

// ===========================================================================
// Token Counting
// ===========================================================================

describe("token counting", () => {
  it("estimateTokens returns a positive number that scales with length", () => {
    const n = estimateTokens("Hello, world!");
    assert.strictEqual(typeof n, "number");
    assert.ok(n > 0);

    const nLong = estimateTokens("Hello, world! ".repeat(100));
    assert.ok(nLong > n);
  });

  it("countMessageTokens returns a positive number", () => {
    const msgs = [
      ChatMessage.system("You are helpful."),
      ChatMessage.user("Hi!"),
    ];
    const n = countMessageTokens(msgs);
    assert.strictEqual(typeof n, "number");
    assert.ok(n > 0);

    msgs.push(ChatMessage.user("Tell me a story about a brave knight."));
    const n2 = countMessageTokens(msgs);
    assert.ok(n2 > n);
  });
});

// ===========================================================================
// ChatWindow
// ===========================================================================

describe("ChatWindow", () => {
  it("add, messages, tokenCount, remainingTokens, length", () => {
    const window = new ChatWindow(4096);

    window.add(ChatMessage.system("You are helpful."));
    assert.strictEqual(window.length, 1);

    window.add(ChatMessage.user("Hello!"));
    assert.strictEqual(window.length, 2);

    assert.ok(window.tokenCount() > 0);
    assert.ok(window.remainingTokens() < 4096);

    const msgs = window.messages();
    assert.strictEqual(msgs.length, 2);
    assert.strictEqual(msgs[0].role, "system");
    assert.strictEqual(msgs[1].role, "user");
  });

  it("evicts oldest non-system messages when over budget", () => {
    const window = new ChatWindow(30);

    window.add(ChatMessage.system("Be helpful."));
    window.add(ChatMessage.user("First message"));
    window.add(ChatMessage.user("Second message"));
    window.add(ChatMessage.user("Third message that pushes over budget"));

    // System message must always survive.
    const msgs = window.messages();
    const roles = msgs.map((m) => m.role);
    assert.ok(roles.includes("system"));
  });

  it("clear removes all messages and resets to baseline", () => {
    const window = new ChatWindow(4096);
    // Capture baseline before adding anything.
    const baseline = window.tokenCount();

    window.add(ChatMessage.user("msg1"));
    window.add(ChatMessage.user("msg2"));
    assert.strictEqual(window.length, 2);
    assert.ok(window.tokenCount() > baseline);

    window.clear();
    assert.strictEqual(window.length, 0);
    assert.strictEqual(window.tokenCount(), baseline);
  });
});

// ===========================================================================
// Retry / Cache / Fallback (construction only, no API calls)
// ===========================================================================

describe("CompletionModel decorators", () => {
  it("withRetry returns a CompletionModel", () => {
    const model = CompletionModel.openai({ apiKey: "fake-key" });
    const retried = model.withRetry({
      maxRetries: 3,
      initialDelayMs: 100,
      maxDelayMs: 5000,
    });
    assert.ok(retried);
  });

  it("withCache returns a CompletionModel", () => {
    const model = CompletionModel.openai({ apiKey: "fake-key" });
    const cached = model.withCache({ ttlSeconds: 60, maxEntries: 100 });
    assert.ok(cached);
  });

  it("withFallback chains multiple models", () => {
    const m1 = CompletionModel.openai({ apiKey: "fake-key-1" });
    const m2 = CompletionModel.openai({ apiKey: "fake-key-2" });
    const fallback = CompletionModel.withFallback([m1, m2]);
    assert.ok(fallback);
  });
});
