// Cloudflare Workers wasi smoke test — generic probe dispatcher.
//
// Each probe is a small named function that exercises one slice of the
// Blazen wasi API. The runner (run-tests.mjs) hits each probe over HTTP
// and prints a table of pass/fail.
//
// Add a probe: just add an entry to PROBES below. Each probe returns
// either a value (treated as `pass`) or throws (treated as `fail` with
// the error message).

import {
  instantiateNapiModuleSync,
  getDefaultContext,
  WASI,
} from "@napi-rs/wasm-runtime";

// @ts-expect-error — wrangler resolves `*.wasm` as `WebAssembly.Module`
import wasmModule from "../../../crates/blazen-node/blazen.wasm32-wasi.wasm";

// ---- napi-rs wasi runtime init -------------------------------------------

const wasi = new WASI({
  version: "preview1",
  args: [],
  env: {},
  preopens: {},
});

const sharedMemory = new WebAssembly.Memory({
  initial: 4000,
  maximum: 65536,
  shared: true,
});

const napi = instantiateNapiModuleSync(wasmModule as WebAssembly.Module, {
  context: getDefaultContext(),
  asyncWorkPoolSize: 0, // workerd has no `Worker` class
  wasi,
  overwriteImports(importObject: any) {
    importObject.env = {
      ...importObject.env,
      ...importObject.napi,
      ...importObject.emnapi,
      memory: sharedMemory,
    };
    return importObject;
  },
  beforeInit({ instance }: any) {
    for (const name of Object.keys(instance.exports)) {
      if (name.startsWith("__napi_register__")) {
        instance.exports[name]();
      }
    }
  },
});

const b = napi.napiModule.exports as any;

// blazen's wasi async dispatcher (Wave 6B) calls
// `globalThis.__blazenDrainAsyncQueue` from a Promise microtask. Wire it.
(globalThis as any).__blazenDrainAsyncQueue = b.__blazenDrainAsyncQueue;

// ---- helpers --------------------------------------------------------------

function mapToObject(value: unknown): unknown {
  if (value instanceof Map) {
    const out: Record<string, unknown> = {};
    for (const [k, v] of value.entries()) out[String(k)] = mapToObject(v);
    return out;
  }
  if (Array.isArray(value)) return value.map(mapToObject);
  if (value && typeof value === "object") {
    const out: Record<string, unknown> = {};
    for (const [k, v] of Object.entries(value as Record<string, unknown>)) {
      out[k] = mapToObject(v);
    }
    return out;
  }
  return value;
}

function registerFetchHttpClient() {
  const client = b.HttpClient.fromCallback(async (req: any) => {
    const headers = new Headers();
    for (const [k, v] of req.headers ?? []) headers.set(k, v);
    const res = await fetch(req.url, {
      method: req.method,
      headers,
      body: req.body ?? undefined,
    });
    const respHeaders: [string, string][] = [];
    res.headers.forEach((v, k) => respHeaders.push([k, v]));
    const body = new Uint8Array(await res.arrayBuffer());
    return { status: res.status, headers: respHeaders, body };
  });
  b.setDefaultHttpClient(client);
  return client;
}

// ---- probes ---------------------------------------------------------------

type Probe = {
  name: string;
  category: "sync" | "async" | "http";
  expect?: "pass" | "fail-known"; // fail-known = expected to fail today
  reason?: string; // when expect=fail-known
  run: (env: any) => unknown | Promise<unknown>;
};

const PROBES: Probe[] = [
  {
    name: "load.version",
    category: "sync",
    run: () => b.version(),
  },
  {
    name: "load.exports-shape",
    category: "sync",
    run: () => ({
      setDefaultHttpClient: typeof b.setDefaultHttpClient,
      HttpClient: typeof b.HttpClient,
      Workflow: typeof b.Workflow,
      Memory: typeof b.Memory,
      UpstashBackend: typeof b.UpstashBackend,
      HttpPeerClient: typeof b.HttpPeerClient,
      // Wave 6B exposes the JS-microtask drain function on the napi
      // module exports; the host wires it onto `globalThis` above so the
      // installed scheduler closure can find it.
      drain: typeof b.__blazenDrainAsyncQueue,
      globalDrain: typeof (globalThis as any).__blazenDrainAsyncQueue,
    }),
  },
  {
    name: "workflow.ctor",
    category: "sync",
    run: () => {
      const wf = new b.Workflow("probe");
      return { ok: typeof wf === "object" };
    },
  },
  {
    name: "workflow.addStep",
    category: "sync",
    run: () => {
      const wf = new b.Workflow("probe");
      wf.addStep("parse", ["blazen::StartEvent"], async (event: any) => ({
        type: "GreetEvent",
        name: event?.data?.name ?? "World",
      }));
      wf.addStep("greet", ["GreetEvent"], async (event: any) => ({
        type: "blazen::StopEvent",
        result: { greeting: `Hello, ${event.name}!` },
      }));
      return { steps: 2 };
    },
  },
  {
    name: "httpclient.fromCallback",
    category: "sync",
    run: () => {
      const client = b.HttpClient.fromCallback(async () => ({
        status: 204,
        headers: [],
        body: new Uint8Array(),
      }));
      return { ok: client !== null && client !== undefined };
    },
  },
  {
    name: "httpclient.setDefault",
    category: "sync",
    run: () => {
      const client = b.HttpClient.fromCallback(async () => ({
        status: 204,
        headers: [],
        body: new Uint8Array(),
      }));
      b.setDefaultHttpClient(client);
      return { ok: true };
    },
  },
  {
    name: "upstash.create",
    category: "sync",
    run: () => {
      if (!b.UpstashBackend?.create) {
        throw new Error("UpstashBackend.create not exported");
      }
      const backend = b.UpstashBackend.create(
        "https://example.upstash.io",
        "fake-token",
        "blazen:probe",
      );
      return { ok: backend !== null && backend !== undefined };
    },
  },
  {
    name: "httppeer.newHttp",
    category: "sync",
    run: () => {
      if (!b.HttpPeerClient?.newHttp) {
        throw new Error("HttpPeerClient.newHttp not exported");
      }
      const peer = b.HttpPeerClient.newHttp(
        "https://peer.example.com",
        "node-1",
      );
      return { ok: peer !== null && peer !== undefined };
    },
  },
  {
    name: "workflow.run",
    category: "async",
    run: async () => {
      const wf = new b.Workflow("greet");
      wf.addStep("parse", ["blazen::StartEvent"], async (event: any) => ({
        type: "GreetEvent",
        name: event?.data?.name ?? "World",
      }));
      wf.addStep("greet", ["GreetEvent"], async (event: any) => ({
        type: "blazen::StopEvent",
        result: { greeting: `Hello, ${event.name}!` },
      }));
      const result = await wf.run({ name: "Workers" });
      return mapToObject(result);
    },
  },
  {
    name: "pipeline.run",
    category: "async",
    run: async () => {
      if (!b.PipelineBuilder || !b.Stage) {
        return {
          skipped: `PipelineBuilder/Stage not exported (PipelineBuilder=${typeof b.PipelineBuilder}, Stage=${typeof b.Stage})`,
        };
      }
      // Build two single-step workflows and wrap each in a Stage.
      // Stage 1: doubles `value`.
      const wf1 = new b.Workflow("step1");
      wf1.addStep("double", ["blazen::StartEvent"], async (event: any) => ({
        type: "blazen::StopEvent",
        result: { doubled: (event?.data?.value ?? 0) * 2 },
      }));
      // Stage 2: adds 10 to `doubled`.
      const wf2 = new b.Workflow("step2");
      wf2.addStep("plus", ["blazen::StartEvent"], async (event: any) => ({
        type: "blazen::StopEvent",
        result: {
          doubled: event?.data?.doubled ?? 0,
          plus: (event?.data?.doubled ?? 0) + 10,
        },
      }));
      const pipeline = new b.PipelineBuilder("smoke")
        .stage(new b.Stage("step1", wf1))
        .stage(new b.Stage("step2", wf2))
        .build();
      const handler = await pipeline.start({ value: 5 });
      const result = await handler.result();
      return mapToObject(result);
    },
  },
  {
    name: "workflow.runStreaming",
    category: "async",
    run: async () => {
      const wf = new b.Workflow("greet-stream");
      wf.addStep("parse", ["blazen::StartEvent"], async (event: any) => ({
        type: "GreetEvent",
        name: event?.data?.name ?? "World",
      }));
      wf.addStep("greet", ["GreetEvent"], async (event: any) => ({
        type: "blazen::StopEvent",
        result: { greeting: `Hello, ${event.name}!` },
      }));
      if (typeof (wf as any).runStreaming !== "function") {
        return { skipped: "runStreaming not exported" };
      }
      // runStreaming(input, onEvent) — collect intermediate events then
      // return the final result alongside the captured event count.
      const events: unknown[] = [];
      const result = await (wf as any).runStreaming(
        { name: "Stream" },
        (event: unknown) => {
          if (events.length < 32) events.push(mapToObject(event));
        },
      );
      return {
        eventCount: events.length,
        result: mapToObject(result),
      };
    },
  },
  {
    name: "workflow.runWithHandler",
    category: "async",
    run: async () => {
      const wf = new b.Workflow("greet-handler");
      wf.addStep("parse", ["blazen::StartEvent"], async (event: any) => ({
        type: "GreetEvent",
        name: event?.data?.name ?? "World",
      }));
      wf.addStep("greet", ["GreetEvent"], async (event: any) => ({
        type: "blazen::StopEvent",
        result: { greeting: `Hello, ${event.name}!` },
      }));
      // runWithHandler exercises the per-step handler spawn path (different
      // from `run`'s direct event-loop spawn).
      if (typeof (wf as any).runWithHandler !== "function") {
        return { skipped: "runWithHandler not exported" };
      }
      const handler = await (wf as any).runWithHandler({ name: "Handler" });
      const result = await handler.result();
      return mapToObject(result);
    },
  },
  {
    name: "openai.viaBlazen",
    category: "http",
    run: async (env: any) => {
      if (!env?.OPENAI_API_KEY) return { skipped: "OPENAI_API_KEY not set" };
      registerFetchHttpClient();
      // Construct an OpenAI provider via from_options; route through our
      // registered HttpClient.
      if (!b.OpenAiProvider?.fromOptions) {
        throw new Error("OpenAiProvider.fromOptions not exported");
      }
      const provider = b.OpenAiProvider.fromOptions({
        apiKey: env.OPENAI_API_KEY,
      });
      // Minimal completion: pick a cheap model and ask for one token.
      const res = await provider.complete({
        model: "gpt-4o-mini",
        messages: [{ role: "user", content: "say 'hi'" }],
        maxTokens: 8,
      });
      return mapToObject(res);
    },
  },
];

// ---- HTTP dispatcher ------------------------------------------------------

async function listProbes(): Promise<Response> {
  return Response.json(
    PROBES.map((p) => ({
      name: p.name,
      category: p.category,
      expect: p.expect ?? "pass",
      reason: p.reason ?? null,
    })),
  );
}

async function runProbe(name: string, env: any): Promise<Response> {
  const probe = PROBES.find((p) => p.name === name);
  if (!probe) {
    return Response.json({ error: `unknown probe: ${name}` }, { status: 404 });
  }
  const t0 = Date.now();
  try {
    const result = await probe.run(env);
    return Response.json({
      name: probe.name,
      category: probe.category,
      status: "pass",
      durationMs: Date.now() - t0,
      result,
    });
  } catch (e: any) {
    return Response.json({
      name: probe.name,
      category: probe.category,
      status: "fail",
      durationMs: Date.now() - t0,
      error: String(e?.message ?? e),
      stack: e?.stack ?? null,
    });
  }
}

export default {
  async fetch(request: Request, env: any): Promise<Response> {
    const url = new URL(request.url);
    if (url.pathname === "/" || url.pathname === "/probes") {
      return listProbes();
    }
    if (url.pathname.startsWith("/run/")) {
      const name = decodeURIComponent(url.pathname.slice("/run/".length));
      return runProbe(name, env);
    }
    return new Response(
      "GET / for probe list, GET /run/<name> to run one",
      { status: 404 },
    );
  },
};
