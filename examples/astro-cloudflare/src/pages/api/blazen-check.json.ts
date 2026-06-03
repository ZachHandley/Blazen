// Smoke test for the `blazen/workers` subpath entry on Cloudflare Workers.
//
// We import a representative spread across Blazen's surface area -- not just
// Model. Blazen ships ~400 napi exports covering providers, models,
// embeddings, image gen, audio/video gen, background removal, workflows,
// agents, document loaders, etc. The workers entry re-exports every one of
// them, identical to `blazen`'s normal entry; the only difference is the
// wasm loader.
//
// The route passes if all the imported symbols are `typeof 'function'`,
// which means:
//   1. `astro build` completed without bundler errors (no .node ELF parsing,
//      no commonjs--resolver fallback to wasi.cjs, no Invalid URL string).
//   2. `wrangler dev` booted without runtime errors at module init.
//   3. The napi module instantiated and `__napi_register__*` ran during
//      init, so each export resolves to a real class constructor.
import type { APIRoute } from "astro";
import {
  Model,
  EmbeddingModel,
  ImageModel,
  AnthropicProvider,
  BackgroundRemovalProvider,
  Workflow,
  AgentResult,
  EstimateCounter,
  TiktokenCounter,
} from "blazen/workers";

// Token-counter behaviour differs between the two wasi variants:
//   * lean `@blazen-dev/blazen-wasm32-wasi`           -> EstimateCounter works;
//       TiktokenCounter.forModel throws an actionable "install …-tiktoken" error.
//   * `@blazen-dev/blazen-wasm32-wasi-tiktoken`        -> both work; exact BPE counts.
// We report the observed result so the same endpoint validates either build.
function counterProbe() {
  const out: Record<string, unknown> = {};

  // EstimateCounter — always available on both builds.
  try {
    const est = new EstimateCounter(128_000);
    out.estimate = {
      ok: true,
      count: est.countTokens("Hello, world!"),
      contextSize: est.contextSize(),
    };
  } catch (e) {
    out.estimate = { ok: false, error: e instanceof Error ? e.message : String(e) };
  }

  // TiktokenCounter — throws on the lean build, returns a counter on the tiktoken build.
  try {
    const tk = TiktokenCounter.forModel("gpt-4o");
    out.tiktoken = { ok: true, count: tk.countTokens("Hello, world!") };
  } catch (e) {
    out.tiktoken = { ok: false, error: e instanceof Error ? e.message : String(e) };
  }

  return out;
}

export const GET: APIRoute = () =>
  new Response(
    JSON.stringify({
      Model: typeof Model,
      EmbeddingModel: typeof EmbeddingModel,
      ImageModel: typeof ImageModel,
      AnthropicProvider: typeof AnthropicProvider,
      BackgroundRemovalProvider: typeof BackgroundRemovalProvider,
      Workflow: typeof Workflow,
      AgentResult: typeof AgentResult,
      TiktokenCounter: typeof TiktokenCounter,
      EstimateCounter: typeof EstimateCounter,
      counters: counterProbe(),
    }),
    { headers: { "content-type": "application/json" } },
  );
