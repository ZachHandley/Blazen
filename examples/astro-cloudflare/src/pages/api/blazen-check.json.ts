// Smoke test for the `blazen/workers` subpath entry on Cloudflare Workers.
//
// We import a representative spread across Blazen's surface area -- not just
// CompletionModel. Blazen ships ~400 napi exports covering providers, models,
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
  CompletionModel,
  EmbeddingModel,
  ImageModel,
  AnthropicProvider,
  BackgroundRemovalProvider,
  Workflow,
  AgentResult,
} from "blazen/workers";

export const GET: APIRoute = () =>
  new Response(
    JSON.stringify({
      CompletionModel: typeof CompletionModel,
      EmbeddingModel: typeof EmbeddingModel,
      ImageModel: typeof ImageModel,
      AnthropicProvider: typeof AnthropicProvider,
      BackgroundRemovalProvider: typeof BackgroundRemovalProvider,
      Workflow: typeof Workflow,
      AgentResult: typeof AgentResult,
    }),
    { headers: { "content-type": "application/json" } },
  );
