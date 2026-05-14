import { defineConfig } from "astro/config";
import cloudflare from "@astrojs/cloudflare";

// IMPORTANT: this example deliberately ships with NO Vite externalize plugins,
// NO ssr.external lists, NO manualChunks, NO ssr.noExternal, NO
// optimizeDeps.exclude, NO resolve.conditions override.
//
// The whole point is to prove that
//   import { ... } from 'blazen/workers'
// Just Works in an Astro+Cloudflare bundle once Blazen ships an explicit
// `./workers` subpath that uses wrangler-style static .wasm imports. If you
// ever feel the need to add a workaround plugin here, the fix in Blazen has
// regressed -- the whole point of the subpath approach is to avoid all of
// that ceremony.
export default defineConfig({
  output: "server",
  adapter: cloudflare(),
});
