import { defineConfig } from "astro/config";
import tailwindcss from "@tailwindcss/vite";
import vue from "@astrojs/vue";
import sitemap from "@astrojs/sitemap";
import mdx from "@astrojs/mdx";

export default defineConfig({
  output: "static",
  site: "https://blazen.dev",
  markdown: {
    shikiConfig: {
      theme: "github-dark",
    },
  },
  vite: {
    plugins: [tailwindcss()],
  },
  integrations: [
    mdx(),
    vue({
      include: "src/components/vue/**/*.vue",
    }),
    sitemap(),
  ],
});
