import { defineConfig } from "astro/config";
import tailwindcss from "@tailwindcss/vite";
import vue from "@astrojs/vue";
import sitemap from "@astrojs/sitemap";

export default defineConfig({
  output: "static",
  site: "https://blazen.dev",
  vite: {
    plugins: [tailwindcss()],
  },
  integrations: [
    vue({
      include: "src/components/vue/**/*.vue",
    }),
    sitemap(),
  ],
});
