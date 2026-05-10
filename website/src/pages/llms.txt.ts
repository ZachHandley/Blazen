import type { APIRoute } from "astro";
import { getCollection } from "astro:content";

const SECTION_ORDER = [
  "getting-started",
  "guides",
  "api",
  "examples",
] as const;

const SECTION_TITLES: Record<(typeof SECTION_ORDER)[number], string> = {
  "getting-started": "Getting Started",
  guides: "Guides",
  api: "API Reference",
  examples: "Examples",
};

export const GET: APIRoute = async ({ site }) => {
  const docs = (await getCollection("docs")).sort((a, b) => {
    const sa = SECTION_ORDER.indexOf(a.data.section);
    const sb = SECTION_ORDER.indexOf(b.data.section);
    if (sa !== sb) return sa - sb;
    return a.data.order - b.data.order;
  });

  const origin = site?.toString().replace(/\/$/, "") ?? "https://blazen.dev";
  const lines: string[] = [
    "# Blazen",
    "",
    "> Rust-first LLM orchestration framework with Python, Node, and WASM bindings. Composable workflows, typed events, durable pause/resume, distributed sub-workflows, and a unified provider layer.",
    "",
    "Companion files for LLMs:",
    "",
    `- [Full documentation](${origin}/llms-full.txt) — every page concatenated`,
    `- [Python slice](${origin}/llms-python.txt) — Python-specific docs plus cross-cutting guides`,
    `- [Node slice](${origin}/llms-node.txt) — Node-specific docs plus cross-cutting guides`,
    `- [Rust slice](${origin}/llms-rust.txt) — Rust-specific docs plus cross-cutting guides`,
    `- [WASM slice](${origin}/llms-wasm.txt) — WASM-specific docs plus cross-cutting guides`,
    "",
  ];

  for (const section of SECTION_ORDER) {
    const inSection = docs.filter((d) => d.data.section === section);
    if (inSection.length === 0) continue;
    lines.push(`## ${SECTION_TITLES[section]}`);
    lines.push("");
    for (const d of inSection) {
      const lang = d.data.language ? ` (${d.data.language})` : "";
      lines.push(
        `- [${d.data.title}${lang}](${origin}/docs/${d.id}): ${d.data.description}`,
      );
    }
    lines.push("");
  }

  return new Response(lines.join("\n"), {
    headers: { "content-type": "text/plain; charset=utf-8" },
  });
};
