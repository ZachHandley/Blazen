import type { APIRoute } from "astro";
import { getCollection } from "astro:content";

const SECTION_ORDER = [
  "getting-started",
  "guides",
  "api",
  "examples",
] as const;

export const GET: APIRoute = async ({ site }) => {
  const docs = (await getCollection("docs")).sort((a, b) => {
    const sa = SECTION_ORDER.indexOf(a.data.section);
    const sb = SECTION_ORDER.indexOf(b.data.section);
    if (sa !== sb) return sa - sb;
    return a.data.order - b.data.order;
  });

  const origin = site?.toString().replace(/\/$/, "") ?? "https://blazen.dev";
  const parts: string[] = [
    "# Blazen — Full Documentation",
    "",
    "Concatenation of every Blazen documentation page. Each entry below names the source URL so you can cite specific pages.",
    "",
    `Companion index: ${origin}/llms.txt`,
    "",
  ];

  for (const d of docs) {
    parts.push("---");
    parts.push("");
    parts.push(`# ${d.data.title}`);
    parts.push("");
    parts.push(`Source: ${origin}/docs/${d.id}`);
    if (d.data.language) parts.push(`Language: ${d.data.language}`);
    parts.push(`Section: ${d.data.section}`);
    parts.push("");
    parts.push(d.body ?? "");
    parts.push("");
  }

  return new Response(parts.join("\n"), {
    headers: { "content-type": "text/plain; charset=utf-8" },
  });
};
