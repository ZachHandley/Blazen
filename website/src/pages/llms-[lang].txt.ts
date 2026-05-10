import type { APIRoute, GetStaticPaths } from "astro";
import { getCollection } from "astro:content";

const LANGS = ["python", "node", "rust", "wasm"] as const;
type Lang = (typeof LANGS)[number];

const LANG_TITLES: Record<Lang, string> = {
  python: "Python",
  node: "Node",
  rust: "Rust",
  wasm: "WASM",
};

const SECTION_ORDER = [
  "getting-started",
  "guides",
  "api",
  "examples",
] as const;

export const getStaticPaths: GetStaticPaths = () =>
  LANGS.map((lang) => ({ params: { lang } }));

export const GET: APIRoute = async ({ params, site }) => {
  const lang = params.lang as Lang;
  const all = await getCollection("docs");
  const docs = all
    .filter(
      (d) => d.data.language === lang || d.data.language === undefined,
    )
    .sort((a, b) => {
      const sa = SECTION_ORDER.indexOf(a.data.section);
      const sb = SECTION_ORDER.indexOf(b.data.section);
      if (sa !== sb) return sa - sb;
      return a.data.order - b.data.order;
    });

  const origin = site?.toString().replace(/\/$/, "") ?? "https://blazen.dev";
  const parts: string[] = [
    `# Blazen — ${LANG_TITLES[lang]} Documentation`,
    "",
    `Concatenation of every ${LANG_TITLES[lang]}-tagged doc plus cross-cutting docs that apply to every SDK.`,
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
