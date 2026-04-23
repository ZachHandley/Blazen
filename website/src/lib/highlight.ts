import { createHighlighter, type Highlighter, type BundledLanguage } from "shiki";

export type CodeLang =
  | "rust"
  | "python"
  | "typescript"
  | "javascript"
  | "bash"
  | "shell"
  | "toml"
  | "json"
  | "yaml";

const THEME = "github-dark";
const LANGS: CodeLang[] = [
  "rust",
  "python",
  "typescript",
  "javascript",
  "bash",
  "shell",
  "toml",
  "json",
  "yaml",
];

let highlighterPromise: Promise<Highlighter> | null = null;

function getHighlighter(): Promise<Highlighter> {
  if (!highlighterPromise) {
    highlighterPromise = createHighlighter({
      themes: [THEME],
      langs: LANGS as BundledLanguage[],
    });
  }
  return highlighterPromise;
}

export async function highlight(code: string, lang: CodeLang): Promise<string> {
  const h = await getHighlighter();
  return h.codeToHtml(code, { lang, theme: THEME });
}
