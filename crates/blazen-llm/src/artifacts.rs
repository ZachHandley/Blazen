//! Inline artifact extraction from LLM text content.
//!
//! Some providers return SVG, code blocks, mermaid diagrams, etc. inline as
//! text within the assistant content. [`extract_inline_artifacts`] is an
//! opt-in post-processor that scans a content string and lifts well-known
//! patterns into typed [`Artifact`] values.

use crate::types::Artifact;

/// Scan a string of LLM-generated text for inline artifacts.
///
/// Detects:
/// - Triple-backtick fenced code blocks with a language hint (`` ```python ... ``` ``).
///   `svg` and `mermaid` languages produce dedicated `Artifact::Svg`/`Artifact::Mermaid`
///   variants; everything else becomes `Artifact::CodeBlock { language, .. }`.
/// - Raw `<svg ...>...</svg>` runs that are not inside a code block.
///
/// Artifacts are returned in source order. Overlapping matches are deduplicated
/// by preferring the fenced-block form.
#[must_use]
pub fn extract_inline_artifacts(content: &str) -> Vec<Artifact> {
    let mut out = Vec::new();
    let mut consumed_ranges: Vec<(usize, usize)> = Vec::new();

    // 1) Fenced code blocks: ```<lang>?\n<body>\n```
    let bytes = content.as_bytes();
    let mut i = 0usize;
    while i + 3 <= bytes.len() {
        if &bytes[i..i + 3] == b"```" {
            // Find end of opening fence line.
            let lang_start = i + 3;
            let line_end = content[lang_start..]
                .find('\n')
                .map_or(bytes.len(), |p| lang_start + p);
            let lang_raw = content[lang_start..line_end].trim();
            let lang = if lang_raw.is_empty() {
                None
            } else {
                Some(lang_raw.to_owned())
            };
            let body_start = (line_end + 1).min(bytes.len());
            // Find closing fence at start of a line.
            let mut search_from = body_start;
            let close = loop {
                if let Some(pos) = content[search_from..].find("```") {
                    let abs = search_from + pos;
                    let at_line_start = abs == 0 || bytes[abs - 1] == b'\n';
                    if at_line_start {
                        break Some(abs);
                    }
                    search_from = abs + 3;
                } else {
                    break None;
                }
            };
            if let Some(close_pos) = close {
                let body_end = if close_pos > 0 && bytes[close_pos - 1] == b'\n' {
                    close_pos - 1
                } else {
                    close_pos
                };
                let body = &content[body_start..body_end];
                let block_end = close_pos + 3;
                consumed_ranges.push((i, block_end));
                let lang_lower = lang.as_deref().map(str::to_ascii_lowercase);
                match lang_lower.as_deref() {
                    Some("svg") => out.push(Artifact::Svg {
                        content: body.to_owned(),
                        title: None,
                    }),
                    Some("mermaid") => out.push(Artifact::Mermaid {
                        content: body.to_owned(),
                    }),
                    Some("html") => out.push(Artifact::Html {
                        content: body.to_owned(),
                    }),
                    Some("latex" | "tex") => out.push(Artifact::Latex {
                        content: body.to_owned(),
                    }),
                    Some("markdown" | "md") => out.push(Artifact::Markdown {
                        content: body.to_owned(),
                    }),
                    _ => out.push(Artifact::CodeBlock {
                        language: lang,
                        content: body.to_owned(),
                        filename: None,
                    }),
                }
                i = block_end;
                continue;
            }
        }
        i += 1;
    }

    // 2) Raw <svg ...>...</svg> runs that are not inside a consumed range.
    let mut search = 0usize;
    while let Some(open_off) = content[search..].find("<svg") {
        let abs_open = search + open_off;
        if consumed_ranges
            .iter()
            .any(|(s, e)| abs_open >= *s && abs_open < *e)
        {
            search = abs_open + 4;
            continue;
        }
        if let Some(close_off) = content[abs_open..].find("</svg>") {
            let abs_close = abs_open + close_off + "</svg>".len();
            out.push(Artifact::Svg {
                content: content[abs_open..abs_close].to_owned(),
                title: None,
            });
            search = abs_close;
        } else {
            break;
        }
    }

    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extract_svg_from_fenced_block() {
        let content = "Here:\n```svg\n<svg width=\"10\"/>\n```\nDone.";
        let arts = extract_inline_artifacts(content);
        assert_eq!(arts.len(), 1);
        assert!(
            matches!(&arts[0], Artifact::Svg { content, .. } if content == "<svg width=\"10\"/>")
        );
    }

    #[test]
    fn extract_mermaid_block() {
        let content = "```mermaid\ngraph TD; A-->B;\n```";
        let arts = extract_inline_artifacts(content);
        assert_eq!(arts.len(), 1);
        assert!(matches!(&arts[0], Artifact::Mermaid { content } if content == "graph TD; A-->B;"));
    }

    #[test]
    fn extract_raw_svg_tag() {
        let content = "Inline: <svg viewBox=\"0 0 1 1\"><rect/></svg> end.";
        let arts = extract_inline_artifacts(content);
        assert_eq!(arts.len(), 1);
        assert!(
            matches!(&arts[0], Artifact::Svg { content, .. } if content.starts_with("<svg") && content.ends_with("</svg>"))
        );
    }

    #[test]
    fn extract_code_block_with_language() {
        let content = "```python\nprint('hi')\n```";
        let arts = extract_inline_artifacts(content);
        assert_eq!(arts.len(), 1);
        assert!(
            matches!(&arts[0], Artifact::CodeBlock { language: Some(l), content, .. } if l == "python" && content == "print('hi')")
        );
    }
}
