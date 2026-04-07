//! Node wrapper for `Artifact` (typed inline content extracted from LLM output).

use napi_derive::napi;

use blazen_llm::Artifact;

/// A typed artifact extracted from or returned by a model.
///
/// SVG / code blocks / markdown / mermaid / html / latex / json / custom payloads
/// can be returned inline as text by an LLM. Use the `kind` field to discriminate,
/// then read the variant-specific fields. Fields not relevant to the variant are
/// `null`.
///
/// ## Example
/// ```ts
/// for (const art of response.artifacts) {
///   if (art.kind === "svg") {
///     renderSvg(art.content as string);
///   } else if (art.kind === "code_block") {
///     console.log(`${art.language}: ${art.content}`);
///   }
/// }
/// ```
#[napi(object)]
pub struct JsArtifact {
    /// Discriminator. One of: `"svg"`, `"code_block"`, `"markdown"`, `"mermaid"`,
    /// `"html"`, `"latex"`, `"json"`, `"custom"`.
    pub kind: String,
    /// The artifact's primary payload. Typically a string; for the `json`
    /// variant it is a parsed JSON value (object/array/scalar).
    pub content: serde_json::Value,
    /// Optional title for `svg` artifacts.
    pub title: Option<String>,
    /// Language hint for `code_block` artifacts.
    pub language: Option<String>,
    /// Filename hint for `code_block` artifacts.
    pub filename: Option<String>,
    /// Provider-specific metadata for `custom` artifacts.
    pub metadata: Option<serde_json::Value>,
    /// Inner `kind` tag for `custom` artifacts. Distinct from the top-level
    /// `kind` field (which equals `"custom"` for this variant).
    #[napi(js_name = "customKind")]
    pub custom_kind: Option<String>,
}

impl From<&Artifact> for JsArtifact {
    fn from(a: &Artifact) -> Self {
        match a {
            Artifact::Svg { content, title } => Self {
                kind: "svg".to_owned(),
                content: serde_json::Value::String(content.clone()),
                title: title.clone(),
                language: None,
                filename: None,
                metadata: None,
                custom_kind: None,
            },
            Artifact::CodeBlock {
                language,
                content,
                filename,
            } => Self {
                kind: "code_block".to_owned(),
                content: serde_json::Value::String(content.clone()),
                title: None,
                language: language.clone(),
                filename: filename.clone(),
                metadata: None,
                custom_kind: None,
            },
            Artifact::Markdown { content } => Self {
                kind: "markdown".to_owned(),
                content: serde_json::Value::String(content.clone()),
                title: None,
                language: None,
                filename: None,
                metadata: None,
                custom_kind: None,
            },
            Artifact::Mermaid { content } => Self {
                kind: "mermaid".to_owned(),
                content: serde_json::Value::String(content.clone()),
                title: None,
                language: None,
                filename: None,
                metadata: None,
                custom_kind: None,
            },
            Artifact::Html { content } => Self {
                kind: "html".to_owned(),
                content: serde_json::Value::String(content.clone()),
                title: None,
                language: None,
                filename: None,
                metadata: None,
                custom_kind: None,
            },
            Artifact::Latex { content } => Self {
                kind: "latex".to_owned(),
                content: serde_json::Value::String(content.clone()),
                title: None,
                language: None,
                filename: None,
                metadata: None,
                custom_kind: None,
            },
            Artifact::Json { content } => Self {
                kind: "json".to_owned(),
                content: content.clone(),
                title: None,
                language: None,
                filename: None,
                metadata: None,
                custom_kind: None,
            },
            Artifact::Custom {
                kind,
                content,
                metadata,
            } => Self {
                kind: "custom".to_owned(),
                content: serde_json::Value::String(content.clone()),
                title: None,
                language: None,
                filename: None,
                metadata: Some(metadata.clone()),
                custom_kind: Some(kind.clone()),
            },
        }
    }
}

impl From<Artifact> for JsArtifact {
    fn from(a: Artifact) -> Self {
        (&a).into()
    }
}
