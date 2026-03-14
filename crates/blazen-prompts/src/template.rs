//! Prompt templates with `{{variable}}` placeholder interpolation.
//!
//! A [`PromptTemplate`] holds a template string with `{{variable}}` placeholders
//! that are replaced at render time. Variables are extracted eagerly on
//! construction and cached for fast lookups.

use std::collections::HashMap;

use blazen_llm::{ChatMessage, MessageContent, Role};
use regex::Regex;
use serde::{Deserialize, Serialize};

use crate::error::PromptError;

// ---------------------------------------------------------------------------
// TemplateRole
// ---------------------------------------------------------------------------

/// The role for a prompt template, mapping to [`blazen_llm::Role`].
///
/// Serializes as a lowercase string (`"system"`, `"user"`, `"assistant"`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum TemplateRole {
    /// A system-level instruction.
    System,
    /// A message from the user.
    User,
    /// A message from the assistant.
    Assistant,
}

impl From<TemplateRole> for Role {
    fn from(role: TemplateRole) -> Self {
        match role {
            TemplateRole::System => Self::System,
            TemplateRole::User => Self::User,
            TemplateRole::Assistant => Self::Assistant,
        }
    }
}

impl From<&TemplateRole> for Role {
    fn from(role: &TemplateRole) -> Self {
        match role {
            TemplateRole::System => Self::System,
            TemplateRole::User => Self::User,
            TemplateRole::Assistant => Self::Assistant,
        }
    }
}

// ---------------------------------------------------------------------------
// PromptTemplate
// ---------------------------------------------------------------------------

/// A reusable prompt template with `{{variable}}` placeholders.
///
/// Templates are constructed with a name, role, and template string. Variable
/// names are automatically extracted from `{{var_name}}` patterns in the
/// template text and cached for validation during rendering.
///
/// # Examples
///
/// ```
/// use std::collections::HashMap;
/// use blazen_prompts::PromptTemplate;
/// use blazen_prompts::template::TemplateRole;
///
/// let template = PromptTemplate::new(
///     "greeting",
///     TemplateRole::User,
///     "Hello {{name}}, welcome to {{place}}!",
/// );
///
/// let mut vars = HashMap::new();
/// vars.insert("name".to_owned(), "Alice".to_owned());
/// vars.insert("place".to_owned(), "Wonderland".to_owned());
///
/// let message = template.render(&vars).unwrap();
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PromptTemplate {
    /// The unique name of this template.
    pub name: String,

    /// The version of this template (e.g. `"1.0"`, `"2.0"`).
    #[serde(default = "default_version")]
    pub version: String,

    /// The chat role this template produces.
    pub role: TemplateRole,

    /// The template string containing `{{variable}}` placeholders.
    pub template: String,

    /// Cached variable names extracted from the template (sorted, deduplicated).
    #[serde(skip)]
    variables: Vec<String>,

    /// An optional human-readable description of the template's purpose.
    #[serde(default)]
    pub description: Option<String>,

    /// Arbitrary key-value metadata attached to the template.
    #[serde(default)]
    pub metadata: HashMap<String, String>,
}

fn default_version() -> String {
    "1.0".to_owned()
}

/// Extract sorted, deduplicated variable names from a template string.
fn extract_variables(template: &str) -> Vec<String> {
    let re = Regex::new(r"\{\{(\w+)\}\}").expect("regex is valid");
    let mut vars: Vec<String> = re
        .captures_iter(template)
        .map(|cap| cap[1].to_owned())
        .collect();
    vars.sort();
    vars.dedup();
    vars
}

impl PromptTemplate {
    /// Create a new prompt template.
    ///
    /// Variable names are automatically extracted from `{{var_name}}` patterns
    /// in the template string.
    #[must_use]
    pub fn new(name: impl Into<String>, role: TemplateRole, template: impl Into<String>) -> Self {
        let name = name.into();
        let template = template.into();
        let variables = extract_variables(&template);

        Self {
            name,
            version: default_version(),
            role,
            template,
            variables,
            description: None,
            metadata: HashMap::new(),
        }
    }

    /// Set the version string (builder pattern).
    #[must_use]
    pub fn with_version(mut self, version: impl Into<String>) -> Self {
        self.version = version.into();
        self
    }

    /// Set the description (builder pattern).
    #[must_use]
    pub fn with_description(mut self, description: impl Into<String>) -> Self {
        self.description = Some(description.into());
        self
    }

    /// Add a metadata key-value pair (builder pattern).
    #[must_use]
    pub fn with_metadata(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }

    /// Return the sorted, deduplicated list of variable names in this template.
    #[must_use]
    pub fn variables(&self) -> &[String] {
        &self.variables
    }

    /// Populate the cached variables field from the template string.
    ///
    /// This must be called after deserialization since the `variables` field is
    /// `#[serde(skip)]`.
    pub fn init(&mut self) {
        self.variables = extract_variables(&self.template);
    }

    /// Render the template by substituting all `{{var}}` placeholders with the
    /// provided values.
    ///
    /// Returns a [`ChatMessage`] with the appropriate role and rendered content.
    ///
    /// # Errors
    ///
    /// Returns [`PromptError::MissingVariable`] if any variable referenced in
    /// the template is not present in `vars`.
    pub fn render(&self, vars: &HashMap<String, String>) -> Result<ChatMessage, PromptError> {
        let mut output = self.template.clone();

        for var_name in &self.variables {
            let Some(value) = vars.get(var_name) else {
                return Err(PromptError::MissingVariable {
                    template: self.name.clone(),
                    name: var_name.clone(),
                });
            };
            let placeholder = format!("{{{{{var_name}}}}}");
            output = output.replace(&placeholder, value);
        }

        Ok(ChatMessage {
            role: Role::from(&self.role),
            content: MessageContent::Text(output),
        })
    }

    /// Convenience wrapper around [`render`](Self::render) that accepts a slice
    /// of `(&str, &str)` tuples instead of a `HashMap`.
    ///
    /// # Errors
    ///
    /// Returns [`PromptError::MissingVariable`] if any variable referenced in
    /// the template is not present in the provided tuples.
    pub fn render_with(&self, vars: &[(&str, &str)]) -> Result<ChatMessage, PromptError> {
        let map: HashMap<String, String> = vars
            .iter()
            .map(|(k, v)| ((*k).to_owned(), (*v).to_owned()))
            .collect();
        self.render(&map)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extract_vars_basic() {
        let vars = extract_variables("Hello {{name}}, you are {{age}} years old.");
        assert_eq!(vars, vec!["age", "name"]);
    }

    #[test]
    fn extract_vars_dedup() {
        let vars = extract_variables("{{x}} and {{x}} again");
        assert_eq!(vars, vec!["x"]);
    }

    #[test]
    fn extract_vars_empty() {
        let vars = extract_variables("No variables here.");
        assert!(vars.is_empty());
    }

    #[test]
    fn template_role_conversion() {
        assert_eq!(Role::from(TemplateRole::System), Role::System);
        assert_eq!(Role::from(TemplateRole::User), Role::User);
        assert_eq!(Role::from(TemplateRole::Assistant), Role::Assistant);
    }
}
