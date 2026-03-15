//! A versioned registry for prompt templates.
//!
//! The [`PromptRegistry`] organises templates by name and version, making it
//! easy to look up the latest version of a prompt or pin to a specific one.
//! It also supports batch rendering into a [`CompletionRequest`] and
//! loading/saving from YAML and JSON files.

use std::collections::HashMap;
use std::path::Path;

use blazen_llm::{ChatMessage, CompletionRequest};

use crate::error::PromptError;
use crate::format::PromptFile;
use crate::template::PromptTemplate;

// ---------------------------------------------------------------------------
// PromptRegistry
// ---------------------------------------------------------------------------

/// A collection of named, versioned prompt templates.
///
/// Templates are grouped by name, with each name potentially having multiple
/// versions. Versions are kept sorted lexicographically so that the "latest"
/// version is always the last entry.
///
/// # Examples
///
/// ```
/// use blazen_prompts::{PromptRegistry, PromptTemplate};
/// use blazen_prompts::template::TemplateRole;
///
/// let mut registry = PromptRegistry::new();
///
/// registry.register(
///     PromptTemplate::new("greet", TemplateRole::User, "Hello {{name}}!")
///         .with_version("1.0"),
/// );
/// registry.register(
///     PromptTemplate::new("greet", TemplateRole::User, "Hey {{name}}, welcome!")
///         .with_version("2.0"),
/// );
///
/// // get() returns the latest version (2.0)
/// let latest = registry.get("greet").unwrap();
/// assert_eq!(latest.version, "2.0");
/// ```
#[derive(Debug, Clone, Default)]
pub struct PromptRegistry {
    /// Maps prompt name to a list of versions, sorted by version string.
    prompts: HashMap<String, Vec<PromptTemplate>>,
}

impl PromptRegistry {
    /// Create an empty prompt registry.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a template in the registry.
    ///
    /// If a template with the same name and version already exists, it is
    /// replaced. Templates are kept sorted by version within each name group.
    pub fn register(&mut self, template: PromptTemplate) {
        let versions = self.prompts.entry(template.name.clone()).or_default();

        // Remove an existing entry with the same version, if any.
        versions.retain(|t| t.version != template.version);

        versions.push(template);
        versions.sort_by(|a, b| a.version.cmp(&b.version));
    }

    /// Get the latest version of a prompt by name.
    #[must_use]
    pub fn get(&self, name: &str) -> Option<&PromptTemplate> {
        self.prompts.get(name).and_then(|v| v.last())
    }

    /// Get a specific version of a prompt by name and version string.
    #[must_use]
    pub fn get_version(&self, name: &str, version: &str) -> Option<&PromptTemplate> {
        self.prompts
            .get(name)
            .and_then(|v| v.iter().find(|t| t.version == version))
    }

    /// Render the latest version of the named prompt with the given variables.
    ///
    /// # Errors
    ///
    /// Returns [`PromptError::NotFound`] if no prompt with that name exists,
    /// or [`PromptError::MissingVariable`] if a required variable is missing.
    pub fn render(
        &self,
        name: &str,
        vars: &HashMap<String, String>,
    ) -> Result<ChatMessage, PromptError> {
        let template = self.get(name).ok_or_else(|| PromptError::NotFound {
            name: name.to_owned(),
        })?;
        template.render(vars)
    }

    /// Render a specific version of the named prompt with the given variables.
    ///
    /// # Errors
    ///
    /// Returns [`PromptError::VersionNotFound`] if the version does not exist,
    /// or [`PromptError::MissingVariable`] if a required variable is missing.
    pub fn render_version(
        &self,
        name: &str,
        version: &str,
        vars: &HashMap<String, String>,
    ) -> Result<ChatMessage, PromptError> {
        let template =
            self.get_version(name, version)
                .ok_or_else(|| PromptError::VersionNotFound {
                    name: name.to_owned(),
                    version: version.to_owned(),
                })?;
        template.render(vars)
    }

    /// Build a [`CompletionRequest`] from a sequence of named prompts.
    ///
    /// Each element in `prompts` is a `(prompt_name, variables)` tuple. The
    /// latest version of each prompt is rendered and the resulting messages
    /// are collected into a single [`CompletionRequest`].
    ///
    /// # Errors
    ///
    /// Returns an error if any prompt is not found or has missing variables.
    pub fn build_request(
        &self,
        prompts: &[(&str, HashMap<String, String>)],
    ) -> Result<CompletionRequest, PromptError> {
        let mut messages = Vec::with_capacity(prompts.len());

        for (name, vars) in prompts {
            let message = self.render(name, vars)?;
            messages.push(message);
        }

        Ok(CompletionRequest::new(messages))
    }

    /// List all registered prompt names.
    #[must_use]
    pub fn list(&self) -> Vec<&str> {
        let mut names: Vec<&str> = self.prompts.keys().map(String::as_str).collect();
        names.sort_unstable();
        names
    }

    /// Load a registry from a YAML or JSON file.
    ///
    /// The file format is detected by extension (`.yaml`/`.yml` for YAML,
    /// `.json` for JSON).
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be read or parsed.
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self, PromptError> {
        let path = path.as_ref();
        let contents = std::fs::read_to_string(path)?;

        let mut prompt_file: PromptFile = match path.extension().and_then(|ext| ext.to_str()) {
            Some("yaml" | "yml") => serde_yaml::from_str(&contents)?,
            Some("json") => serde_json::from_str(&contents)?,
            _ => {
                return Err(PromptError::Validation(format!(
                    "unsupported file extension: {}",
                    path.display()
                )));
            }
        };

        prompt_file.init();

        let mut registry = Self::new();
        for template in prompt_file.prompts {
            registry.register(template);
        }
        Ok(registry)
    }

    /// Load all `.yaml`, `.yml`, and `.json` prompt files from a directory.
    ///
    /// Non-prompt files are silently ignored. Each file is expected to contain
    /// a [`PromptFile`] with a `prompts` array.
    ///
    /// # Errors
    ///
    /// Returns an error if the directory cannot be read or any prompt file
    /// fails to parse.
    pub fn from_dir(path: impl AsRef<Path>) -> Result<Self, PromptError> {
        let path = path.as_ref();
        let mut registry = Self::new();

        for entry in std::fs::read_dir(path)? {
            let entry = entry?;
            let file_path = entry.path();

            let is_prompt_file = file_path
                .extension()
                .and_then(|ext| ext.to_str())
                .is_some_and(|ext| matches!(ext, "yaml" | "yml" | "json"));

            if is_prompt_file {
                let loaded = Self::from_file(&file_path)?;
                for (_, versions) in loaded.prompts {
                    for template in versions {
                        registry.register(template);
                    }
                }
            }
        }

        Ok(registry)
    }

    /// Save all registered prompts to a YAML or JSON file.
    ///
    /// The format is detected by file extension (`.yaml`/`.yml` for YAML,
    /// `.json` for JSON).
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be written or serialization fails.
    pub fn to_file(&self, path: impl AsRef<Path>) -> Result<(), PromptError> {
        let path = path.as_ref();

        let mut all_templates: Vec<&PromptTemplate> =
            self.prompts.values().flat_map(|v| v.iter()).collect();
        all_templates.sort_by(|a, b| (&a.name, &a.version).cmp(&(&b.name, &b.version)));

        let prompt_file = PromptFile {
            prompts: all_templates.into_iter().cloned().collect(),
        };

        let contents = match path.extension().and_then(|ext| ext.to_str()) {
            Some("yaml" | "yml") => serde_yaml::to_string(&prompt_file)?,
            Some("json") => serde_json::to_string_pretty(&prompt_file)?,
            _ => {
                return Err(PromptError::Validation(format!(
                    "unsupported file extension: {}",
                    path.display()
                )));
            }
        };

        std::fs::write(path, contents)?;
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::template::TemplateRole;

    fn sample_registry() -> PromptRegistry {
        let mut registry = PromptRegistry::new();
        registry.register(
            PromptTemplate::new("greet", TemplateRole::User, "Hello {{name}}!").with_version("1.0"),
        );
        registry.register(
            PromptTemplate::new("greet", TemplateRole::User, "Hey {{name}}, welcome!")
                .with_version("2.0"),
        );
        registry
    }

    #[test]
    fn latest_version_is_returned() {
        let registry = sample_registry();
        let latest = registry.get("greet").unwrap();
        assert_eq!(latest.version, "2.0");
    }

    #[test]
    fn specific_version_lookup() {
        let registry = sample_registry();
        let v1 = registry.get_version("greet", "1.0").unwrap();
        assert_eq!(v1.version, "1.0");
        assert!(v1.template.contains("Hello"));
    }

    #[test]
    fn not_found_returns_none() {
        let registry = PromptRegistry::new();
        assert!(registry.get("nonexistent").is_none());
    }
}
