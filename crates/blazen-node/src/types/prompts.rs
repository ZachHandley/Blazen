//! Prompt template and registry bindings for the Node.js SDK.
//!
//! Wraps `blazen-prompts` types for use from JavaScript/TypeScript.

use std::collections::HashMap;

use napi::bindgen_prelude::Result;
use napi_derive::napi;

use blazen_prompts::{PromptRegistry, PromptTemplate, TemplateRole};

use super::message::JsChatMessage;
use crate::error::to_napi_error;

// ---------------------------------------------------------------------------
// Options object for PromptTemplate constructor
// ---------------------------------------------------------------------------

/// Options for creating a `PromptTemplate`.
#[napi(object)]
pub struct PromptTemplateOptions {
    /// The chat role: "system", "user", or "assistant". Defaults to "user".
    pub role: Option<String>,
    /// A unique name for this template. Defaults to "unnamed".
    pub name: Option<String>,
    /// An optional description of this template.
    pub description: Option<String>,
    /// The version string. Defaults to "1.0".
    pub version: Option<String>,
}

// ---------------------------------------------------------------------------
// JsPromptTemplate
// ---------------------------------------------------------------------------

/// A reusable prompt template with `{{variable}}` placeholders.
///
/// Templates are rendered by replacing `{{var}}` placeholders with the
/// provided variables map.
///
/// ```javascript
/// const t = new PromptTemplate("Hello {{name}}!", { role: "user" });
/// const msg = t.render({ name: "Alice" });
/// console.log(msg.content); // "Hello Alice!"
/// ```
#[napi(js_name = "PromptTemplate")]
pub struct JsPromptTemplate {
    pub(crate) inner: PromptTemplate,
}

#[napi]
#[allow(
    clippy::must_use_candidate,
    clippy::missing_errors_doc,
    clippy::needless_pass_by_value
)]
impl JsPromptTemplate {
    /// Create a new prompt template.
    ///
    /// @param template - The template string with `{{variable}}` placeholders.
    /// @param options  - Optional configuration (role, name, description, version).
    #[napi(constructor)]
    pub fn new(template: String, options: Option<PromptTemplateOptions>) -> Result<Self> {
        let opts = options.unwrap_or(PromptTemplateOptions {
            role: None,
            name: None,
            description: None,
            version: None,
        });

        let role = parse_template_role(opts.role.as_deref().unwrap_or("user"))?;
        let name = opts.name.as_deref().unwrap_or("unnamed");

        let mut inner = PromptTemplate::new(name, role, &template);

        if let Some(v) = &opts.version {
            inner = inner.with_version(v.as_str());
        }
        if let Some(d) = &opts.description {
            inner = inner.with_description(d.as_str());
        }

        Ok(Self { inner })
    }

    /// The raw template string.
    #[napi(getter)]
    pub fn template(&self) -> String {
        self.inner.template.clone()
    }

    /// The chat role ("system", "user", or "assistant").
    #[napi(getter)]
    pub fn role(&self) -> &str {
        match &self.inner.role {
            TemplateRole::System => "system",
            TemplateRole::User => "user",
            TemplateRole::Assistant => "assistant",
        }
    }

    /// The template name.
    #[napi(getter)]
    pub fn name(&self) -> String {
        self.inner.name.clone()
    }

    /// The optional description.
    #[napi(getter)]
    pub fn description(&self) -> Option<String> {
        self.inner.description.clone()
    }

    /// The version string.
    #[napi(getter)]
    pub fn version(&self) -> String {
        self.inner.version.clone()
    }

    /// The sorted list of variable names in this template.
    #[napi(getter)]
    pub fn variables(&self) -> Vec<String> {
        self.inner.variables().to_vec()
    }

    /// Render the template with the given variables.
    ///
    /// @param variables - A map of variable name to value.
    /// @returns A `ChatMessage` with the rendered content and template's role.
    #[napi]
    pub fn render(&self, variables: HashMap<String, String>) -> Result<JsChatMessage> {
        let message = self.inner.render(&variables).map_err(to_napi_error)?;
        Ok(JsChatMessage { inner: message })
    }
}

// ---------------------------------------------------------------------------
// JsPromptRegistry
// ---------------------------------------------------------------------------

/// A versioned registry for prompt templates.
///
/// Organises templates by name and version, with convenient lookup,
/// rendering, and file I/O.
///
/// ```javascript
/// const registry = new PromptRegistry();
/// registry.register("greet", new PromptTemplate("Hello {{name}}!"));
/// const msg = registry.render("greet", { name: "Alice" });
/// console.log(msg.content);
/// ```
#[napi(js_name = "PromptRegistry")]
pub struct JsPromptRegistry {
    inner: PromptRegistry,
}

#[napi]
#[allow(
    clippy::must_use_candidate,
    clippy::missing_errors_doc,
    clippy::needless_pass_by_value,
    clippy::new_without_default
)]
impl JsPromptRegistry {
    /// Create a new empty prompt registry.
    #[napi(constructor)]
    pub fn new() -> Self {
        Self {
            inner: PromptRegistry::new(),
        }
    }

    /// Register a template under the given name.
    ///
    /// The template's internal name is updated to match the provided name.
    /// If a template with the same name and version already exists, it is
    /// replaced.
    ///
    /// @param name     - The name to register the template under.
    /// @param template - The `PromptTemplate` to register.
    #[napi]
    pub fn register(&mut self, name: String, template: &JsPromptTemplate) {
        let mut t = template.inner.clone();
        name.clone_into(&mut t.name);
        self.inner.register(t);
    }

    /// Get the latest version of a template by name.
    ///
    /// @param name - The template name.
    /// @returns The `PromptTemplate` or `null` if not found.
    #[napi]
    pub fn get(&self, name: String) -> Option<JsPromptTemplate> {
        self.inner
            .get(&name)
            .map(|t| JsPromptTemplate { inner: t.clone() })
    }

    /// Render the latest version of the named template with the given variables.
    ///
    /// @param name      - The template name.
    /// @param variables - A map of variable name to value.
    /// @returns A `ChatMessage` with the rendered content.
    #[napi]
    pub fn render(
        &self,
        name: String,
        variables: HashMap<String, String>,
    ) -> Result<JsChatMessage> {
        let message = self
            .inner
            .render(&name, &variables)
            .map_err(to_napi_error)?;
        Ok(JsChatMessage { inner: message })
    }

    /// List all registered template names (sorted).
    ///
    /// @returns An array of template name strings.
    #[napi]
    pub fn list(&self) -> Vec<String> {
        self.inner.list().into_iter().map(String::from).collect()
    }

    /// Load a registry from a YAML or JSON file.
    ///
    /// The file format is detected by extension (`.yaml`/`.yml` for YAML,
    /// `.json` for JSON).
    ///
    /// @param path - Path to the prompt file.
    /// @returns A new `PromptRegistry` with the loaded templates.
    #[napi(factory, js_name = "fromFile")]
    pub fn from_file(path: String) -> Result<Self> {
        let registry = PromptRegistry::from_file(&path).map_err(to_napi_error)?;
        Ok(Self { inner: registry })
    }

    /// Load all prompt files from a directory.
    ///
    /// Reads all `.yaml`, `.yml`, and `.json` files in the directory.
    ///
    /// @param path - Path to the directory.
    /// @returns A new `PromptRegistry` with the loaded templates.
    #[napi(factory, js_name = "fromDir")]
    pub fn from_dir(path: String) -> Result<Self> {
        let registry = PromptRegistry::from_dir(&path).map_err(to_napi_error)?;
        Ok(Self { inner: registry })
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Parse a role string into a [`TemplateRole`].
fn parse_template_role(role: &str) -> Result<TemplateRole> {
    match role {
        "system" => Ok(TemplateRole::System),
        "user" => Ok(TemplateRole::User),
        "assistant" => Ok(TemplateRole::Assistant),
        other => Err(napi::Error::new(
            napi::Status::InvalidArg,
            format!("unknown role: '{other}' (expected system, user, or assistant)"),
        )),
    }
}
