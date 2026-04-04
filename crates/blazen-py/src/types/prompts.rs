//! Python wrappers for prompt template types.

use std::collections::HashMap;

use pyo3::prelude::*;
use pyo3::types::PyDict;

use blazen_prompts::{PromptRegistry, PromptTemplate, TemplateRole};

use crate::types::message::PyChatMessage;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Parse a role string into a [`TemplateRole`], defaulting to `User`.
fn parse_role(role: Option<&str>) -> PyResult<TemplateRole> {
    match role {
        None | Some("user") => Ok(TemplateRole::User),
        Some("system") => Ok(TemplateRole::System),
        Some("assistant") => Ok(TemplateRole::Assistant),
        Some(other) => Err(pyo3::exceptions::PyValueError::new_err(format!(
            "unknown role: '{other}' (expected system, user, or assistant)"
        ))),
    }
}

/// Extract `**kwargs` from a Python dict into a `HashMap<String, String>`.
fn kwargs_to_map(kwargs: Option<&Bound<'_, PyDict>>) -> PyResult<HashMap<String, String>> {
    let Some(dict) = kwargs else {
        return Ok(HashMap::new());
    };
    let mut map = HashMap::with_capacity(dict.len());
    for (key, value) in dict {
        let k: String = key.extract()?;
        let v: String = value.extract().map_err(|_| {
            let ty_name = value
                .get_type()
                .name()
                .map_or_else(|_| "unknown".to_owned(), |n| n.to_string());
            pyo3::exceptions::PyTypeError::new_err(format!(
                "variable '{k}' must be a string, got {ty_name}",
            ))
        })?;
        map.insert(k, v);
    }
    Ok(map)
}

/// Convert a [`blazen_prompts::PromptError`] into a Python exception.
fn prompt_err(err: blazen_prompts::PromptError) -> PyErr {
    use blazen_prompts::PromptError;
    match &err {
        PromptError::NotFound { .. } | PromptError::VersionNotFound { .. } => {
            pyo3::exceptions::PyKeyError::new_err(err.to_string())
        }
        PromptError::Io(_) => pyo3::exceptions::PyIOError::new_err(err.to_string()),
        PromptError::MissingVariable { .. }
        | PromptError::Validation(_)
        | PromptError::Yaml(_)
        | PromptError::Json(_) => pyo3::exceptions::PyValueError::new_err(err.to_string()),
    }
}

// ---------------------------------------------------------------------------
// PyPromptTemplate
// ---------------------------------------------------------------------------

/// A reusable prompt template with ``{{variable}}`` placeholders.
///
/// Templates are rendered by replacing ``{{var}}`` placeholders with the
/// provided keyword arguments.
///
/// Example::
///
///     t = PromptTemplate("Hello {{name}}!", role="user")
///     msg = t.render(name="Alice")
///     print(msg.content)  # "Hello Alice!"
#[pyclass(name = "PromptTemplate", from_py_object)]
#[derive(Clone)]
pub struct PyPromptTemplate {
    pub(crate) inner: PromptTemplate,
}

#[pymethods]
impl PyPromptTemplate {
    /// Create a new prompt template.
    ///
    /// Args:
    ///     template: The template string with ``{{variable}}`` placeholders.
    ///     role: The chat role ("system", "user", or "assistant"). Defaults to "user".
    ///     name: A unique name for this template (defaults to "unnamed").
    ///     description: An optional description of this template.
    ///     version: The version string (defaults to "1.0").
    #[new]
    #[pyo3(signature = (template, *, role=None, name=None, description=None, version=None))]
    fn new(
        template: &str,
        role: Option<&str>,
        name: Option<&str>,
        description: Option<&str>,
        version: Option<&str>,
    ) -> PyResult<Self> {
        let parsed_role = parse_role(role)?;
        let template_name = name.unwrap_or("unnamed");

        let mut inner = PromptTemplate::new(template_name, parsed_role, template);

        if let Some(v) = version {
            inner = inner.with_version(v);
        }
        if let Some(d) = description {
            inner = inner.with_description(d);
        }

        Ok(Self { inner })
    }

    /// Render the template with the given keyword arguments.
    ///
    /// Returns a ``ChatMessage`` with the rendered content and the
    /// template's role.
    ///
    /// Args:
    ///     **variables: Template variables as keyword arguments.
    ///
    /// Raises:
    ///     ValueError: If a required variable is missing.
    #[pyo3(signature = (**variables))]
    fn render(&self, variables: Option<&Bound<'_, PyDict>>) -> PyResult<PyChatMessage> {
        let vars = kwargs_to_map(variables)?;
        let message = self.inner.render(&vars).map_err(prompt_err)?;
        Ok(PyChatMessage { inner: message })
    }

    /// The raw template string.
    #[getter]
    fn template(&self) -> &str {
        &self.inner.template
    }

    /// The chat role ("system", "user", or "assistant").
    #[getter]
    fn role(&self) -> &str {
        match &self.inner.role {
            TemplateRole::System => "system",
            TemplateRole::User => "user",
            TemplateRole::Assistant => "assistant",
        }
    }

    /// The template name.
    #[getter]
    fn name(&self) -> &str {
        &self.inner.name
    }

    /// The optional description.
    #[getter]
    fn description(&self) -> Option<&str> {
        self.inner.description.as_deref()
    }

    /// The version string.
    #[getter]
    fn version(&self) -> &str {
        &self.inner.version
    }

    /// The sorted list of variable names in this template.
    #[getter]
    fn variables(&self) -> Vec<String> {
        self.inner.variables().to_vec()
    }

    fn __repr__(&self) -> String {
        format!(
            "PromptTemplate(name='{}', role='{}', version='{}')",
            self.inner.name,
            self.role(),
            self.inner.version,
        )
    }
}

// ---------------------------------------------------------------------------
// PyPromptRegistry
// ---------------------------------------------------------------------------

/// A versioned registry for prompt templates.
///
/// Organises templates by name and version, with convenient lookup,
/// rendering, and file I/O.
///
/// Example::
///
///     registry = PromptRegistry()
///     registry.register("greet", PromptTemplate("Hello {{name}}!"))
///     msg = registry.render("greet", name="Alice")
///     print(msg.content)
#[pyclass(name = "PromptRegistry", from_py_object)]
#[derive(Clone)]
pub struct PyPromptRegistry {
    inner: PromptRegistry,
}

#[pymethods]
impl PyPromptRegistry {
    /// Create a new empty prompt registry.
    #[new]
    fn new() -> Self {
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
    /// Args:
    ///     name: The name to register the template under.
    ///     template: The ``PromptTemplate`` to register.
    fn register(&mut self, name: &str, template: &PyPromptTemplate) {
        let mut t = template.inner.clone();
        name.clone_into(&mut t.name);
        self.inner.register(t);
    }

    /// Get the latest version of a template by name.
    ///
    /// Args:
    ///     name: The template name.
    ///
    /// Returns:
    ///     The ``PromptTemplate`` or ``None`` if not found.
    fn get(&self, name: &str) -> Option<PyPromptTemplate> {
        self.inner
            .get(name)
            .map(|t| PyPromptTemplate { inner: t.clone() })
    }

    /// Render the latest version of the named template.
    ///
    /// Args:
    ///     name: The template name.
    ///     **variables: Template variables as keyword arguments.
    ///
    /// Returns:
    ///     A ``ChatMessage`` with the rendered content.
    ///
    /// Raises:
    ///     KeyError: If no template with that name exists.
    ///     ValueError: If a required variable is missing.
    #[pyo3(signature = (name, **variables))]
    fn render(&self, name: &str, variables: Option<&Bound<'_, PyDict>>) -> PyResult<PyChatMessage> {
        let vars = kwargs_to_map(variables)?;
        let message = self.inner.render(name, &vars).map_err(prompt_err)?;
        Ok(PyChatMessage { inner: message })
    }

    /// List all registered template names (sorted).
    ///
    /// Returns:
    ///     A list of template name strings.
    fn list(&self) -> Vec<String> {
        self.inner.list().into_iter().map(String::from).collect()
    }

    /// Load a registry from a YAML or JSON file.
    ///
    /// The file format is detected by extension (``.yaml``/``.yml`` for YAML,
    /// ``.json`` for JSON).
    ///
    /// Args:
    ///     path: Path to the prompt file.
    ///
    /// Returns:
    ///     A new ``PromptRegistry`` with the loaded templates.
    ///
    /// Raises:
    ///     IOError: If the file cannot be read.
    ///     ValueError: If the file format is unsupported or parsing fails.
    #[staticmethod]
    fn from_file(path: &str) -> PyResult<Self> {
        let registry = PromptRegistry::from_file(path).map_err(prompt_err)?;
        Ok(Self { inner: registry })
    }

    /// Load all prompt files from a directory.
    ///
    /// Reads all ``.yaml``, ``.yml``, and ``.json`` files in the directory.
    ///
    /// Args:
    ///     path: Path to the directory.
    ///
    /// Returns:
    ///     A new ``PromptRegistry`` with the loaded templates.
    ///
    /// Raises:
    ///     IOError: If the directory cannot be read.
    ///     ValueError: If any file fails to parse.
    #[staticmethod]
    fn from_dir(path: &str) -> PyResult<Self> {
        let registry = PromptRegistry::from_dir(path).map_err(prompt_err)?;
        Ok(Self { inner: registry })
    }

    fn __repr__(&self) -> String {
        let names = self.inner.list();
        format!("PromptRegistry(templates={})", names.len())
    }
}
