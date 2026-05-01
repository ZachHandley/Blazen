//! Python wrappers for prompt template types.

use std::collections::HashMap;
use std::path::PathBuf;

use pyo3::prelude::*;
use pyo3::types::PyDict;
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pyclass_enum, gen_stub_pymethods};

use blazen_prompts::{PromptFile, PromptRegistry, PromptTemplate, TemplateRole};

use crate::error::BlazenException;
use crate::types::message::PyChatMessage;

// ---------------------------------------------------------------------------
// PromptException
// ---------------------------------------------------------------------------

pyo3::create_exception!(blazen, PromptException, BlazenException);

/// Register the prompt exception type on the module.
pub fn register_exceptions(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("PromptError", m.py().get_type::<PromptException>())?;
    Ok(())
}

/// Convert a [`blazen_prompts::PromptError`] into a [`PyErr`].
pub fn prompt_err(err: blazen_prompts::PromptError) -> PyErr {
    PromptException::new_err(err.to_string())
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// PyTemplateRole
// ---------------------------------------------------------------------------

/// The role for a prompt template.
///
/// Maps to the chat ``Role`` produced when the template is rendered.
#[gen_stub_pyclass_enum]
#[pyclass(name = "TemplateRole", eq, eq_int, from_py_object)]
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum PyTemplateRole {
    System,
    User,
    Assistant,
}

impl From<PyTemplateRole> for TemplateRole {
    fn from(role: PyTemplateRole) -> Self {
        match role {
            PyTemplateRole::System => Self::System,
            PyTemplateRole::User => Self::User,
            PyTemplateRole::Assistant => Self::Assistant,
        }
    }
}

impl From<TemplateRole> for PyTemplateRole {
    fn from(role: TemplateRole) -> Self {
        match role {
            TemplateRole::System => Self::System,
            TemplateRole::User => Self::User,
            TemplateRole::Assistant => Self::Assistant,
        }
    }
}

impl From<&TemplateRole> for PyTemplateRole {
    fn from(role: &TemplateRole) -> Self {
        match role {
            TemplateRole::System => Self::System,
            TemplateRole::User => Self::User,
            TemplateRole::Assistant => Self::Assistant,
        }
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
/// # Example
///
/// ```text
/// t = PromptTemplate("Hello {{name}}!", role=TemplateRole.User)
/// msg = t.render(name="Alice")
/// print(msg.content)  # "Hello Alice!"
/// ```
#[gen_stub_pyclass]
#[pyclass(name = "PromptTemplate", from_py_object)]
#[derive(Clone)]
pub struct PyPromptTemplate {
    pub(crate) inner: PromptTemplate,
}

#[gen_stub_pymethods]
#[pymethods]
impl PyPromptTemplate {
    /// Create a new prompt template.
    ///
    /// Args:
    ///     template: The template string with ``{{variable}}`` placeholders.
    ///     role: The chat role (``TemplateRole.System``, ``TemplateRole.User``,
    ///         or ``TemplateRole.Assistant``). Defaults to ``TemplateRole.User``.
    ///     name: A unique name for this template (defaults to "unnamed").
    ///     description: An optional description of this template.
    ///     version: The version string (defaults to "1.0").
    #[new]
    #[pyo3(signature = (template, *, role=None, name=None, description=None, version=None))]
    #[allow(clippy::unnecessary_wraps)]
    fn new(
        template: &str,
        role: Option<PyTemplateRole>,
        name: Option<&str>,
        description: Option<&str>,
        version: Option<&str>,
    ) -> PyResult<Self> {
        let parsed_role = role.unwrap_or(PyTemplateRole::User).into();
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
    ///     PromptError: If a required variable is missing.
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

    /// The chat role.
    #[getter]
    fn role(&self) -> PyTemplateRole {
        PyTemplateRole::from(&self.inner.role)
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
        let role = match self.inner.role {
            TemplateRole::System => "system",
            TemplateRole::User => "user",
            TemplateRole::Assistant => "assistant",
        };
        format!(
            "PromptTemplate(name='{}', role='{}', version='{}')",
            self.inner.name, role, self.inner.version,
        )
    }
}

// ---------------------------------------------------------------------------
// PyPromptFile
// ---------------------------------------------------------------------------

/// A serializable collection of prompt templates.
///
/// Mirrors the YAML/JSON file layout used by ``PromptRegistry.from_file``
/// and ``PromptRegistry.to_file``.
///
/// # Example
///
/// ```text
/// pf = PromptFile([template_a, template_b])
/// for t in pf.prompts:
///     registry.register(t.name, t)
/// ```
#[gen_stub_pyclass]
#[pyclass(name = "PromptFile", from_py_object)]
#[derive(Clone)]
pub struct PyPromptFile {
    pub(crate) inner: PromptFile,
}

#[gen_stub_pymethods]
#[pymethods]
impl PyPromptFile {
    /// Create a new ``PromptFile`` from a list of ``PromptTemplate`` objects.
    ///
    /// Args:
    ///     prompts: The prompt templates to include in this file.
    #[new]
    #[pyo3(signature = (prompts=None))]
    fn new(prompts: Option<Vec<PyPromptTemplate>>) -> Self {
        let prompts = prompts
            .map(|ts| ts.into_iter().map(|t| t.inner).collect())
            .unwrap_or_default();
        Self {
            inner: PromptFile { prompts },
        }
    }

    /// Re-extract cached variable lists on every contained template.
    ///
    /// Call after deserializing or after mutating template strings to
    /// refresh the cached variable index used by ``render``.
    fn init(&mut self) {
        self.inner.init();
    }

    /// The prompt templates contained in this file.
    #[getter]
    fn prompts(&self) -> Vec<PyPromptTemplate> {
        self.inner
            .prompts
            .iter()
            .map(|t| PyPromptTemplate { inner: t.clone() })
            .collect()
    }

    fn __repr__(&self) -> String {
        format!("PromptFile(prompts={})", self.inner.prompts.len())
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
/// # Example
///
/// ```text
/// registry = PromptRegistry()
/// registry.register("greet", PromptTemplate("Hello {{name}}!"))
/// msg = registry.render("greet", name="Alice")
/// print(msg.content)
/// ```
#[gen_stub_pyclass]
#[pyclass(name = "PromptRegistry", from_py_object)]
#[derive(Clone)]
pub struct PyPromptRegistry {
    inner: PromptRegistry,
}

#[gen_stub_pymethods]
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
    ///     PromptError: If no template with that name exists, or if a
    ///         required variable is missing.
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
    ///     PromptError: If the file cannot be read, the format is
    ///         unsupported, or parsing fails.
    #[staticmethod]
    fn from_file(path: PathBuf) -> PyResult<Self> {
        let registry = PromptRegistry::from_file(&path).map_err(prompt_err)?;
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
    ///     PromptError: If the directory cannot be read or any file fails
    ///         to parse.
    #[staticmethod]
    fn from_dir(path: PathBuf) -> PyResult<Self> {
        let registry = PromptRegistry::from_dir(&path).map_err(prompt_err)?;
        Ok(Self { inner: registry })
    }

    /// Save all registered prompts to a YAML or JSON file.
    ///
    /// The format is detected by file extension (``.yaml``/``.yml`` for YAML,
    /// ``.json`` for JSON).
    ///
    /// Args:
    ///     path: Path to the output file.
    ///
    /// Raises:
    ///     PromptError: If the file cannot be written, the format is
    ///         unsupported, or serialization fails.
    fn to_file(&self, path: PathBuf) -> PyResult<()> {
        self.inner.to_file(&path).map_err(prompt_err)
    }

    fn __repr__(&self) -> String {
        let names = self.inner.list();
        format!("PromptRegistry(templates={})", names.len())
    }
}
