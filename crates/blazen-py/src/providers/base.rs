//! Python wrapper for [`blazen_llm::BaseProvider`].
//!
//! Wraps any [`CompletionModel`] together with a
//! [`PyCompletionProviderDefaults`] that is applied to every completion
//! request before delegating. Phase A exposes the *structural* surface:
//! constructor, chainable builders, and read-only introspection
//! getters. The actual hook dispatch wiring (running user-supplied
//! coroutines for `before_request` / `before_completion`) is deferred
//! to Phase B; for now the defaults bag stores the Python callables
//! and propagates them through builders unchanged.
//!
//! Once Phase D wires up Python subclassing of `BaseProvider`, the
//! `inner` slot stays as the user-supplied [`PyCompletionModel`] and a
//! true `BlzBaseProvider` is built on demand.

use std::sync::Arc;

use pyo3::prelude::*;
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pymethods};

use blazen_llm::ChatMessage;
use blazen_llm::traits::CompletionModel;
use blazen_llm::types::CompletionRequest;

use crate::providers::completion_model::{PyCompletionModel, arc_from_bound};
use crate::providers::defaults::PyCompletionProviderDefaults;
use crate::types::{PyChatMessage, PyToolDefinition};

// ---------------------------------------------------------------------------
// PyBaseProvider
// ---------------------------------------------------------------------------

/// Wraps any [`CompletionModel`] with instance-level defaults that are
/// applied to every ``complete()`` / ``stream()`` call before delegation.
///
/// Phase A exposes the structural builder surface so foreign-language
/// bindings, audits, and Phase D subclass wiring have something to bind
/// against. The defaults bag is propagated verbatim; full hook dispatch
/// arrives in Phase B.
///
/// Example:
///     >>> from blazen import BaseProvider, CompletionModel
///     >>> inner = CompletionModel.openai()
///     >>> provider = (
///     ...     BaseProvider(inner)
///     ...     .with_system_prompt("be terse")
///     ...     .with_response_format({"type": "json_object"})
///     ... )
#[gen_stub_pyclass]
#[pyclass(name = "BaseProvider", subclass, from_py_object)]
pub struct PyBaseProvider {
    /// User-supplied completion model. Held as the original Py wrapper so
    /// the underlying `Arc<dyn CompletionModel>` (or subclass payload) can
    /// be re-extracted when Phase B/D actually constructs a Rust
    /// [`blazen_llm::BaseProvider`].
    pub(crate) inner: Py<PyCompletionModel>,
    /// The configured completion-role defaults. Cloned on every builder
    /// step so callers see Rust-style chainable, value-returning methods.
    pub(crate) defaults: PyCompletionProviderDefaults,
}

impl Clone for PyBaseProvider {
    fn clone(&self) -> Self {
        Self {
            inner: Python::attach(|py| self.inner.clone_ref(py)),
            defaults: self.defaults.clone(),
        }
    }
}

#[gen_stub_pymethods]
#[pymethods]
impl PyBaseProvider {
    /// Wrap a [`CompletionModel`] with instance-level defaults.
    ///
    /// Args:
    ///     inner: Any [`CompletionModel`] (built-in factory or subclass).
    ///     defaults: Optional [`CompletionProviderDefaults`]; defaults to
    ///         an empty bag.
    #[new]
    #[pyo3(signature = (inner, defaults=None))]
    fn new(inner: Py<PyCompletionModel>, defaults: Option<PyCompletionProviderDefaults>) -> Self {
        Self {
            inner,
            defaults: defaults.unwrap_or_default(),
        }
    }

    // -----------------------------------------------------------------
    // Chainable builders
    // -----------------------------------------------------------------

    /// Return a clone with ``system_prompt`` set on the completion defaults.
    fn with_system_prompt(&self, system_prompt: String) -> Self {
        let mut cloned = self.clone();
        cloned.defaults.system_prompt = Some(system_prompt);
        cloned
    }

    /// Return a clone with the default tool list replaced.
    fn with_tools(&self, tools: Vec<PyToolDefinition>) -> Self {
        let mut cloned = self.clone();
        cloned.defaults.tools = tools;
        cloned
    }

    /// Return a clone with the default ``response_format`` replaced.
    fn with_response_format(&self, response_format: Py<PyAny>) -> Self {
        let mut cloned = self.clone();
        cloned.defaults.response_format = Some(response_format);
        cloned
    }

    /// Return a clone with the universal JSON-level ``before_request`` hook
    /// set on the embedded base defaults.
    fn with_before_request(&self, hook: Py<PyAny>) -> Self {
        let mut cloned = self.clone();
        cloned.defaults.base.before_request = Some(hook);
        cloned
    }

    /// Return a clone with the typed ``before_completion`` hook set.
    fn with_before_completion(&self, hook: Py<PyAny>) -> Self {
        let mut cloned = self.clone();
        cloned.defaults.before_completion = Some(hook);
        cloned
    }

    /// Return a clone with the entire defaults bag replaced.
    fn with_defaults(&self, defaults: PyCompletionProviderDefaults) -> Self {
        let mut cloned = self.clone();
        cloned.defaults = defaults;
        cloned
    }

    // -----------------------------------------------------------------
    // Introspection
    // -----------------------------------------------------------------

    /// The currently configured defaults.
    #[getter]
    fn defaults(&self) -> PyCompletionProviderDefaults {
        self.defaults.clone()
    }

    /// The wrapped completion model's ``model_id``.
    #[getter]
    fn model_id(&self) -> String {
        Python::attach(|py| {
            let bound = self.inner.bind(py);
            // Re-use the public Python `model_id` getter so subclasses that
            // store their id in `config` resolve correctly.
            match bound.getattr("model_id") {
                Ok(attr) => attr.extract::<String>().unwrap_or_default(),
                Err(_) => String::new(),
            }
        })
    }

    /// The wrapped completion model's ``provider_id``, if it exposes one.
    ///
    /// Returns ``None`` for built-in providers that don't expose a
    /// ``provider_id`` attribute (the inner ``CompletionModel`` trait has no
    /// ``provider_id`` method --- it lives on the compute-side
    /// ``ComputeProvider`` trait).
    #[getter]
    fn provider_id(&self) -> Option<String> {
        Python::attach(|py| {
            let bound = self.inner.bind(py);
            match bound.getattr("provider_id") {
                Ok(attr) => {
                    if attr.is_none() {
                        None
                    } else {
                        attr.extract::<String>().ok()
                    }
                }
                Err(_) => None,
            }
        })
    }

    /// Return a fresh handle to the wrapped completion model.
    #[getter]
    fn inner(&self) -> Py<PyCompletionModel> {
        Python::attach(|py| self.inner.clone_ref(py))
    }

    fn __repr__(&self) -> String {
        format!("BaseProvider(model_id={:?})", self.model_id())
    }

    // -----------------------------------------------------------------
    // Typed structured extraction
    // -----------------------------------------------------------------

    /// Typed structured extraction.
    ///
    /// Calls ``complete()`` with a ``response_format`` derived from the
    /// supplied pydantic model's ``model_json_schema()`` and returns
    /// ``schema.model_validate(parsed_json_content)``.
    ///
    /// Args:
    ///     schema: A pydantic ``BaseModel`` subclass (the class, not an
    ///         instance). The class must expose ``model_json_schema()`` and
    ///         ``model_validate()`` (pydantic v2 convention).
    ///     messages: A list of [`ChatMessage`] objects forming the prompt.
    ///
    /// Returns:
    ///     A coroutine resolving to an instance of ``schema``.
    #[gen_stub(override_return_type(type_repr = "typing.Coroutine[typing.Any, typing.Any, typing.Any]", imports = ("typing",)))]
    fn extract<'py>(
        &self,
        py: Python<'py>,
        schema: Bound<'py, PyAny>,
        messages: Vec<PyRef<'py, PyChatMessage>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        // Step 1: derive the JSON schema dict from the pydantic class.
        let schema_dict = schema.call_method0("model_json_schema").map_err(|e| {
            pyo3::exceptions::PyTypeError::new_err(format!(
                "schema must be a pydantic BaseModel subclass exposing \
                     model_json_schema(): {e}"
            ))
        })?;
        let schema_json: serde_json::Value =
            crate::convert::py_to_json(py, &schema_dict).map_err(|e| {
                pyo3::exceptions::PyValueError::new_err(format!(
                    "failed to convert pydantic schema to JSON: {e}"
                ))
            })?;
        let schema_name: String = schema
            .getattr("__name__")
            .and_then(|n| n.extract())
            .unwrap_or_else(|_| "Extracted".to_owned());
        // Wrap as an OpenAI-style structured-output request.
        let response_format = serde_json::json!({
            "type": "json_schema",
            "json_schema": {
                "name": schema_name,
                "schema": schema_json,
                "strict": true,
            }
        });

        // Step 2: build the CompletionRequest with the response_format.
        let rust_messages: Vec<ChatMessage> = messages.iter().map(|m| m.inner.clone()).collect();
        let mut request =
            CompletionRequest::new(rust_messages).with_response_format(response_format);
        if let Some(prompt) = self.defaults.system_prompt.clone() {
            // Prepend a system message if none exists already. Matches the
            // behavior of `BaseProvider`'s defaults application path.
            let has_system = request
                .messages
                .first()
                .is_some_and(|m| matches!(m.role, blazen_llm::types::Role::System));
            if !has_system {
                let mut new_messages = Vec::with_capacity(request.messages.len() + 1);
                new_messages.push(ChatMessage::system(prompt));
                new_messages.append(&mut request.messages);
                request = CompletionRequest::new(new_messages)
                    .with_response_format(request.response_format.clone().unwrap_or_default());
            }
        }

        // Step 3: pull a concrete `Arc<dyn CompletionModel>` from the inner
        // `PyCompletionModel`. Handles built-in providers (direct trait
        // object) and subclasses (via `PySubclassCompletionModel`).
        let inner_bound = self.inner.bind(py);
        let model: Arc<dyn CompletionModel> = arc_from_bound(inner_bound);
        // Keep a handle to the schema class so we can invoke
        // `model_validate` once the response lands.
        let schema_owned: Py<PyAny> = schema.unbind();

        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let response = model
                .complete(request)
                .await
                .map_err(crate::error::blazen_error_to_pyerr)?;
            let content = response.content.clone().unwrap_or_default();
            // Parse JSON content into a serde Value.
            let parsed: serde_json::Value = serde_json::from_str(&content).map_err(|e| {
                pyo3::exceptions::PyValueError::new_err(format!(
                    "extract(): provider response was not valid JSON: {e}; raw content: {content}"
                ))
            })?;
            // Hand the parsed value to `schema.model_validate(...)`.
            Python::attach(|py| {
                let py_value = crate::convert::json_to_py(py, &parsed)?;
                let schema_bound = schema_owned.bind(py);
                let instance = schema_bound
                    .call_method1("model_validate", (py_value,))
                    .map_err(|e| {
                        pyo3::exceptions::PyValueError::new_err(format!(
                            "schema.model_validate() raised: {e}"
                        ))
                    })?;
                Ok(instance.unbind())
            })
        })
    }
}

// Silence "unused" warnings while Phase B/D have yet to wire the inner
// trait object directly into a Rust `BaseProvider`. The import path is
// kept so it surfaces in editor jump-to-definition.
#[allow(dead_code)]
fn _phase_b_anchor(_p: &PyBaseProvider) -> Option<Arc<dyn blazen_llm::CompletionModel>> {
    // Phase B/D will populate this from `_p.inner.bind(py).borrow().inner.clone()`
    // and construct a `blazen_llm::BaseProvider` for actual dispatch.
    None
}
