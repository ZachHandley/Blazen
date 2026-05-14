//! Python wrappers for the provider-defaults hierarchy.
//!
//! Mirrors the Rust types in
//! [`blazen_llm::providers::defaults`]: `BaseProviderDefaults` is the
//! universal cross-role bag (currently just the JSON-level `before_request`
//! hook); each role-specific defaults type (`CompletionProviderDefaults`,
//! `AudioSpeechProviderDefaults`, ...) embeds a `base` field plus its
//! typed `before_*` hook.
//!
//! Phase A scope: STRUCTURAL surface only. The before-hooks are stored
//! verbatim as `Option<Py<PyAny>>` -- no dispatch wiring. Phase B is
//! responsible for actually awaiting the user-supplied coroutines and
//! converting them to the Rust `BeforeRequestHook` / typed callbacks.
//!
//! All public classes carry `#[gen_stub_pyclass]` + `#[gen_stub_pymethods]`
//! so `cargo run --example stub_gen -p blazen-py` captures them in
//! `blazen.pyi`.

use pyo3::prelude::*;
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pymethods};

use crate::types::PyToolDefinition;

// ---------------------------------------------------------------------------
// PyBaseProviderDefaults
// ---------------------------------------------------------------------------

/// Universal provider defaults applicable to any provider role.
///
/// Phase A carries only the JSON-level ``before_request`` hook --- a
/// coroutine that receives the Rust method name (e.g. ``"complete"`` /
/// ``"text_to_speech"``) and a mutable ``dict`` view of the request and
/// may mutate it in place before downstream typed processing.
///
/// Example:
///     >>> async def stamp(method, body):
///     ...     body["trace_id"] = "abc"
///     >>> base = BaseProviderDefaults(before_request=stamp)
#[gen_stub_pyclass]
#[pyclass(name = "BaseProviderDefaults", subclass, from_py_object)]
#[derive(Default)]
pub struct PyBaseProviderDefaults {
    /// Optional Python coroutine factory. Phase B will dispatch this; Phase A
    /// only carries the value through the structural surface.
    pub(crate) before_request: Option<Py<PyAny>>,
}

impl Clone for PyBaseProviderDefaults {
    fn clone(&self) -> Self {
        Self {
            before_request: self
                .before_request
                .as_ref()
                .map(|h| Python::attach(|py| h.clone_ref(py))),
        }
    }
}

#[gen_stub_pymethods]
#[pymethods]
impl PyBaseProviderDefaults {
    /// Construct a new defaults bag.
    ///
    /// Args:
    ///     before_request: Optional ``async def(method: str, body: dict) -> None``
    ///         callable invoked before every provider request. Stored only in
    ///         Phase A; Phase B will wire it through to actual dispatch.
    #[new]
    #[pyo3(signature = (before_request=None))]
    fn new(before_request: Option<Py<PyAny>>) -> Self {
        Self { before_request }
    }

    /// The configured ``before_request`` hook, if any.
    #[getter]
    fn before_request(&self) -> Option<Py<PyAny>> {
        self.before_request
            .as_ref()
            .map(|h| Python::attach(|py| h.clone_ref(py)))
    }

    #[setter]
    fn set_before_request(&mut self, value: Option<Py<PyAny>>) {
        self.before_request = value;
    }

    fn __repr__(&self) -> String {
        format!(
            "BaseProviderDefaults(has_before_request={})",
            self.before_request.is_some()
        )
    }
}

// ---------------------------------------------------------------------------
// PyCompletionProviderDefaults
// ---------------------------------------------------------------------------

/// Completion-role defaults. Carries the universal ``base`` bag plus
/// completion-specific fields: ``system_prompt``, default ``tools``,
/// default ``response_format``, and the typed ``before_completion``
/// hook.
///
/// All fields are read/write so Python users can tweak them after
/// construction.
///
/// Example:
///     >>> async def add_user(req):
///     ...     req["metadata"]["origin"] = "blazen-py"
///     >>> defaults = CompletionProviderDefaults(
///     ...     system_prompt="be terse",
///     ...     tools=[my_tool],
///     ...     response_format={"type": "json_object"},
///     ...     before_completion=add_user,
///     ... )
#[gen_stub_pyclass]
#[pyclass(name = "CompletionProviderDefaults", subclass, from_py_object)]
#[derive(Default)]
pub struct PyCompletionProviderDefaults {
    pub(crate) base: PyBaseProviderDefaults,
    pub(crate) system_prompt: Option<String>,
    pub(crate) tools: Vec<PyToolDefinition>,
    pub(crate) response_format: Option<Py<PyAny>>,
    pub(crate) before_completion: Option<Py<PyAny>>,
}

impl Clone for PyCompletionProviderDefaults {
    fn clone(&self) -> Self {
        Self {
            base: self.base.clone(),
            system_prompt: self.system_prompt.clone(),
            tools: self.tools.clone(),
            response_format: self
                .response_format
                .as_ref()
                .map(|v| Python::attach(|py| v.clone_ref(py))),
            before_completion: self
                .before_completion
                .as_ref()
                .map(|h| Python::attach(|py| h.clone_ref(py))),
        }
    }
}

#[gen_stub_pymethods]
#[pymethods]
impl PyCompletionProviderDefaults {
    /// Construct completion defaults.
    ///
    /// Args:
    ///     base: Universal [`BaseProviderDefaults`]; defaults to an empty bag.
    ///     system_prompt: Default system message prepended when the request
    ///         has none.
    ///     tools: Default tool definitions appended to the request's tools
    ///         list (request entries win on name collision).
    ///     response_format: Default JSON-schema dict applied when the request
    ///         has no ``response_format``.
    ///     before_completion: Optional ``async def(request: dict) -> None``
    ///         typed hook applied after the universal ``before_request``.
    #[new]
    #[pyo3(signature = (base=None, system_prompt=None, tools=None, response_format=None, before_completion=None))]
    fn new(
        base: Option<PyBaseProviderDefaults>,
        system_prompt: Option<String>,
        tools: Option<Vec<PyToolDefinition>>,
        response_format: Option<Py<PyAny>>,
        before_completion: Option<Py<PyAny>>,
    ) -> Self {
        Self {
            base: base.unwrap_or_default(),
            system_prompt,
            tools: tools.unwrap_or_default(),
            response_format,
            before_completion,
        }
    }

    /// The universal base defaults bag.
    #[getter]
    fn base(&self) -> PyBaseProviderDefaults {
        self.base.clone()
    }

    #[setter]
    fn set_base(&mut self, value: PyBaseProviderDefaults) {
        self.base = value;
    }

    /// Default system prompt, if set.
    #[getter]
    fn system_prompt(&self) -> Option<String> {
        self.system_prompt.clone()
    }

    #[setter]
    fn set_system_prompt(&mut self, value: Option<String>) {
        self.system_prompt = value;
    }

    /// Default tool definitions.
    #[getter]
    fn tools(&self) -> Vec<PyToolDefinition> {
        self.tools.clone()
    }

    #[setter]
    fn set_tools(&mut self, value: Vec<PyToolDefinition>) {
        self.tools = value;
    }

    /// Default response-format dict, if set.
    #[getter]
    fn response_format(&self) -> Option<Py<PyAny>> {
        self.response_format
            .as_ref()
            .map(|v| Python::attach(|py| v.clone_ref(py)))
    }

    #[setter]
    fn set_response_format(&mut self, value: Option<Py<PyAny>>) {
        self.response_format = value;
    }

    /// Typed completion-level ``before_completion`` hook, if set.
    #[getter]
    fn before_completion(&self) -> Option<Py<PyAny>> {
        self.before_completion
            .as_ref()
            .map(|h| Python::attach(|py| h.clone_ref(py)))
    }

    #[setter]
    fn set_before_completion(&mut self, value: Option<Py<PyAny>>) {
        self.before_completion = value;
    }

    fn __repr__(&self) -> String {
        format!(
            "CompletionProviderDefaults(has_system_prompt={}, tools_len={}, has_response_format={}, has_before_completion={})",
            self.system_prompt.is_some(),
            self.tools.len(),
            self.response_format.is_some(),
            self.before_completion.is_some(),
        )
    }
}

// ---------------------------------------------------------------------------
// PyEmbeddingProviderDefaults
// ---------------------------------------------------------------------------

/// Embedding-role defaults. Currently only carries the universal ``base``
/// bag (V1 has no embedding-specific hook).
#[gen_stub_pyclass]
#[pyclass(name = "EmbeddingProviderDefaults", subclass, from_py_object)]
#[derive(Default, Clone)]
pub struct PyEmbeddingProviderDefaults {
    pub(crate) base: PyBaseProviderDefaults,
}

#[gen_stub_pymethods]
#[pymethods]
impl PyEmbeddingProviderDefaults {
    #[new]
    #[pyo3(signature = (base=None))]
    fn new(base: Option<PyBaseProviderDefaults>) -> Self {
        Self {
            base: base.unwrap_or_default(),
        }
    }

    #[getter]
    fn base(&self) -> PyBaseProviderDefaults {
        self.base.clone()
    }

    #[setter]
    fn set_base(&mut self, value: PyBaseProviderDefaults) {
        self.base = value;
    }

    fn __repr__(&self) -> String {
        "EmbeddingProviderDefaults()".to_owned()
    }
}

// ---------------------------------------------------------------------------
// Role-specific defaults
// ---------------------------------------------------------------------------
//
// Mirrors the `role_defaults!` macro in `blazen-llm`. Each role's defaults
// hold a `base` and a typed `before` hook (stored as `Option<Py<PyAny>>`
// for Phase A).
//
// The struct ident is the **Python** name (no `Py` prefix) so
// `gen_stub_pyclass` -- which derives the stub class name from the struct
// ident -- emits the desired class names verbatim in `blazen.pyi`. There
// is no ambiguity with the namesake types in `blazen_llm` since those are
// not imported into this file.

macro_rules! role_defaults_py {
    ($struct_name:ident) => {
        #[gen_stub_pyclass]
        #[pyclass(subclass, from_py_object)]
        #[derive(Default)]
        pub struct $struct_name {
            pub(crate) base: PyBaseProviderDefaults,
            pub(crate) before: Option<Py<PyAny>>,
        }

        impl Clone for $struct_name {
            fn clone(&self) -> Self {
                Self {
                    base: self.base.clone(),
                    before: self
                        .before
                        .as_ref()
                        .map(|h| Python::attach(|py| h.clone_ref(py))),
                }
            }
        }

        #[gen_stub_pymethods]
        #[pymethods]
        impl $struct_name {
            /// Construct role-specific defaults.
            ///
            /// Args:
            ///     base: Universal [`BaseProviderDefaults`]; defaults to an empty bag.
            ///     before: Optional ``async def(request: dict) -> None`` typed
            ///         hook applied after the universal ``before_request``.
            #[new]
            #[pyo3(signature = (base=None, before=None))]
            fn new(base: Option<PyBaseProviderDefaults>, before: Option<Py<PyAny>>) -> Self {
                Self {
                    base: base.unwrap_or_default(),
                    before,
                }
            }

            #[getter]
            fn base(&self) -> PyBaseProviderDefaults {
                self.base.clone()
            }

            #[setter]
            fn set_base(&mut self, value: PyBaseProviderDefaults) {
                self.base = value;
            }

            #[getter]
            fn before(&self) -> Option<Py<PyAny>> {
                self.before
                    .as_ref()
                    .map(|h| Python::attach(|py| h.clone_ref(py)))
            }

            #[setter]
            fn set_before(&mut self, value: Option<Py<PyAny>>) {
                self.before = value;
            }

            fn __repr__(&self) -> String {
                format!(
                    "{}(has_before={})",
                    stringify!($struct_name),
                    self.before.is_some()
                )
            }
        }
    };
}

role_defaults_py!(AudioSpeechProviderDefaults);
role_defaults_py!(AudioMusicProviderDefaults);
role_defaults_py!(VoiceCloningProviderDefaults);
role_defaults_py!(ImageGenerationProviderDefaults);
role_defaults_py!(ImageUpscaleProviderDefaults);
role_defaults_py!(VideoProviderDefaults);
role_defaults_py!(TranscriptionProviderDefaults);
role_defaults_py!(ThreeDProviderDefaults);
role_defaults_py!(BackgroundRemovalProviderDefaults);
