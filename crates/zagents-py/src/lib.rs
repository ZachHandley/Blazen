#![allow(clippy::doc_markdown)] // Python docstrings don't follow Rust markdown conventions.
#![allow(clippy::missing_errors_doc)] // PyO3 functions return PyResult; errors are self-evident.
#![allow(clippy::needless_pass_by_value)] // PyO3 extractors require owned types.
#![allow(clippy::must_use_candidate)] // PyO3 methods are called from Python.
#![allow(clippy::unused_self)] // PyO3 requires &self for some methods.
#![allow(clippy::used_underscore_binding)] // Convention for unused params in conversion functions.
#![allow(clippy::doc_link_with_quotes)] // Docstring style preference.

//! `ZAgents` Python bindings.
//!
//! Exposes the `ZAgents` workflow engine and LLM integration layer to Python
//! via PyO3. The native module is named `_zagents` (with underscore prefix,
//! following Python convention for native extension modules).
//!
//! # Quick start (Python)
//!
//! ```python
//! from zagents import Workflow, step, Event, StartEvent, StopEvent, Context
//! from zagents import CompletionModel, ChatMessage
//!
//! @step
//! async def echo(ctx: Context, ev: Event) -> Event:
//!     return StopEvent(result=ev.to_dict())
//!
//! async def main():
//!     wf = Workflow("echo", [echo])
//!     handler = await wf.run(message="hello")
//!     result = await handler.result()
//!     print(result.to_dict())
//! ```

pub mod context;
pub mod error;
pub mod event;
pub mod handler;
pub mod llm;
pub mod step;
pub mod workflow;

use pyo3::prelude::*;

/// The top-level Python module for `ZAgents`.
#[pymodule]
fn _zagents(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Version
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;

    // Event classes
    m.add_class::<event::PyEvent>()?;
    m.add_class::<event::PyStartEvent>()?;
    m.add_class::<event::PyStopEvent>()?;

    // Context
    m.add_class::<context::PyContext>()?;

    // Step decorator
    m.add_function(wrap_pyfunction!(step::step, m)?)?;
    m.add_class::<step::PyStepWrapper>()?;

    // Workflow
    m.add_class::<workflow::PyWorkflow>()?;

    // Handler
    m.add_class::<handler::PyWorkflowHandler>()?;
    m.add_class::<handler::PyEventStream>()?;

    // LLM
    m.add_class::<llm::PyCompletionModel>()?;
    m.add_class::<llm::PyChatMessage>()?;

    Ok(())
}
