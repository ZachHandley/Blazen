#![allow(clippy::doc_markdown)] // Python docstrings don't follow Rust markdown conventions.
#![allow(clippy::missing_errors_doc)] // PyO3 functions return PyResult; errors are self-evident.
#![allow(clippy::needless_pass_by_value)] // PyO3 extractors require owned types.
#![allow(clippy::must_use_candidate)] // PyO3 methods are called from Python.
#![allow(clippy::unused_self)] // PyO3 requires &self for some methods.
#![allow(clippy::used_underscore_binding)] // Convention for unused params in conversion functions.
#![allow(clippy::doc_link_with_quotes)] // Docstring style preference.

//! `Blazen` Python bindings.
//!
//! Exposes the `Blazen` workflow engine and LLM integration layer to Python
//! via PyO3. The native module is named `blazen`.
//!
//! # Quick start (Python)
//!
//! ```python
//! from blazen import Workflow, step, Event, StartEvent, StopEvent, Context
//! from blazen import CompletionModel, ChatMessage
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

pub mod agent;
pub mod compute;
pub mod error;
pub mod providers;
pub mod types;
pub mod workflow;

use pyo3::prelude::*;

/// The top-level Python module for `Blazen`.
#[pymodule]
fn blazen(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Version
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;

    // Event classes
    m.add_class::<workflow::event::PyEvent>()?;
    m.add_class::<workflow::event::PyStartEvent>()?;
    m.add_class::<workflow::event::PyStopEvent>()?;

    // Context
    m.add_class::<workflow::context::PyContext>()?;

    // Step decorator
    m.add_function(wrap_pyfunction!(workflow::step::step, m)?)?;
    m.add_class::<workflow::step::PyStepWrapper>()?;

    // Workflow
    m.add_class::<workflow::workflow::PyWorkflow>()?;

    // Handler
    m.add_class::<workflow::handler::PyWorkflowHandler>()?;
    m.add_class::<workflow::handler::PyEventStream>()?;

    // LLM types
    m.add_class::<types::PyRole>()?;
    m.add_class::<types::PyContentPart>()?;
    m.add_class::<types::PyChatMessage>()?;
    m.add_class::<types::PyToolCall>()?;
    m.add_class::<types::PyTokenUsage>()?;
    m.add_class::<types::PyRequestTiming>()?;
    m.add_class::<types::PyCompletionResponse>()?;
    m.add_class::<types::PyStreamChunk>()?;

    // Completion model (provider)
    m.add_class::<providers::PyCompletionModel>()?;

    // Embedding model
    m.add_class::<types::PyEmbeddingModel>()?;
    m.add_class::<types::PyEmbeddingResponse>()?;

    // Token counting utilities
    m.add_function(wrap_pyfunction!(types::estimate_tokens, m)?)?;
    m.add_function(wrap_pyfunction!(types::count_message_tokens, m)?)?;

    // Agent
    m.add_class::<agent::PyToolDef>()?;
    m.add_class::<agent::PyAgentResult>()?;
    m.add_function(wrap_pyfunction!(agent::run_agent, m)?)?;

    // Compute / Media -- Request types
    m.add_class::<types::PyMediaType>()?;
    m.add_class::<compute::PyImageRequest>()?;
    m.add_class::<compute::PyUpscaleRequest>()?;
    m.add_class::<compute::PyVideoRequest>()?;
    m.add_class::<compute::PySpeechRequest>()?;
    m.add_class::<compute::PyMusicRequest>()?;
    m.add_class::<compute::PyTranscriptionRequest>()?;
    m.add_class::<compute::PyThreeDRequest>()?;

    // Compute / Media -- Job types
    m.add_class::<compute::PyJobStatus>()?;
    m.add_class::<compute::PyJobHandle>()?;
    m.add_class::<compute::PyComputeRequest>()?;

    // Compute / Media -- Generated media types
    m.add_class::<types::PyMediaOutput>()?;
    m.add_class::<types::PyGeneratedImage>()?;
    m.add_class::<types::PyGeneratedVideo>()?;
    m.add_class::<types::PyGeneratedAudio>()?;
    m.add_class::<types::PyGenerated3DModel>()?;

    // Fal provider
    m.add_class::<providers::fal::PyFalProvider>()?;

    // Memory
    m.add_class::<types::PyMemory>()?;
    m.add_class::<types::PyInMemoryBackend>()?;
    m.add_class::<types::PyJsonlBackend>()?;
    m.add_class::<types::PyValkeyBackend>()?;
    m.add_class::<types::PyMemoryResult>()?;

    // Error exception types
    error::register_exceptions(m)?;

    Ok(())
}
