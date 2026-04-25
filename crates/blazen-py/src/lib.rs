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

#[macro_use]
mod macros;

pub(crate) mod convert;
pub(crate) mod session_ref_serializable;

pub mod agent;
pub mod batch;
pub mod compute;
pub mod error;
pub mod manager;
pub mod providers;
pub mod types;
pub mod workflow;

use pyo3::prelude::*;

// Register the stub info gatherer so `cargo run --bin stub_gen` can
// collect all `#[gen_stub_pyclass]` / `#[gen_stub_pymethods]` annotations
// and generate `blazen.pyi` from Rust source (the single source of truth).
pyo3_stub_gen::define_stub_info_gatherer!(stub_info);

/// The top-level Python module for `Blazen`.
#[pymodule]
#[allow(clippy::too_many_lines)]
fn blazen(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Initialize the Rust tracing subscriber once on module load. `try_init`
    // is a no-op if a subscriber is already installed (e.g. the user supplied
    // one before importing blazen). The filter honors `RUST_LOG` and defaults
    // to WARN if the env var is unset. Output goes to stderr so pytest `-s`
    // passes it through without mixing into captured stdout.
    let _ = tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("warn")),
        )
        .with_writer(std::io::stderr)
        .try_init();

    // Version
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;

    // Event classes
    m.add_class::<workflow::event::PyEvent>()?;
    m.add_class::<workflow::event::PyStartEvent>()?;
    m.add_class::<workflow::event::PyStopEvent>()?;

    // Context + state/session namespaces
    m.add_class::<workflow::context::PyContext>()?;
    m.add_class::<workflow::context::PyStateNamespace>()?;
    m.add_class::<workflow::context::PySessionNamespace>()?;

    // State base class
    m.add_class::<workflow::state::PyBlazenState>()?;

    // Step decorator
    m.add_function(wrap_pyfunction!(workflow::step::step, m)?)?;
    m.add_class::<workflow::step::PyStepWrapper>()?;

    // Workflow
    m.add_class::<workflow::workflow::PyWorkflow>()?;
    m.add_class::<workflow::workflow::PySessionPausePolicy>()?;

    // Handler
    m.add_class::<workflow::handler::PyWorkflowHandler>()?;
    m.add_class::<workflow::handler::PyEventStream>()?;

    // LLM types
    m.add_class::<types::PyRole>()?;
    m.add_class::<types::PyContentPart>()?;
    m.add_class::<types::PyChatMessage>()?;
    m.add_class::<types::PyCompletionResponse>()?;
    m.add_class::<types::PyArtifact>()?;
    m.add_class::<types::PyFinishReason>()?;
    m.add_class::<types::PyResponseFormat>()?;
    m.add_class::<types::PyToolOutput>()?;
    m.add_class::<types::PyLlmPayload>()?;

    // Completion model (provider)
    m.add_class::<providers::PyCompletionModel>()?;
    m.add_class::<providers::completion_model::PyCompletionOptions>()?;
    m.add_class::<providers::completion_model::PyLazyCompletionStream>()?;

    // Provider options (typed wrappers)
    m.add_class::<providers::options::PyProviderOptions>()?;
    m.add_class::<providers::options::PyFalLlmEndpointKind>()?;
    m.add_class::<providers::options::PyFalOptions>()?;
    m.add_class::<providers::options::PyAzureOptions>()?;
    m.add_class::<providers::options::PyBedrockOptions>()?;
    m.add_class::<providers::options::PyDevice>()?;

    #[cfg(feature = "embed")]
    m.add_class::<providers::options::PyEmbedOptions>()?;

    m.add_class::<providers::options::PyQuantization>()?;

    #[cfg(feature = "mistralrs")]
    m.add_class::<providers::options::PyMistralRsOptions>()?;

    #[cfg(feature = "whispercpp")]
    m.add_class::<providers::options::PyWhisperModel>()?;
    #[cfg(feature = "whispercpp")]
    m.add_class::<providers::options::PyWhisperOptions>()?;

    // Decorator configs (typed wrappers for retry/cache)
    m.add_class::<providers::config::PyCacheStrategy>()?;
    m.add_class::<providers::config::PyRetryConfig>()?;
    m.add_class::<providers::config::PyCacheConfig>()?;

    // Embedding model + response
    m.add_class::<types::PyEmbeddingModel>()?;
    m.add_class::<types::PyEmbeddingResponse>()?;

    // Transcription provider
    m.add_class::<types::PyTranscription>()?;

    // Token counting utilities
    m.add_function(wrap_pyfunction!(types::estimate_tokens, m)?)?;
    m.add_function(wrap_pyfunction!(types::count_message_tokens, m)?)?;

    // Chat window (token-windowed conversation memory)
    m.add_class::<types::PyChatWindow>()?;

    // Agent
    m.add_class::<agent::PyToolDef>()?;
    m.add_class::<agent::PyAgentResult>()?;
    m.add_function(wrap_pyfunction!(agent::run_agent, m)?)?;

    // Batch completion
    m.add_class::<batch::PyBatchResult>()?;
    m.add_function(wrap_pyfunction!(batch::complete_batch, m)?)?;

    // Compute / Media -- Request types
    m.add_class::<types::PyMediaType>()?;

    // Compute / Media -- Job types
    m.add_class::<compute::PyJobStatus>()?;
    m.add_class::<compute::PyComputeRequest>()?;
    m.add_class::<compute::PyJobHandle>()?;

    // Compute request type wrappers (typed inputs for fal provider methods)
    m.add_class::<compute::request_types::PyImageRequest>()?;
    m.add_class::<compute::request_types::PyUpscaleRequest>()?;
    m.add_class::<compute::request_types::PyVideoRequest>()?;
    m.add_class::<compute::request_types::PySpeechRequest>()?;
    m.add_class::<compute::request_types::PyMusicRequest>()?;
    m.add_class::<compute::request_types::PyTranscriptionRequest>()?;
    m.add_class::<compute::request_types::PyThreeDRequest>()?;
    m.add_class::<compute::request_types::PyBackgroundRemovalRequest>()?;
    m.add_class::<compute::request_types::PyVoiceCloneRequest>()?;

    // Compute result type wrappers (typed outputs from fal provider methods)
    m.add_class::<compute::PyTranscriptionSegment>()?;
    m.add_class::<compute::PyImageResult>()?;
    m.add_class::<compute::PyVideoResult>()?;
    m.add_class::<compute::PyAudioResult>()?;
    m.add_class::<compute::PyThreeDResult>()?;
    m.add_class::<compute::PyTranscriptionResult>()?;
    m.add_class::<compute::PyComputeResult>()?;
    m.add_class::<compute::PyVoiceHandle>()?;

    // Generated media output types
    m.add_class::<types::media::PyMediaOutput>()?;
    m.add_class::<types::media::PyGeneratedImage>()?;
    m.add_class::<types::media::PyGeneratedVideo>()?;
    m.add_class::<types::media::PyGeneratedAudio>()?;
    m.add_class::<types::media::PyGenerated3DModel>()?;
    m.add_class::<types::PyRequestTiming>()?;

    // Fal provider
    m.add_class::<providers::fal::PyFalProvider>()?;
    m.add_class::<providers::fal::PyFalEmbeddingModel>()?;

    // OpenAI provider (for compute capabilities like TTS; LLM completion
    // lives on `CompletionModel.openai`).
    m.add_class::<providers::openai::PyOpenAiProvider>()?;

    // Custom provider (user-defined Python class wrapped as a Blazen provider).
    m.add_class::<providers::custom::PyCustomProvider>()?;

    // Capability providers (subclassable)
    m.add_class::<providers::capability_providers::TTSProvider>()?;
    m.add_class::<providers::capability_providers::MusicProvider>()?;
    m.add_class::<providers::capability_providers::ImageProvider>()?;
    m.add_class::<providers::capability_providers::VideoProvider>()?;
    m.add_class::<providers::capability_providers::ThreeDProvider>()?;
    m.add_class::<providers::capability_providers::BackgroundRemovalProvider>()?;
    m.add_class::<providers::capability_providers::VoiceProvider>()?;

    // Model manager
    m.add_class::<manager::PyModelManager>()?;
    m.add_class::<manager::PyModelStatus>()?;

    // Memory
    m.add_class::<types::PyMemory>()?;
    m.add_class::<types::PyMemoryBackend>()?;
    m.add_class::<types::PyInMemoryBackend>()?;
    m.add_class::<types::PyJsonlBackend>()?;
    m.add_class::<types::PyValkeyBackend>()?;
    m.add_class::<types::PyMemoryResult>()?;

    // Pricing
    m.add_class::<types::pricing::PyModelPricing>()?;
    m.add_function(wrap_pyfunction!(types::pricing::register_pricing, m)?)?;
    m.add_function(wrap_pyfunction!(types::pricing::lookup_pricing, m)?)?;

    // Prompts
    m.add_class::<types::PyPromptTemplate>()?;
    m.add_class::<types::PyPromptRegistry>()?;

    // Error exception types
    error::register_exceptions(m)?;

    Ok(())
}
