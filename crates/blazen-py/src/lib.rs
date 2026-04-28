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
pub mod core_types;
pub mod error;
pub mod manager;
pub mod model_cache;
pub mod peer;
pub mod persist;
pub mod pipeline;
pub mod providers;
pub mod telemetry;
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
    m.add_class::<workflow::event::PyDynamicEvent>()?;
    m.add_class::<workflow::event::PyEventEnvelope>()?;
    m.add_class::<workflow::event::PyInputRequestEvent>()?;
    m.add_class::<workflow::event::PyInputResponseEvent>()?;
    m.add_function(wrap_pyfunction!(
        workflow::event::register_event_deserializer,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(workflow::event::try_deserialize_event, m)?)?;
    m.add_function(wrap_pyfunction!(workflow::event::intern_event_type, m)?)?;

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
    m.add_class::<workflow::builder::PyWorkflowBuilder>()?;

    // Handler
    m.add_class::<workflow::handler::PyWorkflowHandler>()?;
    m.add_class::<workflow::handler::PyEventStream>()?;

    // Core types (session-ref registry, value, snapshot, step registry)
    m.add_class::<core_types::PySessionRefRegistry>()?;
    m.add_class::<core_types::PyRegistryKey>()?;
    m.add_class::<core_types::PyRemoteRefDescriptor>()?;
    m.add_class::<core_types::PyRefLifetime>()?;
    m.add_class::<core_types::PyBytesWrapper>()?;
    m.add_class::<core_types::PyStateValue>()?;
    m.add_class::<core_types::PyStepDeserializerRegistry>()?;
    m.add_class::<core_types::PyStepOutput>()?;
    m.add_class::<core_types::PyStepRegistration>()?;
    m.add_class::<core_types::PyWorkflowSnapshot>()?;
    m.add_class::<core_types::PyWorkflowResult>()?;
    m.add_function(wrap_pyfunction!(
        core_types::step_registry::register_step_builder,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        core_types::step_registry::lookup_step_builder,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        core_types::step_registry::registered_step_ids,
        m
    )?)?;

    #[cfg(feature = "distributed")]
    {
        m.add_class::<core_types::PyRemoteWorkflowRequest>()?;
        m.add_class::<core_types::PyRemoteWorkflowResponse>()?;
        m.add_class::<core_types::PyPeerClient>()?;
    }

    // Pipeline (multi-workflow orchestrator)
    m.add_class::<pipeline::PyPipelineBuilder>()?;
    m.add_class::<pipeline::PyPipeline>()?;
    m.add_class::<pipeline::PyPipelineHandler>()?;
    m.add_class::<pipeline::PyPipelineEventStream>()?;
    m.add_class::<pipeline::PyPipelineEvent>()?;
    m.add_class::<pipeline::PyPipelineSnapshot>()?;
    m.add_class::<pipeline::PyPipelineResult>()?;
    m.add_class::<pipeline::PyStageResult>()?;
    m.add_class::<pipeline::PyActiveWorkflowSnapshot>()?;
    m.add_class::<pipeline::PyPipelineState>()?;
    m.add_class::<pipeline::PyStage>()?;
    m.add_class::<pipeline::PyParallelStage>()?;
    m.add_class::<pipeline::PyJoinStrategy>()?;
    m.add_class::<pipeline::PyStageKind>()?;
    pipeline::register_exceptions(m)?;

    // Persistence (workflow checkpoint storage)
    m.add_class::<persist::PyWorkflowCheckpoint>()?;
    m.add_class::<persist::PyPersistedEvent>()?;
    m.add_class::<persist::PyCheckpointStore>()?;
    m.add_class::<persist::PyRedbCheckpointStore>()?;
    m.add_class::<persist::PyValkeyCheckpointStore>()?;
    persist::register_exceptions(m)?;

    // Peer (distributed sub-workflow gRPC layer)
    m.add_class::<peer::PyBlazenPeerServer>()?;
    m.add_class::<peer::PyBlazenPeerClient>()?;
    m.add_class::<peer::PySubWorkflowRequest>()?;
    m.add_class::<peer::PySubWorkflowResponse>()?;
    m.add_class::<peer::PyDerefRequest>()?;
    m.add_class::<peer::PyDerefResponse>()?;
    m.add_class::<peer::PyReleaseRequest>()?;
    m.add_class::<peer::PyReleaseResponse>()?;
    m.add_class::<peer::PyPeerRemoteRefDescriptor>()?;
    m.add_function(wrap_pyfunction!(peer::resolve_peer_token, m)?)?;
    m.add_function(wrap_pyfunction!(peer::load_server_tls, m)?)?;
    m.add_function(wrap_pyfunction!(peer::load_client_tls, m)?)?;
    m.add("PEER_TOKEN_ENV", blazen_peer::auth::PEER_TOKEN_ENV)?;
    m.add("ENVELOPE_VERSION", blazen_peer::ENVELOPE_VERSION)?;
    peer::register_exceptions(m)?;

    // LLM types
    m.add_class::<types::PyRole>()?;
    m.add_class::<types::PyContentPart>()?;
    m.add_class::<types::PyChatMessage>()?;
    m.add_class::<types::PyCompletionResponse>()?;
    m.add_class::<types::PyCompletionRequest>()?;
    m.add_class::<types::PyArtifact>()?;
    m.add_class::<types::PyCitation>()?;
    m.add_class::<types::PyReasoningTrace>()?;
    m.add_class::<types::PyTokenUsage>()?;
    m.add_class::<types::PyFinishReason>()?;
    m.add_class::<types::PyResponseFormat>()?;
    m.add_class::<types::PyToolOutput>()?;
    m.add_class::<types::PyLlmPayload>()?;
    m.add_class::<types::PyToolCall>()?;
    m.add_class::<types::PyToolDefinition>()?;
    m.add_class::<types::PyHttpClient>()?;
    m.add_class::<types::PyProviderId>()?;
    m.add_class::<types::PyPricingEntry>()?;
    m.add_class::<types::PyMessageContent>()?;
    m.add_class::<types::PyImageContent>()?;
    m.add_class::<types::PyAudioContent>()?;
    m.add_class::<types::PyVideoContent>()?;
    m.add_class::<types::PyFileContent>()?;
    m.add_class::<types::PyImageSource>()?;
    // `MediaSource` is a type alias for `ImageSource` in the Rust crate
    // (`pub type MediaSource = ImageSource;`). Surface the same alias to
    // Python so `from blazen import MediaSource` works.
    m.add("MediaSource", m.py().get_type::<types::PyImageSource>())?;
    m.add_class::<types::PyStreamChunk>()?;
    m.add_class::<types::PyStreamChunkEvent>()?;
    m.add_class::<types::PyStreamCompleteEvent>()?;
    m.add_class::<types::PyProviderCapabilities>()?;
    m.add_class::<types::PyProviderConfig>()?;
    m.add_class::<types::PyModelCapabilities>()?;
    m.add_class::<types::PyProviderInfo>()?;
    m.add_class::<types::PyTool>()?;
    m.add_class::<types::PyLocalModel>()?;
    m.add_class::<types::PyStructuredOutput>()?;
    m.add_class::<types::PyStructuredResponse>()?;
    m.add_class::<types::PyModelRegistry>()?;
    m.add_class::<types::PyImageModel>()?;
    m.add_class::<types::PyHostDispatchAbc>()?;
    m.add_class::<types::PyMemoryEntry>()?;

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

    #[cfg(feature = "llamacpp")]
    m.add_class::<providers::options::PyLlamaCppOptions>()?;

    #[cfg(feature = "candle-llm")]
    m.add_class::<providers::options::PyCandleLlmOptions>()?;

    #[cfg(feature = "candle-embed")]
    m.add_class::<providers::options::PyCandleEmbedOptions>()?;

    #[cfg(feature = "piper")]
    m.add_class::<providers::options::PyPiperOptions>()?;

    #[cfg(feature = "diffusion")]
    {
        m.add_class::<providers::options::PyDiffusionScheduler>()?;
        m.add_class::<providers::options::PyDiffusionOptions>()?;
    }

    #[cfg(all(feature = "embed", not(target_env = "musl")))]
    m.add_class::<providers::options::PyFastEmbedOptions>()?;

    #[cfg(feature = "tract")]
    m.add_class::<providers::options::PyTractOptions>()?;

    // Decorator configs (typed wrappers for retry/cache)
    m.add_class::<providers::config::PyCacheStrategy>()?;
    m.add_class::<providers::config::PyRetryConfig>()?;
    m.add_class::<providers::config::PyCacheConfig>()?;

    // Standalone decorator wrappers (parity with CompletionModel.with_*)
    m.add_class::<providers::PyRetryCompletionModel>()?;
    m.add_class::<providers::PyCachedCompletionModel>()?;
    m.add_class::<providers::PyFallbackModel>()?;

    // Middleware surface
    m.add_class::<providers::PyMiddleware>()?;
    m.add_class::<providers::PyRetryMiddleware>()?;
    m.add_class::<providers::PyCacheMiddleware>()?;
    m.add_class::<providers::PyMiddlewareStack>()?;

    // Embedding model + response
    m.add_class::<types::PyEmbeddingModel>()?;
    m.add_class::<types::PyEmbeddingResponse>()?;

    // Transcription provider
    m.add_class::<types::PyTranscription>()?;

    // Token counting utilities
    m.add_class::<types::PyTokenCounter>()?;
    m.add_class::<types::PyEstimateCounter>()?;
    #[cfg(feature = "tiktoken")]
    m.add_class::<types::PyTiktokenCounter>()?;
    m.add_function(wrap_pyfunction!(types::estimate_tokens, m)?)?;
    m.add_function(wrap_pyfunction!(types::count_message_tokens, m)?)?;

    // Chat window (token-windowed conversation memory)
    m.add_class::<types::PyChatWindow>()?;

    // Agent
    m.add_class::<agent::PyToolDef>()?;
    m.add_class::<agent::PyAgentResult>()?;
    m.add_class::<agent::PyAgentConfig>()?;
    m.add_class::<agent::PyAgentEvent>()?;
    m.add_function(wrap_pyfunction!(agent::run_agent, m)?)?;
    m.add_function(wrap_pyfunction!(agent::run_agent_with_callback, m)?)?;

    // Batch completion
    m.add_class::<batch::PyBatchConfig>()?;
    m.add_class::<batch::PyBatchResult>()?;
    m.add_function(wrap_pyfunction!(batch::complete_batch, m)?)?;

    // Compute / Media -- Request types
    m.add_class::<types::PyMediaType>()?;

    // Compute / Media -- Job types
    m.add_class::<compute::PyJobStatus>()?;
    m.add_class::<compute::PyComputeRequest>()?;
    m.add_class::<compute::PyJobHandle>()?;
    m.add_class::<compute::PyCompute>()?;

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
    m.add_class::<providers::openai_embedding::PyOpenAiEmbeddingModel>()?;

    // Standalone cloud-LLM provider classes (parity with the
    // `CompletionModel.<provider>(...)` factory methods --- wraps the same
    // Rust providers).
    m.add_class::<providers::anthropic::PyAnthropicProvider>()?;
    m.add_class::<providers::gemini::PyGeminiProvider>()?;
    m.add_class::<providers::azure::PyAzureOpenAiProvider>()?;
    m.add_class::<providers::openrouter::PyOpenRouterProvider>()?;
    m.add_class::<providers::groq::PyGroqProvider>()?;
    m.add_class::<providers::together::PyTogetherProvider>()?;
    m.add_class::<providers::mistral::PyMistralProvider>()?;
    m.add_class::<providers::deepseek::PyDeepSeekProvider>()?;
    m.add_class::<providers::fireworks::PyFireworksProvider>()?;
    m.add_class::<providers::perplexity::PyPerplexityProvider>()?;
    m.add_class::<providers::xai::PyXaiProvider>()?;
    m.add_class::<providers::cohere::PyCohereProvider>()?;
    m.add_class::<providers::bedrock::PyBedrockProvider>()?;

    // Generic OpenAI-compatible provider + config + auth-method enum.
    m.add_class::<providers::openai_compat::PyAuthMethod>()?;
    m.add_class::<providers::openai_compat::PyOpenAiCompatConfig>()?;
    m.add_class::<providers::openai_compat::PyOpenAiCompatProvider>()?;
    m.add_class::<providers::openai_compat::PyOpenAiCompatEmbeddingModel>()?;

    // Typed tool authoring (Pydantic-model schema spec).
    m.add_class::<types::typed_tool::PyTypedTool>()?;
    m.add_function(wrap_pyfunction!(types::typed_tool::typed_tool_simple, m)?)?;

    // Provider-helper free functions and ModelInfo.
    m.add_class::<providers::funcs::PyModelInfo>()?;
    m.add_function(wrap_pyfunction!(
        providers::funcs::extract_inline_artifacts,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(providers::funcs::env_var_for_provider, m)?)?;
    m.add_function(wrap_pyfunction!(providers::funcs::resolve_api_key, m)?)?;
    m.add_function(wrap_pyfunction!(providers::funcs::get_context_window, m)?)?;
    m.add_function(wrap_pyfunction!(
        providers::funcs::format_provider_http_tail,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        providers::funcs::register_from_model_info,
        m
    )?)?;
    m.add("PROVIDER_ENV_VARS", providers::funcs::provider_env_vars())?;

    // Local backend providers (feature-gated standalone wrappers).
    #[cfg(feature = "llamacpp")]
    {
        m.add_class::<providers::llamacpp::PyLlamaCppProvider>()?;
        m.add_class::<providers::llamacpp::PyLlamaCppChatMessageInput>()?;
        m.add_class::<providers::llamacpp::PyLlamaCppChatRole>()?;
        m.add_class::<providers::llamacpp::PyLlamaCppInferenceChunk>()?;
        m.add_class::<providers::llamacpp::PyLlamaCppInferenceChunkStream>()?;
        m.add_class::<providers::llamacpp::PyLlamaCppInferenceResult>()?;
        m.add_class::<providers::llamacpp::PyLlamaCppInferenceUsage>()?;
    }
    #[cfg(feature = "candle-llm")]
    m.add_class::<providers::candle_llm::PyCandleLlmProvider>()?;
    #[cfg(feature = "candle-llm")]
    m.add_class::<providers::candle_llm::PyCandleInferenceResult>()?;
    #[cfg(feature = "candle-embed")]
    m.add_class::<providers::candle_embed::PyCandleEmbedModel>()?;
    #[cfg(feature = "mistralrs")]
    {
        m.add_class::<providers::mistralrs::PyMistralRsProvider>()?;
        m.add_class::<providers::mistralrs::PyChatMessageInput>()?;
        m.add_class::<providers::mistralrs::PyChatRole>()?;
        m.add_class::<providers::mistralrs::PyInferenceChunk>()?;
        m.add_class::<providers::mistralrs::PyInferenceChunkStream>()?;
        m.add_class::<providers::mistralrs::PyInferenceImage>()?;
        m.add_class::<providers::mistralrs::PyInferenceImageSource>()?;
        m.add_class::<providers::mistralrs::PyInferenceResult>()?;
        m.add_class::<providers::mistralrs::PyInferenceToolCall>()?;
        m.add_class::<providers::mistralrs::PyInferenceUsage>()?;
    }
    #[cfg(feature = "whispercpp")]
    m.add_class::<providers::whispercpp::PyWhisperCppProvider>()?;
    #[cfg(feature = "piper")]
    m.add_class::<providers::piper::PyPiperProvider>()?;
    #[cfg(feature = "diffusion")]
    m.add_class::<providers::diffusion::PyDiffusionProvider>()?;
    #[cfg(all(feature = "embed", not(target_env = "musl")))]
    m.add_class::<providers::fastembed::PyFastEmbedModel>()?;
    #[cfg(feature = "tract")]
    m.add_class::<providers::tract::PyTractEmbedModel>()?;
    #[cfg(feature = "tract")]
    m.add_class::<providers::tract::PyTractResponse>()?;

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

    // Model cache (HuggingFace Hub downloader / on-disk cache)
    m.add_class::<model_cache::PyModelCache>()?;
    m.add_class::<model_cache::PyProgressCallback>()?;
    model_cache::register_exceptions(m)?;

    // Memory
    m.add_class::<types::PyMemory>()?;
    m.add_class::<types::PyMemoryStore>()?;
    m.add_class::<types::PyMemoryBackend>()?;
    m.add_class::<types::PyInMemoryBackend>()?;
    m.add_class::<types::PyJsonlBackend>()?;
    m.add_class::<types::PyValkeyBackend>()?;
    m.add_class::<types::PyMemoryResult>()?;
    m.add_class::<types::PyStoredEntry>()?;
    m.add_function(wrap_pyfunction!(types::compute_text_simhash_similarity, m)?)?;
    m.add_function(wrap_pyfunction!(
        types::compute_embedding_simhash_similarity,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(types::compute_elid_similarity, m)?)?;
    m.add_function(wrap_pyfunction!(types::simhash_from_hex, m)?)?;
    m.add_function(wrap_pyfunction!(types::simhash_to_hex, m)?)?;
    types::memory::register_exceptions(m)?;

    // Pricing
    m.add_class::<types::pricing::PyModelPricing>()?;
    m.add_function(wrap_pyfunction!(types::pricing::register_pricing, m)?)?;
    m.add_function(wrap_pyfunction!(types::pricing::lookup_pricing, m)?)?;

    // Prompts
    m.add_class::<types::PyTemplateRole>()?;
    m.add_class::<types::PyPromptTemplate>()?;
    m.add_class::<types::PyPromptFile>()?;
    m.add_class::<types::PyPromptRegistry>()?;
    types::prompts::register_exceptions(m)?;

    // Telemetry -- workflow event history
    m.add_class::<telemetry::history::PyWorkflowHistory>()?;
    m.add_class::<telemetry::history::PyHistoryEvent>()?;
    m.add_class::<telemetry::history::PyHistoryEventKind>()?;
    m.add_class::<telemetry::history::PyPauseReason>()?;

    // Telemetry -- tracing-instrumented completion model wrapper
    m.add_function(wrap_pyfunction!(
        telemetry::tracing_model::wrap_with_tracing,
        m
    )?)?;

    // Telemetry -- OTLP exporter (feature-gated)
    #[cfg(feature = "otlp")]
    m.add_class::<telemetry::otlp::PyOtlpConfig>()?;
    #[cfg(feature = "otlp")]
    m.add_function(wrap_pyfunction!(telemetry::otlp::init_otlp, m)?)?;

    // Telemetry -- Prometheus exporter (feature-gated)
    #[cfg(feature = "prometheus")]
    m.add_function(wrap_pyfunction!(telemetry::prometheus::init_prometheus, m)?)?;

    // Telemetry -- Langfuse exporter (feature-gated)
    #[cfg(feature = "langfuse")]
    m.add_class::<telemetry::langfuse::PyLangfuseConfig>()?;
    #[cfg(feature = "langfuse")]
    m.add_function(wrap_pyfunction!(telemetry::langfuse::init_langfuse, m)?)?;

    // Error exception types
    error::register_exceptions(m)?;

    Ok(())
}
