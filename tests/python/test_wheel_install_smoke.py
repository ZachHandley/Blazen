"""Wheel install smoke test.

Verifies that the installed `blazen` wheel exposes every name in a curated
EXPECTED set. This catches platform-specific binding regressions where a
wheel builds successfully but a class is silently missing from the module.

Per-feature tests skip gracefully when the wheel wasn't built with that
feature, but if the feature is present every member of the EXPECTED list
must be importable from `blazen`.
"""

import pytest

import blazen

# Core surface (always present, regardless of optional features).
EXPECTED_CORE = [
    # Workflow + pipeline orchestration.
    "Workflow",
    "WorkflowBuilder",
    "WorkflowHandler",
    "WorkflowResult",
    "WorkflowSnapshot",
    "Pipeline",
    "PipelineBuilder",
    "PipelineHandler",
    "PipelineResult",
    "PipelineSnapshot",
    "PipelineState",
    "PipelineEvent",
    "Stage",
    "ParallelStage",
    "StageResult",
    "StageKind",
    "JoinStrategy",
    "ActiveWorkflowSnapshot",
    # Events.
    "Event",
    "StartEvent",
    "StopEvent",
    "DynamicEvent",
    "EventEnvelope",
    "InputRequestEvent",
    "InputResponseEvent",
    # Context / state.
    "Context",
    "StateNamespace",
    "SessionNamespace",
    "BlazenState",
    "StateValue",
    # Session refs / registries.
    "SessionRefRegistry",
    "RegistryKey",
    "RemoteRefDescriptor",
    "RefLifetime",
    "BytesWrapper",
    "StepDeserializerRegistry",
    "StepOutput",
    "StepRegistration",
    # Memory.
    "Memory",
    "MemoryStore",
    "MemoryBackend",
    "InMemoryBackend",
    "JsonlBackend",
    "ValkeyBackend",
    "MemoryResult",
    "MemoryEntry",
    "StoredEntry",
    # LLM core types.
    "ChatMessage",
    "Role",
    "ContentPart",
    "MessageContent",
    "ImageContent",
    "AudioContent",
    "VideoContent",
    "FileContent",
    "ImageSource",
    "MediaSource",
    "MediaType",
    "MediaOutput",
    "Artifact",
    "Citation",
    "ReasoningTrace",
    "TokenUsage",
    "FinishReason",
    "ResponseFormat",
    "ToolOutput",
    "LlmPayload",
    "ToolCall",
    "ToolDefinition",
    "Tool",
    "ToolDef",
    "TypedTool",
    "HttpClient",
    "ProviderId",
    "PricingEntry",
    "StreamChunk",
    "StreamChunkEvent",
    "StreamCompleteEvent",
    "ProviderCapabilities",
    "ProviderConfig",
    "ProviderInfo",
    "ModelCapabilities",
    "ModelRegistry",
    "ModelInfo",
    "ModelPricing",
    "LocalModel",
    "ImageModel",
    "StructuredOutput",
    "StructuredResponse",
    "CompletionRequest",
    "CompletionResponse",
    "CompletionStream",
    "CompletionOptions",
    "CompletionModel",
    "RetryCompletionModel",
    "CachedCompletionModel",
    "FallbackModel",
    "RetryConfig",
    "CacheConfig",
    "CacheStrategy",
    "Middleware",
    "RetryMiddleware",
    "CacheMiddleware",
    "MiddlewareStack",
    "EmbeddingModel",
    "EmbeddingResponse",
    "Transcription",
    "TokenCounter",
    "EstimateCounter",
    "ChatWindow",
    # Provider options (always-available).
    "ProviderOptions",
    "FalLlmEndpointKind",
    "FalOptions",
    "AzureOptions",
    "BedrockOptions",
    "Device",
    "Quantization",
    # Cloud providers (always built).
    "AnthropicProvider",
    "OpenAiProvider",
    "OpenAiEmbeddingModel",
    "GeminiProvider",
    "AzureOpenAiProvider",
    "OpenRouterProvider",
    "GroqProvider",
    "TogetherProvider",
    "MistralProvider",
    "DeepSeekProvider",
    "FireworksProvider",
    "PerplexityProvider",
    "XaiProvider",
    "CohereProvider",
    "BedrockProvider",
    "FalProvider",
    "FalEmbeddingModel",
    "OpenAiCompatProvider",
    "OpenAiCompatConfig",
    "OpenAiCompatEmbeddingModel",
    "AuthMethod",
    "CustomProvider",
    # Capability provider base classes.
    "TTSProvider",
    "MusicProvider",
    "ImageProvider",
    "VideoProvider",
    "ThreeDProvider",
    "BackgroundRemovalProvider",
    "VoiceProvider",
    # Agent + batch.
    "AgentConfig",
    "AgentResult",
    "AgentEvent",
    "BatchConfig",
    "BatchResult",
    # Compute + media.
    "Compute",
    "ComputeRequest",
    "ComputeResult",
    "JobHandle",
    "JobStatus",
    "ImageRequest",
    "UpscaleRequest",
    "VideoRequest",
    "SpeechRequest",
    "MusicRequest",
    "TranscriptionRequest",
    "ThreeDRequest",
    "BackgroundRemovalRequest",
    "VoiceCloneRequest",
    "ImageResult",
    "VideoResult",
    "AudioResult",
    "ThreeDResult",
    "TranscriptionResult",
    "TranscriptionSegment",
    "GeneratedImage",
    "GeneratedVideo",
    "GeneratedAudio",
    "Generated3DModel",
    "VoiceHandle",
    "RequestTiming",
    # Manager / cache.
    "ModelManager",
    "ModelStatus",
    "ModelCache",
    "ProgressCallback",
    # Persistence.
    "WorkflowCheckpoint",
    "PersistedEvent",
    "CheckpointStore",
    "RedbCheckpointStore",
    "ValkeyCheckpointStore",
    # Peer (distributed).
    "BlazenPeerServer",
    "BlazenPeerClient",
    "SubWorkflowRequest",
    "SubWorkflowResponse",
    "DerefRequest",
    "DerefResponse",
    "ReleaseRequest",
    "ReleaseResponse",
    "PeerRemoteRefDescriptor",
    # Prompts.
    "PromptTemplate",
    "PromptFile",
    "PromptRegistry",
    "TemplateRole",
    # Telemetry / history.
    "WorkflowHistory",
    "HistoryEvent",
    "HistoryEventKind",
    "PauseReason",
    "SessionPausePolicy",
    # Errors.
    "BlazenError",
    "AuthError",
    "RateLimitError",
    "TimeoutError",
    "ValidationError",
    "ContentPolicyError",
    "ProviderError",
    "UnsupportedError",
    "ComputeError",
    "MediaError",
    # Top-level functions.
    "step",
    "typed_tool_simple",
    "wrap_with_tracing",
    "extract_inline_artifacts",
    "estimate_tokens",
    "count_message_tokens",
    "get_context_window",
    "lookup_pricing",
    "register_pricing",
    "register_from_model_info",
    "register_step_builder",
    "lookup_step_builder",
    "registered_step_ids",
    "register_event_deserializer",
    "try_deserialize_event",
    "intern_event_type",
    "env_var_for_provider",
    "resolve_api_key",
    "resolve_peer_token",
    "load_client_tls",
    "load_server_tls",
    "format_provider_http_tail",
    "compute_elid_similarity",
    "compute_text_simhash_similarity",
    "compute_embedding_simhash_similarity",
    "simhash_from_hex",
    "simhash_to_hex",
    # Module constants.
    "PROVIDER_ENV_VARS",
]

# Feature: llamacpp (gated by `--features llamacpp`).
EXPECTED_LLAMACPP = [
    "LlamaCppProvider",
    "LlamaCppOptions",
    "LlamaCppChatMessageInput",
    "LlamaCppChatRole",
    "LlamaCppInferenceChunk",
    "LlamaCppInferenceChunkStream",
    "LlamaCppInferenceResult",
    "LlamaCppInferenceUsage",
]

# Feature: mistralrs.
EXPECTED_MISTRALRS = [
    "MistralRsProvider",
    "MistralRsOptions",
    "ChatMessageInput",
    "ChatRole",
    "InferenceChunk",
    "InferenceChunkStream",
    "InferenceImage",
    "InferenceImageSource",
    "InferenceResult",
    "InferenceToolCall",
    "InferenceUsage",
]

# Feature: candle-llm.
EXPECTED_CANDLE_LLM = [
    "CandleLlmProvider",
    "CandleLlmOptions",
    "CandleInferenceResult",
]

# Feature: candle-embed.
EXPECTED_CANDLE_EMBED = [
    "CandleEmbedModel",
    "CandleEmbedOptions",
]

# Feature: embed (FastEmbed; not built on musl targets).
EXPECTED_FASTEMBED = [
    "FastEmbedModel",
    "FastEmbedOptions",
    "EmbedOptions",
]

# Feature: tract.
EXPECTED_TRACT = [
    "TractEmbedModel",
    "TractOptions",
    "TractResponse",
]

# Feature: whispercpp.
EXPECTED_WHISPERCPP = [
    "WhisperCppProvider",
    "WhisperOptions",
    "WhisperModel",
]

# Feature: piper.
EXPECTED_PIPER = [
    "PiperProvider",
    "PiperOptions",
]

# Feature: diffusion.
EXPECTED_DIFFUSION = [
    "DiffusionProvider",
    "DiffusionOptions",
    "DiffusionScheduler",
]

# Feature: otlp (OpenTelemetry exporter).
EXPECTED_OTLP = [
    "OtlpConfig",
    "init_otlp",
]

# Feature: prometheus (no class, function-only).
EXPECTED_PROMETHEUS = [
    "init_prometheus",
]

# Feature: langfuse.
EXPECTED_LANGFUSE = [
    "LangfuseConfig",
    "init_langfuse",
]

# Feature: tiktoken.
EXPECTED_TIKTOKEN = [
    "TiktokenCounter",
]


def _missing(names: list[str]) -> list[str]:
    return [n for n in names if not hasattr(blazen, n)]


def test_core_surface_present() -> None:
    # Every name in EXPECTED_CORE must exist on the installed wheel.
    missing = _missing(EXPECTED_CORE)
    assert not missing, f"missing core attrs on blazen: {missing}"


def test_blazen_error_hierarchy_is_exception() -> None:
    # BlazenError must be an Exception subclass and the parent of typed errors.
    assert issubclass(blazen.BlazenError, Exception)
    for name in (
        "AuthError",
        "RateLimitError",
        "TimeoutError",
        "ValidationError",
        "ContentPolicyError",
        "ProviderError",
        "UnsupportedError",
        "ComputeError",
        "MediaError",
    ):
        cls = getattr(blazen, name)
        assert issubclass(cls, blazen.BlazenError), f"{name} must subclass BlazenError"


def test_llamacpp_surface_present_when_built() -> None:
    if not hasattr(blazen, "LlamaCppProvider"):
        pytest.skip("blazen wheel not built with --features llamacpp")
    missing = _missing(EXPECTED_LLAMACPP)
    assert not missing, f"missing llamacpp attrs: {missing}"


def test_mistralrs_surface_present_when_built() -> None:
    if not hasattr(blazen, "MistralRsProvider"):
        pytest.skip("blazen wheel not built with --features mistralrs")
    missing = _missing(EXPECTED_MISTRALRS)
    assert not missing, f"missing mistralrs attrs: {missing}"


def test_candle_llm_surface_present_when_built() -> None:
    if not hasattr(blazen, "CandleLlmProvider"):
        pytest.skip("blazen wheel not built with --features candle-llm")
    missing = _missing(EXPECTED_CANDLE_LLM)
    assert not missing, f"missing candle-llm attrs: {missing}"


def test_candle_embed_surface_present_when_built() -> None:
    if not hasattr(blazen, "CandleEmbedModel"):
        pytest.skip("blazen wheel not built with --features candle-embed")
    missing = _missing(EXPECTED_CANDLE_EMBED)
    assert not missing, f"missing candle-embed attrs: {missing}"


def test_fastembed_surface_present_when_built() -> None:
    if not hasattr(blazen, "FastEmbedModel"):
        pytest.skip("blazen wheel not built with --features embed (musl excludes FastEmbed)")
    missing = _missing(EXPECTED_FASTEMBED)
    assert not missing, f"missing fastembed attrs: {missing}"


def test_tract_surface_present_when_built() -> None:
    if not hasattr(blazen, "TractEmbedModel"):
        pytest.skip("blazen wheel not built with --features tract")
    missing = _missing(EXPECTED_TRACT)
    assert not missing, f"missing tract attrs: {missing}"


def test_whispercpp_surface_present_when_built() -> None:
    if not hasattr(blazen, "WhisperCppProvider"):
        pytest.skip("blazen wheel not built with --features whispercpp")
    missing = _missing(EXPECTED_WHISPERCPP)
    assert not missing, f"missing whispercpp attrs: {missing}"


def test_piper_surface_present_when_built() -> None:
    if not hasattr(blazen, "PiperProvider"):
        pytest.skip("blazen wheel not built with --features piper")
    missing = _missing(EXPECTED_PIPER)
    assert not missing, f"missing piper attrs: {missing}"


def test_diffusion_surface_present_when_built() -> None:
    if not hasattr(blazen, "DiffusionProvider"):
        pytest.skip("blazen wheel not built with --features diffusion")
    missing = _missing(EXPECTED_DIFFUSION)
    assert not missing, f"missing diffusion attrs: {missing}"


def test_otlp_surface_present_when_built() -> None:
    if not hasattr(blazen, "OtlpConfig"):
        pytest.skip("blazen wheel not built with --features otlp")
    missing = _missing(EXPECTED_OTLP)
    assert not missing, f"missing otlp attrs: {missing}"


def test_prometheus_surface_present_when_built() -> None:
    # Prometheus exposes only `init_prometheus`; if absent the feature is off.
    if not hasattr(blazen, "init_prometheus"):
        pytest.skip("blazen wheel not built with --features prometheus")
    missing = _missing(EXPECTED_PROMETHEUS)
    assert not missing, f"missing prometheus attrs: {missing}"


def test_langfuse_surface_present_when_built() -> None:
    if not hasattr(blazen, "LangfuseConfig"):
        pytest.skip("blazen wheel not built with --features langfuse")
    missing = _missing(EXPECTED_LANGFUSE)
    assert not missing, f"missing langfuse attrs: {missing}"


def test_tiktoken_surface_present_when_built() -> None:
    if not hasattr(blazen, "TiktokenCounter"):
        pytest.skip("blazen wheel not built with --features tiktoken")
    missing = _missing(EXPECTED_TIKTOKEN)
    assert not missing, f"missing tiktoken attrs: {missing}"
