#!/usr/bin/env python3
"""Binding-parity auditor for the Blazen workspace.

Walks every wrapped `crates/blazen-*/src/lib.rs`, harvests its public surface
(both `pub use` re-exports and direct `pub struct/enum/trait/fn/type`
declarations), and diffs that surface against what is actually exposed by
each language binding:

  - Python : `crates/blazen-py/blazen.pyi`
  - Node   : `crates/blazen-node/index.d.ts`
  - WASM   : `crates/blazen-wasm-sdk/pkg/blazen_wasm_sdk.d.ts`

Name-mapping rules applied before comparison:

  - Python: strip a leading `Py` (pyo3 sometimes leaves these around).
  - Node  : strip a leading `Js` (napi-rs convention -- `JsFoo` -> `Foo`).
  - WASM  : strip a leading `Wasm`.

Items intentionally left unbound live in the `WHITELIST` constant. Add to it
as the surfaces stabilise.

Usage:
    python3 tools/audit_bindings.py            # audit + summary
    python3 tools/audit_bindings.py --verbose  # also print whitelist exclusions
    python3 tools/audit_bindings.py -v         # short alias for --verbose

Exit codes:
    0  no gaps (whitelisted items ignored)
    1  one or more gaps detected
    2  configuration / IO error (missing binding files, etc.)

The script depends on the standard library only.
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent

# Crates we wrap behind PyO3 / napi-rs / wasm-bindgen. The bindings/CLI crates
# (blazen-cli, blazen-py, blazen-node, blazen-wasm, blazen-wasm-sdk,
# blazen-macros) are intentionally excluded -- they are the wrappers, not the
# wrapped libraries.
WRAPPED_CRATES: tuple[str, ...] = (
    "blazen-core",
    "blazen-events",
    "blazen-llm",
    "blazen-manager",
    "blazen-memory",
    "blazen-memory-valkey",
    "blazen-pipeline",
    "blazen-prompts",
    "blazen-telemetry",
    "blazen-persist",
    "blazen-peer",
    "blazen-model-cache",
    "blazen-embed",
    "blazen-embed-fastembed",
    "blazen-embed-tract",
    "blazen-embed-candle",
    "blazen-llm-mistralrs",
    "blazen-llm-llamacpp",
    "blazen-llm-candle",
    "blazen-audio-whispercpp",
    "blazen-audio-piper",
    "blazen-image-diffusion",
)

PY_STUB = REPO_ROOT / "crates" / "blazen-py" / "blazen.pyi"
NODE_DTS = REPO_ROOT / "crates" / "blazen-node" / "index.d.ts"
WASM_DTS = (
    REPO_ROOT
    / "crates"
    / "blazen-wasm-sdk"
    / "pkg"
    / "blazen_wasm_sdk.d.ts"
)

# Names that are deliberately not exposed in any binding. Keep this set tight;
# every entry is a deliberate decision, not a TODO.
#
# Categories:
#   1. Errors bound as exceptions (Python: pyo3::create_exception!; Node:
#      napi::Error mapping). The audit's regex doesn't see them as exposed
#      because they're not `class X:` in the .pyi or `export declare class X`
#      in the .d.ts.
#   2. Closure type aliases -- Rust `pub type FooFn = Arc<dyn Fn(...)>;` cannot
#      be bound as a class; they're exposed indirectly via callback parameters.
#   3. Traits that are pure Rust ergonomics and don't surface to bindings.
#      Traits that ARE bound as ABCs (CheckpointStore, MemoryBackend,
#      Middleware, etc.) are NOT in this list -- they should appear as gaps if
#      they're missing from the .pyi/.d.ts.
#   4. Cross-crate aliases -- when a name is exported under a different name
#      in the binding (e.g. SerializedEvent -> PersistedEvent in py/node).
#   5. Workspace/internal items not meant for bindings.
WHITELIST: frozenset[str] = frozenset(
    {
        # --- Errors bound as exceptions (PyO3 create_exception! / napi Error) ---
        "PiperError",
        "WhisperError",
        "CacheError",
        "PersistError",
        "PeerError",
        "PipelineError",
        "PromptError",
        "MemoryError",
        "BlazenError",
        "LlmError",
        "LlamaCppError",
        "MistralRsError",
        "CandleEmbedError",
        "CandleLlmError",
        "DiffusionError",
        "TractError",
        "EmbedError",
        "FastEmbedError",
        "WorkflowError",
        "SessionRefError",
        "CompletionErrorKind",
        "ComputeErrorKind",
        "MediaErrorKind",
        # --- Closure type aliases -- not bindable as standalone types ---
        "StepFn",
        "ConditionFn",
        "InputMapperFn",
        "PersistFn",
        "PersistJsonFn",
        "EventDeserializerFn",
        "SessionRefDeserializerFn",
        "StepBuilderFn",
        "InputHandlerFn",
        # `pub use submod::{SubWorkflowInputMapper, SubWorkflowOutputMapper}`
        # in `crates/blazen-core/src/lib.rs` re-exports lose the
        # KIND_TYPE_ALIAS auto-skip (the auditor tags re-exports as
        # KIND_REEXPORT, see audit_bindings.py:813-814), so they need
        # explicit whitelist entries even though they're closure aliases.
        "SubWorkflowInputMapper",
        "SubWorkflowOutputMapper",
        # --- Traits exposed only as Rust ergonomics (not as ABCs in bindings) ---
        # Traits that ARE bound as ABCs (CheckpointStore, MemoryBackend,
        # Middleware, PeerClient, MemoryStore, ProgressCallback) intentionally
        # stay OUT of this list -- they should be reported as gaps if missing.
        "Event",
        "AnyEvent",
        "Tool",
        "ModelRegistry",
        "ProviderInfo",
        "ProviderCapabilities",
        "StructuredOutput",
        "SessionRefSerializable",
        "LocalModel",
        # --- Already bound under aliased names -- known cross-crate collisions ---
        # blazen-core::SerializedEvent is bound as PersistedEvent in py/node to
        # avoid a name collision with blazen-persist's own SerializedEvent.
        "SerializedEvent",
        # --- Internal types that flow through bound umbrella/wrapper classes ---
        # These are concrete provider-specific shapes that the bindings expose
        # only via a higher-level type (e.g. EmbeddingResponse wraps
        # FastEmbedResponse / TractResponse; the chat-message ABCs subsume the
        # mistralrs ChatRole/ChatMessageInput pair). The umbrella IS bound,
        # so callers do not need the inner type directly.
        "MediaSource",
        "FetchHttpClient",
        "ReqwestHttpClient",
        "TracingCompletionModel",
        "CandleLlmCompletionModel",
        "CandleEmbedModel",
        "FastEmbedResponse",
        "MetricsLayer",
        # --- Rust-only enums-with-data that don't bind cleanly ---
        # `Device` is a Rust enum carrying a per-variant device-index payload
        # (e.g. `Cuda(usize)`, `Vulkan(usize)`). It cannot be expressed as a
        # JS string enum. Bindings accept device strings ("cuda:0", "metal",
        # etc.) at the call site instead.
        "Device",
        # `EmbedModel` / `EmbedResponse` / `EmbedOptions` in `blazen-llm`
        # are target-conditional aliases for either the fastembed or tract
        # concrete types, depending on `target_env`. Bindings expose the
        # concrete types (FastEmbedModel + TractEmbedModel) directly plus
        # an `EmbedProvider` facade — the alias itself is not bindable as
        # a distinct class.
        "EmbedModel",
        "EmbedResponse",
        "EmbedOptions",
        # --- Workspace/internal items not meant for bindings ---
        "Result",  # type alias for Result<T, FooError> in many crates
        # Free functions that are inherently Rust-only (registries, interning,
        # raw deserializer plumbing). Bindings expose these implicitly via
        # snapshot/restore APIs.
        "register_event_deserializer",
        "try_deserialize_event",
        "intern_event_type",
        "register_step_builder",
        "lookup_step_builder",
        "registered_step_ids",
        "register_pricing",
        "lookup_pricing",
        "compute_cost",
        "current_session_registry",
        "with_session_registry",
        # Rust-side static / lifetime types.
        "CURRENT_SESSION_REGISTRY",
        "SERIALIZED_SESSION_REFS_META_KEY",
        "SESSION_REF_TAG",
        "SNAPSHOT_VERSION",
        "RefLifetime",
        "RegistryKey",
        # `LangfuseLayer` is the `tracing-subscriber` Layer surfaced via
        # `init_langfuse` -- internal, mirrors how `MetricsLayer` is internal
        # to `init_prometheus`. The Config + init_* functions ARE exposed.
        "LangfuseLayer",
        # Internal telemetry error type -- never exposed standalone; errors
        # propagate as plain strings via the `init_*` functions.
        "TelemetryError",
        # HTTP variant of OTLP init -- wasm-sdk uses this internally to back
        # its `initOtlp`; native bindings (py + node) only expose `init_otlp`
        # (the gRPC variant), so this name has no native binding consumer.
        "init_otlp_http",
        # Wasm-internal Tract types -- only compiled under `target_arch =
        # "wasm32"` in `blazen-embed-tract::wasm_provider`. Native bindings
        # (py + node) shouldn't expose them; the wasm-sdk re-binds the
        # underlying class under the JS name `TractEmbedModel` (which IS
        # already exposed via the regular `TractEmbedModel` whitelist entry).
        "WasmTractEmbedModel",
        "WasmTractError",
        "WasmTractResponse",
        # Module-level constants attached via PyModule::add() / napi env
        # exports. The auditor's regex looks for `class X:` / `export
        # declare class X` and doesn't see module attribute literals.
        "ENVELOPE_VERSION",
        # Bound under the `JsMediaType` / `JsQuantization` napi-rs aliases;
        # the prefix-strip already accepts them, but list them explicitly so
        # a Python build that omits the same names doesn't fail the audit.
        "MediaType",
        "Quantization",
        # --- Pipeline internal types not bindable as standalone classes ---
        # `PipelineState` has only `pub(crate)` constructors; it is created
        # internally by `Pipeline::run` and threaded through stage closures
        # via the workflow `Context`. The Node + WASM bindings expose its
        # mutators (`get`, `set`, `last_result`, `stage_result`) directly on
        # the Pipeline / Stage handler closure parameters instead of as a
        # standalone JS class.
        "PipelineState",
        # `StageKind` is a Rust enum-with-data (`Sequential(Stage)` /
        # `Parallel(ParallelStage)`) that cannot round-trip through a JS
        # string enum. Both Node and WASM expose `Stage` and `ParallelStage`
        # as separate classes, which is the JS-idiomatic equivalent of the
        # Rust discriminated union. WASM additionally surfaces a tag-only
        # `StageKind` enum (Sequential / Parallel) for snapshot inspection;
        # Node currently has no consumer for the discriminant.
        "StageKind",
        # `StepKind` is the workflow analogue of `StageKind`: a Rust
        # enum-with-data (`Regular(StepRegistration)` /
        # `SubWorkflow(SubWorkflowStep)` /
        # `ParallelSubWorkflows(ParallelSubWorkflowsStep)`) that cannot
        # round-trip through a JS string enum. Bindings expose the
        # variants as separate classes (Step / SubWorkflowStep /
        # ParallelSubWorkflowsStep), which is the JS-idiomatic equivalent
        # of the discriminated union.
        "StepKind",
        # Module-level string constant exposed via `m.add(...)` in the
        # PyO3 binding (see `crates/blazen-py/src/lib.rs:92`) and via a
        # `#[napi] pub const` re-export in the napi binding -- the
        # auditor's `class X:` / `export declare class X` regex doesn't
        # see module-attribute literals.
        "FINISH_WORKFLOW_TOOL_NAME",
        # `StepDeserializerRegistry` is a Rust runtime registry parameterised
        # by raw `fn` pointers (typed-step builders); function pointers cannot
        # round-trip through PyO3 / napi-rs / wasm-bindgen, so the registry
        # is exposed indirectly via the snapshot/restore APIs that read from
        # it (the registry itself is filled at module init from `inventory`
        # entries on the Rust side).
        "StepDeserializerRegistry",
    }
)

# Items that physically cannot be exposed by the WASM SDK because the
# underlying crate is native-only or its dependency chain (tokio fs/rt,
# h2/tonic, hf-hub, ort/onnxruntime, llama.cpp, candle, mistral.rs, whisper.cpp,
# piper, diffusion, fastembed, tract, redb, valkey, otlp, langfuse, prometheus,
# peer/tls, etc.) does not compile to wasm32.
#
# These names ARE bound in the Python and Node bindings where applicable. The
# WASM auditor filters them out so they don't show up as gaps that nobody can
# fix without a wasm-friendly rewrite of the upstream crate.
#
# Categories (kept in section blocks for grep-friendliness):
#   1. FastEmbed / Tract / Candle embedding backends.
#   2. Candle / llama.cpp / mistral.rs LLM backends and their request/response
#      types (`ChatMessageInput`, `ChatRole`, `InferenceChunk`, etc.).
#   3. Audio backends (whisper.cpp, piper) and image diffusion.
#   4. Peer/RPC server + client (tonic/h2/tls -- never wasm32).
#   5. Persistence (redb, valkey) and `ProgressCallback` / `ModelCache` (rely on
#      tokio fs and a real filesystem).
#   6. Telemetry exporters (OTLP, Langfuse, Prometheus) -- pull in tonic /
#      reqwest server-side / metrics-process which are native-only.
WASM_SKIP: frozenset[str] = frozenset(
    {
        # --- FastEmbed (ONNX Runtime via `ort`) ---
        "FastEmbedModel",
        "FastEmbedOptions",
        "FastEmbedResponse",
        "FastEmbedError",
        # --- Tract embeddings (now bound on wasm32 via fetch loader) ---
        # --- Candle embeddings ---
        "CandleEmbedModel",
        "CandleEmbedError",
        "CandleEmbedOptions",
        # --- Candle LLM ---
        "CandleLlmProvider",
        "CandleLlmCompletionModel",
        "CandleLlmOptions",
        "CandleLlmError",
        "CandleInferenceResult",
        # --- llama.cpp (native FFI bindings) ---
        "LlamaCppProvider",
        "LlamaCppOptions",
        "LlamaCppError",
        "ChatMessageInput",
        "ChatRole",
        "InferenceChunk",
        "InferenceChunkStream",
        "InferenceResult",
        "InferenceUsage",
        # Prefixed re-exports added in W2 so py + node can bind both
        # mistralrs's un-prefixed and llamacpp's prefixed types side by side.
        # llama.cpp is native-FFI-only, so the prefixed names also stay
        # native-only on wasm32.
        "LlamaCppChatMessageInput",
        "LlamaCppChatRole",
        "LlamaCppInferenceChunk",
        "LlamaCppInferenceChunkStream",
        "LlamaCppInferenceResult",
        "LlamaCppInferenceUsage",
        # --- mistral.rs (native, GPU) ---
        "MistralRsProvider",
        "MistralRsOptions",
        "MistralRsError",
        "InferenceImage",
        "InferenceImageSource",
        "InferenceToolCall",
        # --- whisper.cpp (audio transcription, native) ---
        "WhisperCppProvider",
        "WhisperOptions",
        "WhisperModel",
        "WhisperError",
        # --- piper (TTS, native) ---
        "PiperProvider",
        "PiperOptions",
        "PiperError",
        # --- diffusion (native image gen) ---
        "DiffusionProvider",
        "DiffusionOptions",
        "DiffusionScheduler",
        "DiffusionError",
        # --- Peer/RPC (tonic + h2 + TLS -- never wasm32) ---
        # `PeerClient` ABC + `RemoteWorkflowRequest`/`RemoteWorkflowResponse`
        # data envelopes ARE bound on wasm32 via a JS-callback adapter
        # (see crates/blazen-wasm-sdk/src/core_types/distributed.rs); only
        # the tonic-backed concrete transport (`BlazenPeerServer`,
        # `BlazenPeerClient`, the gRPC envelope structs, and TLS helpers)
        # remain native-only.
        "BlazenPeerServer",
        "BlazenPeerClient",
        "PeerError",
        "SubWorkflowRequest",
        "SubWorkflowResponse",
        "DerefRequest",
        "DerefResponse",
        "ReleaseRequest",
        "ReleaseResponse",
        "PeerRemoteRefDescriptor",
        "ENVELOPE_VERSION",
        "PEER_TOKEN_ENV",
        "resolve_peer_token",
        "load_server_tls",
        "load_client_tls",
        # --- Persistence (redb, valkey -- native-only filesystems / sockets) ---
        "WorkflowCheckpoint",
        "PersistedEvent",
        "CheckpointStore",
        "RedbCheckpointStore",
        "ValkeyCheckpointStore",
        "PersistError",
        # --- Memory backends (host-dispatch ABC; wasm-sdk uses
        # `Memory.fromJsBackend` for custom JS-backed stores) ---
        # `MemoryBackend` itself is the host-dispatch ABC trait; wasm-sdk
        # uses `InMemoryBackend` (W4-G) and `Memory.fromJsBackend` instead
        # of binding the trait directly.
        "MemoryBackend",
        "JsonlBackend",
        "ValkeyBackend",
        # --- Manager (Phase 9 wasm-sdk binding pending) ---
        "ModelStatus",
        # --- Model cache + progress callback (tokio fs) ---
        "ModelCache",
        "CacheError",
        "ProgressCallback",
        # --- Telemetry exporters (native-only remainder) ---
        # OtlpConfig + init_otlp are bound on wasm32 via the OTLP HTTP
        # exporter; LangfuseConfig + init_langfuse are bound on wasm32 via
        # web_sys::fetch (see crates/blazen-wasm-sdk/src/telemetry/langfuse.rs).
        # Only Layer handles + the Prometheus pull collector remain native-only:
        # `*Layer` are tracing-subscriber Layer types surfaced via init_*
        # globals; the Prometheus collector binds a TCP listener that wasm32
        # can't open.
        "LangfuseLayer",
        "MetricsLayer",
        "init_prometheus",
        # --- Bundled tiktoken (opt-in via wasm-sdk `tiktoken` feature) ---
        # `TiktokenCounter` is exposed on wasm32 ONLY when blazen-wasm-sdk is
        # built with `--features tiktoken` (intended for the separate
        # `@blazen-dev/sdk-tiktoken` npm package). The default `@blazen-dev/sdk`
        # bundle leaves it out because the embedded BPE tables add ~8 MB. The
        # always-on `TokenCounter` JS-callback class is the bundle-friendly
        # escape hatch.
        "TiktokenCounter",
        # --- `EmbedModel`/`EmbedResponse`/`EmbedOptions`/`EmbedError` are
        # struct re-exports from `blazen-embed` aliasing FastEmbed/Tract.
        # The runtime trait `EmbeddingModel` IS bound (as `EmbeddingModel`);
        # the alias names below are skip-listed for the same reason their
        # underlying types are. ---
        "EmbedModel",
        "EmbedResponse",
        "EmbedOptions",
        "EmbedError",
        # --- blazen-llm types not surfaced as standalone wasm-bindgen classes ---
        # `runAgent()` and `completeBatch()` are exposed as free async
        # functions that resolve to plain JS objects whose shape is
        # documented inline; there is no `class AgentResult` or
        # `class BatchResult` on the WASM side because wasm-bindgen cannot
        # auto-derive a class with arbitrary nested `JsValue` payloads
        # (tool calls, per-call usage, citations, etc.). The Python and
        # Node bindings use the umbrella `CompletionResponse` / a typed
        # batch-result class instead.
        "AgentResult",
        "BatchResult",
        # `ChatWindow` is a sliding-window helper over chat history. It is
        # bound in PyO3 (PyChatWindow) and napi-rs (JsChatWindow) but the
        # WASM SDK currently leaves windowing to TS-side helpers because
        # the Rust struct holds an `Arc<dyn TokenCounter>` whose dyn-dispatch
        # round-trip into JS is not yet wired up.
        "ChatWindow",
        # `CustomProvider` + `HostDispatch` are the host-dispatch facade
        # used by Python and Node to wrap a user-supplied `complete()`
        # callback as a Rust `CompletionModel`. WASM achieves the same
        # capability via `OpenAiCompatProvider` (which already accepts a
        # custom base URL + header map) plus the `HttpClient` host-dispatch
        # class, so the standalone `CustomProvider` class is not duplicated.
        # `HostDispatch` is a pure Rust internal pattern (an Arc<dyn ...>
        # closure) -- it has no equivalent class in any binding.
        "CustomProvider",
        "HostDispatch",
        # `EstimateCounter` is the heuristic token counter. The WASM SDK
        # exposes it via `TokenCounter.estimate()` and the free
        # `estimateTokens()` function instead of as a standalone class --
        # spawning a separate JS class for it would duplicate the
        # `TokenCounter` surface without adding capability.
        "EstimateCounter",
        # `PricingEntry` is exposed as a plain JS object literal
        # (`{ inputPerMillion, outputPerMillion }`) returned from
        # `lookupPricing()` rather than as a class, mirroring how
        # wasm-bindgen idiomatically surfaces small POD records.
        "PricingEntry",
        # `StructuredResponse` carries a typed payload + the raw
        # completion. The WASM SDK exposes structured output via the
        # `responseFormat` field on `CompletionRequest` and returns the
        # parsed JSON inline on `CompletionResponse.content` -- a separate
        # `StructuredResponse` class would duplicate state.
        "StructuredResponse",
        # --- blazen-prompts (template registry / file loader) ---
        # The prompt-template subsystem is bound in PyO3 (PyPromptTemplate /
        # PyPromptRegistry) and napi-rs (JsPromptTemplate / JsPromptRegistry)
        # but not yet wired into the WASM SDK -- the upstream crate's file
        # loader uses `tokio::fs` and a glob crawler that don't compile to
        # wasm32. A wasm-friendly template loader (fed in via `fetch()` or
        # raw strings) is a follow-up; until then these names are skip-listed.
        "PromptTemplate",
        "PromptRegistry",
        "PromptFile",
        "TemplateRole",
    }
)

# Reasons we excluded each whitelisted name -- used by --verbose to show *why*
# an item was filtered. Anything in WHITELIST that is missing from this map
# falls back to "whitelisted (no reason recorded)".
WHITELIST_REASONS: dict[str, str] = {
    # Errors
    **{
        n: "error type bound as exception (PyO3 create_exception! / napi Error)"
        for n in (
            "PiperError",
            "WhisperError",
            "CacheError",
            "PersistError",
            "PeerError",
            "PipelineError",
            "PromptError",
            "MemoryError",
            "BlazenError",
            "LlmError",
            "LlamaCppError",
            "MistralRsError",
            "CandleEmbedError",
            "CandleLlmError",
            "DiffusionError",
            "TractError",
            "EmbedError",
            "FastEmbedError",
            "WorkflowError",
            "SessionRefError",
            "CompletionErrorKind",
            "ComputeErrorKind",
            "MediaErrorKind",
        )
    },
    # Closure aliases
    **{
        n: "closure type alias (Arc<dyn Fn(...)>) -- exposed via callback params"
        for n in (
            "StepFn",
            "ConditionFn",
            "InputMapperFn",
            "PersistFn",
            "PersistJsonFn",
            "EventDeserializerFn",
            "SessionRefDeserializerFn",
            "StepBuilderFn",
            "InputHandlerFn",
            "SubWorkflowInputMapper",
            "SubWorkflowOutputMapper",
        )
    },
    # Pure-Rust traits
    **{
        n: "trait used only as Rust ergonomics -- not surfaced to bindings"
        for n in (
            "Event",
            "AnyEvent",
            "Tool",
            "ModelRegistry",
            "ProviderInfo",
            "ProviderCapabilities",
            "StructuredOutput",
            "SessionRefSerializable",
            "LocalModel",
        )
    },
    # Cross-crate aliases
    "SerializedEvent": "bound under aliased name PersistedEvent in py/node",
    "Device": "Rust enum-with-data (Cuda(usize), Vulkan(usize), ...) -- bindings accept a device string",
    "EmbedModel": "target-conditional alias for FastEmbedModel / TractEmbedModel (concrete types are bound separately)",
    "EmbedResponse": "target-conditional alias for FastEmbedResponse / TractResponse (concrete types are bound separately)",
    "EmbedOptions": "target-conditional alias for FastEmbedOptions / TractOptions (concrete types are bound separately)",
    # Internal provider-specific shapes that flow through a bound umbrella type
    "MediaSource": "type alias for ImageSource (already bound)",
    "FetchHttpClient": "wasm32-only HTTP client; native bindings use ReqwestHttpClient via PyHttpClient",
    "ReqwestHttpClient": "internal reqwest HTTP client; surfaced via PyHttpClient",
    "TracingCompletionModel": "wrapper class -- surfaced via the wrap_with_tracing free fn",
    "CandleLlmCompletionModel": "candle-llm trait-bridge wrapper -- exposed via PyCandleLlmProvider",
    "CandleEmbedModel": "candle-embed type bound as CandleEmbedProvider in node",
    "FastEmbedResponse": "internal -- exposed via PyEmbeddingResponse umbrella",
    "MetricsLayer": "tracing-subscriber Layer -- internal, surfaced via init_prometheus",
    # Workspace internals
    "Result": "Rust Result<T, E> alias -- not a bindable type",
    "register_event_deserializer": "Rust-only registry plumbing",
    "try_deserialize_event": "Rust-only registry plumbing",
    "intern_event_type": "Rust-only registry plumbing",
    "register_step_builder": "Rust-only registry plumbing",
    "lookup_step_builder": "Rust-only registry plumbing",
    "registered_step_ids": "Rust-only registry plumbing",
    "register_pricing": "Rust-only registry plumbing",
    "lookup_pricing": "Rust-only registry plumbing",
    "compute_cost": "Rust-only registry plumbing",
    "current_session_registry": "Rust-only registry plumbing",
    "with_session_registry": "Rust-only registry plumbing",
    "CURRENT_SESSION_REGISTRY": "Rust-side static",
    "SERIALIZED_SESSION_REFS_META_KEY": "Rust-side static",
    "SESSION_REF_TAG": "Rust-side static",
    "SNAPSHOT_VERSION": "Rust-side static",
    "RefLifetime": "Rust-side lifetime helper",
    "RegistryKey": "Rust-side registry key type",
    "LangfuseLayer": "tracing-subscriber Layer -- internal, surfaced via init_langfuse",
    "TelemetryError": "internal error type; not exposed by any binding",
    "init_otlp_http": (
        "internal-only HTTP variant; wasm-sdk's `initOtlp` calls it; native uses gRPC variant"
    ),
    "WasmTractEmbedModel": (
        "wasm32-internal -- bound on wasm-sdk under JS name `TractEmbedModel` via "
        "WasmTractEmbedModel class"
    ),
    "WasmTractError": (
        "wasm32-internal -- errors surface via TractError on wasm-sdk"
    ),
    "WasmTractResponse": (
        "wasm32-internal -- response shape exposed via TractResponse on wasm-sdk"
    ),
    # Module-level constants and napi-aliased enums.
    "ENVELOPE_VERSION": "module-level constant, not a class",
    "FINISH_WORKFLOW_TOOL_NAME": (
        "module-level &'static str constant; bound via PyModule::add() / "
        "napi const re-export, not visible in the auditor's class-regex"
    ),
    "MediaType": "bound as JsMediaType in napi (string enum)",
    "Quantization": "bound as JsQuantization in napi (string enum)",
    # Pipeline internals + Rust enum-with-data + fn-pointer registry.
    "PipelineState": (
        "internal -- pub(crate) ctor, exposed via Stage handler context"
    ),
    "StageKind": (
        "Rust enum-with-data (Sequential(Stage) / Parallel(ParallelStage)) -- "
        "bindings expose Stage + ParallelStage as separate classes"
    ),
    "StepKind": (
        "Rust enum-with-data (Regular(StepRegistration) / SubWorkflow(SubWorkflowStep) / "
        "ParallelSubWorkflows(ParallelSubWorkflowsStep)) -- bindings expose the variants "
        "as separate classes (Step / SubWorkflowStep / ParallelSubWorkflowsStep)"
    ),
    "StepDeserializerRegistry": (
        "fn-pointer registry -- not bindable across FFI; surfaced via snapshot APIs"
    ),
}

# Item-kind tags. Used to route certain kinds of items into separate report
# subsections (traits) and to auto-skip others (type aliases).
KIND_STRUCT = "struct"
KIND_ENUM = "enum"
KIND_TRAIT = "trait"
KIND_FN = "fn"
KIND_TYPE_ALIAS = "type-alias"
KIND_CONST = "const"
KIND_STATIC = "static"
KIND_REEXPORT = "reexport"

# ---------------------------------------------------------------------------
# Rust public-surface extraction
# ---------------------------------------------------------------------------

# `pub use foo::Bar;` and `pub use foo::{Bar, Baz};` (single line or multi-line).
# We strip `as Alias` segments by capturing only the leading identifier.
# The block-form pattern is applied after collapsing whitespace/newlines.
#
# Anchored at column zero -- inner `pub use` re-exports inside private
# modules don't escape the crate and shouldn't be audited.
_PUB_USE_BLOCK_RE = re.compile(
    r"^pub\s+use\s+([^;]+);",
    re.MULTILINE,
)

# Direct declarations on lib.rs (when there is no `pub use`, the items are
# defined inline -- e.g. blazen-events).
#
# Anchored at column zero so we don't accidentally capture methods inside
# `impl` blocks (`    pub fn new(...)`). `lib.rs` items we care about always
# live at the top level.
#
# Captures both the kind keyword and the identifier so we can tag items as
# traits / type-aliases / etc. and route them into separate report sections.
_PUB_DECL_RE = re.compile(
    r"^pub\s+(struct|enum|trait|fn|type|const|static)\s+([A-Za-z_][A-Za-z0-9_]*)",
    re.MULTILINE,
)

# Dedicated regexes for the two kinds we surface specially. `_PUB_DECL_RE`
# already covers them -- these are kept around as documented references for
# the per-task acceptance criteria.
_PUB_TYPE_ALIAS_RE = re.compile(r"^\s*pub type (\w+)", re.MULTILINE)
_PUB_TRAIT_RE = re.compile(r"^\s*pub trait (\w+)", re.MULTILINE)

# Kind keyword (as it appears in Rust source) -> our internal KIND_* tag.
_KIND_KEYWORD_MAP: dict[str, str] = {
    "struct": KIND_STRUCT,
    "enum": KIND_ENUM,
    "trait": KIND_TRAIT,
    "fn": KIND_FN,
    "type": KIND_TYPE_ALIAS,
    "const": KIND_CONST,
    "static": KIND_STATIC,
}

# Match an identifier optionally suffixed with `as Alias`. We always take the
# *exported* name -- the alias if present, otherwise the original.
_USE_ITEM_RE = re.compile(
    r"([A-Za-z_][A-Za-z0-9_]*)\s*(?:as\s+([A-Za-z_][A-Za-z0-9_]*))?"
)


def _strip_block_comments(source: str) -> str:
    """Best-effort removal of `/* ... */` blocks.

    Doc comments (`//!`, `///`) and line comments (`//`) are handled by the
    line-anchored regex itself, but block comments can hide a `pub use` start
    so we drop them up-front. This is intentionally simple -- it does not
    handle nested blocks (Rust does), but those are vanishingly rare in
    `lib.rs` files.
    """
    return re.sub(r"/\*.*?\*/", "", source, flags=re.DOTALL)


def _split_use_items(items_str: str) -> list[str]:
    """Split the body of a `pub use foo::{...};` clause into exported names.

    Handles:
      - `Foo`                       -> `Foo`
      - `Foo as Bar`                -> `Bar`
      - `module::Foo`               -> `Foo`
      - `module::{Foo, Bar as Baz}` -> `Foo`, `Baz`
      - leading `crate_name::`      -> stripped
      - trailing `*`                -> dropped (glob re-export, not auditable)
    """
    # Drop the leading path segments before the optional brace group.
    # `foo::bar::{X, Y}` -> `X, Y`
    # `foo::Bar`         -> `Bar`
    # `foo::*`           -> `*`
    items_str = items_str.strip()

    # Find the right-most `::` outside of any brace group, and drop the prefix.
    brace_idx = items_str.find("{")
    if brace_idx >= 0:
        # Path prefix lives before the brace; the brace contents are the items.
        body = items_str[brace_idx + 1 :].rstrip()
        if body.endswith("}"):
            body = body[:-1]
        # Recurse into each comma-separated entry.
        parts = _split_top_level_commas(body)
    else:
        # No braces -- single item like `foo::bar::Baz` or `Baz as Other`.
        # Take everything after the last `::`.
        last = items_str.rsplit("::", 1)[-1]
        parts = [last]

    names: list[str] = []
    for part in parts:
        part = part.strip()
        if not part or part == "*":
            continue
        # Inside braces an entry can itself be a nested path like
        # `submod::{A, B}`. Recurse to flatten.
        if "{" in part:
            names.extend(_split_use_items(part))
            continue
        # Strip any path prefix on the per-item form.
        item_tail = part.rsplit("::", 1)[-1]
        match = _USE_ITEM_RE.match(item_tail)
        if match is None:
            continue
        original, alias = match.group(1), match.group(2)
        names.append(alias if alias else original)
    return names


def _split_top_level_commas(body: str) -> list[str]:
    """Split a brace body on commas, respecting nested `{...}` groups."""
    parts: list[str] = []
    depth = 0
    buf: list[str] = []
    for ch in body:
        if ch == "{":
            depth += 1
            buf.append(ch)
        elif ch == "}":
            depth -= 1
            buf.append(ch)
        elif ch == "," and depth == 0:
            parts.append("".join(buf))
            buf = []
        else:
            buf.append(ch)
    if buf:
        parts.append("".join(buf))
    return parts


@dataclass
class CrateSurface:
    """The public surface of one wrapped crate.

    `items` is the legacy bag of names (kept for backwards compatibility with
    the `diff_surface` flow). `kinds` records what we *think* each item is
    based on its `pub` declaration -- traits and type aliases get routed into
    their own report sections.
    """

    crate: str
    items: set[str] = field(default_factory=set)
    kinds: dict[str, str] = field(default_factory=dict)

    def kind_of(self, item: str) -> str:
        """Return the recorded kind for `item`, defaulting to KIND_REEXPORT.

        Items pulled from `pub use` re-exports don't have a kind we can
        observe locally (the source crate has it, but for the auditor's
        purposes we treat them as opaque).
        """
        return self.kinds.get(item, KIND_REEXPORT)


def collect_rust_surface(crate_dir: Path) -> CrateSurface:
    """Extract the public surface from a crate's `src/lib.rs`."""
    lib_rs = crate_dir / "src" / "lib.rs"
    surface = CrateSurface(crate=crate_dir.name)
    if not lib_rs.is_file():
        return surface

    raw = lib_rs.read_text(encoding="utf-8")
    src = _strip_block_comments(raw)

    for match in _PUB_USE_BLOCK_RE.finditer(src):
        items_str = match.group(1)
        # Drop any `#[cfg(...)]` style annotations that ended up on the same
        # capture (rare, but possible after comment stripping).
        items_str = items_str.strip()
        for name in _split_use_items(items_str):
            surface.items.add(name)
            # Re-exports keep their default KIND_REEXPORT tag unless a local
            # `pub trait`/`pub type` etc. overrides them below.

    for match in _PUB_DECL_RE.finditer(src):
        keyword, ident = match.group(1), match.group(2)
        surface.items.add(ident)
        surface.kinds[ident] = _KIND_KEYWORD_MAP.get(keyword, KIND_REEXPORT)

    return surface


# ---------------------------------------------------------------------------
# Binding-side surface extraction
# ---------------------------------------------------------------------------

_PY_CLASS_RE = re.compile(r"^class\s+([A-Za-z_][A-Za-z0-9_]*)\s*[:(]", re.MULTILINE)
_PY_DEF_RE = re.compile(r"^def\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(", re.MULTILINE)
_PY_ALL_RE = re.compile(r"\"([A-Za-z_][A-Za-z0-9_]*)\"")

_NODE_CLASS_RE = re.compile(
    r"^export\s+declare\s+class\s+([A-Za-z_][A-Za-z0-9_]*)", re.MULTILINE
)
_NODE_FN_RE = re.compile(
    r"^export\s+declare\s+function\s+([A-Za-z_][A-Za-z0-9_]*)", re.MULTILINE
)
_NODE_IFACE_RE = re.compile(
    r"^export\s+interface\s+([A-Za-z_][A-Za-z0-9_]*)", re.MULTILINE
)
_NODE_TYPE_RE = re.compile(
    r"^export\s+type\s+([A-Za-z_][A-Za-z0-9_]*)", re.MULTILINE
)
# napi-rs emits string enums and numeric enums as `declare const enum`,
# which is neither a class nor an interface. Capture them so the audit
# sees enums like `JsCacheStrategy`, `JsJobStatus`, `SessionPausePolicy`,
# etc. that the bindings expose.
_NODE_CONST_ENUM_RE = re.compile(
    r"^export\s+declare\s+const\s+enum\s+([A-Za-z_][A-Za-z0-9_]*)",
    re.MULTILINE,
)

_WASM_CLASS_RE = re.compile(r"^export\s+class\s+([A-Za-z_][A-Za-z0-9_]*)", re.MULTILINE)
_WASM_FN_RE = re.compile(r"^export\s+function\s+([A-Za-z_][A-Za-z0-9_]*)", re.MULTILINE)
_WASM_IFACE_RE = re.compile(
    r"^export\s+interface\s+([A-Za-z_][A-Za-z0-9_]*)", re.MULTILINE
)
_WASM_TYPE_RE = re.compile(r"^export\s+type\s+([A-Za-z_][A-Za-z0-9_]*)", re.MULTILINE)
# wasm-bindgen emits Rust C-like enums as `export enum X { ... }` in the .d.ts.
# These are neither classes, interfaces, nor type aliases, so we need a
# dedicated regex to see them. Without this, JS-visible enums like
# `JoinStrategy` and `StageKind` were reported as gaps.
_WASM_ENUM_RE = re.compile(r"^export\s+enum\s+([A-Za-z_][A-Za-z0-9_]*)", re.MULTILINE)


def collect_py_surface(path: Path) -> set[str]:
    """Names exposed in the PyO3 stub file."""
    if not path.is_file():
        return set()
    text = path.read_text(encoding="utf-8")
    names: set[str] = set()
    names.update(_PY_CLASS_RE.findall(text))
    # Top-level `def` only -- methods are indented and the regex is line-anchored.
    names.update(_PY_DEF_RE.findall(text))
    # `__all__` carries the canonical export list; pull from there too so we
    # catch enums and constants that don't appear as `class X:`.
    if (all_match := re.search(r"__all__\s*=\s*\[(.*?)\]", text, re.DOTALL)) is not None:
        names.update(_PY_ALL_RE.findall(all_match.group(1)))
    return names


def collect_node_surface(path: Path) -> set[str]:
    """Names exposed in the napi-rs `index.d.ts`."""
    if not path.is_file():
        return set()
    text = path.read_text(encoding="utf-8")
    names: set[str] = set()
    names.update(_NODE_CLASS_RE.findall(text))
    names.update(_NODE_FN_RE.findall(text))
    names.update(_NODE_IFACE_RE.findall(text))
    names.update(_NODE_TYPE_RE.findall(text))
    names.update(_NODE_CONST_ENUM_RE.findall(text))
    return names


def collect_wasm_surface(path: Path) -> set[str] | None:
    """Names exposed in the wasm-bindgen `.d.ts`. Returns None if missing."""
    if not path.is_file():
        return None
    text = path.read_text(encoding="utf-8")
    names: set[str] = set()
    names.update(_WASM_CLASS_RE.findall(text))
    names.update(_WASM_FN_RE.findall(text))
    names.update(_WASM_IFACE_RE.findall(text))
    names.update(_WASM_TYPE_RE.findall(text))
    names.update(_WASM_ENUM_RE.findall(text))
    return names


# ---------------------------------------------------------------------------
# Name normalisation
# ---------------------------------------------------------------------------


def _strip_prefix(name: str, prefix: str) -> str:
    """Return `name` with `prefix` removed if the suffix is still PascalCase.

    We only strip if what follows the prefix starts with an uppercase letter,
    which prevents mangling identifiers like `Pyramid` (does not start with
    `Py` followed by an uppercase letter -- `r` is lowercase, so it stays).
    """
    if name.startswith(prefix) and len(name) > len(prefix):
        rest = name[len(prefix) :]
        if rest[0].isupper():
            return rest
    return name


def _snake_to_camel(name: str) -> str:
    """Convert a snake_case identifier to camelCase.

    napi-rs and wasm-bindgen both rename Rust free functions like
    `complete_batch` to camelCase (`completeBatch`) on the JS side. The Rust
    public-surface harvester captures the original snake_case name, so we
    need to fold the Rust name into camelCase before checking against the
    .d.ts surface. Identifiers that contain no underscores are returned
    unchanged so PascalCase types (`PipelineState`) flow through untouched.
    """
    if "_" not in name:
        return name
    head, _, tail = name.partition("_")
    parts = tail.split("_")
    return head + "".join(p[:1].upper() + p[1:] for p in parts if p)


def _camel_to_snake(name: str) -> str:
    """Convert a camelCase identifier to snake_case.

    Inverse of `_snake_to_camel`. Used to project JS-side names like
    `completeBatch` back to their Rust originals (`complete_batch`) so the
    diff against the snake_case Rust public surface matches. Names already
    in snake_case (no internal uppercase) round-trip unchanged. PascalCase
    type names (`PipelineState`) are returned untouched -- the napi-rs /
    wasm-bindgen rename only applies to free functions, never types.
    """
    if not name or not any(c.isupper() for c in name[1:]):
        return name
    if name[:1].isupper():
        # PascalCase type name -- not a renamed function, leave alone.
        return name
    out: list[str] = [name[0]]
    for ch in name[1:]:
        if ch.isupper():
            out.append("_")
            out.append(ch.lower())
        else:
            out.append(ch)
    return "".join(out)


def normalise_py(name: str) -> str:
    return _strip_prefix(name, "Py")


def normalise_node(name: str) -> str:
    return _strip_prefix(name, "Js")


def normalise_wasm(name: str) -> str:
    return _strip_prefix(name, "Wasm")


def _expand_aliases(names: set[str], normaliser) -> set[str]:
    """Return the set of names *plus* their normalised forms.

    Bindings often ship both `JsFoo` (the napi-rs export) and `Foo` (the type
    alias). Treating both as present lets us match a Rust `Foo` regardless of
    which spelling the binding chose.
    """
    expanded = set(names)
    for n in names:
        expanded.add(normaliser(n))
    return expanded


def _expand_with_camel(names: set[str]) -> set[str]:
    """Augment binding-name set with snake_case<->camelCase variants.

    napi-rs and wasm-bindgen rename Rust free functions like `complete_batch`
    to `completeBatch` on the JS side. The Rust public-surface harvester
    keeps the snake_case original, so to match it we project every binding
    name through both directions:

      - `completeBatch` (in the .d.ts) gets a `complete_batch` alias added,
        so a Rust `complete_batch` matches.
      - `complete_batch` (rare; e.g. if a JS export kept the snake form)
        gets a `completeBatch` alias added for symmetry.

    PascalCase type names are unaffected -- the conversion helpers leave
    them alone so we don't accidentally turn `PipelineState` into
    `pipeline_state`.
    """
    expanded = set(names)
    for n in names:
        camel = _snake_to_camel(n)
        if camel != n:
            expanded.add(camel)
        snake = _camel_to_snake(n)
        if snake != n:
            expanded.add(snake)
    return expanded


# ---------------------------------------------------------------------------
# Diff & report
# ---------------------------------------------------------------------------


@dataclass
class GapReport:
    """Per-binding gap report.

    `by_crate` holds the *real* gaps (structs/enums/free functions that should
    have a binding but don't). `traits_by_crate` is a separate bucket so the
    dev can see at a glance which traits are missing ABC bindings -- those are
    advanced binding choices and we don't want them mixed with the regular
    "you forgot to add this struct" gaps.

    `whitelisted` records every (item, reason) pair that was filtered, so
    --verbose can show what was skipped and why.

    `binding_skipped_total` counts items dropped via a binding-specific skip
    set (currently only WASM_SKIP -- native-only items that can't compile to
    wasm32). These are *not* gaps and are reported separately on the summary
    line so the user understands they were intentionally filtered.
    """

    binding: str
    by_crate: dict[str, list[str]] = field(default_factory=lambda: defaultdict(list))
    traits_by_crate: dict[str, list[str]] = field(
        default_factory=lambda: defaultdict(list)
    )
    total: int = 0
    trait_total: int = 0
    whitelisted: list[tuple[str, str, str]] = field(default_factory=list)
    skipped: bool = False
    skip_reason: str | None = None
    binding_skipped: list[tuple[str, str]] = field(default_factory=list)
    binding_skipped_total: int = 0


def _whitelist_reason(item: str, kind: str) -> str | None:
    """Return a reason string if `item` should be whitelisted, else None.

    The order matters: the explicit `WHITELIST` entries always win, so a
    deliberately-listed type alias still gets its custom message. Type
    aliases not in WHITELIST are auto-skipped because they cannot be bound
    as standalone classes -- they only flow through callback signatures.
    """
    if item in WHITELIST:
        return WHITELIST_REASONS.get(item, "whitelisted (no reason recorded)")
    if kind == KIND_TYPE_ALIAS:
        return "type alias (auto-whitelisted -- not bindable as a standalone type)"
    return None


def diff_surface(
    rust_surfaces: list[CrateSurface],
    binding_names: set[str],
    binding_label: str,
    *,
    binding_skip: frozenset[str] = frozenset(),
) -> GapReport:
    """Diff the Rust surface against a binding's exported names.

    Cross-crate dedup: when a name appears in multiple crates' public surface
    (e.g. `CandleEmbedError` is both defined in `blazen-embed-candle` and
    re-exported through `blazen-llm`), only count it the *first* time we see
    it for a given binding. Otherwise the same gap inflates the total once
    per re-export hop.

    The optional `binding_skip` set lets a caller drop names that are
    physically unbindable in this binding (e.g. native-only crates that
    don't compile to wasm32). Skipped items are recorded on the report's
    `binding_skipped` list so they can be shown in `--verbose` output, but
    they do not count toward `total`/`trait_total` and are not whitelisted.
    """
    report = GapReport(binding=binding_label)
    seen: set[str] = set()
    for surf in rust_surfaces:
        for item in sorted(surf.items):
            if item in seen:
                # Already accounted for via an earlier crate's surface.
                continue
            seen.add(item)

            kind = surf.kind_of(item)

            if item in binding_skip:
                # Drop *before* whitelist so a name that is both skip-listed
                # and whitelisted shows up under the binding-specific bucket
                # (the more specific reason).
                report.binding_skipped.append((surf.crate, item))
                report.binding_skipped_total += 1
                continue

            if (reason := _whitelist_reason(item, kind)) is not None:
                report.whitelisted.append((surf.crate, item, reason))
                continue

            if item in binding_names:
                continue

            if kind == KIND_TRAIT:
                report.traits_by_crate[surf.crate].append(item)
                report.trait_total += 1
                continue

            report.by_crate[surf.crate].append(item)
            report.total += 1
    return report


def print_report(report: GapReport, *, verbose: bool = False) -> None:
    print(f"\n=== {report.binding.upper()} ===")
    if report.skipped:
        print(f"  [skipped] {report.skip_reason}")
        return
    if report.total == 0 and report.trait_total == 0:
        print("  No gaps.")
    else:
        for crate in sorted(report.by_crate):
            items = report.by_crate[crate]
            if not items:
                continue
            print(f"  {crate} ({len(items)}):")
            for item in items:
                print(f"    - {item}")

        if report.trait_total > 0:
            print()
            print("  Traits not bound as ABCs (advanced binding choices):")
            for crate in sorted(report.traits_by_crate):
                items = report.traits_by_crate[crate]
                if not items:
                    continue
                print(f"    {crate} ({len(items)}):")
                for item in items:
                    print(f"      - {item}")

    if verbose:
        print(f"  total real gaps: {report.total}")
        print(f"  total trait gaps: {report.trait_total}")
        if report.binding_skipped_total > 0:
            print(
                "  binding-skipped (native-only, unbindable in this binding) "
                f"({report.binding_skipped_total}):"
            )
            for crate, item in report.binding_skipped:
                print(f"    - [{crate}] {item}")
        if report.whitelisted:
            print(f"  whitelisted ({len(report.whitelisted)}):")
            for crate, item, reason in report.whitelisted:
                print(f"    - [{crate}] {item}: {reason}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help=(
            "Print extra diagnostics: Rust surface sizes plus, for every "
            "filtered item, which whitelist rule excluded it (helps debug "
            "whitelist rot)."
        ),
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=REPO_ROOT,
        help="Override repo root (defaults to the script's parent's parent).",
    )
    args = parser.parse_args(argv)

    crates_dir = args.repo_root / "crates"
    if not crates_dir.is_dir():
        print(f"error: crates directory not found at {crates_dir}", file=sys.stderr)
        return 2

    # Harvest Rust public surfaces.
    rust_surfaces: list[CrateSurface] = []
    missing_crates: list[str] = []
    for crate in WRAPPED_CRATES:
        crate_dir = crates_dir / crate
        if not crate_dir.is_dir():
            missing_crates.append(crate)
            continue
        rust_surfaces.append(collect_rust_surface(crate_dir))

    if missing_crates:
        print(
            "warning: the following wrapped crates were not found and will be "
            f"skipped: {', '.join(missing_crates)}",
            file=sys.stderr,
        )

    if args.verbose:
        print("Rust public surface per crate:")
        for surf in rust_surfaces:
            print(f"  {surf.crate}: {len(surf.items)} items")

    # Harvest binding surfaces.
    py_names = _expand_aliases(collect_py_surface(PY_STUB), normalise_py)
    # Node: napi-rs renames Rust snake_case free functions to camelCase. Add
    # camelCase aliases so a Rust `complete_batch` matches a JS `completeBatch`.
    node_names = _expand_with_camel(
        _expand_aliases(collect_node_surface(NODE_DTS), normalise_node)
    )
    wasm_raw = collect_wasm_surface(WASM_DTS)

    py_report = (
        GapReport(binding="py", skipped=True, skip_reason=f"{PY_STUB} not found")
        if not PY_STUB.is_file()
        else diff_surface(rust_surfaces, py_names, "py")
    )
    node_report = (
        GapReport(binding="node", skipped=True, skip_reason=f"{NODE_DTS} not found")
        if not NODE_DTS.is_file()
        else diff_surface(rust_surfaces, node_names, "node")
    )
    if wasm_raw is None:
        wasm_report = GapReport(
            binding="wasm",
            skipped=True,
            skip_reason=(
                f"{WASM_DTS} not found (wasm-pack output may not be regenerated)"
            ),
        )
    else:
        # WASM: wasm-bindgen also emits camelCase JS names for Rust free
        # functions. Mirror the same camelCase expansion we apply to Node.
        wasm_names = _expand_with_camel(
            _expand_aliases(wasm_raw, normalise_wasm)
        )
        wasm_report = diff_surface(
            rust_surfaces, wasm_names, "wasm", binding_skip=WASM_SKIP
        )

    for report in (py_report, node_report, wasm_report):
        print_report(report, verbose=args.verbose)

    # Aggregate counts for the summary line. Whitelisted/trait totals are
    # surfaced so the dev can see what was filtered without re-running with -v.
    whitelisted_total = (
        len(py_report.whitelisted)
        + len(node_report.whitelisted)
        + len(wasm_report.whitelisted)
    )
    trait_total = (
        py_report.trait_total + node_report.trait_total + wasm_report.trait_total
    )

    print()
    wasm_skipped_note = ""
    if wasm_report.binding_skipped_total > 0:
        wasm_skipped_note = (
            f" ({wasm_report.binding_skipped_total} native-only items skipped)"
        )
    print(
        f"Found {py_report.total} real gaps in py, "
        f"{node_report.total} in node, "
        f"{wasm_report.total} WASM gaps{wasm_skipped_note} "
        f"(plus {whitelisted_total} whitelisted, {trait_total} traits)."
    )

    if py_report.skipped or node_report.skipped or wasm_report.skipped:
        print("(some bindings were skipped -- see per-binding output above)")

    has_gaps = (
        py_report.total > 0
        or node_report.total > 0
        or wasm_report.total > 0
    )
    return 1 if has_gaps else 0


if __name__ == "__main__":
    raise SystemExit(main())
