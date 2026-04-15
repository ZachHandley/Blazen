# Blazen Future Roadmap

This document tracks both **known bugs** that need fixing and **planned features** that expand Blazen beyond API-based providers into **local inference** — running models on the user's own hardware with no network dependency, no per-token costs, and no data leaving the machine.

---

## 0. Critical Bug: Session Ref Lifespan in Sub-Workflows

**Priority: blocker — fix before any new features.**

### Repro (from NBC project)

1. A **scene sub-workflow** runs and puts a `ScenePlan` (a complex non-JSON-serializable Python object) into `StopEvent.result`
2. Blazen stores a `session_ref` ID in the `StopEvent` dict and puts the actual `ScenePlan` in the sub-workflow's process-local `SessionRefRegistry`
3. The scene sub-workflow finishes → Blazen cleans up *its* session registry (the sub-workflow's `Context` goes out of scope, dropping the `SessionRefRegistry`)
4. The **parent orchestrator** tries to read the result → the ref ID is gone → crash

### Root cause

Each `Context` owns its own `SessionRefRegistry` (see `crates/blazen-core/src/session_ref.rs` — "Each Context owns its own SessionRefRegistry so the registry's lifetime is tied to the workflow run"). This is correct for top-level workflows but breaks the parent/child handoff:

- The sub-workflow's registry dies when the sub-workflow finishes
- Refs returned via `StopEvent.result` outlive their registry
- The parent workflow has no way to resolve them

`StopEvent.result` is the one place where a live ref *must* survive past the producing workflow's lifetime, because by definition the consumer is in a different workflow.

### Fix

Two possible approaches — pick one:

**Option A — Parent-owned registry for results (recommended)**

When a sub-workflow is invoked from a parent, pass a reference to the **parent's** `SessionRefRegistry` into the sub-workflow's `Context`. Sub-workflow refs go into the parent's registry directly, so when the sub-workflow finishes, its refs survive in the parent's scope.

- **Pros:** no serialization, no copy, O(1) handoff, live refs work transparently
- **Cons:** requires a "parent registry" slot on `Context`, plus plumbing through sub-workflow invocation

**Option B — Eager materialization at `StopEvent` boundary**

When a sub-workflow emits `StopEvent.result`, walk the payload, resolve every `session_ref` through the dying registry, and inline the resolved objects into the event payload *before* the registry is cleaned up. The parent then gets concrete values, not refs.

- **Pros:** simpler — just a walk + resolve at one known boundary
- **Cons:** defeats the purpose of session refs (loses the "unserializable value" property). Only works if the value *can* be passed by reference/ownership to the parent. For Python, this means holding a `Py<PyAny>` ref across the workflow boundary — which is what the registry already does, so this just moves the storage.

**Recommendation:** Option A is cleaner and matches the mental model. Option B as a fallback if A turns out to need invasive changes.

### Affected files

- `crates/blazen-core/src/session_ref.rs` — registry definition and lifecycle
- `crates/blazen-core/src/context.rs` — `Context` ownership of the registry
- `crates/blazen-core/src/workflow.rs` — sub-workflow invocation and parent/child context wiring
- `crates/blazen-py/src/workflow/session_ref.rs` — Python binding layer for the registry
- `crates/blazen-py/src/workflow/context.rs` — Python Context wrapper
- `crates/blazen-py/src/convert.rs` — session ref detection / resolution during JSON conversion

### Tests to add

- Python e2e: a parent workflow that invokes a sub-workflow, where the sub-workflow returns a non-serializable object (e.g. a `ScenePlan` dataclass holding file handles or a PyTorch tensor) via `StopEvent.result`. The parent must be able to read the result after the sub-workflow finishes. This should go in `tests/python/test_e2e.py`.
- Rust unit test: direct test of parent/child registry handoff in `crates/blazen-core/tests/workflow_integration.rs`.

### Follow-up work (in scope, after the immediate bug fix)

These extend the session ref system beyond the single-process, single-machine case and should land in a phased rollout after the critical fix:

- **Cross-process session refs.** Currently session refs are deliberately excluded from snapshots (see `SessionPausePolicy`). We should add an opt-in path where refs can be snapshotted by serializing the underlying object when it implements a `SessionRefSerializable` marker trait (Python: duck-typed via `__blazen_serialize__`, Node: via a `serialize()` method). Non-serializable refs keep the current pause-on-boundary behavior. This enables resuming a snapshot in a different process — critical for long-running workflows that survive restarts.

- **Distributed workflow execution.** Extend the registry to support refs that live on a remote node. When a parent workflow receives a session ref from a sub-workflow running on a different machine, it should either:
  - (a) Pull the object across the wire on first dereference, caching locally, or
  - (b) Keep a remote proxy handle that forwards method calls back to the owning node
  
  Both modes need a transport layer (initially just HTTP/gRPC against a known peer registry). This is a much bigger undertaking — likely needs its own design doc — but it's the natural end-state of the session ref architecture and should not be blocked by "serialization only" thinking.

- **Ref lifetime policies.** Today the registry auto-cleans on `Context` drop. For distributed / cross-process cases we need explicit lifetime control — `KeepUntilExplicitDrop`, `KeepUntilSnapshot`, `KeepUntilParentFinish` — configurable per-ref at creation time.

- **Ref cloning and sharing.** A sub-workflow should be able to hand a ref to the parent *and* keep using it until the sub-workflow finishes. Current ownership semantics are ambiguous. Add explicit `clone_ref()` / `transfer_ref()` operations and document the semantics.

---

## Guiding Principles

1. **Traits are the contract.** All of our public traits (`CompletionModel`, `EmbeddingModel`, `ImageGeneration`, `AudioGeneration`, `Transcription`, etc.) are already provider-agnostic. Adding local backends means adding new *implementations*, not changing traits.
2. **Feature-gated.** Local backends pull in heavy native dependencies (ONNX Runtime, CUDA, GGML, etc.). Every local backend must be behind a Cargo feature so the default build stays lean.
3. **Separate crates for heavy backends.** Anything that needs its own native build config (e.g. llama.cpp bindings, candle CUDA kernels) belongs in its own crate that depends on `blazen-llm` and implements our traits. The `blazen-llm` crate itself should NOT grow a native-model dependency tree.
4. **Binding coverage is optional per backend.** Not every local backend needs Python + Node + WASM bindings. Some are native-only. Document this per-feature.
5. **Resource-aware defaults.** Local backends must expose device selection (`cpu`, `cuda:0`, `metal`, `vulkan`), quantization choice, and memory budgets. These are real user concerns, not afterthoughts.

---

## 1. Local Embedding Models

### Motivation

Embeddings are the highest-volume call in most RAG pipelines. Sending every document chunk to OpenAI is expensive, slow, and leaks data. Local embedding models are cheap (free), fast (GPU or even fast CPU), and private.

### Tier 1: Local embedding (ONNX) (highest priority)

**Crates:** `blazen-embed` (facade) + `blazen-embed-fastembed` (glibc/mac/windows) + `blazen-embed-tract` (musl/wasm)

- The facade (`blazen-embed`) re-exports `EmbedModel` / `EmbedOptions` and picks the right underlying backend per target triple via target-cfg `Cargo.toml` deps
- On glibc/macOS/Windows: wraps [`fastembed-rs`](https://github.com/Anush008/fastembed-rs) — a Rust port of the Python `fastembed` library — which uses ONNX Runtime under the hood (via `ort` crate)
- On musl/wasm: wraps `tract-onnx` for a pure-Rust ONNX inference path (no C++ runtime dependency)
- Ships pre-quantized models: `BAAI/bge-small-en-v1.5`, `BAAI/bge-base-en-v1.5`, `nomic-embed-text-v1.5`, `jina-embeddings-v2-*`, etc.
- Downloads models on first use, caches in `~/.cache/blazen/models/`
- Implements `EmbeddingModel` trait from `blazen-llm::traits`

**API shape:**
```rust
use blazen_embed::EmbedModel;

let model = EmbedModel::from_options(EmbedOptions {
    model_id: Some("BAAI/bge-small-en-v1.5".into()),
    device: Device::Cpu,
    ..Default::default()
})?;
let response = model.embed(&["hello".into(), "world".into()]).await?;
```

**Python:**
```python
from blazen import EmbeddingModel, EmbedOptions
model = EmbeddingModel.local()  # defaults to bge-small-en-v1.5
model = EmbeddingModel.local(options=EmbedOptions(model_id="nomic-embed-text-v1.5"))
```

**Node:**
```typescript
const model = EmbeddingModel.embed({ modelId: "BAAI/bge-small-en-v1.5" });
```

**Placement:**
- Facade crate: `crates/blazen-embed/`
- Backend crates: `crates/blazen-embed-fastembed/`, `crates/blazen-embed-tract/`
- PyO3 wrapper: `crates/blazen-py/src/types/embedding.rs` — add `embed()` factory behind an `embed` feature flag
- napi-rs wrapper: `crates/blazen-node/src/types/embedding.rs` — same pattern
- WASM: served by the tract backend — bundled automatically on `wasm32-*` via the facade's target-cfg dispatch

### Tier 2: Direct HuggingFace via `candle`

**Crate:** `blazen-embed-candle` (new)

- Wraps [`candle`](https://github.com/huggingface/candle) — HuggingFace's Rust-native ML framework
- Supports any sentence-transformers model from HuggingFace Hub (e.g. `all-MiniLM-L6-v2`, `e5-large-v2`)
- Pure Rust, optional CUDA/Metal backends via Cargo features
- Model download via `hf-hub` crate
- Implements `EmbeddingModel` trait

**Why both this and the embed backend?** The embed backend is faster to get started (pre-packaged models, ONNX). `candle` is more flexible (any HF model, GPU support, no ORT dependency). They serve different use cases.

**Placement:**
- New crate: `crates/blazen-embed-candle/`
- Features: `cpu` (default), `cuda`, `metal`, `accelerate`
- Bindings: Python + Node only (no WASM)

### Tier 3: WASM-compatible embeddings

**Options:**
- `transformers.js` via JS interop from WASM SDK (thin wrapper)
- ONNX Runtime Web (`ort-web`) — experimental, limited model support
- Pure-WASM candle build — theoretically possible, in practice painful

**Status:** Parking lot. The browser ecosystem for local ML is still maturing. Revisit in 2026 when `onnxruntime-web` has better WebGPU support.

---

## 2. Local Completion (Text Generation) Models

### Motivation

Running LLMs locally is the single biggest unlock for privacy-sensitive workloads (healthcare, legal, finance) and cost-sensitive workloads (high-volume background jobs). GGML quantized models (Q4_K_M, Q5_K_M) run 7B-13B parameter models on consumer hardware with acceptable quality.

### Tier 1: `mistralrs` (highest priority)

**Crate:** `blazen-llm-mistralrs` (new)

- Wraps [`mistral.rs`](https://github.com/EricLBuehler/mistral.rs) — a pure-Rust inference engine with GGUF support, CUDA, Metal, paged attention, and a clean async API
- Supports Mistral, LLaMA, Phi, Gemma, Qwen, and more
- Implements `CompletionModel` trait from `blazen-llm::traits`
- Supports streaming via `stream()` method returning `Pin<Box<dyn Stream<Item = ChatStreamChunk>>>`
- Tool calling via the model's native format (where supported)

**API shape:**
```rust
use blazen_llm_mistralrs::MistralRsProvider;

let model = MistralRsProvider::from_options(MistralRsOptions {
    model_id: "TheBloke/Mistral-7B-Instruct-v0.2-GGUF".into(),
    quantization: Quantization::Q4KM,
    device: Device::Cuda(0),
    context_length: 8192,
    ..Default::default()
})?;
```

**Python:**
```python
from blazen import CompletionModel, MistralRsOptions
model = CompletionModel.mistralrs(options=MistralRsOptions(
    model_id="TheBloke/Llama-3.1-8B-Instruct-GGUF",
    quantization="Q5_K_M",
    device="cuda:0",
))
```

**Placement:**
- New crate: `crates/blazen-llm-mistralrs/`
- Features: `cpu` (default), `cuda`, `metal`, `flash-attn`
- Bindings: Python + Node (WASM not supported — too heavy)

### Tier 2: `llama.cpp` bindings via `llama-cpp-rs`

**Crate:** `blazen-llm-llamacpp` (new)

- Wraps [`llama-cpp-rs`](https://github.com/utilityai/llama-cpp-rs) — Rust bindings over `llama.cpp`
- Industry-standard GGUF runtime with the widest model support
- More mature than `mistral.rs` but requires a C++ build step (bundled llama.cpp)
- Implements `CompletionModel` trait

**Why both?** `mistral.rs` is pure Rust (easier build, fewer surprises). `llama.cpp` has broader model coverage and battle-tested performance. Users pick based on their priorities.

**Placement:**
- New crate: `crates/blazen-llm-llamacpp/`
- Features: `cpu`, `cuda`, `metal`, `vulkan`, `rocm`
- Bindings: Python + Node

### Tier 3: `candle` LLM backend

**Crate:** `blazen-llm-candle` (new)

- Uses `candle-transformers` to run HF models directly in Rust
- Supports quantization (GGUF, AWQ, GPTQ)
- Slower than `mistral.rs` for typical chat workloads but more flexible for research use cases
- Implements `CompletionModel` trait

**Placement:** `crates/blazen-llm-candle/` (shares crate with `blazen-embed-candle` if practical — or keep separate, each behind a feature).

---

## 3. Local Image Generation

### Motivation

Stable Diffusion and its descendants (SDXL, SD3, Flux) run on consumer GPUs and produce competitive-quality images. Local image gen is useful for content pipelines, games, and anywhere you can't send prompts to a third-party API.

### Tier 1: `diffusion-rs`

**Crate:** `blazen-image-diffusion` (new)

- Wraps [`diffusion-rs`](https://github.com/EricLBuehler/diffusion-rs) — pure-Rust Stable Diffusion inference built on `mistral.rs` infrastructure
- Supports SD 1.5, SDXL, Flux (with quantization)
- Implements `ImageGeneration` trait from `blazen-llm::compute::traits`
- Implements `ComputeProvider` so it integrates with the existing compute pipeline

**API shape:**
```rust
use blazen_image_diffusion::DiffusionProvider;

let provider = DiffusionProvider::from_options(DiffusionOptions {
    model_id: "stabilityai/stable-diffusion-xl-base-1.0".into(),
    device: Device::Cuda(0),
    ..Default::default()
})?;
let result = provider.generate_image(ImageRequest {
    prompt: "a red circle on white".into(),
    width: Some(1024),
    height: Some(1024),
    ..Default::default()
}).await?;
```

**Placement:**
- New crate: `crates/blazen-image-diffusion/`
- Features: `cpu`, `cuda`, `metal`
- Bindings: Python + Node

### Tier 2: `candle` image backends

- `candle-transformers` has SD pipelines built in
- Alternative path if `diffusion-rs` doesn't keep up with new models
- Same crate-location strategy

### Supporting compute methods

For a full local image-gen story, we also need:

- **Upscaling** (`ImageGeneration::upscale_image`) — ESRGAN, Real-ESRGAN via `candle`
- **Background removal** (`BackgroundRemoval::remove_background`) — BiRefNet, InSPyReNet via ONNX Runtime
- **Image-to-image** — already covered by diffusion pipelines

These can live in the same `blazen-image-diffusion` crate or split out as needed.

---

## 4. Local Audio Models

### Motivation

Transcription (speech-to-text) and TTS (text-to-speech) are extremely well-served by local models now. Whisper runs faster than real-time on a modern laptop. Piper TTS is tiny and fast. No reason to hit an API for these when privacy matters.

### 4.1 Transcription: `whisper.cpp`

**Crate:** `blazen-audio-whispercpp` (new)

- Wraps [`whisper-rs`](https://github.com/tazz4843/whisper-rs) — Rust bindings over `whisper.cpp`
- Supports all Whisper model sizes (tiny, base, small, medium, large-v3)
- Quantized GGUF models for small footprints
- Implements `Transcription` trait from `blazen-llm::compute::traits`
- Supports diarization via `pyannote` integration (optional)

**API:**
```rust
let provider = WhisperCppProvider::from_options(WhisperOptions {
    model: WhisperModel::LargeV3,
    device: Device::Cpu,
    ..Default::default()
})?;
let result = provider.transcribe(TranscriptionRequest {
    audio_url: "file:///path/to/audio.wav".into(),
    language: Some("en".into()),
    ..Default::default()
}).await?;
```

**Placement:**
- New crate: `crates/blazen-audio-whispercpp/`
- Features: `cpu`, `cuda`, `metal`, `coreml`
- Bindings: Python + Node

**Note on `audio_url`:** The current `TranscriptionRequest.audio_url` is designed for HTTP URLs. For local models we need to support `file://` URIs or introduce a `audio_bytes: Vec<u8>` variant. This is a minor trait addition.

### 4.2 TTS: Piper

**Crate:** `blazen-audio-piper` (new)

- Wraps [Piper](https://github.com/rhasspy/piper) via its C API or ONNX Runtime
- Runs on CPU, small models (~30MB), very fast
- Implements `AudioGeneration::text_to_speech` trait

### 4.3 Music / SFX generation

- **MusicGen** (Meta) via `candle-transformers`
- **AudioCraft** via ONNX Runtime
- Both implement `AudioGeneration::generate_music` / `generate_sfx`

**Placement:** `crates/blazen-audio-candle/` or similar, feature-gated.

---

## 5. Local Vision / Multimodal

### Motivation

Vision-language models (LLaVA, MiniCPM-V, Qwen2-VL) run locally now. Useful for image captioning, visual Q&A, and multimodal agents without sending images to cloud APIs.

### Approach

These are just `CompletionModel` implementations with image support in `ChatMessage::user_image_url`. The existing trait already handles this. We add them as another backend in:

- `blazen-llm-mistralrs` (mistral.rs supports LLaVA, Phi-3.5-vision)
- `blazen-llm-candle` (candle has LLaVA and Qwen2-VL implementations)

No new crate needed — just expand the vision model coverage in the existing local-LLM crates.

---

## 6. Local 3D Generation

### Motivation

Triposr, TripoSR, CRM, and friends can generate 3D models from a single image on a consumer GPU. Niche but powerful for games, product design, and 3D content pipelines.

### Status

- Only a handful of open-source 3D models are GPU-ready for consumer hardware
- Triposr has an ONNX export; integrate via ONNX Runtime
- Crate: `blazen-3d-triposr` or a more general `blazen-3d-local`

**Priority:** Lower — the API ecosystem (fal.ai, Replicate) covers this well today. Revisit when local models catch up in quality.

---

## 7. Supporting Infrastructure

### 7.1 Model download and caching

**New crate:** `blazen-model-cache` (optional, shared)

Every local-model crate needs:
- HuggingFace Hub downloads (via `hf-hub` crate)
- Local cache directory (`~/.cache/blazen/models/` or `$BLAZEN_CACHE_DIR`)
- SHA256 verification
- Progress reporting via a `ProgressCallback` trait
- Parallel downloads
- Resume-on-failure

If multiple crates need this, extract it into `blazen-model-cache` rather than duplicating.

### 7.2 Device abstraction

**New module:** `blazen_llm::device` (in the existing `blazen-llm` crate)

```rust
pub enum Device {
    Cpu,
    Cuda(usize),
    Metal,
    Vulkan(usize),
    Rocm(usize),
}

impl Device {
    pub fn from_env() -> Self { /* auto-detect */ }
    pub fn from_str(s: &str) -> Result<Self, BlazenError> { /* parse "cuda:0" */ }
}
```

All local backends take a `Device` parameter. Enables consistent user experience across backends.

### 7.3 Quantization format enum

```rust
pub enum Quantization {
    F32,
    F16,
    BF16,
    Q8_0,
    Q6_K,
    Q5_K_M,
    Q5_K_S,
    Q4_K_M,
    Q4_K_S,
    Q3_K_M,
    Q2_K,
    // GPTQ / AWQ variants
    Gptq4Bit,
    Awq4Bit,
}
```

Lives in `blazen_llm::quantization`.

### 7.4 Typed options for every local backend

Each local backend gets its own options struct following the same pattern as `ProviderOptions`:

```rust
pub struct EmbedOptions {
    pub model_id: Option<String>,
    pub device: Option<Device>,
    pub cache_dir: Option<PathBuf>,
    pub max_batch_size: Option<usize>,
}
```

All implement `Serialize + Deserialize` and have PyO3 / napi-rs wrappers.

---

## 8. Crate Layout (Proposed)

```
crates/
├── blazen-llm/                    # Core traits — NO native deps
├── blazen-embed/                  # Facade — target-cfg dispatch to the backend below
├── blazen-embed-fastembed/        # Local embeddings via fastembed-rs (glibc/mac/windows)
├── blazen-embed-tract/            # Local embeddings via tract-onnx (musl/wasm)
├── blazen-embed-candle/           # Local embeddings via candle
├── blazen-llm-mistralrs/          # Local LLM via mistral.rs
├── blazen-llm-llamacpp/           # Local LLM via llama.cpp
├── blazen-llm-candle/             # Local LLM via candle
├── blazen-image-diffusion/        # Local image gen via diffusion-rs
├── blazen-audio-whispercpp/       # Local STT via whisper.cpp
├── blazen-audio-piper/            # Local TTS via Piper
├── blazen-audio-candle/           # Local music/SFX via candle
├── blazen-model-cache/            # Shared HF download + cache
└── ...existing crates...
```

**Umbrella crate `blazen/`:** Re-exports all local backends behind feature flags:

```toml
[features]
default = []
local-all = ["local-embeddings", "local-llm", "local-image", "local-audio"]
local-embeddings = ["embed", "candle-embed"]
local-llm = ["mistralrs", "llamacpp", "candle-llm"]
local-image = ["diffusion"]
local-audio = ["whispercpp", "piper"]
embed = ["dep:blazen-embed"]
candle-embed = ["dep:blazen-embed-candle"]
mistralrs = ["dep:blazen-llm-mistralrs"]
llamacpp = ["dep:blazen-llm-llamacpp"]
candle-llm = ["dep:blazen-llm-candle"]
diffusion = ["dep:blazen-image-diffusion"]
whispercpp = ["dep:blazen-audio-whispercpp"]
piper = ["dep:blazen-audio-piper"]
```

Users opt into exactly what they need. Default build stays lean.

---

## 9. Binding Strategy

### Python

- Each local backend gets a feature flag in `crates/blazen-py/Cargo.toml`
- Example: `pyo3-stub-gen` + `#[cfg(feature = "embed")]` for the factory method
- Wheels built per-feature-set: `blazen`, `blazen[embed]`, `blazen[local-all]`, etc.
- GPU wheels: `blazen[cuda]`, `blazen[metal]` — pull in the matching native deps

### Node

- Same feature-flag approach in `crates/blazen-node/Cargo.toml`
- napi-rs supports conditional exports
- Pre-built binaries per platform × feature combo (this is the hard part — build matrix explodes)
- Alternative: ship a smaller base package + optional platform-specific addons

### WASM

- **Not supported for most local backends.** See Tier 3 embeddings for why.
- Exception: future WebGPU-based embeddings (transformers.js wrap) may be viable for the WASM SDK.

---

## 10. Priority Order

1. **`embed` embeddings (fastembed on glibc/mac/windows, tract on musl/wasm)** — easiest win, highest value, no GPU required
2. **`mistral.rs` LLM** — the single biggest user-facing feature
3. **`whisper.cpp` transcription** — near-universal demand, mature, fast
4. **`diffusion-rs` image gen** — rounds out the media story
5. **`candle` embeddings** — flexibility alternative to the embed backend
6. **`candle` LLM** — research / experimentation use cases
7. **`llama.cpp` LLM** — alternative for maximum model coverage
8. **Piper TTS** — small addition, low-priority compared to above
9. **Music / SFX / vision LLMs** — nice-to-haves
10. **3D generation** — wait for the ecosystem to mature

---

## 11. Out-of-Scope (Explicit Non-Goals)

- **Training.** Blazen is an inference framework. Fine-tuning and pretraining belong elsewhere.
- **Model conversion tooling.** Users bring their own GGUF / ONNX / safetensors files or download from HF Hub. We don't build a conversion CLI.
- **Custom CUDA kernels.** We rely on upstream backends (candle, mistral.rs, etc.) for GPU work. No in-house kernel authoring.
- **Browser WebGPU inference.** Cool but not our fight — the WASM SDK stays thin.

---

## 12. Migration Path for Existing Users

Once the first local backends land:

- **No breaking changes.** Existing API providers (OpenAI, Anthropic, etc.) keep working unchanged.
- **Memory subsystem** (`blazen-memory`): Already accepts any `Arc<dyn EmbeddingModel>` — local embeddings drop in with zero code changes.
- **Workflows:** Steps that call `model.complete()` are provider-agnostic — swap a local model in, nothing else changes.
- **Fallback chains:** `CompletionModel.with_fallback([local_model, cloud_model])` becomes a powerful pattern — try local first, fall back to cloud on overload.

---

## 13. Open Questions

- **Model licensing.** Several popular open-weight models have restrictive licenses (Llama 2/3 community license, Gemma prohibited-uses policy). Do we surface these in code or just in docs? Lean toward docs — we're not a licensing enforcement tool.
- **Memory budgeting.** Should `from_options` take a `max_memory_mb` field and fail fast if the model won't fit? Probably yes.
- **Streaming token events.** Local backends can expose richer streaming info (per-token logprobs, draft model decisions). Do we extend the `ChatStreamChunk` type? Defer until there's user demand.
- **Batching.** Local embedding/LLM models benefit hugely from batched inference. Does the `embed()` / `complete()` trait need an explicit batch variant, or is the existing `&[String]` input enough? Likely enough for embeddings, not enough for completions — revisit when we get there.
