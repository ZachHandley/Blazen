# Fixing Type Generation Across All Bindings

## Problem Statement

Every user-facing type in Blazen is currently defined **four times**: once in the Rust core (`blazen-llm`, `blazen-core`, `blazen-events`) and then re-declared as a manually-mirrored wrapper in each of the three binding crates (`blazen-py`, `blazen-node`, `blazen-wasm-sdk`). This quadruplication has led to:

- **Field drift**: Node widens `f32` → `f64` and `u64` → `i64` due to napi-rs limitations. Python omits `parameters` on every compute request. WASM's `CompletionResponse` is missing 7 of 14 fields. `TokenUsage` only exposes 3 of 7 fields in *all* bindings.
- **Silent coverage gaps**: the WASM SDK wraps only 4 types total; Node has a full set of 40; Python has 32.
- **Maintenance burden**: adding a field to a core type (e.g. `reasoning_tokens` on `TokenUsage`) requires hunting down 3+ mirror sites and hoping the CI catches the drift (it doesn't — there are no cross-binding schema tests).
- **Dead code**: Node declares `JsImageResult`, `JsVideoResult`, etc. as `#[napi(object)]` structs but every compute method actually returns `serde_json::Value`, so the typed TS interfaces exist only on paper.

The fix: **define each type exactly once in the Rust core**, derive binding-specific traits behind feature gates, and have each binding crate consume the core types directly.

---

## Current Duplication Inventory

### Compute request types (7 core types)

| Type | Core | Python | Node | WASM |
|---|---|---|---|---|
| `ImageRequest` | `blazen-llm/src/compute/requests.rs:11` | `blazen-py/src/compute/requests.rs:17` (`PyImageRequest`, wraps core, 6 getters, **`parameters` not exposed**) | `blazen-node/src/compute/requests.rs:7` (`JsImageRequest`, `#[napi(object)]`, 7 re-declared fields, `parameters: Option<Value>`) | **missing** |
| `UpscaleRequest` | `requests.rs:79` | `requests.rs:90` (`PyUpscaleRequest`, wraps core, **no `parameters`**) | `requests.rs:28` (`JsUpscaleRequest`, 4 fields, **`scale` is `f64` not `f32`**) | **missing** |
| `VideoRequest` | `requests.rs:116` | `requests.rs:138` (`PyVideoRequest`, wraps core, **no `parameters`**) | `requests.rs:42` (`JsVideoRequest`, 8 fields, **`duration_seconds` is `f64`**) | **missing** |
| `SpeechRequest` | `requests.rs:195` | `requests.rs:219` (`PySpeechRequest`, wraps core, **no `parameters`**) | `requests.rs:66` (`JsSpeechRequest`, 7 fields, **`speed` is `f64`**) | **missing** |
| `MusicRequest` | `requests.rs:269` | `requests.rs:292` (`PyMusicRequest`, wraps core, **no `parameters`**) | `requests.rs:86` (`JsMusicRequest`, 4 fields, **`duration_seconds` is `f64`**) | **missing** |
| `TranscriptionRequest` | `requests.rs:313` | `requests.rs:340` (`PyTranscriptionRequest`, wraps core, **no `parameters`**) | `requests.rs:100` (`JsTranscriptionRequest`, 5 fields, **`diarize` is `Option<bool>` not `bool`**) | **missing** |
| `ThreeDRequest` | `requests.rs:367` | `requests.rs:402` (`PyThreeDRequest`, wraps core, **no `parameters`**) | `requests.rs:116` (`JsThreeDRequest`, 5 fields, **`prompt` is `Option<String>` not `String`**) | **missing** |

**Key drift**: Python systematically omits `parameters` on all 7 types. Node systematically widens `f32` → `f64`. Node changes optionality on `diarize` and `prompt`.

### Compute result types (7 core types)

| Type | Core | Python | Node | WASM |
|---|---|---|---|---|
| `ImageResult` | `results.rs:14` | returned as raw JSON | `results.rs:11` (`JsImageResult`, `#[napi(object)]`, **dead code — never returned**) | **missing** |
| `VideoResult` | `results.rs:31` | raw JSON | `results.rs:24` (`JsVideoResult`, dead code) | **missing** |
| `AudioResult` | `results.rs:48` | raw JSON | `results.rs:37` (`JsAudioResult`, dead code) | **missing** |
| `TranscriptionSegment` | `results.rs:82` | raw JSON | `results.rs:50` (`JsTranscriptionSegment`) | **missing** |
| `TranscriptionResult` | `results.rs:95` | raw JSON | `results.rs:63` (`JsTranscriptionResult`, dead code) | **missing** |
| `ThreeDResult` | `results.rs:65` | raw JSON | `results.rs:80` (`JsThreeDResult`, dead code) | **missing** |
| `ComputeResult` | `job.rs:59` | raw JSON | `job.rs:51` (`JsComputeResult`, dead code) | **missing** |

**Key issue**: Node declares these as `#[napi(object)]` TS interfaces but every `FalProvider` method returns `serde_json::Value` (`Promise<any>`). The typed interfaces are dead code. Python doesn't declare them at all and returns plain dicts. This means **no binding provides typed compute results to users today**.

### Job lifecycle types (3 core types)

| Type | Core | Python | Node | WASM |
|---|---|---|---|---|
| `JobHandle` | `job.rs:17` | `job.rs:45` (`PyJobHandle`, wraps core, `submitted_at` → ISO 8601 String) | `job.rs:18` (`JsJobHandle`, `#[napi(object)]`, `submitted_at: String`) | **missing** |
| `JobStatus` | `job.rs:30` (enum with `Failed { error }`) | `job.rs:18` (`PyJobStatus`, frozen class with string classattrs, **loses `error` data**) | `job.rs:32` (`JsJobStatus`, `#[napi(string_enum)]`, **loses `error` data**) | **missing** |
| `ComputeTiming` | `usage.rs:38` (`RequestTiming`, `Option<u64>`) | `usage.rs:74` (`PyRequestTiming`, wraps core) | **DUPLICATED TWICE**: `job.rs:67` (`JsComputeTiming`, `Option<i64>`) AND `usage.rs:18` (`JsRequestTiming`, `Option<i64>`) — structurally identical | **missing** |

### Media output types (5 core types)

| Type | Core | Python | Node | WASM |
|---|---|---|---|---|
| `MediaOutput` | `media.rs:450` (6 fields, `media_type: MediaType` enum) | `media.rs:93` (`PyMediaOutput`, wraps core, `media_type` → MIME string) | `media.rs:11` (`JsMediaOutput`, `media_type: String`, **`file_size: Option<f64>` not `u64`**) | **missing** |
| `GeneratedImage` | `media.rs:499` | `media.rs:183` (`PyGeneratedImage`, wraps core) | `media.rs:31` (`JsGeneratedImage`, re-declared) | **missing** |
| `GeneratedVideo` | `media.rs:510` | `media.rs:241` (`PyGeneratedVideo`, wraps core) | `media.rs:41` (`JsGeneratedVideo`, **`f32` → `f64`**) | **missing** |
| `GeneratedAudio` | `media.rs:525` | `media.rs:313` (`PyGeneratedAudio`, wraps core) | `media.rs:58` (`JsGeneratedAudio`, **`channels: Option<u32>` not `u8`**, **`f32` → `f64`**) | **missing** |
| `Generated3DModel` | `media.rs:538` | `media.rs:384` (`PyGenerated3DModel`, wraps core) | `media.rs:73` (`JsGenerated3DModel`, **`u64` → `f64`**) | **missing** |

### LLM types (10 core types)

| Type | Core | Python | Node | WASM |
|---|---|---|---|---|
| `ChatMessage` | `message.rs:411` (5 fields) | `message.rs:146` (wraps core) | `message.rs:100` (wraps core) | `chat_message.rs:25` (wraps core) |
| `CompletionRequest` | `completion.rs:180` (10 fields, **NO serde**) | constructed inline | `completion.rs:36` (`JsCompletionOptions`, only 6 fields, **renamed**) | via JsValue |
| `CompletionResponse` | `completion.rs:291` (14 fields, **NO serde**) | `completion.rs:22` (wraps core, 14 getters) | `completion.rs:15` (14 re-declared fields) | `types.rs:22` (wraps core, **only 7 getters — missing 7 fields**) |
| `StreamChunk` | `completion.rs:386` (6 fields, **NO serde**) | `stream.rs:28` (wraps core, 6 getters) | `stream.rs:27` (6 re-declared fields) | `types.rs:90` (wraps core, **only 3 getters — missing 3 fields**) |
| `TokenUsage` | `usage.rs:11` (7 fields) | `usage.rs:14` (wraps core, **only 3 getters**) | `usage.rs:7` (**only 3 re-declared fields**) | `types.rs:128` (wraps core, **only 3 getters**) |
| `RequestTiming` | `usage.rs:38` (3 fields, `Option<u64>`) | `usage.rs:74` (wraps core) | `usage.rs:18` (3 fields, **`Option<i64>`**) | **missing** |
| `ToolCall` | `tool.rs:22` (3 fields) | `tool.rs:14` (wraps core) | `tool.rs:7` (3 re-declared) | `types.rs:159` (wraps core) |
| `ToolDefinition` | `tool.rs:11` (3 fields) | via JSON dict | `tool.rs:15` (3 re-declared) | **missing** |
| `FinishReason` | `completion.rs:89` (10-variant enum) | `finish_reason.rs:22` (wraps core, `kind`/`value` getters) | `finish_reason.rs:16` (flattened to `{ kind, value }` strings) | **missing** |
| `ResponseFormat` | `completion.rs:136` (3-variant enum) | `response_format.rs:18` (wraps core + factories) | `response_format.rs:13` (flattened to 4 fields) | **missing** |

### LLM extra types (3 core types, newer additions)

| Type | Core | Python | Node | WASM |
|---|---|---|---|---|
| `ReasoningTrace` | `completion.rs:17` | `reasoning.rs:18` (wraps core) | `reasoning.rs:12` (4 re-declared fields) | **missing** |
| `Citation` | `completion.rs:29` (7 fields, `start: Option<usize>`) | `citation.rs:17` (wraps core) | `citation.rs:12` (7 fields, **`start: Option<i64>`**) | **missing** |
| `Artifact` | `completion.rs:52` (8-variant enum) | `artifact.rs:26` (wraps core + factories) | `artifact.rs:25` (flattened to 7 fields) | **missing** |

### Provider config types (binding-specific)

| Type | Core | Python | Node | WASM |
|---|---|---|---|---|
| `FalLlmEndpoint` | `fal.rs:142` (6 variants, `OpenRouter { enterprise }`, `Custom { path }`) | `fal.rs:38` (7 flat variants, **`enterprise` split, `Custom` dropped**) | `fal.rs:37` (7 string variants, same split) | **missing** |
| `FalOptions` | **NONE** (builder methods) | `fal.rs:75` (`PyFalOptions`, binding-specific) | `fal.rs:92` (`JsFalOptions`, binding-specific) | **missing** |

---

## Overall Statistics

| Category | Core types | Python mirrors | Node mirrors | WASM mirrors |
|---|---|---|---|---|
| Compute requests | 7 | 7 | 7 | 0 |
| Compute results | 7 | 0 (raw JSON) | 7 (dead code) | 0 |
| Job lifecycle | 3 | 3 | 3 (+ 1 intra-dup) | 0 |
| Media output | 5 | 5 | 5 | 0 |
| LLM types | 10 | 8 | 10 | 4 (incomplete) |
| LLM extra | 3 | 3 | 3 | 0 |
| Provider config | 2 | 2 | 2 | 0 |
| **Totals** | **37** | **28** | **37** | **4** |

**37 types × 3 bindings = 111 potential wrappers. Today: 69 exist, ~42 missing. Of the 69 that exist, every one has at least one field-level drift from the source type.**

---

## Target Architecture

### Core Principle

**Every user-facing type is defined exactly once in `blazen-llm` (or `blazen-core` / `blazen-events`).** Each binding crate opts into automatic bridging by enabling a feature gate. The binding's FFI boundary consumes the core type directly — no mirror structs, no re-declared fields, no getter boilerplate.

### Mechanism Per Binding

| Binding | Mechanism | Feature gate | What it generates |
|---|---|---|---|
| **WASM** | [`tsify`](https://crates.io/crates/tsify) derive | `blazen-llm/tsify` | `wasm_bindgen` `IntoWasmAbi` / `FromWasmAbi` impls + TypeScript `.d.ts` interfaces. JS users construct plain objects that match the TS types and get full autocomplete. |
| **Python** | [`pythonize`](https://crates.io/crates/pythonize) (serde ↔ Python dict) | `blazen-llm/pyo3` | `FromPyObject` / `IntoPyObject` impls via `depythonize` / `pythonize`. Python users pass plain dicts; the Rust FFI boundary deserializes them into the core struct. For types that need class-instance ergonomics (e.g. `ChatMessage` with factory methods), keep a thin `#[pyclass]` wrapper that holds `inner: CoreType` but do NOT re-declare fields. |
| **Node** | [`serde-napi`](https://crates.io/crates/serde-napi) or `#[napi(object)]` derive via a proc-macro in `blazen-macros` | `blazen-llm/napi` | For `#[napi(object)]` types (plain TS interfaces), a proc-macro in `blazen-macros` generates the mirror struct automatically from the core type's field list. For full JS class types (`ChatMessage`), keep a thin `#[napi(js_name = "...")]` wrapper with `inner: CoreType`. |

### Feature gate layout in `blazen-llm/Cargo.toml`

```toml
[features]
default = []

# WASM binding support — adds tsify derives for automatic TypeScript interface generation.
tsify = ["dep:tsify", "dep:wasm-bindgen"]

# Python binding support — adds pythonize-based serde ↔ Python dict conversion.
pyo3-serde = ["dep:pythonize", "dep:pyo3"]

# Node binding support — adds napi-derive re-exports for the blazen-macros proc-macro.
napi-bindings = ["dep:napi", "dep:napi-derive"]

[dependencies]
tsify = { version = "0.4", features = ["js"], optional = true }
wasm-bindgen = { version = "0.2", optional = true }
pythonize = { version = "0.22", optional = true }
pyo3 = { version = "0.28", optional = true }
napi = { version = "3", optional = true }
napi-derive = { version = "3", optional = true }
```

### Derive pattern on core types

```rust
// crates/blazen-llm/src/compute/requests.rs

/// Request to generate one or more images from a text prompt.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "tsify", derive(tsify::Tsify))]
#[cfg_attr(feature = "tsify", tsify(into_wasm_abi, from_wasm_abi))]
#[serde(rename_all = "camelCase")]
pub struct ImageRequest {
    pub prompt: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub negative_prompt: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub width: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub height: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub num_images: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    #[serde(default)]
    pub parameters: serde_json::Value,
}
```

**Important**: `#[serde(rename_all = "camelCase")]` is what makes the JS-side field names match JS convention (`negativePrompt`, `numImages`) while keeping Rust field names in `snake_case`. This replaces the manual `#[napi(js_name = "negativePrompt")]` annotations.

### Binding crate consumption

#### WASM (`crates/blazen-wasm-sdk`)

```toml
# Cargo.toml
blazen-llm = { path = "../blazen-llm", features = ["tsify"], default-features = false }
```

```rust
// src/fal.rs
use blazen_llm::compute::{ImageRequest, ImageResult};

#[wasm_bindgen(js_class = "FalProvider")]
impl WasmFalProvider {
    /// Generate one or more images from a text prompt.
    #[wasm_bindgen(js_name = "generateImage")]
    pub fn generate_image(&self, req: ImageRequest) -> js_sys::Promise {
        let provider = Arc::clone(&self.inner);
        future_to_promise(async move {
            let result = provider.generate_image(req).await
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
            // ImageResult also derives Tsify, so it auto-converts to JS
            Ok(serde_wasm_bindgen::to_value(&result)?)
        })
    }
}
```

**Zero mirror structs.** `ImageRequest` is the core type with `Tsify` derived. `ImageResult` is the core type with `Tsify` derived. The generated `.d.ts` has full TypeScript interfaces for both.

#### Python (`crates/blazen-py`)

```toml
# Cargo.toml
blazen-llm = { workspace = true, features = ["pyo3-serde"] }
pythonize = "0.22"
```

```rust
// src/providers/fal.rs
use blazen_llm::compute::{ImageRequest, ImageResult};

#[pymethods]
impl PyFalProvider {
    fn generate_image(&self, py: Python<'_>, req: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        // depythonize: Python dict → core ImageRequest
        let request: ImageRequest = pythonize::depythonize(req)?;
        let inner = self.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let result = inner.generate_image(request).await
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
            // pythonize: core ImageResult → Python dict
            Python::attach(|py| pythonize::pythonize(py, &result))
        })
    }
}
```

**Zero mirror structs for data types.** Users pass a plain Python dict like `{"prompt": "a cat", "width": 1024}` and get a typed dict back. The `.pyi` stubs (generated or hand-maintained) provide autocomplete. For types that need class-instance ergonomics (e.g. `ChatMessage` with factory methods like `ChatMessage.user("hello")`), keep the existing thin `#[pyclass]` wrapper that holds `inner: CoreType`.

#### Node (`crates/blazen-node`)

napi-rs does NOT have a serde-based auto-derive equivalent to `tsify` or `pythonize`. The `#[napi(object)]` macro requires a literal struct definition in the binding crate. Two approaches:

**Approach A — proc-macro in `blazen-macros`** (recommended):

Add a `#[blazen_macros::napi_mirror]` attribute macro that reads the field list from a core type and emits a `#[napi(object)]` struct:

```rust
// crates/blazen-node/src/compute/requests.rs

blazen_macros::napi_mirror! {
    /// Request to generate one or more images from a text prompt.
    blazen_llm::compute::ImageRequest as JsImageRequest
}
```

The macro expands to:

```rust
#[napi(object)]
pub struct JsImageRequest {
    pub prompt: String,
    #[napi(js_name = "negativePrompt")]
    pub negative_prompt: Option<String>,
    // ... all fields from ImageRequest, with:
    //   - f32 → f64 (napi limitation)
    //   - u64 → f64 (napi limitation)
    //   - automatic camelCase js_name
}

impl From<JsImageRequest> for blazen_llm::compute::ImageRequest { /* field-by-field */ }
impl From<blazen_llm::compute::ImageRequest> for JsImageRequest { /* field-by-field */ }
```

This is the most architecturally clean option: the type is defined once in `blazen-llm`, the mirror is auto-generated, and field additions to the core type are automatically picked up at build time (the proc-macro reads the core type's definition via `quote!`).

**Approach B — serde round-trip** (simpler, less type-safe):

Accept `serde_json::Value` at the napi boundary, convert to the core type via `serde_json::from_value`, and return results via `serde_json::to_value`. Users see `any` in TypeScript but the Rust side is fully typed. Ship a hand-maintained `index.d.ts` augmentation file for the typed interfaces.

This is what the Node binding already does for compute results (which return `serde_json::Value`). The downside is TypeScript users get `any` instead of typed interfaces.

**Recommendation**: Start with Approach B for compute types (which are already `serde_json::Value`-based today) and migrate to Approach A incrementally. The proc-macro is the eventual target but can be built after the WASM and Python ports prove the serde-based approach works.

---

## Blockers: Types That Don't Derive Serde

Two critical types **do NOT** derive `Serialize` / `Deserialize`:

1. **`CompletionRequest`** (`blazen-llm/src/types/completion.rs:180`) — 10 fields. Cannot be auto-bridged via serde until derives are added. The `tools: Vec<ToolDefinition>` field and `response_format: Option<ResponseFormat>` field are both `Serialize + Deserialize`, so the blocker is just the missing derive on the parent struct.

2. **`CompletionResponse`** (`blazen-llm/src/types/completion.rs:291`) — 14 fields. Same issue. The `images: Vec<GeneratedImage>` and `audio: Vec<GeneratedAudio>` fields ARE serde-able; the parent just lacks the derive.

3. **`StreamChunk`** (`blazen-llm/src/types/completion.rs:386`) — 6 fields. Same issue.

**Fix**: Add `#[derive(Serialize, Deserialize)]` to all three. These are the two most important types in the entire LLM API surface — without serde on them, the tsify/pythonize/serde-napi approach can't cover the core completion path.

**Risk**: Adding `Serialize` to `CompletionRequest` means any consumer can serialize it to JSON. This is intentional — it enables logging, caching, and cross-process transfer of completion requests, which are all things users want.

---

## Systematic Field Drift to Fix

When the types are centralized, these drifts get fixed for free:

| Drift | Where | Fix |
|---|---|---|
| `f32` → `f64` widening | Node: `duration_seconds`, `speed`, `scale`, `fps` | Core stays `f32`; the `napi_mirror!` macro auto-widens. WASM and Python consume `f32` natively. |
| `u64` → `i64` narrowing | Node: timing fields, `file_size`, `vertex_count`, `face_count` | Core stays `u64`; the `napi_mirror!` macro auto-narrows. WASM and Python consume `u64` natively. |
| `parameters` omitted | Python: all 7 compute request types | Core exposes `parameters: serde_json::Value`; `pythonize` includes it. Python users can pass `{"parameters": {"extra": 1}}`. |
| `TokenUsage` missing 4 fields | All 3 bindings | Core has 7 fields; all bindings auto-expose them. |
| `CompletionResponse` missing 7 fields | WASM | Tsify derives on the full 14-field type → all fields appear in TS. |
| `StreamChunk` missing 3 fields | WASM | Same fix. |
| Node `JsComputeTiming` duplicated | Node has both `JsComputeTiming` and `JsRequestTiming` (identical structs) | Both become one core `RequestTiming` with one napi mirror. |
| `diarize: bool` vs `Option<bool>` | Core vs Node `TranscriptionRequest` | Core `diarize: bool` is authoritative. `#[serde(default)]` handles the JS case where the field is omitted. |
| `prompt: String` vs `Option<String>` | Core vs Node `ThreeDRequest` | Core `prompt: String` is authoritative. If JS needs to omit it, use `#[serde(default)]` with an empty string. |
| `JobStatus::Failed { error }` data loss | Python (classattr strings) and Node (string_enum) | Use `#[serde(tag = "status", content = "error")]` on the core enum; tsify generates a discriminated union in TS; pythonize preserves the data. |
| `MediaType` enum → MIME string flattening | Python and Node | `#[serde(serialize_with = "...")]` on the core `MediaType` field to serialize as MIME string. Or keep the enum in Rust and let tsify generate the union type. |

---

## Phased Execution

### Phase 1 — Add serde derives to `CompletionRequest`, `CompletionResponse`, `StreamChunk`

**Scope**: `crates/blazen-llm/src/types/completion.rs`

Add `#[derive(Serialize, Deserialize)]` to all three types. This unblocks the serde-based automation for the most important types. Verify existing tests still pass.

### Phase 2 — Add `tsify` feature gate to `blazen-llm`, derive `Tsify` on all serde-able types

**Scope**: `crates/blazen-llm/Cargo.toml`, every file with a user-facing `pub struct` / `pub enum` that already derives Serde.

Count from the inventory: **~37 types** across `compute/requests.rs`, `compute/results.rs`, `compute/job.rs`, `media.rs`, `types/completion.rs`, `types/usage.rs`, `types/tool.rs`, `types/message.rs`.

Add:
```rust
#[cfg_attr(feature = "tsify", derive(tsify::Tsify))]
#[cfg_attr(feature = "tsify", tsify(into_wasm_abi, from_wasm_abi))]
```

And add `#[serde(rename_all = "camelCase")]` where the JS convention differs from Rust.

### Phase 3 — Rewrite `blazen-wasm-sdk` to consume core types directly

**Scope**: `crates/blazen-wasm-sdk/`

- Enable `blazen-llm` feature `tsify`.
- Delete all manual wrapper types in `types.rs` and `chat_message.rs`.
- Rewrite `CompletionModel`, `FalProvider`, `Workflow`, `Context` methods to take/return core types.
- Add the missing providers (`anthropic`, `gemini`, `fal`) and compute API surface that were never implemented.
- Result: the WASM SDK becomes a thin set of `#[wasm_bindgen]` methods that delegate to `blazen-llm`, with zero type mirrors.

### Phase 4 — Add `pythonize` feature gate, refactor `blazen-py` to use `depythonize`

**Scope**: `crates/blazen-llm/Cargo.toml`, `crates/blazen-py/`

- Enable `blazen-llm` feature `pyo3-serde`.
- For compute types: delete `Py*Request` wrappers, accept `&Bound<'_, PyAny>` and `depythonize` to core type.
- For result types: `pythonize` core type → Python dict (replacing the existing `serde_json::to_value` → `json_to_py` round-trip).
- For `ChatMessage`: keep the `PyChatMessage` wrapper (it has factory methods like `ChatMessage.user(...)` that pure serde can't express) but have it hold `inner: CoreType` and derive getters from the core fields.
- Update `.pyi` stubs to use TypedDict for the dict-based types.

### Phase 5 — Add `napi_mirror!` proc-macro to `blazen-macros`, refactor `blazen-node`

**Scope**: `crates/blazen-macros/src/`, `crates/blazen-node/`

- Implement `napi_mirror!` proc-macro that generates `#[napi(object)]` structs from a core type reference.
- Replace all manual `Js*` mirror structs with `napi_mirror!` invocations.
- The macro handles `f32` → `f64` widening, `u64` → `f64` narrowing, and camelCase `js_name` annotation automatically.
- For `ChatMessage`: keep the `JsChatMessage` wrapper (has factory methods) but derive field access from the core type.
- Fix the `JsComputeTiming` / `JsRequestTiming` intra-binding duplication (collapse to one mirror of `RequestTiming`).

### Phase 6 — Wire compute results through typed returns (all bindings)

**Scope**: All three binding crates' `FalProvider` methods.

- Replace `-> serde_json::Value` returns with `-> ImageResult`, `-> VideoResult`, etc.
- WASM: already typed via tsify.
- Python: `pythonize` the core result type.
- Node: use the `napi_mirror!`-generated mirror struct.
- **This is what makes the "dead code" result types live.** Users finally get typed compute results instead of `any`.

### Phase 7 — Port all test suites to `tests/wasm/`, add CI `test-wasm` job

Same test suite across all three bindings. Each `tests/{python,node,wasm}/test_*.{py,mjs}` file covers the same surface with the same assertions. 7 test files × 3 bindings = 21 files, all exercising the same types, same providers, same API keys.

---

## Types That Remain Binding-Specific (Not Refactored)

These types genuinely differ across bindings and should NOT be centralized:

| Type | Why it stays binding-specific |
|---|---|
| `PyEvent` / `PyStartEvent` / `PyStopEvent` | Python event class hierarchy with `__init_subclass__` metaclass magic and session-ref auto-routing via `__getattr__`. No analog in Node/WASM. |
| `PyBlazenState` | Python dataclass-style typed state with `Meta.persistent` / `Meta.transient`. Binding-specific persistence protocol. |
| `PyContext` / `JsContext` / `WasmContext` | Each wraps `blazen_core::Context` but adds binding-specific dispatch (4-tier storage in Python, JSON/bytes in Node, `RefCell<HashMap>` in WASM). |
| `PyStepWrapper` / `StepHandlerTsfn` | Language-specific callable wrapping (Python `Py<PyAny>`, Node `ThreadsafeFunction`, WASM `js_sys::Function`). |
| `PyWorkflow` / `JsWorkflow` / `WasmWorkflow` | Binding-specific builder API and step registration. |
| `PyWorkflowHandler` / `JsWorkflowHandler` | Binding-specific async result / stream / pause API. |
| `FalOptions` | Binding-specific options object that maps to builder methods on `FalProvider`. Could be centralized as a serde struct in the core, but the builder pattern is idiomatic Rust. |

---

## References

- [`tsify` crate](https://crates.io/crates/tsify) — `Tsify` derive for automatic `wasm_bindgen` ABI + TypeScript declaration generation from serde types.
- [`pythonize` crate](https://crates.io/crates/pythonize) — `pythonize` / `depythonize` for serde ↔ Python object round-trip.
- [`napi-derive` crate](https://crates.io/crates/napi-derive) — `#[napi(object)]` attribute for napi-rs TypeScript interface generation.
- [`serde` rename_all](https://serde.rs/container-attrs.html#rename_all) — `#[serde(rename_all = "camelCase")]` for JS-convention field names.
- `blazen-macros` crate at `crates/blazen-macros/` — existing proc-macro crate (`#[derive(Event)]`) that would host the new `napi_mirror!` macro.
- Agent survey reports from this session:
  - Node FalProvider survey — full API surface, auth pattern (`Authorization: Key`), factory signatures, test inventory.
  - Node compute API survey — all request/result types, job lifecycle, polling implementation, HTTP threading model.
  - Type duplication inventory — every mirrored type with file paths, line numbers, and field drift.
