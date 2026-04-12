# FIXING_TYPEGENS — Replace untyped Python `serialize_to_py` dicts with proper pyclass wrappers

## Problem

Python result types (`ImageResult`, `AudioResult`, `VideoResult`, `ThreeDResult`, `ComputeResult`, `CompletionResponse`) return their nested collections and structs as **raw Python dicts** via `serialize_to_py()` / `pythonize()` / `json_to_py()`. This means `result.images` returns `list[dict]` instead of `list[GeneratedImage]`, `result.timing` returns `dict` instead of `RequestTiming`, etc. Every such getter shows as `-> typing.Any` in `blazen.pyi`, giving users zero type safety or IDE autocompletion on the nested objects.

**The Node/TypeScript side does NOT have this problem** -- napi-rs's auto-mirror system in `crates/blazen-node/build.rs` generates properly typed interfaces for every nested struct (`JsGeneratedImage`, `JsMediaOutput`, `JsRequestTiming`, etc.) with full field-level types. Python should match.

## Scope

~32 getter methods across 4 files return `Py<PyAny>` via serialization instead of typed wrappers. The fix is: create `#[pyclass]` wrappers for the inner types, then update each getter to return the typed wrapper.

## What already exists (don't rebuild)

| What | Path | Notes |
|---|---|---|
| `PyTranscriptionSegment` (already typed) | `crates/blazen-py/src/compute/result_types.rs:37-76` | Wraps `TranscriptionSegment`. Has text/start/end/speaker getters. **Reference pattern** -- copy this for new wrappers. |
| `PyVoiceHandle` (already typed) | `crates/blazen-py/src/compute/result_types.rs:308-387` | Wraps `VoiceHandle`. Has id/name/provider/language/description getters. |
| `TranscriptionResult.segments` (already typed) | `crates/blazen-py/src/compute/result_types.rs:100-106` | Returns `Vec<PyTranscriptionSegment>` -- shows the correct pattern for list-of-wrapper getters. |
| Node auto-generated types (reference for field shapes) | `crates/blazen-node/index.d.ts` | See `JsGeneratedImage`, `JsGeneratedAudio`, `JsGeneratedVideo`, `JsGenerated3DModel`, `JsMediaOutput`, `JsRequestTiming` for the exact field list per type. |

## Rust inner types that need Python wrappers

All live in `crates/blazen-llm/src/media.rs` unless noted otherwise.

### `MediaOutput` (media.rs:454-471)

```rust
pub struct MediaOutput {
    pub url: Option<String>,
    pub base64: Option<String>,
    pub raw_content: Option<String>,           // text-based formats only (SVG, OBJ, GLTF)
    pub media_type: MediaType,                 // enum -- see media.rs:9-100
    pub file_size: Option<u64>,
    pub metadata: serde_json::Value,           // freeform -- stays dict[str, Any]
}
```

Python wrapper: `PyMediaOutput` (frozen, no constructor).

Getters: `url: Optional[str]`, `base64: Optional[str]`, `raw_content: Optional[str]`, `media_type: str` (expose as the MIME string via `self.inner.media_type.to_mime()`), `file_size: Optional[int]`, `metadata: dict` (stays `json_to_py`).

### `GeneratedImage` (media.rs:509-518)

```rust
pub struct GeneratedImage {
    pub media: MediaOutput,
    pub width: Option<u32>,
    pub height: Option<u32>,
}
```

Python wrapper: `PyGeneratedImage` (frozen, no constructor).

Getters: `media: MediaOutput` (returns `PyMediaOutput`), `width: Optional[int]`, `height: Optional[int]`.

### `GeneratedVideo` (media.rs:524-539)

```rust
pub struct GeneratedVideo {
    pub media: MediaOutput,
    pub width: Option<u32>,
    pub height: Option<u32>,
    pub duration_seconds: Option<f32>,
    pub fps: Option<f32>,
}
```

Python wrapper: `PyGeneratedVideo` (frozen, no constructor).

Getters: `media: MediaOutput`, `width: Optional[int]`, `height: Optional[int]`, `duration_seconds: Optional[float]`, `fps: Optional[float]`.

### `GeneratedAudio` (media.rs:545-557)

```rust
pub struct GeneratedAudio {
    pub media: MediaOutput,
    pub duration_seconds: Option<f32>,
    pub sample_rate: Option<u32>,
    pub channels: Option<u8>,
}
```

Python wrapper: `PyGeneratedAudio` (frozen, no constructor).

Getters: `media: MediaOutput`, `duration_seconds: Optional[float]`, `sample_rate: Optional[int]`, `channels: Optional[int]`.

### `Generated3DModel` (media.rs:563-576)

```rust
pub struct Generated3DModel {
    pub media: MediaOutput,
    pub vertex_count: Option<u64>,
    pub face_count: Option<u64>,
    pub has_textures: bool,
    pub has_animations: bool,
}
```

Python wrapper: `PyGenerated3DModel` (frozen, no constructor).

Getters: `media: MediaOutput`, `vertex_count: Optional[int]`, `face_count: Optional[int]`, `has_textures: bool`, `has_animations: bool`.

### `RequestTiming` (types/usage.rs:42-52)

```rust
pub struct RequestTiming {
    pub queue_ms: Option<u64>,
    pub execution_ms: Option<u64>,
    pub total_ms: Option<u64>,
}
```

Python wrapper: `PyRequestTiming` (frozen, no constructor).

Getters: `queue_ms: Optional[int]`, `execution_ms: Optional[int]`, `total_ms: Optional[int]`.

---

## Files to create

### `crates/blazen-py/src/compute/media_types.rs` (NEW)

All 6 new `#[pyclass]` wrappers go here. Follow the exact pattern of `PyTranscriptionSegment` at `result_types.rs:37-76`:

```rust
use pyo3::prelude::*;
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pymethods};

use blazen_llm::media::{GeneratedAudio, GeneratedImage, GeneratedVideo, Generated3DModel, MediaOutput};
use blazen_llm::types::RequestTiming;

// ---------------------------------------------------------------------------
// MediaOutput
// ---------------------------------------------------------------------------

#[gen_stub_pyclass]
#[pyclass(name = "MediaOutput", frozen)]
pub struct PyMediaOutput {
    pub(crate) inner: MediaOutput,
}

#[gen_stub_pymethods]
#[pymethods]
impl PyMediaOutput {
    /// URL where the media can be downloaded.
    #[getter]
    fn url(&self) -> Option<&str> {
        self.inner.url.as_deref()
    }

    /// Base64-encoded media data.
    #[getter]
    fn base64(&self) -> Option<&str> {
        self.inner.base64.as_deref()
    }

    /// Raw text content for text-based formats (SVG, OBJ, GLTF JSON).
    #[getter]
    fn raw_content(&self) -> Option<&str> {
        self.inner.raw_content.as_deref()
    }

    /// MIME type string (e.g. "image/png", "audio/mpeg", "video/mp4").
    #[getter]
    fn media_type(&self) -> String {
        self.inner.media_type.to_mime().to_owned()
    }

    /// File size in bytes, if known.
    #[getter]
    fn file_size(&self) -> Option<u64> {
        self.inner.file_size
    }

    /// Arbitrary provider-specific metadata (as a dict).
    #[getter]
    fn metadata(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        crate::convert::json_to_py(py, &self.inner.metadata)
    }

    fn __repr__(&self) -> String {
        format!("MediaOutput(url={:?}, media_type={:?})", self.inner.url, self.inner.media_type.to_mime())
    }
}

// ... then PyGeneratedImage, PyGeneratedVideo, PyGeneratedAudio, PyGenerated3DModel, PyRequestTiming
// following the same pattern. Each wraps the Rust struct, provides #[getter] per field,
// and returns PyMediaOutput for the `media` field:
//
// #[getter]
// fn media(&self) -> PyMediaOutput {
//     PyMediaOutput { inner: self.inner.media.clone() }
// }
```

### Register in `crates/blazen-py/src/compute/mod.rs`

Add `pub mod media_types;` and re-export the new types:
```rust
pub use media_types::{
    PyGeneratedAudio, PyGeneratedImage, PyGeneratedVideo, PyGenerated3DModel,
    PyMediaOutput, PyRequestTiming,
};
```

### Register in `crates/blazen-py/src/lib.rs`

Add `m.add_class::<...>()?;` for all 6 new types, in the compute-types registration block (around line 170).

---

## Files to modify (update existing getters)

### `crates/blazen-py/src/compute/result_types.rs`

Every getter that currently calls `serialize_to_py(py, &self.inner.field)` or `json_to_py(py, &self.inner.field)` for a typed struct needs to return the new wrapper instead. Specific changes:

**`PyImageResult`** (lines 148-172):
- `fn images(...)` -> `fn images(&self) -> Vec<PyGeneratedImage>` -- map inner vec to wrapper vec
- `fn timing(...)` -> `fn timing(&self) -> PyRequestTiming` -- wrap inner struct
- `fn metadata(...)` -- LEAVE as `json_to_py` -- metadata is freeform `serde_json::Value`, `dict[str, Any]` is correct

Pattern for list-of-wrapper:
```rust
#[getter]
fn images(&self) -> Vec<PyGeneratedImage> {
    self.inner.images.iter().map(|i| PyGeneratedImage { inner: i.clone() }).collect()
}
```

Pattern for struct wrapper:
```rust
#[getter]
fn timing(&self) -> PyRequestTiming {
    PyRequestTiming { inner: self.inner.timing.clone() }
}
```

**`PyVideoResult`** (lines 185-211):
- `fn videos(...)` -> `fn videos(&self) -> Vec<PyGeneratedVideo>`
- `fn timing(...)` -> `fn timing(&self) -> PyRequestTiming`
- `fn metadata(...)` -- leave as dict

**`PyAudioResult`** (lines 224-250):
- `fn audio(...)` -> `fn audio(&self) -> Vec<PyGeneratedAudio>`
- `fn timing(...)` -> `fn timing(&self) -> PyRequestTiming`
- `fn metadata(...)` -- leave as dict

**`PyThreeDResult`** (lines 263-289):
- `fn models(...)` -> `fn models(&self) -> Vec<PyGenerated3DModel>`
- `fn timing(...)` -> `fn timing(&self) -> PyRequestTiming`
- `fn metadata(...)` -- leave as dict

**`PyTranscriptionResult`** (lines 89-133):
- `fn timing(...)` -> `fn timing(&self) -> PyRequestTiming`
- `fn metadata(...)` -- leave as dict
- `fn segments(...)` -- already returns `Vec<PyTranscriptionSegment>`, no change needed

**`PyComputeResult`** (lines 400-435):
- `fn timing(...)` -> `fn timing(&self) -> PyRequestTiming`
- `fn job(...)` -- leave as dict (JobHandle doesn't have a pyclass wrapper yet; follow-up)
- `fn output(...)` -- leave as dict (arbitrary JSON)
- `fn metadata(...)` -- leave as dict

### `crates/blazen-py/src/types/completion.rs`

**`PyCompletionResponse`** (around lines 29-115):
- `fn content(...)` -- already returns `Option<&str>`, no change
- `fn tool_calls(...)` -> leave as `typing.Any` for now -- `Vec<ToolCall>` needs its own pyclass wrapper (follow-up)
- `fn timing(...)` -> `fn timing(&self) -> Option<PyRequestTiming>` -- currently returns `Option<Py<PyAny>>`, wrap in `PyRequestTiming`
- `fn images(...)` -> leave as `typing.Any` for now -- these are `CompletionResponse` image attachments, different from `ImageResult.images` (follow-up -- may need a different wrapper)
- `fn audio(...)` -> leave as `typing.Any` (same reason)
- `fn videos(...)` -> leave as `typing.Any` (same reason)
- `fn metadata_extra(...)` -- leave as dict (freeform JSON)
- `fn reasoning(...)` -> leave as `typing.Any` for now -- `ReasoningTrace` needs its own pyclass (follow-up)
- `fn citations(...)` -> leave as `typing.Any` -- `Vec<Citation>` needs its own pyclass (follow-up)
- `fn artifacts(...)` -> leave as `typing.Any` -- `Vec<Artifact>` needs its own pyclass (follow-up)

### `crates/blazen-py/src/types/memory.rs`

- `PyMemoryResult.metadata(...)` -- leave as dict (freeform)
- `PyMemory.get(...)` -- leave as `typing.Any` (returns arbitrary stored value)

### `crates/blazen-py/src/workflow/event.rs`

- `PyEvent.to_dict(...)` -- leave as dict (that's the point of `to_dict`)

---

## What stays as `typing.Any` (intentional)

These are CORRECTLY `typing.Any` and should NOT be changed:

| Getter | Reason |
|---|---|
| `*.metadata(...)` on every result type | Freeform `serde_json::Value` -- genuinely arbitrary |
| `ComputeResult.output(...)` | Model-specific JSON -- no fixed schema |
| `ComputeResult.job(...)` | Needs a `PyJobHandle` wrapper -- separate follow-up |
| `CompletionResponse.tool_calls(...)` | Needs a `PyToolCall` wrapper -- separate follow-up |
| `CompletionResponse.reasoning(...)` | Needs a `PyReasoningTrace` wrapper -- separate follow-up |
| `CompletionResponse.citations(...)` | Needs a `PyCitation` wrapper -- separate follow-up |
| `CompletionResponse.artifacts(...)` | Needs a `PyArtifact` wrapper -- separate follow-up |
| `CompletionResponse.images/audio/videos(...)` | Completion-level media refs, not `GeneratedImage` -- follow-up |
| `Memory.get(...)` | Returns arbitrary stored values |
| `Event.to_dict(...)` | Returns the dict representation of an event |
| `EmbeddingModel.embed(...)` | Returns pythonized `EmbeddingResponse` -- needs wrapper |

---

## After all changes

1. Remove the `serialize_to_py` helper function from `result_types.rs:26-30` if nothing uses it anymore.
2. Remove unused `#[gen_stub(override_return_type(...))]` attributes if any remain on the updated getters (the `timing` and `metadata` getters on `TranscriptionResult` at lines 116 and 129 have these from a recent refactor -- remove the ones on getters that now return typed wrappers; leave the ones on getters that still return `json_to_py`).
3. Regenerate stubs: `cargo run --example stub_gen -p blazen-py --features mistralrs`.
4. Verify the `.pyi` shows typed returns:
   ```python
   class ImageResult:
       @property
       def images(self) -> list[GeneratedImage]: ...
       @property
       def timing(self) -> RequestTiming: ...
       @property
       def cost(self) -> float | None: ...
       @property
       def metadata(self) -> typing.Any: ...  # intentional

   class GeneratedImage:
       @property
       def media(self) -> MediaOutput: ...
       @property
       def width(self) -> int | None: ...
       @property
       def height(self) -> int | None: ...

   class MediaOutput:
       @property
       def url(self) -> str | None: ...
       @property
       def base64(self) -> str | None: ...
       @property
       def media_type(self) -> str: ...
       # ...
   ```
5. Run: `cargo check -p blazen-py --all-features && cargo fmt --all -- --check && cargo clippy --workspace --all-features -- -D warnings`.

---

## Verification checklist

- [ ] `PyMediaOutput` -- 6 getters, all typed
- [ ] `PyGeneratedImage` -- 3 getters (media, width, height)
- [ ] `PyGeneratedAudio` -- 4 getters (media, duration_seconds, sample_rate, channels)
- [ ] `PyGeneratedVideo` -- 5 getters (media, width, height, duration_seconds, fps)
- [ ] `PyGenerated3DModel` -- 5 getters (media, vertex_count, face_count, has_textures, has_animations)
- [ ] `PyRequestTiming` -- 3 getters (queue_ms, execution_ms, total_ms)
- [ ] `ImageResult.images` returns `list[GeneratedImage]`
- [ ] `ImageResult.timing` returns `RequestTiming`
- [ ] `AudioResult.audio` returns `list[GeneratedAudio]`
- [ ] `AudioResult.timing` returns `RequestTiming`
- [ ] `VideoResult.videos` returns `list[GeneratedVideo]`
- [ ] `VideoResult.timing` returns `RequestTiming`
- [ ] `ThreeDResult.models` returns `list[Generated3DModel]`
- [ ] `ThreeDResult.timing` returns `RequestTiming`
- [ ] `TranscriptionResult.timing` returns `RequestTiming`
- [ ] `ComputeResult.timing` returns `RequestTiming`
- [ ] All 6 new classes registered in `lib.rs`
- [ ] `blazen.pyi` regenerated with new types
- [ ] `cargo clippy --workspace --all-features -- -D warnings` clean
- [ ] `grep -c "typing.Any:" crates/blazen-py/blazen.pyi` drops from ~45 to ~25

---

## Follow-up tasks (NOT in this ticket)

These typed wrappers are needed for full coverage but are separate work:

- `PyToolCall` wrapping `blazen_llm::types::ToolCall` -- for `CompletionResponse.tool_calls`
- `PyCitation` wrapping `blazen_llm::types::Citation` -- for `CompletionResponse.citations`
- `PyArtifact` wrapping `blazen_llm::types::Artifact` -- for `CompletionResponse.artifacts`
- `PyReasoningTrace` wrapping `blazen_llm::types::ReasoningTrace` -- for `CompletionResponse.reasoning`
- `PyJobHandle` wrapping `blazen_llm::compute::job::JobHandle` -- for `ComputeResult.job`
- `PyEmbeddingResponse` wrapping the embedding result -- for `EmbeddingModel.embed`
