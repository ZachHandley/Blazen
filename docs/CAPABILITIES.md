# Capability Namespaces

`WorkerCapability::kind: String` is intentionally a free-form string in
`blazen_core::distributed`. This document is the **authoritative
namespace registry** — every consumer (Blazen, ZBrain, ZAITUI Hub,
zAssistant, HomeLabs/Arrchitect, zSceneAI, etc.) chooses kinds from the
trees below so the scheduler matches jobs to workers without typed enum
churn.

`WorkerCapability::version: u32` is a **contract-version**, NOT a semver.
Bump it only when the input/output **schema** for the capability changes
in a breaking way. New backend, same I/O = same version. Different
I/O shape = new version.

## Prefix discipline

`agent:<product>[:<role>]` — single product, optional role.

- `agent:claude-code` (no role — one-shot codegen agent)
- `agent:codex` (no role)
- `agent:gemini-cli` (no role)
- `agent:arrchitect:encode` (role required — arrchitect has multiple)
- `agent:arrchitect:subtitle`
- `agent:arrchitect:metadata`
- `agent:architect` (separate from `arrchitect` — different vocabulary)
- `agent:quality-improver` (separate from arrchitect's improver role)

## Namespace tree

### `inference:llm:*` — language-model inference

| Kind | Version | Backend / notes |
|---|---|---|
| `inference:llm:anthropic` | 1 | Remote Anthropic API |
| `inference:llm:openai` | 1 | Remote OpenAI API |
| `inference:llm:local-llama-cpp` | 1 | `blazen-llm-llamacpp` GGUF |
| `inference:llm:local-mistral-rs` | 1 | `blazen-llm-mistralrs` |
| `inference:llm:local-candle` | 1 | `blazen-llm-candle` |
| `inference:llm:ollama` | 1 | `blazen-llm-ollama` |
| `inference:llm:vllm` | 1 | `blazen-llm-vllm` |
| `inference:llm:tgi` | 1 | `blazen-llm-tgi` |
| `inference:llm:lmstudio` | 1 | `blazen-llm-lmstudio` |

### `inference:embed:*` — embeddings

| Kind | Version | Backend |
|---|---|---|
| `inference:embed:bge-large` | 1 | `blazen-embed-fastembed` |
| `inference:embed:nomic` | 1 | `blazen-embed-fastembed` |
| `inference:embed:tract` | 1 | `blazen-embed-tract` |
| `inference:embed:candle` | 1 | `blazen-embed-candle` |

### `inference:stt:*` — speech-to-text

| Kind | Version | Backend |
|---|---|---|
| `inference:stt:whisper` | 1 | `blazen-audio-stt`, version 1 = `large-v3`/`turbo` whisper.cpp contract |

### `inference:tts:*` — text-to-speech

| Kind | Version | Backend |
|---|---|---|
| `inference:tts:kokoro` | 1 | `blazen-audio-tts` Kokoro |
| `inference:tts:piper` | 1 | `blazen-audio-tts` Piper |
| `inference:tts:f5` | 1 | `blazen-audio-tts` F5-TTS |
| `inference:tts:vibevoice` | 1 | `blazen-audio-tts` VibeVoice |
| `inference:tts:qwen3` | 1 | `blazen-audio-tts` Qwen3-TTS |

### `inference:vision:*` — vision models (zSceneAI consumer)

| Kind | Version | Backend |
|---|---|---|
| `inference:vision:detect` | 1 | SAM3 (segment-anything 3) |
| `inference:vision:segment` | 1 | SAM3 |
| `inference:vision:depth` | 1 | Depth-Anything |
| `inference:vision:pose` | 1 | GVHMR |
| `inference:vision:vlm` | 1 | Tarsier2 |
| `inference:vision:scene-graph` | 1 | VGGT |
| `inference:vision:autoshot` | 1 | autoshot scene-detection |

### `inference:3d:*` — 3D generation / manipulation (zSceneAI)

| Kind | Version | Backend |
|---|---|---|
| `inference:3d:image-to-mesh` | 1 | TripoSR |
| `inference:3d:image-to-mesh` | 2 | TRELLIS |
| `inference:3d:image-to-mesh` | 3 | Hunyuan3D-2 |
| `inference:3d:image-to-mesh` | 4 | SF3D |
| `inference:3d:image-to-mesh` | 5 | CRM |
| `inference:3d:image-to-mesh` | 6 | Unique3D |
| `inference:3d:image-to-mesh` | 7 | LGM |
| `inference:3d:rigging` | 1 | UniRig |
| `inference:3d:rigging` | 2 | Blender Rigify (subprocess) |
| `inference:3d:texture` | 1 | SyncMVD |
| `inference:3d:texture` | 2 | TEXTure |
| `inference:3d:texture` | 3 | Text2Tex |
| `inference:3d:texture` | 4 | Flux-texturing |
| `inference:3d:reconstruction` | 1 | COLMAP |
| `inference:3d:reconstruction` | 2 | MASt3R |
| `inference:3d:refine` | 1 | mesh post-processing pipeline |

### `inference:video:*` — video diffusion (zSceneAI)

| Kind | Version | Backend |
|---|---|---|
| `inference:video:diffusion` | 1 | CogVideoX |
| `inference:video:diffusion` | 2 | HunyuanVideo |
| `inference:video:diffusion` | 3 | Hunyuan 1.5 |
| `inference:video:diffusion` | 4 | LTX 2.3 |
| `inference:video:diffusion` | 5 | Wan 2.2 |
| `inference:video:diffusion` | 6 | Cosmos |
| `inference:video:diffusion` | 7 | ConsisID |
| `inference:video:diffusion` | 8 | Hallo4 |
| `inference:video:diffusion` | 9 | Phantom Wan |
| `inference:video:diffusion` | 10 | VACE |

### `inference:motion:*` — motion generation (zSceneAI)

| Kind | Version | Backend |
|---|---|---|
| `inference:motion:generate` | 1 | MotionGPT |
| `inference:motion:generate` | 2 | MoMask |
| `inference:motion:generate` | 3 | AniMo |

### `media:ffmpeg:*` — media transcoding

| Kind | Version | Backend |
|---|---|---|
| `media:ffmpeg:transcode` | 1 | `ffmpeg` shell-out, `-hwaccel` hint per job |
| `media:ffmpeg:thumbnail` | 1 | `ffmpeg` thumbnail extract |
| `media:ffmpeg:remux` | 1 | `ffmpeg` container repackage with `-c copy` (no re-encode) |

#### `media:ffmpeg:remux` v1 schema

**Inputs:**

```json
{
  "input_url": "https://example.com/source.mkv",
  "output_path": "/path/to/dest.mp4",
  "container": "mkv|mp4|mov|webm",
  "copy_streams": true,
  "extra_args": ["-map", "0"]
}
```

- `input_url` — source media; may be `https://...` or `file:///path/to/source`.
- `output_path` — destination on the worker filesystem.
- `container` — target container format (`mkv`, `mp4`, `mov`, `webm`).
- `copy_streams` — default `true`; when `true` uses `-c copy` (stream copy, no re-encode).
- `extra_args` — optional raw `ffmpeg` arg passthrough for power users.

**Outputs:**

```json
{
  "output_url": "file:///path/to/dest.mp4",
  "size_bytes": 123456789,
  "duration_seconds": 1342.5,
  "format": "mp4",
  "stream_summary": [
    { "codec_type": "video", "codec_name": "h264" },
    { "codec_type": "audio", "codec_name": "aac" }
  ]
}
```

**Notes:** Remux = container repackage with `-c copy` (no re-encode). Use `media:ffmpeg:transcode` for re-encoding.

### `media:image:*` — image analysis / generation

| Kind | Version | Backend |
|---|---|---|
| `media:image:analyze` | 1 | Blazen vision provider OR `imagemagick identify` fallback |
| `media:image:generate` | 1 | Flux (`blazen-image-diffusion`) |
| `media:image:generate` | 2 | SDXL |
| `media:image:generate` | 3 | SD3 |

### `media:blender:*` — Blender pool daemon (zSceneAI `crates/blender-controller/`)

| Kind | Version | Backend |
|---|---|---|
| `media:blender:render` | 1 | JSON-RPC 2.0 over WebSocket to long-lived Blender pool |
| `media:blender:rigify` | 1 | Rigify pipeline inside the same pool |

### `agent:*` — agentic workers

| Kind | Version | Notes |
|---|---|---|
| `agent:claude-code` | 1 | Anthropic Claude Code one-shot codegen |
| `agent:codex` | 1 | OpenAI Codex |
| `agent:gemini-cli` | 1 | Google Gemini CLI |
| `agent:arrchitect:encode` | 1 | Arrchitect encode worker (HomeLabs lineage) |
| `agent:arrchitect:subtitle` | 1 | Arrchitect subtitle worker |
| `agent:arrchitect:metadata` | 1 | Arrchitect metadata worker |
| `agent:architect` | 1 | General-purpose architecture agent — NOT arrchitect |
| `agent:quality-improver` | 1 | General-purpose quality-improvement agent |

## Companion concepts

### Node labels (worker-side, declared in `WorkerHello::labels`)

Free-form `BTreeMap<String, String>` of `key:value` schedulable attributes.
Conventional namespaces, not exhaustive:

- **Hardware**: `gpu:nvidia`, `gpu:amd`, `gpu:apple-metal`, `cpu-only`,
  `arch:x86_64`, `arch:aarch64`, `vram:>=8gb`, `vram:>=12gb`,
  `vram:>=16gb`, `vram:>=24gb`, `vram:>=48gb`, `vram:>=80gb`
- **Host**: `host:beastpc`, `host:minibeast`, `host:zminipc`,
  `host:zach-linux`
- **Cluster / network**: `cluster:homelabs`, `cluster:zstack-dev`,
  `mesh:netbird-a`, `mesh:netbird-b`
- **Tenancy**: `tenant:zach`, `tenant:blackleafdigital`
- **Lifecycle**: `lifecycle:prod`, `lifecycle:dev`, `lifecycle:scratch`

`NodeSelector` on `Assignment` filters candidate workers by these labels:
`required` (all must match), `forbidden` (none may match), `preferred`
(soft tie-breaker score).

### Taints + tolerations

`WorkerTaint { key, value, effect: NoSchedule|PreferNoSchedule }` on the
worker keeps un-tolerated jobs off. `TolerationSpec { key, value, effect }`
on the assignment matches a taint when `key + effect` match and `value`
is `None` (wildcard) or equal.

### Priority

`Assignment::priority: u8` — lower numeric value = higher priority.
Default is `DEFAULT_PRIORITY = 128`. The queue uses **deficit-round-robin
WFQ** across 8 priority bands of width 32 so low-priority jobs never
starve: band 0 (0..32) weight 256, band 1 (32..64) weight 224, …, band 7
(224..256) weight 32.

## Upstream dependencies tracked elsewhere

- `inference:llm:local-*` model files live at
  `~/.zbrain/models/<provider>/<model-id>/` (shared across workers on
  the same host so two workers loading the same model share the file
  cache).
- the durable store still lacks `SELECT FOR UPDATE SKIP LOCKED` and per-row TTL — see
  the durable store issue tracker. Until both land, the `DurableAssignmentStore` uses
  OCC retry with K=8-shard sharding (`crates/blazen-controlplane/src/server/durable_store.rs`)
  and a control-plane TTL sweeper (60s).
- Auth for human-submitted jobs uses ZAuth JWT (mirror
  `ZAITUI/crates/zaitui-server/src/auth.rs`); worker-to-CP uses mTLS
  with ZBrain-issued worker certs at `~/.zbrain/workers/<name>.tls/`.

## When to add a new kind vs. bump a version

- **New kind**: a fundamentally different capability (`inference:vision:depth`
  is not `inference:vision:detect`).
- **New version of an existing kind**: same task surface, **different
  input/output schema**. The string `inference:3d:image-to-mesh` is one
  capability; versions 1..7 above are seven backends each with a
  potentially different input schema (some take only an image, some
  take image + text prompt + reference-mesh, etc.).
- **Same kind + same version, different model file**: not a capability
  change — that's just a per-worker config (`~/.zbrain/workers/<name>.toml`
  picks the model).
