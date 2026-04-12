# LOCAL_TTS — TODO

Adds TTS providers to Blazen so it can orchestrate speech synthesis. Two providers in scope:

1. **`openai_tts`** — generic OpenAI-compatible `/v1/audio/speech` client. Points at `zvoice` (local VoxCPM2 wrapper living in `HomeLabs/zvoice/`, exposed on `http://beastpc.lan:8900/v1` or `http://minibeast.lan:8900/v1`) and at OpenAI itself. Primary consumer.
2. **`elevenlabs`** — ElevenLabs HTTP API for managed TTS / voice cloning fallback.

Both plug into the existing `AudioGeneration` capability trait — most of the scaffolding already exists, so this is mainly implementing two providers and wiring them through the Python/Node bindings.

---

## What already exists (don't rebuild)

| Thing | Path |
|---|---|
| `AudioGeneration` trait (`text_to_speech`, `generate_music`, `generate_sfx`) | `crates/blazen-llm/src/compute/traits.rs:117` |
| `SpeechRequest` struct (`text`, `voice`, `voice_url`, `language`, `speed`, `model`, freeform `parameters`) | `crates/blazen-llm/src/compute/requests.rs:222` |
| `AudioResult` return type | `crates/blazen-llm/src/compute/results.rs` (same module) |
| `HttpClient` trait + `ReqwestHttpClient` / `FetchHttpClient` / `WasiHttpClient` impls | `crates/blazen-llm/src/http.rs:192` |
| Existing `AudioGeneration` impl (reference) — fal.ai (async job queue, not sync POST) | `crates/blazen-llm/src/providers/fal.rs:2941` |
| Existing OpenAI provider (completions only, **no TTS yet**) | `crates/blazen-llm/src/providers/openai.rs` |
| Piper stub (`todo!` body, Phase 9) — leave alone | `crates/blazen-audio-piper/src/provider.rs` |
| WASM router stub returning 501 for `POST /v1/audio/speech` | `crates/blazen-wasm/src/router.rs:259` |
| Python binding pattern for `AudioGeneration::text_to_speech` | `crates/blazen-py/src/providers/fal.rs:326` |
| Provider re-exports | `crates/blazen-llm/src/lib.rs:106` |
| Provider design docs | `docs/providers.md` |

---

## Task 1 — `openai_tts` provider

### Target API

Hits any OpenAI-compatible `/v1/audio/speech` endpoint. The `zvoice` service (built alongside this work in `HomeLabs/zvoice/`) matches this shape exactly, so the same provider serves both real OpenAI and local VoxCPM2.

Request body matches OpenAI TTS:

```json
{
  "model": "voxcpm2",                   // or "tts-1" / "tts-1-hd" for OpenAI
  "input": "<text>",
  "voice": "alloy",                     // preset name OR a voice id
  "response_format": "mp3",             // mp3 | wav | flac | opus
  "speed": 1.0                          // 0.25..4.0
}
```

Response: raw audio bytes with `Content-Type: audio/mpeg` (or `audio/wav` / `audio/flac` / `audio/ogg`).

### File layout

Extend the existing `openai.rs` rather than creating a new crate — the provider already holds a base URL and API key, so `impl AudioGeneration for OpenAi` is the cleanest add.

- `crates/blazen-llm/src/providers/openai.rs` — add `impl AudioGeneration for OpenAi` block. `text_to_speech` builds the OpenAI body from `SpeechRequest` (map `voice_url` into `parameters` since OpenAI proper ignores it but `zvoice` reads it), POSTs via `self.http_client.send(HttpRequest { ... })`, returns `AudioResult` with raw bytes and inferred MIME.
- `crates/blazen-llm/src/providers/mod.rs` — no change (provider already registered).
- `crates/blazen-llm/src/lib.rs` — re-export already present for `AudioGeneration`; just confirm `OpenAi` is visible.

### Config plumbing

The `OpenAi` provider struct already takes `base_url` + `api_key`. A `zvoice` instance is configured by constructing `OpenAi::new_with_client(api_key="", base_url="http://beastpc.lan:8900/v1", client)` — empty API key is fine because `zvoice` doesn't auth. Document this in `docs/providers.md` under a new "Local TTS (zvoice)" section.

### Bindings

- `crates/blazen-py/src/providers/openai.rs` — add a Python-exposed `text_to_speech(request)` method mirroring `crates/blazen-py/src/providers/fal.rs:326`.
- `crates/blazen-node/src/providers/openai.rs` — same for Node.
- WASM: flip `crates/blazen-wasm/src/router.rs:259` from 501 to a real `OpenAi::text_to_speech` call.

### Streaming (optional, v2)

OpenAI's TTS endpoint does not stream. `zvoice` exposes an extension: `{..., "stream": true}` returns chunked audio. If we want streaming in Blazen, add a second method `text_to_speech_stream` returning `impl Stream<Item = Result<Bytes>>`. Can be done later — for v1 just do the full-response form.

---

## Task 2 — `elevenlabs` provider

### Target API

`POST https://api.elevenlabs.io/v1/text-to-speech/{voice_id}` with header `xi-api-key: <key>`. Body:

```json
{
  "text": "<text>",
  "model_id": "eleven_multilingual_v2",
  "voice_settings": { "stability": 0.5, "similarity_boost": 0.75 }
}
```

Response: `audio/mpeg` bytes (or a streaming variant via `/stream` suffix).

Voice cloning is `POST /v1/voices/add` (multipart). Out of scope for v1 — just TTS against an existing voice id.

### File layout

New provider file. Follow the `fal.rs` structure but simpler (sync POST, no job polling).

- `crates/blazen-llm/src/providers/elevenlabs.rs` — new file. Struct `ElevenLabs { http_client, api_key, base_url }`. `impl ComputeProvider` + `impl AudioGeneration`. `SpeechRequest.voice` is the ElevenLabs `voice_id`; `SpeechRequest.parameters` can carry `stability` / `similarity_boost` / `model_id`.
- `crates/blazen-llm/src/providers/mod.rs` — `pub mod elevenlabs; pub use elevenlabs::ElevenLabs;`.
- `crates/blazen-llm/src/lib.rs` — re-export `ElevenLabs`.

### Bindings

- `crates/blazen-py/src/providers/elevenlabs.rs` — Python binding.
- `crates/blazen-node/src/providers/elevenlabs.rs` — Node binding.

---

## Shared work

- **Error mapping.** HTTP 401 → `BlazenError::Auth`, 429 → `BlazenError::RateLimit`, 5xx → `BlazenError::Upstream`. See how `fal.rs` handles this (line ~2941+) and match the convention.
- **`AudioResult`.** Confirm the struct has a field for raw bytes and a format/MIME string. If not, extend it — don't invent a second result type.
- **Tests.** Unit tests using a mock `HttpClient` (in-memory responder) that asserts the request body shape matches OpenAI / ElevenLabs. Integration tests gated behind a feature flag or env var so CI doesn't hit real endpoints.
- **Docs.** Update `docs/providers.md` with a "TTS providers" section listing `openai_tts` (pointing at OpenAI or `zvoice`) and `elevenlabs`.

---

## Reference: `zvoice` service endpoint contract

Full service lives in `/var/home/zach/github/HomeLabs/zvoice/`. Relevant bits for Blazen:

| Endpoint | Purpose |
|---|---|
| `POST /v1/audio/speech` | OpenAI-compatible TTS. This is the one the `openai_tts` provider hits. |
| `POST /v1/voices/design` | Create a voice from a natural-language description. Out of scope for Blazen — use zvoice's HTTP API directly or expose via a Blazen extension method later. |
| `POST /v1/voices/clone` | Clone a voice from an audio clip. Same — not in the `AudioGeneration` trait, handle separately. |
| `GET /v1/voices` | List available voices. Could back a future `list_voices()` method on `AudioGeneration`. |
| `GET /health` | Liveness. |

Default port: `8900`. Host: BeastPC or MiniBeast (whichever machine has a CUDA GPU free). Auth: none on LAN.

---

## Suggested sequencing

1. Read `crates/blazen-llm/src/providers/fal.rs` lines ~2941–end to lock in the `AudioGeneration` impl pattern.
2. Add `impl AudioGeneration for OpenAi` in `providers/openai.rs`. Write a unit test with a mock `HttpClient` that returns 200 + fake mp3 bytes.
3. Flip the WASM router stub at `crates/blazen-wasm/src/router.rs:259` to call it.
4. Add Python + Node bindings for `OpenAi::text_to_speech`.
5. Point at `zvoice` running locally and smoke test end-to-end.
6. Repeat for `elevenlabs`.
7. Update `docs/providers.md`.
