# Unsupported audio backends

This document explains audio backends that **don't ship** in Blazen, the reasons
why, and what would unblock each one. It is a deliberate record so users and
contributors can:

- Know where the line is between "supported" and "you're on your own."
- Have a concrete starting point if they want to implement an unsupported
  backend — including the rough size of the project, the upstream reference
  to port from, and which capability crate the new code should live in.
- Understand what was considered vs. what's truly out of scope.

The status snapshot is **after PR-AUDIO ships** (May 2026). Anything not listed
here either ships today (see the table below) or has never been seriously
evaluated for inclusion.

---

## Supported backends (for reference)

These backends ship as part of the PR-AUDIO restructure and are the canonical
recommended path. New work should plug into one of these capability crates
rather than reinvent the trait surface.

| Capability | Crate | Backend | Notes |
|---|---|---|---|
| Text-to-speech | `blazen-audio-tts` | `piper` | Vendored ONNX runtime; offline, English + ~30 other voices. |
| Text-to-speech | `blazen-audio-tts` | `openai` | Cloud proxy (OpenAI `tts-1` / `tts-1-hd`). |
| Text-to-speech | `blazen-audio-tts` | `anytts` | Generic HTTP proxy for self-hosted TTS servers. |
| Speech-to-text | `blazen-audio-stt` | `whispercpp` | whisper.cpp via FFI; GGML quantized models. |
| Speech-to-text | `blazen-audio-stt` | `candle` | Pure-Rust Whisper via `candle-transformers`. |
| Music / SFX generation | `blazen-audio-music` | `musicgen` | Meta MusicGen; ported in Wave 24, autoregressive over EnCodec tokens. |
| Music / SFX generation | `blazen-audio-music` | `audiogen` | Meta AudioGen; shares MusicGen infrastructure, SFX checkpoints. |
| Neural codec | `blazen-audio-codec` | `encodec` | Meta EnCodec, 24 kHz / 48 kHz. |
| Neural codec | `blazen-audio-codec` | `dac` | Descript Audio Codec. |
| Neural codec | `blazen-audio-codec` | `snac` | Multi-scale neural audio codec. |

`blazen-llm` exposes higher-level cloud `AudioGeneration` providers (fal.ai,
Replicate, ElevenLabs, etc.) for backends that don't have a local Rust port —
that is the "escape hatch" referenced throughout this document.

---

## Unsupported backends

### Generation (music / SFX / long-form audio)

#### StableAudio (Stability AI)

- **What it is.** Stable Audio is Stability AI's text-to-audio diffusion
  model, released in two public variants: **Stable Audio Open 1.0**
  (~1.2B parameters, CC-BY-3.0 trained on free-to-use data, max 47 seconds)
  and **Stable Audio 2.0** (closed-weights, hosted on their API, 3-minute
  full songs with structure). The architecture is a **latent diffusion
  transformer**: a custom VAE compresses audio into a continuous latent
  space, a DiT (diffusion transformer) denoises in that latent space
  conditioned on T5 text embeddings, and the VAE decoder reconstructs the
  waveform. Use case is "prompt-to-music with structure" — long-form
  generation, stems, sound design.

- **Status in Blazen.** A stub `StableAudioBackend` ships in
  `crates/blazen-audio-music/src/backends/stable_audio.rs`. Both entry
  points return `MusicError::NotYetImplemented` with a long-form message
  pointing here. The type is part of the binding surface so downstream
  code can route to a clear, actionable error rather than `no such type`.

- **Why not shipped.** No `candle` or `candle-transformers` port of
  Stable Audio exists today. Unlike MusicGen / AudioGen (which share the
  EnCodec + autoregressive-transformer recipe and only differ in
  checkpoints — making Wave 24's MusicGen port reusable), Stable Audio's
  latent-diffusion stack is a fresh architecture: custom VAE, DiT block
  variants, classifier-free-guidance sampler, and a non-standard time
  conditioning. Porting it from the PyTorch reference is a multi-week
  ground-up project, not a checkpoint-swap.

- **Unblocking it.** Two viable paths, in order of preference:
  1. **Wait for upstream.** A `candle-transformers::models::stable_audio`
     module would make this a one-week wrapper job. Watch the
     `huggingface/candle` repo; if a PR appears, file an issue here
     pointing at it.
  2. **Port it ourselves.** ~2000 LOC of fresh Rust against the
     `stability-ai/stable-audio-tools` Python reference. The DiT
     blocks and CFG sampler dominate; the VAE is straightforward
     conv-stack. Land it as a new `candle-transformers` module
     upstream first (so the wider candle community benefits), then
     wire it into `StableAudioBackend` here.
  3. **Cloud escape hatch (already available).** Users who need
     Stable Audio today can call Stability AI's hosted endpoint
     through `blazen-llm`'s generic HTTP `AudioGeneration` provider
     surface. That isn't a local backend, but it covers the use case.

- **Effort.** **Large — 4–6 person-weeks** for a full ground-up port +
  upstream PR + Blazen wiring + tests. Drops to **~1 person-week** if
  upstream lands the port first.

#### AudioLDM / AudioLDM2

- **What it is.** AudioLDM (and its successor AudioLDM2) are
  latent-diffusion audio models from the University of Surrey / OpenSound
  group, in roughly the same architectural family as Stable Audio. They
  target general audio (music + SFX + ambient) from text prompts. License
  is CC-BY-NC-SA (research only), which is part of why they get less
  industry attention than Stable Audio Open.

- **Status in Blazen.** Not present. No stub, no module, no enum variant.

- **Why not shipped.** Same blocker as Stable Audio (no candle port,
  fresh diffusion architecture to port), compounded by the non-commercial
  license — porting effort wouldn't unlock production use for most
  Blazen consumers.

- **Unblocking it.** If a candle port of either AudioLDM or AudioLDM2
  appears, add a new `audioldm.rs` backend file alongside `stable_audio.rs`
  in `crates/blazen-audio-music/src/backends/`. Implement the
  `MusicBackend` trait, gate it on an `audioldm` Cargo feature, and ship
  it as opt-in (so the NC license doesn't surprise anyone). Until then
  this is intentionally lower priority than Stable Audio Open (commercial
  license).

- **Effort.** **Large — 4–6 person-weeks** for a ground-up port.
  Comparable to Stable Audio. Realistically nobody should pick this up
  before Stable Audio Open, given the license gap.

#### Bark (Suno)

- **What it is.** Bark is Suno's open text-to-audio model that handles
  speech, non-verbal cues (`[laughter]`, `[sighs]`), music snippets, and
  sound effects from a single prompt. ~800M parameters, autoregressive
  over **EnCodec tokens**. The architecture is structurally very close
  to MMeta MusicGen — three transformer stages (semantic, coarse acoustic,
  fine acoustic) decoding sequentially into an EnCodec codebook stack.

- **Status in Blazen.** Not present. No stub.

- **Why not shipped.** Genuine candidate for inclusion, just not in
  scope for PR-AUDIO. The architectural similarity to MusicGen means the
  Wave 24 infrastructure (EnCodec wrapper, transformer stage harness,
  sampling utilities) is largely reusable. It's a "next-on-deck" backend,
  not a "wait years for upstream" one.

- **Unblocking it.**
  1. Port the three transformer stages against the
     `suno-ai/bark` Python reference. Use Wave 24's
     `crates/blazen-audio-music/src/backends/musicgen/` modules as the
     template — the autoregressive loop, KV-cache wiring, and EnCodec
     decode path all transfer.
  2. Add `crates/blazen-audio-music/src/backends/bark.rs` (Bark is a
     better fit for `music` than `tts` because of the non-verbal +
     mixed-modality output — pure-speech users should stick with Piper).
     If consensus lands on "primarily TTS," move it to
     `blazen-audio-tts/src/backends/bark.rs` instead.
  3. Land a `BarkBackend` implementing the relevant capability trait,
     feature-gate it (`bark`), wire it into the `blazen-llm` audio
     bridge.

- **Effort.** **Small-to-medium — 1–2 person-weeks**, leveraging
  Wave 24's MusicGen scaffolding. Easiest realistic addition in this
  document.

---

### Text-to-speech (research-grade 2024–2025 models)

The following three TTS models are all from the 2024–2025 research wave
and share the same blocker: **Python-only ecosystem as of May 2026**.
None has a candle, burn, or other production-grade Rust port. They're
grouped here because the unblocking pattern is identical for all three.

#### Spark-TTS

- **What it is.** Spark-TTS (SparkAudio, 2025) is an LLM-based TTS
  system that uses a single autoregressive model over BiCodec tokens
  (no separate acoustic model). Supports zero-shot voice cloning from
  a 3-second reference, cross-lingual synthesis, and fine-grained
  control over pitch/speed/emotion via natural-language instructions.
  Notable for unifying voice cloning and controllable synthesis in
  one model.

- **Status in Blazen.** Not present.

- **Why not shipped.** Python-only reference implementation; depends
  on the BiCodec tokenizer, which itself has no Rust port. Porting
  requires both the codec and the LLM-style autoregressive decoder.

- **Unblocking it.** Port the BiCodec encoder/decoder first (could
  live in `crates/blazen-audio-codec/src/backends/bicodec.rs` alongside
  DAC/SNAC/EnCodec). Then port the autoregressive decoder against the
  `SparkAudio/Spark-TTS` reference and land it as
  `crates/blazen-audio-tts/src/backends/spark.rs` implementing
  `TtsBackend`. The Piper vendoring pattern in
  `crates/blazen-audio-piper-vendored/` is the model for how to
  package a TTS backend whose upstream isn't a polished Rust crate.

- **Effort.** **Medium — 2–4 person-weeks** (codec + LLM-style
  decoder + voice-cloning reference path). The voice-cloning path
  pushes this toward the upper end.

#### F5-TTS

- **What it is.** F5-TTS (Microsoft / SWivid, 2024) is a **non-autoregressive**
  flow-matching TTS using a Diffusion Transformer (DiT) directly on
  mel-spectrograms. Trained on 100k hours; produces native-speaker-quality
  English / Chinese speech with zero-shot voice cloning from short
  references. Notable for being faster than autoregressive systems
  (single forward pass, not token-by-token) while staying competitive
  on quality.

- **Status in Blazen.** Not present.

- **Why not shipped.** Python-only; depends on a DiT block, a Vocos
  vocoder, and a flow-matching sampler — none of which have first-class
  candle implementations today.

- **Unblocking it.** This shares the StableAudio blocker (DiT in
  candle). If a candle DiT block lands for Stable Audio, F5-TTS becomes
  much cheaper because the same block can be reused. Port the
  flow-matching sampler, wire up a Vocos vocoder (also needs a port),
  and add `crates/blazen-audio-tts/src/backends/f5.rs`.

- **Effort.** **Medium — 2–4 person-weeks** **if** a candle DiT
  block already exists, otherwise **Large — 4–6 person-weeks** (the
  DiT port dominates and should land upstream in candle first).

#### MaskGCT

- **What it is.** MaskGCT (Amphion / OpenMMLab, 2024) is a
  non-autoregressive **masked-generative codec transformer** TTS — uses
  iterative parallel decoding (BERT-style mask prediction) over codec
  tokens rather than left-to-right autoregression. Two-stage: T2S
  generates semantic tokens, S2A generates acoustic tokens. Trained on
  100k hours; supports voice cloning.

- **Status in Blazen.** Not present.

- **Why not shipped.** Python-only; non-trivial masked-generation
  sampling loop with no Rust analogue in `candle-transformers`.

- **Unblocking it.** Port the two-stage masked decoder against the
  `open-mmlab/Amphion` reference. The acoustic stage decodes into a
  codec already in `blazen-audio-codec` (likely DAC or a variant), so
  the back half of the pipeline is essentially free. Land as
  `crates/blazen-audio-tts/src/backends/maskgct.rs`.

- **Effort.** **Medium — 2–4 person-weeks**. Of the three 2024–2025
  models, MaskGCT is probably the most tractable because the codec
  reuse is genuine.

---

### Voice cloning / conversion

#### RVC (Real-time Voice Conversion)

- **What it is.** RVC (Retrieval-based Voice Conversion) is the
  open-source voice-cloning toolkit that took off in the 2023–2024
  AI-cover community. Architecturally: an F0 (fundamental frequency)
  extractor + a content encoder (typically HuBERT or ContentVec) + a
  generator that's content-conditioned, F0-conditioned, and uses a
  **FAISS index over training-set content embeddings** for retrieval
  augmentation. Real-time-capable on modest GPUs. Use case is "take a
  source voice recording and re-render it as a target voice you've
  trained on a few minutes of audio."

- **Status in Blazen.** Not present. No stub, no enum variant.

- **Why not shipped.** RVC is a **pure-PyTorch ecosystem with strong
  Python coupling**. The F0 extractor (typically `rmvpe` or `crepe`) is
  Python-only; the content encoder is HuBERT/ContentVec which need
  fairseq-style infrastructure; the FAISS index is C++ with a Python
  wrapper but no first-class Rust binding for the retrieval pattern RVC
  uses. Each piece is portable individually, but the cumulative work is
  large and there is no candle precedent for the retrieval-augmented
  generator architecture.

- **Unblocking it.** A genuine production port would mean:
  1. F0 extractor in Rust (port `rmvpe` against its PyTorch ref, or
     wrap `crepe` via ONNX runtime).
  2. ContentVec / HuBERT encoder in candle.
  3. The RVC generator itself (~20M parameters, but with non-trivial
     conditioning).
  4. FAISS retrieval — `faiss-rs` exists but is thin; the RVC-style
     "top-k blend with content embedding" path needs to be written
     from scratch.
  5. Land as a new `blazen-audio-voice-conversion` crate (it's a
     different capability than TTS), and a new
     `VoiceConversionBackend` trait at the `blazen-audio` level.
     A cleaner short-term move is to ship it as an HTTP-proxy
     backend wrapping a user-hosted RVC server, similar to the
     OpenAI TTS proxy pattern.

- **Effort.** **Large — 4–8 person-weeks** for a full local port.
  Drops to **~1 person-week** for an HTTP-proxy backend that talks to
  a user-managed RVC server (the realistic first step).

---

### Speech-to-text

#### Whisper-streaming (low-latency real-time STT)

- **What it is.** Despite the name, this is **not a model** — it's a
  *technique* for adapting batch Whisper to low-latency real-time
  transcription. The recipe is: chunk audio into short windows (~200ms),
  run voice-activity-detection to skip silence, decode each window
  with KV-cache reuse, emit "stable" hypothesis prefixes (tokens that
  haven't changed across the last N windows) as final, and revise the
  unstable tail. Reference implementation is
  `ufal/whisper_streaming` (Python).

- **Status in Blazen.** Not present. Both Whisper backends
  (`whispercpp`, `candle`) only expose batch / file-based transcription
  today.

- **Why not shipped.** Not a backend, a mode flag on an existing
  backend. The PR-AUDIO scope was "land a clean backend trait surface,"
  not "add streaming modes." Streaming requires plumbing through every
  layer from the FFI binding down to the model invocation, plus a
  voice-activity-detector (Silero VAD is the usual choice, has an ONNX
  release that's straightforward to wire in).

- **Unblocking it.** Add a `stream` method to `SttBackend` returning
  an async `Stream<Item = StreamingTranscript>` (where
  `StreamingTranscript` carries `stable: String` + `unstable: String`
  segments). Implement it on `whispercpp.rs` and `candle.rs` using
  the chunked-decoding + stable-prefix algorithm from
  `ufal/whisper_streaming`. Add Silero VAD as an
  `blazen-audio-stt` dependency (probably as a vendored ONNX session
  similar to Piper's approach).

- **Effort.** **Small — 1 person-week**. The algorithm is well-known
  and the existing backends only need a new entry point, not a new
  model.

#### faster-whisper (CTranslate2-based STT)

- **What it is.** `faster-whisper` is a Python wrapper around
  `CTranslate2` — a C++ inference engine for transformer models with
  aggressive INT8 quantization. Delivers ~4× faster Whisper decoding
  than HuggingFace Transformers at near-identical quality. Widely used
  in self-hosted transcription pipelines.

- **Status in Blazen.** Not present.

- **Why not shipped.** CTranslate2 has C++ headers but **no
  first-class Rust binding**. Wrapping it via `bindgen` is possible
  but the API surface is large and the build story (CTranslate2 needs
  to be present on the host, with the right CUDA/MKL flavor) is messy
  for downstream consumers. `whispercpp` already covers the "fast
  C++ Whisper" niche for Blazen with a much cleaner FFI story.

- **Unblocking it.** Two paths:
  1. **HTTP-proxy backend (recommended first step).** Ship
     `crates/blazen-audio-stt/src/backends/faster_whisper.rs` as
     an HTTP client targeting a user-hosted `faster-whisper-server`
     or `WhisperX` deployment. Identical pattern to the OpenAI
     TTS proxy. Gives users the perf win without Blazen owning the
     C++ build.
  2. **Native binding (only if there's demand).** Wrap CTranslate2
     via `bindgen` or `cxx`, vendor headers, manage the build matrix.
     Significantly more work than `whispercpp` because CTranslate2 is
     a general-purpose engine, not a single-model wrapper.

- **Effort.** **Small — 1 person-week** for the HTTP-proxy backend.
  **Large — 4–8 person-weeks** for a native CTranslate2 binding.

---

## Adding a new audio backend

Each capability crate (`blazen-audio-tts`, `blazen-audio-stt`,
`blazen-audio-music`, `blazen-audio-codec`) follows the same pattern:

```
crates/blazen-audio-<capability>/
  src/
    backends/
      mod.rs              # registers backends, often behind cargo features
      <your_backend>.rs   # your code, implements the capability trait
    traits.rs             # the capability trait you implement
```

To add a backend:

1. Pick the right capability crate. Music/SFX/long-form goes in
   `blazen-audio-music`. Pure speech goes in `blazen-audio-tts`. Neural
   codecs go in `blazen-audio-codec`. STT (batch or streaming) goes in
   `blazen-audio-stt`. If your capability genuinely doesn't fit any of
   these (e.g. voice conversion), open an issue first to discuss adding
   a new capability crate.
2. Add a new file under `src/backends/` named after the backend.
3. Implement the relevant capability trait (`TtsBackend`, `SttBackend`,
   `MusicBackend`, `CodecBackend`).
4. Gate the backend behind a Cargo feature so users only pay for what
   they use. Match the existing feature naming
   (`<crate>/<backend-name>`).
5. Wire the backend into `blazen-llm/src/backends/audio_*.rs` so the
   higher-level Blazen compute traits expose it via the public API.
6. Add tests under `tests/` (or `crates/<crate>/tests/`). Mock the
   model if it's heavy; gate any real-model integration tests behind
   an env var so CI doesn't have to fetch weights.
7. Regenerate typegens for all three high-traffic bindings (`pyi`,
   `index.d.ts`, `blazen_wasm_sdk.d.ts`) and the UniFFI files — see
   the root `CLAUDE.md` for the exact commands. CI's `audit-bindings`
   job blocks merge on drift.

For a **worked example of vendoring + patching an upstream Rust crate
to make it usable in Blazen**, read
`crates/blazen-audio-piper-vendored/VENDORED.md`. That's the canonical
pattern for "the upstream exists but it doesn't quite fit," which is
the most common situation for the backends listed above.

---

## What's *not* in this document

- **Cloud-only providers** (ElevenLabs, PlayHT, Murf, OpenAI Realtime,
  Cartesia, Deepgram, etc.). These are covered by the generic cloud
  `AudioGeneration` / `Stt` surfaces in `blazen-llm` — adding a new
  cloud provider is a config change, not a backend implementation.
  This document is about **local-inference** backends.
- **Models without a permissive-enough license to ship as a default**
  (commercial-API-only models from any vendor). Same rule: if it's
  cloud-only with a public HTTP API, it belongs in `blazen-llm`'s
  provider list, not here.
- **Models that no one has asked for.** This list is curated against
  actual user requests + research-community momentum as of May 2026.
  If you want a backend that isn't listed here, file an issue —
  this document is meant to be added to.
