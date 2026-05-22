# Unsupported audio backends

As of **2026-05-22**, the bulk of the audio backends previously listed in this
document have landed as native Rust ports. This file now tracks the remaining
gaps and points at the upstream public repositories that back each pending
port.

---

## Native backends shipped today

| Capability | Crate | Backend | Notes |
|---|---|---|---|
| Text-to-speech | `blazen-audio-tts` | `piper` | Vendored ONNX runtime; offline, ~30 voices. |
| Text-to-speech | `blazen-audio-tts` | `openai` | Cloud proxy (`tts-1` / `tts-1-hd`). |
| Text-to-speech | `blazen-audio-tts` | `anytts` | Generic HTTP proxy. |
| Text-to-speech | `blazen-audio-tts` | `bark` | Suno Bark, autoregressive over EnCodec; mixed speech + SFX. |
| Text-to-speech | `blazen-audio-tts` | `f5` | F5-TTS, flow-matching DiT, zero-shot voice cloning. |
| Voice conversion | `blazen-audio-voice-conversion` | `rvc` | Retrieval-based VC. **Caveat:** the ContentVec loader is still deferred — see the crate README for the interim path. |
| Speech-to-text | `blazen-audio-stt` | `whispercpp` | whisper.cpp via FFI. |
| Speech-to-text | `blazen-audio-stt` | `candle` | Pure-Rust Whisper. |
| Speech-to-text | `blazen-audio-stt` | `whisper-streaming` | Chunked low-latency mode with Silero VAD on top of either backend. |
| Music / SFX | `blazen-audio-music` | `musicgen` | Meta MusicGen, autoregressive over EnCodec. |
| Music / SFX | `blazen-audio-music` | `audiogen` | Meta AudioGen, shares MusicGen infra. |
| Music / SFX | `blazen-audio-music` | `stable_audio` | Stable Audio Open 1.0, latent-diffusion DiT. |
| Neural codec | `blazen-audio-codec` | `encodec` | Meta EnCodec, 24/48 kHz. |
| Neural codec | `blazen-audio-codec` | `dac` | Descript Audio Codec. |
| Neural codec | `blazen-audio-codec` | `snac` | Multi-scale neural audio codec. |

Cloud providers (ElevenLabs, OpenAI Realtime, Cartesia, Deepgram, fal.ai,
Replicate, etc.) are exposed through the `AudioGeneration` / `Stt` surfaces in
`blazen-llm` and are not tracked here.

---

## Pending native ports

### faster-whisper (CTranslate2-based STT)

**faster-whisper** — CTranslate2 INT8 GEMM dispatch wrapping OpenAI Whisper,
roughly an order of magnitude faster than the HF Transformers reference at
near-identical quality. License: MIT (CTranslate2) + MIT (Whisper). CTranslate2
has no first-class Rust binding and `whispercpp` already covers the
"fast C++ Whisper" niche, so the realistic first step is an HTTP-proxy backend
against a user-hosted `faster-whisper-server` or `WhisperX` deployment. A
native `cxx`/`bindgen` wrapper is possible but expensive due to the breadth of
the CTranslate2 API and the CUDA/MKL build matrix. **Status:** pending native
Rust port; the practical path is a `ct2rs` FFI wrapper over the existing
CTranslate2 C++ library. Upstream: <https://github.com/SYSTRAN/faster-whisper>
(original Whisper paper: <https://arxiv.org/abs/2212.04356>).

### Spark-TTS

**Spark-TTS** — SparkAudio's 2025 LLM-style TTS: a single autoregressive
Qwen2.5-based model over BiCodec tokens with zero-shot voice cloning and
natural-language pitch/speed/emotion control. Blocked on a Rust BiCodec port
(lands in `blazen-audio-codec` alongside DAC/SNAC/EnCodec) plus the
autoregressive decoder itself. The Piper vendoring pattern in
`blazen-audio-piper-vendored/` is the template for packaging the TTS half.
**Status:** pending native Rust port. Upstream:
<https://github.com/SparkAudio/Spark-TTS> (paper:
<https://arxiv.org/abs/2503.01710>).

### MaskGCT

**MaskGCT** — Amphion's 2024 non-autoregressive masked-generative codec
transformer TTS, two-stage (T2S then S2A) with iterative parallel decoding
over codec tokens. Of the 2024-2025 TTS wave this is the most tractable port
because the acoustic stage decodes into a codec already in
`blazen-audio-codec`. The blocker is the masked-generation sampling loop — no
candle analogue exists today. **Status:** pending native Rust port. Upstream:
<https://github.com/open-mmlab/Amphion/tree/main/models/tts/maskgct> (weights:
<https://huggingface.co/amphion/MaskGCT>, paper:
<https://arxiv.org/abs/2409.00750>).

### AudioLDM / AudioLDM2

**AudioLDM / AudioLDM2** — University of Surrey / OpenSound latent-diffusion
audio models targeting music + SFX + ambient from text. **License:
CC-BY-NC-SA-4.0 on the released weights — research / non-commercial only**,
which is why this stays opt-in and lower-priority than Stable Audio Open
(which already ships natively). Architectural blockers overlap with the
already-landed Stable Audio port; the realistic path is to add an
`audioldm.rs` backend in `blazen-audio-music` reusing the DiT / VAE
scaffolding, gated behind an explicit `audioldm` Cargo feature so the
non-commercial license is opt-in. **Status:** pending native Rust port.
Upstream: <https://github.com/haoheliu/AudioLDM2> (paper:
<https://arxiv.org/abs/2308.05734>).

---

## Adding a new audio backend

The mechanics are unchanged from prior revisions — capability crates
(`blazen-audio-tts`, `blazen-audio-stt`, `blazen-audio-music`,
`blazen-audio-codec`, `blazen-audio-voice-conversion`) follow the same
`src/backends/<name>.rs` + Cargo-feature-gated pattern. Wire new backends
into `blazen-llm/src/backends/audio_*.rs` and regenerate all binding
typegens (see root `CLAUDE.md`) before merging.

For a worked vendoring example see
`crates/blazen-audio-piper-vendored/VENDORED.md`.
