# Unsupported audio backends

As of **2026-05-25**, the bulk of the audio backends previously listed in this
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
| Text-to-speech | `blazen-audio-tts` | `spark-tts` | Spark-TTS, Qwen2.5 + BiCodec AR pipeline with per-token streaming; `clone_voice` pending wav2vec2-XLS-R port. |
| Voice conversion | `blazen-audio-voice-conversion` | `rvc` | Retrieval-based VC. **Caveat:** the ContentVec loader is still deferred — see the crate README for the interim path. |
| Speech-to-text | `blazen-audio-stt` | `whispercpp` | whisper.cpp via FFI. |
| Speech-to-text | `blazen-audio-stt` | `candle` | Pure-Rust Whisper. |
| Speech-to-text | `blazen-audio-stt` | `whisper-streaming` | Chunked low-latency mode with Silero VAD on top of either backend. |
| Speech-to-text | `blazen-audio-stt` | `faster-whisper` | CTranslate2-backed Whisper via `ct2rs`; HF download + window-streaming decode. |
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

### MaskGCT

**MaskGCT** — Amphion's 2024 non-autoregressive masked-generative codec
transformer TTS, two-stage (T2S then S2A) with iterative parallel decoding
over codec tokens. Of the 2024-2025 TTS wave this is the most tractable port
because the acoustic stage decodes into a codec already in
`blazen-audio-codec`. The blocker is the masked-generation sampling loop — no
candle analogue exists today. **Status:** Wave M.1 scaffold landed
(`crates/blazen-audio-tts/src/backends/maskgct/`, ~217 LOC) — `synthesize`
currently returns `TtsError::Unsupported` until the T2S/S2A masked-generation
sampling loop ports. Upstream:
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
non-commercial license is opt-in. **Status:** Wave H.1 scaffold landed
(`crates/blazen-audio-music/src/backends/audioldm2/`, ~127 LOC across nine
1-line stub modules); the `audioldm` feature compiles but every entry point
returns `MusicError::NotYetImplemented` until conditioner/UNet/VAE/vocoder/
sampler/pipeline/weights ports complete. Upstream:
<https://github.com/haoheliu/AudioLDM2> (paper:
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
