# blazen-audio-stt

Multi-backend speech-to-text engine crate for Blazen. Surfaces a single `SttBackend` trait (`transcribe(file)` + `stream(audio)`) with three interchangeable implementations selected by Cargo feature.

| Feature | Backend module | Underlying engine |
|---|---|---|
| `whispercpp` | `backends::whispercpp` | [whisper.cpp](https://github.com/ggerganov/whisper.cpp) via [`whisper-rs`](https://crates.io/crates/whisper-rs) |
| `candle` | `backends::candle` | Pure-Rust Whisper via [`candle-transformers`](https://crates.io/crates/candle-transformers) |
| `whisper-streaming` | `backends::whisper_streaming` | Candle Whisper + Silero VAD (`tract-onnx`) for low-latency streaming |
| `faster-whisper` | `backends::faster_whisper` | [SYSTRAN/faster-whisper](https://github.com/SYSTRAN/faster-whisper)-style CTranslate2 INT8 Whisper via [`ct2rs`](https://crates.io/crates/ct2rs) |

Backends can be combined freely; each registers its own `AudioBackend::id`, so multiple instances may coexist in the same process.

## faster-whisper backend

CTranslate2-backed Whisper inference. Substantially faster than `candle` / `whispercpp` on CPU thanks to CTranslate2's INT8 quantisation and AVX-512 / NEON kernels.

- **Feature flag**: `faster-whisper`
- **Backend module**: `blazen_audio_stt::backends::faster_whisper`
- **Rust binding**: [`ct2rs`](https://crates.io/crates/ct2rs) `0.9.18` with the `whisper` sub-feature.
- **License stack**: MIT (`ct2rs`) over MIT (CTranslate2) over MIT (Whisper) — commercially safe end-to-end, unlike Spark-TTS / MaskGCT / AudioLDM.

### System requirements

`ct2rs` vendors and compiles CTranslate2's C++ source on the first build. You need:

| Requirement | Notes |
|---|---|
| C++ toolchain | `gcc` ≥ 9 or `clang` ≥ 10 (any C++17-capable compiler). |
| `cmake` ≥ 3.18 | CTranslate2's build script invokes cmake. |
| **Optional** CUDA toolkit | Enables `ct2rs/cuda` for GPU inference. Not pulled in by `faster-whisper` itself; activate via a direct `ct2rs` dep in your binary crate (see "GPU acceleration" below). |
| **Optional** OpenBLAS / MKL / Apple Accelerate | Faster CPU paths. CTranslate2 auto-detects what's installed at build time; with none of them present it falls back to its bundled Ruy kernels (the `ct2rs/ruy` default). |

On Debian/Ubuntu: `apt install build-essential cmake`. On Fedora/RHEL: `dnf install gcc-c++ cmake`. On macOS: `xcode-select --install` (cmake via `brew install cmake`).

### First-build cost

Building this feature compiles CTranslate2 from source (vendored by `ct2rs`). Expect **~5–10 minutes on the first `cargo build --features faster-whisper`** on a typical workstation; incremental rebuilds are fast (a few seconds, the C++ object files are cached in `target/`). Subsequent `cargo build`s of unrelated crates are unaffected.

CI runs that toggle `--features faster-whisper` cold will pay this cost too — budget accordingly.

### GPU acceleration

`faster-whisper` activates only `ct2rs`'s default features (`all-tokenizers`, `ruy`, `cuda-small-binary`) plus `whisper`. The cuda-small-binary flag means the cuda code path is *available* but not statically linked; CTranslate2 loads it dynamically if a CUDA toolkit is present at runtime. To force CUDA / cuDNN / Flash Attention at compile time, add `ct2rs` as a direct dependency in your binary crate with the appropriate features:

```toml
[dependencies]
ct2rs = { version = "0.9.18", default-features = false, features = ["whisper", "cuda", "cudnn", "flash-attention"] }
```

Cargo feature unification merges them with the `faster-whisper` dep here.

### Status

- **Wave F.0** — scaffolding (`FasterWhisperBackend` returns `SttError::Unsupported`).
- **Wave F.1** — `ct2rs` dependency wired in + FFI link-probe test (`link_probe_ct2rs_loads_successfully`) confirms the CTranslate2 shared library loads and `LogLevel` round-trips across the `cxx` boundary.
- **Wave F.2.x** — real audio frontend, tokenizer, decoder, weight downloader, and pipeline land here; the live `transcribe` / `stream` path goes green.

## Other backends

See the in-module docs (`cargo doc -p blazen-audio-stt --features whispercpp,candle,whisper-streaming,faster-whisper`) for backend-specific knobs, default model ids, and HuggingFace repo layouts.
