# Changelog

## Unreleased

### Added

- WASM-SDK reaches feature parity with Node bindings: pipeline `input_mapper` / `condition` / `onPersist` / `onPersistJson` callbacks, workflow `setSessionPausePolicy` / `runStreaming` / `runWithHandler` / `resumeWithSerializableRefs`, handler `respondToInput` / `snapshot` / `resumeInPlace` / `streamEvents` / `abort`, context `insertSessionRefSerializable` / `getSessionRefSerializable`, real `ModelManager` bind, `InMemoryBackend` + `MemoryResult` standalone classes.
- WASM-SDK `TractEmbedModel.create(modelUrl, tokenizerUrl)` re-enabled via `web_sys::fetch`-based model + tokenizer loader (no `hf-hub` dep on wasm32).
- WASM-SDK `OtlpConfig` + `initOtlp` via custom `WasmFetchHttpClient` HTTP transport (works around `opentelemetry-otlp/grpc-tonic` incompatibility with wasm32).
- `MediaSource` type alias of `ImageSource` exposed in all three bindings.
- Backend inference types bound to py + node: 9 mistralrs (`ChatMessageInput`, `ChatRole`, `InferenceChunk`, `InferenceChunkStream`, `InferenceImage`, `InferenceImageSource`, `InferenceResult`, `InferenceToolCall`, `InferenceUsage`), 6 llamacpp (under `LlamaCpp` prefix), 1 candle (`CandleInferenceResult`).
- `LangfuseConfig` + `init_langfuse` exporter on py + node (new `langfuse` feature in `blazen-telemetry`).
- Python: `OtlpConfig` + `init_otlp` and `init_prometheus(port)` now exposed in `blazen.pyi`.
- Python error hierarchy now includes 9 per-backend `ProviderError` subclasses (`LlamaCppError`, `CandleLlmError`, `CandleEmbedError`, `MistralRsError`, `WhisperError`, `PiperError`, `DiffusionError`, `FastEmbedError`, `TractError`).
- Node binding: typed JS error class hierarchy (~87 classes via JS shim post-processor, rooted at `BlazenError`), `AgentResult` / `BatchResult` upgraded from plain dicts to typed classes with getters and `toString()`, `PipelineBuilder.onPersist` / `.onPersistJson`, `ProgressCallback` subclassable class, `enrichError(err)` helper.
- Python binding: `ProgressCallback` ABC.
- `blazen-telemetry` gained `otlp-http` Cargo feature for HTTP transport (alongside the existing `otlp` gRPC variant).

### Changed

- CI's `audit-bindings` job now regenerates all three typegens (`blazen.pyi`, `index.d.ts`, `pkg/blazen_wasm_sdk.d.ts`) and fails on drift before running the parity audit.
- `cargo run --example stub_gen -p blazen-py` now requires `--features langfuse,otlp,prometheus,tract,distributed` to surface all feature-gated bindings (CI, `.githooks/pre-commit`, and `CLAUDE.md` updated).
- `auto-tag` release job now blocked on `audit-bindings`.

### Fixed

- `blazen-node`: `JsAgentResult.toString()` and `JsBatchResult.toString()` no longer trip clippy's `inherent_to_string` lint (renamed Rust method, kept JS-facing `toString` via napi `js_name`).

### CI

- **`aarch64-unknown-linux-musl` wheels and napi binary now build.** The
  `rust-musl-cross:aarch64-musl` image ships only long-triple binaries
  (`aarch64-unknown-linux-musl-gcc`), but Rust's target spec defaults the
  linker to the short form (`aarch64-linux-musl-gcc`). Prior attempts to
  override this via `CARGO_TARGET_*_LINKER` in `$GITHUB_ENV` did not
  propagate through `pnpm` / `uvx` to the cargo subprocess on the Forgejo
  runner. The probe step now creates short-triple symlinks in both
  `/usr/local/musl/bin` and `/usr/local/bin` so cargo's default lookup
  succeeds with zero env-var plumbing.
- **`x86_64-pc-windows-msvc` wheels and napi binary now build.**
  `llama-cpp-sys-2`'s `build.rs` unconditionally forwards every `CMAKE_*`
  environment variable as a `-D` flag to cmake. The Windows runner host
  injects `CMAKE_C_COMPILER_LAUNCHER=ccache` / `CMAKE_CXX_COMPILER_LAUNCHER=ccache`
  into the process env from some machine- or user-scope source outside the
  workflow, which then reached cmake, which then made Ninja try to spawn
  `ccache` — failing with `CreateProcess: The system cannot find the file
  specified`. The `Build wheels` and `Build napi binary` steps are now
  split into Windows (PowerShell) and non-Windows (bash) siblings; the
  Windows branch does `Remove-Item Env:CMAKE_*_COMPILER_LAUNCHER` before
  spawning `uvx maturin build` / `pnpm exec napi build`, so child
  processes inherit a clean env and the `-D` flag is never emitted.
- Added `Diagnose inherited env (Windows)` step (permanent) that dumps
  `CMAKE_*` / `*LAUNCHER*` variables at Process, Machine, and User scope
  on every Windows build, so the source of any future launcher leak is
  immediately visible in the log.
