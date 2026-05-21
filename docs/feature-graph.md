# Blazen Cargo Feature Graph — Audit & Redesign

Spike output for PR1.5. Read-only research; no source changes other than this doc.

> **2026-05 PR-AUDIO update.** Sections referring to `blazen-audio-whispercpp`,
> `blazen-audio-piper`, and `blazen-audio-candle` are now historical. Those
> crates were dissolved in the PR-AUDIO restructure (Waves 1–28):
>
> - `blazen-audio-whispercpp` → folded into **`blazen-audio-stt`** as the
>   `whispercpp` backend (cargo feature `audio-stt-whispercpp` on `blazen-llm`).
> - `blazen-audio-piper` → folded into **`blazen-audio-tts`** as the `piper`
>   backend (cargo feature `audio-tts-piper`).
> - `blazen-audio-candle` → split into **`blazen-audio-music`** (MusicGen /
>   AudioGen / Stable Audio) and **`blazen-audio-codec`** (EnCodec / DAC /
>   SNAC). Cargo features `audio-music-*` and `audio-codec-*` on `blazen-llm`
>   gate each backend.
> - NEW root crate **`blazen-audio`** holds the capability-agnostic
>   vocabulary (`AudioBackend` trait, `AudioError`, lifecycle helpers).
> - NEW crate **`blazen-audio-piper-vendored`** patches upstream `piper-rs`
>   to use `tract-onnx` + subprocess `espeak-ng`. See `docs/vendored-deps.md`.
>
> Deprecated aliases live on `blazen-llm` for one transition window:
> `whispercpp` → `audio-stt-whispercpp`, `piper` → `audio-tts-piper`,
> `tts` → `audio-tts-anytts`, `candle-audio` → `audio-music-musicgen`.
> See `CHANGELOG.md` and `docs/UNSUPPORTED_AUDIO.md` for the full picture.

This catalogs every Cargo feature across the 17 crates in the Blazen workspace
(14 workspace members + the 3 excluded WASM/component crates), names the sharp
edges, and proposes a unified redesign.

WASM is intentionally gated through `#[cfg(target_arch = "wasm32")]` and
target-conditional dependency blocks, **not** a `wasm` Cargo feature. The
redesign keeps that.

---

## Section 1 — Current-state inventory

Format: per crate, feature table + notable optional deps + a sample of
non-trivial source-cfg gates. "wp" = workspace path. Line numbers point to
the head of the `[features]` block or the relevant target-cfg dep.

### `blazen` (`crates/blazen/Cargo.toml:13`)

| Feature | Default? | Enables |
|---|---|---|
| `llm` | yes | `blazen-llm` (re-export only — see sharp edges) |
| `persist` | no | `blazen-persist`, `blazen-core/persist` |
| `pipeline` | no | `blazen-pipeline` |
| `prompts` | no | `blazen-prompts` |
| `telemetry` | no | `blazen-telemetry`, `blazen-core/telemetry` |
| `all` | no | `llm` + `persist` + `pipeline` + `prompts` + `telemetry` |
| `local-embeddings` | no | `blazen-llm/embed`, `blazen-llm/candle-embed` |
| `local-llm` | no | `blazen-llm/mistralrs`, `blazen-llm/llamacpp`, `blazen-llm/candle-llm` |
| `local-image` | no | `blazen-llm/diffusion` |
| `local-audio` | no | `blazen-llm/whispercpp`, `blazen-llm/piper`, `blazen-llm/candle-audio` |
| `local-all` | no | all four `local-*` groups |

Optional deps: `blazen-llm`, `blazen-persist`, `blazen-pipeline`, `blazen-prompts`,
`blazen-telemetry`.

### `blazen-core` (`crates/blazen-core/Cargo.toml:60`)

| Feature | Default? | Enables |
|---|---|---|
| `persist` | no | `dep:blazen-persist` |
| `telemetry` | no | `dep:blazen-telemetry` (with `history` feature) |
| `distributed` | no | (marker — adds `cfg(feature = "distributed")` code paths) |

Hard deps (not optional) on: `blazen-llm`, `blazen-events`, `blazen-macros`,
`tokio` (`sync`+`macros`), `tokio-stream`, `dashmap`, `futures-{util,core}`,
`tracing`, `serde`, `serde_json`, `rmp-serde`, `serde_bytes`, `uuid`, `chrono`,
`async-trait`, `anyhow`, `thiserror`.

Non-trivial cfg sites: `snapshot.rs:84/186/192/195/252/304/355/422/569/582`
(`feature = "persist"` / `feature = "telemetry"`), `workflow.rs:415` etc.

Per-target dep blocks distinguish:
- native (rt+time)
- `cfg(target_arch = "wasm32")`: `web-time`
- browser wasm (`wasm32, not wasi`): `wasm-bindgen`, `wasm-bindgen-futures`, `js-sys`, `uuid/js`
- wasi: `tokio[rt,time]`, `futures-executor`

### `blazen-events` (`crates/blazen-events/Cargo.toml:147`)

| Feature | Default? | Enables |
|---|---|---|
| `tsify` | no | `dep:tsify-next`, `dep:wasm-bindgen` |

### `blazen-macros`

No `[features]` block. Proc-macro crate with hard deps only.

### `blazen-llm` (`crates/blazen-llm/Cargo.toml:195`)

| Feature | Default? | Enables |
|---|---|---|
| `reqwest` | no | `dep:reqwest` (native HTTP) |
| `tiktoken` | no | `dep:tiktoken-rs` |
| `content-detect` | **yes** | `dep:infer` |
| `tsify` | no | `dep:tsify-next`, `dep:wasm-bindgen` |
| `pyo3-serde` | no | `dep:pythonize`, `dep:pyo3` |
| `embed` | no | `dep:blazen-embed` |
| `candle-embed` | no | `dep:blazen-embed-candle` |
| `mistralrs` | no | `dep:blazen-llm-mistralrs` |
| `llamacpp` | no | `dep:blazen-llm-llamacpp` |
| `candle-llm` | no | `dep:blazen-llm-candle` |
| `diffusion` | no | `dep:blazen-image-diffusion` |
| `whispercpp` | no | `dep:blazen-audio-whispercpp` |
| `piper` | no | `dep:blazen-audio-piper` |
| `candle-audio` | no | **(marker — no `dep:` activator, no source uses, no crate yet)** |

Optional deps: `reqwest`, `tiktoken-rs`, `infer`, `tsify-next`, `wasm-bindgen`,
`pythonize`, `pyo3`, `blazen-embed`, `blazen-llm-mistralrs`,
`blazen-audio-whispercpp`, `blazen-embed-candle`, `blazen-image-diffusion`,
`blazen-llm-candle`, `blazen-llm-llamacpp`, `blazen-audio-piper`.

Source gates: `lib.rs:203,229,232,239,242,245,250,252,257,265`,
`backends/mod.rs:7,10,13,16,19,22`, `content/detect.rs:30,129,141`.

Per-target dep blocks: native (tokio rt+time+fs), browser wasm
(`wasm-bindgen` family, `web-sys[Request,RequestInit,...]`, `getrandom/js`),
cross-wasm (`web-time`).

### `blazen-memory` (`crates/blazen-memory/Cargo.toml:296`)

| Feature | Default? | Enables |
|---|---|---|
| `jsonl` | **yes** | (marker — no `dep:` activator; gates `JsonlMemoryBackend` source) |

Hard deps on `blazen-llm`, `elid["embeddings"]`, etc.

### `blazen-memory-valkey` (`crates/blazen-memory-valkey/Cargo.toml:353`)

| Feature | Default? | Enables |
|---|---|---|
| `redis-tcp` | **yes** | `dep:redis`, `dep:tokio` |
| `upstash` | no | `dep:blazen-llm` (uses its `HttpClient` trait, no tokio) |

### `blazen-persist` (`crates/blazen-persist/Cargo.toml:390`)

| Feature | Default? | Enables |
|---|---|---|
| `redb` | **yes** | `dep:redb` |
| `valkey` | no | `dep:redis` |

### `blazen-pipeline`

No `[features]`. Hard deps only.

### `blazen-prompts`

No `[features]`. Hard deps only.

### `blazen-telemetry` (`crates/blazen-telemetry/Cargo.toml:558`)

| Feature | Default? | Enables |
|---|---|---|
| `spans` | **yes** | `dep:blazen-llm` |
| `history` | no | (marker — gates `HistoryEvent` type) |
| `otlp` | no | OTel SDK + grpc-tonic transport (`opentelemetry*`, `tracing-opentelemetry`) |
| `otlp-http` | no | OTel SDK + http-proto + `dep:reqwest`/`http`/`bytes` + browser-wasm shims (`wasm-bindgen*`, `js-sys`, `web-sys`) + `dep:blazen-llm` |
| `prometheus` | no | `dep:metrics`, `dep:metrics-exporter-prometheus` |
| `langfuse` | no | `dep:reqwest`, `dep:wasm-bindgen-futures`, `dep:blazen-llm`, `dep:base64` |
| `all` | no | all of the above |

Optional deps: `opentelemetry`, `opentelemetry_sdk`, `opentelemetry-otlp`,
`opentelemetry-http`, `tracing-opentelemetry`, `http`, `bytes`, `metrics`,
`metrics-exporter-prometheus`, `base64`, `reqwest`, `blazen-llm`,
plus browser-only `wasm-bindgen`, `wasm-bindgen-futures`, `js-sys`, `web-sys`.

### `blazen-model-cache`

No `[features]`. Target-cfg deps split native (hf-hub + tokio/fs) vs wasi
(`async-trait`) — wasm-unknown-unknown gets no extras.

### `blazen-embed` (`crates/blazen-embed/Cargo.toml`)

**No `[features]`.** The whole crate is a facade chosen by target-cfg:

```toml
[target.'cfg(not(any(target_env = "musl", all(target_os = "macos", target_arch = "x86_64"))))'.dependencies]
blazen-embed-fastembed = { workspace = true }

[target.'cfg(any(target_env = "musl", all(target_os = "macos", target_arch = "x86_64")))'.dependencies]
blazen-embed-tract = { workspace = true }
```

`src/lib.rs:11/17` re-exports under matching cfg.

### `blazen-embed-fastembed`

No `[features]`. Hard deps: `fastembed`, `tokio`, `serde`, `serde_json`,
`tracing`.

### `blazen-embed-tract`

No `[features]`. Tract is gated by target-cfg, not features:
native pulls `tract-onnx` + `tokenizers["onig","progressbar"]` + model-cache;
browser-wasm pulls tract with `getrandom-js` + `tokenizers["unstable_wasm"]`
+ `web-sys`/`wasm-bindgen`; wasi pulls the same tract + tokenizers feature set
plus `reqwest`.

### `blazen-embed-candle` (`crates/blazen-embed-candle/Cargo.toml:790`)

| Feature | Default? | Enables |
|---|---|---|
| `engine` | no | `dep:candle-core`, `dep:candle-nn`, `dep:candle-transformers`, `dep:tokenizers`, `dep:hf-hub` |
| `cpu` | **yes** | `engine` |
| `cuda` | no | `engine` (does **not** forward to `candle-core/cuda` — see sharp edges) |
| `metal` | no | `engine` |
| `accelerate` | no | `engine` |

Source uses: `provider.rs:570` (`cfg(feature = "cuda")`), `provider.rs:595`
(`cfg(feature = "metal")`), `provider.rs:71/755/895` (`cfg(feature = "engine")`).

The Apple target override `Cargo.toml:26` unconditionally sets
`candle-core = { features = ["metal","accelerate"] }` on macOS/iOS — separate
from the feature flag mechanism.

### `blazen-llm-mistralrs` (`crates/blazen-llm-mistralrs/Cargo.toml:842`)

| Feature | Default? | Enables |
|---|---|---|
| `engine` | no | `dep:mistralrs`, `dep:image`, `dep:base64` |
| `cuda` | no | `engine` (documented marker; no `mistralrs/cuda` forwarding) |
| `metal` | no | `engine` (marker) |
| `accelerate` | no | `engine` (marker) |
| `mkl` | no | `engine` (marker) |
| `flash-attn` | no | `cuda` (marker) |

Comment block at `Cargo.toml:846–860` explicitly documents the marker design and
tells users to add `mistralrs` as a direct dep with their target features.

Source uses: only `cfg(feature = "engine")`. No GPU-marker source uses — markers
are pure no-ops here.

### `blazen-llm-candle` (`crates/blazen-llm-candle/Cargo.toml:898`)

| Feature | Default? | Enables |
|---|---|---|
| `engine` | no | `dep:candle-core`, `dep:candle-nn`, `dep:candle-transformers`, `dep:tokenizers`, `dep:hf-hub` |
| `cpu` | **yes** | `engine` |
| `cuda` | no | `engine` (does **not** forward to `candle-core/cuda`) |
| `metal` | no | `engine` |

Source uses: `provider.rs:62` (`cfg(feature = "cuda")`), `provider.rs:71`
(`cfg(feature = "metal")`). Same lie as `blazen-embed-candle`: enabling these
features compiles code paths that call `Device::new_cuda` / `Device::new_metal`,
but `candle-core` never receives the matching feature so those methods either
fail or won't link depending on candle internals.

### `blazen-llm-llamacpp` (`crates/blazen-llm-llamacpp/Cargo.toml:931`)

| Feature | Default? | Enables |
|---|---|---|
| `engine` | no | `dep:llama-cpp-2`, `dep:encoding_rs`, `dep:tokio-stream` |
| `cpu` | **yes** | `engine` |
| `cuda` | no | `engine` (marker) |
| `metal` | no | `engine` (marker) |
| `vulkan` | no | `engine` (marker) |
| `rocm` | no | `engine` (marker) |

Same marker rationale as mistralrs — documented in the `Cargo.toml` comment
block. No source uses of the GPU markers.

### `blazen-audio-whispercpp` (`crates/blazen-audio-whispercpp/Cargo.toml:971`)

| Feature | Default? | Enables |
|---|---|---|
| `engine` | no | `dep:whisper-rs` |
| `cuda` | no | `engine` (marker) |
| `metal` | no | `engine` (marker) |
| `coreml` | no | `engine` (marker) |

### `blazen-audio-piper` (`crates/blazen-audio-piper/Cargo.toml:1017`)

| Feature | Default? | Enables |
|---|---|---|
| `engine` | no | **nothing (no `dep:ort` yet — Phase 9 placeholder)** |

The crate compiles but `engine` is a no-op until the `ort` dep is added back.

### `blazen-image-diffusion` (`crates/blazen-image-diffusion/Cargo.toml:1049`)

| Feature | Default? | Enables |
|---|---|---|
| `engine` | no | **nothing (placeholder)** |
| `cuda` | no | `engine` (placeholder) |
| `metal` | no | `engine` (placeholder) |

Phase 5.3 placeholder; `diffusion-rs` dep is commented out.

### `blazen-cli`

No `[features]`. Skeleton bin crate with only `clap`.

### `blazen-peer` (`crates/blazen-peer/Cargo.toml:1127`)

| Feature | Default? | Enables |
|---|---|---|
| `server` | **yes** | (marker — gates tonic server source) |
| `client` | **yes** | (marker — gates tonic client source) |
| `http-transport` | no | `dep:blazen-llm` |

### `blazen-controlplane` (`crates/blazen-controlplane/Cargo.toml:1198`)

| Feature | Default? | Enables |
|---|---|---|
| `server` | **yes** | (marker) |
| `client` | **yes** | (marker) |
| `http-transport` | no | `dep:axum`, `dep:tower`, `dep:tower-http`, `dep:blazen-llm`, `dep:base64` |
| `valkey-store` | no | `dep:redis` |

### `blazen-manager` (`crates/blazen-manager/Cargo.toml:1236`)

| Feature | Default? | Enables |
|---|---|---|
| `live-models` | no | `dep:blazen-llm-mistralrs`, `dep:blazen-llm-candle`, `dep:blazen-llm-llamacpp`, `dep:blazen-model-cache`, plus their `engine` features and `blazen-llm/{mistralrs,candle-llm,llamacpp}` |

### `blazen-py` (`crates/blazen-py/Cargo.toml:1320`)

| Feature | Default? | Enables |
|---|---|---|
| `extension-module` | no | `pyo3/extension-module` |
| `embed` | no | `blazen-llm/embed` |
| `candle-embed` | no | `blazen-llm/candle-embed` |
| `mistralrs` | no | `blazen-llm/mistralrs` |
| `llamacpp` | no | `blazen-llm/llamacpp` |
| `candle-llm` | no | `blazen-llm/candle-llm` |
| `diffusion` | no | `blazen-llm/diffusion` |
| `whispercpp` | no | `blazen-llm/whispercpp` |
| `piper` | no | `blazen-llm/piper` |
| `tiktoken` | no | `blazen-llm/tiktoken` |
| `tract` | no | `dep:blazen-embed-tract` |
| `otlp` | no | `blazen-telemetry/otlp` |
| `prometheus` | no | `blazen-telemetry/prometheus` |
| `langfuse` | no | `blazen-telemetry/langfuse` |
| `distributed` | no | `blazen-core/distributed`, `blazen-peer/{server,client}`, `dep:blazen-controlplane`, `blazen-controlplane/{client,server}` |
| `local-all` | **yes** | all local backends + `tract` + `tiktoken` + `distributed` + telemetry exporters |

**Missing relative to blazen-node: `fastembed` feature.** blazen-embed-fastembed
is not declared.

### `blazen-node` (`crates/blazen-node/Cargo.toml:1402`)

| Feature | Default? | Enables |
|---|---|---|
| `tiktoken` | no | `blazen-llm/tiktoken` |
| `embed` | no | `blazen-llm/embed` |
| `fastembed` | no | `blazen-llm/embed`, `dep:blazen-embed-fastembed` |
| `tract` | no | `dep:blazen-embed-tract` |
| `candle-embed` | no | `blazen-llm/candle-embed` |
| `mistralrs` | no | `blazen-llm/mistralrs` |
| `llamacpp` | no | `blazen-llm/llamacpp` |
| `candle-llm` | no | `blazen-llm/candle-llm` |
| `diffusion` | no | `blazen-llm/diffusion` |
| `whispercpp` | no | `blazen-llm/whispercpp` |
| `piper` | no | `blazen-llm/piper` |
| `wasi` | no | `tiktoken` + `distributed-http` + `otlp-http` + `langfuse` (Cloudflare Workers / Deno profile) |
| `distributed` | no | gRPC peer (native only) |
| `distributed-http` | no | HTTP/JSON peer (wasi-friendly) |
| `otlp` | no | `blazen-telemetry/otlp` |
| `otlp-http` | no | `blazen-telemetry/otlp-http` |
| `prometheus` | no | `blazen-telemetry/prometheus` |
| `langfuse` | no | `blazen-telemetry/langfuse` |
| `local-all` | no | everything except `wasi` |

Default = `[]`. Note `wasi` is a **profile feature** (composite), not a target
gate — it just bundles the wasi-compatible exporters.

`blazen-embed-fastembed` is target-gated optional (`cfg(not(target_env = "musl"))`).

### `blazen-uniffi` (`crates/blazen-uniffi/Cargo.toml:1522`)

| Feature | Default? | Enables |
|---|---|---|
| `embed` / `candle-embed` / `mistralrs` / `llamacpp` / `candle-llm` / `diffusion` / `whispercpp` / `piper` / `tiktoken` | no | forwards to `blazen-llm/*` |
| `tract` | no | `dep:blazen-embed-tract` |
| `otlp` / `prometheus` / `langfuse` | no | forwards to `blazen-telemetry/*` |
| `distributed` | no | `blazen-core/distributed`, `blazen-peer/{server,client}`, `dep:blazen-controlplane` |
| `local-all` | **yes** | everything above |

Header comment at line 1523 says "Match blazen-py's feature graph one-to-one"
— but blazen-py does *not* expose `fastembed` either, so the divergence with
blazen-node is preserved.

### `blazen-cabi` (`crates/blazen-cabi/Cargo.toml:1618`)

| Feature | Default? | Enables |
|---|---|---|
| All feature names | no | each forwards to `blazen-uniffi/<same-name>` |
| `local-all` | **yes** | all of the above |

Pure pass-through to `blazen-uniffi`.

### `blazen-wasm` (excluded; `crates/blazen-wasm/Cargo.toml:1665`)

No `[features]`. Always depends on `blazen-llm` with `default-features = false`.

### `blazen-wasm-sdk` (excluded; `crates/blazen-wasm-sdk/Cargo.toml:1702`)

| Feature | Default? | Enables |
|---|---|---|
| `tiktoken` | no | `blazen-llm/tiktoken` |

Direct deps already pin `blazen-embed-tract`, `blazen-telemetry["otlp-http","langfuse"]`,
`blazen-peer["client","http-transport"]`. No `local-all` / no LLM-backend features
(the WASM SDK calls remote providers, not native LLM crates).

---

## Section 2 — Sharp edges

1. **`blazen-embed` facade lets aarch64-linux-gnu pull fastembed → ORT, which
   has no prebuilt binary for that triple.** `crates/blazen-embed/Cargo.toml:14`
   only excludes `target_env = "musl"` and `x86_64-apple-darwin`. AWS Graviton
   / Ampere with stock glibc would try to download an `aarch64-linux-gnu` ORT
   archive that `ort-sys-2.0.0-rc.10/build.rs` does not list as a supported
   prebuilt target (see lines 248–255 — only `aarch64-pc-windows-msvc`,
   `aarch64-apple-darwin`, `aarch64-linux-android` are mapped). The
   `blazen-embed-tract` README at line 11 already lists this case as
   tract-supported but the facade's cfg doesn't match. **Confirmed gap.**

2. **`blazen-embed-candle.cuda` + `.metal` + `.accelerate` features are
   half-real.** They activate `#[cfg(feature = "cuda")]` code paths in
   `crates/blazen-embed-candle/src/provider.rs:570/595` but do **not** forward
   to `candle-core/cuda` / `candle-core/metal` / `candle-core/accelerate`
   (`Cargo.toml:32–34`). The Apple-target override at `Cargo.toml:26` only
   helps on macOS/iOS. Enabling `--features cuda` on Linux compiles
   `Device::new_cuda` but candle-core lacks its CUDA backend → link/runtime
   failure depending on candle version. Same issue in `blazen-llm-candle`
   (`Cargo.toml:32–33` + `src/provider.rs:62/71`).

3. **`blazen-llm-mistralrs.{cuda,metal,accelerate,mkl,flash-attn}` are pure
   marker features** with no `dep:` activator and no source uses. Documented at
   `Cargo.toml:846–860` — they exist so downstream `--all-features` checks
   don't break and so users know what to enable in their own binary. They mean
   exactly nothing inside this crate.

4. **`blazen-llm-llamacpp.{cuda,metal,vulkan,rocm}` are the same marker
   pattern** (`Cargo.toml:933–945`). No source uses, no forwarding.

5. **`blazen-audio-whispercpp.{cuda,metal,coreml}` likewise** (`Cargo.toml:975–993`).

6. **`blazen-image-diffusion.{engine,cuda,metal}` are completely empty
   placeholders** (`Cargo.toml:1049`). The `diffusion-rs` dep is commented out.
   Activating any of them does nothing at all — not even a marker, since the
   underlying crate isn't there to forward to.

7. **`blazen-audio-piper.engine` is an empty placeholder** (`Cargo.toml:1017–1024`).
   Same story — `ort` dep is commented out pending Phase 9.

8. **`blazen-llm.candle-audio` is a feature with no `dep:` activator and no
   matching crate** (`Cargo.toml:214`). It exists so `blazen` can list it in
   `local-audio` (`blazen/Cargo.toml:25`), but flipping it does nothing.

9. **`blazen.llm` is a misleading default feature.** `blazen-core` hard-depends
   on `blazen-llm` (`blazen-core/Cargo.toml:20`, used at
   `blazen-core/src/workflow.rs:49`, `step.rs:13/72/137/214/278/440`,
   `context.rs:18`, `handler.rs:23`, `event_loop.rs:1716` and elsewhere). The
   `llm` feature on the umbrella crate just toggles the re-export, not the dep
   tree — `cargo build -p blazen --no-default-features` still pulls blazen-llm.

10. **`blazen-py` is missing `fastembed`.** blazen-node has both `tract` and
    `fastembed` features; blazen-py only has `tract`. This is the inconsistency
    flagged in the user prompt. blazen-py's `local-all` therefore can't activate
    fastembed even when the host could build it.

11. **`blazen-uniffi` / `blazen-cabi` also lack `fastembed`** — they follow
    blazen-py rather than blazen-node. UniFFI bindings are blocked from
    fastembed entirely.

12. **`local-all` is defined in 4 places with diverging contents:**
    - `blazen/Cargo.toml:26` — wraps `blazen-llm/*` features only, no telemetry, no distributed.
    - `blazen-py/Cargo.toml:1335` — backends + `tract` + `tiktoken` + `distributed` + `otlp` + `prometheus` + `langfuse`. Misses `fastembed`.
    - `blazen-node/Cargo.toml:1429` — backends + `fastembed` + `tract` + `tiktoken` + `distributed` (gRPC, not `distributed-http`) + `otlp` (not `otlp-http`) + `prometheus` + `langfuse`.
    - `blazen-uniffi/Cargo.toml:1539` — backends + `tract` + `tiktoken` + `distributed` + telemetry. Misses `fastembed`.
    - `blazen-cabi/Cargo.toml:1634` — forwards to `blazen-uniffi`'s set.

    Effect: the same flag name means three slightly different things across the
    binding surfaces.

13. **`blazen-llm.content-detect` is on by default but adds a non-trivial
    optional dep (`infer`)** — fine for most callers, hostile for footprint-
    sensitive embeds. Combined with the default-features asymmetry below, every
    binding currently pulls `infer` whether they want it or not.

14. **Defaults are inconsistent across crates:**
    - `blazen` defaults `["llm"]` (re-export only)
    - `blazen-llm` defaults `["content-detect"]` (pulls `infer`)
    - `blazen-memory` defaults `["jsonl"]` (functional marker)
    - `blazen-memory-valkey` defaults `["redis-tcp"]` (pulls `redis` + tokio rt)
    - `blazen-persist` defaults `["redb"]`
    - `blazen-telemetry` defaults `["spans"]` (pulls `blazen-llm`)
    - `blazen-embed-candle` defaults `["cpu"]` → `engine` → candle stack
    - `blazen-llm-candle` defaults `["cpu"]` → `engine` → candle stack
    - `blazen-llm-llamacpp` defaults `["cpu"]` → `engine` → llama-cpp-2 C stack
    - `blazen-llm-mistralrs` defaults `[]` (no engine by default)
    - `blazen-audio-whispercpp` defaults `[]`
    - `blazen-audio-piper` defaults `[]`
    - `blazen-image-diffusion` defaults `[]`
    - `blazen-peer` / `blazen-controlplane` default `["server","client"]`
    - `blazen-py` defaults `["local-all"]` (heavy)
    - `blazen-node` defaults `[]` (slim)
    - `blazen-uniffi` defaults `["local-all"]` (heavy)
    - `blazen-cabi` defaults `["local-all"]` (heavy)

    The same family of crates (the local-inference backends) split: candle
    crates auto-enable engine, mistralrs/llamacpp/whispercpp do not. The same
    family of bindings (py/node/uniffi/cabi) split: 3 of 4 default heavy, 1
    defaults slim.

15. **`blazen-telemetry.spans` default pulls `blazen-llm`.** That's a 5-MB-ish
    transitive haul that anyone using telemetry-without-LLM eats by default
    (`Cargo.toml:559–560`).

16. **`--all-features` cannot pass everywhere unaided.** Workspace
    `--all-features` on aarch64-linux-gnu would fail on fastembed (see point
    1); on a machine without CUDA toolkit it would attempt to compile candle
    `cuda` code paths (point 2). Today the marker design papers over this for
    mistralrs/llamacpp/whispercpp, but the candle crates lie about it.

17. **`blazen-memory.jsonl` is a default marker feature with no dep activator
    and a single source-cfg use.** Effectively `default-on` boolean dead
    weight: removing it would require either always-on JSONL or moving it to
    `default = []`. Today it's `default = ["jsonl"]` and nobody disables it.

18. **`blazen-peer.{server,client}` and `blazen-controlplane.{server,client}`
    are marker features that act like "always-on knobs"** — every binding pulls
    both. Worth considering whether they justify their cost (no clear
    dep-tree win from disabling one or the other).

19. **`blazen-llm.tsify` overlaps with `blazen-events.tsify`** — both gate
    `tsify-next`+`wasm-bindgen` independently. Naming is consistent but two
    crates have to be flipped in lockstep for any reasonable TS surface.

20. **`blazen-llm.pyo3-serde` is the only Python-specific feature in the LLM
    crate.** It currently lives there because `blazen-py` activates it
    (`blazen-py/Cargo.toml:1291` — `features = ["reqwest","pyo3-serde"]`),
    but it pulls `pyo3` directly. This bleeds Python-host concerns into the
    provider abstraction layer.

21. **`blazen-controlplane`'s server feature is the cheapest gate but the
    `http-transport` feature pulls axum/tower/tower-http**, all of which always
    compile if `default = ["server","client"]` already gave you tonic. Worth
    asking whether `http-transport` should imply `server`/`client` off or be a
    third transport choice.

22. **`blazen-llm-mistralrs.flash-attn = ["cuda"]`** is the only marker that
    chains to another marker. Since `cuda` itself is a no-op, `flash-attn` is
    a no-op chain. Harmless but odd.

23. **`blazen-llm` build.rs codegen runs regardless of features**
    (`crates/blazen-llm/build.rs:1`). Not a feature-graph bug, but worth
    confirming the redesign doesn't break the pricing-json snapshot path.

24. **`blazen.local-audio` references `blazen-llm/candle-audio`** which is
    itself a phantom (sharp edge #8). Cascading no-ops.

---

## Section 3 — Proposed redesign

### 3.1 ONNX backend selection

**Decision: keep the current target-cfg auto-select, fix the target predicate,
add explicit override features.**

Mechanism: stay with `[target.'cfg(...)'.dependencies]` blocks in
`blazen-embed/Cargo.toml`. We considered three alternatives:

- **`build.rs` writing a cfg via `println!("cargo:rustc-cfg=...")`.** Loses
  predictability (cfg values are invisible in `Cargo.toml`) and requires a
  build script in a facade crate that today has zero code.
- **`cfg_aliases` crate.** Adds a build-dep + still requires per-target
  dep blocks for the actual `optional = true` switch; the alias only helps
  with source `#[cfg]` blocks. The facade has only two `pub use` lines, so
  the alias buys nothing measurable.
- **Status quo (target-cfg deps).** Already in use elsewhere in the workspace
  (`blazen-core/Cargo.toml:87–113`, `blazen-llm/Cargo.toml:249–277`,
  `blazen-model-cache/Cargo.toml:627–640`). Consistent and zero new build deps.

Concretely:

```toml
# crates/blazen-embed/Cargo.toml after redesign

[features]
default = []
# Explicit overrides — when set, win over auto-select.
embed-fastembed = ["dep:blazen-embed-fastembed"]
embed-tract     = ["dep:blazen-embed-tract"]

[dependencies]
blazen-embed-fastembed = { workspace = true, optional = true }
blazen-embed-tract     = { workspace = true, optional = true }

# Auto-select: fastembed only on triples ORT actually ships prebuilts for
# (x86_64 Linux glibc, x86_64/aarch64 macOS, x86_64/aarch64 Windows MSVC,
#  iOS, Android). Tract elsewhere (musl-Linux, aarch64-linux-gnu, wasm*).
[target.'cfg(any(
    all(target_arch = "x86_64", target_os = "linux", target_env = "gnu"),
    all(target_arch = "aarch64", target_os = "macos"),
    all(target_arch = "x86_64",  target_os = "macos"),
    all(target_arch = "x86_64",  target_os = "windows", target_env = "msvc"),
    all(target_arch = "aarch64", target_os = "windows", target_env = "msvc"),
    target_os = "ios",
    target_os = "android"
))'.dependencies]
blazen-embed-fastembed = { workspace = true }   # non-optional on these

[target.'cfg(not(any(
    all(target_arch = "x86_64", target_os = "linux", target_env = "gnu"),
    all(target_arch = "aarch64", target_os = "macos"),
    all(target_arch = "x86_64",  target_os = "macos"),
    all(target_arch = "x86_64",  target_os = "windows", target_env = "msvc"),
    all(target_arch = "aarch64", target_os = "windows", target_env = "msvc"),
    target_os = "ios",
    target_os = "android"
)))'.dependencies]
blazen-embed-tract = { workspace = true }       # non-optional on these
```

`src/lib.rs` becomes:

```rust
#[cfg(any(feature = "embed-fastembed",
          all(not(feature = "embed-tract"),
              any(all(target_arch="x86_64", target_os="linux", target_env="gnu"),
                  all(target_arch="aarch64", target_os="macos"),
                  all(target_arch="x86_64",  target_os="macos"),
                  all(target_arch="x86_64",  target_os="windows", target_env="msvc"),
                  all(target_arch="aarch64", target_os="windows", target_env="msvc"),
                  target_os="ios",
                  target_os="android"))))]
pub use blazen_embed_fastembed::{ ... };

#[cfg(any(feature = "embed-tract",
          all(not(feature = "embed-fastembed"), not(<the same predicate>))))]
pub use blazen_embed_tract::{ ... };
```

This keeps the facade pattern, narrows the fastembed allow-list to triples ORT
actually ships, fixes aarch64-linux-gnu, and gives users two override features
when they want to force one backend (e.g. on a sandboxed host without ORT
binaries, or for benchmarking).

Recommended supplement: a `#[cfg_attr(docsrs, ...)]` block in `src/lib.rs`
documenting which target maps to which backend, so docs.rs shows it.

### 3.2 Default features per crate

**Decision: unify around "cheapest functional default that does not pull a C
toolchain by surprise." For binding crates, default to `[]`.**

Concrete proposal:

| Crate | Today | Proposed | Rationale |
|---|---|---|---|
| `blazen` | `["llm"]` | `[]` | `llm` is a no-op (sharp edge #9); the re-export should be unconditional. |
| `blazen-llm` | `["content-detect"]` | `["content-detect"]` | Keep. `infer` is small; matches existing CI behavior. |
| `blazen-memory` | `["jsonl"]` | `["jsonl"]` | Keep, but add a comment explaining the feature exists for `no_std`-like embeds even though it's default-on. |
| `blazen-memory-valkey` | `["redis-tcp"]` | `[]` | Native binding; `redis-tcp` pulls `redis` + tokio rt unconditionally. Bindings already opt in explicitly. |
| `blazen-persist` | `["redb"]` | `["redb"]` | Keep — most callers want it. |
| `blazen-telemetry` | `["spans"]` | `[]` | `spans` pulls `blazen-llm`; make it explicit. The 3 binding crates already pass `["history","spans"]`. |
| `blazen-embed-candle` | `["cpu"]` (→ engine) | `[]` | Match mistralrs/llamacpp/whisper — engine is opt-in. |
| `blazen-llm-candle` | `["cpu"]` (→ engine) | `[]` | Same. |
| `blazen-llm-llamacpp` | `["cpu"]` (→ engine) | `[]` | Same. C++ stack should be opt-in. |
| `blazen-llm-mistralrs` | `[]` | `[]` | Already correct. |
| `blazen-audio-whispercpp` | `[]` | `[]` | Already correct. |
| `blazen-audio-piper` | `[]` | `[]` | Already correct. |
| `blazen-image-diffusion` | `[]` | `[]` | Already correct. |
| `blazen-peer` | `["server","client"]` | `["client"]` | Most callers want client; server is heavier (gRPC reflection / TLS setup). |
| `blazen-controlplane` | `["server","client"]` | `["client"]` | Same. |
| `blazen-py` | `["local-all"]` | `[]` | Bindings should be lean by default. Heavy build is opt-in via `--features local-all`. |
| `blazen-node` | `[]` | `[]` | Already correct. |
| `blazen-uniffi` | `["local-all"]` | `[]` | Same as blazen-py. |
| `blazen-cabi` | `["local-all"]` | `[]` | Same. |

The Python wheel CI job (and `uv run --no-sync maturin develop --features
langfuse,otlp,prometheus,tract,distributed`) already passes explicit features —
flipping `default = []` for the bindings is a no-op for the canonical build
commands documented in `CLAUDE.md`. Same for the UniFFI/cabi releases (the
Forgejo `release.yaml` passes `--features` flags).

This proposal **does** break callers who consume `blazen-py` / `blazen-uniffi`
/ `blazen-cabi` via `crates.io`-style coordinates and rely on the implicit
`local-all`. Since these crates are bindings (not Rust libs), there are
effectively no such callers — they're consumed via the Python wheel / Go
module / Ruby gem, not via Cargo. Acceptable break.

### 3.3 GPU marker features

**Decision: keep the marker pattern for the four backends that genuinely
cannot forward (mistralrs, llamacpp, whisper, embed-candle's `accelerate`),
but fix the half-real cases in `blazen-llm-candle` and `blazen-embed-candle`
so they're either fully marker or fully forwarding.**

Two-step:

1. **For `blazen-llm-candle.{cuda,metal}` and `blazen-embed-candle.{cuda,metal,accelerate}`:**
   either (a) remove the `cfg(feature = "cuda")` / `cfg(feature = "metal")`
   source gates in their `provider.rs` and make `Device::new_cuda`/`new_metal`
   calls compile-out via `match` against the device string at runtime (relying
   on candle's own internal cfg), or (b) drop the marker features from
   `Cargo.toml` and require callers to add candle-core as a direct dep.

   Recommendation: option (a). Source uses become unconditional code paths;
   candle-core's own feature flags decide whether `Device::new_cuda`
   compiles to a stub-error or a real call. This makes the candle backends
   match how candle is meant to be used.

2. **Standardize the marker convention.** For mistralrs/llamacpp/whisper, keep
   the `cuda = ["engine"]` style and lift the comment block from
   `blazen-llm-mistralrs/Cargo.toml:846–860` into a workspace-level doc snippet
   referenced from all three crates. Add a single `## GPU acceleration` H2 in
   the crate's README with the same copy.

3. **Remove `blazen-llm-mistralrs.flash-attn`** until it actually forwards. It's
   a marker on a marker (sharp edge #22). If users want flash-attn, they enable
   `mistralrs/flash-attn` in their binary. The blazen crate doesn't need to
   pretend.

4. **Remove `blazen-image-diffusion.{engine,cuda,metal}` and
   `blazen-audio-piper.engine`** until the underlying deps are uncommented.
   Phantom features are worse than missing features — they imply functionality
   that doesn't exist. Re-add them with `dep:` activators when Phase 5.3 / 9
   land.

5. **Remove `blazen-llm.candle-audio`** (sharp edge #8). No backing crate.

### 3.4 `local-all` umbrellas

**Decision: keep per-binding `local-all`, but make them mechanically identical
via a doc invariant + CI lint.**

Workspace-level umbrellas in Cargo are not really a thing — features live on
crates. The three bindings (`blazen-py`, `blazen-node`, `blazen-uniffi`,
plus the pass-through `blazen-cabi`) need their own `local-all` because each
forwards through different intermediate dependencies (e.g. blazen-node has
both `embed` and `fastembed` because it can split native vs wasi).

Proposal:

- **Standardize the canonical local-all definition** (kept in sync via the
  `audit-bindings` CI job — extend that job to grep for the per-crate
  `local-all = [...]` blocks and fail on mismatch):

  ```
  local-all = [
      "embed",
      "fastembed",      # NEW: add to py/uniffi/cabi (sharp edge #10, #11)
      "tract",
      "candle-embed",
      "mistralrs",
      "llamacpp",
      "candle-llm",
      "diffusion",
      "whispercpp",
      "piper",
      "tiktoken",
      "distributed",
      "otlp",
      "prometheus",
      "langfuse",
  ]
  ```

- **The umbrella `blazen` crate keeps its modality-grouped umbrellas**
  (`local-embeddings`, `local-llm`, `local-image`, `local-audio`, `local-all`)
  because those map to source-level groupings, not binding profiles.

- **blazen-cabi keeps pass-through** — its `local-all` forwards each name to
  `blazen-uniffi/<name>`. No definition divergence possible.

- **blazen-node keeps a separate `wasi` profile** (Cloudflare Workers / Deno).
  That stays as-is — wasi cannot include `local-all`'s native deps.

### 3.5 Per-binding feature parity

**Decision: `blazen-py`, `blazen-node`, `blazen-uniffi`, `blazen-cabi` expose
exactly the same set of feature names, with one platform-specific exception
in blazen-node.**

Uniform target after redesign:

| Feature | py | node | uniffi | cabi |
|---|---|---|---|---|
| `embed` | yes | yes | yes | yes |
| `fastembed` | **add** | yes | **add** | **add** |
| `tract` | yes | yes | yes | yes |
| `candle-embed` | yes | yes | yes | yes |
| `mistralrs` | yes | yes | yes | yes |
| `llamacpp` | yes | yes | yes | yes |
| `candle-llm` | yes | yes | yes | yes |
| `diffusion` | yes | yes | yes | yes |
| `whispercpp` | yes | yes | yes | yes |
| `piper` | yes | yes | yes | yes |
| `tiktoken` | yes | yes | yes | yes |
| `otlp` | yes | yes | yes | yes |
| `otlp-http` | — | yes | — | — |
| `prometheus` | yes | yes | yes | yes |
| `langfuse` | yes | yes | yes | yes |
| `distributed` | yes | yes | yes | yes |
| `distributed-http` | — | yes | — | — |
| `wasi` | — | yes | — | — |
| `local-all` | yes | yes | yes | yes |
| `extension-module` | yes (pyo3) | — | — | — |

Node-only `otlp-http`, `distributed-http`, `wasi`: justified because only
blazen-node ships a wasi target (the Cloudflare Workers runtime). py/uniffi/
cabi don't.

py-only `extension-module`: justified — it's a pyo3 mechanism.

This means py, uniffi, and cabi each gain a `fastembed` feature. The activator
must be target-cfg gated to skip on musl / aarch64-windows etc. — handled
automatically because `blazen-embed-fastembed` will refuse to build there
anyway and the feature flag merely opts into the dep.

### 3.6 WASM gating

**Decision: confirmed — `target_arch = "wasm32"` stays as the WASM gate.**

The redesign does not introduce a `wasm` Cargo feature. The current pattern
(target-cfg dep blocks + source `#[cfg(target_arch = "wasm32")]`) is the only
mechanism that scales across native / browser-wasm / wasi without explosion.

Touched cfg patterns:
- `cfg(not(target_arch = "wasm32"))` — kept everywhere it currently appears
  (blazen-llm/cache.rs:37, agent.rs:9, providers/*.rs).
- `cfg(all(target_arch = "wasm32", not(target_os = "wasi")))` — kept for
  browser-only deps (wasm-bindgen, web-sys, gloo-timers).
- `cfg(all(target_arch = "wasm32", target_os = "wasi"))` — kept for wasi
  (futures-executor, tokio[rt,time] on wasi).

The only new wrinkle: the embed facade's target predicate widens (Section 3.1).
That predicate uses arch + os + env tuples consistently with the rest of the
workspace.

---

## Section 4 — Per-crate diff plan

### `blazen`
- Remove: `default = ["llm"]` (sharp edge #9). Set `default = []`.
- Keep: feature names `llm`/`persist`/`pipeline`/`prompts`/`telemetry`/`all`.
- Keep: `local-{embeddings,llm,image,audio,all}` umbrellas.
- Remove: `blazen-llm/candle-audio` from `local-audio` (sharp edge #8 — feature is going away).

### `blazen-core`
- No feature changes. Already minimal.
- Optional: document why `blazen-llm` is a hard dep, not behind `persist` or `telemetry`.

### `blazen-events`
- No changes.

### `blazen-macros`
- No changes.

### `blazen-llm`
- Remove: `candle-audio` (phantom).
- Consider moving `pyo3-serde` to a separate crate later (out of scope here).
- Default unchanged: `["content-detect"]`.

### `blazen-memory`
- No changes (default `["jsonl"]` stays — marker feature in widespread use).

### `blazen-memory-valkey`
- Default: `["redis-tcp"]` → `[]`. Callers already pass explicit features.

### `blazen-persist`
- No changes.

### `blazen-pipeline` / `blazen-prompts`
- No changes.

### `blazen-telemetry`
- Default: `["spans"]` → `[]`. Three bindings already pass `["history","spans"]` explicitly.
- Keep all other feature names as-is.

### `blazen-model-cache`
- No changes.

### `blazen-embed`
- Add features: `embed-fastembed`, `embed-tract` (override knobs).
- Replace target predicate (Section 3.1).
- Optional deps on both backends.

### `blazen-embed-fastembed`
- No changes (no `[features]` today; stays that way).

### `blazen-embed-tract`
- No changes.

### `blazen-embed-candle`
- Remove source uses of `#[cfg(feature = "cuda")]` / `#[cfg(feature = "metal")]` (sharp edge #2). Replace with unconditional `match` on device string; rely on candle-core's own cfg.
- Drop `cuda` / `metal` / `accelerate` from `[features]` OR convert them to forwarders (`cuda = ["engine", "candle-core/cuda"]`) — decision depends on whether we want them at all.
- Recommended: drop the marker features; document that GPU is selected by adding `candle-core` as a direct dep with the desired features.
- Default: `["cpu"]` → `[]`.

### `blazen-llm-candle`
- Same as `blazen-embed-candle`: remove half-real `cuda`/`metal` source gates; drop the marker features OR convert to forwarders.
- Default: `["cpu"]` → `[]`.

### `blazen-llm-llamacpp`
- Keep: marker features `cuda`/`metal`/`vulkan`/`rocm` (documented pattern).
- Default: `["cpu"]` → `[]`. Move CPU to opt-in for consistency.

### `blazen-llm-mistralrs`
- Remove: `flash-attn = ["cuda"]` (marker on marker).
- Keep: other markers + the existing doc comment block.
- No default change.

### `blazen-audio-whispercpp`
- No changes — marker pattern stays.

### `blazen-audio-piper`
- Remove: `engine` feature (phantom — no underlying dep).
- Re-add when Phase 9 lands `ort` dep.

### `blazen-image-diffusion`
- Remove: `engine`, `cuda`, `metal` (phantoms).
- Re-add when Phase 5.3 lands `diffusion-rs` dep.

### `blazen-cli`
- No changes.

### `blazen-peer`
- Default: `["server","client"]` → `["client"]`.

### `blazen-controlplane`
- Default: `["server","client"]` → `["client"]`.

### `blazen-manager`
- No changes (its `live-models` feature stays exactly as-is — it's the eval harness gate).

### `blazen-py`
- Add: `fastembed = ["blazen-llm/embed", "dep:blazen-embed-fastembed"]` (target-cfg gate the optional dep).
- Add `fastembed` to `local-all`.
- Default: `["local-all"]` → `[]`.

### `blazen-node`
- Add `fastembed` to `local-all` (it's already a feature name but absent from `local-all`'s list — wait, it IS in `local-all` at line 1429. Verified — no change).
- No default change.

### `blazen-uniffi`
- Add: `fastembed` feature + target-cfg-gated optional dep.
- Add `fastembed` to `local-all`.
- Default: `["local-all"]` → `[]`.

### `blazen-cabi`
- Add: `fastembed = ["blazen-uniffi/fastembed"]`.
- Add `fastembed` to `local-all`.
- Default: `["local-all"]` → `[]`.

### `blazen-wasm` (excluded)
- No changes.

### `blazen-wasm-sdk` (excluded)
- No changes.

---

## Section 5 — Migration strategy

### Breaking changes

| Change | Affected | Semver |
|---|---|---|
| `blazen-py` / `blazen-uniffi` / `blazen-cabi` default `[]` instead of `["local-all"]` | Anyone using these as Rust deps (effectively zero — they're binding crates consumed via wheel/gem/module). | minor for bindings, no-op for end users |
| `blazen-telemetry` default `[]` instead of `["spans"]` | Anyone depending on `blazen-telemetry` without specifying features. CI/scripts that do `cargo build -p blazen-telemetry` get a smaller build. | minor |
| `blazen-memory-valkey` default `[]` instead of `["redis-tcp"]` | Anyone using `blazen-memory-valkey` as a direct dep without features. All in-tree callers already pass explicit features. | minor |
| `blazen-peer` / `blazen-controlplane` default `["client"]` instead of `["server","client"]` | Same as above. | minor |
| `blazen-embed-candle` / `blazen-llm-candle` / `blazen-llm-llamacpp` default `[]` instead of `["cpu"]` | Callers relying on default engine activation get nothing back. blazen-llm's `mistralrs`/`candle-llm`/`llamacpp` features already forward `engine` explicitly. | minor |
| `blazen-llm-mistralrs.flash-attn` removed | None in practice (marker). | patch |
| `blazen-image-diffusion.{engine,cuda,metal}` removed | None (phantom). Re-add with real deps later. | patch |
| `blazen-audio-piper.engine` removed | None (phantom). | patch |
| `blazen-llm.candle-audio` removed | `blazen.local-audio` updated to drop it. | patch |
| `blazen-embed` adds `embed-fastembed` / `embed-tract` features | Additive. | patch |
| Target predicate in `blazen-embed` widens to also reject aarch64-linux-gnu from fastembed | aarch64-linux-gnu callers who were getting fastembed (and crashing at build time) now get tract (and a working build). | bug fix |

### Staging plan

Single PR is feasible because the changes are mostly Cargo.toml edits + a few
source-cfg deletions in candle providers. The PR sequence is:

1. **PR1.5a** (this redesign): land the doc + the embed facade target-predicate
   fix (sharp edge #1) + remove the four phantom features. Doc-only +
   conservative deletions. Low risk.

2. **PR1.5b**: default-features unification. Touches every binding crate's
   Cargo.toml. Higher-risk because CI workflows that omit `--features` will
   see different artifacts. Audit `release.yaml`, `live-models.yaml`, and the
   eight test surfaces from `CLAUDE.md` before/after.

3. **PR1.5c**: per-binding feature parity (add `fastembed` to
   py/uniffi/cabi). Additive only.

4. **PR1.5d**: candle GPU-marker cleanup (sharp edge #2). Touches two
   `provider.rs` files. Pure source cleanup, no API change.

5. **PR1.5e**: extend `audit-bindings` CI to lint the four `local-all`
   blocks for mismatch. Mechanism: a small Python/Bash script in `scripts/`
   that parses `local-all = [...]` and diffs the four definitions.

### `local-all` reconciliation during transition

Between PR1.5b and PR1.5c, `blazen-py.local-all` is missing `fastembed`
relative to `blazen-node.local-all`. This is the **current** state — there's
no regression window. PR1.5c closes the gap.

### ai_brain (sibling project) impact

ai_brain consumes Blazen via the Python wheel. Wheel-side feature flags are
chosen in `release.yaml` / the `pyproject.toml`. As long as the release job
keeps passing `--features langfuse,otlp,prometheus,tract,distributed`
(documented in `CLAUDE.md`'s test commands), ai_brain sees zero change after
PR1.5b. Add `fastembed` to that feature list after PR1.5c lands so the wheel
ships fastembed on x86_64 hosts.

---

## Section 6 — Validation plan

### `live-models` tests

Confirm the eval harness still works under the new feature graph:

```bash
./scripts/build-uniffi-lib.sh linux_amd64   # only if uniffi changes
cargo nextest run -p blazen-manager --features live-models --run-ignored only --test-threads 1
```

`blazen-manager.live-models` is unchanged (Section 4), so this is identity.
Verify on:
- A host with `~/.cache/blazen-tests/loras/test-lora/` pre-staged (Forgejo
  `live-models` job already does this).
- A host *without* the cache: the tests should skip cleanly, not fail.

### `cargo build --workspace --all-features` matrix

After redesign, run on three hosts/targets:

1. **x86_64-unknown-linux-gnu** (host):
   ```bash
   cargo build --workspace --all-features
   cargo test --workspace --all-features --no-run
   cargo nextest run --workspace --all-features
   cargo test --workspace --doc --all-features
   ```
   Expected to pass identically to today, modulo the dropped phantom features.

2. **aarch64-unknown-linux-gnu** (cross):
   ```bash
   rustup target add aarch64-unknown-linux-gnu
   cargo check --target aarch64-unknown-linux-gnu --workspace --all-features
   ```
   **Today this fails on `blazen-embed-fastembed` → ORT.** After the embed
   facade fix (Section 3.1) it should succeed because the facade now picks
   tract on aarch64-gnu and `blazen-embed-fastembed` is no longer a hard
   target-dep on this triple. The `--all-features` `embed-fastembed`
   override flag will still try to pull fastembed (and fail), but `--all-features`
   on a non-supported triple is the user's expressed intent — document that
   `embed-fastembed` requires a supported target.

   If the design goal is "`--all-features` passes on every target," then the
   `embed-fastembed` feature itself must be target-cfg-gated, which Cargo
   doesn't natively support. The accepted compromise: `--all-features` is
   not part of the supported matrix for non-ORT targets; the CI matrix runs
   `--all-features` only on x86_64-linux-gnu and macOS.

3. **wasm32-unknown-unknown** (browser SDK):
   ```bash
   cargo check --target wasm32-unknown-unknown -p blazen-wasm-sdk
   wasm-pack build crates/blazen-wasm-sdk --target web --out-dir pkg --release
   wasm-pack test crates/blazen-wasm-sdk --headless --firefox
   ```
   Identical commands to today. The wasm-sdk pins `blazen-embed-tract`
   directly, so the facade change is transparent.

4. **wasm32-wasip1** (Cloudflare Workers):
   ```bash
   pnpm --filter blazen run build   # builds the wasi artifact via napi-rs
   ```
   blazen-node's `wasi` feature unchanged; should pass.

### Typegen regeneration

Each binding's typegen must still regenerate clean:

```bash
cargo run --example stub_gen -p blazen-py --features langfuse,otlp,prometheus,tract,distributed,fastembed
pnpm --filter blazen run build
wasm-pack build crates/blazen-wasm-sdk --target web --out-dir pkg --release
cargo build -p blazen-uniffi --release && ./scripts/regen-bindings.sh
cargo build -p blazen-cabi --release   # cbindgen via build.rs
```

`blazen.pyi` will gain `fastembed`-feature-gated symbols (PR1.5c). Update the
CI command in `.forgejo/workflows/ci.yaml:259` and `:720` to include
`fastembed` in the feature list once the feature exists.

### Eight test surfaces from CLAUDE.md

All eight surfaces must still pass after the redesign:

1. Rust workspace (`cargo nextest run --workspace --all-features`) — covered above.
2. Python (`uv run --no-sync pytest tests/python/`) — covered (wheel rebuild needed).
3. Node (`pnpm exec ava --timeout 30s`) — covered (typegen + build).
4. WASM SDK (`wasm-pack test`) — covered.
5. Go (`go test ./...` in `bindings/go`) — covered if `./scripts/build-uniffi-lib.sh` succeeds.
6. Swift (`swift test` in `bindings/swift`) — covered.
7. Kotlin (`gradle test` in `bindings/kotlin`) — covered.
8. Ruby (`bundle exec rspec` in `bindings/ruby`) — covered (cabi rebuild via `./scripts/build-uniffi-lib.sh`).

`audit-bindings` (Forgejo CI) fails on typegen drift today. After PR1.5c
extend it to also fail on `local-all` mismatch across the four bindings (a
~30-line shell/python script in `scripts/audit-local-all.sh`).

---

## Notes for the implementation PR author

- The marker-feature pattern is a workaround for `cargo --all-features`
  having no `[target.cfg(...).features]` block. That's a known Cargo
  limitation (rust-lang/cargo#1197 and friends). Don't waste time looking
  for a way around it — accept that `--all-features` is a "supported host"
  contract, not a "every triple" contract.
- The embed facade is the only place that needs target-cfg dep gymnastics
  for arch-specific backend selection. Keep that pattern out of the LLM
  crates; the LLM backends are user-selected via features.
- `blazen-llm-candle` and `blazen-embed-candle` are the only crates with
  genuinely buggy feature gates today (Section 2 #2). Fix those first.
- The user explicitly accepts the `cfg(target_arch)` mechanism for WASM;
  don't propose a `wasm` Cargo feature even if it looks tempting for
  bindings.
