# Vendored dependencies

Blazen vendors a handful of upstream crates when one of the following
conditions holds:

1. **License incompatibility.** Upstream pulls a GPL (or stricter)
   transitive dep that we cannot link into the MPL-2.0 workspace.
2. **Stack divergence.** Upstream depends on a runtime / build tool
   (ONNX runtime, C++ compiler, …) that the rest of the workspace has
   already replaced with a Rust-native equivalent.
3. **Upstream abandonment.** Last commit > 12 months ago and a small
   patch keeps the crate alive.

Each vendored crate lives under `crates/blazen-*-vendored/` and ships a
`VENDORED.md` next to its `Cargo.toml` documenting the upstream URL,
pinned version, applied patches, and a re-sync procedure.

## Current vendored crates

### `crates/blazen-audio-piper-vendored/`

Source: [`thewh1teagle/piper-rs`][piper-rs] v0.1.9 (MIT).

| Driver | Detail |
|--------|--------|
| Reason 1 (stack) | Upstream uses `ort 2.0.0-rc.11`; Blazen standardises on `tract-onnx 0.21` (see `blazen-embed-tract`). |
| Reason 2 (license) | Upstream's `espeak-rs` transitively `cc`-compiles GPL-3.0+ eSpeak NG sources into the artifact. MPL-2.0 + GPL-3.0+ is link-time incompatible. |

Patches:

- Inference path swapped from `ort::session::Session` →
  `tract_onnx::prelude::SimplePlan`. Tensor build uses
  `tract_ndarray::Array{1,2}::from(...).into_tensor()`.
- Phonemizer swapped from in-process `espeak-rs` FFI →
  out-of-process `tokio::process::Command::new("espeak-ng")` /
  `std::process::Command`. The subprocess boundary stops GPL
  obligations from flowing into the calling artifact.

Runtime requirement: the system `espeak-ng` binary must be on `PATH`
(install via `apt install espeak-ng`, `brew install espeak-ng`,
`pacman -S espeak-ng`, etc.). The `Piper::new` constructor preflights
the binary and surfaces `PiperError::EspeakNgMissing` with an install
hint if absent.

Surfaced through: `blazen-audio-tts/piper` feature →
`blazen_audio_tts::PiperBackend`. The bridge crate (`blazen-llm`)
exposes it via `audio-tts-piper`.

Re-sync procedure: see `crates/blazen-audio-piper-vendored/VENDORED.md`.

[piper-rs]: https://github.com/thewh1teagle/piper-rs

## When to add another vendored crate

Before vendoring, try:

1. Upstreaming the patch (always preferable).
2. Forking on GitHub and depending on the fork via a `git = ...`
   coordinate (avoid this — `git` deps don't reproduce well in CI).
3. Filing an issue / PR with the upstream maintainer.

Only vendor when the patch is small, well-scoped, and the upstream
maintenance cadence makes (1)–(3) impractical. Every vendored crate
adds long-tail maintenance load.
