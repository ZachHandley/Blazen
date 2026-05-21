# Vendored: `piper-rs`

This crate is a vendored, patched fork of [`piper-rs`][upstream].

| Field          | Value                                           |
|----------------|-------------------------------------------------|
| Upstream repo  | https://github.com/thewh1teagle/piper-rs        |
| Pinned version | `0.1.9` (Cargo.toml `version`, HEAD as of clone) |
| Pinned commit  | `b5500e6` (HEAD on clone, no v0.1.9 tag exists) |
| Upstream license | MIT                                            |
| Workspace license | MPL-2.0 (this fork)                          |

## Why we vendor

1. **ONNX runtime mismatch.** Upstream depends on
   `ort = 2.0.0-rc.11`. The Blazen workspace already standardises on
   `tract-onnx = 0.21` (used by `blazen-embed-tract`), and shipping
   both runtimes side-by-side bloats every artifact by ~60 MB and
   forces consumers to pick which one the linker resolves.

2. **GPL-3.0+ phonemizer.** Upstream's `espeak-rs` crate transitively
   `cc`-compiles the entire eSpeak NG C source tree (~30 MB) into the
   resulting library. eSpeak NG is **GPL-3.0+** which is link-time
   incompatible with our MPL-2.0 license posture. Linking it would
   force every downstream Blazen consumer to inherit GPL obligations,
   which is unacceptable for a library framework.

## Patches applied

| Subsystem | Upstream                              | Patched fork                                                    |
|-----------|---------------------------------------|-----------------------------------------------------------------|
| ONNX inference | `ort::session::Session` + `ort::value::Tensor` | `tract_onnx::prelude::SimplePlan` + `tract_ndarray` |
| Model build   | `Session::builder().commit_from_file` | `tract_onnx::onnx().model_for_path().into_optimized().into_runnable()` |
| Tensor build  | `Tensor::from_array(...)`             | `tract_ndarray::Array{1,2}::from(...).into_tensor()`            |
| Phonemizer    | `espeak_rs::text_to_phonemes` (FFI)   | `std::process::Command::new("espeak-ng")` invocation            |
| espeak data discovery | `espeak_Initialize` env-var probe | None — `espeak-ng` binary self-locates its data files          |

## What lives where in the patched source

```
src/
├── lib.rs        — public `Piper` handle, `PiperError`, `Piper::new` / `with_voice` / `create`
├── model.rs      — tract plan + tensor build + VITS forward pass; phoneme-id table
└── phonemize.rs  — `espeak-ng` subprocess wrapper (locate + invoke)
```

## API surface (compatibility with upstream)

The public types `Piper`, `PiperError`, `ModelConfig`, and the method
shapes `Piper::new(model_path, config_path)` / `Piper::create(text,
is_phonemes, speaker_id, length_scale, noise_scale, noise_w)` mirror
upstream. We added:

- `Piper::from_model(tract_plan, config)` (mirrors upstream's
  `from_session` with the appropriate type swap)
- `Piper::sample_rate(&self) -> u32` (convenience)
- `PiperError::EspeakNgMissing(String)` (new variant for the
  subprocess phonemizer)

Upstream's `compile-espeak-intonations` feature is dropped (it gated
eSpeak NG source compilation; subprocess invocation makes it moot).

## Runtime requirement

Phonemization happens via a child `espeak-ng` process. The binary
must be installed and on `PATH`:

| OS              | Install command            |
|-----------------|----------------------------|
| Debian / Ubuntu | `apt install espeak-ng`    |
| macOS (Homebrew)| `brew install espeak-ng`   |
| Arch Linux      | `pacman -S espeak-ng`      |
| Fedora / RHEL   | `dnf install espeak-ng`    |
| Alpine          | `apk add espeak-ng-data`   |
| Windows         | https://github.com/espeak-ng/espeak-ng/releases (add install dir to PATH) |

If `espeak-ng` is missing at `Piper::new` time we surface
`PiperError::EspeakNgMissing` with an install hint.

## How to re-sync from upstream

When `piper-rs` ships a new release:

1. `git clone https://github.com/thewh1teagle/piper-rs ~/.cache/piper-rs-upstream`
2. `cd ~/.cache/piper-rs-upstream && git log --oneline` — pick the
   commit / tag you want.
3. Diff against `src/lib.rs` and `src/model.rs` in this crate. Most
   upstream changes will be additive (new helpers, new voice config
   fields) and apply cleanly.
4. Re-apply the two architectural patches:
   - Any new `ort::*` call → tract equivalent.
   - Any new `espeak_rs::*` call → subprocess wrapper in
     `src/phonemize.rs`.
5. Update the **Pinned version / Pinned commit** table at the top of
   this file.
6. Run `cargo nextest run -p blazen-audio-piper-vendored` and the
   `blazen-audio-tts --features piper` smoke + live tests.

[upstream]: https://github.com/thewh1teagle/piper-rs
