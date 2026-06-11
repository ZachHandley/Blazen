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

1. **GPL-3.0+ phonemizer.** Upstream's `espeak-rs` crate transitively
   `cc`-compiles the entire eSpeak NG C source tree (~30 MB) into the
   resulting library. eSpeak NG is **GPL-3.0+** which is link-time
   incompatible with our MPL-2.0 license posture. Linking it would
   force every downstream Blazen consumer to inherit GPL obligations,
   which is unacceptable for a library framework.

2. **Intel-mac target gating.** Upstream's `ort` dependency is
   unconditional; `ort-sys` ships no `x86_64-apple-darwin` prebuilt, so
   this fork target-gates `ort` out of that triple (mirroring
   `blazen-audio-stt`'s `vad-ort` gate) and returns
   `PiperError::Unsupported` from `Piper::new` there.

### History: the tract experiment (reverted)

An earlier revision of this fork swapped `ort` for the workspace's
`tract-onnx` so the whole workspace would converge on one inference
backend. That swap is **reverted**: tract cannot statically analyse
Piper VITS graphs at all — analysis (`into_typed()`) fails before any
plan (optimized or decluttered) can be built:

- tract 0.22: the ONNX `Pad` inference rule demands 3 inputs; Piper
  exports the legal 2-input form (`Wrong input number. Rules expect 3,
  node has 2` on `/enc_p/encoder/attn_layers.0/Pad`).
- tract 0.23: the attention layers' Reshape volume equality
  `2b(2p²+p−1) == 2b(p+1)(2p−1)` is mathematically true but unprovable
  by tract's symbolic-dim algebra (`Reshape volume mismatch` on
  `/enc_p/encoder/attn_layers.0/Reshape_10`).

`ort` 2.0.0-rc.12 was already in the workspace via `blazen-audio-stt`
(`vad-ort`), so reverting adds no new runtime to the dependency tree.

## Patches applied

| Subsystem | Upstream                              | Patched fork                                                    |
|-----------|---------------------------------------|-----------------------------------------------------------------|
| ONNX inference | `ort = 2.0.0-rc.11`, unconditional | `ort = 2.0.0-rc.12`, target-gated out of `x86_64-apple-darwin` (stub engine there) |
| Phonemizer    | `espeak_rs::text_to_phonemes` (FFI)   | `std::process::Command::new("espeak-ng")` invocation            |
| espeak data discovery | `espeak_Initialize` env-var probe | None — `espeak-ng` binary self-locates its data files          |

## What lives where in the patched source

```
src/
├── lib.rs        — public `Piper` handle, `PiperError`, `Piper::new` / `from_session` / `create`
├── model.rs      — voice config structs, phoneme-id table, ort engine (VITS forward pass) + Intel-mac stub
└── phonemize.rs  — `espeak-ng` subprocess wrapper (locate + invoke)
```

## API surface (compatibility with upstream)

The public types `Piper`, `PiperError`, `ModelConfig`, and the method
shapes `Piper::new(model_path, config_path)` / `Piper::create(text,
is_phonemes, speaker_id, length_scale, noise_scale, noise_w)` mirror
upstream. We added:

- `Piper::from_session(ort_session, config)` (mirrors upstream's
  `from_session`; not available on `x86_64-apple-darwin`)
- `Piper::sample_rate(&self) -> u32` (convenience)
- `PiperError::EspeakNgMissing(String)` (new variant for the
  subprocess phonemizer)
- `PiperError::Unsupported(String)` (new variant for the Intel-mac
  stub engine)
- `pub use ort;` re-export (so `from_session` callers can name the
  exact `ort` version this crate links; gated like the dependency)

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
   - Any new `ort::*` call → route through the target-gated engine in
     `src/model.rs` (keep the `x86_64-apple-darwin` stub in sync).
   - Any new `espeak_rs::*` call → subprocess wrapper in
     `src/phonemize.rs`.
5. Update the **Pinned version / Pinned commit** table at the top of
   this file.
6. Run `cargo nextest run -p blazen-audio-piper-vendored` and the
   `blazen-audio-tts --features piper` smoke + live tests.

[upstream]: https://github.com/thewh1teagle/piper-rs
