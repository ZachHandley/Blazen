# Unsupported video backends

This file tracks the state of text-to-video and image-to-video backends in
Blazen as of **2026-05-22**.

In-depth design notes, port plans, license verdicts, and "resume-from-here"
breadcrumbs live in the **BlackLeafDocs** sibling repository
(`../../BlackLeafDocs/blazen/video/`). That is the canonical home of all
research-grade material; this file is a short index that ships in the public
Blazen tree.

---

## Today (HTTP-proxy via FalProvider)

The video models below are reachable today through the
`blazen_llm::compute::VideoGeneration` trait via `FalProvider`. No native
Rust weights inference exists yet, but anything you can render on fal.ai
you can call from Blazen with the standard provider config:

- **CogVideoX** (5B / 2B / I2V)
- **HunyuanVideo**
- **Mochi**

This works for production today. The "pending native ports" section below
is purely about removing the cloud dependency.

---

## Pending native ports

### CogVideoX 5B / 2B / I2V

THUDM's open-weight text-to-video and image-to-video line. DiT-based
diffusion over a 3D VAE latent. The 2B variant is the most realistic
first-native target (smaller VRAM, similar architecture to the 5B). I2V
shares weights with T2V plus an image-conditioning encoder.

[Research notes](../../BlackLeafDocs/blazen/video/cogvideox.md)

### HunyuanVideo

Tencent's open-weight video DiT — substantially larger than CogVideoX 5B,
higher fidelity. Same family of architectural blockers (DiT in candle,
3D VAE codec, flow-matching sampler). License is permissive for
non-commercial; commercial use needs Tencent's separate license — flag
this clearly when wiring the backend.

[Research notes](../../BlackLeafDocs/blazen/video/hunyuanvideo.md)

### Mochi

Genmo's open-weight video model, Apache-2.0. Smaller-than-Hunyuan
parameter count but a non-standard sampler and conditioning scheme.
Notable for the permissive license.

[Research notes](../../BlackLeafDocs/blazen/video/mochi.md)

---

## Shared primitives

Cross-cutting infrastructure that all three video models share — 3D VAE
codec, DiT block variants, flow-matching sampler, frame-interpolation
helpers — lives in:

[Shared primitives](../../BlackLeafDocs/blazen/video/shared_primitives.md)

The cross-cutting Rust ecosystem gap notes for video (video-specific DiT
attention kernels, T2V evaluation harness, etc.) are co-located in
`../../BlackLeafDocs/blazen/video/` alongside the per-model notes — there
is no single `rust_ecosystem_gaps.md` for video today; the per-model
notes each call out their own blockers.
