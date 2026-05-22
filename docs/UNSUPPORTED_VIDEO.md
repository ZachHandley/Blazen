# Unsupported video backends

This file tracks the state of text-to-video and image-to-video backends in
Blazen as of **2026-05-22**. Each pending native port links to its upstream
public repository so a reader can find the architecture, weights, and
license directly at the source.

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

**CogVideoX** — THUDM's open-weight text-to-video and image-to-video line:
DiT-based diffusion over a 3D VAE latent. License: Apache-2.0. The 2B
variant is the most realistic first-native target (smaller VRAM, similar
architecture to the 5B). I2V shares weights with T2V plus an
image-conditioning encoder. **Status:** pending native Rust port; blocked
on a candle 3D VAE codec and DiT block variants. Upstream:
<https://github.com/THUDM/CogVideo>.

### HunyuanVideo

**HunyuanVideo** — Tencent's open-weight video DiT, substantially larger
than CogVideoX 5B at higher fidelity. Same family of architectural
blockers (DiT in candle, 3D VAE codec, flow-matching sampler). **License:
Tencent's commercial-use license requires a separate agreement — flag this
clearly when wiring the backend.** **Status:** pending native Rust port.
Upstream: <https://github.com/Tencent/HunyuanVideo>.

### Mochi

**Mochi** — Genmo's open-weight video model: smaller-than-Hunyuan
parameter count but a non-standard sampler and conditioning scheme.
License: Apache-2.0 — notable for the permissive license. **Status:**
pending native Rust port. Upstream: <https://github.com/genmoai/models>.

---

## Shared primitives

Cross-cutting infrastructure that all three video models share — 3D VAE
codec, DiT block variants, flow-matching sampler, frame-interpolation
helpers — and video-specific Rust ecosystem gaps (DiT attention kernels
tuned for video, T2V evaluation harness, etc.) remain open work. Rust
ecosystem fills like spconv, FlexGEMM and kaolin remain open work; see
issues / future commits in this repo. The per-model sections above call
out each model's individual blockers.
