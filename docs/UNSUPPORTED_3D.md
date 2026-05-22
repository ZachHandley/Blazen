# Unsupported 3D backends

This file tracks the state of 3D generation, rigging, texturing, analysis,
reconstruction, refinement, and animation backends in Blazen as of
**2026-05-22**. Each pending native port links to its upstream public
repository so a reader can find the architecture, weights, and license
directly at the source.

---

## Today (HTTP-proxy via FalProvider)

All of the models below are reachable today through the
`blazen_llm::compute::ThreeDGeneration` trait via `FalProvider`. No native
Rust weights inference exists yet, but anything you can render on fal.ai
you can call from Blazen with the standard provider config:

- **TRELLIS.2** (Microsoft, MIT — 24 GB VRAM)
- **TRELLIS v1** family (text-to-3D, image-to-3D, multi-view)
- **Hunyuan3D family** (Tencent — 2.0 / 2.1 / Mini variants)
- **SF3D** (Stability AI single-image-to-3D)
- **TripoSR** (Stability AI + Tripo single-image-to-3D)
- **UniRig** (auto-rigging)
- **Texturizers** — fal-hosted Texture, SyncMVD, Text2Tex, Flux-Texture,
  PBR material estimators

This works for production today. The "pending native ports" section below
is purely about removing the cloud dependency.

---

## Pending native ports

### TRELLIS.2

**TRELLIS.2** — Microsoft's flagship 3D generator using a sparse-voxel +
Structured Latent (SLAT) representation, ~24 GB VRAM. License: MIT. Blocked
on Rust ecosystem gap-fills for sparse convolutions (`spconv-rs`) and the
FlexGEMM kernel used by the attention blocks. Once those exist, the candle
port is a 3-4 person-week project. **Status:** pending native Rust port.
Upstream: <https://github.com/microsoft/TRELLIS>.

### TRELLIS v1 family

**TRELLIS v1** — the original Microsoft TRELLIS line — smaller and less
memory-hungry than TRELLIS.2, but architecturally similar (sparse-voxel +
transformer). Shares the same `spconv-rs` blocker; a TRELLIS v1 port would
arrive as a smaller companion to the v1 work or as a stepping stone.
License: MIT. **Status:** pending native Rust port. Upstream:
<https://github.com/microsoft/TRELLIS> (earlier release tags).

### Hunyuan3D family

**Hunyuan3D 2.0 / 2.1 / Mini** — Tencent's open-weight DiT-based
image-to-3D and text-to-3D line, comparable quality to TRELLIS v1 at lower
VRAM. **License: Tencent's commercial-use license requires a separate
agreement — flag this clearly when wiring the backend.** **Status:**
pending native Rust port. Upstream:
<https://github.com/Tencent/Hunyuan3D-2>.

### SF3D

**SF3D (Stable Fast 3D)** — Stability AI's fast single-image-to-3D model.
Smaller and faster than TRELLIS or Hunyuan3D at lower fidelity. Good first
candidate for "native on a laptop GPU" because the architecture is more
conventional (no sparse-voxel infra needed). License: Stability AI
Community License. **Status:** pending native Rust port. Upstream:
<https://github.com/Stability-AI/stable-fast-3d>.

### TripoSR

**TripoSR** — Stability AI + Tripo collaboration: single-image-to-3D
transformer (LRM-style). License: MIT. **This is the next planned native
port** because the architecture is conventional and the license is
unambiguously permissive. **Status:** pending native Rust port, scheduled
next in the 3D queue. Upstream:
<https://github.com/VAST-AI-Research/TripoSR>.

### UniRig

**UniRig** — auto-rigging: takes a static mesh and produces a skeleton +
skinning weights. The Rust gap here is mostly tooling (skeleton
manipulation, weight painting primitives) rather than model architecture.
**Status:** pending native Rust port. Upstream:
<https://github.com/VAST-AI-Research/UniRig>.

Adjacent: a Rigify-pattern procedural auto-rigger is also on the roadmap;
it does not have a single upstream model repository — it is a port of the
Blender Rigify metarig methodology.

### Texturizers (Texture, SyncMVD, Text2Tex, Flux-Texture, PBR estimator)

The texturizer family covers two distinct shapes: multi-view synthesis
and PBR material estimation from images. They share the same Rust blockers
(diffusion DiT in candle, multi-view geometry primitives) and benefit from
the same wins. **Status:** pending native Rust ports. Upstream references:

- **TEXTure** (paper): <https://arxiv.org/abs/2302.01721>
- **SyncMVD**: <https://github.com/LIU-Yuxin/SyncMVD>
- **Text2Tex**: <https://github.com/daveredrum/Text2Tex>
- **Flux-Texture / PBR estimators** — fal-hosted today; upstream
  implementations vary per provider.

### Image analyzers

Components consumed by the 3D pipeline but useful standalone:

- **Depth-Anything-V2** — monocular depth estimation. License: Apache-2.0
  (model code; weights vary by checkpoint). Straightforward candle port.
  Upstream: <https://github.com/DepthAnything/Depth-Anything-V2>.
- **YOLOv8-WorldV2** — open-vocabulary detection. **License: AGPL-3.0 —
  flag this clearly for any commercial deployment.** Upstream:
  <https://github.com/ultralytics/ultralytics>.
- **Grounding DINO** — open-vocabulary object grounding. License:
  Apache-2.0. Upstream: <https://github.com/IDEA-Research/GroundingDINO>.
- **SAM2** — Segment Anything 2. License: Apache-2.0; the streaming-video
  mask propagation path adds complexity over SAM v1. Upstream:
  <https://github.com/facebookresearch/sam2>.

### Reconstruction (COLMAP, MASt3R)

Multi-view stereo / structure-from-motion:

- **COLMAP** — classical SfM/MVS baseline with mature C++ bindings
  (`colmap-rs` is partial). Upstream:
  <https://github.com/colmap/colmap>.
- **MASt3R** — Naver's neural feed-forward alternative (DUSt3R successor,
  fast dense matching). Upstream: <https://github.com/naver/mast3r>.

**Status:** both pending native Rust ports / wrappers.

### Mesh refinement

Poisson surface reconstruction, mesh decimation, UV unwrapping, retopo.
This is more of a Rust-ecosystem-gap area than a model port — most of the
work is wrapping or porting classic geometry algorithms rather than ML.
**Status:** pending; no single upstream — see issues / future commits in
this repo for the per-algorithm plan.

### Animators

Motion synthesis for rigged meshes:

- **MotionGPT** — LLM-style motion generation from text. Upstream:
  <https://github.com/OpenMotionLab/MotionGPT>.
- **Motion Lab** — broader motion-synthesis pipeline; the upstream
  documentation is the source of truth for the latest reference repo.
- **CogVideoX-driven** — extract motion from generated video and retarget
  to a rig. Depends on a native CogVideoX port (see `UNSUPPORTED_VIDEO.md`);
  CogVideoX upstream: <https://github.com/THUDM/CogVideo>.

**Status:** all pending native Rust ports.

---

## Rust ecosystem gaps

Cross-cutting infrastructure that several of the above depend on — sparse
voxels (`spconv-rs`), the FlexGEMM kernel, multi-view geometry primitives
(`kaolin`-equivalent helpers), and broader geometry-processing building
blocks — remain open work. Rust ecosystem fills like spconv, FlexGEMM and
kaolin remain open work; see issues / future commits in this repo.
