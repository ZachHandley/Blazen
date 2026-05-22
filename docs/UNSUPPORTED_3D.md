# Unsupported 3D backends

This file tracks the state of 3D generation, rigging, texturing, analysis,
reconstruction, refinement, and animation backends in Blazen as of
**2026-05-22**.

In-depth design notes, port plans, license verdicts, and "resume-from-here"
breadcrumbs live in the **BlackLeafDocs** sibling repository
(`../../BlackLeafDocs/blazen/3d/`). That is the canonical home of all
research-grade material; this file is a short index that ships in the public
Blazen tree.

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

Microsoft's flagship 3D generator — sparse-voxel + SLAT representation,
~24 GB VRAM, MIT license. Blocked on Rust ecosystem gap-fills for sparse
convolutions (`spconv-rs`) and the FlexGEMM kernel used by the attention
blocks. Once those exist, the candle port is a 3-4 person-week project.

[Research notes](../../BlackLeafDocs/blazen/3d/generator/trellis2.md)

### TRELLIS v1 family

The original Microsoft TRELLIS line — smaller and less memory-hungry than
TRELLIS.2, but architecturally similar (sparse-voxel + transformer). Shares
the same `spconv-rs` blocker; a TRELLIS v1 port would arrive as a smaller
companion to the v1 work or as a stepping stone.

[Research notes](../../BlackLeafDocs/blazen/3d/generator/trellis_v1.md)

### Hunyuan3D family

Tencent's open-weight Hunyuan3D 2.0 / 2.1 / Mini line. DiT-based
image-to-3D and text-to-3D, comparable quality to TRELLIS v1 at lower
VRAM. License is permissive for non-commercial; commercial use needs
Tencent's separate license — flag this clearly when wiring the backend.

[Research notes](../../BlackLeafDocs/blazen/3d/generator/hunyuan3d.md)

### SF3D

Stability AI's fast single-image-to-3D model. Smaller and faster than
TRELLIS or Hunyuan3D, lower fidelity. Good first candidate for "native
on a laptop GPU" because the architecture is more conventional (no
sparse-voxel infra needed).

[Research notes](../../BlackLeafDocs/blazen/3d/generator/sf3d.md)

### TripoSR

Stability AI + Tripo collaboration — single-image-to-3D transformer.
**This is the next planned native port** — see the resume-from-here plan
in the BlackLeafDocs note for the staged approach.

[Research notes](../../BlackLeafDocs/blazen/3d/generator/triposr.md)

### UniRig

Auto-rigging — takes a static mesh and produces a skeleton + skinning
weights. The Rust gap here is mostly tooling (skeleton manipulation,
weight painting primitives) rather than model architecture.

[Research notes](../../BlackLeafDocs/blazen/3d/rigger/unirig.md)

Adjacent: the Rigify-pattern auto-rigger note lives at
`../../BlackLeafDocs/blazen/3d/rigger/rigify.md`.

### Texturizers (Texture, SyncMVD, Text2Tex, Flux-Texture, PBR estimator)

The texturizer family covers two distinct shapes: multi-view synthesis
(SyncMVD, Text2Tex, Flux-Texture, Texture) and PBR material estimation
from images. They share the same Rust blockers (diffusion DiT in candle,
multi-view geometry primitives) and benefit from the same wins.

[Research notes](../../BlackLeafDocs/blazen/3d/texturizer/)

### Image analyzers

Components consumed by the 3D pipeline but useful standalone:

- **Depth-Anything-V2** — monocular depth estimation. Apache 2.0,
  straightforward candle port.
- **YOLOv8-WorldV2** — open-vocabulary detection. **License is AGPL-3.0**;
  flag this clearly for any commercial deployment.
- **Grounding DINO** — open-vocabulary object grounding. Apache 2.0.
- **SAM2** — Segment Anything 2. Apache 2.0; the streaming-video mask
  propagation path adds complexity over SAM v1.

[Research notes](../../BlackLeafDocs/blazen/3d/analyzers/)

### Reconstruction (COLMAP, MASt3R)

Multi-view stereo / structure-from-motion. COLMAP has C++ bindings
(`colmap-rs` is partial) and is the conventional baseline. MASt3R is the
neural-feed-forward alternative (DUSt3R successor, fast dense matching).

[Research notes](../../BlackLeafDocs/blazen/3d/reconstruction/)

### Mesh refinement

Poisson surface reconstruction, mesh decimation, UV unwrapping, retopo.
This is more of a Rust-ecosystem-gap area than a model port — most of the
work is wrapping or porting classic geometry algorithms rather than ML.

[Research notes](../../BlackLeafDocs/blazen/3d/refiner/)

### Animators

Motion synthesis for rigged meshes:

- **MotionGPT** — LLM-style motion generation from text.
- **Motion Lab** — broader motion-synthesis pipeline.
- **CogVideoX-driven** — extract motion from generated video and retarget
  to a rig. Depends on a native CogVideoX port (see `UNSUPPORTED_VIDEO.md`).

[Research notes](../../BlackLeafDocs/blazen/3d/animator/)

---

## Rust ecosystem gaps

Cross-cutting infrastructure that several of the above depend on lives in
two index documents:

- [Shared primitives](../../BlackLeafDocs/blazen/3d/shared_primitives.md) —
  sparse voxels, geometry processing helpers, common building blocks.
- [Rust ecosystem gaps](../../BlackLeafDocs/blazen/3d/rust_ecosystem_gaps.md) —
  the `spconv-rs`, FlexGEMM, multi-view geometry, and tooling gaps that
  need to be filled before several of these ports become realistic.
