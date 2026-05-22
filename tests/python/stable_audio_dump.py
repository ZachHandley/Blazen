"""Tensor-dump harness for the StableAudio candle port (PR-A foundation).

This script loads Stability AI's ``stable-audio-open-small`` model via the
upstream ``stable-audio-tools`` Python library, runs a fixed prompt + seed,
and dumps intermediate tensors after each major architectural component to
``~/.cache/blazen-stableaudio-research/dumps/``.

The Rust port (``blazen-audio-music`` candle backend for Stable Audio) then
loads those dumps and asserts block-by-block tensor equality. Tolerances:

* ``1e-5`` for FP32 tensors
* ``1e-3`` for BF16 tensors

Runtime requirements
--------------------
* ``pip install stable-audio-tools`` (also brings ``torch`` + ``descript-audio-codec``)
* HuggingFace credentials with access to ``stabilityai/stable-audio-open-small``
* ~2 GB free disk for the weight cache; ~50 MB for the dump artifacts
* Python 3.10+; a GPU is recommended but not required (CPU works, just slow).

The harness is intentionally **not** invoked by ``cargo test`` -- it gates
on ``stable-audio-tools`` being importable, so plain ``pytest`` discovery
will skip it via :mod:`test_stable_audio_dump_smoke`.

Dumped tensors
--------------
For the fixed prompt below, the following block names are written:

* ``t5_encoder_last_hidden_state`` -- output of the T5-base text encoder
* ``cond_seconds_start_embedding`` -- Fourier-feature embedding of start time
* ``cond_seconds_total_embedding`` -- Fourier-feature embedding of duration
* ``dit_input_latent`` -- noisy latent fed into the first DiT block
* ``dit_input_conditioning`` -- concatenated cross-attention conditioning tensor
* ``dit_block_<i>_out`` -- output of DiT block ``i`` (0-indexed)
* ``dit_final_output`` -- final velocity/noise prediction from the DiT
* ``sampler_step_<i>_latent`` -- latent state after distilled sampler step ``i``
* ``vae_decoder_output`` -- final FP32 stereo waveform at 44.1 kHz

Each tensor is written as raw ``.bin`` (NumPy ``.tobytes()``) plus a sibling
``<name>.meta.json`` describing shape, dtype, and parent block. A top-level
``manifest.json`` lists every dumped tensor for the Rust harness to iterate.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

# Fixed inputs -- the Rust port must reproduce these byte-for-byte.
FIXED_PROMPT: str = "upbeat lofi hip hop with vinyl crackle and warm pads"
FIXED_SEED: int = 42
FIXED_DURATION_SECONDS: float = 10.5
FIXED_SECONDS_START: float = 0.0
SAMPLE_RATE: int = 44_100
DEFAULT_SAMPLER_STEPS: int = 8
MODEL_REPO: str = "stabilityai/stable-audio-open-small"

DUMP_ROOT_ENV: str = "BLAZEN_STABLEAUDIO_DUMP_DIR"
DEFAULT_DUMP_ROOT: Path = (
    Path.home() / ".cache" / "blazen-stableaudio-research" / "dumps"
)


def _resolve_dump_root(override: Path | None = None) -> Path:
    if override is not None:
        root = Path(override)
    else:
        env = os.environ.get(DUMP_ROOT_ENV)
        root = Path(env) if env else DEFAULT_DUMP_ROOT
    root.mkdir(parents=True, exist_ok=True)
    return root


@dataclass
class DumpEntry:
    name: str
    path: str
    meta_path: str
    shape: list[int]
    dtype: str
    parent_block: str


@dataclass
class DumpManifest:
    prompt: str = FIXED_PROMPT
    seed: int = FIXED_SEED
    duration_seconds: float = FIXED_DURATION_SECONDS
    seconds_start: float = FIXED_SECONDS_START
    sample_rate: int = SAMPLE_RATE
    sampler_steps: int = DEFAULT_SAMPLER_STEPS
    model_repo: str = MODEL_REPO
    entries: list[DumpEntry] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "prompt": self.prompt,
            "seed": self.seed,
            "duration_seconds": self.duration_seconds,
            "seconds_start": self.seconds_start,
            "sample_rate": self.sample_rate,
            "sampler_steps": self.sampler_steps,
            "model_repo": self.model_repo,
            "entries": [e.__dict__ for e in self.entries],
        }


def _torch_dtype_to_numpy_name(dtype: Any) -> str:
    """Map a torch dtype to a stable name the Rust loader understands."""
    import torch  # local: only required when actually dumping

    mapping = {
        torch.float32: "float32",
        torch.float16: "float16",
        torch.bfloat16: "bfloat16",
        torch.float64: "float64",
        torch.int64: "int64",
        torch.int32: "int32",
        torch.int16: "int16",
        torch.int8: "int8",
        torch.uint8: "uint8",
        torch.bool: "bool",
    }
    if dtype not in mapping:
        raise ValueError(f"Unsupported torch dtype for dump: {dtype}")
    return mapping[dtype]


def dump_tensor(
    tensor: Any,
    name: str,
    parent_block: str,
    dump_root: Path,
    manifest: DumpManifest,
) -> DumpEntry:
    """Write a torch tensor to ``<dump_root>/<name>.bin`` + ``.meta.json``.

    BF16 tensors are written as raw 2-byte big-endian values via ``tobytes``
    on the underlying view -- the Rust loader interprets the bytes natively.
    """
    import numpy as np
    import torch

    assert isinstance(tensor, torch.Tensor), f"dump_tensor expects torch.Tensor, got {type(tensor)}"

    dtype_name = _torch_dtype_to_numpy_name(tensor.dtype)
    cpu = tensor.detach().to("cpu").contiguous()

    if tensor.dtype == torch.bfloat16:
        # numpy lacks bf16; serialize the raw 2-byte representation.
        raw = cpu.view(torch.uint16).numpy().tobytes()
    else:
        raw = cpu.numpy().tobytes()

    bin_path = dump_root / f"{name}.bin"
    meta_path = dump_root / f"{name}.meta.json"
    bin_path.write_bytes(raw)

    shape = list(cpu.shape)
    meta = {
        "name": name,
        "parent_block": parent_block,
        "shape": shape,
        "dtype": dtype_name,
        "byte_length": len(raw),
    }
    meta_path.write_text(json.dumps(meta, indent=2))

    entry = DumpEntry(
        name=name,
        path=str(bin_path),
        meta_path=str(meta_path),
        shape=shape,
        dtype=dtype_name,
        parent_block=parent_block,
    )
    manifest.entries.append(entry)
    return entry


def _install_dit_block_hooks(
    dit_module: Any,
    dump_root: Path,
    manifest: DumpManifest,
    max_blocks: int | None,
) -> list[Any]:
    """Attach forward hooks to every DiT transformer block."""
    handles: list[Any] = []

    # stable-audio-tools exposes the transformer blocks under
    # ``DiffusionTransformer.transformer.layers`` (a ContinuousTransformer).
    # We probe a few candidate attribute paths so the harness keeps working
    # across the (rare) upstream renames.
    candidates = [
        ("transformer", "layers"),
        ("transformer", "blocks"),
        ("blocks",),
    ]
    blocks = None
    for path in candidates:
        node: Any = dit_module
        ok = True
        for attr in path:
            if not hasattr(node, attr):
                ok = False
                break
            node = getattr(node, attr)
        if ok:
            blocks = node
            break

    if blocks is None:
        raise RuntimeError(
            "Could not locate DiT transformer blocks; upstream layout changed?"
        )

    iter_blocks = list(blocks)
    if max_blocks is not None:
        iter_blocks = iter_blocks[:max_blocks]

    def make_hook(i: int) -> Callable[..., None]:
        def hook(_module: Any, _inputs: Any, output: Any) -> None:
            tensor = output[0] if isinstance(output, tuple) else output
            dump_tensor(
                tensor,
                name=f"dit_block_{i}_out",
                parent_block=f"dit/block_{i}",
                dump_root=dump_root,
                manifest=manifest,
            )

        return hook

    for i, block in enumerate(iter_blocks):
        handles.append(block.register_forward_hook(make_hook(i)))
    return handles


def _dump_t5_output(conditioner: Any, dump_root: Path, manifest: DumpManifest) -> None:
    """Find the T5-based text conditioner and capture its last_hidden_state."""
    # stable-audio-open-small uses a ``T5Conditioner`` whose ``self.model`` is
    # a HuggingFace ``T5EncoderModel``. We wrap its forward to capture
    # last_hidden_state on the next invocation.
    text_cond = None
    if hasattr(conditioner, "conditioners"):
        for value in conditioner.conditioners.values():
            if value.__class__.__name__.lower().startswith("t5"):
                text_cond = value
                break
    if text_cond is None or not hasattr(text_cond, "model"):
        return

    original_forward = text_cond.model.forward

    def wrapped(*args: Any, **kwargs: Any) -> Any:
        out = original_forward(*args, **kwargs)
        last_hidden = getattr(out, "last_hidden_state", None)
        if last_hidden is not None:
            dump_tensor(
                last_hidden,
                name="t5_encoder_last_hidden_state",
                parent_block="conditioner/t5",
                dump_root=dump_root,
                manifest=manifest,
            )
        # Restore on first call so we don't accumulate dumps across sampler steps.
        text_cond.model.forward = original_forward
        return out

    text_cond.model.forward = wrapped


def _dump_numeric_conditioners(
    conditioner: Any, dump_root: Path, manifest: DumpManifest
) -> None:
    """Capture the seconds_start / seconds_total Fourier-feature embeddings."""
    if not hasattr(conditioner, "conditioners"):
        return

    targets = {
        "seconds_start": "cond_seconds_start_embedding",
        "seconds_total": "cond_seconds_total_embedding",
    }
    for key, dump_name in targets.items():
        cond = conditioner.conditioners.get(key)
        if cond is None:
            continue

        original_forward = cond.forward

        def make_wrapper(orig: Any, name: str) -> Any:
            def wrapped(*args: Any, **kwargs: Any) -> Any:
                out = orig(*args, **kwargs)
                # Numeric conditioners typically return (embedding, mask).
                tensor = out[0] if isinstance(out, tuple) else out
                dump_tensor(
                    tensor,
                    name=name,
                    parent_block=f"conditioner/{name}",
                    dump_root=dump_root,
                    manifest=manifest,
                )
                cond.forward = orig
                return out

            return wrapped

        cond.forward = make_wrapper(original_forward, dump_name)


def _install_dit_io_hook(
    dit_module: Any,
    dump_root: Path,
    manifest: DumpManifest,
) -> Any:
    """Hook the DiT's outermost forward to capture input latent + final output."""
    captured = {"done_input": False, "done_output": False}
    original_forward = dit_module.forward

    def wrapped(*args: Any, **kwargs: Any) -> Any:
        if not captured["done_input"]:
            x = args[0] if args else kwargs.get("x")
            if x is not None:
                dump_tensor(
                    x,
                    name="dit_input_latent",
                    parent_block="dit/input",
                    dump_root=dump_root,
                    manifest=manifest,
                )
            cross_attn = kwargs.get("cross_attn_cond")
            if cross_attn is not None:
                dump_tensor(
                    cross_attn,
                    name="dit_input_conditioning",
                    parent_block="dit/input",
                    dump_root=dump_root,
                    manifest=manifest,
                )
            captured["done_input"] = True

        out = original_forward(*args, **kwargs)

        if not captured["done_output"]:
            tensor = out[0] if isinstance(out, tuple) else out
            dump_tensor(
                tensor,
                name="dit_final_output",
                parent_block="dit/output",
                dump_root=dump_root,
                manifest=manifest,
            )
            captured["done_output"] = True
        return out

    dit_module.forward = wrapped
    # Return a sentinel so callers can restore if they want; we leave it
    # patched because the script only runs one generation per invocation.
    return original_forward


def _install_sampler_callback(
    sampler_kwargs: dict[str, Any],
    dump_root: Path,
    manifest: DumpManifest,
) -> None:
    """Inject a per-step callback into the sampler kwargs.

    ``stable-audio-tools`` exposes a ``callback`` kwarg on its sampling
    functions; it receives ``{"i": step, "x": current_latent, ...}``.
    """

    def cb(payload: dict[str, Any]) -> None:
        step = int(payload.get("i", -1))
        latent = payload.get("x") or payload.get("denoised")
        if latent is None:
            return
        dump_tensor(
            latent,
            name=f"sampler_step_{step}_latent",
            parent_block=f"sampler/step_{step}",
            dump_root=dump_root,
            manifest=manifest,
        )

    sampler_kwargs["callback"] = cb


def run_dump(
    dump_root: Path | None = None,
    max_dit_blocks: int | None = None,
    sampler_steps: int = DEFAULT_SAMPLER_STEPS,
) -> DumpManifest:
    """Run the full dump pipeline. Returns the populated manifest.

    Parameters
    ----------
    dump_root
        Output directory; defaults to ``~/.cache/blazen-stableaudio-research/dumps``.
    max_dit_blocks
        If set, only hook the first ``N`` DiT blocks. Used by the smoke test
        to keep CI fast.
    sampler_steps
        Number of distilled sampling steps; the released model is trained
        for 8 but the smoke test overrides this to 1.
    """
    # All heavy imports go inside the function so that simply *parsing* this
    # file (and pytest collection) doesn't require torch or stable-audio-tools.
    import torch
    from stable_audio_tools import get_pretrained_model
    from stable_audio_tools.inference.generation import generate_diffusion_cond

    root = _resolve_dump_root(dump_root)
    manifest = DumpManifest(sampler_steps=sampler_steps)

    torch.manual_seed(FIXED_SEED)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, model_config = get_pretrained_model(MODEL_REPO)
    model = model.to(device).eval()

    sample_rate = int(model_config.get("sample_rate", SAMPLE_RATE))
    manifest.sample_rate = sample_rate
    sample_size = int(model_config.get("sample_size", int(sample_rate * FIXED_DURATION_SECONDS)))

    # Hook installation. Order matters: conditioner hooks run during the
    # cond pre-pass, DiT hooks during the denoise loop, VAE hook at the end.
    conditioner = getattr(model, "conditioner", None)
    if conditioner is not None:
        _dump_t5_output(conditioner, root, manifest)
        _dump_numeric_conditioners(conditioner, root, manifest)

    dit = None
    for attr in ("model", "diffusion", "dit"):
        if hasattr(model, attr):
            candidate = getattr(model, attr)
            if candidate.__class__.__name__.lower().find("transformer") != -1 or attr == "model":
                dit = candidate
                break
    if dit is None:
        raise RuntimeError("Could not locate DiT module on the loaded model")

    dit_handles = _install_dit_block_hooks(dit, root, manifest, max_dit_blocks)
    _install_dit_io_hook(dit, root, manifest)

    sampler_kwargs: dict[str, Any] = {
        "model": model,
        "steps": sampler_steps,
        "cfg_scale": 1.0,  # distilled model: no CFG
        "conditioning": [
            {
                "prompt": FIXED_PROMPT,
                "seconds_start": FIXED_SECONDS_START,
                "seconds_total": FIXED_DURATION_SECONDS,
            }
        ],
        "sample_size": sample_size,
        "sigma_min": 0.3,
        "sigma_max": 500.0,
        "sampler_type": "pingpong",
        "device": device,
        "seed": FIXED_SEED,
    }
    _install_sampler_callback(sampler_kwargs, root, manifest)

    with torch.no_grad():
        audio = generate_diffusion_cond(**sampler_kwargs)

    dump_tensor(
        audio,
        name="vae_decoder_output",
        parent_block="vae/decoder",
        dump_root=root,
        manifest=manifest,
    )

    for h in dit_handles:
        h.remove()

    manifest_path = root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest.to_dict(), indent=2))
    return manifest


def main() -> None:
    manifest = run_dump()
    print(f"Wrote {len(manifest.entries)} tensors to {DEFAULT_DUMP_ROOT}")


if __name__ == "__main__":
    main()
