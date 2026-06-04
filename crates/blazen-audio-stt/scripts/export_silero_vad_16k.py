#!/usr/bin/env python3
"""Export a tract-loadable, control-flow-free Silero VAD v5 (16 kHz) ONNX.

Every stock Silero ONNX export (v4 and v5, from snakers4 and the HF mirrors)
contains nested ONNX ``If`` ops: a top-level ``If(sr == 16000)`` selecting the
16 kHz vs 8 kHz STFT path, plus shape-guard ``If``s inside the STFT and the
decoder RNN. ``tract`` (the pure-Rust ONNX runtime Blazen uses on musl / wasm
where ONNX Runtime doesn't ship) cannot statically analyse those ``If`` nodes
and fails at ``into_optimized()`` -- see snakers4/silero-vad#728.

This script produces a flat, ``If``-free 16 kHz model that BOTH ``ort`` (native)
and ``tract`` (musl/wasm) load and run identically:

  1. Load Silero's PyTorch bundle and grab the *inner* 16 kHz module
     (``model._model``). The sr-branching lives in the outer "merge" wrapper;
     the inner module's ``forward(x, state)`` has no sample-rate branch.
  2. ``torch.onnx.export`` it at a FIXED input shape -- ``input``=[1, 576]
     (512-sample frame + 64-sample lookback context) and ``state``=[2, 1, 128].
     This still emits a handful of shape-guard ``If``s.
  3. Re-serialise through ONNX Runtime's ``ORT_ENABLE_BASIC`` optimiser. With
     the shapes fixed, every ``If`` condition is a compile-time constant, so
     ORT folds all of them away. BASIC (not EXTENDED/ALL) is deliberate: the
     higher levels inject non-standard ``FusedConv`` / ``com.microsoft`` ops
     that tract doesn't implement.

The result is a ~1.2 MB graph (Conv / Pad(reflect) / LSTM / Sigmoid / ...),
no If/Loop/Scan, opset 18, verified to match the PyTorch reference to ~1e-7.

Regenerate with:

    uv run --with silero-vad --with torch --with onnx --with onnxruntime \
        --with numpy python crates/blazen-audio-stt/scripts/export_silero_vad_16k.py

Output: crates/blazen-audio-stt/assets/silero_vad_16k.onnx
"""

from __future__ import annotations

import pathlib
import warnings

import numpy as np
import onnx
import onnxruntime as ort
import silero_vad
import torch

warnings.filterwarnings("ignore")

ASSETS = pathlib.Path(__file__).resolve().parent.parent / "assets"
ASSETS.mkdir(exist_ok=True)
OUT = ASSETS / "silero_vad_16k.onnx"

STATE_DIMS = (2, 1, 128)
FRAME = 512


def main() -> None:
    bundle = silero_vad.load_silero_vad(onnx=False)
    inner = bundle._model  # 16 kHz VADRNNJIT (no sr branching)
    ctx = int(inner.context_size_samples)
    seq = FRAME + ctx
    assert seq == 576, f"expected 16 kHz context 64 -> seq 576, got {seq}"

    x = torch.randn(1, seq)
    state = torch.zeros(*STATE_DIMS)
    with torch.no_grad():
        ref_out, ref_state = inner.forward(x, state)

    tmp_raw = OUT.with_suffix(".raw.onnx")
    torch.onnx.export(
        inner,
        (x, state),
        str(tmp_raw),
        input_names=["input", "state"],
        output_names=["output", "stateN"],
        opset_version=18,
        dynamo=False,
    )

    # Constant-fold the If shape-guards via ORT BASIC (fixed shapes -> constant
    # conditions). BASIC keeps every node in the standard ONNX domain.
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
    so.optimized_model_filepath = str(OUT)
    sess = ort.InferenceSession(
        str(tmp_raw), so, providers=["CPUExecutionProvider"]
    )

    # Parity: ORT-optimised flat graph vs the PyTorch reference.
    got = sess.run(
        None,
        {"input": x.numpy().astype(np.float32), "state": state.numpy().astype(np.float32)},
    )
    prob_diff = float(np.abs(got[0] - ref_out.numpy()).max())
    state_diff = float(np.abs(got[1] - ref_state.numpy()).max())

    flat = onnx.load(str(OUT))
    ops = {n.op_type for n in flat.graph.node}
    cf = ops & {"If", "Loop", "Scan"}
    onnx.checker.check_model(flat)
    tmp_raw.unlink(missing_ok=True)

    print(f"wrote {OUT} ({OUT.stat().st_size} bytes)")
    print(f"  inputs:  input[1,{seq}], state{list(STATE_DIMS)}")
    print(f"  control-flow ops remaining: {cf or 'none'}")
    print(f"  parity vs torch -> prob {prob_diff:.2e}, state {state_diff:.2e}")
    if cf:
        raise SystemExit(f"FAILED: control-flow ops survived: {cf}")
    if prob_diff > 1e-4 or state_diff > 1e-4:
        raise SystemExit("FAILED: parity drift too large")


if __name__ == "__main__":
    main()
