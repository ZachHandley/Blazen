"""Smoke test for the StableAudio tensor-dump harness.

This test is intentionally cheap -- it overrides the dump harness to use a
single DiT block and a single sampler step so it can run on CPU in a few
minutes (still requires the 1.68 GB weight download on first run, but
skips entirely when ``stable-audio-tools`` is not installed).

The full dump (8 sampler steps, all DiT blocks) is intentionally NOT
exercised by the Rust test suite -- run ``python tests/python/stable_audio_dump.py``
manually to regenerate the reference tensors.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

# Skip cleanly when the upstream library is not installed. We import here
# (not inside the test) so pytest's collection step reports a clean skip.
pytest.importorskip(
    "stable_audio_tools",
    reason="stable-audio-tools is not installed; run `pip install stable-audio-tools`",
)
pytest.importorskip("torch", reason="torch is not installed")

# After the importorskips above we can import the harness safely. The
# harness itself defers torch / stable_audio_tools imports into the
# function body, so importing the module is cheap and side-effect free.
import stable_audio_dump  # type: ignore  # noqa: E402


@pytest.fixture
def tiny_dump_root(tmp_path: Path) -> Path:
    root = tmp_path / "stable_audio_dumps"
    root.mkdir()
    yield root
    shutil.rmtree(root, ignore_errors=True)


@pytest.mark.slow
def test_run_dump_smoke(tiny_dump_root: Path) -> None:
    """Exercise the harness end-to-end with a 1-block, 1-step override."""
    manifest = stable_audio_dump.run_dump(
        dump_root=tiny_dump_root,
        max_dit_blocks=1,
        sampler_steps=1,
    )

    # The manifest must list at least: VAE output + DiT final output +
    # one DiT block + one sampler step. Other dumps (T5, numeric cond)
    # depend on whether those subgraphs actually run on the smoke override,
    # so we don't require them.
    names = {e.name for e in manifest.entries}
    assert "vae_decoder_output" in names, names
    assert "dit_final_output" in names, names
    assert any(n.startswith("dit_block_") for n in names), names

    # manifest.json must be written and parseable.
    manifest_path = tiny_dump_root / "manifest.json"
    assert manifest_path.exists()
    text = manifest_path.read_text()
    assert FIXED_PROMPT_MARKER in text


# Keep the prompt assertion as a constant the test can read without
# pulling torch -- this avoids tripping torch's slow import in collection.
FIXED_PROMPT_MARKER = "upbeat lofi hip hop with vinyl crackle and warm pads"
