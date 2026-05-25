"""Smoke tests for the `VcModel` / `VcChunk` / `VcStream` / `TargetVoice` surface.

The construction-free smoke (`test_vc_module_smoke`) always runs and
verifies the typed surface is reachable from Python — it requires only
that blazen-py was built with the `audio-vc-rvc` feature.

The live `RVC convert_voice` test is opt-in via `BLAZEN_RUN_RVC_TESTS=1`
plus `BLAZEN_RVC_VOICE_ID` and `BLAZEN_RVC_INPUT_WAV` because it depends
on a pre-staged voice profile under `$BLAZEN_RVC_VOICE_DIR` and a source
utterance on disk.
"""

import os

import pytest
import blazen as bz

pytestmark = pytest.mark.asyncio


def test_vc_module_smoke():
    """Construction-free smoke: types are importable when the
    `audio-vc-rvc` feature is compiled in."""
    if not hasattr(bz, "VcModel"):
        pytest.skip("blazen built without audio-vc-rvc feature")
    assert hasattr(bz, "VcChunk")
    assert hasattr(bz, "VcStream")
    assert hasattr(bz, "VcModel")
    assert hasattr(bz, "TargetVoice")
    assert hasattr(bz, "RvcError")
    # RvcError is a subclass of ProviderError.
    assert issubclass(bz.RvcError, bz.ProviderError)


async def test_vc_list_empty_dir(tmp_path):
    """`list_target_voices` returns an empty list when `voice_dir`
    contains no profiles."""
    if not hasattr(bz, "VcModel"):
        pytest.skip("blazen built without audio-vc-rvc feature")
    model = bz.VcModel.rvc(voice_dir=tmp_path)
    voices = await model.list_target_voices()
    assert isinstance(voices, list)
    assert voices == []


async def test_vc_register_voice_unsupported(tmp_path):
    """`register_target_voice` raises UnsupportedError on the current
    RVC backend (voice profiles must be pre-staged on disk)."""
    if not hasattr(bz, "VcModel"):
        pytest.skip("blazen built without audio-vc-rvc feature")
    model = bz.VcModel.rvc(voice_dir=tmp_path)
    ref_path = tmp_path / "reference.wav"
    ref_path.write_bytes(b"RIFF\x00\x00\x00\x00WAVE")  # bogus content; never read
    with pytest.raises(bz.UnsupportedError):
        await model.register_target_voice("speaker-01", ref_path)


@pytest.mark.skipif(
    not hasattr(bz, "VcModel")
    or os.getenv("BLAZEN_RUN_RVC_TESTS") != "1"
    or not os.getenv("BLAZEN_RVC_VOICE_ID")
    or not os.getenv("BLAZEN_RVC_INPUT_WAV"),
    reason="requires audio-vc-rvc feature, BLAZEN_RUN_RVC_TESTS=1, "
    "BLAZEN_RVC_VOICE_ID, and BLAZEN_RVC_INPUT_WAV "
    "(plus a voice profile under $BLAZEN_RVC_VOICE_DIR)",
)
async def test_rvc_convert_voice_live():
    """End-to-end live test against a pre-staged RVC voice profile."""
    voice_dir = os.getenv("BLAZEN_RVC_VOICE_DIR")
    voice_id = os.environ["BLAZEN_RVC_VOICE_ID"]
    input_wav = os.environ["BLAZEN_RVC_INPUT_WAV"]
    model = bz.VcModel.rvc(voice_dir=voice_dir) if voice_dir else bz.VcModel.rvc()
    wav_bytes = await model.convert_voice(input_wav, voice_id)
    assert isinstance(wav_bytes, (bytes, bytearray))
    assert len(wav_bytes) > 44  # at least a WAV header
    assert wav_bytes[:4] == b"RIFF"
    assert wav_bytes[8:12] == b"WAVE"
