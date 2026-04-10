"""Local speech-to-text with whisper.cpp (no API key needed).

whisper.cpp runs speech-to-text transcription entirely on-device using
GGML-quantised Whisper models.  The first call downloads the model
weights from HuggingFace Hub (tens to hundreds of MB depending on the
chosen variant) and caches them locally.  Subsequent runs reuse the
cached weights automatically.

Unlike cloud-based transcription providers (fal.ai, AssemblyAI, Whisper
API, etc.), whisper.cpp is ideal for local development, air-gapped
deployments, and on-device inference where audio must not leave the
machine.

Audio format requirements:
    whisper.cpp currently accepts **16-bit PCM mono WAV at 16 kHz**.
    Use ffmpeg to convert other formats:
        ffmpeg -i input.mp3 -ar 16000 -ac 1 -c:a pcm_s16le output.wav

Usage:
    BLAZEN_WHISPER_AUDIO_PATH=/path/to/audio.wav \\
        uv run python crates/blazen-py/examples/audio_whispercpp.py
"""

import asyncio
import os
import sys

from blazen import (
    Transcription,
    TranscriptionRequest,
    WhisperModel,
    WhisperOptions,
)


async def main() -> None:
    # ------------------------------------------------------------------
    # 0. Resolve the audio file path.
    # ------------------------------------------------------------------
    # The example needs a user-supplied WAV file because no small public
    # audio fixtures are checked into the repository.  Pass the path via
    # the BLAZEN_WHISPER_AUDIO_PATH environment variable.
    audio_path = os.environ.get("BLAZEN_WHISPER_AUDIO_PATH")
    if not audio_path:
        print(
            "error: set BLAZEN_WHISPER_AUDIO_PATH=/path/to/audio.wav before running.\n"
            "       whisper.cpp expects 16-bit PCM mono WAV at 16 kHz.\n"
            "       convert with: ffmpeg -i input.mp3 -ar 16000 -ac 1 "
            "-c:a pcm_s16le out.wav",
            file=sys.stderr,
        )
        raise SystemExit(2)

    if not os.path.exists(audio_path):
        print(f"error: audio file does not exist: {audio_path}", file=sys.stderr)
        raise SystemExit(2)

    # ------------------------------------------------------------------
    # 1. Configure the whisper.cpp backend.
    # ------------------------------------------------------------------
    # `WhisperModel.Base` is a good starting point: ~74M parameters,
    # ~1GB RAM, downloads a ~150MB weight file on first use.  Use
    # `WhisperModel.Small` for slightly better accuracy, or
    # `WhisperModel.LargeV3` for state-of-the-art quality (at the cost
    # of ~10GB RAM and a ~3GB download).
    opts = WhisperOptions(
        model=WhisperModel.Base,
        # Uncomment to pin the output language (otherwise whisper auto-detects).
        # language="en",
    )

    # ------------------------------------------------------------------
    # 2. Construct the transcription provider.
    # ------------------------------------------------------------------
    # This downloads and loads the model on the first call.  Subsequent
    # runs reuse the cached model weights from `~/.cache/blazen/models`.
    print("Loading whisper.cpp model (downloads on first run)...")
    transcriber = Transcription.whispercpp(options=opts)
    print(f"Provider: {transcriber.provider_id}")
    print()

    # ------------------------------------------------------------------
    # 3. Transcribe the audio file.
    # ------------------------------------------------------------------
    # `TranscriptionRequest.from_file` is the preferred constructor for
    # local backends; it sets `audio_source` to a local path and leaves
    # `audio_url` empty.
    print(f"Transcribing: {audio_path}")
    result = await transcriber.transcribe(
        TranscriptionRequest.from_file(audio_path),
    )

    # ------------------------------------------------------------------
    # 4. Print the results.
    # ------------------------------------------------------------------
    print()
    print("--- Transcript ---")
    print(result.text)
    print()

    if result.language:
        print(f"Detected language: {result.language}")
    print(f"Segments:          {len(result.segments)}")

    if result.segments:
        print()
        print("--- Segments ---")
        for segment in result.segments:
            print(f"  [{segment.start:7.2f}s - {segment.end:7.2f}s] {segment.text}")

    print()
    print("Done.")


if __name__ == "__main__":
    asyncio.run(main())
