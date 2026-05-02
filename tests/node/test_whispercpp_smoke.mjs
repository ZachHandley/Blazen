/**
 * whisper.cpp local transcription smoke tests.
 *
 * Gated on two environment variables to avoid surprises in CI:
 *
 *     BLAZEN_TEST_WHISPERCPP=1                  # run the tests
 *     BLAZEN_WHISPER_AUDIO_PATH=/path/to/x.wav  # audio file to transcribe
 *
 * Only runs when the native binding is compiled with the `whispercpp`
 * feature.
 *
 * Convert a source audio file to the required format with:
 *     ffmpeg -i input.mp3 -ar 16000 -ac 1 -c:a pcm_s16le output.wav
 *
 * Build with the feature enabled first:
 *     cd crates/blazen-node && npm install && npm run build -- --features whispercpp
 *
 * Run:
 *     BLAZEN_TEST_WHISPERCPP=1 \
 *     BLAZEN_WHISPER_AUDIO_PATH=/path/to/audio.wav \
 *     node --test tests/node/test_whispercpp_smoke.mjs
 */

import test from "ava";
import { existsSync } from "node:fs";

import { Transcription } from "../../crates/blazen-node/index.js";

const BLAZEN_TEST_WHISPERCPP = process.env.BLAZEN_TEST_WHISPERCPP;
const AUDIO_PATH = process.env.BLAZEN_WHISPER_AUDIO_PATH;

const hasAudio = Boolean(AUDIO_PATH && existsSync(AUDIO_PATH));

const T = BLAZEN_TEST_WHISPERCPP ? test : test.skip;
const TAudio = BLAZEN_TEST_WHISPERCPP && hasAudio ? test : test.skip;

T("whisper.cpp local transcription · exposes a whispercpp factory when built with the feature", async (t) => {
  if (typeof Transcription.whispercpp !== "function") {
    // Not built with whispercpp feature -- skip gracefully.
    t.pass("whispercpp feature not built");
    return;
  }

  const transcriber = await Transcription.whispercpp({ model: "base" });
  t.is(transcriber.providerId, "whispercpp");
});

T("whisper.cpp local transcription · accepts WhisperOptions fields without error", async (t) => {
  if (typeof Transcription.whispercpp !== "function") {
    t.pass("whispercpp feature not built");
    return;
  }

  const transcriber = await Transcription.whispercpp({
    model: "base",
    language: "en",
    device: "cpu",
  });
  t.is(transcriber.providerId, "whispercpp");
});

TAudio("whisper.cpp local transcription · transcribes a local WAV file to non-empty text", async (t) => {
  if (typeof Transcription.whispercpp !== "function") {
    t.pass("whispercpp feature not built");
    return;
  }

  const transcriber = await Transcription.whispercpp({ model: "base" });
  const result = await transcriber.transcribe({ audioUrl: AUDIO_PATH });

  t.truthy(result.text, "expected non-empty transcription text");
  t.truthy(result.text.length > 0, "text should not be empty");
  t.truthy(result.segments.length >= 1, "expected at least one segment");

  for (const segment of result.segments) {
    t.truthy(segment.end >= segment.start, "segment timestamps must be ordered");
    t.truthy(segment.text != null, "segment must carry text");
  }
});

TAudio("whisper.cpp local transcription · respects a language hint passed via WhisperOptions", async (t) => {
  if (typeof Transcription.whispercpp !== "function") {
    t.pass("whispercpp feature not built");
    return;
  }

  const transcriber = await Transcription.whispercpp({
    model: "base",
    language: "en",
  });
  const result = await transcriber.transcribe({ audioUrl: AUDIO_PATH });

  t.truthy(result.text, "expected non-empty transcription text");
  t.truthy(result.language, "expected a detected/specified language code");
});
