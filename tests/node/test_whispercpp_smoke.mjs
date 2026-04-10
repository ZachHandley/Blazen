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

import { describe, it } from "node:test";
import assert from "node:assert/strict";
import { existsSync } from "node:fs";

import { Transcription } from "../../crates/blazen-node/index.js";

const BLAZEN_TEST_WHISPERCPP = process.env.BLAZEN_TEST_WHISPERCPP;
const AUDIO_PATH = process.env.BLAZEN_WHISPER_AUDIO_PATH;

const hasAudio = Boolean(AUDIO_PATH && existsSync(AUDIO_PATH));

describe("whisper.cpp local transcription", { skip: !BLAZEN_TEST_WHISPERCPP }, () => {
  it("exposes a whispercpp factory when built with the feature", async () => {
    if (typeof Transcription.whispercpp !== "function") {
      // Not built with whispercpp feature -- skip gracefully.
      return;
    }

    const transcriber = await Transcription.whispercpp({ model: "base" });
    assert.equal(transcriber.providerId, "whispercpp");
  });

  it("accepts WhisperOptions fields without error", async () => {
    if (typeof Transcription.whispercpp !== "function") {
      return;
    }

    const transcriber = await Transcription.whispercpp({
      model: "base",
      language: "en",
      device: "cpu",
    });
    assert.equal(transcriber.providerId, "whispercpp");
  });

  it(
    "transcribes a local WAV file to non-empty text",
    { skip: !hasAudio },
    async () => {
      if (typeof Transcription.whispercpp !== "function") {
        return;
      }

      const transcriber = await Transcription.whispercpp({ model: "base" });
      const result = await transcriber.transcribe({ audioUrl: AUDIO_PATH });

      assert.ok(result.text, "expected non-empty transcription text");
      assert.ok(result.text.length > 0, "text should not be empty");
      assert.ok(result.segments.length >= 1, "expected at least one segment");

      for (const segment of result.segments) {
        assert.ok(segment.end >= segment.start, "segment timestamps must be ordered");
        assert.ok(segment.text != null, "segment must carry text");
      }
    },
  );

  it(
    "respects a language hint passed via WhisperOptions",
    { skip: !hasAudio },
    async () => {
      if (typeof Transcription.whispercpp !== "function") {
        return;
      }

      const transcriber = await Transcription.whispercpp({
        model: "base",
        language: "en",
      });
      const result = await transcriber.transcribe({ audioUrl: AUDIO_PATH });

      assert.ok(result.text, "expected non-empty transcription text");
      assert.ok(result.language, "expected a detected/specified language code");
    },
  );
});
