/**
 * Local speech-to-text with Blazen's whisper.cpp backend.
 *
 * whisper.cpp runs speech-to-text transcription entirely on-device using
 * GGML-quantised Whisper models.  The first call downloads the model
 * weights from HuggingFace Hub (tens to hundreds of MB depending on the
 * chosen variant) and caches them locally.  Subsequent runs reuse the
 * cached weights automatically.
 *
 * Unlike cloud-based transcription providers (fal.ai, AssemblyAI, Whisper
 * API, etc.), whisper.cpp is ideal for local development, air-gapped
 * deployments, and on-device inference where audio must not leave the
 * machine.
 *
 * Audio format requirements:
 *     whisper.cpp currently accepts **16-bit PCM mono WAV at 16 kHz**.
 *     Use ffmpeg to convert other formats:
 *         ffmpeg -i input.mp3 -ar 16000 -ac 1 -c:a pcm_s16le output.wav
 *
 * Build with the feature enabled first:
 *     cd crates/blazen-node
 *     npm install
 *     npm run build -- --features whispercpp
 *
 * Then run:
 *     BLAZEN_WHISPER_AUDIO_PATH=/path/to/audio.wav npx tsx audio_whispercpp.ts
 */

import { Transcription } from "blazen";

// ---------------------------------------------------------------------------
// 0. Resolve the audio file path.
// ---------------------------------------------------------------------------
// The example needs a user-supplied WAV file because no small public
// audio fixtures are checked into the repository.  Pass the path via
// the BLAZEN_WHISPER_AUDIO_PATH environment variable.

const audioPath = process.env.BLAZEN_WHISPER_AUDIO_PATH;
if (!audioPath) {
  console.error(
    "error: set BLAZEN_WHISPER_AUDIO_PATH=/path/to/audio.wav before running.\n" +
      "       whisper.cpp expects 16-bit PCM mono WAV at 16 kHz.\n" +
      "       convert with: ffmpeg -i input.mp3 -ar 16000 -ac 1 " +
      "-c:a pcm_s16le out.wav",
  );
  process.exit(2);
}

// ---------------------------------------------------------------------------
// 1. Construct the whisper.cpp transcription provider.
// ---------------------------------------------------------------------------
// `Transcription.whispercpp` is an async factory because it downloads
// and loads the model on the first call.  Subsequent runs reuse the
// cached model weights from `~/.cache/blazen/models`.

console.log("Loading whisper.cpp model (downloads on first run)...");

// `WhisperModel.Base` is a good starting point: ~74M parameters, ~1GB
// RAM, downloads a ~150MB weight file on first use.  Use `"small"` for
// slightly better accuracy, or `"largeV3"` for state-of-the-art
// quality (at the cost of ~10GB RAM and a ~3GB download).
const transcriber = await Transcription.whispercpp({
  model: "base",
  // Uncomment to pin the output language (otherwise whisper auto-detects).
  // language: "en",
});

console.log(`Provider: ${transcriber.providerId}`);
console.log();

// ---------------------------------------------------------------------------
// 2. Transcribe the audio file.
// ---------------------------------------------------------------------------
// whisper.cpp is offline-only: pass a local file path as `audioUrl`.
// Remote URLs are rejected with an `Unsupported` error.

console.log(`Transcribing: ${audioPath}`);

const result = await transcriber.transcribe({
  audioUrl: audioPath,
});

// ---------------------------------------------------------------------------
// 3. Print the results.
// ---------------------------------------------------------------------------

console.log();
console.log("--- Transcript ---");
console.log(result.text);
console.log();

if (result.language) {
  console.log(`Detected language: ${result.language}`);
}
console.log(`Segments:          ${result.segments.length}`);

if (result.segments.length > 0) {
  console.log();
  console.log("--- Segments ---");
  for (const segment of result.segments) {
    const start = segment.start.toFixed(2).padStart(7);
    const end = segment.end.toFixed(2).padStart(7);
    console.log(`  [${start}s - ${end}s] ${segment.text}`);
  }
}

console.log();
console.log("Done.");
