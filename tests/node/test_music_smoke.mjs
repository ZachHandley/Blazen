// Music + SFX generation bindings — hermetic smoke tests.
//
// Always-on (no env flag, no model downloads): every test in this file
// exercises only the typed surface of the music backends — validation
// errors, factory shapes, modelId round-trips. Tests that actually pull
// MusicGen / AudioGen / Stable Audio weights are gated behind:
//
//     BLAZEN_TEST_MUSIC=1
//
// Build with the music features enabled first:
//     pnpm --filter blazen run build
// (the canonical build uses `local-all` which includes
//  `audio-music-musicgen`, `audio-music-audiogen`, and
//  `audio-music-stable-audio`).
//
// Run:
//     pnpm exec ava tests/node/test_music_smoke.mjs

import test from "ava";

import {
  MusicgenBackend,
  AudioGenBackend,
  StableAudioBackend,
  MusicModel,
  MusicError,
  MusicInvalidInputError,
  MusicNotYetImplementedError,
  MusicEngineNotAvailableError,
  MusicHfHubError,
} from "../../crates/blazen-node/index.js";

const BLAZEN_TEST_MUSIC = process.env.BLAZEN_TEST_MUSIC;

const hasMusicgenExport = typeof MusicgenBackend === "function";
const hasAudioGenExport = typeof AudioGenBackend === "function";
const hasStableAudioExport = typeof StableAudioBackend === "function";
const hasMusicModelExport = typeof MusicModel === "function";

// Hermetic surface tests (run unconditionally — no weights, no network).
const H = (hasMusicgenExport && hasAudioGenExport && hasStableAudioExport && hasMusicModelExport)
  ? test
  : test.skip;

// Live tests that need weights — gated on BLAZEN_TEST_MUSIC.
const L = BLAZEN_TEST_MUSIC ? test : test.skip;

// ---------------------------------------------------------------------------
// MusicgenBackend
// ---------------------------------------------------------------------------

H("music · MusicgenBackend.create({}) builds a small-variant handle", (t) => {
  const backend = MusicgenBackend.create({});
  t.truthy(backend, "expected a MusicgenBackend instance");
  t.is(backend.modelId, "musicgen-small");
});

H("music · MusicgenBackend.create({ variant: 'medium' }) reports musicgen-medium", (t) => {
  const backend = MusicgenBackend.create({ variant: "medium" });
  t.is(backend.modelId, "musicgen-medium");
});

H("music · MusicgenBackend.create({ variant: 'large' }) reports musicgen-large", (t) => {
  const backend = MusicgenBackend.create({ variant: "large" });
  t.is(backend.modelId, "musicgen-large");
});

H("music · MusicgenBackend.create() with no options defaults to small", (t) => {
  const backend = MusicgenBackend.create();
  t.is(backend.modelId, "musicgen-small");
});

H("music · MusicgenBackend.generateMusic('', 8) rejects with MusicInvalidInputError", async (t) => {
  const backend = MusicgenBackend.create({});
  const err = await t.throwsAsync(() => backend.generateMusic("", 8));
  // Real-mode (engine compiled in): the validator rejects empty prompts
  // before any weight load. Stub-mode (engine compiled out): we'd see
  // MusicEngineNotAvailableError instead.
  t.true(
    err instanceof MusicInvalidInputError || err instanceof MusicEngineNotAvailableError,
    `expected MusicInvalidInputError or MusicEngineNotAvailableError, got ${err?.constructor?.name}: ${err?.message}`,
  );
});

H("music · MusicgenBackend.generateMusic('piano', -1) rejects", async (t) => {
  const backend = MusicgenBackend.create({});
  const err = await t.throwsAsync(() => backend.generateMusic("piano", -1));
  t.true(
    err instanceof MusicInvalidInputError || err instanceof MusicEngineNotAvailableError,
    `expected MusicInvalidInputError or MusicEngineNotAvailableError, got ${err?.constructor?.name}: ${err?.message}`,
  );
});

H("music · MusicgenBackend.generateMusic('piano', 9999) rejects past the 60s hard cap", async (t) => {
  const backend = MusicgenBackend.create({});
  const err = await t.throwsAsync(() => backend.generateMusic("piano", 9999));
  t.true(
    err instanceof MusicInvalidInputError || err instanceof MusicEngineNotAvailableError,
    `expected MusicInvalidInputError or MusicEngineNotAvailableError, got ${err?.constructor?.name}: ${err?.message}`,
  );
});

H("music · MusicgenBackend exposes streamGenerate{Music,Sfx} as functions", (t) => {
  const backend = MusicgenBackend.create({});
  t.is(typeof backend.streamGenerateMusic, "function");
  t.is(typeof backend.streamGenerateSfx, "function");
});

// ---------------------------------------------------------------------------
// AudioGenBackend
// ---------------------------------------------------------------------------

H("music · AudioGenBackend.create({}) reports audiogen-medium", (t) => {
  const backend = AudioGenBackend.create({});
  t.truthy(backend);
  t.is(backend.modelId, "audiogen-medium");
});

H("music · AudioGenBackend.generateSfx('', 4) rejects", async (t) => {
  const backend = AudioGenBackend.create({});
  const err = await t.throwsAsync(() => backend.generateSfx("", 4));
  t.true(
    err instanceof MusicInvalidInputError || err instanceof MusicEngineNotAvailableError,
    `expected MusicInvalidInputError or MusicEngineNotAvailableError, got ${err?.constructor?.name}: ${err?.message}`,
  );
});

H("music · AudioGenBackend exposes streamGenerate{Music,Sfx} as functions", (t) => {
  const backend = AudioGenBackend.create({});
  t.is(typeof backend.streamGenerateMusic, "function");
  t.is(typeof backend.streamGenerateSfx, "function");
});

// ---------------------------------------------------------------------------
// StableAudioBackend
// ---------------------------------------------------------------------------

H("music · StableAudioBackend.create({}) reports stable-audio", (t) => {
  const backend = StableAudioBackend.create({});
  t.truthy(backend);
  t.is(backend.modelId, "stable-audio");
});

H("music · StableAudioBackend.generateMusic rejects (NotYetImplemented in stub mode; HF/IO/Invalid in real mode without weights)", async (t) => {
  const backend = StableAudioBackend.create({});
  // In stub mode (audio-music-stable-audio OFF) this throws
  // MusicNotYetImplementedError. In real mode (the canonical build),
  // it'll try to download a tokenizer / weights and fail with one of
  // MusicHfHubError, MusicIoError, or MusicInvalidInputError. All four
  // are acceptable here — what matters is that the call rejects rather
  // than hanging or succeeding without weights.
  const err = await t.throwsAsync(() => backend.generateMusic("ambient", 8));
  t.true(
    err instanceof MusicError,
    `expected a MusicError subclass, got ${err?.constructor?.name}: ${err?.message}`,
  );
});

H("music · StableAudioBackend exposes streamGenerate{Music,Sfx} as functions", (t) => {
  const backend = StableAudioBackend.create({});
  t.is(typeof backend.streamGenerateMusic, "function");
  t.is(typeof backend.streamGenerateSfx, "function");
});

// ---------------------------------------------------------------------------
// MusicModel aggregator
// ---------------------------------------------------------------------------

H("music · MusicModel.musicgen({}) reports the same modelId as MusicgenBackend.create({})", (t) => {
  const direct = MusicgenBackend.create({});
  const unified = MusicModel.musicgen({});
  t.is(unified.modelId, direct.modelId);
});

H("music · MusicModel.audioGen({}) reports audiogen-medium", (t) => {
  const m = MusicModel.audioGen({});
  t.is(m.modelId, "audiogen-medium");
});

H("music · MusicModel.stableAudio({}) reports stable-audio", (t) => {
  const m = MusicModel.stableAudio({});
  t.is(m.modelId, "stable-audio");
});

H("music · MusicModel exposes the full generate + stream surface", (t) => {
  const m = MusicModel.musicgen({});
  t.is(typeof m.generateMusic, "function");
  t.is(typeof m.generateSfx, "function");
  t.is(typeof m.streamGenerateMusic, "function");
  t.is(typeof m.streamGenerateSfx, "function");
});

H("music · MusicModel.musicgen propagates the variant choice", (t) => {
  t.is(MusicModel.musicgen({ variant: "medium" }).modelId, "musicgen-medium");
  t.is(MusicModel.musicgen({ variant: "large" }).modelId, "musicgen-large");
});

// ---------------------------------------------------------------------------
// Error classes are exported and form the expected hierarchy
// ---------------------------------------------------------------------------

H("music · all Music* error classes are exported", (t) => {
  t.is(typeof MusicError, "function");
  t.is(typeof MusicInvalidInputError, "function");
  t.is(typeof MusicEngineNotAvailableError, "function");
  t.is(typeof MusicNotYetImplementedError, "function");
  t.is(typeof MusicHfHubError, "function");
});

H("music · MusicInvalidInputError extends MusicError extends Error", (t) => {
  t.true(MusicInvalidInputError.prototype instanceof MusicError);
  t.true(MusicError.prototype instanceof Error);
});

// ---------------------------------------------------------------------------
// Live tests (gated on BLAZEN_TEST_MUSIC=1) — these pull real weights.
// ---------------------------------------------------------------------------

L("music [live] · MusicgenBackend.generateMusic returns a WAV result", async (t) => {
  const backend = MusicgenBackend.create({ variant: "small" });
  const result = await backend.generateMusic("upbeat piano riff", 4);
  t.truthy(result.bytes);
  t.true(result.bytes.byteLength > 0);
  t.is(result.format, "wav");
  t.is(result.sampleRate, 32_000);
  t.is(result.channels, 1);
});
