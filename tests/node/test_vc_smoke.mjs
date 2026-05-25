// Voice-conversion bindings — hermetic smoke tests.
//
// Always-on (no env flag, no model downloads): every test in this file
// exercises only the typed surface of the RVC backend — option shapes,
// factory wiring, modelId round-trips. Tests that actually pull RVC
// weights or hit `$BLAZEN_RVC_VOICE_DIR/<voice>/` are gated behind:
//
//     BLAZEN_TEST_VC=1
//
// Build with the audio-vc-rvc feature enabled first:
//     pnpm --filter blazen run build
// (the canonical build uses `local-all` which includes `audio-vc-rvc`).
//
// Run:
//     pnpm exec ava tests/node/test_vc_smoke.mjs

import test from "ava";

import {
  RvcBackend,
  VcModel,
  VcError,
  VcEngineNotAvailableError,
  VcModelLoadError,
  VcConversionError,
  VcVoiceNotFoundError,
  VcUnsupportedError,
  VcIoError,
} from "../../crates/blazen-node/index.js";

const BLAZEN_TEST_VC = process.env.BLAZEN_TEST_VC;

const hasRvcExport = typeof RvcBackend === "function";
const hasVcModelExport = typeof VcModel === "function";

// Hermetic surface tests (run unconditionally — no weights, no network).
const H = (hasRvcExport && hasVcModelExport) ? test : test.skip;

// Live tests that need weights — gated on BLAZEN_TEST_VC.
const L = BLAZEN_TEST_VC ? test : test.skip;

// ---------------------------------------------------------------------------
// RvcBackend
// ---------------------------------------------------------------------------

H("vc · RvcBackend.create({}) builds a handle with modelId 'rvc'", (t) => {
  const backend = RvcBackend.create({});
  t.truthy(backend, "expected an RvcBackend instance");
  t.is(backend.modelId, "rvc");
});

H("vc · RvcBackend.create() with no options defaults to modelId 'rvc'", (t) => {
  const backend = RvcBackend.create();
  t.is(backend.modelId, "rvc");
});

H("vc · RvcBackend.create accepts topK / retrievalBlend / rvcVersion", (t) => {
  const backend = RvcBackend.create({
    topK: 16,
    retrievalBlend: 0.5,
    rvcVersion: "v1",
  });
  t.is(backend.modelId, "rvc");
});

H("vc · RvcBackend exposes convertVoice + streamConvertPcm as functions", (t) => {
  const backend = RvcBackend.create({});
  t.is(typeof backend.convertVoice, "function");
  t.is(typeof backend.streamConvertPcm, "function");
  t.is(typeof backend.listTargetVoices, "function");
  t.is(typeof backend.registerTargetVoice, "function");
});

H("vc · RvcBackend.convertVoice with a missing path rejects", async (t) => {
  const backend = RvcBackend.create({});
  const err = await t.throwsAsync(() =>
    backend.convertVoice("/nonexistent/path/that-does-not-exist.wav", "speaker-01"),
  );
  // Real-mode: voice lookup fires first and surfaces VcVoiceNotFoundError
  // (no voice profile registered for "speaker-01"). I/O happens after.
  // Stub-mode: VcEngineNotAvailableError.
  t.true(
    err instanceof VcError,
    `expected a VcError subclass, got ${err?.constructor?.name}: ${err?.message}`,
  );
});

H("vc · RvcBackend.registerTargetVoice surfaces VcUnsupportedError in real mode (or VcEngineNotAvailableError in stub mode)", async (t) => {
  const backend = RvcBackend.create({});
  const err = await t.throwsAsync(() =>
    backend.registerTargetVoice("new-speaker", "/nonexistent/ref.wav"),
  );
  t.true(
    err instanceof VcUnsupportedError || err instanceof VcEngineNotAvailableError,
    `expected VcUnsupportedError or VcEngineNotAvailableError, got ${err?.constructor?.name}: ${err?.message}`,
  );
});

H("vc · RvcBackend.listTargetVoices returns an array (or VcEngineNotAvailableError in stub mode)", async (t) => {
  const backend = RvcBackend.create({});
  try {
    const voices = await backend.listTargetVoices();
    // Real mode: returns either an empty array (no voice dir) or whatever
    // the user has registered under $BLAZEN_RVC_VOICE_DIR.
    t.true(Array.isArray(voices));
  } catch (err) {
    // Stub mode: VcEngineNotAvailableError.
    t.true(
      err instanceof VcEngineNotAvailableError,
      `expected VcEngineNotAvailableError, got ${err?.constructor?.name}: ${err?.message}`,
    );
  }
});

// ---------------------------------------------------------------------------
// VcModel aggregator
// ---------------------------------------------------------------------------

H("vc · VcModel.rvc({}) reports the same modelId as RvcBackend.create({})", (t) => {
  const direct = RvcBackend.create({});
  const unified = VcModel.rvc({});
  t.is(unified.modelId, direct.modelId);
});

H("vc · VcModel exposes the full convert + stream + voice-management surface", (t) => {
  const m = VcModel.rvc({});
  t.is(typeof m.convertVoice, "function");
  t.is(typeof m.streamConvertPcm, "function");
  t.is(typeof m.listTargetVoices, "function");
  t.is(typeof m.registerTargetVoice, "function");
});

H("vc · VcModel.rvc propagates the topK / retrievalBlend / rvcVersion options", (t) => {
  // Construction must succeed and report the same modelId regardless of opts.
  t.is(VcModel.rvc({ topK: 4 }).modelId, "rvc");
  t.is(VcModel.rvc({ retrievalBlend: 1.0 }).modelId, "rvc");
  t.is(VcModel.rvc({ rvcVersion: "v2" }).modelId, "rvc");
});

// ---------------------------------------------------------------------------
// Error classes are exported and form the expected hierarchy
// ---------------------------------------------------------------------------

H("vc · all Vc* error classes are exported", (t) => {
  t.is(typeof VcError, "function");
  t.is(typeof VcEngineNotAvailableError, "function");
  t.is(typeof VcModelLoadError, "function");
  t.is(typeof VcConversionError, "function");
  t.is(typeof VcVoiceNotFoundError, "function");
  t.is(typeof VcUnsupportedError, "function");
  t.is(typeof VcIoError, "function");
});

H("vc · Vc* subclasses all extend VcError extends Error", (t) => {
  t.true(VcEngineNotAvailableError.prototype instanceof VcError);
  t.true(VcModelLoadError.prototype instanceof VcError);
  t.true(VcConversionError.prototype instanceof VcError);
  t.true(VcVoiceNotFoundError.prototype instanceof VcError);
  t.true(VcUnsupportedError.prototype instanceof VcError);
  t.true(VcIoError.prototype instanceof VcError);
  t.true(VcError.prototype instanceof Error);
});

// ---------------------------------------------------------------------------
// Live tests (gated on BLAZEN_TEST_VC=1) — these need a registered voice
// under $BLAZEN_RVC_VOICE_DIR/<voice>/ to actually do anything useful.
// ---------------------------------------------------------------------------

L("vc [live] · RvcBackend.listTargetVoices returns whatever is registered", async (t) => {
  const backend = RvcBackend.create({});
  const voices = await backend.listTargetVoices();
  t.true(Array.isArray(voices));
  for (const v of voices) {
    t.is(typeof v.id, "string");
    t.is(typeof v.sampleRateHz, "number");
  }
});
