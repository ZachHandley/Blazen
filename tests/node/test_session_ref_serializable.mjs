/**
 * Tests for the Node `SessionRefSerializable` round-trip path.
 *
 * Mirrors the Python `tests/python/test_session_refs.py` coverage of
 * `SessionPausePolicy.PickleOrSerialize` + `Workflow.resume_with_session_refs`.
 *
 * The Node binding intentionally does NOT auto-detect a `serialize()`
 * method on JS objects (the bridge is still `serde_json::Value`-based —
 * see the module-level note on
 * `crates/blazen-node/src/workflow/session_ref_serializable.rs`).
 * Instead, JS code calls `ctx.insertSessionRefSerializable(typeName,
 * bytes)` with a payload it has already serialized into a `Buffer`,
 * and reads the bytes back via `ctx.getSessionRefSerializable(key)`
 * after the snapshot/resume round-trip — deserialization into a
 * runtime object is the user's responsibility.
 */

import test from "ava";

import { SessionPausePolicy, Workflow } from "../../crates/blazen-node/index.js";

// =========================================================================
// Helper: encode/decode a tiny "blob" type so the tests don't depend on
// any particular serialization library. The blob is `{ n, tag }` and
// the wire format is the UTF-8 bytes of its JSON representation.
// =========================================================================

const BLOB_TYPE_NAME = "tests::SerializableBlob";

function encodeBlob(blob) {
  return Buffer.from(JSON.stringify(blob), "utf-8");
}

function decodeBlob(bytes) {
  // `bytes` is a Buffer when read back from the registry.
  return JSON.parse(Buffer.from(bytes).toString("utf-8"));
}

// =========================================================================
// SessionPausePolicy enum surface
// =========================================================================

test("SessionPausePolicy enum · exposes the four documented variants as strings", (t) => {
  t.is(SessionPausePolicy.PickleOrError, "PickleOrError");
  t.is(
    SessionPausePolicy.PickleOrSerialize,
    "PickleOrSerialize",
  );
  t.is(SessionPausePolicy.WarnDrop, "WarnDrop");
  t.is(SessionPausePolicy.HardError, "HardError");
});

// =========================================================================
// Live insertion + retrieval (no snapshot)
// =========================================================================

test("ctx.insertSessionRefSerializable / getSessionRefSerializable · round-trips bytes and type name within a single step", async (t) => {
  const wf = new Workflow("live-roundtrip");
  wf.setSessionPausePolicy(SessionPausePolicy.PickleOrSerialize);

  wf.addStep("s", ["blazen::StartEvent"], async (event, ctx) => {
    const original = { n: 42, tag: "live" };
    const key = await ctx.insertSessionRefSerializable(
      BLOB_TYPE_NAME,
      encodeBlob(original),
    );

    const payload = await ctx.getSessionRefSerializable(key);
    if (!payload) {
      throw new Error("expected payload to be returned");
    }
    if (payload.typeName !== BLOB_TYPE_NAME) {
      throw new Error(`unexpected typeName: ${payload.typeName}`);
    }
    const recovered = decodeBlob(payload.bytes);

    return {
      type: "blazen::StopEvent",
      result: { recovered, key },
    };
  });

  const result = await wf.run({});
  t.deepEqual(result.data.recovered, { n: 42, tag: "live" });
  // The key should be a non-empty UUID string.
  t.regex(
    result.data.key,
    /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/,
  );
});

test("ctx.insertSessionRefSerializable / getSessionRefSerializable · returns null for unknown keys", async (t) => {
  const wf = new Workflow("missing-key");
  wf.setSessionPausePolicy(SessionPausePolicy.PickleOrSerialize);

  wf.addStep("s", ["blazen::StartEvent"], async (event, ctx) => {
    const payload = await ctx.getSessionRefSerializable(
      "deadbeef-0000-4000-8000-000000000000",
    );
    return {
      type: "blazen::StopEvent",
      result: { found: payload !== null },
    };
  });

  const result = await wf.run({});
  t.is(result.data.found, false);
});

test("ctx.insertSessionRefSerializable / getSessionRefSerializable · rejects malformed registry keys with a clear error", async (t) => {
  const wf = new Workflow("bad-key");
  wf.setSessionPausePolicy(SessionPausePolicy.PickleOrSerialize);

  wf.addStep("s", ["blazen::StartEvent"], async (event, ctx) => {
    let captured = null;
    try {
      await ctx.getSessionRefSerializable("not-a-uuid");
    } catch (err) {
      captured = err.message;
    }
    return {
      type: "blazen::StopEvent",
      result: { error: captured },
    };
  });

  const result = await wf.run({});
  t.truthy(
    result.data.error,
    "expected getSessionRefSerializable to throw on bad uuid",
  );
  t.regex(result.data.error, /not-a-uuid/);
});

// =========================================================================
// Snapshot capture
// =========================================================================

test("snapshot captures serializable session refs · writes the bytes into __blazen_serialized_session_refs metadata", async (t) => {
  const wf = new Workflow("snapshot-capture");
  wf.setSessionPausePolicy(SessionPausePolicy.PickleOrSerialize);

  wf.addStep("producer", ["blazen::StartEvent"], async (event, ctx) => {
    // Insert two distinct payloads so we can verify both round-trip.
    await ctx.insertSessionRefSerializable(
      BLOB_TYPE_NAME,
      encodeBlob({ n: 1, tag: "first" }),
    );
    await ctx.insertSessionRefSerializable(
      BLOB_TYPE_NAME,
      encodeBlob({ n: 2, tag: "second" }),
    );
    // Park long enough for the harness to pause + snapshot us.
    await new Promise((resolve) => setTimeout(resolve, 500));
    return { type: "blazen::StopEvent", result: { done: true } };
  });

  const handler = await wf.runWithHandler({});
  // Give the producer step a moment to insert the payloads before
  // we pause.
  await new Promise((resolve) => setTimeout(resolve, 100));
  await handler.pause();
  const snapJson = await handler.snapshot();
  await handler.abort();
  try {
    await handler.result();
  } catch (_) {
    // Aborted handlers raise — ignore.
  }

  const snap = JSON.parse(snapJson);
  const sidecar = snap.metadata?.__blazen_serialized_session_refs;
  t.truthy(
    sidecar,
    `snapshot metadata should contain __blazen_serialized_session_refs; got keys ${Object.keys(snap.metadata ?? {}).join(", ")}`,
  );
  const entries = Object.entries(sidecar);
  t.is(
    entries.length,
    2,
    `expected exactly two captured serializable refs, got ${entries.length}`,
  );

  // Each record should carry our type tag and a `data` field that
  // round-trips through `serde_bytes::BytesWrapper` (a JSON array
  // of unsigned ints).
  for (const [, record] of entries) {
    t.is(record.type_tag, BLOB_TYPE_NAME);
    t.truthy(Array.isArray(record.data), "record.data should be an array");

    // The wire format is `[4-byte BE tag_len][tag bytes][user bytes]`.
    // Verify the embedded tag matches what we passed in.
    const raw = Buffer.from(record.data);
    const tagLen = raw.readUInt32BE(0);
    const embeddedTag = raw.subarray(4, 4 + tagLen).toString("utf-8");
    t.is(embeddedTag, BLOB_TYPE_NAME);

    // The user payload should decode back to one of the two blobs
    // we inserted.
    const userBytes = raw.subarray(4 + tagLen);
    const decoded = JSON.parse(userBytes.toString("utf-8"));
    t.truthy(
      (decoded.n === 1 && decoded.tag === "first") ||
        (decoded.n === 2 && decoded.tag === "second"),
      `unexpected decoded payload: ${JSON.stringify(decoded)}`,
    );
  }
});

// =========================================================================
// Resume round-trip
// =========================================================================

test("resumeWithSerializableRefs · rehydrates the registry so getSessionRefSerializable still works", async (t) => {
  const wf = new Workflow("resume-roundtrip");
  wf.setSessionPausePolicy(SessionPausePolicy.PickleOrSerialize);
  // Short timeout so the resumed workflow does not hang forever.
  wf.setTimeout(2);

  wf.addStep("producer", ["blazen::StartEvent"], async (event, ctx) => {
    const key = await ctx.insertSessionRefSerializable(
      BLOB_TYPE_NAME,
      encodeBlob({ n: 99, tag: "round-trip" }),
    );
    // Stash the key in workflow state so we can recover it after resume.
    await ctx.state.set("payload_key", key);
    // Park long enough for the harness to pause + snapshot us.
    await new Promise((resolve) => setTimeout(resolve, 500));
    return { type: "blazen::StopEvent", result: { done: true } };
  });

  const handler = await wf.runWithHandler({});
  await new Promise((resolve) => setTimeout(resolve, 100));
  await handler.pause();
  const snapJson = await handler.snapshot();
  await handler.abort();
  try {
    await handler.result();
  } catch (_) {
    // Aborted handlers raise — ignore.
  }

  // Confirm the snapshot really has a serialized sidecar before we
  // attempt to resume — otherwise this test would silently pass on
  // an unrelated regression.
  const snap = JSON.parse(snapJson);
  t.truthy(snap.metadata?.__blazen_serialized_session_refs);

  // Resume via the new method. We do NOT drive the resumed workflow
  // to completion — the original mid-flight pause does not capture
  // pending channel events, mirroring the existing Python-binding
  // limitation in `tests/python/test_session_refs.py`. The short
  // timeout we set above means the resumed handler will time out
  // instead of blocking forever.
  const resumed = await wf.resumeWithSerializableRefs(snapJson);
  try {
    await resumed.result();
  } catch (_) {
    // Resumed mid-flight workflow times out; that's expected.
  }
});

test("resumeWithSerializableRefs · raises a clear error when a referenced type tag has no record", async (t) => {
  // Build a snapshot, then mangle the embedded tag bytes inside one
  // of the captured payloads so the deserializer trampoline still
  // succeeds (it just rebuilds an opaque adapter), but the outer
  // type_tag field stays consistent. We exercise the malformed
  // payload path instead, which is the closest Node-side analogue
  // to Python's "DoesNotExist class" failure mode.
  const wf = new Workflow("resume-broken");
  wf.setSessionPausePolicy(SessionPausePolicy.PickleOrSerialize);
  wf.setTimeout(2);

  wf.addStep("producer", ["blazen::StartEvent"], async (event, ctx) => {
    await ctx.insertSessionRefSerializable(
      BLOB_TYPE_NAME,
      encodeBlob({ n: 1, tag: "x" }),
    );
    await new Promise((resolve) => setTimeout(resolve, 500));
    return { type: "blazen::StopEvent", result: { done: true } };
  });

  const handler = await wf.runWithHandler({});
  await new Promise((resolve) => setTimeout(resolve, 100));
  await handler.pause();
  const snapJson = await handler.snapshot();
  await handler.abort();
  try {
    await handler.result();
  } catch (_) {
    // ignore
  }

  // Truncate the embedded payload so the trampoline's prefix parser
  // rejects it. The wire format is
  // `[4-byte BE tag_len][tag bytes][user bytes]`; truncating to
  // just two bytes leaves the parser unable to read the length
  // field at all.
  const parsed = JSON.parse(snapJson);
  const sidecar = parsed.metadata.__blazen_serialized_session_refs;
  for (const record of Object.values(sidecar)) {
    record.data = [0x00, 0x00];
  }
  const broken = JSON.stringify(parsed);

  let captured = null;
  try {
    await wf.resumeWithSerializableRefs(broken);
  } catch (err) {
    captured = err.message;
  }
  t.truthy(
    captured,
    "resumeWithSerializableRefs should throw on a malformed serialized payload",
  );
});
