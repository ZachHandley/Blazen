/**
 * Streaming-download tests for the Node.js `ContentStore` binding.
 *
 * Exercises the `ContentStore.fetchStream` wrapper across the three
 * construction paths (custom callback, `ContentStore` subclass, and
 * the built-in `localFile` backend) without touching any external
 * service. Covers correctness, error propagation, and a basic
 * backpressure smoke check.
 *
 * The runtime expectation is:
 *
 *     const iter = await store.fetchStream(handle);
 *     for await (const chunk of iter) {
 *         ...
 *     }
 */

import test from "ava";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { ContentStore } from "../../crates/blazen-node/index.js";

// =========================================================================
// Helpers
// =========================================================================

/** A throwaway handle the custom-store callbacks below ignore. */
function placeholderHandle() {
  return { id: "test-handle", kind: "other" };
}

/** Drain an async iterable into an array of Buffers. */
async function drain(iter) {
  const chunks = [];
  for await (const chunk of iter) {
    chunks.push(Buffer.from(chunk));
  }
  return chunks;
}

// =========================================================================
// 1. Custom store with bytes-returning fetchStream (legacy shape).
// =========================================================================

test("ContentStore.fetchStream · custom store returning Buffer collapses to one chunk", async (t) => {
  const expected = Buffer.from("hello world");

  const store = ContentStore.custom({
    put: async (_body, _hint) => placeholderHandle(),
    resolve: async (_handle) => ({ type: "url", url: "https://example.invalid/blob" }),
    fetchBytes: async (_handle) => expected,
    fetchStream: async (_handle) => expected,
    name: "bytes-stream",
  });

  const handle = await store.put(Buffer.from("placeholder"), {});
  const iter = await store.fetchStream(handle);
  const chunks = await drain(iter);

  t.is(chunks.length, 1);
  t.is(chunks[0].toString("utf8"), "hello world");
});

// =========================================================================
// 2. Custom store with AsyncIterable-returning fetchStream (new shape).
// =========================================================================

test("ContentStore.fetchStream · custom store returning async generator yields chunk-by-chunk", async (t) => {
  const payloadChunks = [
    Buffer.from("hello "),
    Buffer.from("big "),
    Buffer.from("world"),
  ];

  const store = ContentStore.custom({
    put: async (_body, _hint) => placeholderHandle(),
    resolve: async (_handle) => ({ type: "url", url: "https://example.invalid/blob" }),
    fetchBytes: async (_handle) => Buffer.concat(payloadChunks),
    fetchStream: async (_handle) => {
      async function* gen() {
        for (const chunk of payloadChunks) {
          yield new Uint8Array(chunk);
        }
      }
      return gen();
    },
    name: "gen-stream",
  });

  const handle = await store.put(Buffer.from("placeholder"), {});
  const iter = await store.fetchStream(handle);
  const chunks = await drain(iter);

  t.is(chunks.length, 3);
  t.is(Buffer.concat(chunks).toString("utf8"), "hello big world");
  t.is(
    chunks.reduce((n, c) => n + c.length, 0),
    "hello big world".length,
  );
});

// =========================================================================
// 3. Subclass with async-generator fetchStream override.
// =========================================================================

test("ContentStore.fetchStream · subclass async-generator override yields each chunk", async (t) => {
  class MyStore extends ContentStore {
    constructor() {
      super();
    }

    async put(_body, _hint) {
      return placeholderHandle();
    }

    async resolve(_handle) {
      return { type: "url", url: "https://example.invalid/blob" };
    }

    async fetchBytes(_handle) {
      return Buffer.from("ab");
    }

    async *fetchStream(_handle) {
      yield new Uint8Array(Buffer.from("a"));
      yield new Uint8Array(Buffer.from("b"));
    }
  }

  const store = new MyStore();
  const handle = placeholderHandle();
  const iter = await store.fetchStream(handle);
  const chunks = await drain(iter);

  t.is(chunks.length, 2);
  t.is(chunks[0].toString("utf8"), "a");
  t.is(chunks[1].toString("utf8"), "b");
});

// =========================================================================
// 4. Built-in localFile round-trip.
// =========================================================================

test("ContentStore.fetchStream · localFile round-trip preserves the full payload", async (t) => {
  const root = mkdtempSync(join(tmpdir(), "blazen-fetch-stream-"));
  t.teardown(() => rmSync(root, { recursive: true, force: true }));

  const payloadSize = 100_000;
  const payload = Buffer.alloc(payloadSize, 1);

  const store = ContentStore.localFile(root);
  const handle = await store.put(payload, {});

  const iter = await store.fetchStream(handle);
  const chunks = await drain(iter);

  const total = chunks.reduce((n, c) => n + c.length, 0);
  t.is(total, payloadSize);
  t.true(Buffer.concat(chunks).equals(payload));
});

// =========================================================================
// 5. Error path — async generator throws mid-iteration.
// =========================================================================

test("ContentStore.fetchStream · errors raised mid-stream propagate to the consumer", async (t) => {
  class FailingStore extends ContentStore {
    constructor() {
      super();
    }

    async put(_body, _hint) {
      return placeholderHandle();
    }

    async resolve(_handle) {
      return { type: "url", url: "https://example.invalid/blob" };
    }

    async fetchBytes(_handle) {
      throw new Error("not used");
    }

    async *fetchStream(_handle) {
      yield new Uint8Array(Buffer.from("first chunk"));
      throw new Error("boom inside fetchStream");
    }
  }

  const store = new FailingStore();
  const handle = placeholderHandle();
  const iter = await store.fetchStream(handle);

  const collected = [];
  const error = await t.throwsAsync(async () => {
    for await (const chunk of iter) {
      collected.push(Buffer.from(chunk));
    }
  });

  // Don't pin the concrete error class — the bridge may wrap it as
  // ProviderError / generic Error depending on dispatch path. We do
  // assert the underlying message bubbled through and that the first
  // chunk arrived before the failure.
  t.truthy(error);
  t.regex(String(error.message ?? error), /boom inside fetchStream/);
  t.is(collected.length, 1);
  t.is(collected[0].toString("utf8"), "first chunk");
});

// =========================================================================
// 6. Backpressure smoke — many small chunks, slow consumer, ordered delivery.
// =========================================================================

test("ContentStore.fetchStream · backpressure smoke (32 × 16 KiB chunks, slow consumer)", async (t) => {
  const chunkCount = 32;
  const chunkSize = 16 * 1024;
  const expectedChunks = Array.from({ length: chunkCount }, (_, i) =>
    Buffer.alloc(chunkSize, i % 256),
  );

  const store = ContentStore.custom({
    put: async (_body, _hint) => placeholderHandle(),
    resolve: async (_handle) => ({ type: "url", url: "https://example.invalid/blob" }),
    fetchBytes: async (_handle) => Buffer.concat(expectedChunks),
    fetchStream: async (_handle) => {
      async function* gen() {
        for (const chunk of expectedChunks) {
          yield new Uint8Array(chunk);
          // Cooperative yield so producer and consumer interleave.
          await new Promise((resolve) => setImmediate(resolve));
        }
      }
      return gen();
    },
    name: "backpressure",
  });

  const handle = await store.put(Buffer.from("placeholder"), {});
  const iter = await store.fetchStream(handle);

  const received = [];
  for await (const chunk of iter) {
    received.push(Buffer.from(chunk));
    // Cooperative consumer pause — exercises the bounded pipe between
    // the JS-side producer and the Rust-side stream wrapper.
    await new Promise((resolve) => setImmediate(resolve));
  }

  t.is(received.length, chunkCount);
  t.is(
    received.reduce((n, c) => n + c.length, 0),
    chunkCount * chunkSize,
  );
  for (let i = 0; i < chunkCount; i += 1) {
    t.true(
      received[i].equals(expectedChunks[i]),
      `chunk ${i} mismatched (out-of-order or corrupted)`,
    );
  }
});
