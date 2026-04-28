/**
 * Shared helpers for Node smoke tests.
 *
 * Mirrors `_fal_or_skip` in tests/python/test_fal_smoke.py so Node and
 * Python tests treat fal queue / HTTP transient errors and poll-timeout
 * diagnostics the same way: as test skips, not failures.
 */

import assert from "node:assert/strict";

const FAL_TRANSIENT_MARKERS = [
  "404",
  "502",
  "503",
  "504",
  "Bad Gateway",
  "service_unavailable",
  "No endpoints available",
  "poll timeout after",
];

/**
 * Await `promise`. If it rejects with a message matching a known fal
 * transient/infrastructure marker, call `t.skip(reason)` and return
 * `undefined`. Otherwise rethrow.
 *
 * @param {{ skip: (reason?: string) => void }} t  Node test context (the
 *   first argument to `it(name, async (t) => { ... })`).
 * @param {Promise<T>} promise
 * @returns {Promise<T | undefined>}
 * @template T
 */
export async function falOrSkip(t, promise) {
  try {
    return await promise;
  } catch (e) {
    const msg = String(e?.message ?? e);
    const marker = FAL_TRANSIENT_MARKERS.find((m) => msg.includes(m));
    if (marker) {
      t.skip(`fal transient (${marker}): ${msg}`);
      return undefined;
    }
    throw e;
  }
}

/**
 * Await `promise`. If it resolves, return the value. If it rejects, assert
 * that the rejection message contains at least one of the routing-failure
 * markers — proving the request reached the right fal endpoint and the
 * failure was a fal-side file-fetch error, not a routing regression.
 *
 * @param {Promise<T>} promise
 * @param {string[]} markers
 * @returns {Promise<T | undefined>}
 * @template T
 */
export async function expectFalRoutingError(promise, markers) {
  try {
    return await promise;
  } catch (e) {
    const msg = String(e?.message ?? e);
    assert.ok(
      markers.some((m) => msg.includes(m)),
      `unexpected error (routing may have failed): ${msg}`,
    );
    return undefined;
  }
}
