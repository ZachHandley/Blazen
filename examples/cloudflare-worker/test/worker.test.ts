import { unstable_dev, type UnstableDevWorker } from "wrangler";
import { afterAll, beforeAll, describe, expect, test } from "vitest";

let worker: UnstableDevWorker;

beforeAll(async () => {
  worker = await unstable_dev("src/worker.ts", {
    experimental: { disableExperimentalWarning: true },
    // local mode is the default; runs in workerd in-process
  });
}, 60_000); // workerd cold-start can take a bit

afterAll(async () => {
  if (worker) await worker.stop();
});

describe("Blazen real workflow engine on Cloudflare Workers", () => {
  test("runSmokeWorkflow fetch handler returns greeting", async () => {
    const resp = await worker.fetch("/");
    expect(resp.status).toBe(200);
    const body = await resp.json();
    expect(body).toEqual({ greeting: "Hello, World!" });
  });

  test("JS-defined 2-step workflow returns greeting", async () => {
    const resp = await worker.fetch("/js-workflow");
    expect(resp.status).toBe(200);
    const body = await resp.json();
    expect(body).toEqual({ greeting: "Hello, From-JS!" });
  });

  // SKIPPED — the `/snapshot-roundtrip` route hangs on workerd because the
  // `runHandler` -> `awaitResult`/`pause`/`runId` flow never resolves once
  // control has returned to JS. The single-shot `run()` path used by the
  // first two tests works fine, so the engine itself executes; the bug is
  // that `wasm_bindgen_futures::spawn_local` (used inside
  // `crates/blazen-core/src/runtime.rs` to drive the event loop on wasm32)
  // is not re-polled by workerd between two separate JS Promise
  // boundaries inside one request. Even probing `handler.runId()` (which
  // only reads metadata, no broadcast wait) hangs, so the deadlock is at
  // the spawn-local layer, not at the broadcast channel.
  //
  // The route is wired up end-to-end in `worker.ts` with both pause-
  // after-completion and mid-flight pause fallbacks, so once the SDK fix
  // lands (keep the spawned event-loop future tied to the same JsFuture
  // chain `runHandler` returns, or expose a "drive one tick" entry point)
  // change `test.skip` -> `test` and the assertions below will validate
  // the round-trip.
  //
  // Confirmed reproducer: `pnpm wrangler dev --local src/worker.ts` then
  // `curl http://localhost:8787/snapshot-roundtrip` returns
  // 500 "Worker's code had hung" after the workerd timeout.
  test("snapshot/resume round-trip works", async () => {
    const resp = await worker.fetch("/snapshot-roundtrip");
    expect(resp.status).toBe(200);
    const body = (await resp.json()) as {
      status: string;
      result1?: { greeting?: string };
      result2?: { greeting?: string };
      pauseAfterComplete?: boolean;
      pauseAfterCompleteError?: string | null;
      midFlightError?: string;
      error?: string;
      snapshotKeys?: string[];
    };

    if (body.status !== "ok") {
      throw new Error(
        `snapshot-roundtrip route returned status=${body.status}: ${JSON.stringify(body)}`,
      );
    }

    expect(body.result1).toEqual({ greeting: "Hello, Snapshot!" });
    // Whichever pause path won, the resumed workflow should reach the same
    // terminal greeting. After pause-after-completion the engine has no
    // pending events and `awaitResult()` returns the previously-recorded
    // stop payload; after a mid-flight pause the engine replays the
    // GreetEvent and produces the same stop payload deterministically.
    expect(body.result2).toEqual({ greeting: "Hello, Snapshot!" });
  });
});
