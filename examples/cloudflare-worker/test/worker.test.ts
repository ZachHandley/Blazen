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

  // Exercises `runHandler` + `awaitResult` + `pause` + single-shot `run`
  // across multiple JS Promise boundaries. Validates the workerd
  // `setTimeout(0)` yield workaround in both handler methods and
  // `WasmWorkflow::run` — without it, every await in this route hangs at
  // workerd's "code had hung" timeout. See `crate::handler::yield_to_js`.
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
