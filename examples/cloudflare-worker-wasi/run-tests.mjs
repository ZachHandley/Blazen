#!/usr/bin/env node
// Boots `wrangler dev` against the wasi worker, hits every probe, prints
// a summary table, and exits with a non-zero code on any unexpected
// failure.
//
// Usage:
//   node run-tests.mjs                 # run all probes
//   node run-tests.mjs --filter=http   # only http-category probes
//   OPENAI_API_KEY=sk-... node run-tests.mjs   # include OpenAI probe
//
// Designed to run from the example directory:
//   cd examples/cloudflare-worker-wasi && node run-tests.mjs

import { spawn } from "node:child_process";
import { setTimeout as wait } from "node:timers/promises";
import { fileURLToPath } from "node:url";
import path from "node:path";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const PORT = process.env.WRANGLER_PORT ?? "8787";
const ORIGIN = `http://127.0.0.1:${PORT}`;
const READY_TIMEOUT_MS = 30_000;
const PROBE_TIMEOUT_MS = 15_000;

const args = process.argv.slice(2);
const filter = args
  .find((a) => a.startsWith("--filter="))
  ?.slice("--filter=".length);

function color(c, s) {
  if (!process.stdout.isTTY) return s;
  const codes = { red: 31, green: 32, yellow: 33, gray: 90, bold: 1 };
  return `\x1b[${codes[c] ?? 0}m${s}\x1b[0m`;
}

async function startWrangler() {
  const wranglerBin = path.join(
    __dirname,
    "node_modules",
    ".bin",
    "wrangler",
  );
  const proc = spawn(
    wranglerBin,
    ["dev", "--port", PORT, "--ip", "127.0.0.1", "--local"],
    {
      cwd: __dirname,
      env: {
        ...process.env,
        // Pass through OPENAI_API_KEY so the openai probe can run.
        OPENAI_API_KEY: process.env.OPENAI_API_KEY ?? "",
      },
      stdio: ["ignore", "pipe", "pipe"],
    },
  );
  let stdoutBuf = "";
  let stderrBuf = "";
  proc.stdout.on("data", (d) => {
    stdoutBuf += d.toString();
  });
  proc.stderr.on("data", (d) => {
    stderrBuf += d.toString();
  });

  const start = Date.now();
  while (Date.now() - start < READY_TIMEOUT_MS) {
    if (stdoutBuf.includes("Ready on http")) return { proc, stdoutBuf, stderrBuf };
    if (proc.exitCode !== null) {
      throw new Error(
        `wrangler dev exited (code ${proc.exitCode}) before ready:\n${stdoutBuf}\n${stderrBuf}`,
      );
    }
    if (
      stderrBuf.includes("ERROR") ||
      stdoutBuf.includes("✘ [ERROR]") ||
      stdoutBuf.includes("runtime failed to start")
    ) {
      // Give it a moment to flush, then bail.
      await wait(500);
      throw new Error(
        `wrangler dev failed to start:\n${stdoutBuf}\n${stderrBuf}`,
      );
    }
    await wait(200);
  }
  throw new Error(
    `wrangler dev didn't become ready within ${READY_TIMEOUT_MS}ms`,
  );
}

async function fetchJson(url, ms = PROBE_TIMEOUT_MS) {
  const ctl = AbortSignal.timeout(ms);
  const res = await fetch(url, { signal: ctl });
  const text = await res.text();
  try {
    return { status: res.status, json: JSON.parse(text) };
  } catch {
    return { status: res.status, raw: text };
  }
}

async function listProbes() {
  const { json } = await fetchJson(`${ORIGIN}/`);
  return json;
}

async function runProbe(name) {
  const { json, raw } = await fetchJson(
    `${ORIGIN}/run/${encodeURIComponent(name)}`,
  );
  return json ?? { name, status: "fail", error: raw ?? "(non-json)" };
}

function fmtCell(s, w) {
  s = String(s);
  if (s.length > w) return s.slice(0, w - 1) + "…";
  return s.padEnd(w, " ");
}

async function main() {
  console.log(color("bold", "blazen wasi · Cloudflare Workers smoke"));

  // dev-e2e.sh manages wrangler externally so it can tee logs to .dev/.
  // When SKIP_WRANGLER=1 we assume the caller already booted a server on
  // WRANGLER_PORT and just hit it.
  const externalWrangler = process.env.SKIP_WRANGLER === "1";
  let cleanup = () => {};
  if (externalWrangler) {
    console.log(`using external wrangler at ${ORIGIN}…`);
  } else {
    console.log(`booting wrangler dev on ${ORIGIN}…`);
    const { proc } = await startWrangler();
    cleanup = () => {
      try {
        proc.kill("SIGTERM");
      } catch {}
    };
  }
  process.on("SIGINT", cleanup);
  process.on("SIGTERM", cleanup);

  try {
    const probes = await listProbes();
    const filtered = filter
      ? probes.filter(
          (p) =>
            p.name.includes(filter) || p.category === filter,
        )
      : probes;

    console.log(
      `running ${filtered.length}/${probes.length} probes` +
        (filter ? ` (filter: ${filter})` : ""),
    );
    console.log("");
    console.log(
      color("bold", fmtCell("probe", 32)) +
        "  " +
        color("bold", fmtCell("category", 9)) +
        "  " +
        color("bold", fmtCell("status", 6)) +
        "  " +
        color("bold", fmtCell("ms", 5)) +
        "  " +
        color("bold", "detail"),
    );
    console.log("─".repeat(96));

    let pass = 0,
      fail = 0,
      skipped = 0;
    const failures = [];

    for (const probe of filtered) {
      const result = await runProbe(probe.name);
      const skipMsg =
        typeof result.result === "object" &&
        result.result !== null &&
        "skipped" in result.result
          ? result.result.skipped
          : null;

      let statusLabel, statusColor, detail;
      if (skipMsg) {
        skipped++;
        statusLabel = "skip";
        statusColor = "gray";
        detail = String(skipMsg);
      } else if (result.status === "pass") {
        pass++;
        statusLabel = "pass";
        statusColor = "green";
        detail = JSON.stringify(result.result ?? {}).slice(0, 60);
      } else {
        fail++;
        statusLabel = "fail";
        statusColor = "red";
        detail = result.error ?? "(unknown)";
        failures.push(result);
      }
      console.log(
        fmtCell(probe.name, 32) +
          "  " +
          fmtCell(probe.category, 9) +
          "  " +
          color(statusColor, fmtCell(statusLabel, 6)) +
          "  " +
          fmtCell(result.durationMs ?? "", 5) +
          "  " +
          detail,
      );
    }

    console.log("");
    console.log(
      color("bold", "summary: ") +
        color("green", `${pass} pass`) +
        ", " +
        color("red", `${fail} fail`) +
        ", " +
        color("gray", `${skipped} skip`),
    );

    if (failures.length > 0) {
      console.log("");
      console.log(color("bold", "failures (full stack):"));
      for (const f of failures) {
        console.log("");
        console.log(color("red", `▸ ${f.name}`));
        console.log(`  error: ${f.error}`);
        if (f.stack) {
          console.log(
            "  " +
              f.stack
                .split("\n")
                .slice(0, 8)
                .map((l) => l.replace(/^/, "  "))
                .join("\n"),
          );
        }
      }
    }

    cleanup();
    process.exit(fail > 0 ? 1 : 0);
  } catch (e) {
    cleanup();
    console.error(color("red", "harness error:"), e?.message ?? e);
    process.exit(2);
  }
}

main();
