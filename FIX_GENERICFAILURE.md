# Fix plan: GenericFailure opacity + smoke-test flakiness + split e2e workflow

## 1. Context

Three related problems surfaced from one failing CI run (`auto-routes to vision when AnyLlm and image part present` in `tests/node/test_fal_smoke.mjs:115`):

1. **Errors from fal.ai / other providers surface as `BlazenError::Request("result fetch failed: {opaque body}")`** which the Node binding renders as `GenericFailure`. A user hitting this in prod — or us debugging it in CI — has no HTTP status, no request id, no endpoint, no structured body. We can't tell if it's our bug, auth, rate-limit, or provider-side transient.
2. **Smoke tests hit real upstream APIs and hard-fail CI on transient provider errors.** The `auto-routes to vision` test is testing our *routing logic*, not fal's ability to describe an image, yet a fal 5xx kills the whole CI run.
3. **CI runs the full test matrix (Rust workspace + Python smoke + Node smoke, each with real API calls) on every push/PR.** Small unrelated commits (workflow edits, docs, Dockerfiles) pay for 15+ minutes of API-backed test time every time. It costs real money on API keys, takes forever, and leaves contributors waiting.

Fixing all three together because they share the same surface: provider error ergonomics + CI gating.

## 2. Structured provider errors (replaces `GenericFailure`)

### Source locations to change

- `crates/blazen-llm/src/providers/fal.rs`
  - Line 1350-1357: `queue_get_result` — opaque `BlazenError::request(format!("result fetch failed: {error_body}"))`.
  - Line 1487-1494: the second result-fetch path, same opaque pattern.
  - Line 1601-1607: streaming path `"streaming request failed"`.
  - Any other `BlazenError::request(format!(...))` call site on a response body; grep `response.text()` in the file.
- `crates/blazen-llm/src/error.rs` (or wherever `BlazenError` is defined — find with `grep -rn 'enum BlazenError' crates/blazen-llm/`) — add a structured variant.
- `crates/blazen-node/src/error.rs` — Node binding needs to map the new variant to a JS error code more specific than `GenericFailure`.
- `crates/blazen-py/src/error.rs` (or wherever the PyO3 mapping lives) — same mapping treatment on the Python side.

### New variant

Add to `BlazenError`:

```rust
/// Upstream provider returned a non-success HTTP response.
///
/// Use this whenever an HTTP call to a provider (fal, OpenRouter, Anthropic,
/// Gemini, etc.) returns !is_success(). Carries enough context that a caller
/// — human or automated — can decide: retry, fail user request, or report
/// bug. Do NOT use for deserialization errors, auth errors (use a dedicated
/// Auth variant), or network-layer failures (use Network).
ProviderHttp {
    provider: &'static str,          // "fal", "openrouter", "anthropic", …
    endpoint: String,                 // full URL, request-id-stripped (no PII)
    status: u16,                      // HTTP status code
    request_id: Option<String>,       // from x-fal-request-id / x-request-id / etc
    detail: Option<String>,           // parsed from body JSON `{"detail": "..."}` etc.
    raw_body: String,                 // full body text (capped, see below)
},
```

Display implementation: `"{provider} {status} at {endpoint}: {detail or first 200 chars of raw_body} (request-id={request_id or 'none'})"`.

### Body size cap

Cap `raw_body` at 4 KiB on ingest; append `"... [truncated N bytes]"` if longer. Otherwise `BlazenError` becomes a DoS vector if a provider sends a 20 MB HTML error page.

### Call-site migration pattern

Replace:
```rust
if !response.is_success() {
    let error_body = response.text();
    return Err(BlazenError::request(format!(
        "result fetch failed: {error_body}"
    )));
}
```
with:
```rust
if !response.is_success() {
    return Err(provider_http_error(
        "fal",
        &result_url,
        response.status(),
        response.header("x-fal-request-id"),
        response.text(),
    ));
}
```

Add `fn provider_http_error(...) -> BlazenError` in `blazen-llm/src/providers/mod.rs` as a small helper that does the 4 KiB cap, pulls `detail` out of body JSON via `serde_json::from_str::<ProviderErrorBody>(&body)` with `#[derive(Deserialize)] struct ProviderErrorBody { detail: Option<String>, error: Option<String>, message: Option<String> }`, and constructs the `ProviderHttp` variant. Every HTTP-based provider in `crates/blazen-llm/src/providers/` should migrate to it — grep `!response.is_success()` across `providers/*.rs` to find them all.

### Binding layer mapping

**Node** (`crates/blazen-node/src/error.rs`):
- New napi error code `ProviderError` with fields `provider`, `status`, `requestId`, `detail` exposed on the JS side.
- Document: JS code can `catch (e)` and do `if (e.code === 'ProviderError' && e.status === 503) retry()`.

**Python** (`crates/blazen-py/src/error.rs`):
- New exception class `BlazenProviderError(BlazenError)` with `provider`, `status`, `request_id`, `detail` attributes.
- Expose via the PyO3 stub: `blazen.pyi` needs the new class.
- Run `cargo run --example stub_gen -p blazen-py` after and commit the stub delta.

### Clippy + tests

- Add a unit test in `crates/blazen-llm/src/providers/mod.rs` (or a new `tests.rs` alongside) that constructs a fake `HttpResponse` with a known body and asserts the resulting `BlazenError::ProviderHttp` parses `detail` out of `{"detail":"rate limited"}` and caps an oversized body.
- Existing callers that matched on `BlazenError::Request` may need updating — grep `BlazenError::Request` and `BlazenError::request` across the workspace.

## 3. Smoke tests tolerate transient upstream failures

### Helper

Add `tests/node/_smoke_helpers.mjs` (pattern already in use — see other `_helpers` files in the repo):

```js
// Retries `fn` up to `attempts` times on errors that smell like upstream
// transients (5xx, generic provider errors, network/timeout). Routing-layer
// logic mistakes (bad request, auth, our own bugs) propagate on first try.
export async function retryOnUpstream(fn, { attempts = 3, baseDelayMs = 1000 } = {}) {
  let lastErr;
  for (let i = 0; i < attempts; i++) {
    try {
      return await fn();
    } catch (e) {
      lastErr = e;
      const transient = (
        (e.code === 'ProviderError' && e.status >= 500) ||
        (typeof e.message === 'string' &&
          /Provider returned error|result fetch failed|timeout|ECONNRESET|ETIMEDOUT|fetch failed/i.test(e.message))
      );
      if (!transient || i === attempts - 1) throw e;
      await new Promise(r => setTimeout(r, baseDelayMs * (2 ** i)));
    }
  }
  throw lastErr;
}
```

And mirror in `tests/python/_smoke_helpers.py`.

### Test updates

Wrap every `.complete(...)` / `.embed(...)` call in `tests/node/test_fal_smoke.mjs`, `tests/node/test_llm_smoke.mjs`, `tests/python/test_fal_smoke.py`, `tests/python/test_llm_smoke.py`:

```js
const response = await retryOnUpstream(() => model.complete([...]));
```

### Decision on failure mode

- **Hard fail after N retries** is the default. Smoke tests that keep failing mean either (a) our code regressed, (b) the provider is down for extended time. Either warrants CI failure.
- Do NOT turn provider errors into test skips — that hides regressions. Retry is enough.
- If a specific provider becomes chronically flaky (e.g. fal.ai during their outages), gate individual tests behind an env var like `SKIP_FAL_SMOKE=1` that CI can set temporarily without code changes.

## 4. Split CI: fast path always, heavy path conditionally

### Goal

Every push/PR still gets:
- `cargo fmt --check` and `cargo clippy` (lint job, exists today).
- Workspace `cargo nextest` (unit + integration, NO API keys) (exists today).
- Python unit tests (no API keys) — `tests/python/test_workflow.py test_session_refs.py test_e2e.py` (already split on line 128).
- Node free tests (no API keys) — `tests/node/test_workflow.mjs test_session_refs.mjs test_capabilities_smoke.mjs test_e2e.mjs` (already split on lines 170-176).
- PyO3 stub drift check (line 116-125).

Only when provider/test code changes, a PR gets:
- Python smoke (`tests/python/test_*_smoke.py`) with real API keys.
- Node smoke (`tests/node/test_*_smoke.mjs`) with real API keys.

### Layout

**Modify `.forgejo/workflows/ci.yaml`:**
- Remove the `Run Python API smoke tests` step from `test-python` (line 130-134).
- Remove the `Run Node.js LLM smoke tests` step from `test-node` (line 178-182).
- Rename the remaining steps in those jobs to reflect "fast tests only".
- `auto-tag` still depends on `[test, test-python, test-node]` — unchanged.

**New `.forgejo/workflows/e2e.yaml`:**

```yaml
name: E2E smoke tests

on:
  push:
    branches: ["main"]
    paths:
      - 'crates/blazen-llm/src/providers/**'
      - 'crates/blazen-llm/src/compute/**'
      - 'crates/blazen-llm/src/fallback.rs'
      - 'crates/blazen-llm/src/routing/**'
      - 'crates/blazen-node/src/providers/**'
      - 'crates/blazen-node/src/error.rs'
      - 'crates/blazen-py/src/providers/**'
      - 'crates/blazen-py/src/error.rs'
      - 'crates/blazen-py/blazen.pyi'
      - 'crates/blazen-node/index.d.ts'
      - 'crates/blazen-node/index.js'
      - 'tests/node/**'
      - 'tests/python/**'
      - '.forgejo/workflows/e2e.yaml'
      - 'Cargo.lock'
      - 'package.json'
      - 'pnpm-lock.yaml'
      - 'uv.lock'
      - 'pyproject.toml'
  pull_request:
    branches: ["main"]
    paths: [ ... same list ... ]
  workflow_dispatch: {}

concurrency:
  group: e2e-${{ github.ref }}
  cancel-in-progress: true

env:
  CARGO_TERM_COLOR: always
  CARGO_INCREMENTAL: "0"
  CARGO_BUILD_JOBS: "4"

jobs:
  smoke-python:
    name: Python smoke (real API)
    runs-on: ubuntu-latest
    steps:
      # Same Setup Rust / Setup Python / uv sync as ci.yaml test-python
      - name: Run Python smoke tests
        run: uv run pytest -v --timeout=300 tests/python/test_fal_smoke.py tests/python/test_llm_smoke.py tests/python/test_capabilities_smoke.py tests/python/test_provider_smoke.py
        env:
          FAL_API_KEY: ${{ secrets.FAL_API_KEY }}
          OPENROUTER_API_KEY: ${{ secrets.OPENROUTER_API_KEY }}

  smoke-node:
    name: Node smoke (real API)
    runs-on: ubuntu-latest
    steps:
      # Same Setup Rust / Setup Node / pnpm install / napi build as ci.yaml test-node
      - name: Run Node smoke tests
        run: node --test tests/node/test_llm_smoke.mjs tests/node/test_fal_smoke.mjs
        env:
          FAL_API_KEY: ${{ secrets.FAL_API_KEY }}
          OPENROUTER_API_KEY: ${{ secrets.OPENROUTER_API_KEY }}
```

Copy the full Setup Rust / Setup Python / Setup Node blocks from `ci.yaml` — same secrets, same sccache config. Factor into a reusable composite action in a later pass if this pattern keeps spreading.

### Path filter discipline

The paths list MUST include the generated stub files (`crates/blazen-py/blazen.pyi`, `crates/blazen-node/index.d.ts`, `crates/blazen-node/index.js`) because:
- A pure Rust-source change that regenerates stubs will trigger e2e.
- A commit that ONLY updates stubs (without touching Rust) also triggers e2e, which is correct — stub change implies ABI/API surface change.

The paths list intentionally OMITS:
- `.forgejo/workflows/ci.yaml` itself (fast CI handles that).
- `.forgejo/workflows/build-artifacts.yaml` / `build-blazen-ci-base.yaml` / `release.yaml` — packaging, not runtime behaviour.
- `ci-base/**` — Dockerfile changes don't change runtime behaviour of the library.
- `website/**` — deployed separately.
- `*.md`, `LICENSE`, `README*`, top-level Makefile, etc.

### Auto-tag dependency

`auto-tag` in `ci.yaml` currently depends on `[test, test-python, test-node]`. With the split, **should it wait for e2e too?** Options:

- **(A) Fast CI only** — `auto-tag` only waits for the fast jobs (recommended for day-to-day dev velocity). Release gets tagged even if e2e didn't run (because nothing smoke-relevant changed — e2e was skipped by path filter).
- **(B) e2e required** — `auto-tag` also waits for `e2e.yaml` completion. But pull `needs` across workflow files isn't natively supported; would require a polling/check step. Complex.

Recommend (A). Rationale: if e2e runs (because relevant code changed) and it fails, it'll show red on the PR/commit, blocking the human from merging. auto-tag only fires on `push: main` — so by the time a change reaches main, it's been reviewed. If e2e is red on main, the operator can revert or hotfix; not auto-tag-blocked. Matches how big monorepos run flaky external-dep tests.

### Manual override

Add an input to `e2e.yaml`'s `workflow_dispatch`:
```yaml
workflow_dispatch:
  inputs:
    skip_path_filter:
      description: "Force-run even if no paths match (useful for runtime-only regressions)"
      type: boolean
      default: false
```
(The filter is a `paths:` concept — for workflow_dispatch it always runs, this input is just for documentation.)

## 5. Rollout order

1. Add the retry helper (`_smoke_helpers.mjs`, `_smoke_helpers.py`) and wrap existing smoke tests. Commit + push. This alone removes the transient-upstream CI failure class.
2. Add the `BlazenError::ProviderHttp` variant + helper + call-site migration in `fal.rs` first. Commit + push. Re-run the original failing test (if it fails, we at least see WHY fal returned an error, not just "generic failure").
3. Migrate the remaining providers in `crates/blazen-llm/src/providers/*.rs` to the helper. One PR-worth per provider (anthropic, openrouter, gemini, etc.) so each stays reviewable.
4. Update `blazen-node` / `blazen-py` error-mapping layers. Regenerate stubs.
5. Split the workflows: add `e2e.yaml`, remove the smoke-test steps from `ci.yaml`. Make the path filters specific.
6. Watch a few commits land. If path filters are too tight (real provider regressions slipping through without triggering e2e) or too loose (unrelated commits still running e2e), tighten/loosen.

## 6. Non-goals

- NOT moving smoke tests into the core unit test suite via mocks. The whole point of a smoke test is to catch "we updated our provider adapter, does fal's API still accept our payload" — mocking that defeats the purpose.
- NOT making provider errors retryable inside the library. That's the caller's choice. Library reports what happened; caller decides policy.
- NOT changing `build-artifacts.yaml` matrix or the CI base image work. Orthogonal.

## 7. Open questions for when this gets executed

- What should `BlazenError::ProviderHttp` variant's `provider` field be: `&'static str`, `String`, or an enum `ProviderId`? Lean `&'static str` unless we already have a `ProviderId` enum in use (grep).
- Does fal.ai actually send `x-fal-request-id`? Check against an observed response before wiring it up; if the real header name differs, fix the helper.
- Do we want a single `ProviderError` exception class on the Python side, or provider-specific subclasses (`BlazenFalError`, `BlazenOpenRouterError`)? Single class with `provider` attribute is simpler; matches our Rust-side choice.
