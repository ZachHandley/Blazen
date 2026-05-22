# BeastPC GPU CI

End-to-end GPU validation for Blazen's audio, 3D, video, image-diffusion, and
candle/mistralrs LLM crates lives in
[`.forgejo/workflows/beastpc-e2e.yaml`](../.forgejo/workflows/beastpc-e2e.yaml).
Tests in this workflow exercise real CUDA kernels and real model weights — they
do not run on the shared Forgejo `ubuntu-latest` pool, which has no GPU. They
run on a labeled self-hosted Forgejo runner physically attached to the BeastPC
rig.

Today there is **one** BeastPC runner: a 4090 24 GB box. An inbound 5090 +
128 GB VRAM rig will register as a second runner with an extra `big-vram`
label so workflows that need >24 GB of VRAM can target it explicitly.

## Runner registration

The workflow targets `runs-on: beastpc`. Register the runner against the
Blazen repo (or org-wide, if you want the same runner to service Blazen plus
sibling repos) with `act_runner register`:

### 4090 24 GB rig (current)

```bash
# On the 4090 host, after installing act_runner and the CUDA toolkit:
act_runner register \
    --instance https://forge.blackleafdigital.com \
    --token "$FORGEJO_RUNNER_TOKEN" \
    --name beastpc-4090 \
    --labels "beastpc,gpu,cuda,linux_amd64"
```

### 5090 + 128 GB rig (inbound)

Identical command, with `big-vram` added to the label set so the large-VRAM
workflows (when they arrive) can route specifically to this box:

```bash
act_runner register \
    --instance https://forge.blackleafdigital.com \
    --token "$FORGEJO_RUNNER_TOKEN" \
    --name beastpc-5090 \
    --labels "beastpc,gpu,cuda,big-vram,linux_amd64"
```

Run `act_runner daemon` (or install the systemd unit shipped with
`act_runner`) so the runner picks up jobs immediately after registration.

## Required system dependencies

The runner machine must have, **before** the first job runs:

- **CUDA toolkit** matching the version expected by candle 0.10 (`candle-core`,
  `candle-nn`, `candle-transformers` are pinned at `0.10` in the workspace
  root `Cargo.toml`). Candle 0.10 builds against the cudarc 0.x line, which
  works against CUDA 12.x — the 4090 / 5090 driver stack is fine here. Install
  via your distro's package manager or the NVIDIA `.run` installer; ensure
  `nvcc --version` resolves and `nvidia-smi` reports the card.
- **libssl-dev** — required by `reqwest`'s native-tls fallback path.
- **libsndfile1** — `hound` and the audio integration tests read/write WAV
  through it.
- **ffmpeg** — used by video smoke tests and several audio fixtures for
  format conversion.
- **Build basics**: `gcc`, `clang`, `pkg-config`, `cmake`. The `blazen-llm-llamacpp`
  feature path pulls in cmake at build time even when its cuda feature is off.
- **Node 22 + pnpm 10** and **Python 3.12 + uv** — both are bootstrapped by the
  workflow itself via the in-house `setup-python` action and `actions/setup-node@v4`,
  but the runner needs sufficient disk for the pnpm store and uv cache.

## Required env / secrets

Provided through Forgejo repo-level secrets — the workflow references the
following:

- **`BLAZEN_HF_TOKEN`** — optional. Required only for gated HF Hub models
  (some bark voices, certain f5-tts checkpoints). Without it, the gated test
  cases will skip and the rest of the run is unaffected.
- **`FAL_KEY`** — optional. The Video E2E step
  (`cargo nextest run --test cloud_video_smoke`) is `if:`-guarded on a non-empty
  `FAL_KEY` so the job does not turn red when the secret is absent.

`BLAZEN_TEST_*` opt-in env vars set by the workflow:

| Variable                          | Unlocks                                                                          |
| --------------------------------- | -------------------------------------------------------------------------------- |
| `BLAZEN_TEST_STABLE_AUDIO=1`      | `blazen-audio-music` Stable Audio integration tests (real diffusion sampling)     |
| `BLAZEN_TEST_BARK=1`              | `blazen-audio-tts` Bark integration tests                                         |
| `BLAZEN_TEST_F5=1`                | `blazen-audio-tts` F5-TTS integration tests                                       |
| `BLAZEN_TEST_RVC=1`               | `blazen-audio-vc` RVC integration tests                                           |
| `BLAZEN_TEST_WHISPER_STREAMING=1` | `blazen-audio-stt` streaming-whisper integration tests                            |

## Test convention

Both Rust and Python tests participating in this workflow follow a single rule:
they must opt-in. The shared CI pool runs them only when explicitly told to.

### Rust

Mark expensive GPU tests `#[ignore]` so they're skipped by the default
`cargo nextest run --workspace`:

```rust
#[test]
#[ignore = "requires CUDA GPU; run on BeastPC"]
fn generates_30s_stable_audio_clip() {
    // ...
}
```

The workflow runs them via `--run-ignored only`. This matches the pattern
already used by `live-models.yaml`.

### Python

Gate via the `BLAZEN_TEST_<NAME>` env var the workflow already sets:

```python
import os
import pytest

@pytest.mark.skipif(
    os.environ.get("BLAZEN_TEST_BARK") != "1",
    reason="gpu-gated; export BLAZEN_TEST_BARK=1 to run",
)
def test_bark_synthesizes_phrase():
    ...
```

Also add the `gpu` keyword to the test name (or apply
`@pytest.mark.gpu`) — the workflow's pytest invocation filters with
`-k gpu` so unrelated tests don't spin up the bark/whisper/RVC pipelines
implicitly.

### Node

Place GPU-gated specs under `tests/node/*_gpu.mjs`. The workflow runs them
through ava with the `_gpu.mjs` glob; the step exits cleanly if no files match
the glob, so adding the first GPU spec doesn't require a workflow edit.

## Manual trigger

Today, manual `workflow_dispatch` is the primary way this workflow runs.

### Forgejo UI

1. Navigate to **Actions → BeastPC E2E (GPU)** in the Forgejo web UI.
2. Click **Run workflow**, pick `main` (or any branch), and confirm.

### `forgejo-cli`

```bash
forgejo-cli workflow run beastpc-e2e.yaml --ref main
```

Or via the Forgejo API directly (mirrors the dispatch pattern already used in
`ci.yaml`'s `auto-tag` job):

```bash
curl -s -X POST \
  -H "Authorization: token $FORGEJO_ACTIONS_TOKEN" \
  -H "Content-Type: application/json" \
  https://forge.blackleafdigital.com/api/v1/repos/BlackLeafDigital/blazen/actions/workflows/beastpc-e2e.yaml/dispatches \
  -d '{"ref": "refs/heads/main"}'
```

## Push triggers

In addition to manual dispatch, the workflow auto-runs on:

- `push` to `main` when any of the following paths change:
  `crates/blazen-audio-*`, `crates/blazen-3d/`, `crates/blazen-image-diffusion/`,
  `crates/blazen-llm-candle/`, `crates/blazen-llm-mistralrs/`, or the workflow
  file itself.
- `push` of a `v*` tag — pre-release validation on the cut tag itself.

## Nightly schedule status

The `schedule:` block at the top of `beastpc-e2e.yaml` is **commented out**
today. Re-enable it in a follow-up commit once the first clean manual run has
landed; otherwise scaffolding bugs would mask themselves as nightly noise. The
intended cron is `0 5 * * *` (05:00 UTC, off the hour so it doesn't collide
with the broader CI hourly pattern).

## Artifacts

Every run uploads three artifact buckets (via
`https://code.forgejo.org/forgejo/upload-artifact@v4`, matching the pattern
used by `build-artifacts.yaml`):

- `artifacts/wav/**` — WAV outputs from audio integration tests.
- `artifacts/glb/**` — GLB outputs from 3D integration tests.
- `artifacts/reports/summary.md` — small Markdown summary with commit, ref,
  runner, and a flat list of artifacts.

Retention: 14 days. If a run produced no artifacts the upload step emits a
warning rather than failing the job (`if-no-files-found: warn`).
