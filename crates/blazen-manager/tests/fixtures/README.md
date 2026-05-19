# Live-models test fixtures

The `live_models.rs` integration tests are gated behind `--features live-models`
and `--ignored` (they download multi-hundred-MB model weights, so they must
never run on default CI).

## Cache layout

Tests cache downloads under `~/.cache/blazen-tests/`:

```
~/.cache/blazen-tests/
├── Qwen--Qwen2.5-0.5B-Instruct-GGUF/
│   └── qwen2.5-0.5b-instruct-q4_k_m.gguf   (~400 MB, downloaded by all four tests)
└── loras/
    └── test-lora/                          (Test 4 only)
        ├── adapter_model.safetensors
        └── adapter_config.json
```

Path under `~/.cache/blazen-tests/` is canonical per the project's no-`/tmp`
rule — temp dirs would force re-download per CI run, costing bandwidth and
wall-clock time.

## Models

- **Base GGUF**: `Qwen/Qwen2.5-0.5B-Instruct-GGUF`, file
  `qwen2.5-0.5b-instruct-q4_k_m.gguf`. Chosen because it's the smallest
  GGUF-quantised Qwen instruct that mistral.rs / candle / llama.cpp all
  accept; ~400 MB compressed, well under 4 GB RAM working set when loaded.

- **LoRA adapter**: not committed to this repo. The CI runner must
  populate `~/.cache/blazen-tests/loras/test-lora/` with a rank-4 or
  rank-8 LoRA trained on Qwen2.5-0.5B (target_modules covering at least
  `q_proj`, `v_proj`). Any published Qwen2.5-0.5B-Instruct LoRA from
  HuggingFace satisfies this. If the directory is missing, Test 4 emits
  `eprintln!` explaining and returns early instead of failing.

## Running locally

```bash
cargo nextest run -p blazen-manager --features live-models -- --ignored --test-threads 1
```

`--test-threads 1` because each test loads ~500 MB into RAM; concurrent
runs would OOM small CI runners.

## Running on Forgejo

Push a PR with the `live-models` label, or trigger
`.forgejo/workflows/live-models.yaml` via `workflow_dispatch`. The job is
NOT a required check on default PRs and only runs when explicitly
requested.
