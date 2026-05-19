# Live-models test fixtures

The `live_models.rs` integration tests are gated behind `--features live-models`
and `--ignored` (they download multi-hundred-MB model weights, so they must
never run on default CI).

## Cache layout

Tests cache downloads under `~/.cache/blazen-tests/`:

```
~/.cache/blazen-tests/
├── Qwen--Qwen2.5-0.5B-Instruct-GGUF/
│   └── qwen2.5-0.5b-instruct-q4_k_m.gguf   (~400 MB, downloaded by mistralrs/candle/llamacpp infer tests)
├── Qwen--Qwen2.5-0.5B-Instruct/             (Wave E candle safetensors tests)
│   ├── config.json
│   ├── tokenizer.json
│   └── model.safetensors                   (~1 GB)
└── loras/
    └── test-lora/                          (Wave E adapter tests + Test 4)
        ├── adapter_model.safetensors       (PEFT canonical; candle + mistralrs path)
        ├── adapter_config.json             (PEFT canonical; candle + mistralrs path)
        ├── adapter_model.gguf              (llama.cpp path; produced by convert_lora_to_gguf.py)
        └── adapter_model_alt.gguf          (second GGUF for the multi-adapter test)
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
  HuggingFace satisfies this. If the directory is missing, the adapter
  tests `eprintln!` a skip notice and return early instead of failing.

- **Auto-staging (Wave E adapter tests)**: instead of dropping files in
  by hand, you can point the suite at HuggingFace repos via env vars and
  the test harness will download into `loras/test-lora/` on first run:

  | env var | meaning |
  | --- | --- |
  | `BLAZEN_TEST_LORA_REPO` | HF repo id holding `adapter_model.safetensors` + `adapter_config.json` (PEFT canonical). Consumed by `live_mistralrs_load_adapter`, `live_candle_load_adapter_safetensors`, `live_candle_load_adapter_inference_delta`. |
  | `BLAZEN_TEST_LORA_GGUF_REPO` + `BLAZEN_TEST_LORA_GGUF_FILE` | HF repo id + filename of the llama.cpp-compatible GGUF LoRA. Downloaded as `adapter_model.gguf`. Consumed by `live_llamacpp_load_adapter` and (as the first adapter) `live_llamacpp_multi_adapter`. |
  | `BLAZEN_TEST_LORA_GGUF_ALT_REPO` + `BLAZEN_TEST_LORA_GGUF_ALT_FILE` | Second GGUF LoRA used only by `live_llamacpp_multi_adapter`. Downloaded as `adapter_model_alt.gguf`. |

  No HF repo is hard-coded because publishing a curated Qwen2.5-0.5B LoRA
  pair is out of scope for the blazen repo — operators bring their own.
  To produce a GGUF LoRA from a PEFT adapter, run llama.cpp's
  `convert_lora_to_gguf.py`.

- **Candle safetensors base**: `Qwen/Qwen2.5-0.5B-Instruct` (HF). Pulled
  via `ModelCache` on first run; tokenizer + config + single
  safetensors shard, ~1 GB. The candle Wave E tests opt into the
  non-quantized loader path via `CandleLlmOptions::force_safetensors = true`.

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
