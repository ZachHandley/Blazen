//! Real-weights integration tests for the local-inference backends.
//!
//! Opt-in: gated on the `live-models` feature and `#[ignore]` per-test.
//! Run with:
//!
//! ```bash
//! cargo test -p blazen-manager --features live-models -- --ignored
//! ```
//!
//! Tests download a small (~400 MB) Qwen2.5-0.5B-Instruct GGUF into
//! `~/.cache/blazen-tests/` (idempotent — re-runs use the cache) and
//! exercise the full `ModelManager` → `LocalModel` → engine path for each
//! of the three local backends: mistral.rs, candle, and llama.cpp.
//!
//! Test 4 additionally exercises the `LoRA` adapter mount path on mistral.rs.
//! It looks for a pre-staged PEFT adapter directory at
//! `~/.cache/blazen-tests/loras/test-lora/` containing
//! `adapter_model.safetensors` + `adapter_config.json`. If the directory
//! is missing the test prints a skip notice and returns early — no public
//! Qwen2.5-0.5B `LoRA` is reliably available at a stable URL, so operators
//! are expected to drop their own compatible adapter into the path.
//!
//! Tests 5-8 (Wave E of PR3) cover the llama.cpp + candle `load_adapter`
//! wirings: they require an `adapter_model.gguf` (for llama.cpp) and the
//! PEFT pair (for candle) in the same `test-lora/` directory. To stage
//! adapters automatically the suite also honours these env vars:
//!
//! * `BLAZEN_TEST_LORA_REPO` — HF repo id of a PEFT-format Qwen2.5-0.5B
//!   adapter (downloads `adapter_model.safetensors` + `adapter_config.json`
//!   into `~/.cache/blazen-tests/loras/test-lora/`).
//! * `BLAZEN_TEST_LORA_GGUF_REPO` + `BLAZEN_TEST_LORA_GGUF_FILE` — HF
//!   repo id + filename of a GGUF-converted Qwen2.5-0.5B `LoRA`
//!   (downloaded into the same directory as `adapter_model.gguf`).
//!
//! Each test skips with a clear `eprintln!` if its required fixtures are
//! missing.

#![cfg(feature = "live-models")]

use std::path::PathBuf;
use std::sync::Arc;

use blazen_llm::{AdapterMountStrategy, AdapterOptions, LocalModel};
use blazen_llm_candle::{CandleLlmOptions, CandleLlmProvider};
use blazen_llm_llamacpp::{
    ChatMessageInput as LlamaChatMessageInput, ChatRole as LlamaChatRole, LlamaCppOptions,
    LlamaCppProvider,
};
use blazen_llm_mistralrs::{
    ChatMessageInput as MistralChatMessageInput, ChatRole as MistralChatRole, MistralRsOptions,
    MistralRsProvider,
};
use blazen_manager::ModelManager;
use blazen_manager::hf_loader::{BackendHint, HfLoadOptions};
use blazen_model_cache::ModelCache;

const QWEN_REPO: &str = "Qwen/Qwen2.5-0.5B-Instruct-GGUF";
const QWEN_FILE: &str = "qwen2.5-0.5b-instruct-q4_k_m.gguf";
const QWEN_MEM_ESTIMATE_BYTES: u64 = 1_073_741_824; // 1 GB conservative

fn cache_dir() -> PathBuf {
    PathBuf::from(std::env::var("HOME").expect("HOME must be set"))
        .join(".cache")
        .join("blazen-tests")
}

fn lora_dir() -> PathBuf {
    cache_dir().join("loras").join("test-lora")
}

async fn download_qwen_gguf() -> PathBuf {
    let cache = ModelCache::with_dir(cache_dir());
    cache
        .download(QWEN_REPO, QWEN_FILE, None)
        .await
        .expect("download Qwen GGUF")
}

#[tokio::test]
#[ignore = "live-models: downloads ~400 MB Qwen GGUF and runs real inference"]
async fn live_mistralrs_load_infer_unload() {
    let gguf_path = download_qwen_gguf().await;

    let provider = MistralRsProvider::from_options(MistralRsOptions {
        model_id: gguf_path.to_string_lossy().into_owned(),
        ..MistralRsOptions::required(gguf_path.to_string_lossy().into_owned())
    })
    .expect("construct mistralrs provider");
    let provider = Arc::new(provider);

    let mgr = ModelManager::with_budgets_gb(8.0, 0.0);
    mgr.register(
        "qwen",
        Arc::clone(&provider) as Arc<dyn LocalModel>,
        QWEN_MEM_ESTIMATE_BYTES,
    )
    .await;

    mgr.load("qwen").await.expect("manager load");
    assert!(mgr.is_loaded("qwen").await, "model should be loaded");

    let result = provider
        .infer(vec![MistralChatMessageInput::text(
            MistralChatRole::User,
            "Reply with one word: hi",
        )])
        .await
        .expect("mistralrs infer");
    let content = result.content.expect("content present");
    assert!(!content.trim().is_empty(), "inference produced empty text");

    mgr.unload("qwen").await.expect("manager unload");
    assert!(
        !mgr.is_loaded("qwen").await,
        "model should be unloaded after unload()"
    );
}

#[tokio::test]
#[ignore = "live-models: downloads ~400 MB Qwen GGUF and runs real inference"]
async fn live_candle_load_infer_unload() {
    let gguf_path = download_qwen_gguf().await;

    let provider = CandleLlmProvider::from_options(CandleLlmOptions {
        model_id: Some(gguf_path.to_string_lossy().into_owned()),
        device: Some("cpu".into()),
        quantization: Some("q4_k_m".into()),
        ..CandleLlmOptions::default()
    })
    .expect("construct candle provider");
    let provider = Arc::new(provider);

    let mgr = ModelManager::with_budgets_gb(8.0, 0.0);
    mgr.register(
        "qwen",
        Arc::clone(&provider) as Arc<dyn LocalModel>,
        QWEN_MEM_ESTIMATE_BYTES,
    )
    .await;

    mgr.load("qwen").await.expect("manager load");
    assert!(mgr.is_loaded("qwen").await, "model should be loaded");

    let result = provider
        .infer(
            vec![("user".to_owned(), "Reply with one word: hi".to_owned())],
            Some(32),
            None,
            None,
        )
        .await
        .expect("candle infer");
    assert!(
        !result.content.trim().is_empty(),
        "inference produced empty text"
    );

    mgr.unload("qwen").await.expect("manager unload");
    assert!(
        !mgr.is_loaded("qwen").await,
        "model should be unloaded after unload()"
    );
}

#[tokio::test]
#[ignore = "live-models: downloads ~400 MB Qwen GGUF and runs real inference"]
async fn live_llamacpp_load_infer_unload() {
    let gguf_path = download_qwen_gguf().await;

    let provider = LlamaCppProvider::from_options(LlamaCppOptions {
        model_path: Some(gguf_path.to_string_lossy().into_owned()),
        device: Some("cpu".into()),
        ..LlamaCppOptions::default()
    })
    .await
    .expect("construct llamacpp provider");
    let provider = Arc::new(provider);

    let mgr = ModelManager::with_budgets_gb(8.0, 0.0);
    mgr.register(
        "qwen",
        Arc::clone(&provider) as Arc<dyn LocalModel>,
        QWEN_MEM_ESTIMATE_BYTES,
    )
    .await;

    mgr.load("qwen").await.expect("manager load");
    assert!(mgr.is_loaded("qwen").await, "model should be loaded");

    let result = provider
        .infer(vec![LlamaChatMessageInput::new(
            LlamaChatRole::User,
            "Reply with one word: hi",
        )])
        .await
        .expect("llamacpp infer");
    let content = result.content.expect("content present");
    assert!(!content.trim().is_empty(), "inference produced empty text");

    mgr.unload("qwen").await.expect("manager unload");
    assert!(
        !mgr.is_loaded("qwen").await,
        "model should be unloaded after unload()"
    );
}

#[tokio::test]
#[ignore = "live-models: requires operator-staged PEFT LoRA at ~/.cache/blazen-tests/loras/test-lora/"]
async fn live_mistralrs_load_adapter() {
    let adapter_dir = lora_dir();
    let required_files = [
        adapter_dir.join("adapter_model.safetensors"),
        adapter_dir.join("adapter_config.json"),
    ];
    if !required_files.iter().all(|p| p.is_file()) {
        eprintln!(
            "SKIP live_mistralrs_load_adapter: no Qwen2.5-0.5B-compatible LoRA staged at {}.\n\
             Drop a PEFT-format adapter (adapter_model.safetensors + adapter_config.json) into \
             that directory to enable this test.",
            adapter_dir.display()
        );
        return;
    }

    let gguf_path = download_qwen_gguf().await;

    let provider = MistralRsProvider::from_options(MistralRsOptions {
        model_id: gguf_path.to_string_lossy().into_owned(),
        ..MistralRsOptions::required(gguf_path.to_string_lossy().into_owned())
    })
    .expect("construct mistralrs provider");
    let provider = Arc::new(provider);

    let mgr = ModelManager::with_budgets_gb(8.0, 0.0);
    mgr.register(
        "qwen",
        Arc::clone(&provider) as Arc<dyn LocalModel>,
        QWEN_MEM_ESTIMATE_BYTES,
    )
    .await;

    mgr.load("qwen").await.expect("manager load");

    let handle = mgr
        .load_adapter("qwen", &adapter_dir, AdapterOptions::new("test-lora"))
        .await
        .expect("manager load_adapter");

    assert_eq!(handle.adapter_id, "test-lora");
    assert_eq!(handle.mount_strategy, AdapterMountStrategy::Rebuilt);

    let list = mgr.list_adapters("qwen").await.expect("list_adapters");
    assert_eq!(list.len(), 1, "exactly one adapter should be mounted");

    let result = provider
        .infer(vec![MistralChatMessageInput::text(
            MistralChatRole::User,
            "Reply with one word: hello",
        )])
        .await
        .expect("mistralrs infer post-adapter");
    let content = result.content.expect("content present");
    assert!(
        !content.trim().is_empty(),
        "post-adapter inference produced empty text"
    );

    mgr.unload_adapter("qwen", "test-lora")
        .await
        .expect("manager unload_adapter");

    let list = mgr.list_adapters("qwen").await.expect("list_adapters");
    assert!(list.is_empty(), "adapter list should be empty after unload");
}

// ---------------------------------------------------------------------------
// Wave E (PR3): llama.cpp + candle LoRA load_adapter coverage.
//
// The next four tests exercise the just-landed `load_adapter` paths on
// the llama.cpp (Attached, multi-adapter) and candle (Rebuilt) backends.
// All four download the same Qwen2.5-0.5B base model plus a small LoRA
// adapter fixture and skip gracefully when fixtures are unavailable.
// ---------------------------------------------------------------------------

/// HF repo id of a non-quantized Qwen2.5-0.5B-Instruct (safetensors weights
/// + tokenizer + config) — used by the candle safetensors path.
const QWEN_SAFETENSORS_REPO: &str = "Qwen/Qwen2.5-0.5B-Instruct";

fn lora_gguf_path() -> std::path::PathBuf {
    lora_dir().join("adapter_model.gguf")
}

fn lora_gguf_alt_path() -> std::path::PathBuf {
    lora_dir().join("adapter_model_alt.gguf")
}

/// Try to fetch a PEFT-canonical adapter into `lora_dir()` from the
/// `BLAZEN_TEST_LORA_REPO` env var. Returns `true` if the two canonical
/// files end up on disk (either pre-staged or freshly downloaded).
async fn ensure_peft_adapter_staged() -> bool {
    let dir = lora_dir();
    let cfg = dir.join("adapter_config.json");
    let weights = dir.join("adapter_model.safetensors");
    if cfg.is_file() && weights.is_file() {
        return true;
    }
    let Ok(repo) = std::env::var("BLAZEN_TEST_LORA_REPO") else {
        return false;
    };
    if let Err(e) = tokio::fs::create_dir_all(&dir).await {
        eprintln!("failed to create lora dir {}: {e}", dir.display());
        return false;
    }
    let cache = ModelCache::with_dir(cache_dir());
    for filename in ["adapter_config.json", "adapter_model.safetensors"] {
        match cache.download(&repo, filename, None).await {
            Ok(src) => {
                let dst = dir.join(filename);
                if !dst.is_file()
                    && let Err(e) = tokio::fs::copy(&src, &dst).await
                {
                    eprintln!("failed to copy {} -> {}: {e}", src.display(), dst.display());
                    return false;
                }
            }
            Err(e) => {
                eprintln!("failed to download {repo}/{filename}: {e}");
                return false;
            }
        }
    }
    cfg.is_file() && weights.is_file()
}

/// Try to fetch a llama.cpp-compatible GGUF `LoRA` adapter. Returns the
/// downloaded `adapter_model.gguf` path, or `None` if neither the
/// pre-staged file nor `BLAZEN_TEST_LORA_GGUF_REPO` /
/// `BLAZEN_TEST_LORA_GGUF_FILE` were available.
async fn ensure_lora_gguf_staged() -> Option<std::path::PathBuf> {
    let target = lora_gguf_path();
    if target.is_file() {
        return Some(target);
    }
    let repo = std::env::var("BLAZEN_TEST_LORA_GGUF_REPO").ok()?;
    let file = std::env::var("BLAZEN_TEST_LORA_GGUF_FILE").ok()?;
    if let Err(e) = tokio::fs::create_dir_all(lora_dir()).await {
        eprintln!("failed to create lora dir: {e}");
        return None;
    }
    let cache = ModelCache::with_dir(cache_dir());
    match cache.download(&repo, &file, None).await {
        Ok(src) => {
            if !target.is_file()
                && let Err(e) = tokio::fs::copy(&src, &target).await
            {
                eprintln!("failed to stage GGUF lora: {e}");
                return None;
            }
            Some(target)
        }
        Err(e) => {
            eprintln!("failed to download GGUF lora {repo}/{file}: {e}");
            None
        }
    }
}

/// Try to fetch a second llama.cpp-compatible GGUF `LoRA` for the
/// multi-adapter test. Same env protocol as
/// [`ensure_lora_gguf_staged`] but with the `_ALT` suffix.
async fn ensure_lora_gguf_alt_staged() -> Option<std::path::PathBuf> {
    let target = lora_gguf_alt_path();
    if target.is_file() {
        return Some(target);
    }
    let repo = std::env::var("BLAZEN_TEST_LORA_GGUF_ALT_REPO").ok()?;
    let file = std::env::var("BLAZEN_TEST_LORA_GGUF_ALT_FILE").ok()?;
    if let Err(e) = tokio::fs::create_dir_all(lora_dir()).await {
        eprintln!("failed to create lora dir: {e}");
        return None;
    }
    let cache = ModelCache::with_dir(cache_dir());
    match cache.download(&repo, &file, None).await {
        Ok(src) => {
            if !target.is_file()
                && let Err(e) = tokio::fs::copy(&src, &target).await
            {
                eprintln!("failed to stage second GGUF lora: {e}");
                return None;
            }
            Some(target)
        }
        Err(e) => {
            eprintln!("failed to download second GGUF lora {repo}/{file}: {e}");
            None
        }
    }
}

#[tokio::test]
#[ignore = "live-models: downloads Qwen GGUF + GGUF LoRA adapter and runs real inference"]
async fn live_llamacpp_load_adapter() {
    let Some(adapter_gguf) = ensure_lora_gguf_staged().await else {
        eprintln!(
            "SKIP live_llamacpp_load_adapter: no Qwen2.5-0.5B-compatible GGUF LoRA at {}.\n\
             Either stage one manually or set BLAZEN_TEST_LORA_GGUF_REPO + \
             BLAZEN_TEST_LORA_GGUF_FILE. To convert a PEFT safetensors adapter to GGUF, \
             run llama.cpp's `convert_lora_to_gguf.py`.",
            lora_gguf_path().display()
        );
        return;
    };

    let gguf_path = download_qwen_gguf().await;

    let provider = LlamaCppProvider::from_options(LlamaCppOptions {
        model_path: Some(gguf_path.to_string_lossy().into_owned()),
        device: Some("cpu".into()),
        ..LlamaCppOptions::default()
    })
    .await
    .expect("construct llamacpp provider");
    let provider = Arc::new(provider);

    let mgr = ModelManager::with_budgets_gb(8.0, 0.0);
    mgr.register(
        "qwen",
        Arc::clone(&provider) as Arc<dyn LocalModel>,
        QWEN_MEM_ESTIMATE_BYTES,
    )
    .await;
    mgr.load("qwen").await.expect("manager load");

    let handle = mgr
        .load_adapter("qwen", &adapter_gguf, AdapterOptions::new("lora-a"))
        .await
        .expect("llamacpp load_adapter");
    assert_eq!(handle.adapter_id, "lora-a");
    assert_eq!(
        handle.mount_strategy,
        AdapterMountStrategy::Attached,
        "llama.cpp mounts adapters via hot-attach, not a rebuild",
    );

    let list = mgr.list_adapters("qwen").await.expect("list_adapters");
    assert_eq!(list.len(), 1, "exactly one adapter should be mounted");

    let with_adapter = provider
        .infer(vec![LlamaChatMessageInput::new(
            LlamaChatRole::User,
            "Reply with one word: hi",
        )])
        .await
        .expect("infer with adapter");
    let with_content = with_adapter.content.unwrap_or_default();
    assert!(
        !with_content.trim().is_empty(),
        "with-adapter inference produced empty text",
    );
    // Why: 5-token bound is contractual per the task spec — guard the
    // sampler against runaway generation by checking total tokens.
    assert!(
        with_adapter.usage.completion_tokens > 0,
        "expected some tokens generated under adapter"
    );

    mgr.unload_adapter("qwen", "lora-a")
        .await
        .expect("unload_adapter");
    let list = mgr.list_adapters("qwen").await.expect("list_adapters");
    assert!(list.is_empty(), "adapter list empty after unload");

    let post_unload = provider
        .infer(vec![LlamaChatMessageInput::new(
            LlamaChatRole::User,
            "Reply with one word: hi",
        )])
        .await
        .expect("infer after unload");
    assert!(
        !post_unload.content.unwrap_or_default().trim().is_empty(),
        "post-unload inference produced empty text — adapter cleanup likely broke the context",
    );
}

#[tokio::test]
#[ignore = "live-models: downloads Qwen GGUF + two GGUF LoRA adapters and runs real inference"]
async fn live_llamacpp_multi_adapter() {
    let Some(adapter_a) = ensure_lora_gguf_staged().await else {
        eprintln!(
            "SKIP live_llamacpp_multi_adapter: missing primary GGUF LoRA at {}. \
             See live_llamacpp_load_adapter skip notice for env-var setup.",
            lora_gguf_path().display()
        );
        return;
    };
    let Some(adapter_b) = ensure_lora_gguf_alt_staged().await else {
        eprintln!(
            "SKIP live_llamacpp_multi_adapter: missing second GGUF LoRA at {}. \
             Set BLAZEN_TEST_LORA_GGUF_ALT_REPO + BLAZEN_TEST_LORA_GGUF_ALT_FILE to enable.",
            lora_gguf_alt_path().display()
        );
        return;
    };

    let gguf_path = download_qwen_gguf().await;

    let provider = LlamaCppProvider::from_options(LlamaCppOptions {
        model_path: Some(gguf_path.to_string_lossy().into_owned()),
        device: Some("cpu".into()),
        ..LlamaCppOptions::default()
    })
    .await
    .expect("construct llamacpp provider");
    let provider = Arc::new(provider);

    let mgr = ModelManager::with_budgets_gb(8.0, 0.0);
    mgr.register(
        "qwen",
        Arc::clone(&provider) as Arc<dyn LocalModel>,
        QWEN_MEM_ESTIMATE_BYTES,
    )
    .await;
    mgr.load("qwen").await.expect("manager load");

    mgr.load_adapter("qwen", &adapter_a, AdapterOptions::new("lora-a"))
        .await
        .expect("load adapter a");
    mgr.load_adapter("qwen", &adapter_b, AdapterOptions::new("lora-b"))
        .await
        .expect("load adapter b");

    let list = mgr.list_adapters("qwen").await.expect("list_adapters");
    assert_eq!(
        list.len(),
        2,
        "both adapters should be active via the raw multi-adapter FFI; got {list:?}",
    );
    let ids: std::collections::HashSet<_> = list.iter().map(|s| s.adapter_id.as_str()).collect();
    assert!(ids.contains("lora-a"));
    assert!(ids.contains("lora-b"));

    // Why: smoke-test that an inference call with two adapters mounted
    // actually succeeds — the multi-adapter raw-FFI wiring lives on
    // `apply_adapters_to_context` which is only exercised at inference.
    let result = provider
        .infer(vec![LlamaChatMessageInput::new(
            LlamaChatRole::User,
            "Reply with one word: hi",
        )])
        .await
        .expect("infer with two adapters mounted");
    assert!(
        !result.content.unwrap_or_default().trim().is_empty(),
        "multi-adapter inference produced empty text — raw-FFI wiring likely regressed",
    );
}

#[tokio::test]
#[ignore = "live-models: downloads Qwen safetensors (~1 GB) + PEFT LoRA and runs real inference"]
async fn live_candle_load_adapter_safetensors() {
    if !ensure_peft_adapter_staged().await {
        eprintln!(
            "SKIP live_candle_load_adapter_safetensors: no PEFT adapter staged at {}. \
             Stage manually or set BLAZEN_TEST_LORA_REPO to a Qwen2.5-0.5B PEFT repo.",
            lora_dir().display()
        );
        return;
    }
    let adapter_dir = lora_dir();

    let provider = CandleLlmProvider::from_options(CandleLlmOptions {
        model_id: Some(QWEN_SAFETENSORS_REPO.into()),
        device: Some("cpu".into()),
        cache_dir: Some(cache_dir()),
        force_safetensors: true,
        ..CandleLlmOptions::default()
    })
    .expect("construct candle provider");
    let provider = Arc::new(provider);

    let mgr = ModelManager::with_budgets_gb(8.0, 0.0);
    mgr.register(
        "qwen",
        Arc::clone(&provider) as Arc<dyn LocalModel>,
        QWEN_MEM_ESTIMATE_BYTES,
    )
    .await;
    mgr.load("qwen").await.expect("manager load (safetensors)");

    let handle = mgr
        .load_adapter("qwen", &adapter_dir, AdapterOptions::new("peft-a"))
        .await
        .expect("candle load_adapter");
    assert_eq!(handle.adapter_id, "peft-a");
    assert_eq!(
        handle.mount_strategy,
        AdapterMountStrategy::Rebuilt,
        "candle safetensors path rebuilds the engine on each adapter change",
    );

    let result = provider
        .infer(
            vec![("user".into(), "Reply with one word: hi".into())],
            Some(8),
            Some(0.0),
            None,
        )
        .await
        .expect("infer with adapter");
    assert!(
        !result.content.trim().is_empty(),
        "with-adapter inference produced empty text",
    );

    mgr.unload_adapter("qwen", "peft-a")
        .await
        .expect("unload_adapter");
    let list = mgr.list_adapters("qwen").await.expect("list_adapters");
    assert!(list.is_empty(), "adapter list empty after unload");

    // Why: rebuild path must produce a usable base-only engine after
    // unload, not a half-torn-down handle.
    let post = provider
        .infer(
            vec![("user".into(), "Reply with one word: hi".into())],
            Some(8),
            Some(0.0),
            None,
        )
        .await
        .expect("infer after unload");
    assert!(
        !post.content.trim().is_empty(),
        "post-unload base inference produced empty text — rebuild likely broke the engine",
    );
}

#[tokio::test]
#[ignore = "live-models: downloads Qwen safetensors (~1 GB) + PEFT LoRA; greedy-decode A/B compare"]
async fn live_candle_load_adapter_inference_delta() {
    if !ensure_peft_adapter_staged().await {
        eprintln!(
            "SKIP live_candle_load_adapter_inference_delta: no PEFT adapter staged at {}. \
             Stage manually or set BLAZEN_TEST_LORA_REPO to a Qwen2.5-0.5B PEFT repo.",
            lora_dir().display()
        );
        return;
    }
    let adapter_dir = lora_dir();

    let provider = CandleLlmProvider::from_options(CandleLlmOptions {
        model_id: Some(QWEN_SAFETENSORS_REPO.into()),
        device: Some("cpu".into()),
        cache_dir: Some(cache_dir()),
        force_safetensors: true,
        ..CandleLlmOptions::default()
    })
    .expect("construct candle provider");
    let provider = Arc::new(provider);

    let mgr = ModelManager::with_budgets_gb(8.0, 0.0);
    mgr.register(
        "qwen",
        Arc::clone(&provider) as Arc<dyn LocalModel>,
        QWEN_MEM_ESTIMATE_BYTES,
    )
    .await;
    mgr.load("qwen").await.expect("manager load (safetensors)");

    // Why: temperature 0.0 collapses the sampler to greedy decoding so
    // repeated calls with the same prompt produce identical token sequences,
    // which is what makes a deterministic A/B diff meaningful.
    let prompt = vec![("user".into(), "Reply with one word: hi".into())];

    let baseline = provider
        .infer(prompt.clone(), Some(10), Some(0.0), None)
        .await
        .expect("baseline infer without adapter");
    let baseline_text = baseline.content.clone();

    mgr.load_adapter("qwen", &adapter_dir, AdapterOptions::new("delta-test"))
        .await
        .expect("load_adapter");

    let with_adapter = provider
        .infer(prompt, Some(10), Some(0.0), None)
        .await
        .expect("infer with adapter");
    let with_text = with_adapter.content.clone();

    assert!(
        !baseline_text.trim().is_empty() && !with_text.trim().is_empty(),
        "both runs must produce non-empty output (baseline={baseline_text:?}, with={with_text:?})",
    );
    assert_ne!(
        baseline_text, with_text,
        "LoRA delta must change greedy-decoded output; identical text means the adapter \
         weights never reached the forward pass (baseline={baseline_text:?}, with={with_text:?})",
    );
}

// ---------------------------------------------------------------------------
// PR4 Wave 3: ModelManager::load_from_hf auto-detect + hint-override coverage.
//
// These tests hit the HF metadata endpoint (no weight download) to confirm
// the auto-detection rules in `hf_loader::choose_backend` line up with the
// real shape of canonical Qwen2.5-0.5B repos. Each one builds a fresh
// `ModelManager`, calls `load_from_hf`, and skips gracefully when the HF
// metadata fetch fails for reasons unrelated to the assertion (DNS, rate
// limit, timeout, gateway error, …).
// ---------------------------------------------------------------------------

/// Why: distinguish HF-network failures (which should skip) from genuine
/// regressions (which should fail). The substrings cover the surface that
/// `hf-hub`/`reqwest` and the manager's own wrappers stamp into the error
/// chain for transport problems, gateway errors, and HF rate-limiting.
fn is_network_failure(err: &blazen_llm::BlazenError) -> bool {
    let msg = err.to_string().to_ascii_lowercase();
    [
        "dns",
        "timed out",
        "timeout",
        "connection",
        "connect error",
        "network",
        "unreachable",
        "tls",
        "handshake",
        "rate limit",
        "too many requests",
        "429",
        "502",
        "503",
        "504",
    ]
    .iter()
    .any(|needle| msg.contains(needle))
}

#[tokio::test]
#[ignore = "live-models: hits huggingface.co metadata API to auto-detect a GGUF repo"]
async fn live_load_from_hf_autodetect_gguf() {
    let mgr = ModelManager::with_budgets_gb(8.0, 0.0);
    let res = mgr
        .load_from_hf(
            "test".into(),
            "Qwen/Qwen2.5-0.5B-Instruct-GGUF",
            HfLoadOptions::default(),
        )
        .await;
    match res {
        Ok(backend) => assert_eq!(
            backend,
            BackendHint::Llamacpp,
            "GGUF-only repo must auto-detect llamacpp; got {backend:?}",
        ),
        Err(e) if is_network_failure(&e) => {
            eprintln!(
                "[skip] live_load_from_hf_autodetect_gguf requires network access to huggingface.co: {e}",
            );
        }
        Err(e) => panic!("load_from_hf failed for a non-network reason: {e}"),
    }
}

#[tokio::test]
#[ignore = "live-models: hits huggingface.co metadata API to auto-detect a safetensors repo"]
async fn live_load_from_hf_autodetect_safetensors() {
    let mgr = ModelManager::with_budgets_gb(8.0, 0.0);
    let res = mgr
        .load_from_hf(
            "test".into(),
            "Qwen/Qwen2.5-0.5B-Instruct",
            HfLoadOptions::default(),
        )
        .await;
    match res {
        Ok(backend) => assert_eq!(
            backend,
            BackendHint::Mistralrs,
            "safetensors-only repo must auto-detect mistralrs; got {backend:?}",
        ),
        Err(e) if is_network_failure(&e) => {
            eprintln!(
                "[skip] live_load_from_hf_autodetect_safetensors requires network access to huggingface.co: {e}",
            );
        }
        Err(e) => panic!("load_from_hf failed for a non-network reason: {e}"),
    }
}

#[tokio::test]
#[ignore = "live-models: hits huggingface.co metadata API and forces a backend via hint"]
async fn live_load_from_hf_backend_hint_overrides() {
    let mgr = ModelManager::with_budgets_gb(8.0, 0.0);
    let res = mgr
        .load_from_hf(
            "test".into(),
            "Qwen/Qwen2.5-0.5B-Instruct",
            HfLoadOptions {
                backend_hint: Some(BackendHint::Candle),
                ..HfLoadOptions::default()
            },
        )
        .await;
    match res {
        Ok(backend) => assert_eq!(
            backend,
            BackendHint::Candle,
            "explicit backend_hint must override auto-detection; got {backend:?}",
        ),
        Err(e) if is_network_failure(&e) => {
            eprintln!(
                "[skip] live_load_from_hf_backend_hint_overrides requires network access to huggingface.co: {e}",
            );
        }
        Err(e) => panic!("load_from_hf failed for a non-network reason: {e}"),
    }
}
