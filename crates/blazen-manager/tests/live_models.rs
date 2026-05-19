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
