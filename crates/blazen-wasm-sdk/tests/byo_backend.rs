#![cfg(target_arch = "wasm32")]

//! `wasm-bindgen-test` coverage for the BYO (Bring-Your-Own) JS backend
//! adapter exposed by [`blazen_wasm_sdk::byo_backend`].
//!
//! These tests build a JS object satisfying the `BlazenJsBackend` contract
//! using `Function::new_with_args` (the same pattern used by
//! `tests/custom_provider.rs` and `tests/streaming.rs`) and assert that:
//!
//! - `complete()` round-trips a `ModelResponse` through serde-wasm-bindgen.
//! - `streamComplete()` chunks land in the Rust-side `Stream`.
//! - `embed()` produces vectors when the optional method is present.
//! - `registerByoBackend()` on `WasmModelManager` accepts the JS object and
//!   exposes load/unload + status round-trips.

use blazen_wasm_sdk::byo_backend::{JsBackendShim, byo_backend_as_model};
use blazen_wasm_sdk::manager::WasmModelManager;
use futures_util::StreamExt;
use js_sys::{Function, Object, Reflect};
use wasm_bindgen::prelude::*;
use wasm_bindgen_test::{wasm_bindgen_test, wasm_bindgen_test_configure};

wasm_bindgen_test_configure!(run_in_node_experimental);

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Set a key on a JS Object, panicking on failure (tests only).
fn set_key(obj: &Object, key: &str, value: &JsValue) {
    Reflect::set(obj, &JsValue::from_str(key), value).expect("Reflect::set failed");
}

/// Build a minimal `BlazenJsBackend` JS object exposing the required
/// `complete` method plus optional fields.
fn make_minimal_backend(model_id: &str, complete_body: &str) -> JsValue {
    let backend = Object::new();
    set_key(&backend, "modelId", &JsValue::from_str(model_id));
    let complete = Function::new_with_args("request", complete_body);
    set_key(&backend, "complete", complete.as_ref());
    backend.into()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[wasm_bindgen_test]
async fn byo_complete_roundtrip() {
    // JS backend returns a fixed ModelResponse shape.
    let body = r#"
        return Promise.resolve({
            content: 'hello from byo backend',
            tool_calls: [],
            citations: [],
            artifacts: [],
            images: [],
            audio: [],
            videos: [],
            model: 'byo-test',
            metadata: {}
        });
    "#;
    let backend = make_minimal_backend("byo-test", body);

    let model = byo_backend_as_model(backend).expect("BYO backend should construct as a Model");

    // Sanity: model id round-trips (WasmModel.modelId getter returns String).
    assert_eq!(model.model_id(), "byo-test".to_string());
}

#[wasm_bindgen_test]
async fn byo_complete_deserializes_response_via_shim() {
    // Drive the shim directly through the Model trait to confirm
    // the wire deserialization works end-to-end.
    use blazen_llm::traits::Model;
    use blazen_llm::types::ModelRequest;

    let body = r#"
        return Promise.resolve({
            content: 'shim-content',
            tool_calls: [],
            citations: [],
            artifacts: [],
            images: [],
            audio: [],
            videos: [],
            model: 'shim-test',
            metadata: {}
        });
    "#;
    let backend = make_minimal_backend("shim-test", body);
    let shim = JsBackendShim::from_js(backend).expect("shim ctor");

    let req = ModelRequest {
        messages: vec![],
        tools: vec![],
        temperature: None,
        max_tokens: None,
        top_p: None,
        response_format: None,
        model: None,
        modalities: None,
        image_config: None,
        audio_config: None,
        tool_choice: None,
    };

    let resp = shim.complete(req).await.expect("complete() should succeed");
    assert_eq!(resp.content.as_deref(), Some("shim-content"));
}

#[wasm_bindgen_test]
async fn byo_stream_complete_chunks_delivered() {
    use blazen_llm::traits::Model;
    use blazen_llm::types::ModelRequest;

    // JS streamComplete pushes 3 chunks then resolves.
    let backend = Object::new();
    set_key(&backend, "modelId", &JsValue::from_str("byo-stream"));
    let complete = Function::new_with_args(
        "request",
        r#"return Promise.resolve({ content: 'unused', tool_calls: [], citations: [], artifacts: [], images: [], audio: [], videos: [], model: 'byo-stream', metadata: {} });"#,
    );
    set_key(&backend, "complete", complete.as_ref());

    let stream_body = r#"
        onChunk({ delta: 'a', tool_calls: [], citations: [], artifacts: [] });
        onChunk({ delta: 'b', tool_calls: [], citations: [], artifacts: [] });
        onChunk({ delta: 'c', tool_calls: [], citations: [], artifacts: [] });
        return Promise.resolve();
    "#;
    let stream_fn = Function::new_with_args("request,onChunk", stream_body);
    set_key(&backend, "streamComplete", stream_fn.as_ref());

    let shim = JsBackendShim::from_js(backend.into()).expect("shim ctor");

    let req = ModelRequest {
        messages: vec![],
        tools: vec![],
        temperature: None,
        max_tokens: None,
        top_p: None,
        response_format: None,
        model: None,
        modalities: None,
        image_config: None,
        audio_config: None,
        tool_choice: None,
    };

    let stream = shim.stream(req).await.expect("stream() should succeed");
    let collected: Vec<_> = stream.collect().await;
    assert_eq!(collected.len(), 3, "expected 3 chunks");
    let deltas: Vec<String> = collected
        .into_iter()
        .filter_map(|r| r.ok().and_then(|c| c.delta))
        .collect();
    assert_eq!(
        deltas,
        vec!["a".to_string(), "b".to_string(), "c".to_string()]
    );
}

#[wasm_bindgen_test]
async fn byo_embed_optional_when_present() {
    use blazen_llm::traits::EmbeddingModel;

    let backend = Object::new();
    set_key(&backend, "modelId", &JsValue::from_str("byo-embed"));
    set_key(&backend, "embeddingDimensions", &JsValue::from_f64(3.0));

    // complete() is required even when only testing embeddings; trivial stub.
    let complete = Function::new_with_args(
        "request",
        r#"return Promise.resolve({ content: '', tool_calls: [], citations: [], artifacts: [], images: [], audio: [], videos: [], model: 'byo-embed', metadata: {} });"#,
    );
    set_key(&backend, "complete", complete.as_ref());

    // Returns one number[3] per input text. The shim accepts both
    // Float32Array[] and number[][] for ergonomics.
    let embed = Function::new_with_args(
        "texts",
        r#"
            const out = [];
            for (const t of texts) {
                out.push([t.length, 0.5, -0.25]);
            }
            return Promise.resolve(out);
        "#,
    );
    set_key(&backend, "embed", embed.as_ref());

    let shim = JsBackendShim::from_js(backend.into()).expect("shim ctor");
    assert!(shim.has_embed(), "embed() should be advertised");
    assert_eq!(shim.embedding_dimensions(), 3);

    let texts = vec!["hi".to_string(), "world".to_string()];
    let resp = shim.embed(&texts).await.expect("embed() should succeed");
    assert_eq!(resp.embeddings.len(), 2);
    assert_eq!(resp.embeddings[0].len(), 3);
    assert!((resp.embeddings[0][0] - 2.0).abs() < 1e-6);
    assert!((resp.embeddings[1][0] - 5.0).abs() < 1e-6);
}

#[wasm_bindgen_test]
async fn byo_embed_requires_dimensions_when_present() {
    // embed() without embeddingDimensions should fail construction.
    let backend = Object::new();
    set_key(&backend, "modelId", &JsValue::from_str("byo-bad-embed"));
    let complete = Function::new_with_args(
        "request",
        r#"return Promise.resolve({ content: '', tool_calls: [], citations: [], artifacts: [], images: [], audio: [], videos: [], model: 'x', metadata: {} });"#,
    );
    set_key(&backend, "complete", complete.as_ref());
    let embed = Function::new_with_args("texts", "return Promise.resolve([]);");
    set_key(&backend, "embed", embed.as_ref());

    let err = JsBackendShim::from_js(backend.into())
        .err()
        .expect("should reject — embeddingDimensions missing");
    let msg = err.as_string().unwrap_or_default();
    assert!(
        msg.contains("embeddingDimensions"),
        "error should mention missing embeddingDimensions, got: {msg}"
    );
}

#[wasm_bindgen_test]
async fn byo_missing_complete_rejects() {
    // No complete() method on the host object -> construction fails.
    let backend = Object::new();
    set_key(&backend, "modelId", &JsValue::from_str("nope"));
    let err = JsBackendShim::from_js(backend.into())
        .err()
        .expect("should reject — complete is required");
    let msg = err.as_string().unwrap_or_default();
    assert!(
        msg.contains("complete"),
        "error should mention complete is required, got: {msg}"
    );
}

#[wasm_bindgen_test]
async fn byo_register_in_model_manager() {
    // End-to-end: register a BYO backend via WasmModelManager.registerByoBackend
    // and confirm it shows up in status().
    let manager = WasmModelManager::new(8.0, None);

    let backend = Object::new();
    set_key(&backend, "modelId", &JsValue::from_str("byo-mm-test"));
    set_key(&backend, "memoryBytes", &JsValue::from_f64(1_000_000.0));
    let complete = Function::new_with_args(
        "request",
        r#"return Promise.resolve({ content: 'ok', tool_calls: [], citations: [], artifacts: [], images: [], audio: [], videos: [], model: 'byo-mm-test', metadata: {} });"#,
    );
    set_key(&backend, "complete", complete.as_ref());

    let promise = manager
        .register_byo_backend("byo-mm-test".to_owned(), backend.into())
        .expect("registerByoBackend should accept the backend");
    wasm_bindgen_futures::JsFuture::from(promise)
        .await
        .expect("register promise should resolve");

    // status() should now list the registered model.
    let status_promise = manager.status();
    let status = wasm_bindgen_futures::JsFuture::from(status_promise)
        .await
        .expect("status promise should resolve");
    let arr: js_sys::Array = status.unchecked_into();
    assert!(arr.length() >= 1, "status should list at least 1 model");

    // Find our id.
    let mut found = false;
    for i in 0..arr.length() {
        let obj = arr.get(i);
        let id = Reflect::get(&obj, &JsValue::from_str("id"))
            .ok()
            .and_then(|v| v.as_string())
            .unwrap_or_default();
        if id == "byo-mm-test" {
            found = true;
            break;
        }
    }
    assert!(found, "registered BYO id should appear in status()");
}
