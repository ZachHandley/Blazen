#![cfg(target_arch = "wasm32")]

//! `wasm-bindgen-test` coverage for [`blazen_wasm_sdk::providers::custom`].
//!
//! Verifies all four `CustomProvider` factories construct and surface
//! the expected `providerId`, and exercises the
//! [`WasmCustomProviderAdapter`] JS-dispatch path via
//! `WasmCustomProvider::from_js_object`:
//!
//! - Tests 1-3 check that `ollama` / `lmStudio` / `openaiCompat`
//!   produce providers with the expected `providerId` strings.
//! - Test 4 wraps a JS object that exposes a `textToSpeech` method
//!   returning a `Promise<AudioResult>` shape and confirms the typed
//!   `text_to_speech` call round-trips the stub payload back through
//!   `serde_wasm_bindgen` (i.e. the adapter wires the JS-side method up
//!   to the Rust trait dispatch).
//! - Test 5 wraps a JS object that does NOT implement `generateImage`
//!   and asserts the Rust call surfaces an `Unsupported`-flavoured
//!   error.
//!
//! Method bodies are written as raw JS source via
//! `Function::new_with_args` to mirror what host applications actually
//! do (avoids `Closure::wrap` boilerplate; matches `tests/streaming.rs`).

use blazen_wasm_sdk::providers::custom::WasmCustomProvider;
use blazen_wasm_sdk::providers::openai_compat::WasmOpenAiCompatConfig;
use js_sys::{Function, Object, Reflect};
use wasm_bindgen::prelude::*;
use wasm_bindgen_test::{wasm_bindgen_test, wasm_bindgen_test_configure};

// Match the rest of the wasm-sdk test suite (smoke.rs, streaming.rs,
// e2e_parity.rs). `run_in_browser` would let us assert the JS-dispatch
// path under a real DOM, but the existing local test infrastructure
// (and CI's `wasm-pack test --node --release` invocation in
// `.forgejo/workflows/ci.yaml`) is Node-only.
wasm_bindgen_test_configure!(run_in_node_experimental);

// ---------------------------------------------------------------------------
// Factory smoke tests — providerId round-trips through wasm-bindgen.
// ---------------------------------------------------------------------------

#[wasm_bindgen_test]
fn ollama_factory_constructs_provider() {
    let provider = WasmCustomProvider::ollama("llama3".to_owned(), None, None);
    assert_eq!(provider.provider_id(), "ollama");
}

#[wasm_bindgen_test]
fn lm_studio_factory_works() {
    let provider = WasmCustomProvider::lm_studio("mistral".to_owned(), None, None);
    assert_eq!(provider.provider_id(), "lm_studio");
}

#[wasm_bindgen_test]
fn openai_compat_factory_works() {
    let cfg = WasmOpenAiCompatConfig::new(
        "my-llm",
        "https://api.example.test/v1",
        "sk-test",
        "gpt-4o-mini",
    );
    let provider = WasmCustomProvider::openai_compat("my-llm".to_owned(), &cfg);
    assert_eq!(provider.provider_id(), "my-llm");
}

// ---------------------------------------------------------------------------
// from_js_object — typed method routes through the JS prototype chain.
// ---------------------------------------------------------------------------

#[wasm_bindgen_test]
async fn from_js_object_routes_text_to_speech_to_js_method() {
    // Stub `textToSpeech` returns a Promise resolving to an AudioResult
    // shape. Field names match the Rust struct's serde defaults
    // (`serde_wasm_bindgen` preserves the source casing — no rename_all
    // is applied on AudioResult / GeneratedAudio / MediaOutput).
    let tts_body = r"
        return Promise.resolve({
            audio: [{
                media: {
                    url: 'https://example.test/tts.mp3',
                    media_type: { type: 'mp3' },
                    metadata: {}
                }
            }],
            timing: {},
            audio_seconds: 1.5,
            metadata: { stub: true }
        });
    ";
    let tts_fn = Function::new_with_args("req", tts_body);

    let host = Object::new();
    Reflect::set(&host, &JsValue::from_str("textToSpeech"), tts_fn.as_ref()).unwrap();

    let provider = WasmCustomProvider::from_js_object("stub-tts".to_owned(), host);
    assert_eq!(provider.provider_id(), "stub-tts");

    // Build the SpeechRequest JS object (matches the serde shape).
    let req = Object::new();
    Reflect::set(
        &req,
        &JsValue::from_str("text"),
        &JsValue::from_str("hello"),
    )
    .unwrap();
    Reflect::set(
        &req,
        &JsValue::from_str("parameters"),
        &Object::new().into(),
    )
    .unwrap();

    let result_js = provider
        .text_to_speech(req.into())
        .await
        .expect("text_to_speech should resolve via the JS stub");

    // The result must be an Object with an `audio` array of length 1.
    let audio_js = Reflect::get(&result_js, &JsValue::from_str("audio"))
        .expect("result should expose `audio`");
    let audio_arr: js_sys::Array = audio_js.unchecked_into();
    assert_eq!(audio_arr.length(), 1, "stub returned one audio clip");

    // audio_seconds round-trips as a number.
    let seconds = Reflect::get(&result_js, &JsValue::from_str("audio_seconds"))
        .expect("result should expose `audio_seconds`")
        .as_f64()
        .expect("audio_seconds is a number");
    assert!((seconds - 1.5).abs() < 1e-6);
}

#[wasm_bindgen_test]
async fn from_js_object_missing_method_returns_unsupported() {
    // Host object has no `generateImage` method. The adapter must
    // short-circuit to `BlazenError::Unsupported` BEFORE attempting
    // any serde marshaling, and the WasmCustomProvider wrapper must
    // surface that error as a JsValue containing the word "unsupported".
    let host = Object::new();
    let provider = WasmCustomProvider::from_js_object("empty-host".to_owned(), host);
    assert_eq!(provider.provider_id(), "empty-host");

    let req = Object::new();
    Reflect::set(
        &req,
        &JsValue::from_str("prompt"),
        &JsValue::from_str("a corgi"),
    )
    .unwrap();
    Reflect::set(
        &req,
        &JsValue::from_str("parameters"),
        &Object::new().into(),
    )
    .unwrap();

    let err = provider
        .generate_image(req.into())
        .await
        .expect_err("generate_image must error when JS host lacks the method");
    let msg = err.as_string().unwrap_or_default();
    assert!(
        msg.to_lowercase().contains("unsupported"),
        "error should mention `unsupported`, got: {msg}"
    );
}
