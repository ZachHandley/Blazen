#![cfg(target_arch = "wasm32")]

//! Verifies the WASM SDK preserves the original `JsValue` rejection
//! from a tool handler through the `BlazenError::CallerError` path.
//! When a JS tool handler throws / rejects with a custom Error subclass,
//! the original JsValue (with its prototype chain) survives the agent
//! loop's error propagation and is re-thrown verbatim by the wasm-bindgen
//! entrypoint.

use blazen_llm::BlazenError;
use send_wrapper::SendWrapper;
use wasm_bindgen::JsValue;
use wasm_bindgen_test::{wasm_bindgen_test, wasm_bindgen_test_configure};

wasm_bindgen_test_configure!(run_in_browser);

#[wasm_bindgen_test]
fn caller_error_preserves_jsvalue_via_send_wrapper() {
    // Simulate a JS error captured by the agent's tool-handler wrapper.
    let original = JsValue::from_str("simulated JS Error.message");

    let err = BlazenError::caller_error("tool handler threw", SendWrapper::new(original.clone()));

    // Downcast path matches the binding's `blazen_error_to_jsvalue` helper.
    if let BlazenError::CallerError {
        source: Some(s), ..
    } = &err
    {
        let recovered = s
            .downcast_ref::<SendWrapper<JsValue>>()
            .expect("source should be SendWrapper<JsValue>");
        let recovered_value: &JsValue = &**recovered;
        // JsValue equality via string round-trip (it's a primitive here).
        assert_eq!(
            recovered_value.as_string(),
            original.as_string(),
            "recovered JsValue should match the original"
        );
    } else {
        panic!("expected BlazenError::CallerError");
    }
}

#[wasm_bindgen_test]
fn caller_error_helper_returns_original_jsvalue() {
    // The full helper in `blazen_wasm_sdk::agent::blazen_error_to_jsvalue`
    // takes a `BlazenError` and returns a `JsValue` for entrypoint Err
    // returns. Verify the round-trip preserves the original JsValue.
    let original = JsValue::from_str("custom-error-marker");
    let err = BlazenError::caller_error("test", SendWrapper::new(original.clone()));

    let recovered = blazen_wasm_sdk::agent::blazen_error_to_jsvalue(err);
    assert_eq!(recovered.as_string(), original.as_string());
}

#[wasm_bindgen_test]
fn caller_error_helper_falls_back_for_non_caller_variants() {
    let err = BlazenError::tool_error("plain message");
    let js = blazen_wasm_sdk::agent::blazen_error_to_jsvalue(err);
    // Falls back to `JsValue::from_str(&err.to_string())`.
    let s = js.as_string().unwrap_or_default();
    assert!(s.contains("plain message"));
}
