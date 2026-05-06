#![cfg(target_arch = "wasm32")]

//! Minimal wasm-bindgen-test smoke check for the public `Workflow` JS API.
//!
//! The full end-to-end test that exercises `addStep` + `run` with JS
//! callbacks lives in `examples/cloudflare-worker/test/worker.test.ts`
//! (vitest on workerd). Constructing JS `Closure`s from Rust just to
//! re-run the same scenario is awkward and adds no coverage, so this
//! Rust-side test only verifies that `WasmWorkflow` instantiates and the
//! `name` getter round-trips through wasm-bindgen.

use blazen_wasm_sdk::workflow::WasmWorkflow;
use wasm_bindgen_test::{wasm_bindgen_test, wasm_bindgen_test_configure};

wasm_bindgen_test_configure!(run_in_node_experimental);

#[wasm_bindgen_test]
async fn js_api_workflow_instantiates_with_name() {
    let wf = WasmWorkflow::new("smoke");
    assert_eq!(wf.name(), "smoke");
}

// ---------------------------------------------------------------------------
// WasmModelManager construction smoke tests.
//
// These verify only that the post-refactor `register()` signature accepts
// the new arg shapes:
//   - `model: Option<JsValue>` (was a required `JsValue`)
//   - `lifecycle` with optional `isLoaded` / `vramBytes` callbacks alongside
//     the required `load` / `unload`.
//
// Per the convention noted at the top of this file, full callback-driven
// lifecycle exercises live in the JS-hosted suites (cloudflare-worker
// vitest, ava-on-node). These Rust-side tests are intentionally minimal.
// ---------------------------------------------------------------------------

#[wasm_bindgen_test]
async fn js_api_model_manager_registers_with_lifecycle() {
    use blazen_wasm_sdk::manager::WasmModelManager;
    use js_sys::{Function, Object, Reflect};
    use wasm_bindgen::prelude::*;

    let mgr = WasmModelManager::new(8.0); // 8 GB budget

    // Lifecycle object carrying the required `load` / `unload` plus the
    // optional `isLoaded` / `vramBytes` callbacks. Bodies are no-ops; we
    // only care that registration accepts the shape.
    let lifecycle = Object::new();
    let load = Function::new_no_args("return Promise.resolve();");
    let unload = Function::new_no_args("return Promise.resolve();");
    let is_loaded = Function::new_no_args("return false;");
    let vram_bytes = Function::new_no_args("return 1000000000;");
    Reflect::set(&lifecycle, &JsValue::from_str("load"), load.as_ref()).unwrap();
    Reflect::set(&lifecycle, &JsValue::from_str("unload"), unload.as_ref()).unwrap();
    Reflect::set(
        &lifecycle,
        &JsValue::from_str("isLoaded"),
        is_loaded.as_ref(),
    )
    .unwrap();
    Reflect::set(
        &lifecycle,
        &JsValue::from_str("vramBytes"),
        vram_bytes.as_ref(),
    )
    .unwrap();

    // The new register signature: (id, model: Option<JsValue>, vram_estimate, lifecycle).
    // Pass `None` for the model to exercise the `Option<JsValue>` arm.
    let promise = mgr
        .register("smoke-full".to_string(), None, 1_000_000_000.0, lifecycle)
        .expect("register should accept the new shape");
    wasm_bindgen_futures::JsFuture::from(promise)
        .await
        .expect("register promise should resolve");

    // Status should now contain exactly one entry with the expected id.
    let status_promise = mgr.status();
    let status_val = wasm_bindgen_futures::JsFuture::from(status_promise)
        .await
        .expect("status should resolve");
    let arr: js_sys::Array = status_val.dyn_into().unwrap();
    assert_eq!(arr.length(), 1, "status should contain exactly one entry");

    let entry = arr.get(0);
    let id = Reflect::get(&entry, &JsValue::from_str("id"))
        .expect("entry should have id")
        .as_string()
        .expect("id should be a string");
    assert_eq!(id, "smoke-full");
    let loaded = Reflect::get(&entry, &JsValue::from_str("loaded"))
        .expect("entry should have loaded")
        .as_bool()
        .expect("loaded should be a bool");
    assert!(!loaded, "newly registered model should not be loaded");

    // `unregister` is currently a documented no-op upstream, but it must
    // still be callable without panicking on the new manager instance.
    mgr.unregister("smoke-full");
}

#[wasm_bindgen_test]
async fn js_api_model_manager_register_minimal_lifecycle() {
    use blazen_wasm_sdk::manager::WasmModelManager;
    use js_sys::{Function, Object, Reflect};
    use wasm_bindgen::prelude::*;

    let mgr = WasmModelManager::new(4.0); // 4 GB budget

    // Minimal lifecycle: only the required `load` / `unload`. No
    // `isLoaded` / `vramBytes` -- the adapter must fall back to its
    // documented defaults without erroring at registration time.
    let lifecycle = Object::new();
    let load = Function::new_no_args("return Promise.resolve();");
    let unload = Function::new_no_args("return Promise.resolve();");
    Reflect::set(&lifecycle, &JsValue::from_str("load"), load.as_ref()).unwrap();
    Reflect::set(&lifecycle, &JsValue::from_str("unload"), unload.as_ref()).unwrap();

    let promise = mgr
        .register("smoke-minimal".to_string(), None, 500_000_000.0, lifecycle)
        .expect("register should accept a minimal lifecycle");
    wasm_bindgen_futures::JsFuture::from(promise)
        .await
        .expect("register promise should resolve");

    let status_promise = mgr.status();
    let status_val = wasm_bindgen_futures::JsFuture::from(status_promise)
        .await
        .expect("status should resolve");
    let arr: js_sys::Array = status_val.dyn_into().unwrap();
    assert_eq!(arr.length(), 1, "status should contain exactly one entry");

    let entry = arr.get(0);
    let id = Reflect::get(&entry, &JsValue::from_str("id"))
        .expect("entry should have id")
        .as_string()
        .expect("id should be a string");
    assert_eq!(id, "smoke-minimal");

    mgr.unregister("smoke-minimal");
}
