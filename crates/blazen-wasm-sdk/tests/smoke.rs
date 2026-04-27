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
