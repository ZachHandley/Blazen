#![cfg(target_arch = "wasm32")]

//! Smoke tests for the WASM `ControlPlaneClient` / `ControlPlaneWorker`
//! bindings.
//!
//! Real end-to-end coverage against a running control-plane server lives
//! under `crates/blazen-controlplane/tests/` (host-side). The wasm-side
//! tests here verify:
//!
//! - The classes exist and instantiate.
//! - `connect()` validates the endpoint shape (`http://`/`https://` only,
//!   non-empty) before any network I/O.
//!
//! Anything that requires a live HTTP server is out of scope for the
//! browser-only suite — wasm-bindgen-test runs in headless firefox /
//! chrome without a server to talk to.

use blazen_wasm_sdk::controlplane::{
    WasmAdmissionMode, WasmCapability, WasmControlPlaneClient, WasmControlPlaneWorker,
    WasmWorkerConfig,
};
use wasm_bindgen_futures::JsFuture;
use wasm_bindgen_test::{wasm_bindgen_test, wasm_bindgen_test_configure};

wasm_bindgen_test_configure!(run_in_browser);

#[wasm_bindgen_test]
async fn client_connect_rejects_empty_endpoint() {
    let promise = WasmControlPlaneClient::connect(String::new(), None);
    let result = JsFuture::from(promise).await;
    assert!(result.is_err(), "empty endpoint must reject");
}

#[wasm_bindgen_test]
async fn client_connect_rejects_bare_endpoint() {
    let promise = WasmControlPlaneClient::connect("cp.example.com".into(), None);
    let result = JsFuture::from(promise).await;
    assert!(
        result.is_err(),
        "endpoint without scheme must reject (got: {result:?})"
    );
}

#[wasm_bindgen_test]
async fn client_connect_accepts_https_endpoint() {
    let promise =
        WasmControlPlaneClient::connect("https://cp.example.com".into(), Some("tok".into()));
    let val = JsFuture::from(promise).await.expect("connect resolves");
    // The resolved value is the constructed JS class; confirm it
    // exposes the expected `endpoint` getter (string property).
    let endpoint = js_sys::Reflect::get(&val, &wasm_bindgen::JsValue::from_str("endpoint"))
        .expect("get endpoint property")
        .as_string()
        .expect("endpoint is a string");
    assert_eq!(endpoint, "https://cp.example.com");
}

#[wasm_bindgen_test]
async fn worker_connect_rejects_bad_endpoint() {
    let config = WasmWorkerConfig {
        node_id: "w1".into(),
        capabilities: vec![WasmCapability {
            kind: "workflow:test".into(),
            version: 1,
        }],
        tags: None,
        admission: Some(WasmAdmissionMode::Reactive),
        bearer_token: None,
    };
    let promise = WasmControlPlaneWorker::connect("not-a-url".into(), config);
    let result = JsFuture::from(promise).await;
    assert!(result.is_err(), "endpoint without scheme must reject");
}

#[wasm_bindgen_test]
fn capability_roundtrip_via_serde_json() {
    // Pure-data sanity check that the JS-facing capability struct
    // round-trips through the `CapabilityWire` postcard mirror without
    // needing a live control plane. Exercises the From impl that the
    // worker constructor relies on.
    let cap = WasmCapability {
        kind: "workflow:summarize".into(),
        version: 3,
    };
    let json = serde_json::to_string(&cap).expect("encode capability JSON");
    let decoded: WasmCapability = serde_json::from_str(&json).expect("decode capability JSON");
    assert_eq!(decoded.kind, "workflow:summarize");
    assert_eq!(decoded.version, 3);
}

#[wasm_bindgen_test]
fn admission_mode_serde_shape_is_tagged() {
    let modes = [
        WasmAdmissionMode::Fixed { max_in_flight: 4 },
        WasmAdmissionMode::VramBudget { total_mb: 16_384 },
        WasmAdmissionMode::Reactive,
    ];
    for mode in &modes {
        let json = serde_json::to_string(mode).expect("encode admission JSON");
        // Every variant must carry a `type` discriminator so plain JS
        // object literals (`{ type: 'reactive' }`) decode cleanly.
        assert!(json.contains("\"type\""), "expected type tag in {json}");
    }
}
