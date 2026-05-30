#![cfg(target_arch = "wasm32")]

//! Live-`JsValue` identity passthrough tests for the workflow bridge.
//!
//! These verify the `js_native` lane: an event returned by one step keeps
//! its JS object identity (including non-JSON members such as functions)
//! when handed to the next step's handler, instead of being flattened
//! through `serde_json::Value` and rebuilt with `DynamicEvent::from_json`.
//!
//! Run under `wasm-bindgen-test` (a JS heap is required to host the live
//! `JsValue`s). The two-step workflow is wired entirely through the public
//! `wasm-bindgen` surface so the test exercises the same path production JS
//! callers hit.

use blazen_wasm_sdk::workflow::{EventTypesTs, StepHandlerTs, WasmWorkflow};
use js_sys::{Array, Function, Reflect};
use wasm_bindgen::prelude::*;
use wasm_bindgen_test::{wasm_bindgen_test, wasm_bindgen_test_configure};

wasm_bindgen_test_configure!(run_in_node_experimental);

/// Reinterpret a single event-type string as the `EventTypesTs` newtype that
/// `add_step` accepts.
fn event_types(event_type: &str) -> EventTypesTs {
    let arr = Array::new();
    arr.push(&JsValue::from_str(event_type));
    JsValue::from(arr).unchecked_into()
}

/// Reinterpret a JS step handler function as the `StepHandlerTs` newtype.
fn handler(body: &str) -> StepHandlerTs {
    JsValue::from(Function::new_with_args("event, ctx", body)).unchecked_into()
}

/// Step A returns an event object carrying a live JS function under
/// `marker`. Step B reads `event.marker`: if identity survived the hop it is
/// still a callable function returning `42`; if the event had been
/// JSON-flattened the function would be gone (serialised to `null` /
/// dropped). Step B terminates the run with `survived: <bool>`.
#[wasm_bindgen_test]
async fn live_js_object_identity_survives_step_hop() {
    let mut wf = WasmWorkflow::new("identity-passthrough");

    // Step A: emit an event whose payload includes a live function member.
    wf.add_step(
        "a",
        event_types("blazen::StartEvent"),
        handler(
            "return { type: 'Mid', marker: () => 42, plain: 'kept' };",
        ),
    )
    .expect("addStep a");

    // Step B: probe the marker. A surviving live object still has a callable
    // `marker`; a JSON round-trip would have stripped it.
    wf.add_step(
        "b",
        event_types("Mid"),
        handler(
            "const survived = typeof event.marker === 'function' && event.marker() === 42; \
             return { type: 'StopEvent', result: { survived, plain: event.plain } };",
        ),
    )
    .expect("addStep b");

    let result_js = wf.run(JsValue::NULL).await.expect("workflow run should resolve");

    let survived = Reflect::get(&result_js, &JsValue::from_str("survived"))
        .expect("result.survived present")
        .as_bool()
        .expect("result.survived is a boolean");
    assert!(
        survived,
        "live JS object identity must survive the step-to-step hop (marker function lost)"
    );

    let plain = Reflect::get(&result_js, &JsValue::from_str("plain"))
        .expect("result.plain present")
        .as_string()
        .unwrap_or_default();
    assert_eq!(plain, "kept", "plain JSON members must also round-trip");
}

/// A native-backed event still serialises to its JSON snapshot on the
/// streaming / snapshot lane (the registered native serializer), so an
/// observer that only sees the JSON form gets real data — not the
/// placeholder `null`.
#[wasm_bindgen_test]
async fn native_backed_event_serializes_to_json_snapshot() {
    let mut wf = WasmWorkflow::new("snapshot-json");

    // Single step returns a StopEvent whose result is a plain JSON object;
    // the engine renders the terminal payload through `to_json()`, which for
    // a native-backed `DynamicEvent` consults the registered serializer.
    wf.add_step(
        "only",
        event_types("blazen::StartEvent"),
        handler("return { type: 'StopEvent', result: { ok: true, n: 7 } };"),
    )
    .expect("addStep only");

    let result_js = wf.run(JsValue::NULL).await.expect("workflow run should resolve");

    let ok = Reflect::get(&result_js, &JsValue::from_str("ok"))
        .expect("result.ok present")
        .as_bool()
        .expect("result.ok is a boolean");
    assert!(ok, "terminal payload must carry the real JSON snapshot");

    let n = Reflect::get(&result_js, &JsValue::from_str("n"))
        .expect("result.n present")
        .as_f64()
        .expect("result.n is a number");
    assert_eq!(n, 7.0);
}
