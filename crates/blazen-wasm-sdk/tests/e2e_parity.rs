#![cfg(target_arch = "wasm32")]

//! End-to-end parity test for the wasm-sdk bindings.
//!
//! Mirrors `tests/python/test_e2e_parity.py` and
//! `tests/node/test_e2e_parity.mjs`: verifies that a 2-stage `Pipeline`
//! composed of `Stage` + `Workflow` + `InMemoryBackend`-backed `Memory`
//! can be wired together via the public `wasm-bindgen` surface without
//! throwing at construction time, and that the resulting `Pipeline`
//! exposes the expected method shape.
//!
//! Surface deltas vs. the Python/Node sister files:
//!
//! - `PromptTemplate` is not currently exposed by `blazen-wasm-sdk`. The
//!   Python and Node bindings re-export `blazen-prompts`, but the WASM
//!   surface does not (yet). Once `PromptTemplate` is bound, this file
//!   should grow a parallel `render()` round-trip assertion.
//! - `CustomProvider` is likewise not bound on the WASM surface (the
//!   `napi-rs` and `pyo3` bindings expose it via `JsCustomProvider` /
//!   `PyCustomProvider`; the WASM crate does not). Once a
//!   `JsCustomProvider`-shaped binding lands here, this file should grow
//!   a `providerId` round-trip assertion.
//!
//! Test bodies are construction/shape-only, mirroring the Node sister
//! file. We deliberately do NOT call `pipeline.start().await` here:
//! `blazen_pipeline::Pipeline::start` internally calls `tokio::spawn`,
//! and `wasm-bindgen-test` does not stand up a Tokio reactor on
//! `wasm32-unknown-unknown` (the production WASM caller embeds its own
//! runtime, e.g. via `wasm-bindgen-futures` plus a JS event loop). The
//! native Rust integration tests in `crates/blazen-pipeline/tests` cover
//! the runtime path; the parity contract this file enforces is that
//! every JS-facing class composes cleanly without panicking.

use blazen_wasm_sdk::memory::{WasmInMemoryBackend, WasmMemory};
use blazen_wasm_sdk::pipeline::{WasmPipeline, WasmPipelineBuilder, WasmStage};
use blazen_wasm_sdk::workflow::WasmWorkflow;
use js_sys::Array;
use wasm_bindgen::prelude::*;
use wasm_bindgen_test::{wasm_bindgen_test, wasm_bindgen_test_configure};

wasm_bindgen_test_configure!(run_in_node_experimental);

// Build a JS function that, when invoked as a step handler, returns a
// `StopEvent` envelope carrying the supplied JSON payload.
//
// The pipeline engine awaits whatever the JS function returns; for shape
// wiring we hand back a synchronous plain object with the
// `blazen::StopEvent` discriminator that the dispatch layer recognises.
fn make_stop_step_handler(payload_json: &'static str) -> js_sys::Function {
    let body = format!(
        "return {{ type: 'blazen::StopEvent', result: {payload_json} }};"
    );
    // Two-arg signature matches the `(event, ctx)` calling convention the
    // workflow engine uses for step handlers.
    js_sys::Function::new_with_args("event, ctx", &body)
}

// Build a JS array containing a single event-type string. The
// `Workflow.addStep` binding expects a JS array of strings, so a Rust
// `Vec<&str>` would not deserialise — we hand it an actual `js_sys::Array`.
fn event_types_array(event_type: &str) -> JsValue {
    let arr = Array::new();
    arr.push(&JsValue::from_str(event_type));
    arr.into()
}

#[wasm_bindgen_test]
fn e2e_parity_pipeline_with_memory() {
    // 1. InMemoryBackend wrapped in a Memory (local-only mode -- no
    //    embedder needed, mirrors the Node sister test's
    //    `Memory.local(backend)` construction).
    let backend = WasmInMemoryBackend::new();
    let memory = WasmMemory::local_from_backend(&backend);
    // We can't downcast across `wasm-bindgen` boundaries, so we just
    // assert the value drops cleanly. Construction not panicking is the
    // contract under test.
    drop(memory);

    // 2. Build two distinct workflows, each with a single step that
    //    terminates the run via a `StopEvent` envelope.
    let mut wf1 = WasmWorkflow::new("stage-1");
    wf1.add_step(
        "ingest",
        event_types_array("blazen::StartEvent"),
        make_stop_step_handler("{ prompt: 'hello world' }"),
    )
    .expect("addStep on wf1 should succeed");

    let mut wf2 = WasmWorkflow::new("stage-2");
    wf2.add_step(
        "respond",
        event_types_array("blazen::StartEvent"),
        make_stop_step_handler("{ reply: 'ok' }"),
    )
    .expect("addStep on wf2 should succeed");

    // 3. Wrap each workflow as a Stage. The Stage constructor consumes
    //    the workflow's pending steps into a real engine workflow.
    let stage1 = WasmStage::new("ingest".to_owned(), &mut wf1, None, None)
        .expect("Stage construction for stage-1 should succeed");
    let stage2 = WasmStage::new("respond".to_owned(), &mut wf2, None, None)
        .expect("Stage construction for stage-2 should succeed");

    assert_eq!(stage1.name(), "ingest");
    assert_eq!(stage2.name(), "respond");

    // 4. Build the 2-stage Pipeline via the fluent builder. We assert
    //    the build succeeds and yields a `WasmPipeline`; the runtime
    //    behavior of `start()` is covered by the native pipeline
    //    integration tests, which is the same split the Node sister
    //    file uses (it never invokes `pipeline.start()` either).
    let builder = WasmPipelineBuilder::new("e2e-parity".to_owned());
    builder.stage(&stage1).expect("stage1 should append");
    builder.stage(&stage2).expect("stage2 should append");
    let pipeline: WasmPipeline = builder.build().expect("pipeline build should succeed");

    // The `WasmPipeline` JS class doesn't expose getters; reaching
    // `build()` without panicking is the wiring contract under test.
    drop(pipeline);
}

#[wasm_bindgen_test]
fn pipeline_builder_rejects_empty_pipelines() {
    // Mirrors the Python and Node parity assertion that an empty
    // pipeline cannot be built -- the builder must require at least one
    // stage. The underlying `blazen_pipeline::PipelineBuilder::build`
    // returns `PipelineError::ValidationFailed` which surfaces as a
    // `JsValue` error here.
    let empty = WasmPipelineBuilder::new("empty".to_owned());
    // `WasmPipeline` doesn't implement `Debug`, so we can't use
    // `expect_err`; pattern-match the `Result` instead.
    let err = match empty.build() {
        Ok(_) => panic!("empty pipeline build must error"),
        Err(e) => e,
    };
    let msg = err.as_string().unwrap_or_default().to_lowercase();
    assert!(
        msg.contains("stage") || msg.contains("empty") || msg.contains("at least"),
        "error message should reference the missing-stage condition: {msg}"
    );
}

#[wasm_bindgen_test]
fn in_memory_backend_constructs_standalone() {
    // Smoke check that the standalone `InMemoryBackend` wrapper exposes
    // the `Default` constructor and that `Memory::local_from_backend`
    // accepts a borrowed reference (so the JS-side `new InMemoryBackend()`
    // pattern works without consuming the backend).
    let backend = WasmInMemoryBackend::new();
    let memory_a = WasmMemory::local_from_backend(&backend);
    let memory_b = WasmMemory::local_from_backend(&backend);
    drop(memory_a);
    drop(memory_b);
}
