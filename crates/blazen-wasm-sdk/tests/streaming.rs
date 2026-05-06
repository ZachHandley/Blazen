#![cfg(target_arch = "wasm32")]

//! `wasm-bindgen-test` coverage for the `ContentStore.fetchStream` JS surface.
//!
//! Exercises three scenarios end-to-end:
//!
//! 1. Custom store whose `fetchStream` callback returns a buffered
//!    `Uint8Array` — verifies the legacy single-chunk shim still works.
//! 2. Custom store whose `fetchStream` callback returns a live
//!    `ReadableStream<Uint8Array>` enqueueing multiple chunks — verifies
//!    that the WASM bridge preserves chunk boundaries through both the
//!    JS->Rust adapter (`readable_stream_to_byte_stream`) and the
//!    Rust->JS adapter (`byte_stream_to_readable_stream`) wired up by the
//!    `fetchStream` JS method.
//! 3. Built-in `inMemory` store — round-trips bytes through `put` then
//!    `fetchStream`, confirming the default `ByteStream` fallback also
//!    surfaces correctly through the new method.
//! 4. Error path — a `fetchStream` callback that hands back a
//!    `ReadableStream` whose underlying source errors. The promise itself
//!    resolves with the wrapped stream; the error must surface when the
//!    JS consumer drains the stream.
//!
//! The tests deliberately keep all chunk construction in JS-evaluated
//! source strings (via `js_sys::Function::new_with_args`) to mirror what
//! a host application would actually do — building `Uint8Array`s
//! directly in Rust loses some of the integration value.

use blazen_wasm_sdk::content::WasmContentStore;
use js_sys::{Function, Object, Promise, Reflect, Uint8Array};
use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;
use wasm_bindgen_futures::JsFuture;
use wasm_bindgen_test::{wasm_bindgen_test, wasm_bindgen_test_configure};
use web_sys::{ReadableStream, ReadableStreamDefaultReader};

wasm_bindgen_test_configure!(run_in_node_experimental);

// ---------------------------------------------------------------------------
// JS plumbing helpers
// ---------------------------------------------------------------------------

/// Build a JS handle object `{ id, kind, mimeType?, byteSize? }`.
fn make_handle(id: &str, kind: &str) -> JsValue {
    let obj = Object::new();
    Reflect::set(&obj, &JsValue::from_str("id"), &JsValue::from_str(id)).unwrap();
    Reflect::set(&obj, &JsValue::from_str("kind"), &JsValue::from_str(kind)).unwrap();
    obj.into()
}

/// Build a minimal JS options object with `put`, `resolve`, `fetchBytes`,
/// and `fetchStream` methods. Each method is supplied as raw JS source via
/// `Function::new_with_args` — this avoids building Rust closures (which
/// require `Closure::wrap` boilerplate and add no test value) and matches
/// how a host application would wire callbacks in practice.
fn build_options_with_fetch_stream(fetch_stream_body: &str) -> Object {
    let opts = Object::new();

    // `put(body, hint)` — synchronous; just hands back a fixed handle.
    let put = Function::new_with_args(
        "body, hint",
        "return { id: 'h-1', kind: 'data', mimeType: 'application/octet-stream' };",
    );
    // `resolve(handle)` — synchronous; returns a URL `MediaSource` shape.
    let resolve = Function::new_with_args(
        "handle",
        "return { type: 'url', url: 'https://example.test/' + handle.id };",
    );
    // `fetchBytes(handle)` — synchronous; empty payload (the streaming
    // tests only exercise `fetchStream`).
    let fetch_bytes = Function::new_with_args("handle", "return new Uint8Array();");
    let fetch_stream = Function::new_with_args("handle", fetch_stream_body);

    Reflect::set(&opts, &JsValue::from_str("put"), put.as_ref()).unwrap();
    Reflect::set(&opts, &JsValue::from_str("resolve"), resolve.as_ref()).unwrap();
    Reflect::set(
        &opts,
        &JsValue::from_str("fetchBytes"),
        fetch_bytes.as_ref(),
    )
    .unwrap();
    Reflect::set(
        &opts,
        &JsValue::from_str("fetchStream"),
        fetch_stream.as_ref(),
    )
    .unwrap();

    opts
}

/// Drain a `ReadableStream<Uint8Array>` end-to-end, returning every chunk
/// as its own `Vec<u8>`. The boundaries are preserved so we can assert
/// that the underlying `pull` semantics aren't coalescing chunks.
async fn drain_stream(stream: ReadableStream) -> Result<Vec<Vec<u8>>, JsValue> {
    let reader: ReadableStreamDefaultReader = stream.get_reader().unchecked_into();
    let mut chunks: Vec<Vec<u8>> = Vec::new();
    loop {
        let result = JsFuture::from(reader.read()).await?;
        let done = Reflect::get(&result, &JsValue::from_str("done"))?
            .as_bool()
            .unwrap_or(false);
        if done {
            break;
        }
        let value = Reflect::get(&result, &JsValue::from_str("value"))?;
        if !value.is_instance_of::<Uint8Array>() {
            return Err(JsValue::from_str("chunk is not a Uint8Array"));
        }
        let arr: Uint8Array = value.unchecked_into();
        chunks.push(arr.to_vec());
    }
    Ok(chunks)
}

// ---------------------------------------------------------------------------
// Test 1 — custom store, `fetchStream` returns buffered `Uint8Array`
// ---------------------------------------------------------------------------

#[wasm_bindgen_test]
async fn fetch_stream_custom_returns_buffered_bytes() {
    // Callback returns a single `Uint8Array` rather than a ReadableStream.
    // The bridge must wrap this as a single-chunk ByteStream and surface
    // it back as a one-chunk ReadableStream on the JS side.
    let opts = build_options_with_fetch_stream(
        "return new Uint8Array([1, 2, 3, 4, 5]);",
    );
    let store = WasmContentStore::custom(opts.unchecked_into())
        .expect("custom store should construct");

    let handle = make_handle("h-1", "data");
    let promise: Promise = store.fetch_stream(handle.unchecked_into()).unchecked_into();
    let stream_val = JsFuture::from(promise)
        .await
        .expect("fetchStream promise should resolve");

    assert!(
        stream_val.is_instance_of::<ReadableStream>(),
        "fetchStream must resolve with a ReadableStream"
    );
    let stream: ReadableStream = stream_val.unchecked_into();
    let chunks = drain_stream(stream).await.expect("drain should succeed");

    // Single buffered chunk is mirrored verbatim — no coalescing, no
    // splitting.
    assert_eq!(chunks.len(), 1, "buffered bytes should yield one chunk");
    assert_eq!(chunks[0], vec![1, 2, 3, 4, 5]);
}

// ---------------------------------------------------------------------------
// Test 2 — custom store, `fetchStream` returns a live `ReadableStream`
// ---------------------------------------------------------------------------

#[wasm_bindgen_test]
async fn fetch_stream_custom_returns_readable_stream_three_chunks() {
    // The callback constructs a ReadableStream whose `start` enqueues
    // three distinct chunks then closes the controller. The Rust bridge
    // (readable_stream_to_byte_stream -> ByteStream ->
    // byte_stream_to_readable_stream) must preserve all three chunks
    // and their boundaries.
    let body = r#"
        return new ReadableStream({
            start(controller) {
                controller.enqueue(new Uint8Array([10, 20]));
                controller.enqueue(new Uint8Array([30, 40, 50]));
                controller.enqueue(new Uint8Array([60]));
                controller.close();
            }
        });
    "#;
    let opts = build_options_with_fetch_stream(body);
    let store = WasmContentStore::custom(opts.unchecked_into())
        .expect("custom store should construct");

    let handle = make_handle("h-stream", "data");
    let promise: Promise = store.fetch_stream(handle.unchecked_into()).unchecked_into();
    let stream_val = JsFuture::from(promise)
        .await
        .expect("fetchStream promise should resolve");

    let stream: ReadableStream = stream_val.unchecked_into();
    let chunks = drain_stream(stream).await.expect("drain should succeed");

    assert_eq!(chunks.len(), 3, "three enqueued chunks should round-trip");
    assert_eq!(chunks[0], vec![10, 20]);
    assert_eq!(chunks[1], vec![30, 40, 50]);
    assert_eq!(chunks[2], vec![60]);
}

// ---------------------------------------------------------------------------
// Test 3 — built-in `inMemory` store: put bytes, fetchStream them back
// ---------------------------------------------------------------------------

#[wasm_bindgen_test]
async fn fetch_stream_in_memory_store_round_trip() {
    // `inMemory` is the only built-in store reachable on wasm32 without
    // external API keys (per the module-level docs in `content.rs`).
    // Round-trip a buffer of bytes via `put` -> `fetchStream` and verify
    // the payload survives the default ByteStream fallback path.
    let store = WasmContentStore::in_memory();

    let payload: Vec<u8> = (0u8..32).collect();
    let arr = Uint8Array::from(payload.as_slice());

    // `put(body, kindHint?, mimeType?, displayName?)` — first arg is the
    // typed body. We pass the Uint8Array directly, which the wasm-bindgen
    // signature expects as `ContentPutBodyTs` (a `JsValue` newtype).
    let body_js: JsValue = arr.into();
    let put_promise: Promise = store
        .put(
            body_js.unchecked_ref(),
            Some("data".to_string()),
            Some("application/octet-stream".to_string()),
            Some("payload.bin".to_string()),
        )
        .unchecked_into();
    let handle_val = JsFuture::from(put_promise)
        .await
        .expect("put promise should resolve");

    // `fetchStream` against the issued handle; drain and confirm bytes.
    let stream_promise: Promise = store
        .fetch_stream(handle_val.unchecked_into())
        .unchecked_into();
    let stream_val = JsFuture::from(stream_promise)
        .await
        .expect("fetchStream promise should resolve");
    let stream: ReadableStream = stream_val.unchecked_into();
    let chunks = drain_stream(stream).await.expect("drain should succeed");

    let total: Vec<u8> = chunks.into_iter().flatten().collect();
    assert_eq!(total, payload, "round-tripped bytes should match");
}

// ---------------------------------------------------------------------------
// Test 4 — error path: a ReadableStream whose source errors mid-flight
// ---------------------------------------------------------------------------

#[wasm_bindgen_test]
async fn fetch_stream_propagates_underlying_stream_error() {
    // The callback constructs a ReadableStream that enqueues one chunk
    // and then synchronously errors the controller. The `fetchStream`
    // promise itself resolves (the bridge installs the stream lazily),
    // but draining the resulting JS-side stream must surface the error
    // either as a rejected `read()` or as a chunk-boundary error.
    let body = r#"
        return new ReadableStream({
            start(controller) {
                controller.enqueue(new Uint8Array([7, 8, 9]));
                controller.error(new Error('boom'));
            }
        });
    "#;
    let opts = build_options_with_fetch_stream(body);
    let store = WasmContentStore::custom(opts.unchecked_into())
        .expect("custom store should construct");

    let handle = make_handle("h-err", "data");
    let promise: Promise = store.fetch_stream(handle.unchecked_into()).unchecked_into();
    let stream_val = JsFuture::from(promise)
        .await
        .expect("fetchStream promise should resolve even when source errors");
    let stream: ReadableStream = stream_val.unchecked_into();

    // Drain — at least one chunk may arrive before the error surfaces,
    // but `drain_stream` MUST eventually return Err when the underlying
    // source's error has propagated through both bridges.
    let result = drain_stream(stream).await;
    assert!(
        result.is_err(),
        "errored ReadableStream must surface as an Err on the consumer side, got {result:?}",
    );
}
