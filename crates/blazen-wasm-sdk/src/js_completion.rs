//! JavaScript callback bridge for local/custom completion models.
//!
//! Allows TypeScript code to supply a completion handler function that is
//! called from the Rust `CompletionModel` trait implementation. This mirrors
//! the `JsTool` pattern in `agent.rs` and enables local or custom models
//! (e.g. `transformers.js`, WebLLM, or any JS-side inference library) to
//! participate in the Blazen completion pipeline.
//!
//! # Usage (TypeScript)
//!
//! ```typescript
//! const model = CompletionModel.fromJsHandler(
//!   'my-local-model',
//!   async (request) => {
//!     // `request` is a CompletionRequest serialized as a plain JS object.
//!     const response = await myLocalModel.generate(request);
//!     // Return a CompletionResponse-shaped object.
//!     return {
//!       content: response.text,
//!       toolCalls: [],
//!       citations: [],
//!       artifacts: [],
//!       images: [],
//!       audio: [],
//!       videos: [],
//!       model: 'my-local-model',
//!       metadata: {},
//!     };
//!   },
//! );
//!
//! const result = await model.complete([ChatMessage.user('Hello!')]);
//! ```

use std::pin::Pin;

use async_trait::async_trait;
use futures_util::Stream;
use wasm_bindgen::prelude::*;

use blazen_llm::types::{CompletionRequest, CompletionResponse, StreamChunk};
use blazen_llm::BlazenError;

// ---------------------------------------------------------------------------
// SendFuture wrapper (same pattern as agent.rs)
// ---------------------------------------------------------------------------

/// Wrapper that unsafely implements `Send` for a non-Send future.
/// SAFETY: WASM is single-threaded.
struct SendFuture<F>(F);

unsafe impl<F> Send for SendFuture<F> {}

impl<F: std::future::Future> std::future::Future for SendFuture<F> {
    type Output = F::Output;

    fn poll(
        self: Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Self::Output> {
        // SAFETY: We are not moving F, just projecting through the wrapper.
        let inner = unsafe { self.map_unchecked_mut(|s| &mut s.0) };
        inner.poll(cx)
    }
}

// ---------------------------------------------------------------------------
// TypeScript type declarations
// ---------------------------------------------------------------------------

#[wasm_bindgen(typescript_custom_section)]
const TS_COMPLETE_HANDLER: &str = r#"
/**
 * A JavaScript function that performs a completion given a request object.
 * The request is a plain JS object matching the CompletionRequest schema.
 * Must return (or resolve to) a CompletionResponse-shaped object.
 */
export type CompleteHandler = (request: any) => Promise<any>;

/**
 * A JavaScript function that performs a streaming completion.
 * Called with the request and a callback that should be invoked for each
 * chunk. The callback accepts a StreamChunk-shaped object.
 * Must return (or resolve to) void when streaming is complete.
 *
 * **Note:** For v1, if no stream handler is provided, streaming falls back
 * to a single-chunk approach that calls the complete handler and yields one
 * chunk containing the full response.
 */
export type StreamHandler = (request: any, onChunk: (chunk: any) => void) => Promise<void>;
"#;

// ---------------------------------------------------------------------------
// JsCompletionHandler
// ---------------------------------------------------------------------------

/// A `CompletionModel` whose execution is delegated to JavaScript functions.
///
/// The `complete_handler` is called for non-streaming completions.
/// The `stream_handler` is optional: if absent, `stream()` falls back to
/// calling `complete()` and yielding the result as a single `StreamChunk`.
pub(crate) struct JsCompletionHandler {
    model_id: String,
    complete_handler: js_sys::Function,
    stream_handler: Option<js_sys::Function>,
}

// SAFETY: WASM is single-threaded.
unsafe impl Send for JsCompletionHandler {}
unsafe impl Sync for JsCompletionHandler {}

impl std::fmt::Debug for JsCompletionHandler {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("JsCompletionHandler")
            .field("model_id", &self.model_id)
            .field("has_stream_handler", &self.stream_handler.is_some())
            .finish_non_exhaustive()
    }
}

impl JsCompletionHandler {
    /// Create a new handler from JS functions.
    pub(crate) fn new(
        model_id: String,
        complete_handler: js_sys::Function,
        stream_handler: Option<js_sys::Function>,
    ) -> Self {
        Self {
            model_id,
            complete_handler,
            stream_handler,
        }
    }

    /// Internal non-Send async complete implementation.
    async fn complete_impl(
        &self,
        request: CompletionRequest,
    ) -> Result<CompletionResponse, BlazenError> {
        // Serialize the request to a JS value.
        let js_request = serde_wasm_bindgen::to_value(&request)
            .map_err(|e| BlazenError::provider("js_handler", e.to_string()))?;

        // Call the JS handler.
        let result = self
            .complete_handler
            .call1(&JsValue::NULL, &js_request)
            .map_err(|e| BlazenError::provider("js_handler", format!("{e:?}")))?;

        // If the result is a Promise, await it.
        let result = if result.has_type::<js_sys::Promise>() {
            let promise: js_sys::Promise = result.unchecked_into();
            wasm_bindgen_futures::JsFuture::from(promise)
                .await
                .map_err(|e| BlazenError::provider("js_handler", format!("{e:?}")))?
        } else {
            result
        };

        // Deserialize the JS result back into a CompletionResponse.
        serde_wasm_bindgen::from_value::<CompletionResponse>(result)
            .map_err(|e| BlazenError::invalid_response(e.to_string()))
    }

    /// Internal non-Send async stream implementation using the JS stream handler.
    async fn stream_with_handler_impl(
        &self,
        request: CompletionRequest,
        stream_handler: &js_sys::Function,
    ) -> Result<Vec<StreamChunk>, BlazenError> {
        use std::cell::RefCell;
        use std::rc::Rc;

        // Serialize the request to a JS value.
        let js_request = serde_wasm_bindgen::to_value(&request)
            .map_err(|e| BlazenError::provider("js_handler", e.to_string()))?;

        // Collect chunks in a shared buffer via a JS callback closure.
        let chunks: Rc<RefCell<Vec<StreamChunk>>> = Rc::new(RefCell::new(Vec::new()));
        let chunks_ref = Rc::clone(&chunks);

        let on_chunk = Closure::wrap(Box::new(move |js_chunk: JsValue| {
            if let Ok(chunk) = serde_wasm_bindgen::from_value::<StreamChunk>(js_chunk) {
                chunks_ref.borrow_mut().push(chunk);
            }
        }) as Box<dyn FnMut(JsValue)>);

        // Call stream_handler(request, onChunk).
        let result = stream_handler
            .call2(&JsValue::NULL, &js_request, on_chunk.as_ref().unchecked_ref())
            .map_err(|e| BlazenError::provider("js_handler", format!("{e:?}")))?;

        // Await the promise if returned.
        if result.has_type::<js_sys::Promise>() {
            let promise: js_sys::Promise = result.unchecked_into();
            wasm_bindgen_futures::JsFuture::from(promise)
                .await
                .map_err(|e| BlazenError::stream_error(format!("{e:?}")))?;
        }

        // Drop the closure so its prevent-GC prevent is released.
        drop(on_chunk);

        // Extract the collected chunks.
        let result = chunks.borrow().clone();
        Ok(result)
    }
}

#[async_trait]
impl blazen_llm::traits::CompletionModel for JsCompletionHandler {
    fn model_id(&self) -> &str {
        &self.model_id
    }

    async fn complete(
        &self,
        request: CompletionRequest,
    ) -> Result<CompletionResponse, BlazenError> {
        // SAFETY: WASM is single-threaded, Send is vacuously satisfied.
        SendFuture(self.complete_impl(request)).await
    }

    async fn stream(
        &self,
        request: CompletionRequest,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<StreamChunk, BlazenError>> + Send>>, BlazenError>
    {
        if let Some(ref handler) = self.stream_handler {
            // Use the JS stream handler: collect all chunks, then yield them.
            // This is a v1 approach -- true incremental streaming would require
            // an async-iterable bridge (Phase 13.5.c).
            let chunks =
                SendFuture(self.stream_with_handler_impl(request, handler)).await?;
            Ok(Box::pin(futures_util::stream::iter(
                chunks.into_iter().map(Ok),
            )))
        } else {
            // Fallback: call complete() and yield one chunk with the full response.
            let response = self.complete(request).await?;
            let chunk = StreamChunk {
                delta: response.content,
                tool_calls: response.tool_calls,
                finish_reason: response.finish_reason,
                reasoning_delta: response.reasoning.map(|r| r.text),
                citations: response.citations,
                artifacts: response.artifacts,
            };
            Ok(Box::pin(futures_util::stream::once(async { Ok(chunk) })))
        }
    }
}
