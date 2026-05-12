//! Streaming completion surface for the UniFFI bindings.
//!
//! Streaming sits behind a *foreign-callable sink trait* rather than the
//! async-iterator pattern used by `blazen-py` and `blazen-node`. UniFFI's
//! async-iterator support across Go, Swift, Kotlin, and Ruby is uneven;
//! foreign-callable traits are the universally supported primitive. Each
//! foreign-language idiomatic wrapper (in `bindings/<lang>/`) adapts the sink
//! callbacks into the host language's native streaming type:
//!
//! - Go: sink callbacks push to a `chan StreamChunk`
//! - Swift: sink callbacks build an `AsyncStream<StreamChunk>`
//! - Kotlin: sink callbacks emit into a `Flow<StreamChunk>`
//! - Ruby: sink callbacks yield to an `Enumerator::Lazy`
//!
//! ## Wire-format shape
//!
//! Upstream `blazen_llm::StreamChunk` carries `delta: Option<String>`,
//! `tool_calls`, `finish_reason`, `reasoning_delta`, `citations`, and
//! `artifacts`. The FFI [`StreamChunk`] flattens that into the fields most
//! foreign callers actually need (`content_delta`, `tool_calls`, `is_final`).
//! The terminal `finish_reason` is delivered via [`CompletionStreamSink::on_done`]
//! rather than on the chunk itself, mirroring the
//! `on_chunk` / `on_done` / `on_error` callback shape of typical foreign
//! streaming idioms.
//!
//! ## Usage (Rust-side; foreign sides look analogous)
//!
//! ```ignore
//! use std::sync::Arc;
//!
//! struct PrintSink;
//!
//! #[async_trait::async_trait]
//! impl CompletionStreamSink for PrintSink {
//!     async fn on_chunk(&self, chunk: StreamChunk) -> BlazenResult<()> {
//!         print!("{}", chunk.content_delta);
//!         Ok(())
//!     }
//!     async fn on_done(&self, finish_reason: String, _usage: TokenUsage) -> BlazenResult<()> {
//!         println!("\n[done: {finish_reason}]");
//!         Ok(())
//!     }
//!     async fn on_error(&self, err: BlazenError) -> BlazenResult<()> {
//!         eprintln!("stream error: {err}");
//!         Ok(())
//!     }
//! }
//!
//! complete_streaming(model, request, Arc::new(PrintSink)).await?;
//! ```

use std::sync::Arc;

use futures_util::StreamExt;

use blazen_llm::CompletionRequest as CoreCompletionRequest;

use crate::errors::{BlazenError, BlazenResult};
use crate::llm::{CompletionModel, CompletionRequest, TokenUsage, ToolCall};
use crate::runtime::runtime;

// ---------------------------------------------------------------------------
// Wire-format records
// ---------------------------------------------------------------------------

/// A single chunk from a streaming chat completion.
///
/// Chunks arrive in order. `content_delta` is the incremental text since the
/// last chunk (empty when the chunk carries only tool-call deltas or
/// reasoning trace). `tool_calls` is the latest known set of tool
/// invocations — upstream providers may emit tool-call deltas across
/// multiple chunks, so consumers should treat each chunk's `tool_calls` as a
/// snapshot rather than an append-only list.
///
/// `is_final` is set on the last content-bearing chunk before
/// [`CompletionStreamSink::on_done`] fires. It is a UI hint (e.g. "stop
/// showing the typing cursor") and does not replace `on_done` for cleanup.
#[derive(Debug, Clone, uniffi::Record)]
pub struct StreamChunk {
    /// Incremental text delta since the previous chunk. Empty when the
    /// chunk carries only tool-call deltas, reasoning trace, citations, or
    /// artifacts.
    pub content_delta: String,
    /// Tool-call snapshot for this chunk. May grow as the provider streams
    /// tool-call arguments piecewise; consumers should replace, not append.
    pub tool_calls: Vec<ToolCall>,
    /// True if this is the final content-bearing chunk before `on_done`.
    /// Hint only — `on_done` is the authoritative completion signal.
    pub is_final: bool,
}

// ---------------------------------------------------------------------------
// Foreign-implementable sink trait
// ---------------------------------------------------------------------------

/// Sink for streaming chat completion output, implemented in foreign code.
///
/// The streaming engine calls [`on_chunk`](Self::on_chunk) for each chunk as
/// it arrives, then exactly one of [`on_done`](Self::on_done) (success) or
/// [`on_error`](Self::on_error) (failure). Implementations should treat
/// `on_done` / `on_error` as cleanup hooks (close channels, complete async
/// iterators, etc.).
///
/// ## Async story
///
/// All three methods are `async` on the Rust side. UniFFI exposes them as:
/// - Go: blocking functions, safe from goroutines (compose with channels)
/// - Swift: `async throws` methods
/// - Kotlin: `suspend fun` methods
/// - Ruby: blocking methods (wrap in `Async { ... }` for fiber concurrency)
#[uniffi::export(with_foreign)]
#[async_trait::async_trait]
pub trait CompletionStreamSink: Send + Sync {
    /// Receive a single chunk from the streaming response.
    ///
    /// Returning an `Err` aborts the stream — the streaming engine will not
    /// call further `on_chunk` callbacks and will not call `on_done`.
    async fn on_chunk(&self, chunk: StreamChunk) -> BlazenResult<()>;

    /// Receive the terminal completion signal. Called exactly once at the
    /// end of a successful stream. Implementations should perform any
    /// cleanup here (close channels, signal completion to async iterators).
    ///
    /// `finish_reason` is the last `finish_reason` reported by the
    /// provider (e.g. `"stop"`, `"tool_calls"`, `"length"`) — empty string
    /// when the provider didn't report one. `usage` is the running token
    /// usage; some providers don't surface usage via the stream, in which
    /// case all counters are zero.
    async fn on_done(&self, finish_reason: String, usage: TokenUsage) -> BlazenResult<()>;

    /// Receive a fatal error from the stream. Called exactly once when the
    /// stream fails midway. After `on_error` fires, neither further
    /// `on_chunk` nor `on_done` will be called.
    async fn on_error(&self, err: BlazenError) -> BlazenResult<()>;
}

// ---------------------------------------------------------------------------
// Free function: drive a stream into a sink
// ---------------------------------------------------------------------------

/// Drive a streaming chat completion, dispatching each chunk to the sink.
///
/// On success, calls `sink.on_done(finish_reason, usage)` exactly once and
/// returns `Ok(())`. On a provider-side failure (or sink-side
/// `on_chunk`/`on_done` failure), calls `sink.on_error(...)` exactly once
/// and returns `Ok(())` — the error is *delivered* via the sink, not
/// propagated to this function's caller. This keeps the foreign-language
/// surface symmetric: the sink owns both happy-path and error-path
/// observation.
///
/// The only way this function itself returns `Err` is when the initial
/// request conversion fails (malformed JSON in tool definitions, etc.) or
/// when the upstream `stream()` call fails to *start* the stream. Sink
/// callback failures are surfaced via `on_error`.
#[uniffi::export(async_runtime = "tokio")]
pub async fn complete_streaming(
    model: Arc<CompletionModel>,
    request: CompletionRequest,
    sink: Arc<dyn CompletionStreamSink>,
) -> BlazenResult<()> {
    let core_request = CoreCompletionRequest::try_from(request)?;
    let stream = match model.inner.stream(core_request).await {
        Ok(s) => s,
        Err(err) => {
            // Why: surface the start-of-stream failure both to the caller
            // (return Err) and to the sink (so foreign-language consumers
            // observing only the sink also see it).
            let wire_err = BlazenError::from(err);
            let _ = sink.on_error(clone_error(&wire_err)).await;
            return Err(wire_err);
        }
    };

    let mut stream = std::pin::pin!(stream);
    let mut last_finish_reason = String::new();
    let mut pending: Option<StreamChunk> = None;

    while let Some(item) = stream.next().await {
        match item {
            Ok(chunk) => {
                if let Some(reason) = chunk.finish_reason.as_ref() {
                    last_finish_reason = reason.clone();
                }
                let wire = StreamChunk {
                    content_delta: chunk.delta.unwrap_or_default(),
                    tool_calls: chunk.tool_calls.into_iter().map(ToolCall::from).collect(),
                    is_final: false,
                };
                // Why: defer dispatch by one step so we can mark the last
                // content-bearing chunk's `is_final = true` without
                // peeking past the end of the stream.
                if let Some(prev) = pending.take()
                    && let Err(sink_err) = sink.on_chunk(prev).await
                {
                    let _ = sink.on_error(clone_error(&sink_err)).await;
                    return Ok(());
                }
                pending = Some(wire);
            }
            Err(err) => {
                if let Some(prev) = pending.take() {
                    let _ = sink.on_chunk(prev).await;
                }
                let wire_err = BlazenError::from(err);
                let _ = sink.on_error(wire_err).await;
                return Ok(());
            }
        }
    }

    if let Some(mut last) = pending.take() {
        last.is_final = true;
        if let Err(sink_err) = sink.on_chunk(last).await {
            let _ = sink.on_error(clone_error(&sink_err)).await;
            return Ok(());
        }
    }

    // Why: the upstream stream() trait does not surface a TokenUsage on its
    // own — providers emit usage only on the non-streaming path. Pass
    // zeroed usage so foreign sinks have a stable signature regardless of
    // provider capabilities.
    if let Err(sink_err) = sink
        .on_done(last_finish_reason, TokenUsage::default())
        .await
    {
        let _ = sink.on_error(clone_error(&sink_err)).await;
    }
    Ok(())
}

/// Synchronous variant of [`complete_streaming`] — blocks the current
/// thread on the shared Tokio runtime.
///
/// Handy for Ruby scripts and quick Go main fns where async machinery is
/// overkill. The sink's `async` methods still run on the shared runtime
/// (they're just driven synchronously from the caller's thread).
#[uniffi::export]
pub fn complete_streaming_blocking(
    model: Arc<CompletionModel>,
    request: CompletionRequest,
    sink: Arc<dyn CompletionStreamSink>,
) -> BlazenResult<()> {
    runtime().block_on(complete_streaming(model, request, sink))
}

// ---------------------------------------------------------------------------
// Internals
// ---------------------------------------------------------------------------

/// Produce a structurally-identical clone of a [`BlazenError`].
///
/// `BlazenError` is `#[uniffi(flat_error)]` and not `Clone`, but the sink
/// callback chain needs to deliver an error to both `on_chunk` callers and
/// `on_error` in a few edge cases. Re-builds the variant by hand rather
/// than going through `to_string()` so structured fields survive.
fn clone_error(err: &BlazenError) -> BlazenError {
    match err {
        BlazenError::Auth { message } => BlazenError::Auth {
            message: message.clone(),
        },
        BlazenError::RateLimit {
            message,
            retry_after_ms,
        } => BlazenError::RateLimit {
            message: message.clone(),
            retry_after_ms: *retry_after_ms,
        },
        BlazenError::Timeout {
            message,
            elapsed_ms,
        } => BlazenError::Timeout {
            message: message.clone(),
            elapsed_ms: *elapsed_ms,
        },
        BlazenError::Validation { message } => BlazenError::Validation {
            message: message.clone(),
        },
        BlazenError::ContentPolicy { message } => BlazenError::ContentPolicy {
            message: message.clone(),
        },
        BlazenError::Unsupported { message } => BlazenError::Unsupported {
            message: message.clone(),
        },
        BlazenError::Compute { message } => BlazenError::Compute {
            message: message.clone(),
        },
        BlazenError::Media { message } => BlazenError::Media {
            message: message.clone(),
        },
        BlazenError::Provider {
            kind,
            message,
            provider,
            status,
            endpoint,
            request_id,
            detail,
            retry_after_ms,
        } => BlazenError::Provider {
            kind: kind.clone(),
            message: message.clone(),
            provider: provider.clone(),
            status: *status,
            endpoint: endpoint.clone(),
            request_id: request_id.clone(),
            detail: detail.clone(),
            retry_after_ms: *retry_after_ms,
        },
        BlazenError::Workflow { message } => BlazenError::Workflow {
            message: message.clone(),
        },
        BlazenError::Tool { message } => BlazenError::Tool {
            message: message.clone(),
        },
        BlazenError::Peer { kind, message } => BlazenError::Peer {
            kind: kind.clone(),
            message: message.clone(),
        },
        BlazenError::Persist { message } => BlazenError::Persist {
            message: message.clone(),
        },
        BlazenError::Prompt { kind, message } => BlazenError::Prompt {
            kind: kind.clone(),
            message: message.clone(),
        },
        BlazenError::Memory { kind, message } => BlazenError::Memory {
            kind: kind.clone(),
            message: message.clone(),
        },
        BlazenError::Cache { kind, message } => BlazenError::Cache {
            kind: kind.clone(),
            message: message.clone(),
        },
        BlazenError::Cancelled => BlazenError::Cancelled,
        BlazenError::Internal { message } => BlazenError::Internal {
            message: message.clone(),
        },
    }
}
