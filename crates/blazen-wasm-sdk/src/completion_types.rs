//! TS-facing copies of the streaming-event surface from
//! [`blazen_llm::events`].
//!
//! [`blazen_llm::types::StreamChunk`], [`blazen_llm::types::CompletionRequest`],
//! and the message-content / media types already cross the WASM ABI as
//! [`tsify_next::Tsify`]-derived TypeScript interfaces because the
//! `blazen-llm` dep enables the `tsify` feature. The two streaming workflow
//! events ([`blazen_llm::events::StreamChunkEvent`] /
//! [`blazen_llm::events::StreamCompleteEvent`]) live behind the
//! `Box<dyn AnyEvent>` boundary so they need their own flattened wrappers
//! to reach JS.

use serde::{Deserialize, Serialize};
use tsify_next::Tsify;

use blazen_llm::events::{StreamChunkEvent, StreamCompleteEvent};
use blazen_llm::types::TokenUsage;

// ---------------------------------------------------------------------------
// WasmStreamChunkEvent
// ---------------------------------------------------------------------------

/// TS-facing copy of [`blazen_llm::events::StreamChunkEvent`].
///
/// Emitted for each incremental chunk received during a streaming
/// completion.
#[derive(Debug, Clone, Serialize, Deserialize, Tsify)]
#[tsify(into_wasm_abi, from_wasm_abi)]
pub struct WasmStreamChunkEvent {
    /// The incremental text content from this chunk.
    pub delta: String,
    /// Present in the final chunk to indicate why generation stopped.
    pub finish_reason: Option<String>,
    /// The model that produced this chunk.
    pub model: String,
}

impl From<StreamChunkEvent> for WasmStreamChunkEvent {
    fn from(value: StreamChunkEvent) -> Self {
        Self {
            delta: value.delta,
            finish_reason: value.finish_reason,
            model: value.model,
        }
    }
}

impl From<WasmStreamChunkEvent> for StreamChunkEvent {
    fn from(value: WasmStreamChunkEvent) -> Self {
        Self {
            delta: value.delta,
            finish_reason: value.finish_reason,
            model: value.model,
        }
    }
}

// ---------------------------------------------------------------------------
// WasmStreamCompleteEvent
// ---------------------------------------------------------------------------

/// TS-facing copy of [`blazen_llm::events::StreamCompleteEvent`].
///
/// Emitted once a streaming completion has fully finished.
#[derive(Debug, Clone, Serialize, Deserialize, Tsify)]
#[tsify(into_wasm_abi, from_wasm_abi)]
pub struct WasmStreamCompleteEvent {
    /// The concatenated full text of the streamed response.
    pub full_text: String,
    /// Token usage statistics, if the provider reported them.
    pub usage: Option<TokenUsage>,
    /// The model that produced the response.
    pub model: String,
}

impl From<StreamCompleteEvent> for WasmStreamCompleteEvent {
    fn from(value: StreamCompleteEvent) -> Self {
        Self {
            full_text: value.full_text,
            usage: value.usage,
            model: value.model,
        }
    }
}

impl From<WasmStreamCompleteEvent> for StreamCompleteEvent {
    fn from(value: WasmStreamCompleteEvent) -> Self {
        Self {
            full_text: value.full_text,
            usage: value.usage,
            model: value.model,
        }
    }
}
