//! LLM-specific events for streaming integration with the workflow engine.
//!
//! These events allow workflow steps to observe streaming progress from LLM
//! providers in real time via the standard event infrastructure.

use std::any::Any;

use serde::{Deserialize, Serialize};
use zagents_events::{AnyEvent, Event};

use crate::types::TokenUsage;

// ---------------------------------------------------------------------------
// StreamChunkEvent
// ---------------------------------------------------------------------------

/// Emitted for each incremental chunk received during a streaming completion.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamChunkEvent {
    /// The incremental text content from this chunk.
    pub delta: String,
    /// Present in the final chunk to indicate why generation stopped.
    pub finish_reason: Option<String>,
    /// The model that produced this chunk.
    pub model: String,
}

impl Event for StreamChunkEvent {
    fn event_type() -> &'static str
    where
        Self: Sized,
    {
        "zagents_llm::StreamChunkEvent"
    }

    fn event_type_id(&self) -> &'static str {
        "zagents_llm::StreamChunkEvent"
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn clone_boxed(&self) -> Box<dyn AnyEvent> {
        Box::new(self.clone())
    }

    fn to_json(&self) -> serde_json::Value {
        serde_json::to_value(self).expect("StreamChunkEvent serialization should never fail")
    }
}

// ---------------------------------------------------------------------------
// StreamCompleteEvent
// ---------------------------------------------------------------------------

/// Emitted once a streaming completion has fully finished.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamCompleteEvent {
    /// The concatenated full text of the streamed response.
    pub full_text: String,
    /// Token usage statistics, if the provider reported them.
    pub usage: Option<TokenUsage>,
    /// The model that produced the response.
    pub model: String,
}

impl Event for StreamCompleteEvent {
    fn event_type() -> &'static str
    where
        Self: Sized,
    {
        "zagents_llm::StreamCompleteEvent"
    }

    fn event_type_id(&self) -> &'static str {
        "zagents_llm::StreamCompleteEvent"
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn clone_boxed(&self) -> Box<dyn AnyEvent> {
        Box::new(self.clone())
    }

    fn to_json(&self) -> serde_json::Value {
        serde_json::to_value(self).expect("StreamCompleteEvent serialization should never fail")
    }
}
