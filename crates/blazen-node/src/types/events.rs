//! Typed class wrappers for built-in workflow events and LLM streaming events.
//!
//! The plain `#[napi(object)]` shapes for [`blazen_events::InputRequestEvent`]
//! and [`blazen_events::InputResponseEvent`] live in
//! [`crate::workflow::events_typed`]. This module adds:
//!
//! - [`JsStartEventClass`] / [`JsStopEventClass`] -- ergonomic class-form
//!   wrappers around the built-in [`blazen_events::StartEvent`] /
//!   [`blazen_events::StopEvent`] types so callers can construct them via
//!   `new StartEvent({ data })` / `new StopEvent({ result })` and inspect
//!   `.kind` / `.data` / `.result` / `.toJSON()` without going through the
//!   flat-object event convention used by the step-handler return path.
//! - [`JsStreamChunkEvent`] / [`JsStreamCompleteEvent`] -- typed mirrors of
//!   [`blazen_llm::StreamChunkEvent`] / [`blazen_llm::StreamCompleteEvent`]
//!   for use with the workflow event bus.
//! - [`JsAgentConfig`] / [`JsAgentEvent`] -- typed wrappers for the agent
//!   loop configuration and per-iteration events. The existing
//!   [`crate::agent::JsAgentRunOptions`] flat options object stays as the
//!   ergonomic JS surface for [`crate::agent::run_agent`]; this typed
//!   `AgentConfig` class exists for callers who want to build configuration
//!   imperatively and inspect it.

use napi_derive::napi;

use blazen_llm::events::{
    StreamChunkEvent as RustStreamChunkEvent, StreamCompleteEvent as RustStreamCompleteEvent,
};

use crate::generated::JsTokenUsage;

// ---------------------------------------------------------------------------
// JsStartEventClass
// ---------------------------------------------------------------------------

/// Options for constructing a [`JsStartEventClass`] from JavaScript.
#[napi(object)]
pub struct StartEventOptions {
    /// Arbitrary payload passed into the workflow at start.
    pub data: Option<serde_json::Value>,
}

/// Class-form wrapper for [`blazen_events::StartEvent`].
///
/// Mirrors the plain-object form `{ type: "blazen::StartEvent", ...data }`
/// used by the step-handler return path while exposing a constructor and
/// typed getters for ergonomic JS use.
///
/// ```javascript
/// const ev = new StartEvent({ data: { topic: "weather" } });
/// ev.kind;     // "blazen::StartEvent"
/// ev.data;     // { topic: "weather" }
/// ev.toJSON(); // { type: "blazen::StartEvent", topic: "weather" }
/// ```
#[napi(js_name = "StartEvent")]
pub struct JsStartEventClass {
    inner: blazen_events::StartEvent,
}

#[napi]
#[allow(clippy::must_use_candidate)]
impl JsStartEventClass {
    /// Build a new `StartEvent`.
    #[napi(constructor)]
    pub fn new(options: Option<StartEventOptions>) -> Self {
        let data = options
            .and_then(|o| o.data)
            .unwrap_or(serde_json::Value::Null);
        Self {
            inner: blazen_events::StartEvent { data },
        }
    }

    /// The canonical event type identifier (`"blazen::StartEvent"`).
    #[napi(getter)]
    pub fn kind(&self) -> String {
        "blazen::StartEvent".to_owned()
    }

    /// The arbitrary payload attached to this event.
    #[napi(getter)]
    pub fn data(&self) -> serde_json::Value {
        self.inner.data.clone()
    }

    /// Convert to the flat `{ type, ...data }` JS object form used by the
    /// step-handler return path.
    #[napi(js_name = "toJSON")]
    pub fn to_json(&self) -> serde_json::Value {
        let mut obj = match &self.inner.data {
            serde_json::Value::Object(map) => map.clone(),
            other => {
                let mut m = serde_json::Map::new();
                m.insert("data".to_owned(), other.clone());
                m
            }
        };
        obj.insert(
            "type".to_owned(),
            serde_json::Value::String("blazen::StartEvent".to_owned()),
        );
        serde_json::Value::Object(obj)
    }
}

// ---------------------------------------------------------------------------
// JsStopEventClass
// ---------------------------------------------------------------------------

/// Options for constructing a [`JsStopEventClass`] from JavaScript.
#[napi(object)]
pub struct StopEventOptions {
    /// The final result of the workflow.
    pub result: Option<serde_json::Value>,
}

/// Class-form wrapper for [`blazen_events::StopEvent`].
///
/// ```javascript
/// const ev = new StopEvent({ result: { answer: 42 } });
/// ev.kind;     // "blazen::StopEvent"
/// ev.result;   // { answer: 42 }
/// ev.toJSON(); // { type: "blazen::StopEvent", result: { answer: 42 } }
/// ```
#[napi(js_name = "StopEvent")]
pub struct JsStopEventClass {
    inner: blazen_events::StopEvent,
}

#[napi]
#[allow(clippy::must_use_candidate)]
impl JsStopEventClass {
    /// Build a new `StopEvent`.
    #[napi(constructor)]
    pub fn new(options: Option<StopEventOptions>) -> Self {
        let result = options
            .and_then(|o| o.result)
            .unwrap_or(serde_json::Value::Null);
        Self {
            inner: blazen_events::StopEvent { result },
        }
    }

    /// The canonical event type identifier (`"blazen::StopEvent"`).
    #[napi(getter)]
    pub fn kind(&self) -> String {
        "blazen::StopEvent".to_owned()
    }

    /// The workflow result attached to this event.
    #[napi(getter)]
    pub fn result(&self) -> serde_json::Value {
        self.inner.result.clone()
    }

    /// Convert to the flat `{ type, result }` JS object form used by the
    /// step-handler return path.
    #[napi(js_name = "toJSON")]
    pub fn to_json(&self) -> serde_json::Value {
        let mut obj = serde_json::Map::new();
        obj.insert(
            "type".to_owned(),
            serde_json::Value::String("blazen::StopEvent".to_owned()),
        );
        obj.insert("result".to_owned(), self.inner.result.clone());
        serde_json::Value::Object(obj)
    }
}

// ---------------------------------------------------------------------------
// JsStreamChunkEvent
// ---------------------------------------------------------------------------

/// Emitted for each incremental chunk received during a streaming completion.
///
/// Mirrors [`blazen_llm::StreamChunkEvent`] for the workflow event bus.
#[napi(object, js_name = "StreamChunkEvent")]
pub struct JsStreamChunkEvent {
    /// The incremental text content from this chunk.
    pub delta: String,
    /// Present in the final chunk to indicate why generation stopped.
    #[napi(js_name = "finishReason")]
    pub finish_reason: Option<String>,
    /// The model that produced this chunk.
    pub model: String,
}

impl From<RustStreamChunkEvent> for JsStreamChunkEvent {
    fn from(event: RustStreamChunkEvent) -> Self {
        Self {
            delta: event.delta,
            finish_reason: event.finish_reason,
            model: event.model,
        }
    }
}

impl From<JsStreamChunkEvent> for RustStreamChunkEvent {
    fn from(event: JsStreamChunkEvent) -> Self {
        Self {
            delta: event.delta,
            finish_reason: event.finish_reason,
            model: event.model,
        }
    }
}

// ---------------------------------------------------------------------------
// JsStreamCompleteEvent
// ---------------------------------------------------------------------------

/// Emitted once a streaming completion has fully finished.
///
/// Mirrors [`blazen_llm::StreamCompleteEvent`] for the workflow event bus.
#[napi(object, js_name = "StreamCompleteEvent")]
pub struct JsStreamCompleteEvent {
    /// The concatenated full text of the streamed response.
    #[napi(js_name = "fullText")]
    pub full_text: String,
    /// Token usage statistics, if the provider reported them.
    pub usage: Option<JsTokenUsage>,
    /// The model that produced the response.
    pub model: String,
}

impl From<RustStreamCompleteEvent> for JsStreamCompleteEvent {
    fn from(event: RustStreamCompleteEvent) -> Self {
        Self {
            full_text: event.full_text,
            usage: event.usage.map(Into::into),
            model: event.model,
        }
    }
}

// ---------------------------------------------------------------------------
// JsAgentConfig
// ---------------------------------------------------------------------------

/// Typed configuration for the agentic tool execution loop.
///
/// Mirrors the [`blazen_llm::AgentConfig`] fields. The existing
/// [`crate::agent::JsAgentRunOptions`] plain-object form remains the
/// ergonomic JS surface for [`crate::agent::run_agent`]; this typed object
/// is the canonical data-shape mirror.
#[napi(object, js_name = "AgentConfig")]
pub struct JsAgentConfig {
    /// Maximum number of tool call rounds before forcing a stop.
    #[napi(js_name = "maxIterations")]
    pub max_iterations: u32,
    /// Whether to add the legacy implicit "finish" tool the model can call
    /// to exit early.
    #[napi(js_name = "addFinishTool")]
    pub add_finish_tool: bool,
    /// Suppress automatic registration of the built-in `finish_workflow`
    /// exit tool. Default `false`. Mirrors
    /// [`blazen_llm::AgentConfig::no_finish_tool`] (Wave 6).
    #[napi(js_name = "noFinishTool")]
    pub no_finish_tool: bool,
    /// Override the name of the built-in `finish_workflow` tool. `null`
    /// uses the canonical [`crate::agent::FINISH_WORKFLOW_TOOL_NAME`].
    /// Mirrors [`blazen_llm::AgentConfig::finish_tool_name`] (Wave 6).
    #[napi(js_name = "finishToolName")]
    pub finish_tool_name: Option<String>,
    /// Optional system prompt prepended to messages.
    #[napi(js_name = "systemPrompt")]
    pub system_prompt: Option<String>,
    /// Sampling temperature.
    pub temperature: Option<f64>,
    /// Maximum tokens per completion call.
    #[napi(js_name = "maxTokens")]
    pub max_tokens: Option<u32>,
    /// Maximum number of tool calls to execute concurrently within a single
    /// model response. `0` means unlimited.
    #[napi(js_name = "toolConcurrency")]
    pub tool_concurrency: u32,
}

impl Default for JsAgentConfig {
    fn default() -> Self {
        Self {
            max_iterations: 10,
            add_finish_tool: false,
            no_finish_tool: false,
            finish_tool_name: None,
            system_prompt: None,
            temperature: None,
            max_tokens: None,
            tool_concurrency: 0,
        }
    }
}

// ---------------------------------------------------------------------------
// JsAgentEvent
// ---------------------------------------------------------------------------

/// Typed agent loop event emitted to the [`run_agent_with_callback`]
/// observer.
///
/// `kind` is one of:
/// - `"toolCalled"` -- the model requested a tool invocation. `iteration`,
///   `toolName`, `toolCallId`, and `arguments` are populated.
/// - `"toolResult"` -- a tool execution completed. `iteration`, `toolName`,
///   and `result` are populated.
/// - `"iterationComplete"` -- the model produced a response. `iteration`
///   and `hadToolCalls` are populated.
#[napi(object, js_name = "AgentEvent")]
pub struct JsAgentEvent {
    /// Discriminant: `"toolCalled"`, `"toolResult"`, or `"iterationComplete"`.
    pub kind: String,
    /// Iteration index (0-based).
    pub iteration: u32,
    /// Tool name. Populated for `"toolCalled"` and `"toolResult"`.
    #[napi(js_name = "toolName")]
    pub tool_name: Option<String>,
    /// Tool call ID. Populated for `"toolCalled"`.
    #[napi(js_name = "toolCallId")]
    pub tool_call_id: Option<String>,
    /// Tool arguments. Populated for `"toolCalled"`.
    pub arguments: Option<serde_json::Value>,
    /// Tool result payload. Populated for `"toolResult"`.
    pub result: Option<serde_json::Value>,
    /// Whether this iteration contained tool calls. Populated for
    /// `"iterationComplete"`.
    #[napi(js_name = "hadToolCalls")]
    pub had_tool_calls: Option<bool>,
}

impl JsAgentEvent {
    /// Build a [`JsAgentEvent`] from a Rust [`blazen_llm::AgentEvent`].
    #[must_use]
    pub fn from_rust(event: blazen_llm::AgentEvent) -> Self {
        match event {
            blazen_llm::AgentEvent::ToolCalled {
                iteration,
                tool_call,
            } => Self {
                kind: "toolCalled".to_owned(),
                iteration,
                tool_name: Some(tool_call.name),
                tool_call_id: Some(tool_call.id),
                arguments: Some(tool_call.arguments),
                result: None,
                had_tool_calls: None,
            },
            blazen_llm::AgentEvent::ToolResult {
                iteration,
                tool_name,
                result,
            } => Self {
                kind: "toolResult".to_owned(),
                iteration,
                tool_name: Some(tool_name),
                tool_call_id: None,
                arguments: None,
                result: Some(result),
                had_tool_calls: None,
            },
            blazen_llm::AgentEvent::IterationComplete {
                iteration,
                had_tool_calls,
            } => Self {
                kind: "iterationComplete".to_owned(),
                iteration,
                tool_name: None,
                tool_call_id: None,
                arguments: None,
                result: None,
                had_tool_calls: Some(had_tool_calls),
            },
        }
    }
}
