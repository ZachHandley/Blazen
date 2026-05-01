//! JavaScript bindings for the agentic tool execution loop.
//!
//! Exposes [`run_agent`] which wraps the Rust [`blazen_llm::agent::run_agent`]
//! function, allowing Node.js / TypeScript users to run an LLM agent with
//! tool-calling capabilities.

use std::sync::Arc;

use napi::Status;
use napi::bindgen_prelude::*;
use napi::threadsafe_function::ThreadsafeFunction;
use napi_derive::napi;

use blazen_llm::agent::{
    AgentConfig, run_agent as rust_run_agent,
    run_agent_with_callback as rust_run_agent_with_callback,
};
use blazen_llm::error::BlazenError;
use blazen_llm::traits::Tool;
use blazen_llm::types::{ChatMessage, CompletionResponse, ToolDefinition};
use napi::threadsafe_function::ThreadsafeFunctionCallMode;

use crate::error::llm_error_to_napi;
use crate::providers::JsCompletionModel;
use crate::types::events::JsAgentEvent;
use crate::types::{JsChatMessage, JsCompletionResponse, build_response};

// ---------------------------------------------------------------------------
// Type alias for the tool handler ThreadsafeFunction
// ---------------------------------------------------------------------------

/// Tool handler: takes (toolName: string, arguments: object) and returns a
/// JSON-serializable result (or Promise thereof).
///
/// See [`crate::workflow`] for a detailed explanation of the generic parameters.
type ToolHandlerTsfn = ThreadsafeFunction<
    FnArgs<(String, serde_json::Value)>,
    Promise<serde_json::Value>,
    FnArgs<(String, serde_json::Value)>,
    Status,
    false,
    true,
>;

// ---------------------------------------------------------------------------
// Wave 6 — finish-workflow tool surface
// ---------------------------------------------------------------------------

/// Canonical name of the built-in `finish_workflow` exit tool. Mirrors
/// [`blazen_llm::FINISH_WORKFLOW_TOOL_NAME`].
#[napi]
pub const FINISH_WORKFLOW_TOOL_NAME: &str = blazen_llm::FINISH_WORKFLOW_TOOL_NAME;

/// Build a fresh JSON-Schema description of the built-in `finish_workflow`
/// exit tool. The shape mirrors [`blazen_llm::finish_workflow_tool`] —
/// callers that want to surface the same tool to a JS-side agent loop can
/// use this object as a [`JsToolDef`] entry.
#[napi(js_name = "finishWorkflowToolDef")]
#[must_use]
pub fn finish_workflow_tool_def() -> JsToolDef {
    let tool = blazen_llm::finish_workflow_tool();
    let def = tool.definition();
    JsToolDef {
        name: def.name,
        description: def.description,
        parameters: def.parameters,
    }
}

// ---------------------------------------------------------------------------
// JsToolDef
// ---------------------------------------------------------------------------

/// Describes a tool that the agent may invoke during its execution loop.
#[napi(object)]
pub struct JsToolDef {
    /// The unique name of the tool.
    pub name: String,
    /// A human-readable description of what the tool does.
    pub description: String,
    /// JSON Schema describing the tool's parameters.
    pub parameters: serde_json::Value,
}

// ---------------------------------------------------------------------------
// JsAgentRunOptions
// ---------------------------------------------------------------------------

/// Options for configuring an agent run.
#[derive(Default)]
#[napi(object)]
pub struct JsAgentRunOptions {
    /// Maximum number of tool-calling iterations before forcing a final answer.
    /// Defaults to 10.
    #[napi(js_name = "maxIterations")]
    pub max_iterations: Option<i32>,
    /// Optional system prompt prepended to the conversation.
    #[napi(js_name = "systemPrompt")]
    pub system_prompt: Option<String>,
    /// Sampling temperature (0.0 - 2.0).
    pub temperature: Option<f64>,
    /// Maximum tokens per completion call.
    #[napi(js_name = "maxTokens")]
    pub max_tokens: Option<i32>,
    /// Whether to add the legacy implicit "finish" tool the model can call
    /// to signal it has a final answer.
    #[napi(js_name = "addFinishTool")]
    pub add_finish_tool: Option<bool>,
    /// Suppress automatic registration of the built-in `finish_workflow`
    /// exit tool (default `false`, i.e. it is auto-added). Mirrors
    /// [`blazen_llm::AgentConfig::no_finish_tool`] (Wave 6).
    #[napi(js_name = "noFinishTool")]
    pub no_finish_tool: Option<bool>,
    /// Override the name of the built-in `finish_workflow` tool. Defaults
    /// to [`FINISH_WORKFLOW_TOOL_NAME`]. Mirrors
    /// [`blazen_llm::AgentConfig::finish_tool_name`] (Wave 6).
    #[napi(js_name = "finishToolName")]
    pub finish_tool_name: Option<String>,
    /// Maximum number of tool calls to execute concurrently within a single
    /// model response. `0` means unlimited (all in parallel). Defaults to 0.
    #[napi(js_name = "toolConcurrency")]
    pub tool_concurrency: Option<i32>,
}

// ---------------------------------------------------------------------------
// JsAgentResult
// ---------------------------------------------------------------------------

/// The result of an agent run.
//
// The `response` field stores the source-of-truth Rust `CompletionResponse`
// (which is `Clone`) and rebuilds the JS-shape on each getter call. This
// mirrors `PyAgentResult` (which stores the inner Rust type and rebuilds the
// Python wrapper per access) and avoids requiring `Clone` on
// `JsCompletionResponse` (a `#[napi(object)]` shape whose transitively
// generated mirror types are not `Clone`).
#[napi(js_name = "AgentResult")]
pub struct JsAgentResult {
    response: CompletionResponse,
    messages: Vec<serde_json::Value>,
    iterations: u32,
    total_cost: Option<f64>,
}

#[napi]
#[allow(clippy::must_use_candidate)]
impl JsAgentResult {
    /// The final completion response from the model.
    #[napi(getter)]
    pub fn response(&self) -> JsCompletionResponse {
        build_response(self.response.clone())
    }

    /// Full message history including all tool calls and results.
    #[napi(getter)]
    pub fn messages(&self) -> Vec<serde_json::Value> {
        self.messages.clone()
    }

    /// Number of tool-calling iterations that occurred.
    #[napi(getter)]
    pub fn iterations(&self) -> u32 {
        self.iterations
    }

    /// Aggregated cost across all iterations, if available.
    #[napi(getter, js_name = "totalCost")]
    pub fn total_cost(&self) -> Option<f64> {
        self.total_cost
    }

    /// String representation matching the Python `AgentResult.__repr__`.
    #[napi(js_name = "toString")]
    pub fn display_string(&self) -> String {
        format!(
            "AgentResult(iterations={}, cost={:?})",
            self.iterations, self.total_cost
        )
    }
}

impl JsAgentResult {
    pub(crate) fn new(
        response: CompletionResponse,
        messages: Vec<serde_json::Value>,
        iterations: u32,
        total_cost: Option<f64>,
    ) -> Self {
        Self {
            response,
            messages,
            iterations,
            total_cost,
        }
    }
}

// ---------------------------------------------------------------------------
// JsToolWrapper -- bridges JS tool handler to the Rust Tool trait
// ---------------------------------------------------------------------------

/// A wrapper that implements [`Tool`] by delegating to a JavaScript callback.
///
/// The handler is a `ThreadsafeFunction` that receives `(toolName, arguments)`
/// and returns the tool result as a JSON value (or Promise thereof).
struct JsToolWrapper {
    def: ToolDefinition,
    handler: Arc<ToolHandlerTsfn>,
}

#[async_trait::async_trait]
impl Tool for JsToolWrapper {
    fn definition(&self) -> ToolDefinition {
        self.def.clone()
    }

    async fn execute(
        &self,
        arguments: serde_json::Value,
    ) -> std::result::Result<blazen_llm::types::ToolOutput<serde_json::Value>, BlazenError> {
        let name = self.def.name.clone();

        // Call the JS handler with (toolName, arguments).
        // call_async returns a Future<Result<Promise<Value>>>.
        // We await the outer call, then await the inner promise.
        let promise = self
            .handler
            .call_async(FnArgs::from((name, arguments)))
            .await
            .map_err(|e| BlazenError::tool_error(e.to_string()))?;

        let result = promise
            .await
            .map_err(|e| BlazenError::tool_error(e.to_string()))?;

        // Dispatch on shape: if the value is an object with an explicit
        // `data` key, treat it as a structured `ToolOutput` and decode an
        // optional `llmOverride`. The `data`-key guard prevents
        // misinterpreting an arbitrary user dict like `{"items": [1,2,3]}`
        // as a ToolOutput. Otherwise, auto-wrap the bare value with no
        // override.
        if let Some(obj) = result.as_object()
            && obj.contains_key("data")
        {
            return decode_structured_tool_output(result)
                .map_err(|e| BlazenError::tool_error(e.to_string()));
        }
        Ok(result.into())
    }
}

/// Decode a JS object of shape `{ data, llmOverride? }` into a Rust
/// [`blazen_llm::types::ToolOutput`].
///
/// The wrapper is `#[napi(object)]` and therefore not `Deserialize`, so we
/// pull `data` and `llmOverride` out by hand and let the inner [`LlmPayload`]
/// (which derives `Deserialize`) parse itself. The serde tag/rename rules on
/// [`LlmPayload`] match the JS shape: `{ kind: "text" | "json" | ... }` plus
/// the variant-specific fields.
fn decode_structured_tool_output(
    value: serde_json::Value,
) -> napi::Result<blazen_llm::types::ToolOutput<serde_json::Value>> {
    let serde_json::Value::Object(mut obj) = value else {
        return Err(napi::Error::from_reason(
            "structured ToolOutput must be a JS object with a `data` field",
        ));
    };

    let data = obj
        .remove("data")
        .ok_or_else(|| napi::Error::from_reason("structured ToolOutput is missing `data`"))?;

    // Accept either `llmOverride` (camelCase preferred) or `llm_override`
    // (snake_case). Strip explicit nulls so they round-trip as "no override".
    let raw_override = obj
        .remove("llmOverride")
        .or_else(|| obj.remove("llm_override"));
    let llm_override = match raw_override {
        None | Some(serde_json::Value::Null) => None,
        Some(v) => Some(
            serde_json::from_value::<blazen_llm::types::LlmPayload>(v)
                .map_err(|e| napi::Error::from_reason(format!("invalid llmOverride: {e}")))?,
        ),
    };

    Ok(blazen_llm::types::ToolOutput { data, llm_override })
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Run an agentic tool execution loop.
///
/// The agent repeatedly calls the model, executes any tool calls via the
/// `toolHandler` callback, feeds results back, and repeats until the model
/// stops calling tools or `maxIterations` is reached.
///
/// The `toolHandler` callback receives `(toolName: string, arguments: object)`
/// and must return the tool result as a JSON-serializable value (or a Promise
/// that resolves to one).
///
/// ```typescript
/// import { CompletionModel, ChatMessage, runAgent } from 'blazen';
///
/// const model = CompletionModel.openai({ apiKey: "sk-..." });
///
/// const result = await runAgent(
///   model,
///   [ChatMessage.user("What is the weather in NYC?")],
///   [{ name: "getWeather", description: "Get weather", parameters: { type: "object", properties: { city: { type: "string" } }, required: ["city"] } }],
///   async (toolName, args) => {
///     if (toolName === "getWeather") return { temp: 72, condition: "sunny" };
///     throw new Error(`Unknown tool: ${toolName}`);
///   },
///   { maxIterations: 5 }
/// );
/// ```
#[napi(js_name = "runAgent")]
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::missing_errors_doc
)]
pub async fn run_agent(
    model: &JsCompletionModel,
    messages: Vec<&JsChatMessage>,
    tools: Vec<JsToolDef>,
    tool_handler: ToolHandlerTsfn,
    options: Option<JsAgentRunOptions>,
) -> napi::Result<JsAgentResult> {
    let rust_messages: Vec<ChatMessage> = messages.iter().map(|m| m.inner.clone()).collect();

    let handler = Arc::new(tool_handler);

    let rust_tools: Vec<Arc<dyn Tool>> = tools
        .into_iter()
        .map(|t| {
            Arc::new(JsToolWrapper {
                def: ToolDefinition {
                    name: t.name,
                    description: t.description,
                    parameters: t.parameters,
                },
                handler: Arc::clone(&handler),
            }) as Arc<dyn Tool>
        })
        .collect();

    let opts = options.unwrap_or_default();

    let mut config =
        AgentConfig::new(rust_tools).with_max_iterations(opts.max_iterations.unwrap_or(10) as u32);

    if let Some(sp) = opts.system_prompt {
        config = config.with_system_prompt(sp);
    }
    if let Some(t) = opts.temperature {
        config = config.with_temperature(t as f32);
    }
    if let Some(mt) = opts.max_tokens {
        config = config.with_max_tokens(mt as u32);
    }
    if opts.add_finish_tool.unwrap_or(false) {
        config = config.with_finish_tool();
    }
    if opts.no_finish_tool.unwrap_or(false) {
        config = config.no_finish_tool();
    }
    if let Some(name) = opts.finish_tool_name {
        config = config.finish_tool_name(name);
    }
    if let Some(tc) = opts.tool_concurrency {
        config = config.with_tool_concurrency(tc as usize);
    }

    let inner = model.inner.as_ref().ok_or_else(|| {
        napi::Error::from_reason(
            "runAgent() is not supported on subclassed CompletionModel instances",
        )
    })?;
    let result = rust_run_agent(inner.as_ref(), rust_messages, config)
        .await
        .map_err(llm_error_to_napi)?;

    let js_messages: Vec<serde_json::Value> = result
        .messages
        .iter()
        .map(|m| {
            serde_json::to_value(m).map_err(|e| {
                napi::Error::new(
                    Status::GenericFailure,
                    format!("Failed to serialize ChatMessage: {e}"),
                )
            })
        })
        .collect::<napi::Result<Vec<_>>>()?;

    Ok(JsAgentResult::new(
        result.response,
        js_messages,
        result.iterations,
        result.total_cost,
    ))
}

// ---------------------------------------------------------------------------
// runAgentWithCallback
// ---------------------------------------------------------------------------

/// Event observer callback: receives a typed [`JsAgentEvent`] for each
/// tool call, tool result, and iteration the agent loop emits.
///
/// `Weak = true` so it does not prevent Node.js from exiting once the
/// agent run resolves. `CalleeHandled = false` to avoid the error-first
/// callback convention.
type AgentEventCallbackTsfn =
    ThreadsafeFunction<JsAgentEvent, Unknown<'static>, JsAgentEvent, Status, false, true>;

/// Run an agent loop with an event-observer callback.
///
/// Identical to [`run_agent`] but additionally invokes `onEvent` for each
/// [`JsAgentEvent`] emitted during the loop. The callback is fire-and-forget
/// — its return value is ignored, and any exception it raises does not
/// abort the loop.
///
/// ```typescript
/// import { CompletionModel, ChatMessage, runAgentWithCallback } from 'blazen';
///
/// const model = CompletionModel.openai({ apiKey: "sk-..." });
///
/// const result = await runAgentWithCallback(
///   model,
///   [ChatMessage.user("What is the weather in NYC?")],
///   [{ name: "getWeather", description: "Get weather", parameters: { type: "object" } }],
///   async (toolName, args) => ({ temp: 72, condition: "sunny" }),
///   (event) => {
///     console.log(event.kind, event.iteration, event.toolName);
///   },
///   { maxIterations: 5 },
/// );
/// ```
#[napi(js_name = "runAgentWithCallback")]
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::missing_errors_doc,
    clippy::too_many_arguments
)]
pub async fn run_agent_with_callback(
    model: &JsCompletionModel,
    messages: Vec<&JsChatMessage>,
    tools: Vec<JsToolDef>,
    tool_handler: ToolHandlerTsfn,
    on_event: AgentEventCallbackTsfn,
    options: Option<JsAgentRunOptions>,
) -> napi::Result<JsAgentResult> {
    let rust_messages: Vec<ChatMessage> = messages.iter().map(|m| m.inner.clone()).collect();

    let handler = Arc::new(tool_handler);

    let rust_tools: Vec<Arc<dyn Tool>> = tools
        .into_iter()
        .map(|t| {
            Arc::new(JsToolWrapper {
                def: ToolDefinition {
                    name: t.name,
                    description: t.description,
                    parameters: t.parameters,
                },
                handler: Arc::clone(&handler),
            }) as Arc<dyn Tool>
        })
        .collect();

    let opts = options.unwrap_or_default();

    let mut config =
        AgentConfig::new(rust_tools).with_max_iterations(opts.max_iterations.unwrap_or(10) as u32);

    if let Some(sp) = opts.system_prompt {
        config = config.with_system_prompt(sp);
    }
    if let Some(t) = opts.temperature {
        config = config.with_temperature(t as f32);
    }
    if let Some(mt) = opts.max_tokens {
        config = config.with_max_tokens(mt as u32);
    }
    if opts.add_finish_tool.unwrap_or(false) {
        config = config.with_finish_tool();
    }
    if opts.no_finish_tool.unwrap_or(false) {
        config = config.no_finish_tool();
    }
    if let Some(name) = opts.finish_tool_name {
        config = config.finish_tool_name(name);
    }
    if let Some(tc) = opts.tool_concurrency {
        config = config.with_tool_concurrency(tc as usize);
    }

    let inner = model.inner.as_ref().ok_or_else(|| {
        napi::Error::from_reason(
            "runAgentWithCallback() is not supported on subclassed CompletionModel instances",
        )
    })?;

    let on_event_arc = Arc::new(on_event);

    let on_event_fn = move |ev: blazen_llm::AgentEvent| {
        let typed = JsAgentEvent::from_rust(ev);
        on_event_arc.call(typed, ThreadsafeFunctionCallMode::Blocking);
    };

    let result = rust_run_agent_with_callback(inner.as_ref(), rust_messages, config, on_event_fn)
        .await
        .map_err(llm_error_to_napi)?;

    let js_messages: Vec<serde_json::Value> = result
        .messages
        .iter()
        .map(|m| {
            serde_json::to_value(m).map_err(|e| {
                napi::Error::new(
                    Status::GenericFailure,
                    format!("Failed to serialize ChatMessage: {e}"),
                )
            })
        })
        .collect::<napi::Result<Vec<_>>>()?;

    Ok(JsAgentResult::new(
        result.response,
        js_messages,
        result.iterations,
        result.total_cost,
    ))
}
