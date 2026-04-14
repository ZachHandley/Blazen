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

use blazen_llm::agent::{AgentConfig, run_agent as rust_run_agent};
use blazen_llm::error::BlazenError;
use blazen_llm::traits::Tool;
use blazen_llm::types::{ChatMessage, ToolDefinition};

use crate::error::llm_error_to_napi;
use crate::providers::JsCompletionModel;
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
    /// Whether to add a built-in "finish" tool that the model can call to
    /// signal it has a final answer.
    #[napi(js_name = "addFinishTool")]
    pub add_finish_tool: Option<bool>,
    /// Maximum number of tool calls to execute concurrently within a single
    /// model response. `0` means unlimited (all in parallel). Defaults to 0.
    #[napi(js_name = "toolConcurrency")]
    pub tool_concurrency: Option<i32>,
}

// ---------------------------------------------------------------------------
// JsAgentResult
// ---------------------------------------------------------------------------

/// The result of an agent run.
#[napi(object)]
pub struct JsAgentResult {
    /// The final completion response from the model.
    pub response: JsCompletionResponse,
    /// Full message history including all tool calls and results.
    pub messages: Vec<serde_json::Value>,
    /// Number of tool-calling iterations that occurred.
    pub iterations: u32,
    /// Aggregated cost across all iterations, if available.
    #[napi(js_name = "totalCost")]
    pub total_cost: Option<f64>,
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
    ) -> std::result::Result<serde_json::Value, BlazenError> {
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

        Ok(result)
    }
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

    Ok(JsAgentResult {
        response: build_response(result.response),
        messages: js_messages,
        iterations: result.iterations,
        total_cost: result.total_cost,
    })
}
