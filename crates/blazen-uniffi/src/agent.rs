//! LLM agent loop for the UniFFI bindings.
//!
//! Wraps upstream [`blazen_llm::agent::run_agent`] so foreign callers can run
//! the canonical "completion → execute tool calls → feed results back →
//! repeat" loop without re-implementing it in Go, Swift, Kotlin, or Ruby.
//!
//! ## Shape vs. `blazen-py`
//!
//! The Python binding ([`blazen-py`'s `agent.rs`](crates/blazen-py/src/agent.rs))
//! attaches one Python callable per [`PyToolDef`] and bridges each one through
//! `PyToolWrapper`. Across the UniFFI surface a single foreign-implementable
//! [`ToolHandler`] dispatcher is more idiomatic — Go / Swift / Kotlin / Ruby
//! cannot conjure a fresh trait object per tool name without a heap of
//! boilerplate. The handler receives the tool name + JSON-encoded arguments
//! and returns a JSON-encoded result string; the wrapper synthesises one
//! `Arc<dyn CoreTool>` per declared tool that forwards into the handler.
//!
//! ## Reused upstream API
//!
//! The body of the loop is upstream [`blazen_llm::agent::run_agent`] — we do
//! not duplicate iteration / fan-out / exit-tool logic here.

use std::sync::Arc;

use blazen_llm::agent::{
    AgentConfig as CoreAgentConfig, AgentResult as CoreAgentResult, run_agent as core_run_agent,
};
use blazen_llm::error::BlazenError as CoreBlazenError;
use blazen_llm::traits::Tool as CoreTool;
use blazen_llm::types::{
    ChatMessage as CoreChatMessage, ToolDefinition as CoreToolDefinition,
    ToolOutput as CoreToolOutput,
};

use crate::errors::{BlazenError, BlazenResult};
use crate::llm::{CompletionModel, TokenUsage, Tool};
use crate::runtime::runtime;

// ---------------------------------------------------------------------------
// Foreign-implementable tool dispatcher
// ---------------------------------------------------------------------------

/// Foreign-language tool executor invoked by the agent loop.
///
/// Implementations receive the LLM's chosen `tool_name` plus a JSON-encoded
/// `arguments_json` string and return a JSON-encoded result string that is
/// fed back to the model on the next turn.
///
/// ## Errors
///
/// Returning [`BlazenError`] from `execute` aborts the agent loop with that
/// error. Use [`BlazenError::Tool`] for handler-side failures; the message is
/// surfaced verbatim to the foreign caller.
#[uniffi::export(with_foreign)]
#[async_trait::async_trait]
pub trait ToolHandler: Send + Sync {
    /// Execute the named tool with JSON-encoded arguments.
    ///
    /// The returned string is JSON-encoded and round-trips back into the LLM
    /// as the tool result on the next turn. Return `"null"` (the JSON literal)
    /// when the tool produced no useful result.
    async fn execute(&self, tool_name: String, arguments_json: String) -> BlazenResult<String>;
}

// ---------------------------------------------------------------------------
// Wire-format result
// ---------------------------------------------------------------------------

/// Outcome of an [`Agent::run`] call.
///
/// `total_cost_usd` is the sum of per-iteration costs; when the provider did
/// not report cost data it is `0.0` (the wire format does not distinguish
/// "zero" from "unknown" — foreign callers wanting fidelity should pull
/// pricing from telemetry).
#[derive(Debug, Clone, uniffi::Record)]
pub struct AgentResult {
    /// The model's final textual response after the loop terminates.
    pub final_message: String,
    /// Number of iterations (LLM round-trips) the loop executed before
    /// terminating.
    pub iterations: u32,
    /// Total number of tool calls executed across all iterations.
    pub tool_call_count: u32,
    /// Aggregated token usage across every completion call in the loop.
    pub total_usage: TokenUsage,
    /// Aggregated USD cost across every completion call in the loop.
    pub total_cost_usd: f64,
}

// ---------------------------------------------------------------------------
// Internal adapter: ToolHandler -> dyn CoreTool
// ---------------------------------------------------------------------------

/// Bridges a single tool definition + the multi-tool [`ToolHandler`] into the
/// upstream [`CoreTool`] trait. One instance is constructed per tool declared
/// to [`Agent::new`].
struct ToolHandlerAdapter {
    definition: CoreToolDefinition,
    handler: Arc<dyn ToolHandler>,
}

#[async_trait::async_trait]
impl CoreTool for ToolHandlerAdapter {
    fn definition(&self) -> CoreToolDefinition {
        self.definition.clone()
    }

    async fn execute(
        &self,
        arguments: serde_json::Value,
    ) -> Result<CoreToolOutput<serde_json::Value>, CoreBlazenError> {
        let arguments_json = arguments.to_string();
        let tool_name = self.definition.name.clone();
        let raw = self
            .handler
            .execute(tool_name, arguments_json)
            .await
            .map_err(|e| CoreBlazenError::tool_error(e.to_string()))?;
        let value = if raw.is_empty() {
            serde_json::Value::Null
        } else {
            serde_json::from_str(&raw).map_err(|e| {
                CoreBlazenError::tool_error(format!("invalid JSON returned by tool handler: {e}"))
            })?
        };
        Ok(CoreToolOutput::new(value))
    }
}

// ---------------------------------------------------------------------------
// Agent opaque handle
// ---------------------------------------------------------------------------

/// A configured LLM agent that drives the standard tool-execution loop.
///
/// Construct via [`Agent::new`] with a model, optional system prompt, the
/// list of [`Tool`] definitions the model may call, a foreign-language
/// [`ToolHandler`] that executes those tools, and a `max_iterations` budget.
/// Then invoke [`run`](Self::run) (async) or
/// [`run_blocking`](Self::run_blocking) (sync).
///
/// Reuse a single `Agent` across calls when configuration is stable — the
/// underlying model handle is reference-counted, so cloning is cheap.
#[derive(uniffi::Object)]
pub struct Agent {
    model: Arc<CompletionModel>,
    system_prompt: Option<String>,
    tools: Vec<Tool>,
    handler: Arc<dyn ToolHandler>,
    max_iterations: u32,
}

#[uniffi::export]
impl Agent {
    /// Build an agent.
    ///
    /// - `model`: the completion model to drive.
    /// - `system_prompt`: optional system prompt prepended to the conversation
    ///   on every iteration.
    /// - `tools`: the tools the model may invoke. The names embedded in each
    ///   [`Tool`] must match the names the [`ToolHandler`] dispatches on.
    /// - `tool_handler`: the foreign-language executor for tool calls.
    /// - `max_iterations`: hard cap on LLM round-trips before the loop is
    ///   forced to produce a final answer.
    #[uniffi::constructor]
    #[must_use]
    pub fn new(
        model: Arc<CompletionModel>,
        system_prompt: Option<String>,
        tools: Vec<Tool>,
        tool_handler: Arc<dyn ToolHandler>,
        max_iterations: u32,
    ) -> Arc<Self> {
        Arc::new(Self {
            model,
            system_prompt,
            tools,
            handler: tool_handler,
            max_iterations,
        })
    }
}

#[uniffi::export(async_runtime = "tokio")]
impl Agent {
    /// Run the agent loop until the model produces a final answer (no tool
    /// calls) or `max_iterations` is reached.
    ///
    /// `user_input` is sent as the initial user-role message. The final
    /// answer is returned in [`AgentResult::final_message`].
    pub async fn run(self: Arc<Self>, user_input: String) -> BlazenResult<AgentResult> {
        let core_tools = build_core_tools(&self.tools, &self.handler)?;
        let mut config = CoreAgentConfig::new(core_tools).with_max_iterations(self.max_iterations);
        if let Some(prompt) = self.system_prompt.as_ref() {
            config = config.with_system_prompt(prompt.clone());
        }
        let messages = vec![CoreChatMessage::user(user_input)];
        let result = core_run_agent(self.model.inner.as_ref(), messages, config)
            .await
            .map_err(BlazenError::from)?;
        Ok(agent_result_from_core(&result))
    }
}

#[uniffi::export]
impl Agent {
    /// Synchronous variant of [`run`](Self::run) — blocks the current thread
    /// on the shared Tokio runtime.
    pub fn run_blocking(self: Arc<Self>, user_input: String) -> BlazenResult<AgentResult> {
        let this = Arc::clone(&self);
        runtime().block_on(async move { this.run(user_input).await })
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Materialise the wire [`Tool`] list into upstream `Arc<dyn CoreTool>`
/// objects, each forwarding execution through the shared [`ToolHandler`].
fn build_core_tools(
    tools: &[Tool],
    handler: &Arc<dyn ToolHandler>,
) -> BlazenResult<Vec<Arc<dyn CoreTool>>> {
    tools
        .iter()
        .map(|t| {
            let definition = CoreToolDefinition::try_from(t.clone())?;
            let adapter = ToolHandlerAdapter {
                definition,
                handler: Arc::clone(handler),
            };
            Ok(Arc::new(adapter) as Arc<dyn CoreTool>)
        })
        .collect()
}

/// Convert an upstream [`CoreAgentResult`] into the wire [`AgentResult`].
fn agent_result_from_core(result: &CoreAgentResult) -> AgentResult {
    let final_message = result.response.content.clone().unwrap_or_default();
    let tool_call_count = count_tool_calls(&result.messages);
    let total_usage = result
        .total_usage
        .clone()
        .map(TokenUsage::from)
        .unwrap_or_default();
    AgentResult {
        final_message,
        iterations: result.iterations,
        tool_call_count,
        total_usage,
        total_cost_usd: result.total_cost.unwrap_or(0.0),
    }
}

/// Count tool invocations across the full message history. Assistant
/// messages carry `tool_calls`; summing their lengths yields the total
/// number of executions the agent loop performed.
fn count_tool_calls(messages: &[CoreChatMessage]) -> u32 {
    let total: usize = messages.iter().map(|m| m.tool_calls.len()).sum();
    u32::try_from(total).unwrap_or(u32::MAX)
}
