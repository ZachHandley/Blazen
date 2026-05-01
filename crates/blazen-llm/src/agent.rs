//! Agentic tool execution loop.
//!
//! Provides [`run_agent`] which implements the standard LLM + tool calling
//! pattern: send messages with tool definitions, execute any tool calls the
//! model makes, feed results back, and repeat until the model stops calling
//! tools or a maximum iteration count is reached.

use std::sync::Arc;
#[cfg(not(target_arch = "wasm32"))]
use std::time::Instant;
#[cfg(target_arch = "wasm32")]
use web_time::Instant;

use crate::error::BlazenError;
use crate::traits::{CompletionModel, Tool};
use crate::types::{
    ChatMessage, CompletionRequest, CompletionResponse, RequestTiming, TokenUsage, ToolCall,
    ToolDefinition, ToolOutput,
};

/// Default name of the built-in `finish_workflow` exit tool.
pub const FINISH_WORKFLOW_TOOL_NAME: &str = "finish_workflow";

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for the agentic tool execution loop.
#[derive(Clone)]
pub struct AgentConfig {
    /// Maximum number of tool call rounds before forcing a stop.
    /// Default: 10.
    pub max_iterations: u32,
    /// Tools available to the agent.
    pub tools: Vec<Arc<dyn Tool>>,
    /// Whether to add the legacy implicit "finish" tool (single `answer`
    /// string argument) the model can call to exit early. Default: `false`.
    ///
    /// This is the original Wave-0 finish tool. Most callers should prefer
    /// the always-on `finish_workflow` tool (see [`Self::no_finish_tool`])
    /// which carries a typed `result: serde_json::Value` plus optional
    /// `summary: String`.
    pub add_finish_tool: bool,
    /// When `true`, suppress automatic registration of the built-in
    /// `finish_workflow` exit tool. Default: `false` (the tool is always
    /// added unless opted out).
    pub no_finish_tool: bool,
    /// Override the name of the built-in `finish_workflow` tool. Useful
    /// when the host wants to namespace it. Default: `None`, meaning the
    /// canonical name [`FINISH_WORKFLOW_TOOL_NAME`] is used.
    pub finish_tool_name: Option<String>,
    /// Optional system prompt prepended to messages.
    pub system_prompt: Option<String>,
    /// Sampling temperature.
    pub temperature: Option<f32>,
    /// Maximum tokens per completion call.
    pub max_tokens: Option<u32>,
    /// Maximum number of tool calls to execute concurrently within a single
    /// model response. `0` means unlimited (all in parallel). Default: `0`.
    pub tool_concurrency: usize,
}

impl AgentConfig {
    #[must_use]
    pub fn new(tools: Vec<Arc<dyn Tool>>) -> Self {
        Self {
            max_iterations: 10,
            tools,
            add_finish_tool: false,
            no_finish_tool: false,
            finish_tool_name: None,
            system_prompt: None,
            temperature: None,
            max_tokens: None,
            tool_concurrency: 0,
        }
    }

    #[must_use]
    pub fn with_max_iterations(mut self, n: u32) -> Self {
        self.max_iterations = n;
        self
    }

    #[must_use]
    pub fn with_system_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.system_prompt = Some(prompt.into());
        self
    }

    #[must_use]
    pub fn with_finish_tool(mut self) -> Self {
        self.add_finish_tool = true;
        self
    }

    /// Suppress automatic registration of the built-in `finish_workflow`
    /// exit tool. By default the tool is auto-added to every agent run.
    #[must_use]
    pub fn no_finish_tool(mut self) -> Self {
        self.no_finish_tool = true;
        self
    }

    /// Rename the built-in `finish_workflow` exit tool. The schema and
    /// behavior are unchanged; only the name surfaced to the model differs.
    #[must_use]
    pub fn finish_tool_name(mut self, name: impl Into<String>) -> Self {
        self.finish_tool_name = Some(name.into());
        self
    }

    #[must_use]
    pub fn with_temperature(mut self, temp: f32) -> Self {
        self.temperature = Some(temp);
        self
    }

    #[must_use]
    pub fn with_max_tokens(mut self, tokens: u32) -> Self {
        self.max_tokens = Some(tokens);
        self
    }

    #[must_use]
    pub fn with_tool_concurrency(mut self, n: usize) -> Self {
        self.tool_concurrency = n;
        self
    }
}

// ---------------------------------------------------------------------------
// Finish tool (internal)
// ---------------------------------------------------------------------------

/// Built-in tool that signals the agent should stop and return a final answer.
struct FinishTool;

const FINISH_TOOL_NAME: &str = "finish";

#[async_trait::async_trait]
impl Tool for FinishTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: FINISH_TOOL_NAME.to_owned(),
            description: "Call this tool when you have the final answer and want to stop \
                          using tools. Pass the final answer as the 'answer' argument."
                .to_owned(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "answer": {
                        "type": "string",
                        "description": "The final answer to return to the user."
                    }
                },
                "required": ["answer"]
            }),
        }
    }

    async fn execute(
        &self,
        arguments: serde_json::Value,
    ) -> Result<crate::types::ToolOutput<serde_json::Value>, BlazenError> {
        // The finish tool doesn't really "execute" -- the loop checks for it
        // before execution happens.
        Ok(arguments.into())
    }
}

/// Built-in `finish_workflow` exit tool. Carries a typed
/// `result: serde_json::Value` plus an optional human-readable `summary`.
///
/// When the LLM calls this tool, the agent loop returns immediately and the
/// tool's arguments become the final result. Auto-registered on every
/// [`run_agent`] call unless [`AgentConfig::no_finish_tool`] is used.
struct FinishWorkflowTool {
    name: String,
}

impl FinishWorkflowTool {
    fn new(name: impl Into<String>) -> Self {
        Self { name: name.into() }
    }
}

#[async_trait::async_trait]
impl Tool for FinishWorkflowTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: self.name.clone(),
            description: "Call this tool to terminate the agent loop and return a final \
                          structured result. The `result` argument can be any JSON value \
                          (object, array, string, number, etc.); `summary` is an optional \
                          human-readable description of what was accomplished."
                .to_owned(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "result": {
                        "description": "The final structured result of the workflow."
                    },
                    "summary": {
                        "type": "string",
                        "description": "Optional human-readable summary of the result."
                    }
                },
                "required": ["result"]
            }),
        }
    }

    async fn execute(
        &self,
        arguments: serde_json::Value,
    ) -> Result<ToolOutput<serde_json::Value>, BlazenError> {
        // The early-return is performed by the agent loop after dispatch
        // observes `is_exit() == true`; the body simply echoes the args
        // so the tool-result message stays well-formed.
        Ok(ToolOutput::new(arguments))
    }

    fn is_exit(&self) -> bool {
        true
    }
}

/// Construct the built-in `finish_workflow` exit tool with its canonical name.
///
/// Most callers do not need to invoke this directly: [`run_agent`] auto-registers
/// it on every run unless [`AgentConfig::no_finish_tool`] is set. The accessor
/// is exposed primarily for tests, custom agent loops, and binding shims.
#[must_use]
pub fn finish_workflow_tool() -> Arc<dyn Tool> {
    Arc::new(FinishWorkflowTool::new(FINISH_WORKFLOW_TOOL_NAME)) as Arc<dyn Tool>
}

// ---------------------------------------------------------------------------
// Agent result
// ---------------------------------------------------------------------------

/// Result of an agent run.
pub struct AgentResult {
    /// The final completion response.
    pub response: CompletionResponse,
    /// Full message history including all tool calls and results.
    pub messages: Vec<ChatMessage>,
    /// Number of tool call rounds that occurred.
    pub iterations: u32,
    /// Aggregated token usage across all rounds.
    pub total_usage: Option<TokenUsage>,
    /// Aggregated cost across all rounds.
    pub total_cost: Option<f64>,
    /// Total wall-clock time for the entire agent run.
    pub timing: Option<RequestTiming>,
}

// ---------------------------------------------------------------------------
// Agent events (for callbacks)
// ---------------------------------------------------------------------------

/// Events emitted during agent execution.
#[derive(Debug, Clone)]
pub enum AgentEvent {
    /// A tool was called by the model.
    ToolCalled { iteration: u32, tool_call: ToolCall },
    /// A tool execution completed.
    ToolResult {
        iteration: u32,
        tool_name: String,
        result: serde_json::Value,
    },
    /// An iteration completed (model responded).
    IterationComplete {
        iteration: u32,
        had_tool_calls: bool,
    },
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Run the agent loop without event callbacks.
///
/// This is a convenience wrapper around [`run_agent_with_callback`] that
/// discards all [`AgentEvent`]s.
///
/// # Errors
///
/// Returns [`BlazenError`] if any model completion call fails, if a tool call
/// references an unknown tool, or if a tool execution itself fails.
pub async fn run_agent(
    model: &dyn CompletionModel,
    messages: Vec<ChatMessage>,
    config: AgentConfig,
) -> Result<AgentResult, BlazenError> {
    run_agent_with_callback(model, messages, config, |_| {}).await
}

/// Run the agent loop, emitting [`AgentEvent`]s to the supplied callback.
///
/// The loop works as follows:
///
/// 1. Build a [`CompletionRequest`] containing the full message history and
///    all tool definitions.
/// 2. Call the model.
/// 3. If the model responds with no tool calls, return immediately.
/// 4. If the model invoked the built-in "finish" tool (when enabled), extract
///    the answer and return.
/// 5. Otherwise, execute each tool call, append the results to the message
///    history, and go back to step 1.
/// 6. If `max_iterations` is reached, make one final call **without** tools
///    to force the model to produce a text answer.
///
/// # Errors
///
/// Returns [`BlazenError`] if any model completion call fails, if a tool call
/// references an unknown tool, or if a tool execution itself fails.
#[allow(clippy::too_many_lines)]
pub async fn run_agent_with_callback(
    model: &dyn CompletionModel,
    messages: Vec<ChatMessage>,
    config: AgentConfig,
    on_event: impl Fn(AgentEvent) + Send + Sync,
) -> Result<AgentResult, BlazenError> {
    let start = Instant::now();
    let mut messages = messages;
    let mut total_usage: Option<TokenUsage> = None;
    let mut total_cost: Option<f64> = None;

    // Prepend system prompt if provided.
    if let Some(system) = &config.system_prompt {
        messages.insert(0, ChatMessage::system(system));
    }

    // Build tool definitions.
    let mut tool_defs: Vec<ToolDefinition> = config.tools.iter().map(|t| t.definition()).collect();
    let mut all_tools = config.tools.clone();

    if config.add_finish_tool {
        let finish = Arc::new(FinishTool) as Arc<dyn Tool>;
        tool_defs.push(finish.definition());
        all_tools.push(finish);
    }

    // Auto-register the built-in `finish_workflow` exit tool unless opted out.
    if !config.no_finish_tool {
        let name = config
            .finish_tool_name
            .clone()
            .unwrap_or_else(|| FINISH_WORKFLOW_TOOL_NAME.to_owned());
        // Don't double-register if the caller already supplied a tool with the
        // same name (e.g. their own custom exit tool).
        if !all_tools.iter().any(|t| t.definition().name == name) {
            let finish = Arc::new(FinishWorkflowTool::new(name)) as Arc<dyn Tool>;
            tool_defs.push(finish.definition());
            all_tools.push(finish);
        }
    }

    for iteration in 0..config.max_iterations {
        let request = build_request(&messages, &tool_defs, &config);

        // Call the model.
        let response = model.complete(request).await?;

        // Accumulate usage.
        accumulate_usage(&mut total_usage, response.usage.as_ref());
        accumulate_cost(&mut total_cost, response.cost);

        // No tool calls -- model is done.
        if response.tool_calls.is_empty() {
            on_event(AgentEvent::IterationComplete {
                iteration,
                had_tool_calls: false,
            });

            return Ok(AgentResult {
                response,
                messages,
                iterations: iteration,
                total_usage,
                total_cost,
                timing: Some(elapsed_timing(&start)),
            });
        }

        // Check for finish tool.
        if config.add_finish_tool
            && let Some(result) = check_finish_tool(
                &response,
                &messages,
                iteration,
                total_usage.as_ref(),
                total_cost.as_ref(),
                &start,
            )
        {
            return Ok(result);
        }

        // Append the assistant's response including any tool calls so that
        // the full conversation history is preserved for the provider.
        messages.push(ChatMessage::assistant_with_tool_calls(
            response.content.clone(),
            response.tool_calls.clone(),
        ));

        // Detect an exit-tool call BEFORE dispatch so we can short-circuit
        // immediately with the tool's arguments as the final result. This
        // covers both the built-in `finish_workflow` tool and any user tool
        // that returns `is_exit() == true`.
        if let Some(exit_call) = find_exit_call(&response.tool_calls, &all_tools) {
            on_event(AgentEvent::ToolCalled {
                iteration,
                tool_call: exit_call.clone(),
            });
            on_event(AgentEvent::ToolResult {
                iteration,
                tool_name: exit_call.name.clone(),
                result: exit_call.arguments.clone(),
            });
            on_event(AgentEvent::IterationComplete {
                iteration,
                had_tool_calls: true,
            });
            return Ok(build_exit_result(
                exit_call,
                &response,
                &messages,
                iteration,
                total_usage,
                total_cost,
                &start,
            ));
        }

        // Execute each tool call and add results.
        execute_tool_calls(
            &response.tool_calls,
            &all_tools,
            &mut messages,
            iteration,
            &on_event,
            config.tool_concurrency,
        )
        .await?;

        on_event(AgentEvent::IterationComplete {
            iteration,
            had_tool_calls: true,
        });
    }

    // Max iterations reached -- force a final call **without** tools so the
    // model must produce a plain text response.
    let request = build_request_no_tools(&messages, &config);
    let response = model.complete(request).await?;
    accumulate_usage(&mut total_usage, response.usage.as_ref());
    accumulate_cost(&mut total_cost, response.cost);

    Ok(AgentResult {
        response,
        messages,
        iterations: config.max_iterations,
        total_usage,
        total_cost,
        timing: Some(elapsed_timing(&start)),
    })
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Build a [`CompletionRequest`] with tools and optional sampling parameters.
fn build_request(
    messages: &[ChatMessage],
    tool_defs: &[ToolDefinition],
    config: &AgentConfig,
) -> CompletionRequest {
    let mut request = CompletionRequest::new(messages.to_vec()).with_tools(tool_defs.to_vec());
    if let Some(temp) = config.temperature {
        request = request.with_temperature(temp);
    }
    if let Some(max) = config.max_tokens {
        request = request.with_max_tokens(max);
    }
    request
}

/// Build a [`CompletionRequest`] **without** tools (for the final forced call).
fn build_request_no_tools(messages: &[ChatMessage], config: &AgentConfig) -> CompletionRequest {
    let mut request = CompletionRequest::new(messages.to_vec());
    if let Some(temp) = config.temperature {
        request = request.with_temperature(temp);
    }
    if let Some(max) = config.max_tokens {
        request = request.with_max_tokens(max);
    }
    request
}

/// Check whether the model called the built-in "finish" tool. If so, build
/// and return the final [`AgentResult`].
fn check_finish_tool(
    response: &CompletionResponse,
    messages: &[ChatMessage],
    iteration: u32,
    total_usage: Option<&TokenUsage>,
    total_cost: Option<&f64>,
    start: &Instant,
) -> Option<AgentResult> {
    let finish_call = response
        .tool_calls
        .iter()
        .find(|tc| tc.name == FINISH_TOOL_NAME)?;

    let answer = finish_call
        .arguments
        .get("answer")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_owned();

    let final_response = CompletionResponse {
        content: Some(answer),
        tool_calls: vec![],
        reasoning: None,
        citations: vec![],
        artifacts: vec![],
        usage: response.usage.clone(),
        model: response.model.clone(),
        finish_reason: Some("finish_tool".to_owned()),
        cost: response.cost,
        timing: response.timing.clone(),
        images: response.images.clone(),
        audio: response.audio.clone(),
        videos: response.videos.clone(),
        metadata: response.metadata.clone(),
    };

    Some(AgentResult {
        response: final_response,
        messages: messages.to_vec(),
        iterations: iteration,
        total_usage: total_usage.cloned(),
        total_cost: total_cost.copied(),
        timing: Some(elapsed_timing(start)),
    })
}

/// Execute all tool calls from a single model response concurrently and append
/// the results to the message history in the original order.
///
/// When `tool_concurrency` is `0`, all tool calls run in parallel with no limit.
/// When it is a positive number, at most that many tool calls execute at once
/// (bounded via a [`tokio::sync::Semaphore`]).
async fn execute_tool_calls(
    tool_calls: &[ToolCall],
    all_tools: &[Arc<dyn Tool>],
    messages: &mut Vec<ChatMessage>,
    iteration: u32,
    on_event: &(impl Fn(AgentEvent) + Send + Sync),
    tool_concurrency: usize,
) -> Result<(), BlazenError> {
    // Validate all tool names up front so we fail fast before executing any.
    let resolved_tools: Vec<Arc<dyn Tool>> = tool_calls
        .iter()
        .map(|tc| {
            find_tool(all_tools, &tc.name)
                .ok_or_else(|| BlazenError::tool_error(format!("unknown tool: {}", tc.name)))
        })
        .collect::<Result<Vec<_>, _>>()?;

    // Fire ToolCalled events for every tool before we start execution.
    for tc in tool_calls {
        on_event(AgentEvent::ToolCalled {
            iteration,
            tool_call: tc.clone(),
        });
    }

    // Build a semaphore for bounded concurrency (None = unlimited).
    let semaphore = if tool_concurrency > 0 {
        Some(Arc::new(tokio::sync::Semaphore::new(tool_concurrency)))
    } else {
        None
    };

    // Execute all tool calls concurrently, preserving order via `join_all`.
    let futures: Vec<_> = tool_calls
        .iter()
        .zip(resolved_tools.iter())
        .map(|(tc, tool)| {
            let tool = Arc::clone(tool);
            let args = tc.arguments.clone();
            let sem = semaphore.clone();
            async move {
                // Acquire a permit when concurrency is bounded.
                let _permit = match sem {
                    Some(ref s) => Some(s.acquire().await.expect("semaphore closed")),
                    None => None,
                };
                tool.execute(args).await
            }
        })
        .collect();

    let results = futures_util::future::join_all(futures).await;

    // Process results in the original order.
    for (tc, result) in tool_calls.iter().zip(results) {
        let output = result?;

        on_event(AgentEvent::ToolResult {
            iteration,
            tool_name: tc.name.clone(),
            result: output.data.clone(),
        });

        messages.push(ChatMessage::tool_result(&tc.id, &tc.name, output));
    }

    Ok(())
}

fn find_tool(tools: &[Arc<dyn Tool>], name: &str) -> Option<Arc<dyn Tool>> {
    tools.iter().find(|t| t.definition().name == name).cloned()
}

/// Scan `tool_calls` and return the first call that targets an exit-marked
/// tool (`Tool::is_exit() == true`). Returns `None` when no exit tool was
/// invoked or the tool name is unknown.
fn find_exit_call<'a>(
    tool_calls: &'a [ToolCall],
    all_tools: &[Arc<dyn Tool>],
) -> Option<&'a ToolCall> {
    tool_calls
        .iter()
        .find(|tc| find_tool(all_tools, &tc.name).is_some_and(|t| t.is_exit()))
}

/// Build the [`AgentResult`] returned when an exit tool terminates the loop.
///
/// The exit tool's `arguments` become the response `metadata` (so callers can
/// recover the structured `result` / `summary` fields verbatim) and, when a
/// `summary` string is present in the args, the `content` of the synthesized
/// final response.
fn build_exit_result(
    exit_call: &ToolCall,
    response: &CompletionResponse,
    messages: &[ChatMessage],
    iteration: u32,
    total_usage: Option<TokenUsage>,
    total_cost: Option<f64>,
    start: &Instant,
) -> AgentResult {
    let summary = exit_call
        .arguments
        .get("summary")
        .and_then(|v| v.as_str())
        .map(str::to_owned);

    let final_response = CompletionResponse {
        content: summary,
        tool_calls: vec![],
        reasoning: None,
        citations: vec![],
        artifacts: vec![],
        usage: response.usage.clone(),
        model: response.model.clone(),
        finish_reason: Some("exit_tool".to_owned()),
        cost: response.cost,
        timing: response.timing.clone(),
        images: response.images.clone(),
        audio: response.audio.clone(),
        videos: response.videos.clone(),
        metadata: exit_call.arguments.clone(),
    };

    AgentResult {
        response: final_response,
        messages: messages.to_vec(),
        iterations: iteration,
        total_usage,
        total_cost,
        timing: Some(elapsed_timing(start)),
    }
}

fn accumulate_usage(total: &mut Option<TokenUsage>, new: Option<&TokenUsage>) {
    let Some(new_usage) = new else {
        return;
    };
    match total {
        Some(existing) => existing.add(new_usage),
        None => *total = Some(new_usage.clone()),
    }
}

fn accumulate_cost(total: &mut Option<f64>, new: Option<f64>) {
    if let Some(c) = new {
        *total = Some(total.unwrap_or(0.0) + c);
    }
}

/// Build a [`RequestTiming`] from the given start instant, using saturating
/// conversion to avoid truncation warnings.
fn elapsed_timing(start: &Instant) -> RequestTiming {
    let ms = start.elapsed().as_millis();
    // Saturate to u64::MAX if the duration somehow exceeds ~584 million years.
    #[allow(clippy::cast_possible_truncation)]
    let total_ms = if ms > u128::from(u64::MAX) {
        u64::MAX
    } else {
        ms as u64
    };
    RequestTiming {
        queue_ms: None,
        execution_ms: None,
        total_ms: Some(total_ms),
    }
}
