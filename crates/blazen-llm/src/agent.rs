//! Agentic tool execution loop.
//!
//! Provides [`run_agent`] which implements the standard LLM + tool calling
//! pattern: send messages with tool definitions, execute any tool calls the
//! model makes, feed results back, and repeat until the model stops calling
//! tools or a maximum iteration count is reached.

use std::sync::Arc;
use std::time::Instant;

use crate::error::BlazenError;
use crate::traits::{CompletionModel, Tool};
use crate::types::{
    ChatMessage, CompletionRequest, CompletionResponse, RequestTiming, TokenUsage, ToolCall,
    ToolDefinition,
};

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
    /// Whether to add an implicit "finish" tool the model can call to exit early.
    /// Default: false.
    pub add_finish_tool: bool,
    /// Optional system prompt prepended to messages.
    pub system_prompt: Option<String>,
    /// Sampling temperature.
    pub temperature: Option<f32>,
    /// Maximum tokens per completion call.
    pub max_tokens: Option<u32>,
}

impl AgentConfig {
    #[must_use]
    pub fn new(tools: Vec<Arc<dyn Tool>>) -> Self {
        Self {
            max_iterations: 10,
            tools,
            add_finish_tool: false,
            system_prompt: None,
            temperature: None,
            max_tokens: None,
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
    ) -> Result<serde_json::Value, BlazenError> {
        // The finish tool doesn't really "execute" -- the loop checks for it
        // before execution happens.
        Ok(arguments)
    }
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

        // Execute each tool call and add results.
        execute_tool_calls(
            &response.tool_calls,
            &all_tools,
            &mut messages,
            iteration,
            &on_event,
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

/// Execute all tool calls from a single model response and append the results
/// to the message history.
async fn execute_tool_calls(
    tool_calls: &[ToolCall],
    all_tools: &[Arc<dyn Tool>],
    messages: &mut Vec<ChatMessage>,
    iteration: u32,
    on_event: &(impl Fn(AgentEvent) + Send + Sync),
) -> Result<(), BlazenError> {
    for tc in tool_calls {
        on_event(AgentEvent::ToolCalled {
            iteration,
            tool_call: tc.clone(),
        });

        let tool = find_tool(all_tools, &tc.name)
            .ok_or_else(|| BlazenError::tool_error(format!("unknown tool: {}", tc.name)))?;

        let result = tool.execute(tc.arguments.clone()).await?;

        on_event(AgentEvent::ToolResult {
            iteration,
            tool_name: tc.name.clone(),
            result: result.clone(),
        });

        // Serialize the tool result and add it as a tool message with the
        // matching tool_call_id so providers can correlate results.
        let result_str = if let Some(s) = result.as_str() {
            s.to_owned()
        } else {
            serde_json::to_string(&result).unwrap_or_default()
        };
        messages.push(ChatMessage::tool_result(&tc.id, &tc.name, &result_str));
    }
    Ok(())
}

fn find_tool(tools: &[Arc<dyn Tool>], name: &str) -> Option<Arc<dyn Tool>> {
    tools.iter().find(|t| t.definition().name == name).cloned()
}

fn accumulate_usage(total: &mut Option<TokenUsage>, new: Option<&TokenUsage>) {
    if let Some(new_usage) = new {
        if let Some(existing) = total {
            existing.prompt_tokens += new_usage.prompt_tokens;
            existing.completion_tokens += new_usage.completion_tokens;
            existing.total_tokens += new_usage.total_tokens;
        } else {
            *total = Some(new_usage.clone());
        }
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
