//! Integration tests for the agent loop's exit-tool semantics (Wave 6).
//!
//! Two paths terminate the loop early with a structured result:
//!
//! 1. The auto-registered built-in `finish_workflow` tool.
//! 2. Any user-supplied tool whose [`Tool::is_exit`] returns `true`.
//!
//! These tests drive [`run_agent`] with a tiny in-memory `MockCompletionModel`
//! that emits a single tool-call on the first turn, then verify that the loop
//! returned immediately and that the tool's arguments survive in the result.

use std::pin::Pin;
use std::sync::Arc;
use std::sync::Mutex;

use async_trait::async_trait;
use futures_util::Stream;
use serde_json::json;

use blazen_llm::error::BlazenError;
use blazen_llm::traits::{CompletionModel, Tool};
use blazen_llm::types::{
    ChatMessage, CompletionRequest, CompletionResponse, StreamChunk, ToolCall, ToolDefinition,
    ToolOutput,
};
use blazen_llm::{AgentConfig, FINISH_WORKFLOW_TOOL_NAME, finish_workflow_tool, run_agent};

// ---------------------------------------------------------------------------
// Mock completion model
// ---------------------------------------------------------------------------

/// A `CompletionModel` whose `complete` calls return successive pre-canned
/// responses from an internal queue.
struct MockCompletionModel {
    responses: Mutex<Vec<CompletionResponse>>,
    cursor: Mutex<usize>,
    calls: Mutex<u32>,
}

impl MockCompletionModel {
    fn new(responses: Vec<CompletionResponse>) -> Self {
        Self {
            responses: Mutex::new(responses),
            cursor: Mutex::new(0),
            calls: Mutex::new(0),
        }
    }

    fn call_count(&self) -> u32 {
        *self.calls.lock().unwrap()
    }
}

#[async_trait]
impl CompletionModel for MockCompletionModel {
    fn model_id(&self) -> &'static str {
        "mock-model"
    }

    async fn complete(
        &self,
        _request: CompletionRequest,
    ) -> Result<CompletionResponse, BlazenError> {
        *self.calls.lock().unwrap() += 1;
        let mut idx = self.cursor.lock().unwrap();
        let responses = self.responses.lock().unwrap();
        let response = responses
            .get(*idx)
            .cloned()
            .ok_or_else(|| BlazenError::tool_error("mock: no more pre-canned responses"))?;
        *idx += 1;
        Ok(response)
    }

    async fn stream(
        &self,
        _request: CompletionRequest,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<StreamChunk, BlazenError>> + Send>>, BlazenError>
    {
        Err(BlazenError::tool_error(
            "mock: stream() not used by run_agent",
        ))
    }
}

/// Build a minimal [`CompletionResponse`].
fn make_response(content: Option<&str>, tool_calls: Vec<ToolCall>) -> CompletionResponse {
    CompletionResponse {
        content: content.map(str::to_owned),
        tool_calls,
        reasoning: None,
        citations: vec![],
        artifacts: vec![],
        usage: None,
        model: "mock-model".to_owned(),
        finish_reason: None,
        cost: None,
        timing: None,
        images: vec![],
        audio: vec![],
        videos: vec![],
        metadata: serde_json::Value::Null,
    }
}

fn tool_call(id: &str, name: &str, args: serde_json::Value) -> ToolCall {
    ToolCall {
        id: id.to_owned(),
        name: name.to_owned(),
        arguments: args,
    }
}

// ---------------------------------------------------------------------------
// User-defined tools
// ---------------------------------------------------------------------------

/// A user tool that is marked as an exit tool. Calling it should terminate
/// the loop with the args as the final result.
struct UserExitTool;

#[async_trait]
impl Tool for UserExitTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "submit_answer".to_owned(),
            description: "Submit the final answer and exit.".to_owned(),
            parameters: json!({
                "type": "object",
                "properties": { "answer": { "type": "string" } },
                "required": ["answer"]
            }),
        }
    }

    async fn execute(
        &self,
        arguments: serde_json::Value,
    ) -> Result<ToolOutput<serde_json::Value>, BlazenError> {
        Ok(ToolOutput::new(arguments))
    }

    fn is_exit(&self) -> bool {
        true
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[tokio::test]
async fn builtin_finish_workflow_terminates_agent() {
    // The built-in `finish_workflow` tool must be auto-registered, and calling
    // it must terminate the loop immediately.
    let exit_args = json!({
        "result": { "score": 42, "label": "ok" },
        "summary": "found the answer"
    });
    let model = MockCompletionModel::new(vec![make_response(
        None,
        vec![tool_call(
            "call_1",
            FINISH_WORKFLOW_TOOL_NAME,
            exit_args.clone(),
        )],
    )]);

    // No user tools, default config -> finish_workflow is auto-added.
    let config = AgentConfig::new(vec![]);
    let result = run_agent(&model, vec![ChatMessage::user("hi")], config)
        .await
        .expect("agent run failed");

    // Only one model call should have happened.
    assert_eq!(model.call_count(), 1, "agent should not loop after exit");
    assert_eq!(result.iterations, 0, "exit on iteration 0");

    // The metadata field carries the exit tool's arguments verbatim so callers
    // can recover the structured `result` and `summary`.
    assert_eq!(result.response.metadata, exit_args);
    assert_eq!(
        result.response.finish_reason.as_deref(),
        Some("exit_tool"),
        "synthesized finish reason should be exit_tool"
    );
    assert_eq!(result.response.content.as_deref(), Some("found the answer"));
    assert!(
        result.response.tool_calls.is_empty(),
        "synthesized response carries no further tool calls"
    );
}

#[tokio::test]
async fn user_marked_is_exit_tool_terminates_agent() {
    // A user-defined tool with `is_exit() == true` must also short-circuit
    // the loop and return its arguments as the result.
    let exit_args = json!({ "answer": "the rain in spain" });
    let model = MockCompletionModel::new(vec![make_response(
        None,
        vec![tool_call("call_1", "submit_answer", exit_args.clone())],
    )]);

    let user_tool: Arc<dyn Tool> = Arc::new(UserExitTool);
    let config = AgentConfig::new(vec![user_tool]);

    let result = run_agent(&model, vec![ChatMessage::user("hi")], config)
        .await
        .expect("agent run failed");

    assert_eq!(model.call_count(), 1);
    assert_eq!(result.iterations, 0);
    assert_eq!(result.response.metadata, exit_args);
    assert_eq!(result.response.finish_reason.as_deref(), Some("exit_tool"));
}

#[tokio::test]
async fn agent_with_no_finish_tool_uses_normal_completion() {
    // With `no_finish_tool()`, the built-in must be absent. A response that
    // emits no tool calls (just plain text) should be returned via the normal
    // completion path -- no exit-tool synthesis.
    let model = MockCompletionModel::new(vec![make_response(Some("plain answer"), vec![])]);

    let config = AgentConfig::new(vec![]).no_finish_tool();
    let result = run_agent(&model, vec![ChatMessage::user("hi")], config)
        .await
        .expect("agent run failed");

    assert_eq!(model.call_count(), 1);
    assert_eq!(result.iterations, 0);
    assert_eq!(result.response.content.as_deref(), Some("plain answer"));
    // Normal completion path -> no exit_tool finish_reason synthesized,
    // and metadata stays at the model's value (Null in this mock).
    assert_ne!(result.response.finish_reason.as_deref(), Some("exit_tool"));
    assert_eq!(result.response.metadata, serde_json::Value::Null);
}

#[tokio::test]
async fn custom_finish_tool_name_is_respected() {
    // Renaming the built-in via `finish_tool_name` should expose it under the
    // new name. A model call that targets that name must still trigger the
    // exit-tool early-return.
    let exit_args = json!({ "result": "done", "summary": "ok" });
    let model = MockCompletionModel::new(vec![make_response(
        None,
        vec![tool_call("call_1", "all_done", exit_args.clone())],
    )]);

    let config = AgentConfig::new(vec![]).finish_tool_name("all_done");
    let result = run_agent(&model, vec![ChatMessage::user("hi")], config)
        .await
        .expect("agent run failed");

    assert_eq!(model.call_count(), 1);
    assert_eq!(result.iterations, 0);
    assert_eq!(result.response.metadata, exit_args);
    assert_eq!(result.response.finish_reason.as_deref(), Some("exit_tool"));
}

#[tokio::test]
async fn finish_workflow_tool_helper_constructs_named_tool() {
    // Sanity check: the public `finish_workflow_tool()` helper constructs a
    // tool exposing the canonical name and `is_exit() == true`.
    let tool = finish_workflow_tool();
    let def = tool.definition();
    assert_eq!(def.name, FINISH_WORKFLOW_TOOL_NAME);
    assert!(tool.is_exit());
    // Schema should require `result`.
    let required = def
        .parameters
        .get("required")
        .and_then(|v| v.as_array())
        .expect("required array");
    assert!(required.iter().any(|v| v == "result"));
}
