//! Integration tests for the agent loop's tool-error recovery semantics.
//!
//! A tool-execution error — most commonly the LLM sending malformed arguments,
//! or invoking a tool name that doesn't exist — must NOT abort the run. The
//! agent loop turns the error into a `tool_result` message fed back to the
//! model (so it can read the failure and retry) and emits an
//! [`AgentEvent::ToolError`], mirroring the Anthropic / `OpenAI` tool-use loops.
//!
//! These tests drive [`run_agent_with_callback`] with an in-memory `MockModel`
//! that emits a tool call on the first turn, then a plain answer on the second,
//! and assert the run recovers instead of failing.

use std::pin::Pin;
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use futures_util::Stream;
use serde_json::json;

use blazen_llm::error::BlazenError;
use blazen_llm::traits::{Model, Tool};
use blazen_llm::types::{
    ChatMessage, ModelRequest, ModelResponse, Role, StreamChunk, ToolCall, ToolDefinition,
    ToolOutput,
};
use blazen_llm::{AgentConfig, AgentEvent, run_agent_with_callback};

// ---------------------------------------------------------------------------
// Mock completion model (queue of pre-canned responses)
// ---------------------------------------------------------------------------

struct MockModel {
    responses: Mutex<Vec<ModelResponse>>,
    cursor: Mutex<usize>,
    calls: Mutex<u32>,
}

impl MockModel {
    fn new(responses: Vec<ModelResponse>) -> Self {
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
impl Model for MockModel {
    fn model_id(&self) -> &'static str {
        "mock-model"
    }

    async fn complete(&self, _request: ModelRequest) -> Result<ModelResponse, BlazenError> {
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
        _request: ModelRequest,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<StreamChunk, BlazenError>> + Send>>, BlazenError>
    {
        Err(BlazenError::tool_error("mock: stream() not used"))
    }
}

fn make_response(content: Option<&str>, tool_calls: Vec<ToolCall>) -> ModelResponse {
    ModelResponse {
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
// A tool whose arg parsing is strict — mirrors `TypedTool`'s deserialization
// error path without pulling a schemars/serde-derive dev-dep into the test.
// ---------------------------------------------------------------------------

struct StrictAdder;

#[async_trait]
impl Tool for StrictAdder {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "strict_adder".to_owned(),
            description: "Add two integers a and b.".to_owned(),
            parameters: json!({
                "type": "object",
                "properties": { "a": { "type": "integer" }, "b": { "type": "integer" } },
                "required": ["a", "b"]
            }),
        }
    }

    async fn execute(
        &self,
        arguments: serde_json::Value,
    ) -> Result<ToolOutput<serde_json::Value>, BlazenError> {
        let a = arguments
            .get("a")
            .and_then(serde_json::Value::as_i64)
            .ok_or_else(|| {
                BlazenError::tool_error("argument deserialization failed: expected integer 'a'")
            })?;
        let b = arguments
            .get("b")
            .and_then(serde_json::Value::as_i64)
            .ok_or_else(|| {
                BlazenError::tool_error("argument deserialization failed: expected integer 'b'")
            })?;
        Ok(ToolOutput::new(json!({ "sum": a + b })))
    }
}

/// Collect every emitted [`AgentEvent`] for post-run assertions.
fn event_collector() -> (
    Arc<Mutex<Vec<AgentEvent>>>,
    impl Fn(AgentEvent) + Send + Sync,
) {
    let sink = Arc::new(Mutex::new(Vec::new()));
    let sink_clone = Arc::clone(&sink);
    let cb = move |ev: AgentEvent| sink_clone.lock().unwrap().push(ev);
    (sink, cb)
}

/// Find the first `ToolError` event and return `(tool_name, error)`.
fn first_tool_error(events: &[AgentEvent]) -> Option<(String, String)> {
    events.iter().find_map(|e| match e {
        AgentEvent::ToolError {
            tool_name, error, ..
        } => Some((tool_name.clone(), error.clone())),
        _ => None,
    })
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[tokio::test]
async fn arg_error_is_fed_back_and_run_recovers() {
    // Turn 1: model calls strict_adder with a non-integer `a` -> deserialization
    // error. Turn 2: model returns a plain answer -> loop terminates.
    let model = MockModel::new(vec![
        make_response(
            None,
            vec![tool_call(
                "c1",
                "strict_adder",
                json!({ "a": "oops", "b": 2 }),
            )],
        ),
        make_response(Some("recovered"), vec![]),
    ]);

    let (events, on_event) = event_collector();
    let config = AgentConfig::new(vec![Arc::new(StrictAdder) as Arc<dyn Tool>]).no_finish_tool();

    let result = run_agent_with_callback(
        &model,
        vec![ChatMessage::user("add them")],
        config,
        on_event,
    )
    .await
    .expect("run must NOT abort on a recoverable tool error");

    // The run completed via the second turn.
    assert_eq!(
        model.call_count(),
        2,
        "model should get a second turn to retry"
    );
    assert_eq!(result.response.content.as_deref(), Some("recovered"));

    // A ToolError event fired, carrying the deserialization message.
    let evs = events.lock().unwrap();
    let (name, err) = first_tool_error(&evs).expect("a ToolError event must fire");
    assert_eq!(name, "strict_adder");
    assert!(
        err.contains("argument deserialization failed"),
        "error should carry the deser failure, got: {err}"
    );

    // The error was fed back to the model as a tool_result message between turns.
    let tool_msg = result
        .messages
        .iter()
        .find(|m| m.role == Role::Tool && m.tool_call_id.as_deref() == Some("c1"))
        .expect("a tool_result message for c1 must be appended");
    let rendered = format!("{tool_msg:?}");
    assert!(
        rendered.contains("error"),
        "fed-back tool_result should carry an error payload, got: {rendered}"
    );
}

#[tokio::test]
async fn unknown_tool_is_fed_back_with_available_list() {
    // Turn 1: model hallucinates a tool name. Turn 2: plain answer.
    let model = MockModel::new(vec![
        make_response(None, vec![tool_call("c1", "ghost_tool", json!({}))]),
        make_response(Some("ok"), vec![]),
    ]);

    let (events, on_event) = event_collector();
    // Register one real tool so the "available tools" list is non-empty.
    let config = AgentConfig::new(vec![Arc::new(StrictAdder) as Arc<dyn Tool>]).no_finish_tool();

    let result = run_agent_with_callback(&model, vec![ChatMessage::user("go")], config, on_event)
        .await
        .expect("unknown-tool call must not abort the run");

    assert_eq!(model.call_count(), 2);
    assert_eq!(result.response.content.as_deref(), Some("ok"));

    let evs = events.lock().unwrap();
    let (name, err) = first_tool_error(&evs).expect("a ToolError event must fire");
    assert_eq!(name, "ghost_tool");
    assert!(err.contains("unknown tool 'ghost_tool'"), "got: {err}");
    assert!(
        err.contains("strict_adder"),
        "error should list available tools, got: {err}"
    );
}

#[tokio::test]
async fn success_path_still_emits_tool_result() {
    // Regression guard: a tool that succeeds still produces a ToolResult event
    // (not ToolError) and the run completes normally.
    let model = MockModel::new(vec![
        make_response(
            None,
            vec![tool_call("c1", "strict_adder", json!({ "a": 2, "b": 3 }))],
        ),
        make_response(Some("the sum is 5"), vec![]),
    ]);

    let (events, on_event) = event_collector();
    let config = AgentConfig::new(vec![Arc::new(StrictAdder) as Arc<dyn Tool>]).no_finish_tool();

    let result = run_agent_with_callback(
        &model,
        vec![ChatMessage::user("add 2 and 3")],
        config,
        on_event,
    )
    .await
    .expect("success path must complete");

    assert_eq!(result.response.content.as_deref(), Some("the sum is 5"));

    let evs = events.lock().unwrap();
    assert!(
        evs.iter().any(
            |e| matches!(e, AgentEvent::ToolResult { tool_name, .. } if tool_name == "strict_adder")
        ),
        "a ToolResult event should fire on success"
    );
    assert!(
        first_tool_error(&evs).is_none(),
        "no ToolError on the success path"
    );
}

#[tokio::test]
async fn repeated_tool_errors_are_bounded_by_max_iterations() {
    // A model that ALWAYS sends a bad tool call must still terminate via the
    // forced no-tools final call at max_iterations — never an aborted run.
    let max = 3u32;
    let mut responses: Vec<ModelResponse> = (0..max)
        .map(|i| {
            make_response(
                None,
                vec![tool_call(
                    &format!("c{i}"),
                    "strict_adder",
                    json!({ "a": "bad", "b": "bad" }),
                )],
            )
        })
        .collect();
    // The forced final call (build_request_no_tools) consumes one more response.
    responses.push(make_response(Some("giving up cleanly"), vec![]));

    let (events, on_event) = event_collector();
    let config = AgentConfig::new(vec![Arc::new(StrictAdder) as Arc<dyn Tool>])
        .no_finish_tool()
        .with_max_iterations(max);

    let result = run_agent_with_callback(
        &model_for(responses),
        vec![ChatMessage::user("go")],
        config,
        on_event,
    )
    .await
    .expect("repeated tool errors must not abort the run");

    assert_eq!(
        result.iterations, max,
        "loop should run to the iteration cap"
    );
    let evs = events.lock().unwrap();
    let tool_errors = evs
        .iter()
        .filter(|e| matches!(e, AgentEvent::ToolError { .. }))
        .count();
    assert_eq!(tool_errors, max as usize, "one ToolError per bad iteration");
}

fn model_for(responses: Vec<ModelResponse>) -> MockModel {
    MockModel::new(responses)
}
