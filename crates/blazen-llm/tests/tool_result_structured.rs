//! Integration tests for the agent loop's handling of structured tool results.
//!
//! These tests drive [`blazen_llm::run_agent`] with a tiny in-memory
//! [`MockCompletionModel`] that returns pre-canned responses on each turn. The
//! goal is to verify that the agent loop preserves the full `ToolOutput`
//! structure on `Role::Tool` messages (data + optional `llm_override`) instead
//! of stringifying everything into `content`.

use std::pin::Pin;
use std::sync::Arc;
use std::sync::Mutex;

use async_trait::async_trait;
use futures_util::Stream;
use serde_json::json;

use blazen_llm::error::BlazenError;
use blazen_llm::traits::{CompletionModel, Tool};
use blazen_llm::types::{
    ChatMessage, CompletionRequest, CompletionResponse, LlmPayload, MessageContent, Role,
    StreamChunk, ToolCall, ToolDefinition, ToolOutput,
};
use blazen_llm::{AgentConfig, run_agent};

// ---------------------------------------------------------------------------
// Mock completion model
// ---------------------------------------------------------------------------

/// A `CompletionModel` whose `complete` calls return successive pre-canned
/// responses from an internal queue. `stream` is not exercised by `run_agent`,
/// so it returns an error if ever invoked.
struct MockCompletionModel {
    responses: Mutex<Vec<CompletionResponse>>,
    cursor: Mutex<usize>,
}

impl MockCompletionModel {
    fn new(responses: Vec<CompletionResponse>) -> Self {
        Self {
            responses: Mutex::new(responses),
            cursor: Mutex::new(0),
        }
    }
}

#[async_trait]
impl CompletionModel for MockCompletionModel {
    #[allow(clippy::unnecessary_literal_bound)]
    fn model_id(&self) -> &str {
        "mock-model"
    }

    async fn complete(
        &self,
        _request: CompletionRequest,
    ) -> Result<CompletionResponse, BlazenError> {
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

/// Build a [`CompletionResponse`] with only the fields the agent loop reads.
fn make_response(
    content: Option<&str>,
    tool_calls: Vec<ToolCall>,
    finish_reason: Option<&str>,
) -> CompletionResponse {
    CompletionResponse {
        content: content.map(str::to_owned),
        tool_calls,
        reasoning: None,
        citations: vec![],
        artifacts: vec![],
        usage: None,
        model: "mock-model".to_owned(),
        finish_reason: finish_reason.map(str::to_owned),
        cost: None,
        timing: None,
        images: vec![],
        audio: vec![],
        videos: vec![],
        metadata: serde_json::Value::Null,
    }
}

// ---------------------------------------------------------------------------
// Mock tool
// ---------------------------------------------------------------------------

/// A tool that ignores its arguments and returns a pre-set [`ToolOutput`].
struct CannedTool {
    output: Mutex<Option<ToolOutput<serde_json::Value>>>,
}

impl CannedTool {
    fn new(output: ToolOutput<serde_json::Value>) -> Self {
        Self {
            output: Mutex::new(Some(output)),
        }
    }
}

#[async_trait]
impl Tool for CannedTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "search".to_owned(),
            description: "search for stuff".to_owned(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "q": { "type": "string" }
                },
                "required": ["q"]
            }),
        }
    }

    async fn execute(
        &self,
        _arguments: serde_json::Value,
    ) -> Result<ToolOutput<serde_json::Value>, BlazenError> {
        let mut slot = self.output.lock().unwrap();
        slot.take()
            .ok_or_else(|| BlazenError::tool_error("canned tool already consumed"))
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Build the standard two-turn pre-canned sequence: turn 1 invokes `search`,
/// turn 2 returns a final `"done"` message with `finish_reason: stop`.
fn standard_responses() -> Vec<CompletionResponse> {
    vec![
        make_response(
            None,
            vec![ToolCall {
                id: "call_1".to_owned(),
                name: "search".to_owned(),
                arguments: json!({ "q": "cats" }),
            }],
            Some("tool_calls"),
        ),
        make_response(Some("done"), vec![], Some("stop")),
    ]
}

/// Locate the single `Role::Tool` message in the agent's message history.
fn find_tool_message(messages: &[ChatMessage]) -> &ChatMessage {
    messages
        .iter()
        .find(|m| m.role == Role::Tool)
        .expect("expected a Role::Tool message in the agent's history")
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[tokio::test]
async fn agent_loop_preserves_structured_data() {
    let model = MockCompletionModel::new(standard_responses());
    let tool = Arc::new(CannedTool::new(ToolOutput::new(
        json!({ "items": [1, 2, 3] }),
    ))) as Arc<dyn Tool>;

    let result = run_agent(
        &model,
        vec![ChatMessage::user("find cats")],
        AgentConfig::new(vec![tool]),
    )
    .await
    .expect("agent run should succeed");

    let tool_msg = find_tool_message(&result.messages);

    let tool_result = tool_msg
        .tool_result
        .as_ref()
        .expect("structured tool_result should be populated for object data");
    assert_eq!(tool_result.data, json!({ "items": [1, 2, 3] }));
    assert!(tool_result.llm_override.is_none());

    // The structured payload bypasses the text channel — content stays empty.
    assert!(
        tool_msg
            .content
            .text_content()
            .unwrap_or_default()
            .is_empty()
    );
}

#[tokio::test]
async fn agent_loop_string_result_lives_in_content() {
    let model = MockCompletionModel::new(standard_responses());
    let tool = Arc::new(CannedTool::new(ToolOutput::new(json!("hello")))) as Arc<dyn Tool>;

    let result = run_agent(
        &model,
        vec![ChatMessage::user("find cats")],
        AgentConfig::new(vec![tool]),
    )
    .await
    .expect("agent run should succeed");

    let tool_msg = find_tool_message(&result.messages);

    assert!(
        tool_msg.tool_result.is_none(),
        "plain-string data should live in content, not in tool_result"
    );
    assert_eq!(tool_msg.content, MessageContent::Text("hello".into()));
}

#[tokio::test]
async fn agent_loop_preserves_llm_override() {
    let model = MockCompletionModel::new(standard_responses());
    let output = ToolOutput::with_override(
        json!({ "items": [1, 2, 3], "_debug": "..." }),
        LlmPayload::Text {
            text: "3 items".into(),
        },
    );
    let tool = Arc::new(CannedTool::new(output)) as Arc<dyn Tool>;

    let result = run_agent(
        &model,
        vec![ChatMessage::user("find cats")],
        AgentConfig::new(vec![tool]),
    )
    .await
    .expect("agent run should succeed");

    let tool_msg = find_tool_message(&result.messages);

    let tool_result = tool_msg
        .tool_result
        .as_ref()
        .expect("tool_result with an llm_override must be preserved");
    assert_eq!(
        tool_result.data,
        json!({ "items": [1, 2, 3], "_debug": "..." })
    );
    assert_eq!(
        tool_result.llm_override,
        Some(LlmPayload::Text {
            text: "3 items".into()
        })
    );
}
