//! Typed stream chunk for the Node.js bindings.

use napi_derive::napi;

use super::artifact::JsArtifact;
use super::citation::JsCitation;
use crate::generated::JsToolCall;

// ---------------------------------------------------------------------------
// StreamChunk
// ---------------------------------------------------------------------------

/// A single chunk from a streaming completion response.
///
/// Passed to the `onChunk` callback during streaming operations.
///
/// ```javascript
/// await model.stream(
///   [ChatMessage.user("Tell me a story")],
///   (chunk) => {
///     if (chunk.delta) process.stdout.write(chunk.delta);
///     if (chunk.finishReason) console.log("done:", chunk.finishReason);
///   }
/// );
/// ```
#[napi(object)]
pub struct JsStreamChunk {
    /// Incremental text content, if any.
    pub delta: Option<String>,
    /// The reason the model stopped generating (`"stop"`, `"tool_use"`, etc.).
    /// Present only in the final chunk.
    #[napi(js_name = "finishReason")]
    pub finish_reason: Option<String>,
    /// Tool invocations completed in this chunk.
    #[napi(js_name = "toolCalls")]
    pub tool_calls: Vec<JsToolCall>,
    /// Reasoning text delta from models that stream a chain-of-thought trace
    /// (Anthropic extended thinking, `DeepSeek` R1, `OpenAI` o-series).
    #[napi(js_name = "reasoningDelta")]
    pub reasoning_delta: Option<String>,
    /// Citations completed in this chunk.
    pub citations: Vec<JsCitation>,
    /// Artifacts completed in this chunk.
    pub artifacts: Vec<JsArtifact>,
}

/// Build a [`JsStreamChunk`] from the internal [`blazen_llm::StreamChunk`].
pub(crate) fn build_stream_chunk(chunk: blazen_llm::StreamChunk) -> JsStreamChunk {
    JsStreamChunk {
        delta: chunk.delta,
        finish_reason: chunk.finish_reason,
        tool_calls: chunk
            .tool_calls
            .into_iter()
            .map(|tc| JsToolCall {
                id: tc.id,
                name: tc.name,
                arguments: tc.arguments,
            })
            .collect(),
        reasoning_delta: chunk.reasoning_delta,
        citations: chunk.citations.iter().map(JsCitation::from).collect(),
        artifacts: chunk.artifacts.iter().map(JsArtifact::from).collect(),
    }
}
