//! Shared SSE (Server-Sent Events) parser for OpenAI-compatible streaming
//! responses.
//!
//! The `OpenAI` SSE protocol sends lines of the form `data: <json>` separated
//! by blank lines. The special value `data: [DONE]` signals the end of the
//! stream. This parser is reused by the OpenAI-compatible provider, the Azure
//! provider, and as a building block for the Gemini provider.

use std::pin::Pin;
use std::task::{Context, Poll};

use futures_util::Stream;
use serde::Deserialize;
use tracing::warn;

use crate::error::BlazenError;
use crate::http::ByteStream;
use crate::types::{StreamChunk, ToolCall};

// ---------------------------------------------------------------------------
// Wire types (deserialization only -- not part of public API)
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
pub(crate) struct OaiResponse {
    pub choices: Vec<OaiChoice>,
    pub model: String,
    pub usage: Option<OaiUsage>,
}

#[derive(Debug, Deserialize)]
pub(crate) struct OaiChoice {
    pub message: OaiMessage,
    pub finish_reason: Option<String>,
}

#[derive(Debug, Deserialize)]
pub(crate) struct OaiMessage {
    pub content: Option<String>,
    #[serde(default)]
    pub tool_calls: Vec<OaiToolCall>,
    #[serde(default)]
    pub reasoning_content: Option<String>,
    #[serde(default)]
    pub reasoning: Option<String>,
    #[serde(default)]
    pub citations: Vec<serde_json::Value>,
}

#[derive(Debug, Deserialize)]
pub(crate) struct OaiToolCall {
    pub id: String,
    pub function: OaiFunction,
}

#[derive(Debug, Deserialize)]
pub(crate) struct OaiFunction {
    pub name: String,
    pub arguments: String,
}

#[derive(Debug, Deserialize)]
#[allow(clippy::struct_field_names)]
pub(crate) struct OaiUsage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
    #[serde(default)]
    pub completion_tokens_details: Option<OaiCompletionTokensDetails>,
}

#[derive(Debug, Clone, Deserialize)]
pub(crate) struct OaiCompletionTokensDetails {
    #[serde(default)]
    pub reasoning_tokens: u32,
}

// Streaming wire types

#[derive(Debug, Deserialize)]
pub(crate) struct OaiStreamChunk {
    pub choices: Vec<OaiStreamChoice>,
    /// Present in the wire format but not used for mapping.
    #[allow(dead_code)]
    pub model: Option<String>,
}

#[derive(Debug, Deserialize)]
pub(crate) struct OaiStreamChoice {
    pub delta: OaiStreamDelta,
    pub finish_reason: Option<String>,
}

#[derive(Debug, Deserialize)]
pub(crate) struct OaiStreamDelta {
    pub content: Option<String>,
    #[serde(default)]
    pub tool_calls: Vec<OaiStreamToolCall>,
    /// Present in the wire format (`DeepSeek` R1 etc.); not yet surfaced in
    /// streaming chunks.
    #[serde(default)]
    #[allow(dead_code)]
    pub reasoning_content: Option<String>,
    /// Present in the wire format (Grok); not yet surfaced in streaming chunks.
    #[serde(default)]
    #[allow(dead_code)]
    pub reasoning: Option<String>,
    /// Present in the wire format (Perplexity); not yet surfaced in streaming
    /// chunks.
    #[serde(default)]
    #[allow(dead_code)]
    pub citations: Vec<serde_json::Value>,
}

#[derive(Debug, Deserialize)]
pub(crate) struct OaiStreamToolCall {
    /// Present in the wire format but not used for mapping.
    #[allow(dead_code)]
    pub index: Option<u32>,
    pub id: Option<String>,
    pub function: Option<OaiStreamFunction>,
}

#[derive(Debug, Deserialize)]
pub(crate) struct OaiStreamFunction {
    pub name: Option<String>,
    pub arguments: Option<String>,
}

// ---------------------------------------------------------------------------
// SSE stream parser
// ---------------------------------------------------------------------------

/// Parses an SSE byte stream from an OpenAI-compatible API into
/// [`StreamChunk`]s.
///
/// The protocol sends lines of the form `data: <json>` separated by blank
/// lines. The special value `data: [DONE]` signals the end of the stream.
pub(crate) struct SseParser {
    inner: ByteStream,
    buffer: String,
}

impl SseParser {
    pub fn new(inner: ByteStream) -> Self {
        Self {
            inner,
            buffer: String::new(),
        }
    }
}

impl Stream for SseParser {
    type Item = Result<StreamChunk, BlazenError>;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let this = self.get_mut();

        loop {
            // Try to extract a complete SSE event from the buffer.
            if let Some(chunk) = parse_next_event(&mut this.buffer) {
                return Poll::Ready(Some(chunk));
            }

            // Need more data from the underlying byte stream.
            match Pin::new(&mut this.inner).poll_next(cx) {
                Poll::Ready(Some(Ok(bytes))) => {
                    let text = String::from_utf8_lossy(&bytes);
                    this.buffer.push_str(&text);
                    // Loop back to try parsing again.
                }
                Poll::Ready(Some(Err(e))) => {
                    return Poll::Ready(Some(Err(BlazenError::stream_error(e.to_string()))));
                }
                Poll::Ready(None) => {
                    // Stream ended. If there's leftover data, try one more parse.
                    if !this.buffer.is_empty()
                        && let Some(chunk) = parse_next_event(&mut this.buffer)
                    {
                        return Poll::Ready(Some(chunk));
                    }
                    return Poll::Ready(None);
                }
                Poll::Pending => {
                    return Poll::Pending;
                }
            }
        }
    }
}

/// Try to extract the next SSE event from the buffer.
///
/// Returns `Some` if a complete event was found (including `[DONE]` which
/// maps to a final chunk with `finish_reason: "stop"`). Returns `None` if
/// more data is needed.
pub(crate) fn parse_next_event(buffer: &mut String) -> Option<Result<StreamChunk, BlazenError>> {
    loop {
        // Look for a complete line.
        let newline_pos = buffer.find('\n')?;
        let line = buffer[..newline_pos].trim().to_owned();
        // Remove the line (and the newline) from the buffer.
        buffer.drain(..=newline_pos);

        // Skip empty lines and comment lines.
        if line.is_empty() || line.starts_with(':') {
            continue;
        }

        if let Some(data) = line.strip_prefix("data: ") {
            let data = data.trim();

            if data == "[DONE]" {
                // Signal end of stream.
                return Some(Ok(StreamChunk {
                    delta: None,
                    tool_calls: Vec::new(),
                    finish_reason: Some("stop".to_owned()),
                    ..Default::default()
                }));
            }

            match serde_json::from_str::<OaiStreamChunk>(data) {
                Ok(chunk) => {
                    let Some(choice) = chunk.choices.into_iter().next() else {
                        continue; // No choice in this chunk, skip.
                    };

                    let tool_calls: Vec<ToolCall> = choice
                        .delta
                        .tool_calls
                        .into_iter()
                        .filter_map(|tc| {
                            let func = tc.function?;
                            let name = func.name?;
                            let args_str = func.arguments.unwrap_or_default();
                            let args =
                                serde_json::from_str(&args_str).unwrap_or(serde_json::Value::Null);
                            Some(ToolCall {
                                id: tc.id.unwrap_or_default(),
                                name,
                                arguments: args,
                            })
                        })
                        .collect();

                    return Some(Ok(StreamChunk {
                        delta: choice.delta.content,
                        tool_calls,
                        finish_reason: choice.finish_reason,
                        ..Default::default()
                    }));
                }
                Err(e) => {
                    warn!(error = %e, data, "failed to parse OpenAI SSE chunk");
                    return Some(Err(BlazenError::stream_error(format!(
                        "failed to parse SSE chunk: {e}"
                    ))));
                }
            }
        }

        // Lines that don't start with "data:" are ignored (e.g. "event:", "id:").
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_sse_text_delta() {
        let mut buf = "data: {\"choices\":[{\"delta\":{\"content\":\"Hello\"},\"finish_reason\":null}],\"model\":\"gpt-4o\"}\n\n".to_owned();

        let result = parse_next_event(&mut buf).unwrap().unwrap();
        assert_eq!(result.delta.as_deref(), Some("Hello"));
        assert!(result.finish_reason.is_none());
    }

    #[test]
    fn parse_sse_done() {
        let mut buf = "data: [DONE]\n\n".to_owned();

        let result = parse_next_event(&mut buf).unwrap().unwrap();
        assert!(result.delta.is_none());
        assert_eq!(result.finish_reason.as_deref(), Some("stop"));
    }

    #[test]
    fn parse_sse_skips_empty_lines() {
        let mut buf = "\n\ndata: {\"choices\":[{\"delta\":{\"content\":\"Hi\"},\"finish_reason\":null}],\"model\":\"gpt-4o\"}\n\n".to_owned();

        let result = parse_next_event(&mut buf).unwrap().unwrap();
        assert_eq!(result.delta.as_deref(), Some("Hi"));
    }

    #[test]
    fn parse_sse_incomplete_returns_none() {
        let mut buf = "data: {\"choices\"".to_owned();

        // No newline yet, so we can't extract a complete line.
        let result = parse_next_event(&mut buf);
        assert!(result.is_none());
    }

    #[test]
    fn parse_sse_with_finish_reason() {
        let mut buf = "data: {\"choices\":[{\"delta\":{\"content\":null},\"finish_reason\":\"stop\"}],\"model\":\"gpt-4o\"}\n\n".to_owned();

        let result = parse_next_event(&mut buf).unwrap().unwrap();
        assert!(result.delta.is_none());
        assert_eq!(result.finish_reason.as_deref(), Some("stop"));
    }

    #[test]
    fn parse_sse_tool_call() {
        let mut buf = "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"call_123\",\"function\":{\"name\":\"get_weather\",\"arguments\":\"{\\\"city\\\":\\\"NYC\\\"}\"}}]},\"finish_reason\":null}],\"model\":\"gpt-4o\"}\n\n".to_owned();

        let result = parse_next_event(&mut buf).unwrap().unwrap();
        assert!(result.delta.is_none());
        assert_eq!(result.tool_calls.len(), 1);
        assert_eq!(result.tool_calls[0].id, "call_123");
        assert_eq!(result.tool_calls[0].name, "get_weather");
        assert_eq!(result.tool_calls[0].arguments["city"], "NYC");
    }
}
