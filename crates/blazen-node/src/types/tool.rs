//! Tool call and tool definition types.

use napi_derive::napi;

/// A tool invocation requested by the model.
#[napi(object)]
pub struct JsToolCall {
    pub id: String,
    pub name: String,
    pub arguments: serde_json::Value,
}

/// Describes a tool that the model may invoke during a conversation.
#[napi(object)]
pub struct JsToolDefinition {
    pub name: String,
    pub description: String,
    pub parameters: serde_json::Value,
}
