//! Tool definition and invocation types.

use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Tool definitions and calls
// ---------------------------------------------------------------------------

/// Describes a tool that the model may invoke during a conversation.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "tsify", derive(tsify_next::Tsify))]
#[cfg_attr(feature = "tsify", tsify(into_wasm_abi, from_wasm_abi))]
pub struct ToolDefinition {
    /// The unique name of the tool.
    pub name: String,
    /// A human-readable description of what the tool does.
    pub description: String,
    /// A JSON Schema object describing the tool's input parameters.
    pub parameters: serde_json::Value,
}

/// A tool invocation requested by the model.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "tsify", derive(tsify_next::Tsify))]
#[cfg_attr(feature = "tsify", tsify(into_wasm_abi, from_wasm_abi))]
pub struct ToolCall {
    /// Provider-assigned identifier for this specific invocation.
    pub id: String,
    /// The name of the tool to invoke.
    pub name: String,
    /// The arguments to pass, as a JSON value.
    pub arguments: serde_json::Value,
}
