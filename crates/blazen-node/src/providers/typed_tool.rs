//! NAPI bindings for typed tool definitions.
//!
//! Exposes [`JsTypedTool`] -- a class that wraps a JavaScript handler and
//! a JSON Schema parameter description into a [`Tool`] implementation,
//! plus the [`typed_tool_simple`] free function that builds one from
//! `(name, description, parameters, handler)` without the class
//! constructor boilerplate.

use std::sync::Arc;

use napi::Status;
use napi::bindgen_prelude::*;
use napi::threadsafe_function::ThreadsafeFunction;
use napi_derive::napi;

use blazen_llm::error::BlazenError;
use blazen_llm::traits::Tool;
use blazen_llm::types::ToolDefinition;

// ---------------------------------------------------------------------------
// Tool handler ThreadsafeFunction
// ---------------------------------------------------------------------------

/// Tool handler: takes (toolName: string, arguments: object) and returns a
/// JSON-serializable result (or Promise thereof).
type TypedToolHandlerTsfn = ThreadsafeFunction<
    FnArgs<(String, serde_json::Value)>,
    Promise<serde_json::Value>,
    FnArgs<(String, serde_json::Value)>,
    Status,
    false,
    true,
>;

// ---------------------------------------------------------------------------
// JsTypedTool
// ---------------------------------------------------------------------------

/// A typed tool definition wrapping a JS handler.
///
/// ```typescript
/// const tool = new TypedTool(
///   "getWeather",
///   "Get current weather for a city",
///   { type: "object", properties: { city: { type: "string" } }, required: ["city"] },
///   async (name, args) => ({ temp: 72, city: args.city }),
/// );
/// ```
#[napi(js_name = "TypedTool")]
pub struct JsTypedTool {
    pub(crate) inner: Arc<JsToolImpl>,
}

pub(crate) struct JsToolImpl {
    pub def: ToolDefinition,
    pub handler: Arc<TypedToolHandlerTsfn>,
    /// When `true`, [`Tool::is_exit`] returns `true` so the agent loop
    /// stops after this tool is invoked. Mirrors
    /// [`blazen_llm::TypedTool::exit_tool`] (Wave 6).
    pub is_exit: bool,
}

#[async_trait::async_trait]
impl Tool for JsToolImpl {
    fn definition(&self) -> ToolDefinition {
        self.def.clone()
    }

    fn is_exit(&self) -> bool {
        self.is_exit
    }

    async fn execute(
        &self,
        arguments: serde_json::Value,
    ) -> std::result::Result<blazen_llm::types::ToolOutput<serde_json::Value>, BlazenError> {
        let name = self.def.name.clone();
        let promise = self
            .handler
            .call_async(FnArgs::from((name, arguments)))
            .await
            .map_err(|e| BlazenError::tool_error(e.to_string()))?;
        let result = promise
            .await
            .map_err(|e| BlazenError::tool_error(e.to_string()))?;
        Ok(result.into())
    }
}

#[napi]
#[allow(clippy::must_use_candidate, clippy::missing_errors_doc)]
impl JsTypedTool {
    /// Construct a typed tool from a name, description, JSON Schema parameter
    /// object, and a JavaScript handler.
    #[napi(constructor)]
    pub fn new(
        name: String,
        description: String,
        parameters: serde_json::Value,
        handler: TypedToolHandlerTsfn,
    ) -> Result<Self> {
        Ok(Self {
            inner: Arc::new(JsToolImpl {
                def: ToolDefinition {
                    name,
                    description,
                    parameters,
                },
                handler: Arc::new(handler),
                is_exit: false,
            }),
        })
    }

    /// The tool name.
    #[napi(getter)]
    pub fn name(&self) -> String {
        self.inner.def.name.clone()
    }

    /// The tool description.
    #[napi(getter)]
    pub fn description(&self) -> String {
        self.inner.def.description.clone()
    }

    /// The JSON Schema parameter definition.
    #[napi(getter)]
    pub fn parameters(&self) -> serde_json::Value {
        self.inner.def.parameters.clone()
    }

    /// Whether this tool is marked as an "exit" tool, signalling the agent
    /// loop to stop after it runs. Mirrors
    /// [`blazen_llm::Tool::is_exit`] (Wave 6).
    #[napi(getter, js_name = "isExit")]
    pub fn is_exit(&self) -> bool {
        self.inner.is_exit
    }

    /// Builder-style toggle for the exit-tool marker. Returns a new
    /// [`JsTypedTool`] that shares the same underlying handler but
    /// reports the requested [`Tool::is_exit`] value. Mirrors
    /// [`blazen_llm::TypedTool::exit_tool`] (Wave 6).
    ///
    /// ```typescript
    /// const exit = new TypedTool(
    ///   "submit",
    ///   "Submit the final answer and exit",
    ///   { type: "object", properties: { answer: { type: "string" } }, required: ["answer"] },
    ///   async (_n, args) => ({ submitted: args.answer }),
    /// ).exitTool(true);
    /// ```
    #[napi(js_name = "exitTool")]
    #[must_use]
    pub fn exit_tool(&self, exit: bool) -> JsTypedTool {
        JsTypedTool {
            inner: Arc::new(JsToolImpl {
                def: self.inner.def.clone(),
                handler: Arc::clone(&self.inner.handler),
                is_exit: exit,
            }),
        }
    }
}

// ---------------------------------------------------------------------------
// typedToolSimple free function
// ---------------------------------------------------------------------------

/// Build a typed tool without invoking `new TypedTool(...)`.
///
/// ```typescript
/// const tool = typedToolSimple(
///   "echo",
///   "Echo back the input",
///   { type: "object", properties: { msg: { type: "string" } }, required: ["msg"] },
///   async (_name, args) => ({ echoed: args.msg }),
/// );
/// ```
#[napi(js_name = "typedToolSimple")]
#[allow(clippy::missing_errors_doc)]
pub fn typed_tool_simple(
    name: String,
    description: String,
    parameters: serde_json::Value,
    handler: TypedToolHandlerTsfn,
) -> Result<JsTypedTool> {
    JsTypedTool::new(name, description, parameters, handler)
}
