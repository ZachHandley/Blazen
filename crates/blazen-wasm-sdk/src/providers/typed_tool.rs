//! `wasm-bindgen` wrapper for ergonomic tool authoring.
//!
//! [`WasmTypedTool`] adapts a name + description + JSON Schema + JS handler
//! into the tool shape expected by [`crate::agent::run_agent`] (a plain JS
//! object with `name`, `description`, `parameters`, and `handler`).
//!
//! [`typed_tool_simple`] is a free function that constructs a `WasmTypedTool`
//! and returns it ready to pass in a `tools` array.

use wasm_bindgen::prelude::*;

/// A typed tool wrapper.
///
/// ```js
/// const addTool = new TypedTool(
///     'add',
///     'Add two numbers',
///     { type: 'object', properties: { a: { type: 'number' }, b: { type: 'number' } }, required: ['a', 'b'] },
///     ({ a, b }) => ({ sum: a + b }),
/// );
///
/// const result = await runAgent(model, [ChatMessage.user('What is 5+3?')], [addTool.asTool()]);
/// ```
#[wasm_bindgen(js_name = "TypedTool")]
pub struct WasmTypedTool {
    name: String,
    description: String,
    parameters: JsValue,
    handler: js_sys::Function,
}

// SAFETY: WASM is single-threaded.
unsafe impl Send for WasmTypedTool {}
unsafe impl Sync for WasmTypedTool {}

#[wasm_bindgen(js_class = "TypedTool")]
impl WasmTypedTool {
    /// Construct a typed tool.
    ///
    /// - `name` -- tool identifier sent to the LLM.
    /// - `description` -- natural-language description for the LLM.
    /// - `parameters` -- a JSON Schema object describing the handler's
    ///   argument shape.
    /// - `handler` -- a function called with the parsed argument object;
    ///   may be async.
    #[wasm_bindgen(constructor)]
    #[must_use]
    pub fn new(
        name: String,
        description: String,
        parameters: JsValue,
        handler: js_sys::Function,
    ) -> WasmTypedTool {
        Self {
            name,
            description,
            parameters,
            handler,
        }
    }

    /// The tool name.
    #[wasm_bindgen(getter)]
    pub fn name(&self) -> String {
        self.name.clone()
    }

    /// The tool description.
    #[wasm_bindgen(getter)]
    pub fn description(&self) -> String {
        self.description.clone()
    }

    /// The JSON Schema for the tool's argument object.
    #[wasm_bindgen(getter)]
    pub fn parameters(&self) -> JsValue {
        self.parameters.clone()
    }

    /// The JS handler function.
    #[wasm_bindgen(getter)]
    pub fn handler(&self) -> js_sys::Function {
        self.handler.clone()
    }

    /// Materialize this typed tool as the plain JS object shape expected by
    /// [`crate::agent::run_agent`]:
    /// `{ name, description, parameters, handler }`.
    #[wasm_bindgen(js_name = "asTool")]
    pub fn as_tool(&self) -> Result<JsValue, JsValue> {
        let obj = js_sys::Object::new();
        js_sys::Reflect::set(&obj, &JsValue::from_str("name"), &JsValue::from_str(&self.name))?;
        js_sys::Reflect::set(
            &obj,
            &JsValue::from_str("description"),
            &JsValue::from_str(&self.description),
        )?;
        js_sys::Reflect::set(&obj, &JsValue::from_str("parameters"), &self.parameters)?;
        js_sys::Reflect::set(&obj, &JsValue::from_str("handler"), &self.handler)?;
        Ok(obj.into())
    }
}

/// Construct a [`WasmTypedTool`] in a single free-function call.
///
/// Equivalent to `new TypedTool(name, description, parameters, handler)`.
#[wasm_bindgen(js_name = "typedToolSimple")]
#[must_use]
pub fn typed_tool_simple(
    name: String,
    description: String,
    parameters: JsValue,
    handler: js_sys::Function,
) -> WasmTypedTool {
    WasmTypedTool::new(name, description, parameters, handler)
}
