//! `wasm-bindgen` wrapper for the agent loop from `blazen-llm`.
//!
//! Exposes `runAgent()` as an async function that orchestrates the LLM +
//! tool calling pattern entirely in WASM, with tool execution delegated to
//! JavaScript callback functions.

use std::pin::Pin;
use std::sync::Arc;

use wasm_bindgen::prelude::*;
use wasm_bindgen_futures::future_to_promise;

use blazen_llm::agent::AgentConfig;
use blazen_llm::traits::Tool;
use blazen_llm::types::ToolDefinition;

use crate::chat_message::js_messages_to_vec;
use crate::completion_model::WasmCompletionModel;

// ---------------------------------------------------------------------------
// SendFuture wrapper (same as in http_fetch.rs)
// ---------------------------------------------------------------------------

/// Wrapper that unsafely implements `Send` for a non-Send future.
/// SAFETY: WASM is single-threaded.
struct SendFuture<F>(F);

unsafe impl<F> Send for SendFuture<F> {}

impl<F: std::future::Future> std::future::Future for SendFuture<F> {
    type Output = F::Output;

    fn poll(
        self: Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Self::Output> {
        // SAFETY: We are not moving F, just projecting through the wrapper.
        let inner = unsafe { self.map_unchecked_mut(|s| &mut s.0) };
        inner.poll(cx)
    }
}

// ---------------------------------------------------------------------------
// JS Tool wrapper
// ---------------------------------------------------------------------------

/// A tool whose execution is delegated to a JavaScript function.
struct JsTool {
    definition: ToolDefinition,
    handler: js_sys::Function,
}

// SAFETY: WASM is single-threaded.
unsafe impl Send for JsTool {}
unsafe impl Sync for JsTool {}

impl std::fmt::Debug for JsTool {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("JsTool")
            .field("name", &self.definition.name)
            .finish_non_exhaustive()
    }
}

/// After the JS handler resolves to a `JsValue`, decide whether to treat it
/// as a structured [`blazen_llm::types::ToolOutput`] (must be an object with
/// a `data` key) or a bare value (wrapped in `ToolOutput::new`).
///
/// Accepts either `llm_override` (snake) or `llmOverride` (camel) for the
/// optional override field — JS callers naturally produce the camelCase
/// spelling, but the serde derive on `ToolOutput` expects the snake form, so
/// we canonicalize before delegating to `serde_json::from_value`.
fn js_to_tool_output(
    js_value: JsValue,
) -> Result<blazen_llm::types::ToolOutput<serde_json::Value>, blazen_llm::BlazenError> {
    use blazen_llm::types::ToolOutput;

    // First, normalize string results: try JSON-parse, falling back to a
    // string-typed Value. This preserves the prior behavior so that handlers
    // returning a JSON string of a structured ToolOutput still get unpacked.
    let raw: serde_json::Value = if let Some(s) = js_value.as_string() {
        serde_json::from_str(&s).unwrap_or(serde_json::Value::String(s))
    } else {
        serde_wasm_bindgen::from_value(js_value)
            .map_err(|e| blazen_llm::BlazenError::tool_error(e.to_string()))?
    };

    // Heuristic: an object with a `data` key is treated as a structured
    // ToolOutput. Anything else is wrapped as `ToolOutput::new(raw)`.
    if let serde_json::Value::Object(map) = &raw
        && map.contains_key("data")
    {
        let mut normalized = map.clone();
        if !normalized.contains_key("llm_override")
            && let Some(v) = normalized.remove("llmOverride")
        {
            normalized.insert("llm_override".into(), v);
        }
        let normalized_value = serde_json::Value::Object(normalized);
        if let Ok(out) = serde_json::from_value::<ToolOutput<serde_json::Value>>(normalized_value) {
            return Ok(out);
        }
    }

    Ok(ToolOutput::new(raw))
}

impl JsTool {
    /// The actual async execute implementation (non-Send).
    ///
    /// Returns a [`blazen_llm::types::ToolOutput`] directly: if the JS
    /// handler returned an object with a `data` key it is unpacked as a
    /// structured tool output (with optional `llm_override` /
    /// `llmOverride`); otherwise the entire value is treated as `data`
    /// with no override.
    async fn execute_impl(
        &self,
        arguments: serde_json::Value,
    ) -> Result<blazen_llm::types::ToolOutput<serde_json::Value>, blazen_llm::BlazenError> {
        // Convert arguments to JsValue.
        let js_args = serde_wasm_bindgen::to_value(&arguments)
            .map_err(|e| blazen_llm::BlazenError::tool_error(e.to_string()))?;

        // Call the JS handler.
        let result = self
            .handler
            .call1(&JsValue::NULL, &js_args)
            .map_err(|e| blazen_llm::BlazenError::tool_error(format!("{e:?}")))?;

        // If the result is a Promise, await it.
        let result = if result.has_type::<js_sys::Promise>() {
            let promise: js_sys::Promise = result.unchecked_into();
            wasm_bindgen_futures::JsFuture::from(promise)
                .await
                .map_err(|e| blazen_llm::BlazenError::tool_error(format!("{e:?}")))?
        } else {
            result
        };

        js_to_tool_output(result)
    }
}

#[async_trait::async_trait]
impl Tool for JsTool {
    fn definition(&self) -> ToolDefinition {
        self.definition.clone()
    }

    async fn execute(
        &self,
        arguments: serde_json::Value,
    ) -> Result<blazen_llm::types::ToolOutput<serde_json::Value>, blazen_llm::BlazenError> {
        // SAFETY: WASM is single-threaded, Send is vacuously satisfied.
        SendFuture(self.execute_impl(arguments)).await
    }
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Run the agent loop with the given model, messages, and tools.
///
/// Each tool in the `tools` array should be a JS object with:
/// - `name` (string) -- the tool name
/// - `description` (string) -- what the tool does
/// - `parameters` (object) -- JSON Schema for the tool's input
/// - `handler` (function) -- called with the arguments object, should return
///   a string or JSON value (may be async / return a Promise)
///
/// The optional `options` object supports:
/// - `toolConcurrency` (number) -- max concurrent tool calls per round
///   (default: 0 = unlimited)
/// - `maxIterations` (number) -- max tool call rounds (default: 10)
/// - `systemPrompt` (string) -- optional system prompt prepended to messages
/// - `temperature` (number) -- sampling temperature
/// - `maxTokens` (number) -- max tokens per completion call
/// - `addFinishTool` (boolean) -- add built-in finish tool the model can
///   call to exit early
///
/// Returns a `Promise` that resolves to a JS object with:
/// - `content` (string | undefined) -- the final text response
/// - `messages` (array) -- the full message history
/// - `iterations` (number) -- how many tool call rounds occurred
/// - `totalUsage` (object | undefined) -- aggregated token usage
/// - `totalCost` (number | undefined) -- aggregated cost in USD
///
/// ```js
/// const result = await runAgent(model, [ChatMessage.user('What is 15*7?')], [
///   {
///     name: 'multiply',
///     description: 'Multiply two numbers',
///     parameters: { type: 'object', properties: { a: { type: 'number' }, b: { type: 'number' } }, required: ['a', 'b'] },
///     handler: (args) => JSON.stringify({ result: args.a * args.b })
///   }
/// ], { toolConcurrency: 2, maxIterations: 5 });
/// ```
#[wasm_bindgen(js_name = "runAgent")]
pub fn run_agent(
    model: &WasmCompletionModel,
    messages: JsValue,
    tools: JsValue,
    options: JsValue,
) -> js_sys::Promise {
    let model_arc = model.inner_arc();

    future_to_promise(async move {
        let msgs = js_messages_to_vec(&messages)?;

        // Parse tool definitions from the JS array.
        let tools_array = js_sys::Array::from(&tools);
        let mut tool_impls: Vec<Arc<dyn Tool>> = Vec::with_capacity(tools_array.length() as usize);

        for i in 0..tools_array.length() {
            let tool_obj = tools_array.get(i);

            let name = js_sys::Reflect::get(&tool_obj, &JsValue::from_str("name"))
                .ok()
                .and_then(|v| v.as_string())
                .ok_or_else(|| JsValue::from_str(&format!("Tool at index {i} missing 'name'")))?;

            let description = js_sys::Reflect::get(&tool_obj, &JsValue::from_str("description"))
                .ok()
                .and_then(|v| v.as_string())
                .ok_or_else(|| {
                    JsValue::from_str(&format!("Tool '{name}' missing 'description'"))
                })?;

            let params_js = js_sys::Reflect::get(&tool_obj, &JsValue::from_str("parameters"))
                .map_err(|e| JsValue::from_str(&format!("Tool '{name}' missing 'parameters': {e:?}")))?;
            let parameters: serde_json::Value = serde_wasm_bindgen::from_value(params_js)
                .map_err(|e| JsValue::from_str(&format!("Tool '{name}' invalid 'parameters': {e}")))?;

            let handler_js = js_sys::Reflect::get(&tool_obj, &JsValue::from_str("handler"))
                .map_err(|e| JsValue::from_str(&format!("Tool '{name}' missing 'handler': {e:?}")))?;
            let handler: js_sys::Function = handler_js
                .dyn_into()
                .map_err(|_| JsValue::from_str(&format!("Tool '{name}' handler is not a function")))?;

            tool_impls.push(Arc::new(JsTool {
                definition: ToolDefinition {
                    name,
                    description,
                    parameters,
                },
                handler,
            }));
        }

        let mut config = AgentConfig::new(tool_impls);

        // Parse optional configuration from the options object.
        if options.is_object() {
            if let Ok(tc) = js_sys::Reflect::get(&options, &JsValue::from_str("toolConcurrency"))
            {
                if let Some(n) = tc.as_f64() {
                    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
                    {
                        config = config.with_tool_concurrency(n as usize);
                    }
                }
            }
            if let Ok(mi) = js_sys::Reflect::get(&options, &JsValue::from_str("maxIterations")) {
                if let Some(n) = mi.as_f64() {
                    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
                    {
                        config = config.with_max_iterations(n as u32);
                    }
                }
            }
            if let Ok(sp) = js_sys::Reflect::get(&options, &JsValue::from_str("systemPrompt")) {
                if let Some(s) = sp.as_string() {
                    config = config.with_system_prompt(s);
                }
            }
            if let Ok(t) = js_sys::Reflect::get(&options, &JsValue::from_str("temperature")) {
                if let Some(n) = t.as_f64() {
                    #[allow(clippy::cast_possible_truncation)]
                    {
                        config = config.with_temperature(n as f32);
                    }
                }
            }
            if let Ok(mt) = js_sys::Reflect::get(&options, &JsValue::from_str("maxTokens")) {
                if let Some(n) = mt.as_f64() {
                    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
                    {
                        config = config.with_max_tokens(n as u32);
                    }
                }
            }
            if let Ok(af) = js_sys::Reflect::get(&options, &JsValue::from_str("addFinishTool")) {
                if af.is_truthy() {
                    config = config.with_finish_tool();
                }
            }
        }

        let result = blazen_llm::run_agent(model_arc.as_ref(), msgs, config)
            .await
            .map_err(|e| JsValue::from_str(&e.to_string()))?;

        // Build the result JS object.
        let obj = js_sys::Object::new();
        js_sys::Reflect::set(
            &obj,
            &JsValue::from_str("content"),
            &match result.response.content {
                Some(ref c) => JsValue::from_str(c),
                None => JsValue::UNDEFINED,
            },
        )?;
        js_sys::Reflect::set(
            &obj,
            &JsValue::from_str("iterations"),
            &JsValue::from_f64(f64::from(result.iterations)),
        )?;

        if let Some(ref usage) = result.total_usage {
            let usage_obj = js_sys::Object::new();
            js_sys::Reflect::set(
                &usage_obj,
                &JsValue::from_str("promptTokens"),
                &JsValue::from_f64(f64::from(usage.prompt_tokens)),
            )?;
            js_sys::Reflect::set(
                &usage_obj,
                &JsValue::from_str("completionTokens"),
                &JsValue::from_f64(f64::from(usage.completion_tokens)),
            )?;
            js_sys::Reflect::set(
                &usage_obj,
                &JsValue::from_str("totalTokens"),
                &JsValue::from_f64(f64::from(usage.total_tokens)),
            )?;
            js_sys::Reflect::set(&obj, &JsValue::from_str("totalUsage"), &usage_obj)?;
        }

        if let Some(cost) = result.total_cost {
            js_sys::Reflect::set(
                &obj,
                &JsValue::from_str("totalCost"),
                &JsValue::from_f64(cost),
            )?;
        }

        // Serialize the full message history as JSON.
        let messages_js = serde_wasm_bindgen::to_value(&result.messages)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        js_sys::Reflect::set(&obj, &JsValue::from_str("messages"), &messages_js)?;

        Ok(obj.into())
    })
}
