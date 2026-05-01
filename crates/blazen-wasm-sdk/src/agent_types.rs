//! TS-facing copies of the agent-loop configuration and event surface
//! from [`blazen_llm::agent`].
//!
//! - [`WasmAgentConfig`] mirrors the data fields of
//!   [`blazen_llm::agent::AgentConfig`], minus the `tools` vec which can't
//!   cross the WASM ABI as a typed array of `Arc<dyn Tool>` (tools are
//!   passed as a separate JS array of `{ name, description, parameters,
//!   handler }` objects to [`crate::agent::run_agent`]).
//! - [`WasmAgentEvent`] mirrors [`blazen_llm::agent::AgentEvent`] using a
//!   tagged-union shape so the TS side can pattern-match on `kind`.
//! - [`run_agent_with_callback`] is the variant of [`crate::agent::run_agent`]
//!   that forwards each [`WasmAgentEvent`] to a JS callback as it occurs.

use std::pin::Pin;
use std::sync::Arc;

use serde::{Deserialize, Serialize};
use tsify_next::Tsify;
use wasm_bindgen::prelude::*;
use wasm_bindgen_futures::future_to_promise;

use blazen_llm::agent::{AgentConfig, AgentEvent};
use blazen_llm::traits::Tool;
use blazen_llm::types::{ToolCall, ToolDefinition};

use crate::chat_message::js_messages_to_vec;
use crate::completion_model::WasmCompletionModel;

// ---------------------------------------------------------------------------
// WasmAgentConfig (Tsify)
// ---------------------------------------------------------------------------

/// TS-facing copy of [`blazen_llm::agent::AgentConfig`].
///
/// All fields are optional on the JS side; defaults match
/// [`AgentConfig::new`] when omitted. Field names are serialised in
/// `camelCase` to match the convention already used by
/// [`crate::agent::run_agent`]'s manually-parsed options object.
#[derive(Debug, Clone, Default, Serialize, Deserialize, Tsify)]
#[serde(rename_all = "camelCase")]
#[tsify(into_wasm_abi, from_wasm_abi)]
pub struct WasmAgentConfig {
    /// Maximum number of tool call rounds before forcing a stop. Default: `10`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_iterations: Option<u32>,
    /// Whether to add an implicit "finish" tool the model can call to exit
    /// early. Default: `false`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub add_finish_tool: Option<bool>,
    /// When `true`, the canonical `finish_workflow` tool is **not** injected.
    /// Mirrors [`AgentConfig::no_finish_tool`]. Default: `false`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub no_finish_tool: Option<bool>,
    /// Override the canonical name used for the auto-injected
    /// `finish_workflow` tool. When `None` the default
    /// [`blazen_llm::FINISH_WORKFLOW_TOOL_NAME`] is used.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub finish_tool_name: Option<String>,
    /// Optional system prompt prepended to messages.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system_prompt: Option<String>,
    /// Sampling temperature.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    /// Maximum tokens per completion call.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
    /// Maximum number of tool calls to execute concurrently within a single
    /// model response. `0` means unlimited. Default: `0`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_concurrency: Option<u32>,
}

impl WasmAgentConfig {
    /// Apply this config onto a native [`AgentConfig`] (which already has its
    /// `tools` populated).
    fn apply_to(self, mut config: AgentConfig) -> AgentConfig {
        if let Some(n) = self.max_iterations {
            config = config.with_max_iterations(n);
        }
        if let Some(true) = self.add_finish_tool {
            config = config.with_finish_tool();
        }
        if let Some(true) = self.no_finish_tool {
            config = config.no_finish_tool();
        }
        if let Some(name) = self.finish_tool_name {
            config = config.finish_tool_name(name);
        }
        if let Some(s) = self.system_prompt {
            config = config.with_system_prompt(s);
        }
        if let Some(t) = self.temperature {
            config = config.with_temperature(t);
        }
        if let Some(mt) = self.max_tokens {
            config = config.with_max_tokens(mt);
        }
        if let Some(tc) = self.tool_concurrency {
            config = config.with_tool_concurrency(tc as usize);
        }
        config
    }
}

// ---------------------------------------------------------------------------
// WasmAgentEvent (Tsify tagged union)
// ---------------------------------------------------------------------------

/// TS-facing copy of [`blazen_llm::agent::AgentEvent`].
///
/// Serialized as `{ kind: "toolCalled", iteration, toolCall }` etc. so JS
/// callers can dispatch on `kind` in a single switch. Both the variant
/// discriminator and the inner fields use `camelCase`.
#[derive(Debug, Clone, Serialize, Deserialize, Tsify)]
#[serde(tag = "kind", rename_all = "camelCase")]
#[tsify(into_wasm_abi, from_wasm_abi)]
pub enum WasmAgentEvent {
    /// A tool was called by the model.
    #[serde(rename_all = "camelCase")]
    ToolCalled {
        /// Iteration number (1-based) in which the call was issued.
        iteration: u32,
        /// The full tool call payload from the model.
        tool_call: ToolCall,
    },
    /// A tool execution completed with a result.
    #[serde(rename_all = "camelCase")]
    ToolResult {
        /// Iteration number in which the tool ran.
        iteration: u32,
        /// Name of the tool that produced the result.
        tool_name: String,
        /// JSON-encoded tool result.
        result: serde_json::Value,
    },
    /// A model iteration completed.
    #[serde(rename_all = "camelCase")]
    IterationComplete {
        /// Iteration number that just finished.
        iteration: u32,
        /// Whether the model emitted any tool calls in this iteration.
        had_tool_calls: bool,
    },
}

impl From<AgentEvent> for WasmAgentEvent {
    fn from(value: AgentEvent) -> Self {
        match value {
            AgentEvent::ToolCalled {
                iteration,
                tool_call,
            } => Self::ToolCalled {
                iteration,
                tool_call,
            },
            AgentEvent::ToolResult {
                iteration,
                tool_name,
                result,
            } => Self::ToolResult {
                iteration,
                tool_name,
                result,
            },
            AgentEvent::IterationComplete {
                iteration,
                had_tool_calls,
            } => Self::IterationComplete {
                iteration,
                had_tool_calls,
            },
        }
    }
}

// ---------------------------------------------------------------------------
// run_agent_with_callback
// ---------------------------------------------------------------------------

/// `Send`-marker wrapper around a `!Send` future. SAFETY: WASM is single-threaded.
struct SendFuture<F>(F);

unsafe impl<F> Send for SendFuture<F> {}

impl<F: std::future::Future> std::future::Future for SendFuture<F> {
    type Output = F::Output;

    fn poll(
        self: Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Self::Output> {
        let inner = unsafe { self.map_unchecked_mut(|s| &mut s.0) };
        inner.poll(cx)
    }
}

/// Tool whose execution is delegated to a JS function.
///
/// Mirrors [`crate::agent::JsTool`] (which is module-private); duplicated
/// here because we need the same delegation semantics for the
/// `runAgentWithCallback` entry point but cannot reach the private type.
struct JsTool {
    definition: ToolDefinition,
    handler: js_sys::Function,
    /// When `true`, the agent loop exits immediately after this tool runs
    /// and the tool's arguments become the final result. Mirrors
    /// [`blazen_llm::traits::Tool::is_exit`].
    is_exit: bool,
}

unsafe impl Send for JsTool {}
unsafe impl Sync for JsTool {}

impl std::fmt::Debug for JsTool {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("JsTool")
            .field("name", &self.definition.name)
            .finish_non_exhaustive()
    }
}

fn js_to_tool_output(
    js_value: JsValue,
) -> Result<blazen_llm::types::ToolOutput<serde_json::Value>, blazen_llm::BlazenError> {
    use blazen_llm::types::ToolOutput;

    let raw: serde_json::Value = if let Some(s) = js_value.as_string() {
        serde_json::from_str(&s).unwrap_or(serde_json::Value::String(s))
    } else {
        serde_wasm_bindgen::from_value(js_value)
            .map_err(|e| blazen_llm::BlazenError::tool_error(e.to_string()))?
    };

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
    async fn execute_impl(
        &self,
        arguments: serde_json::Value,
    ) -> Result<blazen_llm::types::ToolOutput<serde_json::Value>, blazen_llm::BlazenError> {
        let js_args = serde_wasm_bindgen::to_value(&arguments)
            .map_err(|e| blazen_llm::BlazenError::tool_error(e.to_string()))?;

        let result = self
            .handler
            .call1(&JsValue::NULL, &js_args)
            .map_err(|e| blazen_llm::BlazenError::tool_error(format!("{e:?}")))?;

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

    fn is_exit(&self) -> bool {
        self.is_exit
    }

    async fn execute(
        &self,
        arguments: serde_json::Value,
    ) -> Result<blazen_llm::types::ToolOutput<serde_json::Value>, blazen_llm::BlazenError> {
        SendFuture(self.execute_impl(arguments)).await
    }
}

fn parse_tools(tools: &JsValue) -> Result<Vec<Arc<dyn Tool>>, JsValue> {
    let tools_array = js_sys::Array::from(tools);
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
            .ok_or_else(|| JsValue::from_str(&format!("Tool '{name}' missing 'description'")))?;

        let params_js =
            js_sys::Reflect::get(&tool_obj, &JsValue::from_str("parameters")).map_err(|e| {
                JsValue::from_str(&format!("Tool '{name}' missing 'parameters': {e:?}"))
            })?;
        let parameters: serde_json::Value = serde_wasm_bindgen::from_value(params_js)
            .map_err(|e| JsValue::from_str(&format!("Tool '{name}' invalid 'parameters': {e}")))?;

        let handler_js = js_sys::Reflect::get(&tool_obj, &JsValue::from_str("handler"))
            .map_err(|e| JsValue::from_str(&format!("Tool '{name}' missing 'handler': {e:?}")))?;
        let handler: js_sys::Function = handler_js
            .dyn_into()
            .map_err(|_| JsValue::from_str(&format!("Tool '{name}' handler is not a function")))?;

        let is_exit = js_sys::Reflect::get(&tool_obj, &JsValue::from_str("isExit"))
            .ok()
            .map(|v| v.is_truthy())
            .unwrap_or(false);

        tool_impls.push(Arc::new(JsTool {
            definition: ToolDefinition {
                name,
                description,
                parameters,
            },
            handler,
            is_exit,
        }));
    }

    Ok(tool_impls)
}

/// Run the agent loop, forwarding each [`WasmAgentEvent`] to a JS callback.
///
/// The callback is invoked synchronously for every event (`tool_called`,
/// `tool_result`, `iteration_complete`) with a tagged-union object matching
/// [`WasmAgentEvent`]. The callback's return value is ignored; throwing an
/// exception will surface as a rejected promise from the outer call.
///
/// Tool definitions and `options` follow the same shape documented on
/// [`crate::agent::run_agent`].
#[wasm_bindgen(js_name = "runAgentWithCallback")]
pub fn run_agent_with_callback(
    model: &WasmCompletionModel,
    messages: JsValue,
    tools: JsValue,
    options: JsValue,
    on_event: js_sys::Function,
) -> js_sys::Promise {
    let model_arc = model.inner_arc();

    future_to_promise(async move {
        let msgs = js_messages_to_vec(&messages)?;
        let tool_impls = parse_tools(&tools)?;

        let mut config = AgentConfig::new(tool_impls);

        if options.is_object() {
            let parsed: WasmAgentConfig = serde_wasm_bindgen::from_value(options)
                .map_err(|e| JsValue::from_str(&format!("invalid agent options: {e}")))?;
            config = parsed.apply_to(config);
        }

        let serializer = serde_wasm_bindgen::Serializer::new().serialize_maps_as_objects(true);

        // SAFETY: WASM is single-threaded, so the `!Send` `js_sys::Function`
        // is never observed across threads. We wrap it in a transparent
        // newtype that asserts `Send + Sync` to satisfy the
        // `run_agent_with_callback` trait bound.
        struct SyncFn(js_sys::Function);
        unsafe impl Send for SyncFn {}
        unsafe impl Sync for SyncFn {}

        let on_event_cb = SyncFn(on_event);

        let result = SendFuture(blazen_llm::run_agent_with_callback(
            model_arc.as_ref(),
            msgs,
            config,
            move |event| {
                let wasm_event: WasmAgentEvent = event.into();
                if let Ok(js_event) = wasm_event.serialize(&serializer) {
                    let _ = on_event_cb.0.call1(&JsValue::NULL, &js_event);
                }
            },
        ))
        .await
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

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

        let messages_js = serde_wasm_bindgen::to_value(&result.messages)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        js_sys::Reflect::set(&obj, &JsValue::from_str("messages"), &messages_js)?;

        Ok(obj.into())
    })
}
