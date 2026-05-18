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

use crate::agent::blazen_error_to_jsvalue;
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

    // Structured ToolOutput shape: object with a `data` key. Anything else
    // is auto-wrapped as the raw `data` payload.
    let serde_json::Value::Object(mut obj) = raw else {
        return Ok(ToolOutput::new(raw));
    };
    if !obj.contains_key("data") {
        return Ok(ToolOutput::new(serde_json::Value::Object(obj)));
    }

    let data = obj.remove("data").unwrap_or(serde_json::Value::Null);

    // Accept either camelCase `llmOverride` (canonical JS) or snake_case
    // `llm_override` (legacy / Rust-style).
    let raw_override = obj
        .remove("llmOverride")
        .or_else(|| obj.remove("llm_override"));

    let llm_override = match raw_override {
        None | Some(serde_json::Value::Null) => None,
        Some(v) => {
            let wasm_payload: WasmLlmPayload = serde_json::from_value(v).map_err(|e| {
                blazen_llm::BlazenError::tool_error(format!("invalid llmOverride payload: {e}"))
            })?;
            let payload: blazen_llm::types::LlmPayload =
                wasm_payload.try_into().map_err(|e: String| {
                    blazen_llm::BlazenError::tool_error(format!(
                        "invalid llmOverride payload: {e}"
                    ))
                })?;
            Some(payload)
        }
    };

    Ok(ToolOutput { data, llm_override })
}

impl JsTool {
    async fn execute_impl(
        &self,
        arguments: serde_json::Value,
    ) -> Result<blazen_llm::types::ToolOutput<serde_json::Value>, blazen_llm::BlazenError> {
        let js_args = serde_wasm_bindgen::to_value(&arguments)
            .map_err(|e| blazen_llm::BlazenError::tool_error(e.to_string()))?;

        // JS-thrown Errors are preserved as `CallerError` with the original
        // `JsValue` payload so JS callers can still `instanceof` / inspect
        // properties after the rejection.
        let result = self
            .handler
            .call1(&JsValue::NULL, &js_args)
            .map_err(|e| {
                blazen_llm::BlazenError::caller_error(
                    format!(
                        "tool handler `{}` threw on invocation",
                        self.definition.name
                    ),
                    send_wrapper::SendWrapper::new(e),
                )
            })?;

        let result = if result.has_type::<js_sys::Promise>() {
            let promise: js_sys::Promise = result.unchecked_into();
            wasm_bindgen_futures::JsFuture::from(promise)
                .await
                .map_err(|e| {
                    blazen_llm::BlazenError::caller_error(
                        format!(
                            "tool handler `{}` promise rejected",
                            self.definition.name
                        ),
                        send_wrapper::SendWrapper::new(e),
                    )
                })?
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
        .map_err(blazen_error_to_jsvalue)?;

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

// ---------------------------------------------------------------------------
// WasmLlmPayload + content-part mirrors (JS-facing tool-output override)
// ---------------------------------------------------------------------------
//
// These mirror the core [`blazen_llm::types::LlmPayload`] and
// [`blazen_llm::types::ContentPart`] families with a JS-friendly,
// discriminator-consistent shape: every variant uses a flat string tag
// (`kind` on the outer payload, `partType` on each content part,
// `sourceType` on each media source). The core types use serde tags
// `kind` / `type` (mixed) which is fine for Rust↔Rust wire traffic but
// confusing when round-tripping through JSON-shaped JS objects — and in
// the past has caused subtle "kind=parts but inner block has no `type`
// field" deserialization failures for honest JS callers.
//
// These types deserialize from the JS-side shape directly; conversion
// into the core types happens explicitly via [`TryFrom`].

/// JS-facing image source discriminator (TS-friendly `sourceType`).
#[derive(Debug, Clone, Serialize, Deserialize, Tsify)]
#[serde(rename_all = "camelCase")]
#[tsify(into_wasm_abi, from_wasm_abi)]
pub struct WasmImageSource {
    /// `"url"` or `"base64"`.
    pub source_type: String,
    /// Set for `sourceType: "url"`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub url: Option<String>,
    /// Set for `sourceType: "base64"`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub data: Option<String>,
}

/// JS-facing media source (used by File/Audio/Video).
#[derive(Debug, Clone, Serialize, Deserialize, Tsify)]
#[serde(rename_all = "camelCase")]
#[tsify(into_wasm_abi, from_wasm_abi)]
pub struct WasmMediaSource {
    /// `"url"` or `"base64"`.
    pub source_type: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub url: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub data: Option<String>,
}

/// JS-facing mirror of [`blazen_llm::types::ImageContent`].
#[derive(Debug, Clone, Serialize, Deserialize, Tsify)]
#[serde(rename_all = "camelCase")]
#[tsify(into_wasm_abi, from_wasm_abi)]
pub struct WasmImageContent {
    pub source: WasmImageSource,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub media_type: Option<String>,
}

/// JS-facing mirror of [`blazen_llm::types::FileContent`].
#[derive(Debug, Clone, Serialize, Deserialize, Tsify)]
#[serde(rename_all = "camelCase")]
#[tsify(into_wasm_abi, from_wasm_abi)]
pub struct WasmFileContent {
    pub source: WasmMediaSource,
    pub media_type: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub filename: Option<String>,
}

/// JS-facing mirror of [`blazen_llm::types::AudioContent`].
#[derive(Debug, Clone, Serialize, Deserialize, Tsify)]
#[serde(rename_all = "camelCase")]
#[tsify(into_wasm_abi, from_wasm_abi)]
pub struct WasmAudioContent {
    pub source: WasmMediaSource,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub media_type: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub duration_seconds: Option<f32>,
}

/// JS-facing mirror of [`blazen_llm::types::VideoContent`].
#[derive(Debug, Clone, Serialize, Deserialize, Tsify)]
#[serde(rename_all = "camelCase")]
#[tsify(into_wasm_abi, from_wasm_abi)]
pub struct WasmVideoContent {
    pub source: WasmMediaSource,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub media_type: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub duration_seconds: Option<f32>,
}

/// JS-facing flat `ContentPart` discriminator (`partType` field).
///
/// `partType` is one of `"text"`, `"image"`, `"file"`, `"audio"`,
/// `"video"`. Set the matching field (`text`, `image`, `file`, `audio`,
/// `video`) accordingly.
#[derive(Debug, Clone, Serialize, Deserialize, Tsify)]
#[serde(rename_all = "camelCase")]
#[tsify(into_wasm_abi, from_wasm_abi)]
pub struct WasmContentPart {
    pub part_type: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub text: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub image: Option<WasmImageContent>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub file: Option<WasmFileContent>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub audio: Option<WasmAudioContent>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub video: Option<WasmVideoContent>,
}

/// JS-facing `LlmPayload` mirror (`kind` outer + `partType` inner).
///
/// Variants:
/// - `{ kind: "text", text }`
/// - `{ kind: "json", value }`
/// - `{ kind: "parts", parts: WasmContentPart[] }`
/// - `{ kind: "provider_raw", provider, value }`
///
/// The `kind` tag uses the same `snake_case` strings as the core enum
/// (`text`, `json`, `parts`, `provider_raw`) — the camelCase
/// `rename_all` on this struct applies to FIELD names only, not the tag
/// value. The converter matches the kind string literally.
#[derive(Debug, Clone, Serialize, Deserialize, Tsify)]
#[serde(rename_all = "camelCase")]
#[tsify(into_wasm_abi, from_wasm_abi)]
pub struct WasmLlmPayload {
    /// `"text"`, `"json"`, `"parts"`, or `"provider_raw"`.
    pub kind: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub text: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub value: Option<serde_json::Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parts: Option<Vec<WasmContentPart>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub provider: Option<String>,
}

impl TryFrom<WasmLlmPayload> for blazen_llm::types::LlmPayload {
    type Error = String;

    fn try_from(p: WasmLlmPayload) -> Result<Self, String> {
        use blazen_llm::types::{LlmPayload, ProviderId};
        match p.kind.as_str() {
            "text" => Ok(LlmPayload::Text {
                text: p
                    .text
                    .ok_or_else(|| "kind=text requires `text`".to_string())?,
            }),
            "json" => Ok(LlmPayload::Json {
                value: p
                    .value
                    .ok_or_else(|| "kind=json requires `value`".to_string())?,
            }),
            "parts" => {
                let raw = p.parts.unwrap_or_default();
                let mut parts = Vec::with_capacity(raw.len());
                for part in raw {
                    parts.push(wasm_content_part_to_rust(part)?);
                }
                Ok(LlmPayload::Parts { parts })
            }
            "provider_raw" => {
                let provider_str = p
                    .provider
                    .ok_or_else(|| "kind=provider_raw requires `provider`".to_string())?;
                let provider = match provider_str.as_str() {
                    "openai" => ProviderId::OpenAi,
                    "openai_compat" => ProviderId::OpenAiCompat,
                    "azure" => ProviderId::Azure,
                    "anthropic" => ProviderId::Anthropic,
                    "gemini" => ProviderId::Gemini,
                    "responses" => ProviderId::Responses,
                    "fal" => ProviderId::Fal,
                    other => return Err(format!("unknown provider: {other}")),
                };
                Ok(LlmPayload::ProviderRaw {
                    provider,
                    value: p
                        .value
                        .ok_or_else(|| "kind=provider_raw requires `value`".to_string())?,
                })
            }
            other => Err(format!("unknown llmOverride kind: {other}")),
        }
    }
}

fn wasm_content_part_to_rust(
    part: WasmContentPart,
) -> Result<blazen_llm::types::ContentPart, String> {
    use blazen_llm::types::{
        AudioContent, ContentPart, FileContent, ImageContent, VideoContent,
    };

    match part.part_type.as_str() {
        "text" => Ok(ContentPart::Text {
            text: part
                .text
                .ok_or_else(|| "partType=text requires `text`".to_string())?,
        }),
        "image" => {
            let img = part
                .image
                .ok_or_else(|| "partType=image requires `image`".to_string())?;
            Ok(ContentPart::Image(ImageContent {
                source: wasm_image_source_to_rust(img.source)?,
                media_type: img.media_type,
            }))
        }
        "file" => {
            let f = part
                .file
                .ok_or_else(|| "partType=file requires `file`".to_string())?;
            Ok(ContentPart::File(FileContent {
                source: wasm_media_source_to_rust(f.source)?,
                media_type: f.media_type,
                filename: f.filename,
            }))
        }
        "audio" => {
            let a = part
                .audio
                .ok_or_else(|| "partType=audio requires `audio`".to_string())?;
            Ok(ContentPart::Audio(AudioContent {
                source: wasm_media_source_to_rust(a.source)?,
                media_type: a.media_type,
                duration_seconds: a.duration_seconds,
            }))
        }
        "video" => {
            let v = part
                .video
                .ok_or_else(|| "partType=video requires `video`".to_string())?;
            Ok(ContentPart::Video(VideoContent {
                source: wasm_media_source_to_rust(v.source)?,
                media_type: v.media_type,
                duration_seconds: v.duration_seconds,
            }))
        }
        other => Err(format!("unknown partType: {other}")),
    }
}

fn wasm_image_source_to_rust(
    s: WasmImageSource,
) -> Result<blazen_llm::types::ImageSource, String> {
    use blazen_llm::types::ImageSource;
    match s.source_type.as_str() {
        "url" => Ok(ImageSource::Url {
            url: s
                .url
                .ok_or_else(|| "sourceType=url requires `url`".to_string())?,
        }),
        "base64" => Ok(ImageSource::Base64 {
            data: s
                .data
                .ok_or_else(|| "sourceType=base64 requires `data`".to_string())?,
        }),
        other => Err(format!("unknown image sourceType: {other}")),
    }
}

fn wasm_media_source_to_rust(
    s: WasmMediaSource,
) -> Result<blazen_llm::types::MediaSource, String> {
    use blazen_llm::types::MediaSource;
    match s.source_type.as_str() {
        "url" => Ok(MediaSource::Url {
            url: s
                .url
                .ok_or_else(|| "sourceType=url requires `url`".to_string())?,
        }),
        "base64" => Ok(MediaSource::Base64 {
            data: s
                .data
                .ok_or_else(|| "sourceType=base64 requires `data`".to_string())?,
        }),
        other => Err(format!("unknown media sourceType: {other}")),
    }
}
