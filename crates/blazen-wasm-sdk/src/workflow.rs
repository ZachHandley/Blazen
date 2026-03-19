//! `wasm-bindgen` wrapper for the Blazen workflow engine.
//!
//! The full workflow engine from `blazen-core` relies on `tokio::spawn` and
//! multi-producer channels which are not available in the WASM single-threaded
//! runtime. This module provides a simplified workflow abstraction that runs
//! steps sequentially via JS callback functions.
//!
//! **Status**: Stub implementation. The `addStep` / `run` API shape is
//! defined so that TypeScript consumers can code against the interface now,
//! but execution is not yet wired up. A follow-up will implement a
//! WASM-compatible event loop.

use wasm_bindgen::prelude::*;
use wasm_bindgen_futures::future_to_promise;

// ---------------------------------------------------------------------------
// WasmWorkflow
// ---------------------------------------------------------------------------

/// A simplified workflow engine that runs entirely in WASM.
///
/// Steps are JavaScript callback functions that receive an event and a
/// context object and return the next event (or `null` to stop).
///
/// ```js
/// const wf = new Workflow('my-flow');
/// wf.addStep('process', ['StartEvent'], (event, ctx) => {
///   return { type: 'StopEvent', result: event.data };
/// });
/// const result = await wf.run({ data: 'hello' });
/// ```
#[wasm_bindgen(js_name = "Workflow")]
pub struct WasmWorkflow {
    name: String,
    steps: Vec<WasmStepRegistration>,
}

/// Internal representation of a registered step.
struct WasmStepRegistration {
    name: String,
    event_types: Vec<String>,
    handler: js_sys::Function,
}

#[wasm_bindgen(js_class = "Workflow")]
impl WasmWorkflow {
    /// Create a new workflow with the given name.
    #[wasm_bindgen(constructor)]
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_owned(),
            steps: Vec::new(),
        }
    }

    /// The workflow name.
    #[wasm_bindgen(getter)]
    pub fn name(&self) -> String {
        self.name.clone()
    }

    /// Register a step handler.
    ///
    /// - `name` -- unique step identifier
    /// - `event_types` -- array of event type strings this step responds to
    ///   (e.g. `["StartEvent"]`)
    /// - `handler` -- a function `(event, context) => Event | null`.
    ///   Returning `null` or an object with `type: "StopEvent"` ends the
    ///   workflow. The handler may be async (return a `Promise`).
    #[wasm_bindgen(js_name = "addStep")]
    pub fn add_step(
        &mut self,
        name: &str,
        event_types: JsValue,
        handler: js_sys::Function,
    ) -> Result<(), JsValue> {
        let types_array = js_sys::Array::from(&event_types);
        let mut event_types_vec = Vec::with_capacity(types_array.length() as usize);
        for i in 0..types_array.length() {
            let t = types_array
                .get(i)
                .as_string()
                .ok_or_else(|| JsValue::from_str("event_types must be an array of strings"))?;
            event_types_vec.push(t);
        }

        self.steps.push(WasmStepRegistration {
            name: name.to_owned(),
            event_types: event_types_vec,
            handler,
        });

        Ok(())
    }

    /// Execute the workflow with the given input data.
    ///
    /// Returns a `Promise` that resolves to the final result (the `result`
    /// field of the `StopEvent`).
    ///
    /// The input is passed as the `data` field of a synthetic `StartEvent`.
    #[wasm_bindgen]
    pub fn run(&self, input: JsValue) -> js_sys::Promise {
        // Clone the step handlers into the async block.
        let steps: Vec<(String, Vec<String>, js_sys::Function)> = self
            .steps
            .iter()
            .map(|s| (s.name.clone(), s.event_types.clone(), s.handler.clone()))
            .collect();
        let workflow_name = self.name.clone();

        future_to_promise(async move {
            // Build the initial StartEvent.
            let start_event = js_sys::Object::new();
            js_sys::Reflect::set(
                &start_event,
                &JsValue::from_str("type"),
                &JsValue::from_str("StartEvent"),
            )?;
            js_sys::Reflect::set(&start_event, &JsValue::from_str("data"), &input)?;

            // Simple sequential event loop.
            let mut current_event: JsValue = start_event.into();
            let max_iterations = 100_u32;

            for _ in 0..max_iterations {
                // Determine the event type.
                let event_type = js_sys::Reflect::get(&current_event, &JsValue::from_str("type"))
                    .ok()
                    .and_then(|v| v.as_string())
                    .unwrap_or_else(|| "unknown".to_owned());

                // Check for StopEvent.
                if event_type == "StopEvent" {
                    let result =
                        js_sys::Reflect::get(&current_event, &JsValue::from_str("result"))
                            .unwrap_or(JsValue::UNDEFINED);
                    return Ok(result);
                }

                // Find a matching step.
                let matching_step = steps.iter().find(|(_, types, _)| {
                    types.iter().any(|t| t == &event_type)
                });

                let Some((_step_name, _types, handler)) = matching_step else {
                    return Err(JsValue::from_str(&format!(
                        "Workflow '{workflow_name}': no step handles event type '{event_type}'"
                    )));
                };

                // Build a simple context object.
                let ctx = js_sys::Object::new();
                js_sys::Reflect::set(
                    &ctx,
                    &JsValue::from_str("workflowName"),
                    &JsValue::from_str(&workflow_name),
                )?;

                // Call the handler.
                let result = handler.call2(&JsValue::NULL, &current_event, &ctx)
                    .map_err(|e| JsValue::from_str(&format!("Step handler error: {e:?}")))?;

                // If the result is a Promise, await it.
                let result = if result.has_type::<js_sys::Promise>() {
                    let promise: js_sys::Promise = result.unchecked_into();
                    wasm_bindgen_futures::JsFuture::from(promise)
                        .await
                        .map_err(|e| {
                            JsValue::from_str(&format!("Step handler promise rejected: {e:?}"))
                        })?
                } else {
                    result
                };

                // If the handler returned null/undefined, treat as StopEvent.
                if result.is_null() || result.is_undefined() {
                    return Ok(JsValue::UNDEFINED);
                }

                current_event = result;
            }

            Err(JsValue::from_str(&format!(
                "Workflow '{workflow_name}': exceeded maximum iterations ({max_iterations})"
            )))
        })
    }
}
