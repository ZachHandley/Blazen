//! `wasm-bindgen` wrapper for [`blazen_llm::compute::ComputeProvider`].
//!
//! Exposes a generic compute provider class with the full job lifecycle:
//! `submit`, `status`, `cancel`, `awaitCompletion`, and the convenience
//! `run` (submit + result) helper.
//!
//! Job lifecycle methods accept and return [`JobHandle`]-shaped objects that
//! match the tsify interface generated from
//! [`blazen_llm::compute::JobHandle`]. The provider itself is backed by a JS
//! handlers object so users can plug in their own backend (fal.ai,
//! Replicate, RunPod, an internal queue, etc.) without writing Rust.

use std::pin::Pin;

use wasm_bindgen::prelude::*;
use wasm_bindgen_futures::future_to_promise;

// ---------------------------------------------------------------------------
// SendFuture wrapper
// ---------------------------------------------------------------------------

/// Wrapper that unsafely implements `Send` for a non-Send future.
/// SAFETY: WASM is single-threaded.
struct SendFuture<F>(F);

// SAFETY: WASM is single-threaded.
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
// Helpers
// ---------------------------------------------------------------------------

async fn call_with_arg(
    handlers: &JsValue,
    method: &str,
    arg: &JsValue,
) -> Result<JsValue, JsValue> {
    let handler = js_sys::Reflect::get(handlers, &JsValue::from_str(method))
        .map_err(|e| JsValue::from_str(&format!("failed to get handler '{method}': {e:?}")))?;
    if !handler.is_function() {
        return Err(JsValue::from_str(&format!(
            "handler '{method}' is not a function"
        )));
    }
    let func: &js_sys::Function = handler.unchecked_ref();
    let result = func
        .call1(&JsValue::NULL, arg)
        .map_err(|e| JsValue::from_str(&format!("handler '{method}' threw: {e:?}")))?;
    if result.has_type::<js_sys::Promise>() {
        let promise: js_sys::Promise = result.unchecked_into();
        wasm_bindgen_futures::JsFuture::from(promise)
            .await
            .map_err(|e| JsValue::from_str(&format!("handler '{method}' rejected: {e:?}")))
    } else {
        Ok(result)
    }
}

async fn sleep_ms(ms: u32) {
    let promise = js_sys::Promise::new(&mut |resolve, _| {
        let global = js_sys::global();
        if let Ok(set_timeout) =
            js_sys::Reflect::get(&global, &JsValue::from_str("setTimeout"))
        {
            if set_timeout.is_function() {
                let func: &js_sys::Function = set_timeout.unchecked_ref();
                let _ = func.call2(
                    &JsValue::NULL,
                    &resolve,
                    #[allow(clippy::cast_lossless)]
                    &JsValue::from(ms as f64),
                );
            }
        }
    });
    let _ = wasm_bindgen_futures::JsFuture::from(promise).await;
}

fn extract_status_string(value: &JsValue) -> Option<String> {
    if let Some(s) = value.as_string() {
        return Some(s);
    }
    if value.is_object() {
        if let Ok(field) = js_sys::Reflect::get(value, &JsValue::from_str("status")) {
            if let Some(s) = field.as_string() {
                return Some(s);
            }
        }
    }
    None
}

// ---------------------------------------------------------------------------
// TypeScript type declarations
// ---------------------------------------------------------------------------

#[wasm_bindgen(typescript_custom_section)]
const TS_COMPUTE_PROVIDER_TYPES: &str = r#"
/** Handlers for `ComputeProvider`. Pass to the constructor. */
export interface ComputeProviderHandlers {
    /** Submit a `ComputeRequest` and return a `JobHandle`. */
    submit: (request: any) => Promise<any> | any;
    /** Return a `JobStatus` for a given job. May return a string variant or the full object. */
    status: (job: any) => Promise<any> | any;
    /** Return a `ComputeResult` once the job has finished. */
    result: (job: any) => Promise<any> | any;
    /** Cancel a running or queued job. */
    cancel: (job: any) => Promise<void> | void;
}

/** Options accepted by `ComputeProvider.awaitCompletion`. */
export interface AwaitCompletionOptions {
    /** Polling interval in milliseconds (default: 1000). */
    pollIntervalMs?: number;
    /** Maximum number of polls before giving up (default: 600). */
    maxPolls?: number;
}
"#;

// ---------------------------------------------------------------------------
// WasmComputeProvider
// ---------------------------------------------------------------------------

/// A generic compute provider backed by JavaScript handler functions.
///
/// Mirrors [`blazen_llm::compute::ComputeProvider`] from the Rust API and
/// exposes the full job lifecycle (`submit`, `status`, `cancel`,
/// `awaitCompletion`, `run`).
///
/// ```js
/// const compute = new ComputeProvider('fal', {
///   submit:  async (req)  => ({ id: 'job-1', provider: 'fal', model: req.model, submitted_at: new Date() }),
///   status:  async (job)  => 'Running',
///   result:  async (job)  => ({ output: { url: '...' }, timing: { ttft_ms: 0 }, metadata: {} }),
///   cancel:  async (job)  => {},
/// });
/// const job = await compute.submit({ model: 'fal-ai/flux/dev', input: { prompt: 'a fox' } });
/// const result = await compute.awaitCompletion(job, { pollIntervalMs: 500 });
/// ```
#[wasm_bindgen(js_name = "ComputeProvider")]
pub struct WasmComputeProvider {
    provider_id: String,
    handlers: JsValue,
}

// SAFETY: WASM is single-threaded.
unsafe impl Send for WasmComputeProvider {}
unsafe impl Sync for WasmComputeProvider {}

#[wasm_bindgen(js_class = "ComputeProvider")]
impl WasmComputeProvider {
    /// Create a new compute provider.
    ///
    /// @param providerId - Short identifier (e.g. `"fal"`, `"replicate"`).
    /// @param handlers   - Object with `submit`, `status`, `result`, and
    ///                     `cancel` async functions.
    #[wasm_bindgen(constructor)]
    #[must_use]
    pub fn new(provider_id: &str, handlers: JsValue) -> Self {
        Self {
            provider_id: provider_id.to_owned(),
            handlers,
        }
    }

    /// The provider identifier.
    #[wasm_bindgen(getter, js_name = "providerId")]
    #[must_use]
    pub fn provider_id(&self) -> String {
        self.provider_id.clone()
    }

    /// Submit a compute job and return a `JobHandle`.
    #[wasm_bindgen]
    pub fn submit(&self, request: JsValue) -> js_sys::Promise {
        let handlers = self.handlers.clone();
        future_to_promise(SendFuture(async move {
            call_with_arg(&handlers, "submit", &request).await
        }))
    }

    /// Poll the current status of a submitted job.
    #[wasm_bindgen]
    pub fn status(&self, job: JsValue) -> js_sys::Promise {
        let handlers = self.handlers.clone();
        future_to_promise(SendFuture(async move {
            call_with_arg(&handlers, "status", &job).await
        }))
    }

    /// Wait for a job to finish and return its `ComputeResult`.
    #[wasm_bindgen]
    pub fn result(&self, job: JsValue) -> js_sys::Promise {
        let handlers = self.handlers.clone();
        future_to_promise(SendFuture(async move {
            call_with_arg(&handlers, "result", &job).await
        }))
    }

    /// Cancel a running or queued job.
    #[wasm_bindgen]
    pub fn cancel(&self, job: JsValue) -> js_sys::Promise {
        let handlers = self.handlers.clone();
        future_to_promise(SendFuture(async move {
            call_with_arg(&handlers, "cancel", &job).await
        }))
    }

    /// Submit a job and wait for the result. Equivalent to
    /// `submit().then(awaitCompletion)`.
    #[wasm_bindgen]
    pub fn run(&self, request: JsValue) -> js_sys::Promise {
        let handlers = self.handlers.clone();
        future_to_promise(SendFuture(async move {
            let job = call_with_arg(&handlers, "submit", &request).await?;
            call_with_arg(&handlers, "result", &job).await
        }))
    }

    /// Poll `status` until the job reaches a terminal state, then resolve
    /// with the full `ComputeResult` from `result(job)`. Rejects if the
    /// job ends in `Failed` or `Cancelled` or if `maxPolls` is exceeded.
    #[wasm_bindgen(js_name = "awaitCompletion")]
    pub fn await_completion(&self, job: JsValue, options: JsValue) -> js_sys::Promise {
        let handlers = self.handlers.clone();
        future_to_promise(SendFuture(async move {
            let mut poll_interval_ms: u32 = 1000;
            let mut max_polls: u32 = 600;
            if options.is_object() {
                if let Ok(v) =
                    js_sys::Reflect::get(&options, &JsValue::from_str("pollIntervalMs"))
                {
                    if let Some(n) = v.as_f64() {
                        if n.is_finite() && n >= 0.0 {
                            #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
                            {
                                poll_interval_ms = n as u32;
                            }
                        }
                    }
                }
                if let Ok(v) = js_sys::Reflect::get(&options, &JsValue::from_str("maxPolls")) {
                    if let Some(n) = v.as_f64() {
                        if n.is_finite() && n >= 0.0 {
                            #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
                            {
                                max_polls = n as u32;
                            }
                        }
                    }
                }
            }
            for _ in 0..max_polls {
                let status_value = call_with_arg(&handlers, "status", &job).await?;
                let status_str = extract_status_string(&status_value).unwrap_or_default();
                match status_str.as_str() {
                    "Completed" => {
                        return call_with_arg(&handlers, "result", &job).await;
                    }
                    "Failed" => {
                        return Err(JsValue::from_str("compute job failed"));
                    }
                    "Cancelled" => {
                        return Err(JsValue::from_str("compute job was cancelled"));
                    }
                    _ => {}
                }
                sleep_ms(poll_interval_ms).await;
            }
            Err(JsValue::from_str(
                "compute job did not complete within the configured maxPolls",
            ))
        }))
    }
}
