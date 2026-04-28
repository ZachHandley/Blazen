use std::future::Future;
use std::pin::Pin;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use napi::Status;
use napi::bindgen_prelude::*;
use napi::threadsafe_function::ThreadsafeFunction;
use napi_derive::napi;

use blazen_pipeline::{PipelineError, PipelineSnapshot};

use crate::error::pipeline_error_to_napi;
use crate::pipeline::pipeline::JsPipeline;
use crate::pipeline::snapshot::JsPipelineSnapshot;
use crate::pipeline::stage::{JsParallelStage, JsStage};

/// JS persist callback receiving a typed `PipelineSnapshot` and returning a
/// `Promise<void>`.
///
/// Generic parameters follow the napi-rs 3 convention used elsewhere in this
/// crate (see [`crate::providers::custom`] for a detailed walkthrough):
/// - `T = JsPipelineSnapshot`: the snapshot is moved into JS as the
///   `PipelineSnapshot` class instance.
/// - `Return = Promise<()>`: the JS handler must return a `Promise` (or be
///   `async`); resolution drives the persist future.
/// - `CalleeHandled = false`: no error-first callback convention -- the JS
///   handler resolves or rejects.
/// - `Weak = true`: the TSFN does not keep the Node event loop alive on its
///   own.
type PersistTsfn =
    ThreadsafeFunction<JsPipelineSnapshot, Promise<()>, JsPipelineSnapshot, Status, false, true>;

/// JS persist callback receiving the JSON-serialized snapshot and returning a
/// `Promise<void>`. See [`PersistTsfn`] for an explanation of the generic
/// parameters.
type PersistJsonTsfn = ThreadsafeFunction<String, Promise<()>, String, Status, false, true>;

/// Fluent builder for constructing a Pipeline.
#[napi(js_name = "PipelineBuilder")]
pub struct JsPipelineBuilder {
    inner: Mutex<Option<blazen_pipeline::PipelineBuilder>>,
}

#[napi]
#[allow(
    clippy::must_use_candidate,
    clippy::missing_errors_doc,
    clippy::missing_panics_doc,
    clippy::needless_pass_by_value
)]
impl JsPipelineBuilder {
    #[napi(constructor)]
    pub fn new(name: String) -> Self {
        Self {
            inner: Mutex::new(Some(blazen_pipeline::PipelineBuilder::new(name))),
        }
    }

    /// Append a sequential Stage to the pipeline.
    #[napi]
    pub fn stage(&self, stage: &JsStage) -> Result<&Self> {
        let core_stage = stage.take()?;
        let mut guard = self.inner.lock().expect("poisoned");
        let builder = guard.take().ok_or_else(|| {
            napi::Error::new(
                napi::Status::GenericFailure,
                "PipelineBuilder already consumed (build() was called)",
            )
        })?;
        *guard = Some(builder.stage(core_stage));
        Ok(self)
    }

    /// Append a `ParallelStage` to the pipeline.
    #[napi]
    pub fn parallel(&self, parallel: &JsParallelStage) -> Result<&Self> {
        let core_parallel = parallel.take()?;
        let mut guard = self.inner.lock().expect("poisoned");
        let builder = guard.take().ok_or_else(|| {
            napi::Error::new(
                napi::Status::GenericFailure,
                "PipelineBuilder already consumed (build() was called)",
            )
        })?;
        *guard = Some(builder.parallel(core_parallel));
        Ok(self)
    }

    /// Set a per-stage timeout in seconds. Each stage's workflow gets this duration.
    #[napi(js_name = "timeoutPerStage")]
    pub fn timeout_per_stage(&self, seconds: f64) -> Result<&Self> {
        let mut guard = self.inner.lock().expect("poisoned");
        let builder = guard.take().ok_or_else(|| {
            napi::Error::new(
                napi::Status::GenericFailure,
                "PipelineBuilder already consumed (build() was called)",
            )
        })?;
        *guard = Some(builder.timeout_per_stage(Duration::from_secs_f64(seconds)));
        Ok(self)
    }

    /// Register a persist callback that receives a typed `PipelineSnapshot`
    /// after each stage completes.
    ///
    /// The JS callback must return a `Promise<void>` (or be `async`). If the
    /// promise rejects, the rejection is wrapped as a `PipelineError` and
    /// propagated to the running pipeline, aborting it.
    #[napi(js_name = "onPersist")]
    pub fn on_persist(&self, callback: PersistTsfn) -> Result<&Self> {
        let tsfn = Arc::new(callback);
        let persist_fn: blazen_pipeline::PersistFn = Arc::new(
            move |snapshot: PipelineSnapshot| -> Pin<
                Box<dyn Future<Output = std::result::Result<(), PipelineError>> + Send>,
            > {
                let tsfn = Arc::clone(&tsfn);
                Box::pin(async move {
                    let js_snap = JsPipelineSnapshot::from_inner(snapshot);
                    // Phase 1: schedule the JS callback and capture the
                    // returned Promise.
                    let promise = tsfn.call_async(js_snap).await.map_err(|e| {
                        PipelineError::PersistFailed(format!(
                            "persist callback dispatch failed: {e}"
                        ))
                    })?;
                    // Phase 2: drive the JS Promise to completion.
                    promise.await.map_err(|e| {
                        PipelineError::PersistFailed(format!("persist callback rejected: {e}"))
                    })?;
                    Ok(())
                })
            },
        );

        let mut guard = self.inner.lock().expect("poisoned");
        let builder = guard.take().ok_or_else(|| {
            napi::Error::new(
                napi::Status::GenericFailure,
                "PipelineBuilder already consumed (build() was called)",
            )
        })?;
        *guard = Some(builder.on_persist(persist_fn));
        Ok(self)
    }

    /// Register a persist callback that receives the snapshot as a JSON
    /// string after each stage completes.
    ///
    /// The JS callback must return a `Promise<void>` (or be `async`). If the
    /// promise rejects, the rejection is wrapped as a `PipelineError` and
    /// propagated to the running pipeline, aborting it.
    #[napi(js_name = "onPersistJson")]
    pub fn on_persist_json(&self, callback: PersistJsonTsfn) -> Result<&Self> {
        let tsfn = Arc::new(callback);
        let persist_json_fn: blazen_pipeline::PersistJsonFn = Arc::new(
            move |json: String| -> Pin<
                Box<dyn Future<Output = std::result::Result<(), PipelineError>> + Send>,
            > {
                let tsfn = Arc::clone(&tsfn);
                Box::pin(async move {
                    // Phase 1: schedule the JS callback and capture the
                    // returned Promise.
                    let promise = tsfn.call_async(json).await.map_err(|e| {
                        PipelineError::PersistFailed(format!(
                            "persist callback dispatch failed: {e}"
                        ))
                    })?;
                    // Phase 2: drive the JS Promise to completion.
                    promise.await.map_err(|e| {
                        PipelineError::PersistFailed(format!("persist callback rejected: {e}"))
                    })?;
                    Ok(())
                })
            },
        );

        let mut guard = self.inner.lock().expect("poisoned");
        let builder = guard.take().ok_or_else(|| {
            napi::Error::new(
                napi::Status::GenericFailure,
                "PipelineBuilder already consumed (build() was called)",
            )
        })?;
        *guard = Some(builder.on_persist_json(persist_json_fn));
        Ok(self)
    }

    /// Validate and build the pipeline. Throws if no stages or duplicate names.
    #[napi]
    pub fn build(&self) -> Result<JsPipeline> {
        let mut guard = self.inner.lock().expect("poisoned");
        let builder = guard.take().ok_or_else(|| {
            napi::Error::new(
                napi::Status::GenericFailure,
                "PipelineBuilder already consumed",
            )
        })?;
        let pipeline = builder.build().map_err(pipeline_error_to_napi)?;
        Ok(JsPipeline::from_inner(pipeline))
    }
}
