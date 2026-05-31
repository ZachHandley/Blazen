//! JavaScript bindings for [`blazen_core::SubExecutable`].
//!
//! Exposes [`JsSubExecutable`] as a subclassable NAPI class (JS-class →
//! Rust-trait ABC, mirroring the
//! [`CustomProvider`](crate::providers::custom::JsCustomProvider) bridge).
//! A user writes:
//!
//! ```typescript
//! class MyChild extends SubExecutable {
//!   async execute(input) {
//!     return { result: input.value * 2 };
//!   }
//! }
//! ```
//!
//! and the resulting object is a first-class
//! [`blazen_core::SubExecutable`] that the workflow engine can embed as a
//! child runner inside a parent `Workflow` via
//! [`SubPipelineStep`](crate::workflow::subpipeline_step::JsSubPipelineStep).
//!
//! ## Bridging JavaScript async → Rust async
//!
//! When the constructor is invoked through a JS subclass, it installs a
//! [`JsSubExecutableAdapter`] that holds a [`ThreadsafeFunction`] bound to
//! the JS `execute` override. Each Rust
//! [`SubExecutable::execute`](blazen_core::SubExecutable::execute) call
//! serializes the input JSON, schedules the JS callback on the v8 main
//! thread, awaits the returned `Promise`, and surfaces the resolved value
//! as the terminal JSON result.
//!
//! ## v1 limitation
//!
//! The parent [`Context`](blazen_core::Context) is *not* forwarded into the
//! JS `execute` override — napi-rs's JS references are not safe to
//! materialize off the v8 main thread, and `Context` carries a live session
//! registry that cannot be cheaply projected to JSON. The JS override
//! therefore receives only the input payload. This matches the input-only
//! shape used by the other v1 binding surfaces and is sufficient for the
//! request → response child-runner contract.

use std::sync::Arc;

use async_trait::async_trait;
use napi::bindgen_prelude::*;
use napi::threadsafe_function::ThreadsafeFunction;
use napi_derive::napi;

use blazen_core::{Context, SubExecutable, WorkflowError};

// ---------------------------------------------------------------------------
// ThreadsafeFunction alias
// ---------------------------------------------------------------------------

/// Pre-built JS callback for the `execute` override.
///
/// - `T = serde_json::Value`: the single JSON input argument.
/// - `Return = Promise<Option<serde_json::Value>>`: the JS override must
///   return a `Promise`; `Option<_>` maps `undefined`/`null` to
///   [`serde_json::Value::Null`].
/// - `CalleeHandled = false`: the JS override resolves or rejects directly.
/// - `Weak = true`: does not keep the Node event loop alive on its own.
type ExecuteTsfn = ThreadsafeFunction<
    serde_json::Value,
    Promise<Option<serde_json::Value>>,
    serde_json::Value,
    Status,
    false,
    true,
>;

// ---------------------------------------------------------------------------
// JsSubExecutableAdapter
// ---------------------------------------------------------------------------

/// Rust [`SubExecutable`] implementation backed by a JavaScript subclass
/// instance's `execute` override.
struct JsSubExecutableAdapter {
    execute: Arc<ExecuteTsfn>,
}

impl std::fmt::Debug for JsSubExecutableAdapter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("JsSubExecutableAdapter")
            .finish_non_exhaustive()
    }
}

#[async_trait]
impl SubExecutable for JsSubExecutableAdapter {
    async fn execute(
        &self,
        input: serde_json::Value,
        _ctx: Context,
    ) -> std::result::Result<serde_json::Value, WorkflowError> {
        // Phase 1: schedule the JS callback on the Node main thread and
        // await napi capturing its returned `Promise`.
        let promise = self.execute.call_async_catch(input).await.map_err(|e| {
            WorkflowError::Context(format!("SubExecutable: `execute` dispatch failed: {e}"))
        })?;

        // Phase 2: drive the JS async body to completion.
        let resolved = promise
            .await
            .map_err(|e| WorkflowError::Context(format!("SubExecutable: `execute` raised: {e}")))?;

        Ok(resolved.unwrap_or(serde_json::Value::Null))
    }
}

// ---------------------------------------------------------------------------
// JsSubExecutable
// ---------------------------------------------------------------------------

/// A user-defined child runner embeddable inside a parent `Workflow`.
///
/// Subclass `SubExecutable` and override `execute(input)` to run an opaque
/// JSON payload to completion, returning the terminal JSON value. The
/// resulting object can be embedded as a step via `SubPipelineStep`'s
/// `fromExecutable` factory.
///
/// ```typescript
/// import { SubExecutable, SubPipelineStep, Workflow } from "blazen";
///
/// class Doubler extends SubExecutable {
///   async execute(input) {
///     return { value: input.value * 2 };
///   }
/// }
///
/// const step = SubPipelineStep.fromExecutable(
///   "double", ["blazen::StartEvent"], ["double::output"], new Doubler(),
/// );
/// ```
///
/// Constructing `SubExecutable` directly (without a subclass override)
/// yields a runner whose `execute` reports an error — override `execute` to
/// give it behavior.
#[napi(js_name = "SubExecutable")]
pub struct JsSubExecutable {
    pub(crate) inner: Arc<dyn SubExecutable>,
}

#[napi]
#[allow(
    clippy::must_use_candidate,
    clippy::missing_errors_doc,
    clippy::needless_pass_by_value
)]
impl JsSubExecutable {
    /// Construct a `SubExecutable`.
    ///
    /// When invoked through a JS subclass that overrides `execute`, the
    /// constructor binds that override and dispatches every Rust
    /// [`SubExecutable::execute`](blazen_core::SubExecutable::execute) call
    /// to it. When invoked directly (no override), `execute` reports an
    /// error until overridden.
    #[napi(constructor)]
    pub fn new(this: This<'_>) -> Result<Self> {
        let inner: Arc<dyn SubExecutable> = match build_execute_tsfn(&this.object)? {
            Some(execute) => Arc::new(JsSubExecutableAdapter { execute }),
            None => Arc::new(UnimplementedSubExecutable),
        };
        Ok(Self { inner })
    }
}

impl JsSubExecutable {
    /// Clone the underlying `Arc<dyn SubExecutable>` so it can be embedded in
    /// a parent workflow step.
    pub(crate) fn executable(&self) -> Arc<dyn SubExecutable> {
        Arc::clone(&self.inner)
    }
}

// ---------------------------------------------------------------------------
// Direct-construction fallback
// ---------------------------------------------------------------------------

/// `SubExecutable` impl used when the class is constructed directly without
/// a JS `execute` override. Every call reports an error.
#[derive(Debug)]
struct UnimplementedSubExecutable;

#[async_trait]
impl SubExecutable for UnimplementedSubExecutable {
    async fn execute(
        &self,
        _input: serde_json::Value,
        _ctx: Context,
    ) -> std::result::Result<serde_json::Value, WorkflowError> {
        Err(WorkflowError::Context(
            "SubExecutable subclass must override `execute(input)`".to_owned(),
        ))
    }
}

// ---------------------------------------------------------------------------
// Helper: build the `execute` TSFN from the host object's override
// ---------------------------------------------------------------------------

/// Extract the JS `execute` override off `host_object` and build a
/// [`ThreadsafeFunction`] bound to the host instance. Returns `None` when no
/// `execute` override is present (direct construction).
fn build_execute_tsfn(host_object: &Object<'_>) -> Result<Option<Arc<ExecuteTsfn>>> {
    if !host_object.has_named_property("execute").unwrap_or(false) {
        return Ok(None);
    }

    let js_function: Function<'_, serde_json::Value, Promise<Option<serde_json::Value>>> =
        match host_object.get_named_property("execute") {
            Ok(f) => f,
            Err(_) => return Ok(None),
        };

    // Bind `this` so user code referencing `this.foo` resolves when the TSFN
    // callback fires on the Node main thread.
    let bound = js_function.bind(host_object).map_err(|e| {
        napi::Error::from_reason(format!(
            "SubExecutable: failed to bind `this` for `execute`: {e}"
        ))
    })?;

    let tsfn: ExecuteTsfn = bound
        .build_threadsafe_function::<serde_json::Value>()
        .weak::<true>()
        .build()
        .map_err(|e| {
            napi::Error::from_reason(format!(
                "SubExecutable: failed to build threadsafe function for `execute`: {e}"
            ))
        })?;

    Ok(Some(Arc::new(tsfn)))
}
