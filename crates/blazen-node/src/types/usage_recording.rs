//! Node bindings for `blazen_llm::usage_recording`: the `UsageEmitter` family
//! and the `UsageRecording*` decorators that emit `UsageEvent` after each
//! provider call.
//!
//! The Rust trait `UsageEmitter::emit` is sync, so the JS callback is wrapped
//! in a `ThreadsafeFunction` and invoked from whichever tokio worker the
//! provider call lands on. Errors thrown by the JS callback are caught and
//! discarded -- a buggy emitter must not be allowed to abort a completion.

use std::sync::Arc;

use napi::Either;
use napi::bindgen_prelude::Result;
use napi::threadsafe_function::{ThreadsafeFunction, ThreadsafeFunctionCallMode};
use napi_derive::napi;
use uuid::Uuid;

use blazen_events::UsageEvent;
use blazen_llm::traits::{CompletionModel, EmbeddingModel};
use blazen_llm::usage_recording::{
    NoopUsageEmitter as CoreNoopUsageEmitter, UsageEmitter,
    UsageRecordingCompletionModel as CoreUsageRecordingCompletionModel,
    UsageRecordingEmbeddingModel as CoreUsageRecordingEmbeddingModel,
};

use super::embedding::JsEmbeddingModel;
use super::events_wave::JsUsageEvent;
use crate::providers::completion_model::JsCompletionModel;

// ---------------------------------------------------------------------------
// JsUsageEmitter
// ---------------------------------------------------------------------------

/// A sink for emitted [`JsUsageEvent`]s.
///
/// Construct with a JS callback that handles each event. The callback runs
/// on the libuv main thread, so it can do anything synchronous; a thrown
/// error is caught and logged, never propagated into the completion call.
///
/// ```javascript
/// const events: UsageEvent[] = [];
/// const emitter = new UsageEmitter((event) => { events.push(event); });
/// const model = new UsageRecordingCompletionModel(base, emitter, "openai");
/// ```
#[napi(js_name = "UsageEmitter")]
pub struct JsUsageEmitter {
    pub(crate) handler:
        Arc<ThreadsafeFunction<JsUsageEvent, (), JsUsageEvent, napi::Status, false>>,
}

#[napi]
#[allow(clippy::must_use_candidate, clippy::needless_pass_by_value)]
impl JsUsageEmitter {
    /// Create an emitter from a JS callback. The callback is invoked once
    /// per emitted event.
    #[napi(constructor)]
    pub fn new(
        callback: ThreadsafeFunction<JsUsageEvent, (), JsUsageEvent, napi::Status, false>,
    ) -> Self {
        Self {
            handler: Arc::new(callback),
        }
    }
}

impl JsUsageEmitter {
    /// Build an `Arc<dyn UsageEmitter>` that dispatches into the JS callback.
    pub(crate) fn as_dyn_emitter(&self) -> Arc<dyn UsageEmitter> {
        Arc::new(JsUsageEmitterAdapter {
            handler: Arc::clone(&self.handler),
        })
    }
}

// Internal adapter that implements the Rust trait by calling the TSFN.
#[derive(Clone)]
struct JsUsageEmitterAdapter {
    handler: Arc<ThreadsafeFunction<JsUsageEvent, (), JsUsageEvent, napi::Status, false>>,
}

impl std::fmt::Debug for JsUsageEmitterAdapter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("JsUsageEmitterAdapter")
    }
}

impl UsageEmitter for JsUsageEmitterAdapter {
    fn emit(&self, event: UsageEvent) {
        // ThreadsafeFunction::call is non-blocking: it queues the call on the
        // libuv loop and returns immediately. Dropped events (queue full or
        // function disposed) are intentionally swallowed so a failing JS
        // listener can't abort the completion.
        let js_event = JsUsageEvent::from(event);
        let _ = self
            .handler
            .call(js_event, ThreadsafeFunctionCallMode::NonBlocking);
    }
}

// ---------------------------------------------------------------------------
// JsNoopUsageEmitter
// ---------------------------------------------------------------------------

/// A no-op emitter that drops every event.
///
/// Useful as a default when no downstream observer is wired up:
///
/// ```javascript
/// const model = new UsageRecordingCompletionModel(base, new NoopUsageEmitter(), "openai");
/// ```
#[napi(js_name = "NoopUsageEmitter")]
pub struct JsNoopUsageEmitter;

#[napi]
#[allow(clippy::new_without_default, clippy::must_use_candidate)]
impl JsNoopUsageEmitter {
    /// Construct a no-op emitter.
    #[napi(constructor)]
    pub fn new() -> Self {
        Self
    }
}

impl JsNoopUsageEmitter {
    pub(crate) fn as_dyn_emitter() -> Arc<dyn UsageEmitter> {
        Arc::new(CoreNoopUsageEmitter)
    }
}

/// napi-friendly union of the two emitter shapes.
type AnyEmitter<'a> = Either<&'a JsUsageEmitter, &'a JsNoopUsageEmitter>;

fn emitter_arc(any: AnyEmitter<'_>) -> Arc<dyn UsageEmitter> {
    match any {
        Either::A(e) => e.as_dyn_emitter(),
        Either::B(_) => JsNoopUsageEmitter::as_dyn_emitter(),
    }
}

fn parse_run_id(run_id: Option<String>) -> Result<Uuid> {
    match run_id {
        Some(s) => Uuid::parse_str(&s)
            .map_err(|e| napi::Error::from_reason(format!("invalid runId UUID: {e}"))),
        None => Ok(Uuid::new_v4()),
    }
}

// ---------------------------------------------------------------------------
// JsUsageRecordingCompletionModel
// ---------------------------------------------------------------------------

fn require_completion_inner(model: &JsCompletionModel) -> Result<Arc<dyn CompletionModel>> {
    model.inner.clone().ok_or_else(|| {
        napi::Error::from_reason(
            "UsageRecordingCompletionModel: source CompletionModel has no inner provider",
        )
    })
}

/// A `CompletionModel` decorator that emits a `UsageEvent` after each
/// successful `complete` call. Mirrors
/// `blazen_llm::usage_recording::UsageRecordingCompletionModel`.
///
/// ```javascript
/// const base = CompletionModel.openai();
/// const events = [];
/// const emitter = new UsageEmitter((e) => events.push(e));
/// const model = new UsageRecordingCompletionModel(base, emitter, "openai");
/// const response = await model.complete([ChatMessage.user("hi")]);
/// ```
#[napi(js_name = "UsageRecordingCompletionModel")]
pub struct JsUsageRecordingCompletionModel {
    inner: Arc<dyn CompletionModel>,
}

#[napi]
#[allow(
    clippy::must_use_candidate,
    clippy::missing_errors_doc,
    clippy::needless_pass_by_value
)]
impl JsUsageRecordingCompletionModel {
    /// Wrap a `CompletionModel` with a usage-recording layer.
    #[napi(constructor)]
    pub fn new(
        model: &JsCompletionModel,
        emitter: AnyEmitter<'_>,
        provider_label: String,
        run_id: Option<String>,
    ) -> Result<Self> {
        let inner_model = require_completion_inner(model)?;
        let emitter_arc = emitter_arc(emitter);
        let run_id_uuid = parse_run_id(run_id)?;
        let wrapped = CoreUsageRecordingCompletionModel::from_arc(
            inner_model,
            emitter_arc,
            provider_label,
            run_id_uuid,
        );
        Ok(Self {
            inner: Arc::new(wrapped),
        })
    }

    /// The underlying provider's model id.
    #[napi(js_name = "modelId", getter)]
    pub fn model_id(&self) -> String {
        self.inner.model_id().to_owned()
    }

    /// Convert this decorator into a `CompletionModel` so it can be passed to
    /// APIs that expect the base type (`runAgent`, further decorators, …).
    #[napi(js_name = "toCompletionModel")]
    pub fn to_completion_model(&self) -> JsCompletionModel {
        JsCompletionModel {
            inner: Some(Arc::clone(&self.inner)),
            local_model: None,
            config: None,
        }
    }
}

// ---------------------------------------------------------------------------
// JsUsageRecordingEmbeddingModel
// ---------------------------------------------------------------------------

fn require_embedding_inner(model: &JsEmbeddingModel) -> Result<Arc<dyn EmbeddingModel>> {
    model.inner_arc().ok_or_else(|| {
        napi::Error::from_reason(
            "UsageRecordingEmbeddingModel: source EmbeddingModel has no inner provider",
        )
    })
}

/// An `EmbeddingModel` decorator that emits a `UsageEvent` after each
/// successful `embed` call.
#[napi(js_name = "UsageRecordingEmbeddingModel")]
pub struct JsUsageRecordingEmbeddingModel {
    inner: Arc<dyn EmbeddingModel>,
}

#[napi]
#[allow(
    clippy::must_use_candidate,
    clippy::missing_errors_doc,
    clippy::needless_pass_by_value
)]
impl JsUsageRecordingEmbeddingModel {
    /// Wrap an `EmbeddingModel` with a usage-recording layer.
    #[napi(constructor)]
    pub fn new(
        model: &JsEmbeddingModel,
        emitter: AnyEmitter<'_>,
        provider_label: String,
        run_id: Option<String>,
    ) -> Result<Self> {
        let inner_model = require_embedding_inner(model)?;
        let emitter_arc = emitter_arc(emitter);
        let run_id_uuid = parse_run_id(run_id)?;
        let wrapped = CoreUsageRecordingEmbeddingModel::from_arc(
            inner_model,
            emitter_arc,
            provider_label,
            run_id_uuid,
        );
        Ok(Self {
            inner: Arc::new(wrapped),
        })
    }

    /// The underlying provider's model id.
    #[napi(js_name = "modelId", getter)]
    pub fn model_id(&self) -> String {
        self.inner.model_id().to_owned()
    }

    /// Output dimensionality.
    #[napi(getter)]
    pub fn dimensions(&self) -> u32 {
        u32::try_from(self.inner.dimensions()).unwrap_or(u32::MAX)
    }
}
