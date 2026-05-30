//! Workflow surface for the UniFFI bindings.
//!
//! The Blazen workflow engine is built around the `Event`/`Step` proc-macro
//! abstraction in `blazen-macros` — strongly-typed at compile time. That model
//! doesn't survive an FFI boundary; foreign-language code can't define new
//! `Event` types at compile time. Instead, this module exposes a
//! **JSON-event** workflow surface that mirrors what `blazen-py` does
//! internally (see `crates/blazen-py/src/workflow/step.rs`):
//!
//! - Events on the wire are `{ event_type: String, data_json: String }`.
//! - Step handlers are `Arc<dyn StepHandler>` trait objects with foreign-
//!   language implementations (Go func, Swift `Sendable` closure-bearing
//!   struct, Kotlin object, Ruby class).
//! - Each step declares the event types it `accepts` and `emits` so the
//!   workflow engine routes events correctly.
//!
//! ## Example (Go)
//!
//! ```go,ignore
//! type GreetHandler struct{}
//! func (g *GreetHandler) Invoke(ctx context.Context, input blazen.Event) (blazen.StepOutput, error) {
//!     var data map[string]any
//!     json.Unmarshal([]byte(input.DataJson), &data)
//!     name, _ := data["name"].(string)
//!     return blazen.StepOutput_Single(blazen.Event{
//!         EventType: "StopEvent",
//!         DataJson:  fmt.Sprintf(`{"result":"Hello, %s!"}`, name),
//!     }), nil
//! }
//!
//! wf, err := blazen.NewWorkflowBuilder("greeter").
//!     Step("greet", []string{"StartEvent"}, []string{"StopEvent"}, &GreetHandler{}).
//!     Build()
//! result, err := wf.Run(ctx, `{"name":"Zach"}`)
//! ```

use std::sync::Arc;

use tokio::sync::Mutex as AsyncMutex;
use tokio_stream::StreamExt;

use crate::errors::{BlazenError, BlazenResult};
use crate::llm::TokenUsage;
use crate::runtime::runtime;

use blazen_core::{
    Context as CoreContext, StepOutput as CoreStepOutput, StepRegistration,
    Workflow as CoreWorkflow, WorkflowBuilder as CoreWorkflowBuilder,
    WorkflowHandler as CoreWorkflowHandler,
};
use blazen_events::{AnyEvent, DynamicEvent, StartEvent, StopEvent, intern_event_type};

// ---------------------------------------------------------------------------
// Wire-format records
// ---------------------------------------------------------------------------

/// Event crossed across the FFI boundary.
///
/// `event_type` is a free-form string naming the event class (e.g.
/// `"StartEvent"`, `"StopEvent"`, `"MyCustomEvent"`). `data_json` is a
/// JSON-encoded payload. Foreign-language wrappers typically marshal these
/// to/from native types (Go structs, Swift `Codable`, Kotlin `@Serializable`,
/// Ruby hashes) just outside this module's boundary.
#[derive(Debug, Clone, uniffi::Record)]
pub struct Event {
    pub event_type: String,
    pub data_json: String,
}

/// What a [`StepHandler`] returns: zero, one, or many events to publish.
#[derive(Debug, Clone, uniffi::Enum)]
pub enum StepOutput {
    /// Step performed work but produced no event.
    None,
    /// Step produced exactly one event (the common case).
    Single { event: Event },
    /// Step fans out — produced multiple events at once.
    Multiple { events: Vec<Event> },
}

/// Final result of a workflow run.
#[derive(Debug, Clone, uniffi::Record)]
pub struct WorkflowResult {
    /// The terminal event (typically `"StopEvent"`).
    pub event: Event,
    /// Total LLM token usage across the run, if any LLM steps ran.
    pub total_input_tokens: u64,
    pub total_output_tokens: u64,
    /// Total cost in USD across the run, if pricing data was available.
    pub total_cost_usd: f64,
}

/// A human-in-the-loop response delivered to a workflow that auto-parked on an
/// `InputRequestEvent` (event type `"blazen::InputRequestEvent"`).
///
/// `request_id` must match the `request_id` carried by the
/// `InputRequestEvent` the workflow emitted; `response_json` is the JSON-
/// encoded answer handed back to the waiting step.
#[derive(Debug, Clone, uniffi::Record)]
pub struct InputResponse {
    /// Matches the `request_id` of the `InputRequestEvent` being answered.
    pub request_id: String,
    /// JSON-encoded answer delivered to the parked step.
    pub response_json: String,
}

// ---------------------------------------------------------------------------
// Foreign-implementable step handler trait
// ---------------------------------------------------------------------------

/// Step handler implemented in foreign code (Go / Swift / Kotlin / Ruby).
///
/// The Rust workflow engine calls `invoke` whenever an event matching the
/// step's `accepts` list arrives, and routes the returned [`StepOutput`]
/// back into the event queue.
///
/// ## Async story
///
/// `invoke` is `async` on the Rust side. UniFFI exposes this as:
/// - Go: blocking function, safe to call from goroutines (composes with channels)
/// - Swift: `async throws` method
/// - Kotlin: `suspend fun` method
/// - Ruby: blocking method (wrap in `Async { ... }` block for fiber concurrency)
#[uniffi::export(with_foreign)]
#[async_trait::async_trait]
pub trait StepHandler: Send + Sync {
    async fn invoke(&self, event: Event) -> BlazenResult<StepOutput>;
}

// ---------------------------------------------------------------------------
// WorkflowBuilder
// ---------------------------------------------------------------------------

/// Builder for [`Workflow`]. Use [`Workflow::builder`] or
/// `WorkflowBuilder::new()` to start.
#[derive(uniffi::Object)]
pub struct WorkflowBuilder {
    inner: parking_lot::Mutex<Option<CoreWorkflowBuilder>>,
}

impl WorkflowBuilder {
    fn take_builder(&self) -> BlazenResult<CoreWorkflowBuilder> {
        self.inner.lock().take().ok_or(BlazenError::Validation {
            message: "WorkflowBuilder already consumed".into(),
        })
    }

    fn replace_builder(&self, builder: CoreWorkflowBuilder) {
        *self.inner.lock() = Some(builder);
    }
}

#[uniffi::export]
impl WorkflowBuilder {
    /// Create a new builder with the given workflow name.
    #[uniffi::constructor]
    #[must_use]
    pub fn new(name: String) -> Arc<Self> {
        Arc::new(Self {
            inner: parking_lot::Mutex::new(Some(CoreWorkflowBuilder::new(name))),
        })
    }

    /// Register a step.
    ///
    /// - `name`: step identifier, must be unique within the workflow.
    /// - `accepts`: event-type names this step should be invoked for
    ///   (e.g. `["StartEvent"]`).
    /// - `emits`: event-type names this step is expected to produce. Used for
    ///   workflow validation and routing; provide every type the handler can
    ///   return.
    /// - `handler`: the foreign-implemented step handler.
    pub fn step(
        self: Arc<Self>,
        name: String,
        accepts: Vec<String>,
        emits: Vec<String>,
        handler: Arc<dyn StepHandler>,
    ) -> BlazenResult<Arc<Self>> {
        let builder = self.take_builder()?;
        let registration = build_step_registration(name, accepts, emits, handler);
        let next = builder.step(registration);
        self.replace_builder(next);
        Ok(self)
    }

    /// Per-step timeout in milliseconds. Steps that exceed this are aborted.
    pub fn step_timeout_ms(self: Arc<Self>, millis: u64) -> BlazenResult<Arc<Self>> {
        let builder = self.take_builder()?;
        let next = builder.step_timeout(std::time::Duration::from_millis(millis));
        self.replace_builder(next);
        Ok(self)
    }

    /// Workflow-wide timeout in milliseconds. Whole run aborts after this.
    pub fn timeout_ms(self: Arc<Self>, millis: u64) -> BlazenResult<Arc<Self>> {
        let builder = self.take_builder()?;
        let next = builder.timeout(std::time::Duration::from_millis(millis));
        self.replace_builder(next);
        Ok(self)
    }

    /// Consume the builder and produce a [`Workflow`] ready to run.
    pub fn build(self: Arc<Self>) -> BlazenResult<Arc<Workflow>> {
        let builder = self.take_builder()?;
        let workflow = builder.build().map_err(BlazenError::from)?;
        Ok(Arc::new(Workflow {
            inner: Arc::new(workflow),
        }))
    }
}

// ---------------------------------------------------------------------------
// Workflow + handler
// ---------------------------------------------------------------------------

/// A built workflow ready to run.
#[derive(uniffi::Object)]
pub struct Workflow {
    inner: Arc<CoreWorkflow>,
}

#[uniffi::export(async_runtime = "tokio")]
impl Workflow {
    /// Run the workflow to completion with the given JSON input as the
    /// `StartEvent` payload. Blocks (in Go) / suspends (in Swift/Kotlin)
    /// until the workflow emits its `StopEvent` (or fails).
    ///
    /// This is the result-only shorthand. For streaming intermediate events,
    /// pausing, snapshotting, or human-in-the-loop input, use
    /// [`run_with_handler`](Self::run_with_handler) instead.
    pub async fn run(self: Arc<Self>, input_json: String) -> BlazenResult<WorkflowResult> {
        let input: serde_json::Value = serde_json::from_str(&input_json)?;
        let handler = self.inner.run(input).await.map_err(BlazenError::from)?;
        core_handler_to_wire_result(handler).await
    }

    /// Run the workflow and return a live [`WorkflowHandler`] instead of
    /// blocking for the final result.
    ///
    /// The returned handler exposes the full control surface — stream
    /// intermediate events to a foreign [`WorkflowEventSink`], `pause` /
    /// `resume_in_place`, `snapshot`, `respond_to_input` for human-in-the-
    /// loop, `abort`, and running `usage_total` / `cost_total_usd` — plus
    /// `result()` to await the terminal event. This mirrors the
    /// `run_with_handler` surface in the Python / Node / WASM bindings.
    pub async fn run_with_handler(
        self: Arc<Self>,
        input_json: String,
    ) -> BlazenResult<Arc<WorkflowHandler>> {
        let input: serde_json::Value = serde_json::from_str(&input_json)?;
        let handler = self.inner.run(input).await.map_err(BlazenError::from)?;
        Ok(WorkflowHandler::new(handler))
    }

    /// Names of all registered steps, in registration order.
    #[must_use]
    pub fn step_names(self: Arc<Self>) -> Vec<String> {
        self.inner.step_names()
    }
}

#[uniffi::export]
impl Workflow {
    /// Synchronous variant of [`run`] — blocks the current thread on the
    /// shared Tokio runtime. Provided for callers that want fire-and-forget
    /// usage without the host language's async machinery (handy for Ruby
    /// scripts and quick Go main fns). Prefer the async [`run`] in long-
    /// running services.
    pub fn run_blocking(self: Arc<Self>, input_json: String) -> BlazenResult<WorkflowResult> {
        let this = Arc::clone(&self);
        runtime().block_on(async move { this.run(input_json).await })
    }

    /// Synchronous variant of [`run_with_handler`](Self::run_with_handler) —
    /// blocks the current thread on the shared Tokio runtime while the
    /// workflow is launched, then returns the live handler. The workflow
    /// keeps running on the shared runtime after this returns.
    pub fn run_with_handler_blocking(
        self: Arc<Self>,
        input_json: String,
    ) -> BlazenResult<Arc<WorkflowHandler>> {
        let this = Arc::clone(&self);
        runtime().block_on(async move { this.run_with_handler(input_json).await })
    }
}

/// Await a core [`CoreWorkflowHandler`] to completion and convert it into the
/// wire-format [`WorkflowResult`]. Shared by [`Workflow::run`] and
/// [`WorkflowHandler::result`] so both produce an identical result object.
async fn core_handler_to_wire_result(handler: CoreWorkflowHandler) -> BlazenResult<WorkflowResult> {
    let result = handler.result().await.map_err(BlazenError::from)?;
    Ok(WorkflowResult {
        event: any_event_to_wire(&*result.event),
        total_input_tokens: u64::from(result.usage_total.prompt_tokens),
        total_output_tokens: u64::from(result.usage_total.completion_tokens),
        total_cost_usd: result.cost_total_usd,
    })
}

// ---------------------------------------------------------------------------
// Streaming sink + live handler
// ---------------------------------------------------------------------------

/// Foreign-implementable sink for intermediate workflow events.
///
/// UniFFI's async-iterator support across Go, Swift, Kotlin, and Ruby is
/// uneven, so streaming uses a *foreign-callable sink trait* (mirroring
/// [`CompletionStreamSink`](crate::streaming::CompletionStreamSink)) rather
/// than an async iterator. Each foreign-language idiomatic wrapper adapts the
/// callbacks into its host streaming type:
///
/// - Go: `on_event` pushes to a `chan Event`
/// - Swift: callbacks build an `AsyncStream<Event>`
/// - Kotlin: callbacks emit into a `Flow<Event>`
/// - Ruby: callbacks yield to an `Enumerator::Lazy`
///
/// Steps publish to the stream via
/// `ctx.write_event_to_stream(...)`. The pump invokes [`on_event`](Self::on_event)
/// for each event in order, then exactly one [`on_close`](Self::on_close) when
/// the workflow completes (the internal `"blazen::StreamEnd"` sentinel is
/// consumed and never forwarded).
#[uniffi::export(with_foreign)]
pub trait WorkflowEventSink: Send + Sync {
    /// One intermediate event arrived from a step.
    fn on_event(&self, event: Event);
    /// The stream ended — the workflow reached a terminal state (or the
    /// subscription was cancelled). Fires exactly once.
    fn on_close(&self);
}

/// A live handle to a running workflow.
///
/// Returned by [`Workflow::run_with_handler`]. Provides:
///
/// **Consumption (consumes the handler):**
/// - [`result`](Self::result) — await the final [`WorkflowResult`].
///
/// **Streaming (borrows the handler):**
/// - [`stream_events`](Self::stream_events) — pump intermediate events to a
///   foreign [`WorkflowEventSink`]. Returns immediately; the pump runs on the
///   shared Tokio runtime until the workflow completes.
///
/// **Control (borrows the handler, may be called repeatedly):**
/// - [`pause`](Self::pause) / [`resume_in_place`](Self::resume_in_place)
/// - [`snapshot`](Self::snapshot) — capture resumable state as a JSON string
/// - [`respond_to_input`](Self::respond_to_input) — human-in-the-loop
/// - [`abort`](Self::abort)
/// - [`usage_total`](Self::usage_total) / [`cost_total_usd`](Self::cost_total_usd)
#[derive(uniffi::Object)]
pub struct WorkflowHandler {
    /// `Option` because [`result`](Self::result) consumes the inner handler.
    /// `AsyncMutex` so the control methods can borrow it across `.await`.
    inner: Arc<AsyncMutex<Option<CoreWorkflowHandler>>>,
    /// Pre-subscribed initial stream, taken at construction so events
    /// published before [`stream_events`](Self::stream_events) is called are
    /// not lost. Consumed on the first `stream_events` call.
    initial_stream: parking_lot::Mutex<Option<PinnedEventStream>>,
}

type PinnedEventStream =
    std::pin::Pin<Box<dyn tokio_stream::Stream<Item = Box<dyn AnyEvent>> + Send + Unpin>>;

impl WorkflowHandler {
    /// Wrap a fresh core [`CoreWorkflowHandler`], capturing its pre-subscribed
    /// initial stream so the first `stream_events` subscriber sees every event
    /// from the very first step.
    fn new(mut handler: CoreWorkflowHandler) -> Arc<Self> {
        let initial_stream: Option<PinnedEventStream> = handler
            .take_initial_stream()
            .map(|s| Box::pin(s) as PinnedEventStream);
        Arc::new(Self {
            inner: Arc::new(AsyncMutex::new(Some(handler))),
            initial_stream: parking_lot::Mutex::new(initial_stream),
        })
    }

    /// Borrow the inner handler for a control operation, erroring if it was
    /// already consumed by [`result`](Self::result).
    async fn with_handler<T>(
        &self,
        f: impl FnOnce(&CoreWorkflowHandler) -> Result<T, BlazenError>,
    ) -> BlazenResult<T> {
        let guard = self.inner.lock().await;
        let handler = guard.as_ref().ok_or(BlazenError::Validation {
            message: "WorkflowHandler already consumed by result()".into(),
        })?;
        f(handler)
    }
}

#[uniffi::export(async_runtime = "tokio")]
impl WorkflowHandler {
    /// Await the final workflow result, consuming the handler.
    ///
    /// # Errors
    /// [`BlazenError::Validation`] if the handler was already consumed;
    /// [`BlazenError::Workflow`] if the run failed.
    pub async fn result(self: Arc<Self>) -> BlazenResult<WorkflowResult> {
        let handler = {
            let mut guard = self.inner.lock().await;
            guard.take().ok_or(BlazenError::Validation {
                message: "WorkflowHandler already consumed by result()".into(),
            })?
        };
        core_handler_to_wire_result(handler).await
    }

    /// Pump intermediate events to `sink` until the workflow completes.
    ///
    /// Returns immediately; the pump runs on the shared Tokio runtime. The
    /// first call consumes the pre-subscribed initial stream (so no events are
    /// lost between `run_with_handler` and this call); subsequent calls
    /// subscribe from the current point in time. `sink.on_close()` fires
    /// exactly once when the stream ends.
    ///
    /// # Errors
    /// [`BlazenError::Validation`] if the handler was already consumed.
    pub async fn stream_events(
        self: Arc<Self>,
        sink: Arc<dyn WorkflowEventSink>,
    ) -> BlazenResult<()> {
        // Prefer the pre-subscribed initial stream on the first call so the
        // very first step's events are not raced; fall back to a fresh
        // subscription for later subscribers. Take from the (sync) parking_lot
        // mutex and drop its guard BEFORE any `.await` so the returned future
        // stays `Send`.
        let initial = self.initial_stream.lock().take();
        let stream: PinnedEventStream = match initial {
            Some(s) => s,
            None => {
                let guard = self.inner.lock().await;
                let handler = guard.as_ref().ok_or(BlazenError::Validation {
                    message: "WorkflowHandler already consumed by result()".into(),
                })?;
                Box::pin(handler.stream_events()) as PinnedEventStream
            }
        };
        tokio::spawn(async move {
            let mut stream = stream;
            while let Some(event) = stream.next().await {
                // The event loop sends a sentinel to mark stream end; consume
                // it and stop rather than forwarding it to foreign code.
                if event.event_type_id() == "blazen::StreamEnd" {
                    break;
                }
                sink.on_event(any_event_to_wire(&*event));
            }
            sink.on_close();
        });
        Ok(())
    }

    /// Capture a resumable [`crate::persist::WorkflowCheckpoint`]-compatible
    /// snapshot of the current workflow state, encoded as a JSON string.
    ///
    /// For a quiescent snapshot (no in-flight steps), call [`pause`](Self::pause)
    /// first, then `snapshot()`, then optionally [`resume_in_place`](Self::resume_in_place)
    /// or [`abort`](Self::abort).
    ///
    /// # Errors
    /// [`BlazenError::Validation`] if the handler was already consumed;
    /// [`BlazenError::Workflow`] if the event loop has already exited.
    pub async fn snapshot(self: Arc<Self>) -> BlazenResult<String> {
        let guard = self.inner.lock().await;
        let handler = guard.as_ref().ok_or(BlazenError::Validation {
            message: "WorkflowHandler already consumed by result()".into(),
        })?;
        let snap = handler.snapshot().await.map_err(BlazenError::from)?;
        snap.to_json().map_err(BlazenError::from)
    }

    /// Snapshot the running aggregate [`TokenUsage`] for this run. Safe to
    /// call at any point; matches `WorkflowResult` totals once `result()`
    /// completes.
    ///
    /// # Errors
    /// [`BlazenError::Validation`] if the handler was already consumed.
    pub async fn usage_total(self: Arc<Self>) -> BlazenResult<TokenUsage> {
        let guard = self.inner.lock().await;
        let handler = guard.as_ref().ok_or(BlazenError::Validation {
            message: "WorkflowHandler already consumed by result()".into(),
        })?;
        Ok(TokenUsage::from(handler.usage_total().await))
    }

    /// Snapshot the running aggregate cost in USD for this run.
    ///
    /// # Errors
    /// [`BlazenError::Validation`] if the handler was already consumed.
    pub async fn cost_total_usd(self: Arc<Self>) -> BlazenResult<f64> {
        let guard = self.inner.lock().await;
        let handler = guard.as_ref().ok_or(BlazenError::Validation {
            message: "WorkflowHandler already consumed by result()".into(),
        })?;
        Ok(handler.cost_total_usd().await)
    }

    /// Park the event loop after the current step. The loop stays alive and
    /// responsive to `resume_in_place`, `snapshot`, `respond_to_input`, and
    /// `abort`.
    ///
    /// # Errors
    /// [`BlazenError::Validation`] if the handler was already consumed;
    /// [`BlazenError::Workflow`] if the event loop has already exited.
    pub async fn pause(self: Arc<Self>) -> BlazenResult<()> {
        self.with_handler(|h| h.pause().map_err(BlazenError::from))
            .await
    }

    /// Resume a parked event loop.
    ///
    /// # Errors
    /// [`BlazenError::Validation`] if the handler was already consumed;
    /// [`BlazenError::Workflow`] if the event loop has already exited.
    pub async fn resume_in_place(self: Arc<Self>) -> BlazenResult<()> {
        self.with_handler(|h| h.resume_in_place().map_err(BlazenError::from))
            .await
    }

    /// Deliver a human-in-the-loop response to a workflow that auto-parked on
    /// an `InputRequestEvent`. The loop unparks and routes the response.
    ///
    /// # Errors
    /// [`BlazenError::Validation`] if the handler was already consumed or
    /// `response.response_json` is not valid JSON; [`BlazenError::Workflow`]
    /// if the event loop has already exited.
    pub async fn respond_to_input(self: Arc<Self>, response: InputResponse) -> BlazenResult<()> {
        let parsed: serde_json::Value = serde_json::from_str(&response.response_json)?;
        let core_response = blazen_events::InputResponseEvent {
            request_id: response.request_id,
            response: parsed,
        };
        self.with_handler(move |h| h.respond_to_input(core_response).map_err(BlazenError::from))
            .await
    }

    /// Tear down the event loop. Any pending `result()` resolves with a
    /// workflow error.
    ///
    /// # Errors
    /// [`BlazenError::Validation`] if the handler was already consumed;
    /// [`BlazenError::Workflow`] if the event loop has already exited.
    pub async fn abort(self: Arc<Self>) -> BlazenResult<()> {
        self.with_handler(|h| h.abort().map_err(BlazenError::from))
            .await
    }
}

// ---------------------------------------------------------------------------
// Internals: bridge a foreign StepHandler into a blazen-core StepRegistration.
// ---------------------------------------------------------------------------

/// Build a [`StepRegistration`] whose handler closure dispatches into a
/// foreign-language [`StepHandler`] trait object.
fn build_step_registration(
    name: String,
    accepts: Vec<String>,
    emits: Vec<String>,
    handler: Arc<dyn StepHandler>,
) -> StepRegistration {
    let accepts_static: Vec<&'static str> = accepts
        .iter()
        .map(|s| intern_event_type(s.as_str()))
        .collect();
    let emits_static: Vec<&'static str> = emits
        .iter()
        .map(|s| intern_event_type(s.as_str()))
        .collect();

    let step_name = name.clone();
    let handler_arc = Arc::clone(&handler);

    let step_fn: blazen_core::StepFn =
        Arc::new(move |event: Box<dyn AnyEvent>, _ctx: CoreContext| {
            let handler = Arc::clone(&handler_arc);
            let step_name = step_name.clone();
            Box::pin(async move {
                let wire = any_event_to_wire(&*event);
                let out = handler.invoke(wire).await.map_err(|e| {
                    blazen_core::WorkflowError::StepFailed {
                        step_name: step_name.clone(),
                        source: Box::new(std::io::Error::other(e.to_string())),
                    }
                })?;
                Ok(wire_step_output_to_core(out))
            })
        });

    StepRegistration::new(name, accepts_static, emits_static, step_fn, 0)
}

/// Convert a `blazen-core` `AnyEvent` (dyn trait object) into the wire-format
/// [`Event`] crossed across the FFI. Built-in event types (`StartEvent`,
/// `StopEvent`, `InputRequestEvent`, etc.) are recognised by downcast;
/// everything else is serialised via the generic `AnyEvent::to_json` path.
fn any_event_to_wire(event: &dyn AnyEvent) -> Event {
    let event_type = event.event_type_id().to_string();
    let data_json = event.to_json().to_string();
    Event {
        event_type,
        data_json,
    }
}

/// Convert the wire-format [`StepOutput`] to `blazen-core`'s native form.
fn wire_step_output_to_core(out: StepOutput) -> CoreStepOutput {
    match out {
        StepOutput::None => CoreStepOutput::None,
        StepOutput::Single { event } => CoreStepOutput::Single(wire_event_to_any(event)),
        StepOutput::Multiple { events } => {
            CoreStepOutput::Multiple(events.into_iter().map(wire_event_to_any).collect())
        }
    }
}

/// Convert a wire-format [`Event`] to a `Box<dyn AnyEvent>` suitable for the
/// blazen event queue. Recognises a small set of built-in event types so they
/// round-trip with their native struct identities; unknown types are wrapped
/// in a [`DynamicEvent`] carrying the JSON payload.
fn wire_event_to_any(ev: Event) -> Box<dyn AnyEvent> {
    let parsed: serde_json::Value = serde_json::from_str(&ev.data_json)
        .unwrap_or(serde_json::Value::Object(serde_json::Map::new()));
    match ev.event_type.as_str() {
        "StartEvent" => Box::new(StartEvent { data: parsed }),
        "StopEvent" => Box::new(StopEvent { result: parsed }),
        other => Box::new(DynamicEvent::from_json(other.to_string(), parsed)),
    }
}
