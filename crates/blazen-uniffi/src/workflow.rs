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

use crate::errors::{BlazenError, BlazenResult};
use crate::runtime::runtime;

use blazen_core::{
    Context as CoreContext, StepOutput as CoreStepOutput, StepRegistration,
    Workflow as CoreWorkflow, WorkflowBuilder as CoreWorkflowBuilder,
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
    pub async fn run(self: Arc<Self>, input_json: String) -> BlazenResult<WorkflowResult> {
        let input: serde_json::Value = serde_json::from_str(&input_json)?;
        let handler = self.inner.run(input).await.map_err(BlazenError::from)?;
        let usage = handler.usage_total().await;
        let cost = handler.cost_total_usd().await;
        let result = handler.result().await.map_err(BlazenError::from)?;
        Ok(WorkflowResult {
            event: any_event_to_wire(&*result.event),
            total_input_tokens: u64::from(usage.prompt_tokens),
            total_output_tokens: u64::from(usage.completion_tokens),
            total_cost_usd: cost,
        })
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
        other => Box::new(DynamicEvent {
            event_type: other.to_string(),
            data: parsed,
        }),
    }
}
