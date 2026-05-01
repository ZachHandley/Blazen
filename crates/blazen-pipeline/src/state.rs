//! Pipeline shared state.
//!
//! [`PipelineState`] carries data between stages -- the pipeline's shared
//! key/value store, per-stage results, and the original input. Each stage
//! can read previous results via [`PipelineState::last_result`] or
//! [`PipelineState::stage_result`], and can store arbitrary data via
//! [`PipelineState::set`] / [`PipelineState::get`] (when `S = serde_json::Value`).
//!
//! `PipelineState` is generic over `S`, the typed shared-state container.
//! When `S = serde_json::Value` (the default), the JSON-only `get`/`set`
//! accessors are available. Custom `S` types can opt into snapshot
//! persistence by implementing `Serialize` and `DeserializeOwned`.
//!
//! In addition to the typed `shared` field, `PipelineState` carries an
//! `objects` bag for opaque, non-`Serialize` resources (DB connections,
//! file handles, etc.). The `objects` bag mirrors `blazen_core::Context.objects`
//! and is always excluded from snapshots; entries do not survive a
//! snapshot/restore cycle.

use std::any::Any;
use std::collections::HashMap;

use indexmap::IndexMap;
use serde::Serialize;

/// Mutable state that flows through the pipeline.
///
/// Stages can read from and write to the shared typed state. Each
/// completed stage's output is also recorded in insertion order so later
/// stages can reference earlier results.
///
/// # Type parameters
///
/// * `S` -- the typed shared-state container. Defaults to
///   `serde_json::Value` for backward compatibility with existing
///   JSON-shaped pipelines. Must be `Default + Clone + Send + Sync + 'static`.
pub struct PipelineState<S = serde_json::Value>
where
    S: Default + Clone + Send + Sync + 'static,
{
    /// Typed shared state that persists across stages.
    shared: S,
    /// Snapshot-excluded typed objects bag (DB conns, file handles, ...).
    /// Mirrors `blazen_core::Context.objects` -- opaque, downcast on get.
    objects: HashMap<String, Box<dyn Any + Send + Sync>>,
    /// Per-stage output values, in the order stages completed.
    stage_results: IndexMap<String, serde_json::Value>,
    /// The original input that was passed to `Pipeline::run`.
    input: serde_json::Value,
    /// Running total of `TokenUsage` aggregated from `UsageEvent`s emitted
    /// during pipeline execution.
    pub(crate) usage_total: blazen_llm::types::TokenUsage,
    /// Running total cost in USD aggregated from `UsageEvent.cost_usd`
    /// values when present.
    pub(crate) cost_total_usd: f64,
}

impl<S> Default for PipelineState<S>
where
    S: Default + Clone + Send + Sync + 'static,
{
    fn default() -> Self {
        Self {
            shared: S::default(),
            objects: HashMap::new(),
            stage_results: IndexMap::new(),
            input: serde_json::Value::Null,
            usage_total: blazen_llm::types::TokenUsage::default(),
            cost_total_usd: 0.0,
        }
    }
}

impl<S> PipelineState<S>
where
    S: Default + Clone + Send + Sync + 'static,
{
    /// Create a new state with the given input JSON. `shared` is `S::default()`.
    pub(crate) fn new(input: serde_json::Value) -> Self {
        Self {
            input,
            ..Self::default()
        }
    }

    /// Restore from snapshot. Used by `Pipeline::resume`. Bounded on
    /// `S: DeserializeOwned` so we can decode the typed shared state from
    /// the persisted JSON. The `objects` bag is always reconstructed empty
    /// because in-process resources cannot survive a snapshot/restore cycle.
    pub(crate) fn restore(
        shared_json: serde_json::Value,
        stage_results: IndexMap<String, serde_json::Value>,
        input: serde_json::Value,
    ) -> Result<Self, serde_json::Error>
    where
        S: serde::de::DeserializeOwned,
    {
        let shared: S = serde_json::from_value(shared_json)?;
        Ok(Self {
            shared,
            objects: HashMap::new(),
            stage_results,
            input,
            usage_total: blazen_llm::types::TokenUsage::default(),
            cost_total_usd: 0.0,
        })
    }

    /// Typed reference to the shared state.
    #[must_use]
    pub fn shared(&self) -> &S {
        &self.shared
    }

    /// Mutable typed reference to the shared state.
    pub fn shared_mut(&mut self) -> &mut S {
        &mut self.shared
    }

    /// Insert an opaque object. Snapshot-excluded.
    pub fn put_object<T: Send + Sync + 'static>(&mut self, key: impl Into<String>, value: T) {
        self.objects.insert(key.into(), Box::new(value));
    }

    /// Get an opaque object by key, downcast to `T`.
    #[must_use]
    pub fn get_object<T: Send + Sync + 'static>(&self, key: &str) -> Option<&T> {
        self.objects.get(key).and_then(|b| b.downcast_ref::<T>())
    }

    /// Remove and return an opaque object by key, downcast to `T`.
    pub fn remove_object<T: Send + Sync + 'static>(&mut self, key: &str) -> Option<T> {
        let any = self.objects.remove(key)?;
        any.downcast::<T>().ok().map(|b| *b)
    }

    /// Get the original pipeline input.
    #[must_use]
    pub fn input(&self) -> &serde_json::Value {
        &self.input
    }

    /// Get the output of a specific completed stage by name.
    #[must_use]
    pub fn stage_result(&self, stage_name: &str) -> Option<&serde_json::Value> {
        self.stage_results.get(stage_name)
    }

    /// Get the output of the most recently completed stage, or the pipeline
    /// input if no stages have completed yet.
    #[must_use]
    pub fn last_result(&self) -> &serde_json::Value {
        self.stage_results
            .values()
            .next_back()
            .unwrap_or(&self.input)
    }

    /// Record a stage's output.
    pub(crate) fn record_stage_result(&mut self, name: String, value: serde_json::Value) {
        self.stage_results.insert(name, value);
    }

    /// Aggregated token usage across the pipeline run so far.
    #[must_use]
    pub fn usage_total(&self) -> &blazen_llm::types::TokenUsage {
        &self.usage_total
    }

    /// Aggregated cost in USD across the pipeline run so far.
    #[must_use]
    pub fn cost_total_usd(&self) -> f64 {
        self.cost_total_usd
    }

    /// Record a single usage event into the running totals.
    ///
    /// Adds the seven token counters from the [`blazen_events::UsageEvent`]
    /// into [`Self::usage_total`] using saturating arithmetic, and
    /// accumulates `event.cost_usd` (when present) into
    /// [`Self::cost_total_usd`].
    pub fn record_usage(&mut self, event: &blazen_events::UsageEvent) {
        let tu = blazen_llm::types::TokenUsage {
            prompt_tokens: event.prompt_tokens,
            completion_tokens: event.completion_tokens,
            total_tokens: event.total_tokens,
            reasoning_tokens: event.reasoning_tokens,
            cached_input_tokens: event.cached_input_tokens,
            audio_input_tokens: event.audio_input_tokens,
            audio_output_tokens: event.audio_output_tokens,
        };
        self.usage_total.add(&tu);
        if let Some(cost) = event.cost_usd {
            self.cost_total_usd += cost;
        }
    }
}

/// JSON-specific helpers available only when `S = serde_json::Value`.
impl PipelineState<serde_json::Value> {
    /// Get a key from the JSON-backed shared state.
    #[must_use]
    pub fn get(&self, key: &str) -> Option<&serde_json::Value> {
        self.shared.as_object().and_then(|o| o.get(key))
    }

    /// Set a key in the JSON-backed shared state.
    pub fn set(&mut self, key: impl Into<String>, value: serde_json::Value) {
        if !self.shared.is_object() {
            self.shared = serde_json::json!({});
        }
        if let Some(map) = self.shared.as_object_mut() {
            map.insert(key.into(), value);
        }
    }
}

/// Snapshot helpers available when `S: Serialize`.
impl<S> PipelineState<S>
where
    S: Default + Clone + Send + Sync + 'static + Serialize,
{
    /// Serialize the typed shared state to JSON for snapshot persistence.
    pub(crate) fn shared_to_json(&self) -> Result<serde_json::Value, serde_json::Error> {
        serde_json::to_value(&self.shared)
    }
}
