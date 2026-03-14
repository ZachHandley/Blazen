//! Pipeline shared state.
//!
//! [`PipelineState`] carries data between stages -- the pipeline's shared
//! key/value store, per-stage results, and the original input. Each stage
//! can read previous results via [`PipelineState::last_result`] or
//! [`PipelineState::stage_result`], and can store arbitrary data via
//! [`PipelineState::set`] / [`PipelineState::get`].

use std::collections::HashMap;

use indexmap::IndexMap;
use serde_json::Value;

/// Mutable state that flows through the pipeline.
///
/// Stages can read from and write to the shared key/value store. Each
/// completed stage's output is also recorded in insertion order so later
/// stages can reference earlier results.
#[derive(Debug, Clone)]
pub struct PipelineState {
    /// Arbitrary shared key/value data that persists across stages.
    shared: HashMap<String, Value>,
    /// Per-stage output values, in the order stages completed.
    stage_results: IndexMap<String, Value>,
    /// The original input that was passed to `Pipeline::run`.
    input: Value,
}

impl PipelineState {
    /// Create a new empty pipeline state with the given input.
    pub(crate) fn new(input: Value) -> Self {
        Self {
            shared: HashMap::new(),
            stage_results: IndexMap::new(),
            input,
        }
    }

    /// Restore pipeline state from a snapshot's shared state and completed
    /// stage results.
    #[allow(dead_code)]
    pub(crate) fn restore(
        input: Value,
        shared: HashMap<String, Value>,
        stage_results: IndexMap<String, Value>,
    ) -> Self {
        Self {
            shared,
            stage_results,
            input,
        }
    }

    /// Get a value from the shared key/value store.
    #[must_use]
    pub fn get(&self, key: &str) -> Option<&Value> {
        self.shared.get(key)
    }

    /// Set a value in the shared key/value store.
    pub fn set(&mut self, key: impl Into<String>, value: Value) {
        self.shared.insert(key.into(), value);
    }

    /// Get the original pipeline input.
    #[must_use]
    pub fn input(&self) -> &Value {
        &self.input
    }

    /// Get the output of a specific completed stage by name.
    #[must_use]
    pub fn stage_result(&self, name: &str) -> Option<&Value> {
        self.stage_results.get(name)
    }

    /// Get the output of the most recently completed stage, or the pipeline
    /// input if no stages have completed yet.
    #[must_use]
    pub fn last_result(&self) -> &Value {
        self.stage_results
            .values()
            .next_back()
            .unwrap_or(&self.input)
    }

    /// Record a stage's output.
    pub(crate) fn record_stage_result(&mut self, name: impl Into<String>, value: Value) {
        self.stage_results.insert(name.into(), value);
    }

    /// Returns a clone of the shared state map (for snapshotting).
    #[must_use]
    pub(crate) fn shared_clone(&self) -> HashMap<String, Value> {
        self.shared.clone()
    }

    /// Returns a clone of the stage results map (for snapshotting).
    #[must_use]
    #[allow(dead_code)]
    pub(crate) fn stage_results_clone(&self) -> IndexMap<String, Value> {
        self.stage_results.clone()
    }
}
