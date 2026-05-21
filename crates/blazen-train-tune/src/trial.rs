//! Trial record + lifecycle state machine.
//!
//! A `Trial` is the unit of work the runner produces and the searcher
//! consumes. Every mutation (proposal, completion, failure) is appended to
//! the journal so the run can be replayed after a crash.

use std::collections::HashMap;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use serde_json::Value as JsonValue;

/// Stable identifier for a trial. Monotonic, assigned by the runner; not
/// re-issued on replay.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, PartialOrd, Ord)]
#[serde(transparent)]
pub struct TrialId(pub u64);

impl std::fmt::Display for TrialId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "trial-{:06}", self.0)
    }
}

/// Lifecycle of a single trial.
///
/// `Running` → `Completed` is the happy path; `Failed(msg)` covers
/// evaluator errors; `Pruned` is reserved for early-stopping policies that
/// abort a trial mid-flight (not used by the v1 runner — recorded
/// faithfully if a searcher / outside system writes one).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", content = "detail")]
pub enum TrialStatus {
    /// Proposed by the searcher; evaluation has begun (or is queued).
    Running,
    /// Evaluator returned a finite metric.
    Completed,
    /// Evaluator returned an error. The string is the formatted error.
    Failed(String),
    /// Trial was terminated early (e.g. ASHA, median-stopping).
    Pruned,
}

impl TrialStatus {
    /// `true` once the trial will never mutate again.
    #[must_use]
    pub fn is_terminal(&self) -> bool {
        matches!(self, Self::Completed | Self::Failed(_) | Self::Pruned)
    }
}

/// A single hyperparameter trial.
///
/// `metric` is `None` until the evaluator reports back; lower-is-better by
/// convention (matches the `eval_loss` minimization story in `blazen-train`).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Trial {
    pub id: TrialId,
    pub config: HashMap<String, JsonValue>,
    pub metric: Option<f64>,
    pub status: TrialStatus,
    pub started_at: DateTime<Utc>,
    pub finished_at: Option<DateTime<Utc>>,
}

impl Trial {
    /// Build a fresh `Running` trial with the current UTC timestamp.
    #[must_use]
    pub fn new(id: TrialId, config: HashMap<String, JsonValue>) -> Self {
        Self {
            id,
            config,
            metric: None,
            status: TrialStatus::Running,
            started_at: Utc::now(),
            finished_at: None,
        }
    }

    /// Mark this trial completed with `metric` (lower is better).
    pub fn complete(&mut self, metric: f64) {
        self.metric = Some(metric);
        self.status = TrialStatus::Completed;
        self.finished_at = Some(Utc::now());
    }

    /// Mark this trial failed; metric stays `None`.
    pub fn fail(&mut self, msg: impl Into<String>) {
        self.status = TrialStatus::Failed(msg.into());
        self.finished_at = Some(Utc::now());
    }

    /// Mark this trial pruned (early-stopped).
    pub fn prune(&mut self) {
        self.status = TrialStatus::Pruned;
        self.finished_at = Some(Utc::now());
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn lifecycle_completion_sets_metric_and_terminal() {
        let mut t = Trial::new(
            TrialId(7),
            HashMap::from([("lr".to_string(), json!(0.001))]),
        );
        assert_eq!(t.metric, None);
        assert!(matches!(t.status, TrialStatus::Running));
        t.complete(0.42);
        assert_eq!(t.metric, Some(0.42));
        assert!(t.status.is_terminal());
        assert!(t.finished_at.is_some());
    }

    #[test]
    fn failure_path_records_message() {
        let mut t = Trial::new(TrialId(1), HashMap::new());
        t.fail("evaluator exploded");
        match &t.status {
            TrialStatus::Failed(msg) => assert_eq!(msg, "evaluator exploded"),
            other => panic!("unexpected status: {other:?}"),
        }
        assert!(t.status.is_terminal());
        assert_eq!(t.metric, None);
    }

    #[test]
    fn trial_id_display_format_is_stable() {
        assert_eq!(TrialId(0).to_string(), "trial-000000");
        assert_eq!(TrialId(42).to_string(), "trial-000042");
    }
}
