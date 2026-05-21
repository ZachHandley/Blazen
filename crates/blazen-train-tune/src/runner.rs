//! The async search loop.
//!
//! Wires a `SearchSpace` + `Searcher` + `Evaluator` + `TrialJournal` and
//! drives them until a budget (max trials or wall-clock seconds) is met.
//!
//! Concurrency: if `parallel > 1`, up to `parallel` trials run at once
//! inside a `tokio::task::JoinSet`. The searcher's `ask`/`tell` calls
//! remain serial — only the evaluator runs in parallel — so each in-flight
//! trial sees the same `history` snapshot for proposal purposes. Searchers
//! that depend strictly on every prior result being `tell`ed before the
//! next `ask` should use `parallel = 1`.

use std::{
    collections::HashMap,
    future::Future,
    pin::Pin,
    sync::Arc,
    time::{Duration, Instant},
};

use serde_json::Value as JsonValue;
use tokio::task::JoinSet;

use crate::{
    error::TuneError,
    journal::TrialJournal,
    searcher::Searcher,
    space::SearchSpace,
    trial::{Trial, TrialId},
};

/// Closed-form objective function. Lower is better.
///
/// Implementations typically wrap `blazen_train::Trainer::run()` and return
/// the final `eval_loss`. The closure may be synchronous or `async`.
pub type EvalFuture = Pin<Box<dyn Future<Output = Result<f64, String>> + Send>>;

/// Evaluator trait. Implementors take a sampled config and return a metric.
///
/// The trait is the only coupling point between this crate and any actual
/// training backend; the runner doesn't know what's being optimized.
pub trait Evaluator: Send + Sync {
    /// Evaluate `config` and return the metric (lower-is-better). On error,
    /// the runner records the trial as `Failed` and continues.
    fn evaluate(&self, config: HashMap<String, JsonValue>) -> EvalFuture;
}

/// Convenience blanket impl for closures.
impl<F> Evaluator for F
where
    F: Fn(HashMap<String, JsonValue>) -> EvalFuture + Send + Sync,
{
    fn evaluate(&self, config: HashMap<String, JsonValue>) -> EvalFuture {
        self(config)
    }
}

/// Stop condition for the search loop.
#[derive(Debug, Clone, Copy)]
pub enum RunnerBudget {
    /// Run exactly this many trials, then stop.
    MaxTrials(usize),
    /// Run until elapsed wall-clock exceeds this duration.
    TimeBudget(Duration),
    /// Stop at the first of either bound.
    Either { max_trials: usize, time: Duration },
}

impl RunnerBudget {
    fn reached(&self, trials_done: usize, start: Instant) -> bool {
        match self {
            Self::MaxTrials(n) => trials_done >= *n,
            Self::TimeBudget(d) => start.elapsed() >= *d,
            Self::Either { max_trials, time } => {
                trials_done >= *max_trials || start.elapsed() >= *time
            }
        }
    }
}

/// The search loop driver.
pub struct Runner {
    space: SearchSpace,
    searcher: Box<dyn Searcher + Send>,
    evaluator: Arc<dyn Evaluator>,
    journal: Option<TrialJournal>,
    budget: RunnerBudget,
    parallel: usize,
    next_id: u64,
}

impl Runner {
    pub fn new(
        space: SearchSpace,
        searcher: Box<dyn Searcher + Send>,
        evaluator: Arc<dyn Evaluator>,
        budget: RunnerBudget,
    ) -> Self {
        Self {
            space,
            searcher,
            evaluator,
            journal: None,
            budget,
            parallel: 1,
            next_id: 0,
        }
    }

    /// Attach a journal. If the journal file already contains records,
    /// they are loaded as initial history and `next_id` is bumped past the
    /// highest id seen.
    pub fn with_journal(mut self, journal: TrialJournal) -> Result<Self, TuneError> {
        let existing = TrialJournal::replay(journal.path())?;
        if let Some(max) = existing.iter().map(|t| t.id.0).max() {
            self.next_id = max + 1;
            // Feed history forward to the searcher so it isn't starting cold.
            for t in &existing {
                if t.status.is_terminal() {
                    self.searcher.tell(t);
                }
            }
        }
        self.journal = Some(journal);
        Ok(self)
    }

    /// Set max in-flight trials (default 1). Values >1 only help when the
    /// evaluator is expensive enough that proposal-without-feedback is OK.
    #[must_use]
    pub fn with_parallel(mut self, parallel: usize) -> Self {
        self.parallel = parallel.max(1);
        self
    }

    /// Borrow the search space (for introspection / logging).
    #[must_use]
    pub fn space(&self) -> &SearchSpace {
        &self.space
    }

    /// Run the loop to completion. Returns the recorded history in order.
    pub async fn run(&mut self) -> Result<Vec<Trial>, TuneError> {
        let start = Instant::now();
        let mut history: Vec<Trial> = self
            .journal
            .as_ref()
            .map(|j| TrialJournal::replay(j.path()))
            .transpose()?
            .unwrap_or_default();
        // Sort by id so the in-memory view matches journal order.
        history.sort_by_key(|t| t.id.0);

        let mut completed = history.iter().filter(|t| t.status.is_terminal()).count();

        if self.parallel == 1 {
            self.run_serial(&mut history, &mut completed, start).await
        } else {
            self.run_parallel(&mut history, &mut completed, start).await
        }
    }

    async fn run_serial(
        &mut self,
        history: &mut Vec<Trial>,
        completed: &mut usize,
        start: Instant,
    ) -> Result<Vec<Trial>, TuneError> {
        while !self.budget.reached(*completed, start) {
            let cfg = match self.searcher.ask(history) {
                Ok(c) => c,
                Err(TuneError::GridExhausted(n)) => {
                    tracing::info!(total = n, "grid exhausted; stopping");
                    break;
                }
                Err(e) => return Err(e),
            };
            let id = TrialId(self.next_id);
            self.next_id += 1;
            let mut trial = Trial::new(id, cfg.clone());
            self.record(&trial)?;
            let result = self.evaluator.evaluate(cfg).await;
            match result {
                Ok(metric) if metric.is_finite() => trial.complete(metric),
                Ok(metric) => trial.fail(format!("non-finite metric: {metric}")),
                Err(msg) => trial.fail(msg),
            }
            self.record(&trial)?;
            self.searcher.tell(&trial);
            history.push(trial);
            *completed += 1;
        }
        Ok(std::mem::take(history))
    }

    async fn run_parallel(
        &mut self,
        history: &mut Vec<Trial>,
        completed: &mut usize,
        start: Instant,
    ) -> Result<Vec<Trial>, TuneError> {
        type InFlightItem = (TrialId, Result<f64, String>, HashMap<String, JsonValue>);
        let mut in_flight: JoinSet<InFlightItem> = JoinSet::new();
        let mut launched_running: HashMap<u64, Trial> = HashMap::new();

        loop {
            // Top up the in-flight pool.
            while in_flight.len() < self.parallel && !self.budget.reached(*completed, start) {
                let cfg = match self.searcher.ask(history) {
                    Ok(c) => c,
                    Err(TuneError::GridExhausted(n)) => {
                        tracing::info!(total = n, "grid exhausted; draining in-flight");
                        break;
                    }
                    Err(e) => return Err(e),
                };
                let id = TrialId(self.next_id);
                self.next_id += 1;
                let trial = Trial::new(id, cfg.clone());
                self.record(&trial)?;
                launched_running.insert(id.0, trial);
                let evaluator = self.evaluator.clone();
                in_flight.spawn(async move {
                    let res = evaluator.evaluate(cfg.clone()).await;
                    (id, res, cfg)
                });
            }

            if in_flight.is_empty() {
                break;
            }

            // Drain at least one completion.
            let Some(joined) = in_flight.join_next().await else {
                break;
            };
            let (id, result, cfg) = match joined {
                Ok(tuple) => tuple,
                Err(e) => return Err(TuneError::Evaluator(format!("join error: {e}"))),
            };
            let mut trial = launched_running.remove(&id.0).unwrap_or_else(|| {
                // Defensive: rebuild a fresh Running record if we lost it.
                Trial::new(id, cfg)
            });
            match result {
                Ok(metric) if metric.is_finite() => trial.complete(metric),
                Ok(metric) => trial.fail(format!("non-finite metric: {metric}")),
                Err(msg) => trial.fail(msg),
            }
            self.record(&trial)?;
            self.searcher.tell(&trial);
            history.push(trial);
            *completed += 1;
        }
        // Keep deterministic ordering by id.
        history.sort_by_key(|t| t.id.0);
        Ok(std::mem::take(history))
    }

    fn record(&mut self, trial: &Trial) -> Result<(), TuneError> {
        if let Some(j) = self.journal.as_mut() {
            j.record(trial)?;
        }
        Ok(())
    }
}

#[cfg(test)]
#[allow(clippy::unreadable_literal, clippy::cast_sign_loss)]
mod tests {
    use super::*;
    use crate::{
        searcher::{RandomSearch, TpeSearch},
        space::Distribution,
        trial::TrialStatus,
    };

    fn unit_space() -> SearchSpace {
        let mut s = SearchSpace::new();
        s.add(
            "x",
            Distribution::Uniform {
                low: 0.0,
                high: 1.0,
            },
        )
        .unwrap();
        s
    }

    /// Evaluator that returns `(x - target)^2`.
    struct Quadratic {
        target: f64,
    }
    impl Evaluator for Quadratic {
        fn evaluate(&self, config: HashMap<String, JsonValue>) -> EvalFuture {
            let x = config["x"].as_f64().unwrap();
            let target = self.target;
            Box::pin(async move { Ok((x - target).powi(2)) })
        }
    }

    #[tokio::test]
    async fn runner_records_trials_in_order() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("j.jsonl");
        let journal = TrialJournal::open(&path).unwrap();
        let space = unit_space();
        let searcher = Box::new(RandomSearch::new(space.clone(), 0));
        let evaluator: Arc<dyn Evaluator> = Arc::new(Quadratic { target: 0.4 });
        let mut runner = Runner::new(space, searcher, evaluator, RunnerBudget::MaxTrials(8))
            .with_journal(journal)
            .unwrap();
        let history = runner.run().await.unwrap();
        assert_eq!(history.len(), 8);
        for (i, t) in history.iter().enumerate() {
            assert_eq!(t.id, TrialId(i as u64));
            assert!(matches!(t.status, TrialStatus::Completed));
            assert!(t.metric.unwrap() >= 0.0);
        }
        // Journal should hold 16 lines (8 running + 8 completed).
        let raw = std::fs::read_to_string(&path).unwrap();
        let lines: Vec<_> = raw.lines().filter(|l| !l.trim().is_empty()).collect();
        assert_eq!(lines.len(), 16);
    }

    /// Evaluator that always errors.
    struct AlwaysFails;
    impl Evaluator for AlwaysFails {
        fn evaluate(&self, _config: HashMap<String, JsonValue>) -> EvalFuture {
            Box::pin(async { Err::<f64, _>("boom".to_string()) })
        }
    }

    #[tokio::test]
    async fn runner_handles_evaluator_failure_gracefully() {
        let space = unit_space();
        let searcher = Box::new(RandomSearch::new(space.clone(), 1));
        let evaluator: Arc<dyn Evaluator> = Arc::new(AlwaysFails);
        let mut runner = Runner::new(space, searcher, evaluator, RunnerBudget::MaxTrials(3));
        let history = runner.run().await.unwrap();
        assert_eq!(history.len(), 3);
        for t in &history {
            assert!(t.metric.is_none());
            assert!(matches!(t.status, TrialStatus::Failed(_)));
        }
    }

    #[tokio::test]
    async fn runner_with_tpe_makes_progress_toward_target() {
        let space = unit_space();
        let searcher = Box::new(TpeSearch::new(space.clone(), 0xabad1dea));
        let evaluator: Arc<dyn Evaluator> = Arc::new(Quadratic { target: 0.3 });
        let mut runner = Runner::new(space, searcher, evaluator, RunnerBudget::MaxTrials(40));
        let history = runner.run().await.unwrap();
        let best = history
            .iter()
            .filter_map(|t| t.metric.map(|m| (m, t)))
            .min_by(|a, b| a.0.partial_cmp(&b.0).unwrap())
            .unwrap();
        // 40 trials at gamma=0.25 with startup=10 gives ~30 TPE-driven
        // proposals; 0.15 tolerance is comfortable.
        let best_x = best.1.config["x"].as_f64().unwrap();
        assert!(
            (best_x - 0.3).abs() < 0.15,
            "best_x={best_x} did not converge near 0.3"
        );
    }
}
