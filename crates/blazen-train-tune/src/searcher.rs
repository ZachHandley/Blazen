//! Search strategies: `RandomSearch`, `GridSearch`, `TpeSearch`.
//!
//! All three implement [`Searcher`]; the runner only calls `ask` and `tell`
//! against the trait, so callers can swap searchers without restructuring.
//!
//! ## TPE (Tree-structured Parzen Estimator)
//!
//! Reference: Bergstra, Bardenet, Bengio, Kégl (2011), *"Algorithms for
//! Hyper-Parameter Optimization"*, NeurIPS — Algorithm 1.
//!
//! For each parameter independently (a v1 factorization — multivariate
//! TPE per Falkner et al. is a follow-up):
//! 1. Split trial history at quantile γ = 0.25 by observed metric (lower
//!    is better): `good = bottom γ`, `bad = rest`.
//! 2. Fit a kernel density estimate (KDE) to each group. For continuous
//!    params we use Gaussian KDE on the parameter's natural scale
//!    (`log` for `LogUniform`); for discrete params we use additive
//!    Laplace-smoothed counts.
//! 3. Sample `n_candidates` points from the `good` KDE, score each by
//!    `l(x)/g(x)` (good density / bad density), and propose the maximizer.
//!
//! Under the warm-up budget (fewer than `n_startup_trials` completed
//! trials), TPE falls back to random sampling — there isn't enough data to
//! split into good/bad meaningfully.

use std::collections::HashMap;

use rand::{RngExt, SeedableRng, rngs::StdRng};
use serde_json::{Value as JsonValue, json};

use crate::{
    error::TuneError,
    space::{Distribution, SearchSpace},
    trial::Trial,
};

/// Search strategy contract.
///
/// `ask` proposes the next config to evaluate. `tell` feeds back a completed
/// trial (the searcher may persist its own state or just remember it for the
/// next `ask`). Implementations are not required to be `Send + Sync`; the
/// runner serializes ask/tell on a single task.
pub trait Searcher {
    /// Propose the next config. `history` is the full ordered list of every
    /// trial recorded so far (running + completed + failed + pruned).
    fn ask(&mut self, history: &[Trial]) -> Result<HashMap<String, JsonValue>, TuneError>;

    /// Notify the searcher of a trial that just reached a terminal status.
    /// Default no-op; stateful searchers (TPE) override.
    fn tell(&mut self, _trial: &Trial) {}
}

// =====================================================================
// RandomSearch
// =====================================================================

/// Independent draws from each parameter's prior. Seeded RNG for repro.
pub struct RandomSearch {
    space: SearchSpace,
    rng: StdRng,
}

impl RandomSearch {
    #[must_use]
    pub fn new(space: SearchSpace, seed: u64) -> Self {
        Self {
            space,
            rng: StdRng::seed_from_u64(seed),
        }
    }
}

impl Searcher for RandomSearch {
    fn ask(&mut self, _history: &[Trial]) -> Result<HashMap<String, JsonValue>, TuneError> {
        Ok(self.space.sample(&mut self.rng))
    }
}

// =====================================================================
// GridSearch
// =====================================================================

/// Cartesian-product enumeration. Outer-most axis is the first registered
/// parameter (mirrors how scikit-learn's `ParameterGrid` iterates).
///
/// All parameters must have finite cardinality (`Categorical`, `Discrete`,
/// or `IntUniform`). Continuous priors are rejected at construction.
pub struct GridSearch {
    axes: Vec<(String, Vec<JsonValue>)>,
    total: usize,
    cursor: usize,
}

impl GridSearch {
    /// Build the grid by enumerating every distribution in `space`.
    pub fn new(space: &SearchSpace) -> Result<Self, TuneError> {
        let mut axes = Vec::with_capacity(space.len());
        let mut total: usize = 1;
        for (name, dist) in space.params() {
            let vals = dist.enumerate()?;
            total = total
                .checked_mul(vals.len())
                .ok_or_else(|| TuneError::InvalidBounds("grid size overflows usize".into()))?;
            axes.push((name.clone(), vals));
        }
        Ok(Self {
            axes,
            total,
            cursor: 0,
        })
    }

    /// Total number of points in the grid (product of axis sizes).
    #[must_use]
    pub fn total(&self) -> usize {
        self.total
    }
}

impl Searcher for GridSearch {
    fn ask(&mut self, _history: &[Trial]) -> Result<HashMap<String, JsonValue>, TuneError> {
        if self.cursor >= self.total {
            return Err(TuneError::GridExhausted(self.total));
        }
        let mut idx = self.cursor;
        let mut out = HashMap::with_capacity(self.axes.len());
        // Right-most axis varies fastest (column-major). This matches the
        // intuition that the first-registered axis is the outer loop.
        for (name, vals) in self.axes.iter().rev() {
            let pick = idx % vals.len();
            idx /= vals.len();
            out.insert(name.clone(), vals[pick].clone());
        }
        self.cursor += 1;
        Ok(out)
    }
}

// =====================================================================
// TPE (Tree-structured Parzen Estimator)
// =====================================================================

/// TPE hyperparameters.
#[derive(Debug, Clone, Copy)]
pub struct TpeOptions {
    /// Quantile threshold splitting `good` vs `bad` history. Bergstra
    /// recommends 0.10–0.25; 0.25 is the v1 default.
    pub gamma: f64,
    /// Random-search startup count. Below this, TPE falls back to random
    /// sampling because the good/bad split isn't well-defined yet.
    pub n_startup_trials: usize,
    /// Number of candidate draws from `l(x)` per `ask`; the candidate
    /// maximizing `l/g` is returned.
    pub n_candidates: usize,
    /// Gaussian KDE bandwidth as a fraction of each parameter's range
    /// (Scott's rule is overkill for low-N regimes; a fixed fraction
    /// matches Optuna's default behavior for the warm-up phase).
    pub bandwidth_frac: f64,
    /// Additive smoothing for discrete-parameter density estimates
    /// (prevents 0/0 in the `l/g` ratio when a category is unseen).
    pub categorical_prior: f64,
}

impl Default for TpeOptions {
    fn default() -> Self {
        Self {
            gamma: 0.25,
            n_startup_trials: 10,
            n_candidates: 24,
            bandwidth_frac: 0.10,
            categorical_prior: 1.0,
        }
    }
}

pub struct TpeSearch {
    space: SearchSpace,
    opts: TpeOptions,
    rng: StdRng,
}

impl TpeSearch {
    #[must_use]
    pub fn new(space: SearchSpace, seed: u64) -> Self {
        Self {
            space,
            opts: TpeOptions::default(),
            rng: StdRng::seed_from_u64(seed),
        }
    }

    /// Override default TPE options (gamma, candidate count, etc.).
    #[must_use]
    pub fn with_options(mut self, opts: TpeOptions) -> Self {
        self.opts = opts;
        self
    }

    /// Split history into (good, bad) trials at the γ quantile by metric.
    /// Only completed trials with a finite metric participate.
    fn split_history<'a>(&self, history: &'a [Trial]) -> (Vec<&'a Trial>, Vec<&'a Trial>) {
        let mut done: Vec<&Trial> = history
            .iter()
            .filter(|t| t.metric.is_some_and(f64::is_finite))
            .collect();
        // Lower-is-better, so the smallest metrics are the "good" group.
        done.sort_by(|a, b| {
            a.metric
                .unwrap()
                .partial_cmp(&b.metric.unwrap())
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        // n_good = ceil(gamma * N), at least 1 if any trial is present so
        // the good group is never empty.
        #[allow(
            clippy::cast_precision_loss,
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss
        )]
        let n_good = ((self.opts.gamma * done.len() as f64).ceil() as usize)
            .max(1)
            .min(done.len().saturating_sub(1).max(1));
        let bad = done.split_off(n_good);
        (done, bad)
    }
}

impl Searcher for TpeSearch {
    fn ask(&mut self, history: &[Trial]) -> Result<HashMap<String, JsonValue>, TuneError> {
        let completed = history
            .iter()
            .filter(|t| t.metric.is_some_and(f64::is_finite))
            .count();

        // Warm-up: not enough finished trials yet, draw from the prior.
        if completed < self.opts.n_startup_trials {
            return Ok(self.space.sample(&mut self.rng));
        }

        let (good, bad) = self.split_history(history);
        // If everything collapsed into one group (e.g. only one finite
        // metric exists), fall back to prior sampling.
        if good.is_empty() || bad.is_empty() {
            return Ok(self.space.sample(&mut self.rng));
        }

        let mut chosen = HashMap::with_capacity(self.space.len());
        // Snapshot the param list so the per-axis `self.propose_*` calls
        // below can take `&mut self.rng` without clashing with the
        // `&self.space` borrow on the loop.
        let params: Vec<(String, Distribution)> = self.space.params().to_vec();
        for (name, dist) in &params {
            let good_vals: Vec<&JsonValue> =
                good.iter().filter_map(|t| t.config.get(name)).collect();
            let bad_vals: Vec<&JsonValue> = bad.iter().filter_map(|t| t.config.get(name)).collect();

            let picked = match dist {
                Distribution::Uniform { low, high } => {
                    self.propose_continuous(*low, *high, &good_vals, &bad_vals, false)
                }
                Distribution::LogUniform { low, high } => {
                    self.propose_continuous(*low, *high, &good_vals, &bad_vals, true)
                }
                Distribution::IntUniform { low, high } => {
                    self.propose_int(*low, *high, &good_vals, &bad_vals)
                }
                Distribution::Categorical { choices } => {
                    self.propose_categorical(choices, &good_vals, &bad_vals)
                }
                Distribution::Discrete { values } => {
                    self.propose_discrete(values, &good_vals, &bad_vals)
                }
            };
            chosen.insert(name.clone(), picked);
        }
        Ok(chosen)
    }
}

impl TpeSearch {
    /// Gaussian-KDE density of `x` under observations `obs` with bandwidth
    /// `h`. Returns 1e-12 floor so the `l/g` ratio is always finite.
    fn kde_pdf(x: f64, obs: &[f64], h: f64) -> f64 {
        if obs.is_empty() || h <= 0.0 {
            return 1e-12;
        }
        let norm = 1.0 / ((2.0 * std::f64::consts::PI).sqrt() * h);
        #[allow(clippy::cast_precision_loss)]
        let n = obs.len() as f64;
        let mut acc = 0.0;
        for o in obs {
            let z = (x - o) / h;
            acc += (-0.5 * z * z).exp();
        }
        ((norm * acc) / n).max(1e-12)
    }

    fn extract_f64(v: &JsonValue, log: bool) -> Option<f64> {
        v.as_f64().map(|x| if log { x.ln() } else { x })
    }

    fn propose_continuous(
        &mut self,
        low: f64,
        high: f64,
        good_vals: &[&JsonValue],
        bad_vals: &[&JsonValue],
        log: bool,
    ) -> JsonValue {
        let (lo, hi) = if log {
            (low.ln(), high.ln())
        } else {
            (low, high)
        };
        let h = (hi - lo) * self.opts.bandwidth_frac;
        let good_obs: Vec<f64> = good_vals
            .iter()
            .filter_map(|v| Self::extract_f64(v, log))
            .collect();
        let bad_obs: Vec<f64> = bad_vals
            .iter()
            .filter_map(|v| Self::extract_f64(v, log))
            .collect();

        // Candidate sampler: 50% Gaussian-around-good observations
        // (Parzen-window draw), 50% uniform across the range. This keeps
        // exploration alive when the good cluster collapses.
        let mut best_x = self.rng.random_range(lo..hi);
        let mut best_score = f64::NEG_INFINITY;
        for _ in 0..self.opts.n_candidates {
            let x = if good_obs.is_empty() || self.rng.random_range(0.0..1.0) < 0.5 {
                self.rng.random_range(lo..hi)
            } else {
                let idx = self.rng.random_range(0..good_obs.len());
                let center = good_obs[idx];
                // Sample N(center, h) via Box–Muller (avoids extra dep).
                let u1: f64 = self.rng.random_range(f64::EPSILON..1.0);
                let u2: f64 = self.rng.random_range(0.0..1.0);
                let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
                (center + z * h).clamp(lo, hi)
            };
            let l = Self::kde_pdf(x, &good_obs, h);
            let g = Self::kde_pdf(x, &bad_obs, h);
            let score = l / g;
            if score > best_score {
                best_score = score;
                best_x = x;
            }
        }
        let value = if log { best_x.exp() } else { best_x };
        json!(value)
    }

    #[allow(clippy::cast_precision_loss)]
    fn propose_int(
        &mut self,
        low: i64,
        high: i64,
        good_vals: &[&JsonValue],
        bad_vals: &[&JsonValue],
    ) -> JsonValue {
        let lo_f = low as f64;
        let hi_f = high as f64;
        let h = ((hi_f - lo_f) * self.opts.bandwidth_frac).max(1.0);
        let good_obs: Vec<f64> = good_vals
            .iter()
            .filter_map(|v| v.as_i64().map(|x| x as f64))
            .collect();
        let bad_obs: Vec<f64> = bad_vals
            .iter()
            .filter_map(|v| v.as_i64().map(|x| x as f64))
            .collect();

        let mut best_x: i64 = self.rng.random_range(low..high);
        let mut best_score = f64::NEG_INFINITY;
        for _ in 0..self.opts.n_candidates {
            let candidate_f = if good_obs.is_empty() || self.rng.random_range(0.0..1.0) < 0.5 {
                self.rng.random_range(lo_f..hi_f)
            } else {
                let idx = self.rng.random_range(0..good_obs.len());
                let center = good_obs[idx];
                let u1: f64 = self.rng.random_range(f64::EPSILON..1.0);
                let u2: f64 = self.rng.random_range(0.0..1.0);
                let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
                (center + z * h).clamp(lo_f, hi_f - 1.0)
            };
            #[allow(clippy::cast_possible_truncation)]
            let candidate = (candidate_f.floor() as i64).clamp(low, high - 1);
            let cf = candidate as f64;
            let l = Self::kde_pdf(cf, &good_obs, h);
            let g = Self::kde_pdf(cf, &bad_obs, h);
            let score = l / g;
            if score > best_score {
                best_score = score;
                best_x = candidate;
            }
        }
        json!(best_x)
    }

    fn propose_categorical(
        &mut self,
        choices: &[JsonValue],
        good_vals: &[&JsonValue],
        bad_vals: &[&JsonValue],
    ) -> JsonValue {
        let counts = |obs: &[&JsonValue]| -> Vec<f64> {
            let mut c = vec![self.opts.categorical_prior; choices.len()];
            for v in obs {
                if let Some(idx) = choices.iter().position(|cand| cand == *v) {
                    c[idx] += 1.0;
                }
            }
            c
        };
        let good = counts(good_vals);
        let bad = counts(bad_vals);
        let good_sum: f64 = good.iter().sum();
        let bad_sum: f64 = bad.iter().sum();
        let mut best_idx = 0;
        let mut best_score = f64::NEG_INFINITY;
        for (i, _) in choices.iter().enumerate() {
            let l = good[i] / good_sum;
            let g = bad[i] / bad_sum;
            let score = l / g;
            if score > best_score {
                best_score = score;
                best_idx = i;
            }
        }
        choices[best_idx].clone()
    }

    fn propose_discrete(
        &mut self,
        values: &[f64],
        good_vals: &[&JsonValue],
        bad_vals: &[&JsonValue],
    ) -> JsonValue {
        let as_json: Vec<JsonValue> = values.iter().map(|v| json!(*v)).collect();
        let refs: Vec<&JsonValue> = as_json.iter().collect();
        // Treat each discrete value as a category.
        let counts = |obs: &[&JsonValue]| -> Vec<f64> {
            let mut c = vec![self.opts.categorical_prior; values.len()];
            for v in obs {
                if let Some(x) = v.as_f64()
                    && let Some(idx) = values
                        .iter()
                        .position(|cand| (cand - x).abs() < f64::EPSILON)
                {
                    c[idx] += 1.0;
                }
            }
            c
        };
        let good = counts(good_vals);
        let bad = counts(bad_vals);
        let good_sum: f64 = good.iter().sum();
        let bad_sum: f64 = bad.iter().sum();
        let mut best_idx = 0;
        let mut best_score = f64::NEG_INFINITY;
        for i in 0..values.len() {
            let l = good[i] / good_sum;
            let g = bad[i] / bad_sum;
            let score = l / g;
            if score > best_score {
                best_score = score;
                best_idx = i;
            }
        }
        let _ = refs; // hush dead_code in some configs
        json!(values[best_idx])
    }
}

#[cfg(test)]
#[allow(clippy::unreadable_literal, clippy::cast_sign_loss)]
mod tests {
    use super::*;
    use crate::trial::{Trial, TrialId};
    use std::collections::HashMap;

    fn space_one_continuous() -> SearchSpace {
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

    #[test]
    fn random_search_samples_uniformly_in_bounds() {
        let mut rs = RandomSearch::new(space_one_continuous(), 42);
        for _ in 0..200 {
            let s = rs.ask(&[]).unwrap();
            let x = s["x"].as_f64().unwrap();
            assert!((0.0..1.0).contains(&x));
        }
    }

    #[test]
    fn grid_search_covers_full_product() {
        let mut space = SearchSpace::new();
        space
            .add(
                "a",
                Distribution::Categorical {
                    choices: vec![json!("x"), json!("y"), json!("z")],
                },
            )
            .unwrap();
        space
            .add("b", Distribution::IntUniform { low: 0, high: 2 })
            .unwrap();
        let mut gs = GridSearch::new(&space).unwrap();
        assert_eq!(gs.total(), 6);
        let mut seen: std::collections::HashSet<(String, i64)> = std::collections::HashSet::new();
        for _ in 0..6 {
            let s = gs.ask(&[]).unwrap();
            let a = s["a"].as_str().unwrap().to_string();
            let b = s["b"].as_i64().unwrap();
            assert!(seen.insert((a, b)), "grid produced a duplicate");
        }
        assert!(gs.ask(&[]).is_err(), "grid should be exhausted");
        assert_eq!(seen.len(), 6);
    }

    /// Synthetic objective f(x) = (x - 0.7)^2 on [0, 1]. TPE should pull
    /// the best trial within 0.1 of 0.7 after 50 trials.
    #[test]
    fn tpe_converges_on_synthetic_unimodal_objective() {
        let space = space_one_continuous();
        let mut tpe = TpeSearch::new(space, 0xdeadbeef);
        let mut history: Vec<Trial> = Vec::new();
        for i in 0..50 {
            let cfg = tpe.ask(&history).unwrap();
            let x = cfg["x"].as_f64().unwrap();
            let metric = (x - 0.7).powi(2);
            let mut t = Trial::new(TrialId(i as u64), cfg);
            t.complete(metric);
            tpe.tell(&t);
            history.push(t);
        }
        let best = history
            .iter()
            .filter_map(|t| t.metric.map(|m| (m, t)))
            .min_by(|a, b| a.0.partial_cmp(&b.0).unwrap())
            .unwrap()
            .1;
        let best_x = best.config["x"].as_f64().unwrap();
        assert!(
            (best_x - 0.7).abs() < 0.1,
            "TPE failed to converge: best_x={best_x}"
        );
    }

    #[test]
    fn tpe_falls_back_to_random_before_startup() {
        let space = space_one_continuous();
        let mut tpe = TpeSearch::new(space, 1).with_options(TpeOptions {
            n_startup_trials: 5,
            ..TpeOptions::default()
        });
        // Empty history → should not panic, should yield a valid sample.
        for _ in 0..5 {
            let cfg = tpe.ask(&[]).unwrap();
            let x = cfg["x"].as_f64().unwrap();
            assert!((0.0..1.0).contains(&x));
        }
    }

    #[test]
    fn tpe_handles_categorical_param() {
        let mut space = SearchSpace::new();
        space
            .add(
                "opt",
                Distribution::Categorical {
                    choices: vec![json!("adam"), json!("sgd"), json!("lion")],
                },
            )
            .unwrap();
        let mut tpe = TpeSearch::new(space, 7).with_options(TpeOptions {
            n_startup_trials: 4,
            ..TpeOptions::default()
        });
        let mut history: Vec<Trial> = Vec::new();
        // Plant a history where "adam" is consistently good and others bad.
        for i in 0..12 {
            let cfg = tpe.ask(&history).unwrap();
            let opt = cfg["opt"].as_str().unwrap().to_string();
            let metric = if opt == "adam" { 0.05 } else { 0.9 };
            let mut t = Trial::new(TrialId(i), HashMap::from([("opt".into(), json!(opt))]));
            t.complete(metric);
            tpe.tell(&t);
            history.push(t);
        }
        // After 12 trials, TPE should prefer "adam".
        let mut adam_count = 0;
        for _ in 0..20 {
            let cfg = tpe.ask(&history).unwrap();
            if cfg["opt"].as_str() == Some("adam") {
                adam_count += 1;
            }
        }
        assert!(
            adam_count >= 18,
            "TPE should favor adam; got {adam_count}/20"
        );
    }
}
