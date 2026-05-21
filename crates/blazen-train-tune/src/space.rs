//! `SearchSpace` + `Distribution` types.
//!
//! A `SearchSpace` is an ordered list of named hyperparameter priors. Order
//! matters for grid search (outer-most first), and the name is what the
//! caller's evaluator reads back out of the per-trial config map.

use std::collections::HashMap;

use rand::{Rng, RngExt};
use serde::{Deserialize, Serialize};
use serde_json::{Value as JsonValue, json};

use crate::error::TuneError;

/// Prior distribution over one hyperparameter.
///
/// The `Categorical` arm carries `JsonValue`s rather than `String`s so that
/// callers can pass typed options (e.g. integer ranks, boolean flags, or
/// even nested JSON objects for module-target lists).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind")]
pub enum Distribution {
    /// Discrete choice over an arbitrary list of values.
    Categorical { choices: Vec<JsonValue> },
    /// Uniform integer in `[low, high)` (high-exclusive, matching `rand`).
    IntUniform { low: i64, high: i64 },
    /// Uniform real in `[low, high)`.
    Uniform { low: f64, high: f64 },
    /// Log-uniform real in `[low, high)`. Both bounds must be strictly
    /// positive. Sampling is `exp(U(log(low), log(high)))`.
    LogUniform { low: f64, high: f64 },
    /// Discrete choice over a fixed list of floats. Useful for learning-rate
    /// schedules / batch sizes you want grid-searched exactly.
    Discrete { values: Vec<f64> },
}

impl Distribution {
    /// Friendly type tag for error messages.
    #[must_use]
    pub fn kind_str(&self) -> &'static str {
        match self {
            Self::Categorical { .. } => "Categorical",
            Self::IntUniform { .. } => "IntUniform",
            Self::Uniform { .. } => "Uniform",
            Self::LogUniform { .. } => "LogUniform",
            Self::Discrete { .. } => "Discrete",
        }
    }

    fn validate(&self) -> Result<(), TuneError> {
        match self {
            Self::IntUniform { low, high } => {
                if low >= high {
                    return Err(TuneError::InvalidBounds(format!(
                        "IntUniform low={low} >= high={high}"
                    )));
                }
            }
            Self::Uniform { low, high } => {
                if !(low.is_finite() && high.is_finite()) || low >= high {
                    return Err(TuneError::InvalidBounds(format!(
                        "Uniform low={low} high={high} (need finite, low<high)"
                    )));
                }
            }
            Self::LogUniform { low, high } => {
                if !(low.is_finite() && high.is_finite())
                    || *low <= 0.0
                    || *high <= 0.0
                    || low >= high
                {
                    return Err(TuneError::InvalidBounds(format!(
                        "LogUniform low={low} high={high} (need finite, both >0, low<high)"
                    )));
                }
            }
            Self::Categorical { choices } => {
                if choices.is_empty() {
                    return Err(TuneError::InvalidBounds(
                        "Categorical with no choices".into(),
                    ));
                }
            }
            Self::Discrete { values } => {
                if values.is_empty() {
                    return Err(TuneError::InvalidBounds("Discrete with no values".into()));
                }
            }
        }
        Ok(())
    }

    /// Draw a single sample.
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> JsonValue {
        match self {
            Self::Categorical { choices } => {
                // SAFETY (panic-free): validate() guarantees `choices` is
                // non-empty before any sample call.
                let idx = rng.random_range(0..choices.len());
                choices[idx].clone()
            }
            Self::IntUniform { low, high } => {
                let v: i64 = rng.random_range(*low..*high);
                json!(v)
            }
            Self::Uniform { low, high } => {
                let v: f64 = rng.random_range(*low..*high);
                json!(v)
            }
            Self::LogUniform { low, high } => {
                let lo = low.ln();
                let hi = high.ln();
                let u: f64 = rng.random_range(lo..hi);
                json!(u.exp())
            }
            Self::Discrete { values } => {
                let idx = rng.random_range(0..values.len());
                json!(values[idx])
            }
        }
    }

    /// Number of distinct values this distribution can take, if any.
    /// `None` for continuous distributions (used by grid search).
    #[must_use]
    pub fn cardinality(&self) -> Option<usize> {
        match self {
            Self::Categorical { choices } => Some(choices.len()),
            Self::Discrete { values } => Some(values.len()),
            Self::IntUniform { low, high } => {
                let n = high - low;
                if n > 0 {
                    Some(usize::try_from(n).unwrap_or(usize::MAX))
                } else {
                    Some(0)
                }
            }
            Self::Uniform { .. } | Self::LogUniform { .. } => None,
        }
    }

    /// Enumerate this distribution's values as JSON. Returns
    /// `UngriddableDistribution` for continuous priors.
    pub fn enumerate(&self) -> Result<Vec<JsonValue>, TuneError> {
        match self {
            Self::Categorical { choices } => Ok(choices.clone()),
            Self::Discrete { values } => Ok(values.iter().map(|v| json!(*v)).collect()),
            Self::IntUniform { low, high } => Ok((*low..*high).map(|v| json!(v)).collect()),
            Self::Uniform { .. } | Self::LogUniform { .. } => {
                Err(TuneError::UngriddableDistribution(
                    self.kind_str().to_string(),
                    "continuous distributions cannot be enumerated for grid search; \
                     convert to Discrete or Categorical",
                ))
            }
        }
    }
}

/// An ordered, named collection of hyperparameter priors.
///
/// `add` returns `&mut Self` so the builder reads top-to-bottom. Insertion
/// order is preserved (important for `GridSearch` axis order and for
/// deterministic sampling under a fixed seed).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SearchSpace {
    params: Vec<(String, Distribution)>,
}

impl SearchSpace {
    #[must_use]
    pub fn new() -> Self {
        Self { params: Vec::new() }
    }

    /// Register a hyperparameter. Validates the distribution; later
    /// duplicate names overwrite earlier ones (and warn via `tracing`).
    pub fn add(
        &mut self,
        name: impl Into<String>,
        dist: Distribution,
    ) -> Result<&mut Self, TuneError> {
        dist.validate()?;
        let name = name.into();
        if let Some(slot) = self.params.iter_mut().find(|(n, _)| n == &name) {
            tracing::warn!(param = %name, "SearchSpace::add overwrote existing distribution");
            slot.1 = dist;
        } else {
            self.params.push((name, dist));
        }
        Ok(self)
    }

    /// Number of parameters registered.
    #[must_use]
    pub fn len(&self) -> usize {
        self.params.len()
    }

    /// `true` if no parameters are registered.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.params.is_empty()
    }

    /// Slice of `(name, distribution)` pairs in insertion order.
    #[must_use]
    pub fn params(&self) -> &[(String, Distribution)] {
        &self.params
    }

    /// Look up a parameter's distribution by name.
    #[must_use]
    pub fn get(&self, name: &str) -> Option<&Distribution> {
        self.params
            .iter()
            .find_map(|(n, d)| (n == name).then_some(d))
    }

    /// Draw one sample for every parameter.
    pub fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> HashMap<String, JsonValue> {
        self.params
            .iter()
            .map(|(name, dist)| (name.clone(), dist.sample(rng)))
            .collect()
    }
}

#[cfg(test)]
#[allow(clippy::unreadable_literal)]
mod tests {
    use super::*;
    use rand::{SeedableRng, rngs::StdRng};

    #[test]
    fn add_validates_bounds() {
        let mut s = SearchSpace::new();
        let err = s
            .add(
                "bad",
                Distribution::Uniform {
                    low: 1.0,
                    high: 0.0,
                },
            )
            .unwrap_err();
        assert!(matches!(err, TuneError::InvalidBounds(_)));
        let err = s
            .add(
                "bad_log",
                Distribution::LogUniform {
                    low: 0.0,
                    high: 1.0,
                },
            )
            .unwrap_err();
        assert!(matches!(err, TuneError::InvalidBounds(_)));
    }

    #[test]
    fn sample_produces_one_value_per_param() {
        let mut s = SearchSpace::new();
        s.add(
            "lr",
            Distribution::LogUniform {
                low: 1e-5,
                high: 1e-2,
            },
        )
        .unwrap();
        s.add(
            "rank",
            Distribution::Categorical {
                choices: vec![json!(4), json!(8), json!(16)],
            },
        )
        .unwrap();
        let mut rng = StdRng::seed_from_u64(0xc0ffee);
        let sample = s.sample(&mut rng);
        assert_eq!(sample.len(), 2);
        let lr = sample["lr"].as_f64().unwrap();
        assert!((1e-5..1e-2).contains(&lr));
        assert!([4, 8, 16].contains(&sample["rank"].as_i64().unwrap()));
    }

    #[test]
    fn enumerate_rejects_continuous() {
        let d = Distribution::Uniform {
            low: 0.0,
            high: 1.0,
        };
        assert!(d.enumerate().is_err());
    }
}
