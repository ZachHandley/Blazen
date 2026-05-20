//! Learning-rate schedulers.
//!
//! Wave 2 of PR7 implements constant / linear-decay-with-warmup /
//! cosine-decay-with-warmup. The returned closure maps a 0-indexed step
//! to the learning rate to apply that step. Callers feed the result into
//! the optimizer's `set_learning_rate` before each `backward_step`.

// Why: step counters are bounded by `max_steps: usize` which is sized for
// training runs (millions of steps at most); 2^52 mantissa headroom on f64
// is many orders of magnitude beyond anything we'd see.
#![allow(clippy::cast_precision_loss)]

use crate::config::{SchedulerConfig, SchedulerKind};

/// Build the LR schedule closure described by `cfg`.
///
/// All three shapes share the same linear warmup: at step `s < warmup_steps`
/// the LR is `base_lr * (s + 1) / (warmup_steps + 1)`. After warmup, the
/// `Constant` shape holds `base_lr`, `Linear` decays to 0 at `max_steps`,
/// and `Cosine` follows a half-cosine to 0 at `max_steps`.
///
/// `max_steps` must be greater than `warmup_steps` for `Linear` and `Cosine`
/// to make sense. When equal (or `max_steps == 0`), the post-warmup region
/// is empty and the closure returns 0 for any step at or past `max_steps`.
#[must_use]
pub fn make_scheduler(
    cfg: &SchedulerConfig,
    base_lr: f64,
    max_steps: usize,
) -> Box<dyn Fn(usize) -> f64 + Send + Sync> {
    let warmup = cfg.warmup_steps;
    let warmup_denom = warmup as f64 + 1.0;
    match cfg.kind {
        SchedulerKind::Constant => Box::new(move |step| {
            if step < warmup {
                base_lr * (step as f64 + 1.0) / warmup_denom
            } else {
                base_lr
            }
        }),
        SchedulerKind::Linear => Box::new(move |step| {
            if step < warmup {
                base_lr * (step as f64 + 1.0) / warmup_denom
            } else if step >= max_steps || max_steps <= warmup {
                0.0
            } else {
                let progress = (step - warmup) as f64 / (max_steps - warmup) as f64;
                base_lr * (1.0 - progress)
            }
        }),
        SchedulerKind::Cosine => Box::new(move |step| {
            if step < warmup {
                base_lr * (step as f64 + 1.0) / warmup_denom
            } else if step >= max_steps || max_steps <= warmup {
                0.0
            } else {
                let progress = (step - warmup) as f64 / (max_steps - warmup) as f64;
                base_lr * 0.5 * (1.0 + (std::f64::consts::PI * progress).cos())
            }
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg(kind: SchedulerKind, warmup: usize) -> SchedulerConfig {
        SchedulerConfig {
            kind,
            warmup_steps: warmup,
        }
    }

    #[test]
    fn constant_lr_with_warmup_ramps() {
        let base_lr = 1.0;
        let warmup = 10;
        let sched = make_scheduler(&cfg(SchedulerKind::Constant, warmup), base_lr, 100);

        assert!((sched(0) - 1.0 / 11.0).abs() < 1e-12);
        assert!((sched(4) - 5.0 / 11.0).abs() < 1e-12);
        assert!((sched(9) - 10.0 / 11.0).abs() < 1e-12);
        assert!((sched(10) - base_lr).abs() < 1e-12);
        assert!((sched(50) - base_lr).abs() < 1e-12);
        assert!((sched(99_999) - base_lr).abs() < 1e-12);
    }

    #[test]
    fn linear_lr_decays_to_zero_at_max_steps() {
        let base_lr = 2.0;
        let warmup = 10;
        let max = 100;
        let sched = make_scheduler(&cfg(SchedulerKind::Linear, warmup), base_lr, max);

        assert!((sched(10) - base_lr).abs() < 1e-12);
        let midpoint = warmup + (max - warmup) / 2;
        assert!((sched(midpoint) - base_lr * 0.5).abs() < 1e-12);
        assert!((sched(max) - 0.0).abs() < 1e-12);
        assert!((sched(max + 999) - 0.0).abs() < 1e-12);
        let one_before = sched(max - 1);
        assert!(one_before > 0.0 && one_before < base_lr * 0.05);
    }

    #[test]
    fn cosine_lr_at_midpoint_is_half() {
        let base_lr = 3.0;
        let warmup = 0;
        let max = 200;
        let sched = make_scheduler(&cfg(SchedulerKind::Cosine, warmup), base_lr, max);

        let midpoint = max / 2;
        assert!((sched(midpoint) - base_lr * 0.5).abs() < 1e-12);
    }

    #[test]
    fn cosine_lr_at_warmup_end_equals_base_lr() {
        let base_lr = 1.5;
        let warmup = 25;
        let max = 500;
        let sched = make_scheduler(&cfg(SchedulerKind::Cosine, warmup), base_lr, max);

        assert!((sched(warmup) - base_lr).abs() < 1e-12);
        assert!((sched(max) - 0.0).abs() < 1e-12);
    }

    #[test]
    fn cosine_and_linear_handle_zero_post_warmup_region() {
        let sched = make_scheduler(&cfg(SchedulerKind::Cosine, 50), 1.0, 50);
        assert!((sched(50) - 0.0).abs() < 1e-12);
        let sched = make_scheduler(&cfg(SchedulerKind::Linear, 50), 1.0, 50);
        assert!((sched(50) - 0.0).abs() < 1e-12);
    }
}
