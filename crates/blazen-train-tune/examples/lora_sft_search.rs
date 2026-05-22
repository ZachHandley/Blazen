//! Example: TPE hyperparameter search over `LoRA` SFT.
//!
//! Wires `blazen-train`'s `LoraConfig` + `OptimConfig` axes into a
//! `SearchSpace`, then runs a TPE search whose evaluator is a *mock*
//! training callback — the example is hermetic so it runs in CI without
//! a model download. Swap the `evaluator` body for a real
//! `blazen_train::Trainer::run()` call when you want live tuning.
//!
//! Usage:
//!   `cargo run -p blazen-train-tune --example lora_sft_search -- [max_trials]`

use std::{collections::HashMap, sync::Arc};

use blazen_train_tune::{
    Distribution, Evaluator, Runner, RunnerBudget, SearchSpace, TpeSearch, TrialJournal,
};
use serde_json::Value as JsonValue;

/// Mock objective: a smooth bowl over (log-lr, rank, alpha) with the
/// minimum near lr=3e-4, rank=16, alpha=32 — biologically plausible
/// `LoRA`-SFT sweet spot. Useful as a CI-friendly stand-in for actual
/// training.
fn mock_eval_loss(cfg: &HashMap<String, JsonValue>) -> f64 {
    let lr = cfg["learning_rate"].as_f64().unwrap();
    #[allow(
        clippy::cast_precision_loss,
        reason = "rank choices live in {4, 8, 16, 32, 64} — well within \
                  f64's 53-bit mantissa, so the cast is exact."
    )]
    let rank = cfg["rank"].as_i64().unwrap() as f64;
    let alpha = cfg["alpha"].as_f64().unwrap();
    let log_lr_target = (3e-4f64).ln();
    let log_lr = lr.ln();
    let lr_term = (log_lr - log_lr_target).powi(2);
    let rank_term = ((rank - 16.0) / 16.0).powi(2);
    let alpha_term = ((alpha - 32.0) / 32.0).powi(2);
    0.5 * (lr_term + rank_term + alpha_term) + 0.05
}

struct MockSftEvaluator;
impl Evaluator for MockSftEvaluator {
    fn evaluate(
        &self,
        config: HashMap<String, JsonValue>,
    ) -> blazen_train_tune::runner::EvalFuture {
        Box::pin(async move { Ok::<f64, String>(mock_eval_loss(&config)) })
    }
}

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber_init();

    let max_trials: usize = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(30);

    let mut space = SearchSpace::new();
    space.add(
        "learning_rate",
        Distribution::LogUniform {
            low: 1e-5,
            high: 1e-2,
        },
    )?;
    space.add(
        "rank",
        Distribution::Categorical {
            choices: vec![
                serde_json::json!(4),
                serde_json::json!(8),
                serde_json::json!(16),
                serde_json::json!(32),
                serde_json::json!(64),
            ],
        },
    )?;
    space.add(
        "alpha",
        Distribution::Discrete {
            values: vec![8.0, 16.0, 32.0, 64.0, 128.0],
        },
    )?;

    let journal_dir = dirs::cache_dir()
        .unwrap_or_else(|| std::path::PathBuf::from("."))
        .join("blazen-train-tune");
    std::fs::create_dir_all(&journal_dir)?;
    let journal_path = journal_dir.join("lora_sft_search.jsonl");
    // Fresh run each invocation (uncomment to *resume* instead):
    let _ = std::fs::remove_file(&journal_path);
    let journal = TrialJournal::open(&journal_path)?;

    let searcher = Box::new(TpeSearch::new(space.clone(), 0xb1a2e_u64));
    let evaluator: Arc<dyn Evaluator> = Arc::new(MockSftEvaluator);
    let mut runner = Runner::new(
        space,
        searcher,
        evaluator,
        RunnerBudget::MaxTrials(max_trials),
    )
    .with_journal(journal)?;

    println!("running {max_trials} TPE trials...");
    let history = runner.run().await?;

    let best = history
        .iter()
        .filter_map(|t| t.metric.map(|m| (m, t)))
        .min_by(|a, b| a.0.partial_cmp(&b.0).unwrap())
        .ok_or("no completed trial")?;
    println!(
        "best trial: {} metric={:.4} lr={:.2e} rank={} alpha={}",
        best.1.id,
        best.0,
        best.1.config["learning_rate"].as_f64().unwrap(),
        best.1.config["rank"].as_i64().unwrap(),
        best.1.config["alpha"].as_f64().unwrap(),
    );
    println!("journal: {}", journal_path.display());
    Ok(())
}

fn tracing_subscriber_init() {
    // Best-effort: don't fail the example if tracing isn't available.
    let _ = std::panic::catch_unwind(|| {
        // No tracing-subscriber dep here on purpose — keep example light.
    });
}
