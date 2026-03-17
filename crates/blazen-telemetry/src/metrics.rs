//! Prometheus metrics layer for Blazen workflows.
//!
//! Provides a [`MetricsLayer`] that intercepts `tracing` spans as they close
//! and records Prometheus-compatible metrics via the `metrics` crate.
//!
//! # Recorded Metrics
//!
//! | Metric Name                        | Type      | Source Span      | Labels                |
//! |------------------------------------|-----------|------------------|-----------------------|
//! | `blazen_workflow_duration_seconds`  | Histogram | `workflow.run`   | --                    |
//! | `blazen_workflow_total`             | Counter   | `workflow.run`   | `status`              |
//! | `blazen_step_duration_seconds`      | Histogram | `workflow.step`  | --                    |
//! | `blazen_step_total`                 | Counter   | `workflow.step`  | `status`              |
//! | `blazen_llm_duration_seconds`       | Histogram | `llm.complete`   | --                    |
//! | `blazen_llm_tokens_total`           | Counter   | `llm.complete`   | `type` (prompt/completion) |
//! | `blazen_llm_requests_total`         | Counter   | `llm.complete`   | `status`              |

use metrics::{counter, histogram};
use tracing::Subscriber;
use tracing::field::{Field, Visit};
use tracing_subscriber::Layer;
use tracing_subscriber::layer::Context;
use tracing_subscriber::registry::LookupSpan;

/// A `tracing` layer that records span metrics into the `metrics` crate.
///
/// Attach this layer to your subscriber stack alongside the Prometheus
/// recorder (see [`crate::exporters::prometheus::init_prometheus`]) to get
/// automatic workflow/step/LLM metrics.
pub struct MetricsLayer;

/// Visitor that extracts known fields from a span's recorded attributes.
#[derive(Default)]
struct SpanFieldVisitor {
    duration_ms: Option<f64>,
    status: Option<String>,
    prompt_tokens: Option<u64>,
    completion_tokens: Option<u64>,
}

impl Visit for SpanFieldVisitor {
    fn record_f64(&mut self, field: &Field, value: f64) {
        if field.name() == "duration_ms" {
            self.duration_ms = Some(value);
        }
    }

    #[allow(clippy::cast_precision_loss)]
    fn record_u64(&mut self, field: &Field, value: u64) {
        match field.name() {
            "duration_ms" => self.duration_ms = Some(value as f64),
            "prompt_tokens" => self.prompt_tokens = Some(value),
            "completion_tokens" => self.completion_tokens = Some(value),
            _ => {}
        }
    }

    #[allow(clippy::cast_precision_loss)]
    fn record_i64(&mut self, field: &Field, value: i64) {
        match field.name() {
            "duration_ms" => self.duration_ms = Some(value as f64),
            "prompt_tokens" => self.prompt_tokens = Some(value.unsigned_abs()),
            "completion_tokens" => self.completion_tokens = Some(value.unsigned_abs()),
            _ => {}
        }
    }

    fn record_str(&mut self, field: &Field, value: &str) {
        if field.name() == "status" {
            self.status = Some(value.to_owned());
        }
    }

    fn record_debug(&mut self, field: &Field, value: &dyn std::fmt::Debug) {
        if field.name() == "status" {
            self.status = Some(format!("{value:?}"));
        }
    }
}

impl<S> Layer<S> for MetricsLayer
where
    S: Subscriber + for<'a> LookupSpan<'a>,
{
    fn on_close(&self, id: tracing::span::Id, ctx: Context<'_, S>) {
        let Some(span) = ctx.span(&id) else {
            return;
        };

        let name = span.name();

        // Read metrics data stored by on_new_span / on_record
        let extensions = span.extensions();
        let data = extensions.get::<RecordedFields>();

        let status_label = data.and_then(|d| d.status.as_deref()).unwrap_or("ok");
        let duration_secs = data
            .and_then(|d| d.duration_ms)
            .map_or(0.0, |ms| ms / 1000.0);

        match name {
            "workflow.run" => {
                histogram!("blazen_workflow_duration_seconds").record(duration_secs);
                counter!("blazen_workflow_total", "status" => status_label.to_owned()).increment(1);
            }
            "workflow.step" => {
                histogram!("blazen_step_duration_seconds").record(duration_secs);
                counter!("blazen_step_total", "status" => status_label.to_owned()).increment(1);
            }
            "llm.complete" => {
                histogram!("blazen_llm_duration_seconds").record(duration_secs);
                counter!("blazen_llm_requests_total", "status" => status_label.to_owned())
                    .increment(1);

                if let Some(d) = data {
                    if let Some(prompt_tokens) = d.prompt_tokens {
                        counter!("blazen_llm_tokens_total", "type" => "prompt")
                            .increment(prompt_tokens);
                    }
                    if let Some(completion_tokens) = d.completion_tokens {
                        counter!("blazen_llm_tokens_total", "type" => "completion")
                            .increment(completion_tokens);
                    }
                }
            }
            _ => {}
        }
    }

    fn on_record(
        &self,
        id: &tracing::span::Id,
        values: &tracing::span::Record<'_>,
        ctx: Context<'_, S>,
    ) {
        let Some(span) = ctx.span(id) else {
            return;
        };

        let mut visitor = SpanFieldVisitor::default();
        values.record(&mut visitor);

        let mut extensions = span.extensions_mut();
        if let Some(existing) = extensions.get_mut::<RecordedFields>() {
            // Merge new values into existing recorded fields
            if visitor.duration_ms.is_some() {
                existing.duration_ms = visitor.duration_ms;
            }
            if visitor.status.is_some() {
                existing.status = visitor.status;
            }
            if visitor.prompt_tokens.is_some() {
                existing.prompt_tokens = visitor.prompt_tokens;
            }
            if visitor.completion_tokens.is_some() {
                existing.completion_tokens = visitor.completion_tokens;
            }
        } else {
            extensions.insert(RecordedFields {
                duration_ms: visitor.duration_ms,
                status: visitor.status,
                prompt_tokens: visitor.prompt_tokens,
                completion_tokens: visitor.completion_tokens,
            });
        }
    }

    fn on_new_span(
        &self,
        attrs: &tracing::span::Attributes<'_>,
        id: &tracing::span::Id,
        ctx: Context<'_, S>,
    ) {
        let Some(span) = ctx.span(id) else {
            return;
        };

        // Capture initial span attributes
        let mut visitor = SpanFieldVisitor::default();
        attrs.record(&mut visitor);

        let mut extensions = span.extensions_mut();
        extensions.insert(RecordedFields {
            duration_ms: visitor.duration_ms,
            status: visitor.status,
            prompt_tokens: visitor.prompt_tokens,
            completion_tokens: visitor.completion_tokens,
        });
    }
}

/// Stored span field data, inserted into span extensions by [`MetricsLayer`].
struct RecordedFields {
    duration_ms: Option<f64>,
    status: Option<String>,
    prompt_tokens: Option<u64>,
    completion_tokens: Option<u64>,
}
