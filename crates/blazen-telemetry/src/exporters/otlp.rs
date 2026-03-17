//! OpenTelemetry OTLP exporter.
//!
//! Exports traces to any OTLP-compatible backend (Jaeger, Grafana Tempo,
//! Honeycomb, Datadog, etc.) via gRPC.

use opentelemetry::global;
use opentelemetry::trace::TracerProvider;
use opentelemetry_otlp::{SpanExporter, WithExportConfig};
use opentelemetry_sdk::{resource::Resource, trace::SdkTracerProvider};
use serde::{Deserialize, Serialize};
use tracing_opentelemetry::OpenTelemetryLayer;
use tracing_subscriber::{EnvFilter, layer::SubscriberExt, util::SubscriberInitExt};

/// Configuration for the OTLP exporter.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OtlpConfig {
    /// The OTLP endpoint URL (e.g. `"http://localhost:4317"`).
    pub endpoint: String,
    /// The service name reported to the backend.
    pub service_name: String,
}

/// Initialize the OTLP trace exporter and install it as the global tracing
/// subscriber layer.
///
/// This sets up:
/// 1. An OTLP gRPC (tonic) span exporter pointed at `config.endpoint`
/// 2. A `SdkTracerProvider` with batch export and the configured service name
/// 3. A `tracing_opentelemetry` layer bridging `tracing` spans to OpenTelemetry
/// 4. A combined subscriber with both the `OTel` layer and a `fmt` layer
///
/// # Errors
///
/// Returns an error if the OTLP exporter or tracer provider cannot be created.
pub fn init_otlp(config: OtlpConfig) -> Result<(), Box<dyn std::error::Error>> {
    // 1. Build the OTLP gRPC span exporter via tonic transport
    let exporter = SpanExporter::builder()
        .with_tonic()
        .with_endpoint(&config.endpoint)
        .build()?;

    // 2. Build the tracer provider with batch export and service.name resource
    let resource = Resource::builder_empty()
        .with_service_name(config.service_name)
        .build();

    let provider = SdkTracerProvider::builder()
        .with_batch_exporter(exporter)
        .with_resource(resource)
        .build();

    // 3. Set the provider as the global tracer provider
    global::set_tracer_provider(provider.clone());

    // 4. Create the tracing-opentelemetry layer using a tracer from the provider
    let tracer = provider.tracer("blazen");
    let otel_layer = OpenTelemetryLayer::new(tracer);

    // 5. Compose with an env-filter and fmt layer, then install
    let env_filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info"));

    tracing_subscriber::registry()
        .with(env_filter)
        .with(otel_layer)
        .with(tracing_subscriber::fmt::layer())
        .init();

    Ok(())
}
