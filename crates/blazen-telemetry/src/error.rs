//! Error types for the `blazen-telemetry` crate.

use std::fmt;

/// Errors that can occur while initializing or operating telemetry exporters.
#[derive(Debug)]
pub enum TelemetryError {
    /// A Langfuse exporter failed to initialize or operate.
    Langfuse(String),
}

impl fmt::Display for TelemetryError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Langfuse(msg) => write!(f, "langfuse error: {msg}"),
        }
    }
}

impl std::error::Error for TelemetryError {}
