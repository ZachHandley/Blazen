//! Shared state for the REST surface.
//!
//! Holds the [`ManagerHandle`] trait object the routes dispatch into,
//! plus the [`ContentStore`] used for chunked uploads and a minimal
//! metrics counter that backs `GET /v1/blazen/metrics`. All fields are
//! interior-mutable (`Arc` / `Mutex` / atomic) so handlers can hold an
//! `Arc<RestState>` without taking a write lock.

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use dashmap::DashMap;

use crate::server::model_manager::ManagerHandle;

use super::uploads::ContentStore;

/// Per-RPC counter pair surfaced by `GET /v1/blazen/metrics`.
#[derive(Default)]
pub struct RestMetrics {
    /// Total successful + failed dispatches into [`ManagerHandle`].
    pub total_calls: AtomicU64,
    /// Per-RPC call counts, keyed by the wire-protocol verb name
    /// (`"complete"`, `"embed"`, `"transcribe"`, ...).
    pub by_rpc: DashMap<String, u64>,
    /// Per-RPC last-call wall-clock timestamp in milliseconds since the
    /// Unix epoch. Useful for liveness alerts (e.g. "no chat completion
    /// in the last 5 minutes").
    pub last_call_at_ms: DashMap<String, u64>,
}

impl RestMetrics {
    /// Record a single dispatch.
    pub fn record(&self, rpc: &str) {
        self.total_calls.fetch_add(1, Ordering::Relaxed);
        *self.by_rpc.entry(rpc.to_owned()).or_insert(0) += 1;
        let now_ms = match SystemTime::now().duration_since(UNIX_EPOCH) {
            Ok(d) => u64::try_from(d.as_millis()).unwrap_or(u64::MAX),
            Err(_) => 0,
        };
        self.last_call_at_ms.insert(rpc.to_owned(), now_ms);
    }

    /// Render the metrics in Prometheus exposition format.
    #[must_use]
    pub fn render_prometheus(&self) -> String {
        use std::fmt::Write;
        let mut out = String::new();
        out.push_str("# HELP blazen_rest_calls_total Total REST dispatches into ManagerHandle.\n");
        out.push_str("# TYPE blazen_rest_calls_total counter\n");
        let _ = writeln!(
            out,
            "blazen_rest_calls_total {}",
            self.total_calls.load(Ordering::Relaxed)
        );
        out.push_str("# HELP blazen_rest_calls Per-RPC dispatch count.\n");
        out.push_str("# TYPE blazen_rest_calls counter\n");
        for entry in &self.by_rpc {
            let _ = writeln!(
                out,
                "blazen_rest_calls{{rpc=\"{}\"}} {}",
                entry.key(),
                *entry.value()
            );
        }
        out.push_str("# HELP blazen_rest_last_call_at_ms Wall-clock timestamp of the last dispatch per RPC.\n");
        out.push_str("# TYPE blazen_rest_last_call_at_ms gauge\n");
        for entry in &self.last_call_at_ms {
            let _ = writeln!(
                out,
                "blazen_rest_last_call_at_ms{{rpc=\"{}\"}} {}",
                entry.key(),
                *entry.value()
            );
        }
        out
    }
}

/// Shared, cheaply-cloned state passed to every REST handler.
pub struct RestState {
    /// Trait object the routes dispatch into.
    pub handle: Arc<dyn ManagerHandle>,
    /// In-memory blob store for chunked uploads (adapter weights, audio
    /// transcription inputs > one request body).
    pub content_store: ContentStore,
    /// Lightweight call counters.
    pub metrics: RestMetrics,
}

impl RestState {
    /// Build a state around the given `ManagerHandle`.
    #[must_use]
    pub fn new(handle: Arc<dyn ManagerHandle>) -> Self {
        Self {
            handle,
            content_store: ContentStore::default(),
            metrics: RestMetrics::default(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn metrics_record_and_render() {
        let m = RestMetrics::default();
        m.record("complete");
        m.record("complete");
        m.record("embed");
        let prom = m.render_prometheus();
        assert!(prom.contains("blazen_rest_calls_total 3"));
        assert!(prom.contains("rpc=\"complete\"} 2"));
        assert!(prom.contains("rpc=\"embed\"} 1"));
        assert!(prom.contains("blazen_rest_last_call_at_ms"));
    }
}
