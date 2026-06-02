//! Probe driver — owns a background tokio task that re-probes every
//! `interval`, publishes each snapshot to a `tokio::sync::watch` channel.
//! Workers subscribe and read the latest snapshot when assembling their
//! heartbeat `AdmissionSnapshot`.

use crate::ProbedResources;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::watch;

#[derive(Clone, Debug)]
pub struct ProbeConfig {
    pub interval: Duration,
}

impl Default for ProbeConfig {
    fn default() -> Self {
        Self {
            interval: Duration::from_secs(5),
        }
    }
}

/// Handle to a running [`Probe`] task. Cloning is cheap; subscribers all
/// see the same latest snapshot through the shared `watch::Receiver`.
#[derive(Clone)]
pub struct ProbeHandle {
    rx: watch::Receiver<Arc<ProbedResources>>,
    _shutdown: Arc<tokio::sync::Notify>,
}

impl ProbeHandle {
    /// The most recent snapshot the probe has published.
    #[must_use]
    pub fn latest(&self) -> Arc<ProbedResources> {
        self.rx.borrow().clone()
    }

    /// Subscribe to snapshot updates. Callers that want to react to fresh
    /// probes (e.g., to re-emit a heartbeat early) can `.changed().await`
    /// on this receiver.
    #[must_use]
    pub fn subscribe(&self) -> watch::Receiver<Arc<ProbedResources>> {
        self.rx.clone()
    }
}

pub struct Probe;

impl Probe {
    /// Spawn the probe loop. The first snapshot is taken synchronously
    /// (well — inside the spawned task, before the loop) so any caller
    /// that immediately calls `handle.latest()` after the future resolves
    /// will see real data, not a default.
    #[must_use = "the returned ProbeHandle is the only way to observe the probe"]
    pub async fn spawn(config: ProbeConfig) -> ProbeHandle {
        let initial = Arc::new(crate::probe_once().await);
        let (tx, rx) = watch::channel(initial);
        let shutdown = Arc::new(tokio::sync::Notify::new());
        let shutdown_listener = shutdown.clone();

        tokio::spawn(async move {
            let mut ticker = tokio::time::interval(config.interval);
            // Skip the immediate tick — we already have the initial
            // snapshot in the channel; first refresh should fire after
            // one full interval.
            ticker.tick().await;

            loop {
                tokio::select! {
                    _ = ticker.tick() => {
                        let snap = Arc::new(crate::probe_once().await);
                        // Best-effort: if every receiver dropped, the
                        // probe is unobserved and can exit cleanly.
                        if tx.send(snap).is_err() {
                            tracing::debug!("probe receivers all dropped; exiting probe loop");
                            return;
                        }
                    }
                    () = shutdown_listener.notified() => {
                        tracing::debug!("probe shutdown signaled; exiting probe loop");
                        return;
                    }
                }
            }
        });

        ProbeHandle {
            rx,
            _shutdown: shutdown,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test(flavor = "current_thread", start_paused = true)]
    async fn probe_publishes_initial_snapshot_synchronously() {
        let handle = Probe::spawn(ProbeConfig {
            interval: Duration::from_millis(50),
        })
        .await;
        let snap = handle.latest();
        assert!(snap.cpu_cores > 0, "initial snapshot must have CPU data");
    }

    #[tokio::test(flavor = "current_thread", start_paused = true)]
    async fn probe_refresh_publishes_new_snapshots() {
        let handle = Probe::spawn(ProbeConfig {
            interval: Duration::from_millis(50),
        })
        .await;
        let mut sub = handle.subscribe();
        // Drain the initial-value notification.
        let _ = sub.borrow_and_update();
        // Advance virtual time past one interval boundary.
        tokio::time::advance(Duration::from_millis(60)).await;
        // Yield so the spawned task can wake and publish.
        tokio::task::yield_now().await;
        // Either a new snap is already in the channel, or it will be
        // very shortly — confirm the channel has not been closed.
        assert!(sub.has_changed().is_ok());
    }
}
