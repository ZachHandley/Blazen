//! Non-commercial license warning helper for audio backends whose
//! upstream weights ship under CC-BY-NC-* terms (`AudioLDM`, Spark-TTS,
//! `MaskGCT`, `MusicGen`, ...).
//!
//! Use case: at the start of the first inference call, the backend
//! invokes [`warn_nc_once`] which emits a single `tracing::warn!` per
//! `(backend_id, model_id)` pair. Subsequent calls are no-ops. The set
//! of pairs that have already warned is tracked via a process-wide
//! `OnceLock<Mutex<HashSet<(String, String)>>>` so the warning is
//! idempotent per-pair without coordination from individual backends.

use std::collections::HashSet;
use std::sync::{Mutex, OnceLock};

static EMITTED: OnceLock<Mutex<HashSet<(String, String)>>> = OnceLock::new();

fn emitted_set() -> &'static Mutex<HashSet<(String, String)>> {
    EMITTED.get_or_init(|| Mutex::new(HashSet::new()))
}

/// Emit a one-time non-commercial-use warning for this `(backend_id,
/// model_id)` pair. Subsequent calls with the same pair are no-ops.
///
/// `license` should be a short license identifier (e.g.
/// `"CC-BY-NC-SA-4.0"`, `"CC-BY-NC-4.0"`, `"Stability AI Community
/// License"`) that gets surfaced in the warning text so users know
/// what they're agreeing to.
///
/// # Panics
///
/// Panics if the internal mutex has been poisoned by a previous
/// caller panicking while holding the lock. In practice the critical
/// section only mutates a `HashSet`, so poisoning should not occur in
/// well-behaved code.
pub fn warn_nc_once(backend_id: &str, model_id: &str, license: &str) {
    let set = emitted_set();
    let key = (backend_id.to_string(), model_id.to_string());
    let already = {
        let mut guard = set.lock().expect("nc-license set poisoned");
        if guard.contains(&key) {
            true
        } else {
            guard.insert(key);
            false
        }
    };
    if !already {
        tracing::warn!(
            backend = backend_id,
            model = model_id,
            license = license,
            "Loaded under {license} — non-commercial use only without separate authorial license. \
             Confirm your deployment is non-commercial OR has obtained commercial licensing from the model author."
        );
    }
}

/// Test-only helper that returns the current set of `(backend_id,
/// model_id)` pairs that have emitted a warning so tests can assert
/// idempotency without depending on a `tracing` subscriber.
#[cfg(test)]
pub(crate) fn entries_for_test() -> Vec<(String, String)> {
    let set = emitted_set();
    let guard = set.lock().expect("nc-license set poisoned");
    let mut v: Vec<(String, String)> = guard.iter().cloned().collect();
    v.sort();
    v
}

#[cfg(test)]
mod tests {
    use super::{entries_for_test, warn_nc_once};
    use std::sync::{Arc, Barrier};
    use std::thread;

    // The static `EMITTED` set is process-wide and shared across all
    // tests in this module. To keep the per-test assertions
    // independent, each test uses a unique backend/model id prefix so
    // entries inserted by other tests do not interfere.

    fn count_with_backend(prefix: &str) -> usize {
        entries_for_test()
            .into_iter()
            .filter(|(b, _)| b.starts_with(prefix))
            .count()
    }

    #[test]
    fn warn_nc_once_emits_once_per_pair() {
        let backend = "test-once-emits";
        let model = "model-a";
        warn_nc_once(backend, model, "CC-BY-NC-4.0");
        warn_nc_once(backend, model, "CC-BY-NC-4.0");
        warn_nc_once(backend, model, "CC-BY-NC-4.0");
        assert_eq!(count_with_backend(backend), 1);
    }

    #[test]
    fn warn_nc_once_emits_for_distinct_pairs() {
        let backend = "test-distinct-pairs";
        warn_nc_once(backend, "m1", "CC-BY-NC-4.0");
        warn_nc_once(backend, "m2", "CC-BY-NC-SA-4.0");
        warn_nc_once(backend, "m3", "Stability AI Community License");
        assert_eq!(count_with_backend(backend), 3);
        let entries = entries_for_test();
        assert!(entries.contains(&(backend.to_string(), "m1".to_string())));
        assert!(entries.contains(&(backend.to_string(), "m2".to_string())));
        assert!(entries.contains(&(backend.to_string(), "m3".to_string())));
    }

    #[test]
    fn warn_nc_once_handles_concurrent_callers() {
        let backend = "test-concurrent";
        let model = "shared-model";
        let license = "CC-BY-NC-SA-4.0";
        let n = 16;
        let barrier = Arc::new(Barrier::new(n));
        let mut handles = Vec::with_capacity(n);
        for _ in 0..n {
            let b = Arc::clone(&barrier);
            handles.push(thread::spawn(move || {
                b.wait();
                warn_nc_once(backend, model, license);
            }));
        }
        for h in handles {
            h.join().expect("worker thread panicked");
        }
        assert_eq!(count_with_backend(backend), 1);
    }
}
