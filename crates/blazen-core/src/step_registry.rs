//! Global registry of step builders, keyed by a stable step ID.
//!
//! This registry is the foundation for distributed workflow execution: a
//! peer node that receives a workflow request over the wire cannot
//! deserialize the original step closures (they are
//! [`Arc<dyn Fn(...) -> Future<StepOutput>>`](crate::StepFn)), so instead
//! it rebuilds the workflow by looking up step IDs in a process-global
//! registry. Both sides of the distributed call must have the same step
//! code compiled in — the wire protocol only carries opaque string IDs.
//!
//! # Step IDs
//!
//! A step ID is a `&'static str` that uniquely identifies a step builder
//! within a process. The intended format — once the `#[step]` proc-macro
//! is updated in a follow-up task — is
//! `concat!(module_path!(), "::", fn_name)` so both sides of a
//! distributed call naturally agree on a name.
//!
//! # Manual registration (current API)
//!
//! Until the `#[step]` proc-macro emits automatic registration, user code
//! can opt in by calling [`register_step_builder`] explicitly at startup:
//!
//! ```no_run
//! use blazen_core::{register_step_builder, StepFn, StepOutput, StepRegistration};
//! use blazen_events::{Event, StartEvent, StopEvent};
//! use std::sync::Arc;
//!
//! fn my_step() -> StepRegistration {
//!     let handler: StepFn = Arc::new(|event, _ctx| {
//!         Box::pin(async move {
//!             let start = event.as_any().downcast_ref::<StartEvent>().unwrap();
//!             Ok(StepOutput::Single(Box::new(StopEvent {
//!                 result: start.data.clone(),
//!             })))
//!         })
//!     });
//!
//!     StepRegistration {
//!         name: "my_step".into(),
//!         accepts: vec![StartEvent::event_type()],
//!         emits: vec![StopEvent::event_type()],
//!         handler,
//!         max_concurrency: 0,
//!     }
//! }
//!
//! // Call once at process startup (e.g. from `main` or a `ctor` fn).
//! register_step_builder("my_crate::my_step", my_step);
//! ```
//!
//! # Thread safety
//!
//! The registry is backed by [`dashmap::DashMap`] and is `Send + Sync`.
//! Registrations and lookups can be performed concurrently from any
//! thread.

use std::sync::OnceLock;

use dashmap::DashMap;

use crate::step::StepRegistration;

/// A function that, when called, produces a fresh [`StepRegistration`].
///
/// Used by [`StepDeserializerRegistry`] to reconstruct a step from its
/// stable ID — typically at resume time or when a peer workflow server
/// rebuilds a sub-workflow from the wire.
pub type StepBuilderFn = fn() -> StepRegistration;

/// Global registry of step builders, keyed by a stable step ID.
///
/// Step IDs are produced by the `#[step]` proc-macro as
/// `module_path!()::fn_name` so both sides of a distributed workflow can
/// agree on a name.
///
/// See the [module docs](self) for a full example.
pub struct StepDeserializerRegistry {
    inner: DashMap<&'static str, StepBuilderFn>,
}

impl StepDeserializerRegistry {
    /// Create a new empty registry.
    #[must_use]
    fn new() -> Self {
        Self {
            inner: DashMap::new(),
        }
    }

    /// Register a step builder under the given ID.
    ///
    /// Idempotent — if the same ID is registered twice with the same
    /// function, the second call is a no-op. A different function under
    /// the same ID panics in debug builds (to catch ID collisions during
    /// development) and is silently ignored in release (to avoid
    /// crashing a running service).
    pub fn register(&self, step_id: &'static str, builder: StepBuilderFn) {
        // Insert-if-absent: the closure is only run when the entry is
        // vacant, so the common (already-registered) path is lock-free
        // on the write side.
        let entry = self.inner.entry(step_id).or_insert(builder);

        // Collision detection: compare by function pointer. In debug
        // builds a mismatch is a programmer error and we panic; in
        // release builds we swallow the mismatch to keep the process
        // up.
        debug_assert!(
            std::ptr::fn_addr_eq(*entry.value(), builder),
            "step ID collision: `{step_id}` already registered with a different builder"
        );
    }

    /// Look up a step builder by ID.
    ///
    /// Returns `Some(StepRegistration)` if a builder is registered for
    /// `step_id`, `None` otherwise. Each call invokes the builder
    /// function and produces a fresh [`StepRegistration`].
    #[must_use]
    pub fn lookup(&self, step_id: &str) -> Option<StepRegistration> {
        self.inner.get(step_id).map(|builder| (*builder)())
    }

    /// Return all registered step IDs.
    ///
    /// Intended for diagnostics and introspection (e.g. a `/steps`
    /// admin endpoint on a distributed workflow peer). The returned
    /// order is unspecified.
    #[must_use]
    pub fn step_ids(&self) -> Vec<&'static str> {
        self.inner.iter().map(|entry| *entry.key()).collect()
    }

    /// Return the number of registered step builders.
    #[must_use]
    pub fn len(&self) -> usize {
        self.inner.len()
    }

    /// Returns `true` if no step builders are registered.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }
}

impl std::fmt::Debug for StepDeserializerRegistry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("StepDeserializerRegistry")
            .field("len", &self.inner.len())
            .finish()
    }
}

/// Process-global singleton registry.
fn global_registry() -> &'static StepDeserializerRegistry {
    static REGISTRY: OnceLock<StepDeserializerRegistry> = OnceLock::new();
    REGISTRY.get_or_init(StepDeserializerRegistry::new)
}

/// Register a step builder in the process-global
/// [`StepDeserializerRegistry`].
///
/// Typically called from a `ctor`- or
/// `linkme::distributed_slice`-initialized static so registration
/// happens at program startup without requiring explicit `fn main()`
/// setup. For now — until the `#[step]` proc-macro emits automatic
/// registration — user code must call this explicitly.
///
/// See the [module docs](self) for an example.
pub fn register_step_builder(step_id: &'static str, builder: StepBuilderFn) {
    global_registry().register(step_id, builder);
}

/// Look up a registered step builder in the process-global
/// [`StepDeserializerRegistry`].
#[must_use]
pub fn lookup_step_builder(step_id: &str) -> Option<StepRegistration> {
    global_registry().lookup(step_id)
}

/// Return all step IDs registered in the process-global
/// [`StepDeserializerRegistry`].
///
/// Intended for diagnostics. The returned order is unspecified.
#[must_use]
pub fn registered_step_ids() -> Vec<&'static str> {
    global_registry().step_ids()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use blazen_events::{Event, StartEvent, StopEvent};

    use crate::step::{StepFn, StepOutput, StepRegistration};

    use super::{
        StepDeserializerRegistry, lookup_step_builder, register_step_builder, registered_step_ids,
    };

    /// Build an echo step (`StartEvent` → `StopEvent`) whose registration
    /// carries the given `name`.
    fn make_echo_step(name: &'static str) -> StepRegistration {
        let handler: StepFn = Arc::new(|event, _ctx| {
            Box::pin(async move {
                let start = event
                    .as_any()
                    .downcast_ref::<StartEvent>()
                    .expect("expected StartEvent");
                let stop = StopEvent {
                    result: start.data.clone(),
                };
                Ok(StepOutput::Single(Box::new(stop)))
            })
        });

        StepRegistration {
            name: name.into(),
            accepts: vec![StartEvent::event_type()],
            emits: vec![StopEvent::event_type()],
            handler,
            max_concurrency: 0,
        }
    }

    fn echo_step_a() -> StepRegistration {
        make_echo_step("step_a")
    }

    fn echo_step_b() -> StepRegistration {
        make_echo_step("step_b")
    }

    #[test]
    fn register_and_lookup_single_step() {
        let registry = StepDeserializerRegistry::new();
        registry.register("test::step_a", echo_step_a);

        let looked_up = registry
            .lookup("test::step_a")
            .expect("registered step must resolve");
        assert_eq!(looked_up.name, "step_a");
        assert_eq!(looked_up.accepts, vec![StartEvent::event_type()]);
    }

    #[test]
    fn lookup_unknown_step_returns_none() {
        let registry = StepDeserializerRegistry::new();
        assert!(registry.lookup("test::does_not_exist").is_none());
    }

    #[test]
    fn register_same_builder_twice_is_idempotent() {
        let registry = StepDeserializerRegistry::new();
        registry.register("test::step_a", echo_step_a);
        registry.register("test::step_a", echo_step_a);
        assert_eq!(registry.len(), 1);
    }

    #[test]
    fn step_ids_lists_all_registered() {
        let registry = StepDeserializerRegistry::new();
        registry.register("test::step_a", echo_step_a);
        registry.register("test::step_b", echo_step_b);

        let mut ids = registry.step_ids();
        ids.sort_unstable();
        assert_eq!(ids, vec!["test::step_a", "test::step_b"]);
    }

    #[test]
    fn debug_impl_reports_len() {
        let registry = StepDeserializerRegistry::new();
        registry.register("test::step_a", echo_step_a);
        let s = format!("{registry:?}");
        assert!(s.contains("len: 1"));
    }

    /// Use unique IDs to avoid interfering with other tests that may
    /// mutate the global registry in parallel.
    #[test]
    fn global_register_and_lookup() {
        register_step_builder("blazen_core::step_registry::tests::global_a", echo_step_a);

        let looked_up = lookup_step_builder("blazen_core::step_registry::tests::global_a")
            .expect("global registration must resolve");
        assert_eq!(looked_up.name, "step_a");

        let ids = registered_step_ids();
        assert!(
            ids.contains(&"blazen_core::step_registry::tests::global_a"),
            "registered_step_ids() must include globally-registered IDs"
        );
    }
}
