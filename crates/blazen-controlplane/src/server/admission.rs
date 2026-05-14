//! Server-side admission routing.
//!
//! Decides which connected worker should receive a submitted assignment,
//! and how to deliver it. Three admission modes are supported per worker
//! (see [`blazen_core::distributed::AdmissionMode`]):
//!
//! - `Fixed { max_in_flight }` — count cap. Skip if `in_flight >= cap`.
//! - `VramBudget { max_vram_mb }` — VRAM-sum cap. Skip if assignment's
//!   `resource_hint.vram_mb` plus the worker's reported
//!   `in_flight_vram_mb` exceeds the budget.
//! - `Reactive` — worker self-decides via offer/claim/decline. The
//!   admission logic returns `Decision::Offer`; the queue handles the
//!   retry-on-decline loop separately.
//!
//! ## Routing algorithm
//!
//! 1. **Hard filter (capability match)** — select workers whose
//!    `capabilities` set contains the assignment's required capability.
//! 2. **Hard filter (tag predicate)** — drop workers whose tags don't
//!    AND-match every `key=value` (or `key=*` wildcard) requirement.
//! 3. **Capacity filter** — drop workers whose admission mode reports
//!    no room for this assignment.
//! 4. **Tie-break: least-loaded** — pick the worker with the lowest
//!    normalized load (0..1). Ties broken by round-robin.
//! 5. **Reactive verification** — if the chosen worker is Reactive,
//!    return `Decision::Offer`; the queue layer pushes an Offer and
//!    handles Decline by re-routing (max 3 retries) — this module is
//!    purely stateless.

use std::sync::atomic::{AtomicUsize, Ordering};

use blazen_core::distributed::{AdmissionMode, ResourceHint, WorkerCapability};

use super::registry::{WorkerHandle, WorkerRegistry};

/// Server-side admission policy. Currently stateless except for a
/// round-robin counter to break ties among equally-loaded workers.
pub struct Admission {
    /// Monotonic counter for round-robin tie-breaking.
    round_robin: AtomicUsize,
}

/// Result of [`Admission::route`].
#[derive(Debug, Clone)]
pub enum Decision {
    /// Push the assignment directly to this worker (Fixed / `VramBudget`).
    Push {
        /// Session identifier of the chosen worker.
        session_id: uuid::Uuid,
        /// Stable node identifier of the chosen worker (for logging /
        /// metrics / `RunStateSnapshot::assigned_to`).
        node_id: String,
    },
    /// Offer the assignment to this worker (Reactive). Caller must
    /// await the `OfferDecision` and re-route on Decline.
    Offer {
        /// Session identifier of the chosen worker.
        session_id: uuid::Uuid,
        /// Stable node identifier of the chosen worker.
        node_id: String,
    },
    /// No worker matches. Caller decides whether to queue (if
    /// `wait_for_worker`) or return `FAILED_PRECONDITION`.
    NoCandidate {
        /// Why no candidate was found, so callers can surface a useful
        /// error to the orchestrator.
        reason: NoCandidateReason,
    },
}

/// Reason no candidate worker was available for an assignment. Carried
/// inside [`Decision::NoCandidate`].
#[derive(Debug, Clone)]
pub enum NoCandidateReason {
    /// No connected worker advertises the required capability.
    NoCapability,
    /// Workers exist with the capability but none match the tag predicate.
    TagMismatch,
    /// Workers match but all are saturated.
    Saturated,
    /// Submission targeted a `VramBudget` worker but omitted
    /// `resource_hint.vram_mb`.
    MissingVramHint,
}

/// Inputs to a single routing decision.
pub struct RouteRequest<'a> {
    /// Required capability (e.g. `workflow:my-pipeline@1`).
    pub required_capability: &'a WorkerCapability,
    /// `key=value` (or `key=*` wildcard) tag requirements, AND-conjunction.
    pub required_tags: &'a [String],
    /// Resource estimate. Required for `VramBudget` workers.
    pub resource_hint: Option<&'a ResourceHint>,
}

impl Admission {
    /// Construct a fresh admission controller with a zeroed round-robin
    /// counter.
    #[must_use]
    pub fn new() -> Self {
        Self {
            round_robin: AtomicUsize::new(0),
        }
    }

    /// Route a single assignment. Stateless w.r.t. registry — every
    /// call re-reads the live registry state.
    ///
    /// The five-step algorithm is documented at module level; the gist:
    /// filter by capability, then tags, then capacity, then pick the
    /// least-loaded worker with round-robin tie-breaking, finally choose
    /// `Push` vs `Offer` based on the worker's admission mode.
    pub fn route(&self, registry: &WorkerRegistry, req: &RouteRequest<'_>) -> Decision {
        // Step 1: capability match.
        let candidates = registry.workers_with_capability(req.required_capability);
        if candidates.is_empty() {
            return Decision::NoCandidate {
                reason: NoCandidateReason::NoCapability,
            };
        }

        // Step 2: tag predicate filter.
        let after_tags: Vec<WorkerHandle> = candidates
            .into_iter()
            .filter(|w| tags_match(req.required_tags, &w.tags))
            .collect();
        if after_tags.is_empty() {
            return Decision::NoCandidate {
                reason: NoCandidateReason::TagMismatch,
            };
        }

        // Step 3: capacity filter.
        let mut had_vram_miss = false;
        let with_capacity: Vec<WorkerHandle> = after_tags
            .into_iter()
            .filter(|w| match has_capacity(w, req.resource_hint) {
                CapacityCheck::HasRoom => true,
                CapacityCheck::Saturated => false,
                CapacityCheck::MissingVramHint => {
                    had_vram_miss = true;
                    false
                }
            })
            .collect();

        if with_capacity.is_empty() {
            if had_vram_miss {
                return Decision::NoCandidate {
                    reason: NoCandidateReason::MissingVramHint,
                };
            }
            return Decision::NoCandidate {
                reason: NoCandidateReason::Saturated,
            };
        }

        // Step 4: least-loaded with round-robin tie-break.
        let min_load = with_capacity
            .iter()
            .map(load_score)
            .min_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap_or(0.0);

        let tied: Vec<WorkerHandle> = with_capacity
            .into_iter()
            .filter(|w| (load_score(w) - min_load).abs() < f32::EPSILON)
            .collect();

        let rr = self.round_robin.fetch_add(1, Ordering::Relaxed);
        let chosen = &tied[rr % tied.len()];

        // Step 5: dispatch shape depends on admission mode.
        match chosen.admission {
            AdmissionMode::Reactive => Decision::Offer {
                session_id: chosen.session_id,
                node_id: chosen.node_id.clone(),
            },
            AdmissionMode::Fixed { .. } | AdmissionMode::VramBudget { .. } => Decision::Push {
                session_id: chosen.session_id,
                node_id: chosen.node_id.clone(),
            },
        }
    }
}

impl Default for Admission {
    fn default() -> Self {
        Self::new()
    }
}

/// Outcome of the per-worker capacity check inside [`Admission::route`].
enum CapacityCheck {
    /// Worker has room for this assignment.
    HasRoom,
    /// Worker is at its admission-mode cap.
    Saturated,
    /// Worker is `VramBudget` but the request omitted `vram_mb`.
    MissingVramHint,
}

/// Decide whether `w` currently has room for an assignment described by
/// `hint`. The semantics depend on the worker's admission mode:
///
/// - `Fixed` — strictly `in_flight < max_in_flight`.
/// - `VramBudget` — `hint.vram_mb` is required; reject if absent.
///   Otherwise check `in_flight_vram_mb + need <= max_vram_mb`.
/// - `Reactive` — accept if the worker's last reported `capacity_score`
///   is `> 0.0`; the real decision is made by the worker via offer/claim.
fn has_capacity(w: &WorkerHandle, hint: Option<&ResourceHint>) -> CapacityCheck {
    match w.admission {
        AdmissionMode::Fixed { max_in_flight } => {
            if w.in_flight < max_in_flight {
                CapacityCheck::HasRoom
            } else {
                CapacityCheck::Saturated
            }
        }
        AdmissionMode::VramBudget { max_vram_mb } => {
            let Some(need) = hint.and_then(|h| h.vram_mb) else {
                return CapacityCheck::MissingVramHint;
            };
            let used = w
                .admission_snapshot
                .as_ref()
                .map_or(0, |s| s.in_flight_vram_mb);
            if used.saturating_add(need) <= max_vram_mb {
                CapacityCheck::HasRoom
            } else {
                CapacityCheck::Saturated
            }
        }
        AdmissionMode::Reactive => {
            let score = w
                .admission_snapshot
                .as_ref()
                .map_or(1.0, |s| s.capacity_score);
            if score > 0.0 {
                CapacityCheck::HasRoom
            } else {
                CapacityCheck::Saturated
            }
        }
    }
}

/// Normalized load 0..1. Lower = more idle. Used for least-loaded
/// selection and tie-breaking.
///
/// This is a coarse 0..1 ratio used purely for ranking — exact float
/// precision is not required, so `clippy::cast_precision_loss` is
/// allowed here.
#[allow(clippy::cast_precision_loss)]
fn load_score(w: &WorkerHandle) -> f32 {
    match w.admission {
        AdmissionMode::Fixed { max_in_flight } if max_in_flight > 0 => {
            f32::from(u16::try_from(w.in_flight).unwrap_or(u16::MAX))
                / f32::from(u16::try_from(max_in_flight).unwrap_or(u16::MAX))
        }
        AdmissionMode::Fixed { .. } | AdmissionMode::VramBudget { max_vram_mb: 0 } => 1.0,
        AdmissionMode::VramBudget { max_vram_mb } => {
            let used = w
                .admission_snapshot
                .as_ref()
                .map_or(0, |s| s.in_flight_vram_mb);
            (used as f32) / (max_vram_mb as f32)
        }
        AdmissionMode::Reactive => {
            let score = w
                .admission_snapshot
                .as_ref()
                .map_or(1.0, |s| s.capacity_score);
            1.0 - score
        }
    }
}

/// AND-conjunction tag predicate match. Each entry in `required` is
/// `key=value` (exact) or `key=*` (any value, must just be present).
/// Returns `true` iff every entry matches.
fn tags_match(required: &[String], have: &std::collections::BTreeMap<String, String>) -> bool {
    required.iter().all(|entry| match entry.split_once('=') {
        Some((k, "*")) => have.contains_key(k),
        Some((k, v)) => have.get(k).map(String::as_str) == Some(v),
        None => false, // malformed predicate — fail closed.
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::{BTreeMap, HashSet};

    use blazen_core::distributed::WorkerCapability;
    use tokio::sync::mpsc;

    fn cap(kind: &str, version: u32) -> WorkerCapability {
        WorkerCapability {
            kind: kind.to_string(),
            version,
        }
    }

    fn register_worker(
        reg: &WorkerRegistry,
        node_id: &str,
        capabilities: Vec<WorkerCapability>,
        tags: BTreeMap<String, String>,
        admission: AdmissionMode,
    ) -> (uuid::Uuid, mpsc::Receiver<crate::protocol::ServerToWorker>) {
        let (tx, rx) = mpsc::channel(8);
        let caps: HashSet<_> = capabilities.into_iter().collect();
        let sid = reg.register(node_id.into(), caps, tags, admission, tx);
        (sid, rx)
    }

    #[test]
    fn route_no_capability_match() {
        let reg = WorkerRegistry::new();
        let _ = register_worker(
            &reg,
            "a",
            vec![cap("workflow:other", 1)],
            BTreeMap::new(),
            AdmissionMode::Fixed { max_in_flight: 4 },
        );
        let admission = Admission::new();
        let need = cap("workflow:wanted", 1);
        let d = admission.route(
            &reg,
            &RouteRequest {
                required_capability: &need,
                required_tags: &[],
                resource_hint: None,
            },
        );
        assert!(matches!(
            d,
            Decision::NoCandidate {
                reason: NoCandidateReason::NoCapability
            }
        ));
    }

    #[test]
    fn route_fixed_push_when_under_cap() {
        let reg = WorkerRegistry::new();
        let (sid, _rx) = register_worker(
            &reg,
            "a",
            vec![cap("workflow:hello", 1)],
            BTreeMap::new(),
            AdmissionMode::Fixed { max_in_flight: 4 },
        );
        let admission = Admission::new();
        let need = cap("workflow:hello", 1);
        let d = admission.route(
            &reg,
            &RouteRequest {
                required_capability: &need,
                required_tags: &[],
                resource_hint: None,
            },
        );
        match d {
            Decision::Push { session_id, .. } => assert_eq!(session_id, sid),
            other => panic!("expected Push, got {other:?}"),
        }
    }

    #[test]
    fn route_fixed_saturated_returns_no_candidate() {
        let reg = WorkerRegistry::new();
        let (sid, _rx) = register_worker(
            &reg,
            "a",
            vec![cap("workflow:hello", 1)],
            BTreeMap::new(),
            AdmissionMode::Fixed { max_in_flight: 2 },
        );
        reg.record_heartbeat(sid, 2, None); // at cap
        let admission = Admission::new();
        let need = cap("workflow:hello", 1);
        let d = admission.route(
            &reg,
            &RouteRequest {
                required_capability: &need,
                required_tags: &[],
                resource_hint: None,
            },
        );
        assert!(matches!(
            d,
            Decision::NoCandidate {
                reason: NoCandidateReason::Saturated
            }
        ));
    }

    #[test]
    fn route_vram_budget_requires_hint() {
        let reg = WorkerRegistry::new();
        let _ = register_worker(
            &reg,
            "a",
            vec![cap("workflow:hello", 1)],
            BTreeMap::new(),
            AdmissionMode::VramBudget {
                max_vram_mb: 24_000,
            },
        );
        let admission = Admission::new();
        let need = cap("workflow:hello", 1);
        let d = admission.route(
            &reg,
            &RouteRequest {
                required_capability: &need,
                required_tags: &[],
                resource_hint: None,
            },
        );
        assert!(matches!(
            d,
            Decision::NoCandidate {
                reason: NoCandidateReason::MissingVramHint
            }
        ));
    }

    #[test]
    fn route_vram_budget_accepts_when_fits() {
        let reg = WorkerRegistry::new();
        let (sid, _rx) = register_worker(
            &reg,
            "a",
            vec![cap("workflow:hello", 1)],
            BTreeMap::new(),
            AdmissionMode::VramBudget {
                max_vram_mb: 24_000,
            },
        );
        let admission = Admission::new();
        let need = cap("workflow:hello", 1);
        let hint = ResourceHint {
            vram_mb: Some(8_000),
            ..Default::default()
        };
        let d = admission.route(
            &reg,
            &RouteRequest {
                required_capability: &need,
                required_tags: &[],
                resource_hint: Some(&hint),
            },
        );
        match d {
            Decision::Push { session_id, .. } => assert_eq!(session_id, sid),
            other => panic!("expected Push, got {other:?}"),
        }
    }

    #[test]
    fn route_reactive_returns_offer() {
        let reg = WorkerRegistry::new();
        let (sid, _rx) = register_worker(
            &reg,
            "a",
            vec![cap("workflow:hello", 1)],
            BTreeMap::new(),
            AdmissionMode::Reactive,
        );
        let admission = Admission::new();
        let need = cap("workflow:hello", 1);
        let d = admission.route(
            &reg,
            &RouteRequest {
                required_capability: &need,
                required_tags: &[],
                resource_hint: None,
            },
        );
        match d {
            Decision::Offer { session_id, .. } => assert_eq!(session_id, sid),
            other => panic!("expected Offer, got {other:?}"),
        }
    }

    #[test]
    fn route_tag_predicate_filters() {
        let reg = WorkerRegistry::new();
        let mut tags_a = BTreeMap::new();
        tags_a.insert("region".into(), "us-west".into());
        let (_sid_a, _) = register_worker(
            &reg,
            "a",
            vec![cap("workflow:hello", 1)],
            tags_a,
            AdmissionMode::Fixed { max_in_flight: 4 },
        );
        let mut tags_b = BTreeMap::new();
        tags_b.insert("region".into(), "eu-central".into());
        let (sid_b, _) = register_worker(
            &reg,
            "b",
            vec![cap("workflow:hello", 1)],
            tags_b,
            AdmissionMode::Fixed { max_in_flight: 4 },
        );

        let admission = Admission::new();
        let need = cap("workflow:hello", 1);
        let req_tags = vec!["region=eu-central".to_string()];
        let d = admission.route(
            &reg,
            &RouteRequest {
                required_capability: &need,
                required_tags: &req_tags,
                resource_hint: None,
            },
        );
        match d {
            Decision::Push { session_id, .. } => assert_eq!(session_id, sid_b),
            other => panic!("expected Push to b, got {other:?}"),
        }

        // Wildcard match.
        let req_tags_wild = vec!["region=*".to_string()];
        let d2 = admission.route(
            &reg,
            &RouteRequest {
                required_capability: &need,
                required_tags: &req_tags_wild,
                resource_hint: None,
            },
        );
        assert!(matches!(d2, Decision::Push { .. }));

        // No match.
        let req_tags_no = vec!["region=apac".to_string()];
        let d3 = admission.route(
            &reg,
            &RouteRequest {
                required_capability: &need,
                required_tags: &req_tags_no,
                resource_hint: None,
            },
        );
        assert!(matches!(
            d3,
            Decision::NoCandidate {
                reason: NoCandidateReason::TagMismatch
            }
        ));
    }

    #[test]
    fn route_least_loaded_picks_lower_ratio() {
        let reg = WorkerRegistry::new();
        let (sid_a, _) = register_worker(
            &reg,
            "a",
            vec![cap("workflow:hello", 1)],
            BTreeMap::new(),
            AdmissionMode::Fixed { max_in_flight: 10 },
        );
        let (sid_b, _) = register_worker(
            &reg,
            "b",
            vec![cap("workflow:hello", 1)],
            BTreeMap::new(),
            AdmissionMode::Fixed { max_in_flight: 10 },
        );
        // A heavily loaded, B idle.
        reg.record_heartbeat(sid_a, 9, None);
        reg.record_heartbeat(sid_b, 1, None);

        let admission = Admission::new();
        let need = cap("workflow:hello", 1);
        let d = admission.route(
            &reg,
            &RouteRequest {
                required_capability: &need,
                required_tags: &[],
                resource_hint: None,
            },
        );
        match d {
            Decision::Push { session_id, .. } => assert_eq!(session_id, sid_b),
            other => panic!("expected Push to b, got {other:?}"),
        }
        // sid_a unused
        let _ = sid_a;
    }
}
