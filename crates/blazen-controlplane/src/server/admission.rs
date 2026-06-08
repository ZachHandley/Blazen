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

use blazen_core::distributed::{
    AdmissionMode, NodeSelector, ResourceHint, TaintEffect, TolerationSpec, WorkerCapability,
};

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
    /// Workers exist (capability + tag matched) but none satisfied the
    /// node selector and/or taint+toleration constraints.
    SelectorMismatch,
    /// Workers match but all are saturated.
    Saturated,
    /// Submission targeted a `VramBudget` worker but omitted
    /// `resource_hint.vram_mb`.
    MissingVramHint,
}

/// Inputs to a single routing decision.
pub struct RouteRequest<'a> {
    /// Tenant/place this routing decision is scoped to. Only workers
    /// serving this place are considered as candidates (tenancy
    /// isolation). `"__default__"` for single-tenant deployments.
    pub place: &'a str,
    /// Required capability (e.g. `workflow:my-pipeline@1`).
    pub required_capability: &'a WorkerCapability,
    /// `key=value` (or `key=*` wildcard) tag requirements, AND-conjunction.
    pub required_tags: &'a [String],
    /// Resource estimate. Required for `VramBudget` workers.
    pub resource_hint: Option<&'a ResourceHint>,
    /// Node selector — `required` labels must match, `forbidden` must not,
    /// `preferred` contributes to candidate scoring.
    pub selector: &'a NodeSelector,
    /// Tolerations carried by the job — admit workers carrying matching
    /// taints. `NoSchedule` taints without a matching toleration exclude
    /// the worker; `PreferNoSchedule` taints without a matching toleration
    /// penalize the worker's score but do not exclude it.
    pub tolerations: &'a [TolerationSpec],
    /// Decoded assignment input. Used for `model_residency` affinity
    /// scoring — if the input is a JSON object containing a `model` or
    /// `model_name` string field, workers with that model already loaded
    /// gain a score bonus.
    pub input: Option<&'a serde_json::Value>,
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
        // Step 1: capability match, scoped to the request's place.
        let candidates = registry.workers_with_capability(req.place, req.required_capability);
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

        // Step 3 (new): NodeSelector + taint/toleration filter, with
        // PreferNoSchedule penalties + preferred-label bonuses + model
        // residency bonuses tracked per-worker. Score is signed; higher
        // is better.
        let requested_model = req.input.and_then(requested_model_name);
        let scored: Vec<(WorkerHandle, i32)> = after_tags
            .into_iter()
            .filter_map(|w| {
                score_candidate(
                    &w,
                    req.selector,
                    req.tolerations,
                    requested_model.as_deref(),
                )
                .map(|s| (w, s))
            })
            .collect();
        if scored.is_empty() {
            return Decision::NoCandidate {
                reason: NoCandidateReason::SelectorMismatch,
            };
        }

        // Step 4: capacity filter.
        let mut had_vram_miss = false;
        let with_capacity: Vec<(WorkerHandle, i32)> = scored
            .into_iter()
            .filter(|(w, _)| match has_capacity(w, req.resource_hint) {
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

        // Step 5: pick the best candidate. Primary key = highest score
        // (selector/taint/affinity); secondary = lowest load; ties broken
        // by round-robin.
        let max_score = with_capacity.iter().map(|(_, s)| *s).max().unwrap_or(0);
        let top_scored: Vec<WorkerHandle> = with_capacity
            .into_iter()
            .filter(|(_, s)| *s == max_score)
            .map(|(w, _)| w)
            .collect();

        let min_load = top_scored
            .iter()
            .map(load_score)
            .min_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap_or(0.0);

        let tied: Vec<WorkerHandle> = top_scored
            .into_iter()
            .filter(|w| (load_score(w) - min_load).abs() < f32::EPSILON)
            .collect();

        let rr = self.round_robin.fetch_add(1, Ordering::Relaxed);
        let chosen = &tied[rr % tied.len()];

        // Step 6: dispatch shape depends on admission mode.
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

/// Score penalty subtracted per un-tolerated `PreferNoSchedule` taint.
/// `-100` is large enough to distinguish from a preferred-label bonus
/// (`+10`) or a model-residency bonus (`+50`).
const PREFER_NO_SCHEDULE_PENALTY: i32 = 100;
/// Score bonus per matched preferred label in the node selector.
const PREFERRED_LABEL_BONUS: i32 = 10;
/// Score bonus when the worker already has the requested model resident.
const MODEL_RESIDENCY_BONUS: i32 = 50;

/// Extract the model name the caller is asking for, if the assignment
/// input is a JSON object containing `model_name` or `model` as a string.
fn requested_model_name(input: &serde_json::Value) -> Option<String> {
    let obj = input.as_object()?;
    for key in ["model_name", "model"] {
        if let Some(serde_json::Value::String(s)) = obj.get(key) {
            return Some(s.clone());
        }
    }
    None
}

/// Per-worker selector + taint + score evaluation.
///
/// Returns `None` if the worker is excluded by the selector or by a
/// `NoSchedule` taint without a matching toleration. Otherwise returns
/// the candidate's signed score: `PreferNoSchedule` taints subtract,
/// preferred-label matches and model-residency affinity add.
fn score_candidate(
    w: &WorkerHandle,
    selector: &NodeSelector,
    tolerations: &[TolerationSpec],
    requested_model: Option<&str>,
) -> Option<i32> {
    if !selector.admits(&w.labels) {
        return None;
    }
    let mut score: i32 = 0;
    for taint in &w.taints {
        if tolerations.iter().any(|t| t.matches(taint)) {
            continue;
        }
        match taint.effect {
            TaintEffect::NoSchedule => return None,
            TaintEffect::PreferNoSchedule => score -= PREFER_NO_SCHEDULE_PENALTY,
        }
    }
    let pref = i32::try_from(selector.preferred_match_count(&w.labels)).unwrap_or(i32::MAX);
    score += pref * PREFERRED_LABEL_BONUS;
    if let Some(model) = requested_model
        && w.admission_snapshot
            .as_ref()
            .is_some_and(|s| s.model_residency.contains(model))
    {
        score += MODEL_RESIDENCY_BONUS;
    }
    Some(score)
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
    use std::collections::{BTreeMap, BTreeSet, HashSet};

    use blazen_core::distributed::{AdmissionSnapshot, WorkerCapability, WorkerTaint};
    use tokio::sync::mpsc;

    fn cap(kind: &str, version: u32) -> WorkerCapability {
        WorkerCapability {
            kind: kind.to_string(),
            version,
        }
    }

    /// Default `RouteRequest` builder for tests. Most tests don't care about
    /// the selector/toleration fields, so they default to empty. `place`
    /// is the first arg — single-place tests pass `"__default__"`.
    fn route_req<'a>(
        place: &'a str,
        capability: &'a WorkerCapability,
        tags: &'a [String],
        hint: Option<&'a ResourceHint>,
        selector: &'a NodeSelector,
        tolerations: &'a [TolerationSpec],
        input: Option<&'a serde_json::Value>,
    ) -> RouteRequest<'a> {
        RouteRequest {
            place,
            required_capability: capability,
            required_tags: tags,
            resource_hint: hint,
            selector,
            tolerations,
            input,
        }
    }

    fn register_worker(
        reg: &WorkerRegistry,
        node_id: &str,
        capabilities: Vec<WorkerCapability>,
        tags: BTreeMap<String, String>,
        admission: AdmissionMode,
    ) -> (uuid::Uuid, mpsc::Receiver<crate::protocol::ServerToWorker>) {
        register_worker_in_place(
            reg,
            "__default__",
            node_id,
            capabilities,
            tags,
            admission,
            BTreeMap::new(),
            Vec::new(),
        )
    }

    /// Register a worker into the default place. Thin wrapper over
    /// [`register_worker_in_place`] for the many single-place tests.
    fn register_worker_full(
        reg: &WorkerRegistry,
        node_id: &str,
        capabilities: Vec<WorkerCapability>,
        tags: BTreeMap<String, String>,
        admission: AdmissionMode,
        labels: BTreeMap<String, String>,
        taints: Vec<WorkerTaint>,
    ) -> (uuid::Uuid, mpsc::Receiver<crate::protocol::ServerToWorker>) {
        register_worker_in_place(
            reg,
            "__default__",
            node_id,
            capabilities,
            tags,
            admission,
            labels,
            taints,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn register_worker_in_place(
        reg: &WorkerRegistry,
        place: &str,
        node_id: &str,
        capabilities: Vec<WorkerCapability>,
        tags: BTreeMap<String, String>,
        admission: AdmissionMode,
        labels: BTreeMap<String, String>,
        taints: Vec<WorkerTaint>,
    ) -> (uuid::Uuid, mpsc::Receiver<crate::protocol::ServerToWorker>) {
        let (tx, rx) = mpsc::channel(8);
        let caps: HashSet<_> = capabilities.into_iter().collect();
        let sid = reg.register(
            node_id.into(),
            place.into(),
            caps,
            tags,
            admission,
            tx,
            labels,
            taints,
            Vec::new(),
        );
        (sid, rx)
    }

    fn labels_from(pairs: &[(&str, &str)]) -> BTreeMap<String, String> {
        pairs
            .iter()
            .map(|(k, v)| ((*k).to_string(), (*v).to_string()))
            .collect()
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
        let sel = NodeSelector::default();
        let d = admission.route(&reg, &route_req("__default__", &need, &[], None, &sel, &[], None));
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
        let sel = NodeSelector::default();
        let d = admission.route(&reg, &route_req("__default__", &need, &[], None, &sel, &[], None));
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
        let sel = NodeSelector::default();
        let d = admission.route(&reg, &route_req("__default__", &need, &[], None, &sel, &[], None));
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
        let sel = NodeSelector::default();
        let d = admission.route(&reg, &route_req("__default__", &need, &[], None, &sel, &[], None));
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
        let sel = NodeSelector::default();
        let d = admission.route(&reg, &route_req("__default__", &need, &[], Some(&hint), &sel, &[], None));
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
        let sel = NodeSelector::default();
        let d = admission.route(&reg, &route_req("__default__", &need, &[], None, &sel, &[], None));
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
        let sel = NodeSelector::default();
        let req_tags = vec!["region=eu-central".to_string()];
        let d = admission.route(&reg, &route_req("__default__", &need, &req_tags, None, &sel, &[], None));
        match d {
            Decision::Push { session_id, .. } => assert_eq!(session_id, sid_b),
            other => panic!("expected Push to b, got {other:?}"),
        }

        // Wildcard match.
        let req_tags_wild = vec!["region=*".to_string()];
        let d2 = admission.route(
            &reg,
            &route_req("__default__", &need, &req_tags_wild, None, &sel, &[], None),
        );
        assert!(matches!(d2, Decision::Push { .. }));

        // No match.
        let req_tags_no = vec!["region=apac".to_string()];
        let d3 = admission.route(&reg, &route_req("__default__", &need, &req_tags_no, None, &sel, &[], None));
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
        let sel = NodeSelector::default();
        let d = admission.route(&reg, &route_req("__default__", &need, &[], None, &sel, &[], None));
        match d {
            Decision::Push { session_id, .. } => assert_eq!(session_id, sid_b),
            other => panic!("expected Push to b, got {other:?}"),
        }
        // sid_a unused
        let _ = sid_a;
    }

    // -----------------------------------------------------------------
    // NodeSelector + taint/toleration tests.
    // -----------------------------------------------------------------

    #[test]
    fn selector_required_filter_excludes_non_matching_workers() {
        let reg = WorkerRegistry::new();
        let need = cap("workflow:hello", 1);
        let (sid_nv, _) = register_worker_full(
            &reg,
            "nv",
            vec![need.clone()],
            BTreeMap::new(),
            AdmissionMode::Fixed { max_in_flight: 4 },
            labels_from(&[("gpu", "nvidia")]),
            Vec::new(),
        );
        let (_sid_cpu, _) = register_worker_full(
            &reg,
            "cpu",
            vec![need.clone()],
            BTreeMap::new(),
            AdmissionMode::Fixed { max_in_flight: 4 },
            labels_from(&[("cpu-only", "true")]),
            Vec::new(),
        );

        let sel = NodeSelector {
            required: vec!["gpu:nvidia".into()],
            forbidden: vec![],
            preferred: vec![],
        };
        let admission = Admission::new();
        // Run several times — round-robin must never pick the non-nvidia worker.
        for _ in 0..5 {
            let d = admission.route(&reg, &route_req("__default__", &need, &[], None, &sel, &[], None));
            match d {
                Decision::Push { session_id, .. } => assert_eq!(session_id, sid_nv),
                other => panic!("expected Push to nv, got {other:?}"),
            }
        }
    }

    #[test]
    fn selector_forbidden_excludes_matching_workers() {
        let reg = WorkerRegistry::new();
        let need = cap("workflow:hello", 1);
        let (_sid, _) = register_worker_full(
            &reg,
            "dev",
            vec![need.clone()],
            BTreeMap::new(),
            AdmissionMode::Fixed { max_in_flight: 4 },
            labels_from(&[("lifecycle", "dev")]),
            Vec::new(),
        );

        let sel = NodeSelector {
            required: vec![],
            forbidden: vec!["lifecycle:dev".into()],
            preferred: vec![],
        };
        let admission = Admission::new();
        let d = admission.route(&reg, &route_req("__default__", &need, &[], None, &sel, &[], None));
        assert!(matches!(
            d,
            Decision::NoCandidate {
                reason: NoCandidateReason::SelectorMismatch
            }
        ));
    }

    #[test]
    fn selector_preferred_breaks_ties_in_score() {
        let reg = WorkerRegistry::new();
        let need = cap("workflow:hello", 1);
        let (sid_pref, _) = register_worker_full(
            &reg,
            "pref",
            vec![need.clone()],
            BTreeMap::new(),
            AdmissionMode::Fixed { max_in_flight: 4 },
            labels_from(&[("region", "us-west"), ("host", "beastpc")]),
            Vec::new(),
        );
        let (_sid_plain, _) = register_worker_full(
            &reg,
            "plain",
            vec![need.clone()],
            BTreeMap::new(),
            AdmissionMode::Fixed { max_in_flight: 4 },
            labels_from(&[("region", "eu-central")]),
            Vec::new(),
        );

        let sel = NodeSelector {
            required: vec![],
            forbidden: vec![],
            preferred: vec![
                "region:us-west".into(),
                "host:beastpc".into(),
                "tier:premium".into(),
            ],
        };
        let admission = Admission::new();
        for _ in 0..5 {
            let d = admission.route(&reg, &route_req("__default__", &need, &[], None, &sel, &[], None));
            match d {
                Decision::Push { session_id, .. } => assert_eq!(session_id, sid_pref),
                other => panic!("expected Push to pref, got {other:?}"),
            }
        }
    }

    #[test]
    fn taint_no_schedule_blocks_without_toleration() {
        let reg = WorkerRegistry::new();
        let need = cap("workflow:hello", 1);
        let (_sid, _) = register_worker_full(
            &reg,
            "tainted",
            vec![need.clone()],
            BTreeMap::new(),
            AdmissionMode::Fixed { max_in_flight: 4 },
            BTreeMap::new(),
            vec![WorkerTaint {
                key: "dedicated".into(),
                value: Some("arrchitect".into()),
                effect: TaintEffect::NoSchedule,
            }],
        );

        let sel = NodeSelector::default();
        let admission = Admission::new();
        let d = admission.route(&reg, &route_req("__default__", &need, &[], None, &sel, &[], None));
        assert!(matches!(
            d,
            Decision::NoCandidate {
                reason: NoCandidateReason::SelectorMismatch
            }
        ));
    }

    #[test]
    fn taint_no_schedule_admits_with_toleration() {
        let reg = WorkerRegistry::new();
        let need = cap("workflow:hello", 1);
        let (sid, _) = register_worker_full(
            &reg,
            "tainted",
            vec![need.clone()],
            BTreeMap::new(),
            AdmissionMode::Fixed { max_in_flight: 4 },
            BTreeMap::new(),
            vec![WorkerTaint {
                key: "dedicated".into(),
                value: Some("arrchitect".into()),
                effect: TaintEffect::NoSchedule,
            }],
        );

        let sel = NodeSelector::default();
        let tols = vec![TolerationSpec {
            key: "dedicated".into(),
            value: Some("arrchitect".into()),
            effect: TaintEffect::NoSchedule,
        }];
        let admission = Admission::new();
        let d = admission.route(&reg, &route_req("__default__", &need, &[], None, &sel, &tols, None));
        match d {
            Decision::Push { session_id, .. } => assert_eq!(session_id, sid),
            other => panic!("expected Push to tainted worker, got {other:?}"),
        }
    }

    #[test]
    fn taint_prefer_no_schedule_only_penalizes_score() {
        let reg = WorkerRegistry::new();
        let need = cap("workflow:hello", 1);
        let (_sid_tainted, _) = register_worker_full(
            &reg,
            "tainted",
            vec![need.clone()],
            BTreeMap::new(),
            AdmissionMode::Fixed { max_in_flight: 4 },
            BTreeMap::new(),
            vec![WorkerTaint {
                key: "experimental".into(),
                value: None,
                effect: TaintEffect::PreferNoSchedule,
            }],
        );
        let (sid_clean, _) = register_worker_full(
            &reg,
            "clean",
            vec![need.clone()],
            BTreeMap::new(),
            AdmissionMode::Fixed { max_in_flight: 4 },
            BTreeMap::new(),
            Vec::new(),
        );

        let sel = NodeSelector::default();
        let admission = Admission::new();
        // Untainted worker should always win across multiple rounds.
        for _ in 0..5 {
            let d = admission.route(&reg, &route_req("__default__", &need, &[], None, &sel, &[], None));
            match d {
                Decision::Push { session_id, .. } => assert_eq!(session_id, sid_clean),
                other => panic!("expected Push to clean, got {other:?}"),
            }
        }
    }

    #[test]
    fn model_residency_preferred_bumps_score() {
        let reg = WorkerRegistry::new();
        let need = cap("workflow:hello", 1);
        let (sid_loaded, _) = register_worker_full(
            &reg,
            "loaded",
            vec![need.clone()],
            BTreeMap::new(),
            AdmissionMode::Fixed { max_in_flight: 4 },
            BTreeMap::new(),
            Vec::new(),
        );
        let (_sid_cold, _) = register_worker_full(
            &reg,
            "cold",
            vec![need.clone()],
            BTreeMap::new(),
            AdmissionMode::Fixed { max_in_flight: 4 },
            BTreeMap::new(),
            Vec::new(),
        );
        let mut residency = BTreeSet::new();
        residency.insert("llama-3-8b".to_string());
        reg.record_heartbeat(
            sid_loaded,
            0,
            Some(AdmissionSnapshot {
                capacity_score: 1.0,
                model_residency: residency,
                vram_free_mb: None,
                in_flight_vram_mb: 0,
            }),
        );

        let sel = NodeSelector::default();
        let input = serde_json::json!({ "model_name": "llama-3-8b" });
        let admission = Admission::new();
        for _ in 0..5 {
            let d = admission.route(&reg, &route_req("__default__", &need, &[], None, &sel, &[], Some(&input)));
            match d {
                Decision::Push { session_id, .. } => assert_eq!(session_id, sid_loaded),
                other => panic!("expected Push to loaded, got {other:?}"),
            }
        }
    }

    #[test]
    fn selector_mismatch_returns_dedicated_no_candidate_reason() {
        let reg = WorkerRegistry::new();
        let need = cap("workflow:hello", 1);
        // Two workers that pass capability but neither has the required label.
        let (_sid_a, _) = register_worker_full(
            &reg,
            "a",
            vec![need.clone()],
            BTreeMap::new(),
            AdmissionMode::Fixed { max_in_flight: 4 },
            labels_from(&[("region", "us-east")]),
            Vec::new(),
        );
        let (_sid_b, _) = register_worker_full(
            &reg,
            "b",
            vec![need.clone()],
            BTreeMap::new(),
            AdmissionMode::Fixed { max_in_flight: 4 },
            labels_from(&[("region", "us-east")]),
            Vec::new(),
        );

        let sel = NodeSelector {
            required: vec!["gpu:nvidia".into()],
            forbidden: vec![],
            preferred: vec![],
        };
        let admission = Admission::new();
        let d = admission.route(&reg, &route_req("__default__", &need, &[], None, &sel, &[], None));
        assert!(
            matches!(
                d,
                Decision::NoCandidate {
                    reason: NoCandidateReason::SelectorMismatch
                }
            ),
            "expected SelectorMismatch, got {d:?}"
        );
    }

    #[test]
    fn route_isolates_by_place_no_candidate_in_other_place() {
        // A worker exists, but only in place-B. A place-A submission must
        // see NoCandidate even though the capability matches in place-B.
        let reg = WorkerRegistry::new();
        let need = cap("workflow:hello", 1);
        let _ = register_worker_in_place(
            &reg,
            "place-b",
            "b-worker",
            vec![need.clone()],
            BTreeMap::new(),
            AdmissionMode::Fixed { max_in_flight: 4 },
            BTreeMap::new(),
            Vec::new(),
        );
        let admission = Admission::new();
        let sel = NodeSelector::default();
        let d = admission.route(&reg, &route_req("place-a", &need, &[], None, &sel, &[], None));
        assert!(
            matches!(
                d,
                Decision::NoCandidate {
                    reason: NoCandidateReason::NoCapability
                }
            ),
            "place-A submission must not route to a place-B worker, got {d:?}"
        );
    }

    #[test]
    fn route_matches_worker_in_same_place() {
        // The same submission routes when a worker serves place-A.
        let reg = WorkerRegistry::new();
        let need = cap("workflow:hello", 1);
        let (sid, _rx) = register_worker_in_place(
            &reg,
            "place-a",
            "a-worker",
            vec![need.clone()],
            BTreeMap::new(),
            AdmissionMode::Fixed { max_in_flight: 4 },
            BTreeMap::new(),
            Vec::new(),
        );
        let admission = Admission::new();
        let sel = NodeSelector::default();
        let d = admission.route(&reg, &route_req("place-a", &need, &[], None, &sel, &[], None));
        match d {
            Decision::Push { session_id, .. } => assert_eq!(session_id, sid),
            other => panic!("expected Push to the place-A worker, got {other:?}"),
        }
    }
}
