//! Session-ref JSON walker for `blazen-node`.
//!
//! This is the Node equivalent of `blazen-py/src/convert.rs:226-323` —
//! the code that detects `__blazen_session_ref__` markers in event
//! payloads and routes lookups through the active
//! [`SessionRefRegistry`][blazen_core::session_ref::SessionRefRegistry].
//!
//! ## Current scope (Phase 0.6)
//!
//! napi-rs currently passes step-handler arguments as
//! `serde_json::Value` (see
//! [`crate::workflow::workflow::JsWorkflow`]), which means **raw JS
//! object identity does not survive the Rust↔JS boundary today** — a JS
//! object becomes a plain `serde_json::Value` on entry and loses its
//! reference on the way back. Fully-preserving session-ref semantics for
//! JS objects therefore needs a broader bridge refactor where step
//! handlers take `napi::Unknown` instead of `serde_json::Value`. That
//! work is tracked by the WASM / JS-bridge phase and is out of scope
//! for the Phase 0 bug fix.
//!
//! What this module **does** provide today:
//!
//! - [`SESSION_REF_TAG`] — re-export of the canonical JSON marker.
//! - [`find_session_ref_key`] — check whether a JSON value is a
//!   `{"__blazen_session_ref__": "<uuid>"}` marker, returning the UUID
//!   if so.
//! - [`make_session_ref_json`] — build the canonical marker JSON from
//!   a [`RegistryKey`], for Rust-side code that wants to insert a
//!   value into the registry and emit the tag itself.
//! - [`resolve_session_refs_in_json`] — walk a
//!   `serde_json::Value`, replace any `{"__blazen_session_ref__":
//!   "<uuid>"}` markers with the underlying value **if** the underlying
//!   value is `Arc<serde_json::Value>` (i.e. something Rust code stored
//!   via `registry.insert(Arc::new(value))`). Non-`Value` payloads are
//!   left tagged — the caller must resolve them out-of-band.
//!
//! Once the napi-rs bridge is refactored to carry `napi::Unknown` end
//! to end, this module gains the full walker that stores
//! `Arc<napi::Ref<Unknown>>` entries in the registry and resolves them
//! back out on the next step boundary.

use std::sync::Arc;

#[doc(inline)]
pub use blazen_core::session_ref::SESSION_REF_TAG;
use blazen_core::session_ref::{RegistryKey, RemoteRefDescriptor, SessionRefRegistry};
use serde_json::Value;

/// If `value` is a `{"__blazen_session_ref__": "<uuid>"}` marker,
/// return the parsed [`RegistryKey`]. Otherwise return `None`.
#[must_use]
pub fn find_session_ref_key(value: &Value) -> Option<RegistryKey> {
    let obj = value.as_object()?;
    if obj.len() != 1 {
        return None;
    }
    let raw = obj.get(SESSION_REF_TAG)?.as_str()?;
    RegistryKey::parse(raw).ok()
}

/// Build a `{"__blazen_session_ref__": "<uuid>"}` JSON marker from a key.
///
/// Used by Rust-side code that stores a value in the registry via
/// [`SessionRefRegistry::insert`] and wants to include the marker
/// inside an event payload.
#[must_use]
pub fn make_session_ref_json(key: RegistryKey) -> Value {
    serde_json::json!({
        SESSION_REF_TAG: key.to_string(),
    })
}

/// Recursively walk `value`, replacing any
/// `{"__blazen_session_ref__": "<uuid>"}` markers with the underlying
/// stored value if that value is an `Arc<serde_json::Value>`.
///
/// Markers whose stored value is *not* an `Arc<Value>` (e.g. a future
/// `Arc<napi::Ref<Unknown>>`) are left in place so the caller can
/// resolve them via a more specific downcast.
///
/// This is a tree rewrite, not an in-place mutation — returns a new
/// `Value`. The async traversal is needed because
/// [`SessionRefRegistry::get_any`] takes the internal `RwLock` read
/// guard asynchronously.
pub async fn resolve_session_refs_in_json(value: Value, registry: &SessionRefRegistry) -> Value {
    match value {
        Value::Object(mut map) => {
            // If the whole object is a session-ref marker, try to resolve it.
            if map.len() == 1
                && let Some(tag) = map.get(SESSION_REF_TAG).and_then(Value::as_str)
                && let Ok(key) = RegistryKey::parse(tag)
                && let Some(any_arc) = registry.get_any(key).await
                && let Ok(value_arc) = Arc::downcast::<Value>(any_arc)
            {
                return (*value_arc).clone();
            }
            // Otherwise recurse into every entry.
            for (_k, v) in &mut map {
                let taken = std::mem::take(v);
                *v = Box::pin(resolve_session_refs_in_json(taken, registry)).await;
            }
            Value::Object(map)
        }
        Value::Array(mut arr) => {
            for v in &mut arr {
                let taken = std::mem::take(v);
                *v = Box::pin(resolve_session_refs_in_json(taken, registry)).await;
            }
            Value::Array(arr)
        }
        other => other,
    }
}

/// Walk a [`serde_json::Value`] tree and collect every
/// `{"__blazen_session_ref__": "<uuid>"}` marker whose key exists in
/// the `remote_refs` sidecar of `registry`.
///
/// Returns a list of `(RegistryKey, RemoteRefDescriptor)` pairs for
/// markers that represent values living on a peer node. The caller
/// can use these descriptors to issue `DerefSessionRef` RPCs and
/// materialise the values locally before handing the payload to JS
/// user code.
///
/// Markers that are **not** in the remote sidecar (either locally
/// resolved or simply unknown) are silently skipped.
pub async fn find_remote_refs_in_json(
    value: &Value,
    registry: &SessionRefRegistry,
) -> Vec<(RegistryKey, RemoteRefDescriptor)> {
    let mut out = Vec::new();
    collect_remote_refs(value, registry, &mut out).await;
    out
}

/// Recursive helper — collects remote-ref descriptors into `out`.
async fn collect_remote_refs(
    value: &Value,
    registry: &SessionRefRegistry,
    out: &mut Vec<(RegistryKey, RemoteRefDescriptor)>,
) {
    match value {
        Value::Object(map) => {
            // Check if this object *is* a session-ref marker.
            if map.len() == 1
                && let Some(tag) = map.get(SESSION_REF_TAG).and_then(Value::as_str)
                && let Ok(key) = RegistryKey::parse(tag)
            {
                if let Some(desc) = registry.get_remote(key).await {
                    out.push((key, desc));
                }
                return;
            }
            // Not a marker — recurse into every value.
            for v in map.values() {
                Box::pin(collect_remote_refs(v, registry, out)).await;
            }
        }
        Value::Array(arr) => {
            for v in arr {
                Box::pin(collect_remote_refs(v, registry, out)).await;
            }
        }
        _ => {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn find_session_ref_key_matches_marker_shape() {
        let key = RegistryKey::new();
        let marker = make_session_ref_json(key);
        assert_eq!(find_session_ref_key(&marker), Some(key));
    }

    #[test]
    fn find_session_ref_key_rejects_non_marker() {
        let not_a_marker = serde_json::json!({"other": "field"});
        assert!(find_session_ref_key(&not_a_marker).is_none());

        let extra_fields = serde_json::json!({
            SESSION_REF_TAG: "deadbeef-0000-4000-8000-000000000000",
            "extra": 1,
        });
        assert!(find_session_ref_key(&extra_fields).is_none());

        let wrong_type = serde_json::json!({SESSION_REF_TAG: 42});
        assert!(find_session_ref_key(&wrong_type).is_none());
    }

    #[tokio::test]
    async fn resolve_substitutes_stored_value_arc() {
        let reg = SessionRefRegistry::new();
        let payload = serde_json::json!({"hello": "world"});
        let key = reg.insert_arc(Arc::new(payload.clone())).await.unwrap();
        let marker = make_session_ref_json(key);

        let resolved = resolve_session_refs_in_json(marker, &reg).await;
        assert_eq!(resolved, payload);
    }

    #[tokio::test]
    async fn resolve_walks_nested_structures() {
        let reg = SessionRefRegistry::new();
        let payload = serde_json::json!("the-value");
        let key = reg.insert_arc(Arc::new(payload.clone())).await.unwrap();

        let nested = serde_json::json!({
            "top": [
                {"inner": make_session_ref_json(key)},
                "plain",
            ],
        });

        let resolved = resolve_session_refs_in_json(nested, &reg).await;
        assert_eq!(resolved["top"][0]["inner"], payload);
        assert_eq!(resolved["top"][1], serde_json::json!("plain"));
    }

    #[tokio::test]
    async fn resolve_leaves_unknown_type_refs_in_place() {
        // A ref stored as a non-Value type should be left as its marker.
        let reg = SessionRefRegistry::new();
        let key = reg.insert(42_i32).await.unwrap();
        let marker = make_session_ref_json(key);

        let resolved = resolve_session_refs_in_json(marker.clone(), &reg).await;
        assert_eq!(resolved, marker);
    }

    #[tokio::test]
    async fn find_remote_refs_discovers_remote_markers() {
        let reg = SessionRefRegistry::new();
        let key = RegistryKey::new();
        let desc = RemoteRefDescriptor {
            origin_node_id: "peer-node-1".to_owned(),
            type_tag: "blazen::TestObj".to_owned(),
            created_at_epoch_ms: 1_700_000_000_000,
        };
        reg.insert_remote(key, desc).await.unwrap();

        let payload = serde_json::json!({
            "result": make_session_ref_json(key),
            "plain": 42,
        });

        let found = find_remote_refs_in_json(&payload, &reg).await;
        assert_eq!(found.len(), 1);
        assert_eq!(found[0].0, key);
        assert_eq!(found[0].1.origin_node_id, "peer-node-1");
    }

    #[tokio::test]
    async fn find_remote_refs_skips_local_refs() {
        let reg = SessionRefRegistry::new();
        let local_key = reg
            .insert_arc(Arc::new(serde_json::json!("local-value")))
            .await
            .unwrap();
        let marker = make_session_ref_json(local_key);

        let found = find_remote_refs_in_json(&marker, &reg).await;
        assert!(found.is_empty(), "local-only refs must not appear");
    }

    #[tokio::test]
    async fn find_remote_refs_walks_nested_arrays() {
        let reg = SessionRefRegistry::new();
        let key = RegistryKey::new();
        let desc = RemoteRefDescriptor {
            origin_node_id: "peer-2".to_owned(),
            type_tag: "blazen::Tensor".to_owned(),
            created_at_epoch_ms: 42,
        };
        reg.insert_remote(key, desc).await.unwrap();

        let payload = serde_json::json!([
            {"nested": make_session_ref_json(key)},
            "ignored",
        ]);

        let found = find_remote_refs_in_json(&payload, &reg).await;
        assert_eq!(found.len(), 1);
        assert_eq!(found[0].0, key);
    }
}
