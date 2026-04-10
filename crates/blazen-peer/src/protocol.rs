//! Postcard-serializable wire types for the Blazen peer protocol.
//!
//! Every gRPC method on the [`crate::pb`] service takes and returns a
//! single `bytes` field whose contents are one of the structs defined
//! in this module, encoded with [`postcard`]. Versioning is handled
//! per-message by the [`ENVELOPE_VERSION`] field rather than by
//! evolving the proto schema, so adding a new field here is a
//! source-only change for both client and server.
//!
//! ## A note on `serde_json::Value`
//!
//! Postcard is a non-self-describing format and cannot round-trip
//! `serde_json::Value` directly — its untagged enum requires
//! `deserialize_any`, which postcard explicitly does not implement.
//! To stay compatible we carry JSON payloads as `Vec<u8>` of
//! pre-serialized JSON text. Helper constructors
//! ([`SubWorkflowRequest::new`],
//! [`SubWorkflowResponse::input_value`], etc.) take and return
//! `serde_json::Value` for ergonomics, doing the JSON encode / decode
//! at the boundary so callers never have to think about the byte
//! representation.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// Current envelope version. Bump this whenever you change the shape
/// of any struct in this module in a way that is not forward-compatible.
///
/// The convention is:
///
/// - **Adding a new optional field at the end** of a struct is
///   forward-compatible — postcard will skip unknown trailing bytes on
///   decode and `Option::None` for missing fields. No version bump.
/// - **Renaming, reordering, or removing fields** is *not*
///   forward-compatible. Bump this constant and update
///   [`crate::error::PeerError::EnvelopeVersion`] handling on the
///   server side.
pub const ENVELOPE_VERSION: u32 = 1;

/// Request to invoke a sub-workflow on a remote peer.
///
/// The receiving peer looks up `workflow_name` in its local
/// [`blazen_core::step_registry`] (or its workflow catalog), feeds
/// `input_json` into the entry step, and runs the listed `step_ids`
/// to completion. The `timeout_secs` field bounds the total
/// wall-clock time the server is willing to spend on this request.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubWorkflowRequest {
    /// Envelope version of this payload. See [`ENVELOPE_VERSION`].
    pub envelope_version: u32,
    /// Symbolic name of the workflow to invoke on the remote peer.
    pub workflow_name: String,
    /// Ordered list of step IDs to execute as part of this sub-workflow.
    /// Empty means "run the workflow's default step set".
    pub step_ids: Vec<String>,
    /// Initial input value passed to the workflow's first step,
    /// carried as JSON-encoded bytes. See the module-level docs for
    /// why we use `Vec<u8>` here instead of `serde_json::Value`. Use
    /// [`Self::new`] / [`Self::input_value`] for the ergonomic
    /// `serde_json::Value` interface.
    #[serde(with = "serde_bytes")]
    pub input_json: Vec<u8>,
    /// Optional timeout in seconds. `None` means "use the server's
    /// default deadline".
    pub timeout_secs: Option<u64>,
}

impl SubWorkflowRequest {
    /// Build a [`SubWorkflowRequest`] from a `serde_json::Value`,
    /// handling the JSON encode for the caller.
    ///
    /// # Errors
    /// Returns [`serde_json::Error`] if `input` cannot be serialized
    /// to JSON. In practice this only happens for non-finite floats
    /// or other inputs that already violate JSON's value model.
    pub fn new(
        workflow_name: impl Into<String>,
        step_ids: Vec<String>,
        input: &serde_json::Value,
        timeout_secs: Option<u64>,
    ) -> Result<Self, serde_json::Error> {
        Ok(Self {
            envelope_version: ENVELOPE_VERSION,
            workflow_name: workflow_name.into(),
            step_ids,
            input_json: serde_json::to_vec(input)?,
            timeout_secs,
        })
    }

    /// Decode the inner `input_json` field back into a
    /// `serde_json::Value`.
    ///
    /// # Errors
    /// Returns [`serde_json::Error`] if `input_json` does not contain
    /// valid JSON. This should never happen for payloads produced by
    /// [`Self::new`].
    pub fn input_value(&self) -> Result<serde_json::Value, serde_json::Error> {
        serde_json::from_slice(&self.input_json)
    }
}

/// Result of a remote sub-workflow invocation.
///
/// The `state_json` map mirrors the parent workflow's
/// [`blazen_core::Context`] state values that the sub-workflow chose
/// to expose, with each value carried as JSON-encoded bytes (see the
/// module-level docs for why). `result_json` is the optional terminal
/// value (if any), and `remote_refs` describes session refs that the
/// sub-workflow created but could not serialize — the parent should
/// treat each entry as a proxy handle and use [`DerefRequest`] /
/// [`ReleaseRequest`] to interact with them.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubWorkflowResponse {
    /// Envelope version of this payload. See [`ENVELOPE_VERSION`].
    pub envelope_version: u32,
    /// Public state values exported by the sub-workflow, each
    /// carried as JSON-encoded bytes. Use [`Self::state_values`] for
    /// the ergonomic `serde_json::Value` interface.
    pub state_json: HashMap<String, Vec<u8>>,
    /// Optional terminal result, JSON-encoded. `None` when the
    /// workflow exited without producing one.
    #[serde(with = "serde_bytes_option")]
    pub result_json: Option<Vec<u8>>,
    /// Descriptors for any session refs the sub-workflow registered
    /// that the parent should be able to dereference remotely.
    pub remote_refs: HashMap<Uuid, RemoteRefDescriptor>,
    /// Error message if the sub-workflow failed. When `Some`, callers
    /// should ignore `result_json` and `state_json`.
    pub error: Option<String>,
}

impl SubWorkflowResponse {
    /// Build a successful response from a `serde_json::Value` result
    /// and a state map keyed by `String`. Performs the JSON encode
    /// for every value.
    ///
    /// # Errors
    /// Returns [`serde_json::Error`] if any value cannot be encoded
    /// as JSON.
    pub fn ok(
        state: &HashMap<String, serde_json::Value>,
        result: Option<&serde_json::Value>,
        remote_refs: HashMap<Uuid, RemoteRefDescriptor>,
    ) -> Result<Self, serde_json::Error> {
        let mut state_json = HashMap::with_capacity(state.len());
        for (k, v) in state {
            state_json.insert(k.clone(), serde_json::to_vec(v)?);
        }
        let result_json = match result {
            Some(v) => Some(serde_json::to_vec(v)?),
            None => None,
        };
        Ok(Self {
            envelope_version: ENVELOPE_VERSION,
            state_json,
            result_json,
            remote_refs,
            error: None,
        })
    }

    /// Build an error response carrying just an error message and an
    /// otherwise empty payload.
    #[must_use]
    pub fn err(message: impl Into<String>) -> Self {
        Self {
            envelope_version: ENVELOPE_VERSION,
            state_json: HashMap::new(),
            result_json: None,
            remote_refs: HashMap::new(),
            error: Some(message.into()),
        }
    }

    /// Decode the entire `state_json` map back into
    /// `serde_json::Value` form.
    ///
    /// # Errors
    /// Returns [`serde_json::Error`] if any entry does not contain
    /// valid JSON.
    pub fn state_values(&self) -> Result<HashMap<String, serde_json::Value>, serde_json::Error> {
        let mut out = HashMap::with_capacity(self.state_json.len());
        for (k, bytes) in &self.state_json {
            out.insert(k.clone(), serde_json::from_slice(bytes)?);
        }
        Ok(out)
    }

    /// Decode the optional `result_json` payload into a
    /// `serde_json::Value`.
    ///
    /// # Errors
    /// Returns [`serde_json::Error`] if `result_json` is `Some` but
    /// does not contain valid JSON.
    pub fn result_value(&self) -> Result<Option<serde_json::Value>, serde_json::Error> {
        match &self.result_json {
            Some(bytes) => Ok(Some(serde_json::from_slice(bytes)?)),
            None => Ok(None),
        }
    }
}

/// Helper module for `#[serde(with = "...")]` on `Option<Vec<u8>>`
/// fields. Wraps the inner `Vec<u8>` in [`serde_bytes::ByteBuf`] so
/// the value is encoded as a length-prefixed byte string instead of
/// a sequence of individual `u8` items. This matches the convention
/// [`SubWorkflowRequest::input_json`] uses for symmetry and keeps the
/// wire format compact under postcard.
mod serde_bytes_option {
    use serde::{Deserialize, Deserializer, Serialize, Serializer};

    #[derive(Serialize, Deserialize)]
    #[serde(transparent)]
    struct Wrapper(serde_bytes::ByteBuf);

    #[allow(clippy::ref_option)]
    pub fn serialize<S>(value: &Option<Vec<u8>>, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        // We must let the serializer drive the Some/None encoding,
        // otherwise non-self-describing formats (postcard) end up with
        // a tag mismatch on decode. Wrap in `ByteBuf` purely to switch
        // the inner serializer from "sequence of u8" to "byte string".
        //
        // The signature must be `&Option<T>` because serde's `with`
        // attribute passes a reference to the field, not the inner
        // value. Clippy's `ref_option` lint does not apply here.
        value
            .as_ref()
            .map(|bytes| Wrapper(serde_bytes::ByteBuf::from(bytes.clone())))
            .serialize(serializer)
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<Option<Vec<u8>>, D::Error>
    where
        D: Deserializer<'de>,
    {
        Option::<Wrapper>::deserialize(deserializer)
            .map(|opt| opt.map(|wrapper| wrapper.0.into_vec()))
    }
}

/// Metadata describing a remote session ref.
///
/// A `RemoteRefDescriptor` is a small, owned, serializable handle that
/// the parent process holds onto as a stand-in for an opaque value
/// living on a different node. To actually read the underlying value
/// the parent issues a [`DerefRequest`] over gRPC; to drop it, a
/// [`ReleaseRequest`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RemoteRefDescriptor {
    /// Stable identifier of the node that owns the underlying value.
    pub origin_node_id: String,
    /// Type tag mirroring
    /// [`blazen_core::SessionRefSerializable::blazen_type_tag`]. Used
    /// by the parent's deserializer registry to figure out how to
    /// rehydrate the bytes returned by [`DerefResponse`].
    pub type_tag: String,
    /// Wall-clock creation time on the origin node, in milliseconds
    /// since the Unix epoch. Useful for tracing and TTL bookkeeping.
    pub created_at_epoch_ms: u64,
}

/// Request to dereference a remote session ref.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DerefRequest {
    /// Envelope version of this payload. See [`ENVELOPE_VERSION`].
    pub envelope_version: u32,
    /// UUID of the registry entry on the origin node, taken from a
    /// [`RemoteRefDescriptor`] returned by an earlier
    /// [`SubWorkflowResponse`].
    pub ref_uuid: Uuid,
}

/// Response containing the dereferenced bytes for a session ref.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DerefResponse {
    /// Envelope version of this payload. See [`ENVELOPE_VERSION`].
    pub envelope_version: u32,
    /// Raw payload returned by
    /// [`blazen_core::SessionRefSerializable::blazen_serialize`] on
    /// the origin node. The parent decodes this with the deserializer
    /// keyed by [`RemoteRefDescriptor::type_tag`].
    #[serde(with = "serde_bytes")]
    pub payload: Vec<u8>,
}

/// Request to release (drop) a remote session ref.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReleaseRequest {
    /// Envelope version of this payload. See [`ENVELOPE_VERSION`].
    pub envelope_version: u32,
    /// UUID of the registry entry to drop on the origin node.
    pub ref_uuid: Uuid,
}

/// Acknowledgement for a [`ReleaseRequest`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReleaseResponse {
    /// Envelope version of this payload. See [`ENVELOPE_VERSION`].
    pub envelope_version: u32,
    /// `true` if the registry entry was found and dropped, `false` if
    /// it was already gone (e.g. expired by lifetime policy).
    pub released: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn roundtrip<T>(value: &T) -> T
    where
        T: Serialize + for<'de> Deserialize<'de>,
    {
        let bytes = postcard::to_allocvec(value).expect("postcard encode");
        postcard::from_bytes(&bytes).expect("postcard decode")
    }

    #[test]
    fn sub_workflow_request_roundtrips() {
        let input = serde_json::json!({ "url": "https://example.com" });
        let original = SubWorkflowRequest::new(
            "summarize",
            vec!["fetch".to_string(), "summarize".to_string()],
            &input,
            Some(30),
        )
        .expect("encode input");

        let decoded = roundtrip(&original);
        assert_eq!(decoded.envelope_version, original.envelope_version);
        assert_eq!(decoded.workflow_name, original.workflow_name);
        assert_eq!(decoded.step_ids, original.step_ids);
        assert_eq!(decoded.timeout_secs, original.timeout_secs);
        assert_eq!(decoded.input_value().expect("decode input"), input);
    }

    #[test]
    fn sub_workflow_request_helper_construction() {
        let input = serde_json::json!({ "id": 7 });
        let req = SubWorkflowRequest::new("noop", vec![], &input, None).unwrap();
        assert_eq!(req.envelope_version, ENVELOPE_VERSION);
        assert!(req.step_ids.is_empty());
        assert!(req.timeout_secs.is_none());
        assert_eq!(req.input_value().unwrap(), input);
    }

    #[test]
    fn sub_workflow_response_roundtrips() {
        let mut state = HashMap::new();
        state.insert("count".to_string(), serde_json::json!(42));
        state.insert(
            "labels".to_string(),
            serde_json::json!(["alpha", "beta", "gamma"]),
        );

        let mut remote_refs = HashMap::new();
        let ref_uuid = Uuid::new_v4();
        remote_refs.insert(
            ref_uuid,
            RemoteRefDescriptor {
                origin_node_id: "node-a".to_string(),
                type_tag: "FileHandle".to_string(),
                created_at_epoch_ms: 1_700_000_000_000,
            },
        );

        let result = serde_json::json!("ok");
        let original = SubWorkflowResponse::ok(&state, Some(&result), remote_refs).unwrap();

        let decoded = roundtrip(&original);
        assert_eq!(decoded.envelope_version, original.envelope_version);
        assert_eq!(decoded.state_values().unwrap(), state);
        assert_eq!(decoded.result_value().unwrap(), Some(result));
        assert_eq!(decoded.remote_refs.len(), 1);
        let descriptor = decoded
            .remote_refs
            .get(&ref_uuid)
            .expect("remote ref present");
        assert_eq!(descriptor.origin_node_id, "node-a");
        assert_eq!(descriptor.type_tag, "FileHandle");
        assert_eq!(descriptor.created_at_epoch_ms, 1_700_000_000_000);
        assert!(decoded.error.is_none());
    }

    #[test]
    fn sub_workflow_response_error_roundtrips() {
        let original = SubWorkflowResponse::err("workflow exploded");
        let decoded = roundtrip(&original);
        assert_eq!(decoded.error.as_deref(), Some("workflow exploded"));
        assert!(decoded.result_json.is_none());
        assert!(decoded.state_json.is_empty());
        assert!(decoded.remote_refs.is_empty());
        assert!(decoded.result_value().unwrap().is_none());
        assert!(decoded.state_values().unwrap().is_empty());
    }

    #[test]
    fn remote_ref_descriptor_roundtrips() {
        let original = RemoteRefDescriptor {
            origin_node_id: "node-b".to_string(),
            type_tag: "ModelWeights".to_string(),
            created_at_epoch_ms: 1_234_567_890,
        };
        let decoded = roundtrip(&original);
        assert_eq!(decoded.origin_node_id, original.origin_node_id);
        assert_eq!(decoded.type_tag, original.type_tag);
        assert_eq!(decoded.created_at_epoch_ms, original.created_at_epoch_ms);
    }

    #[test]
    fn deref_request_roundtrips() {
        let original = DerefRequest {
            envelope_version: ENVELOPE_VERSION,
            ref_uuid: Uuid::new_v4(),
        };
        let decoded = roundtrip(&original);
        assert_eq!(decoded.envelope_version, original.envelope_version);
        assert_eq!(decoded.ref_uuid, original.ref_uuid);
    }

    #[test]
    fn deref_response_roundtrips() {
        let original = DerefResponse {
            envelope_version: ENVELOPE_VERSION,
            payload: vec![1, 2, 3, 4, 5],
        };
        let decoded = roundtrip(&original);
        assert_eq!(decoded.envelope_version, original.envelope_version);
        assert_eq!(decoded.payload, original.payload);
    }

    #[test]
    fn release_request_roundtrips() {
        let original = ReleaseRequest {
            envelope_version: ENVELOPE_VERSION,
            ref_uuid: Uuid::new_v4(),
        };
        let decoded = roundtrip(&original);
        assert_eq!(decoded.envelope_version, original.envelope_version);
        assert_eq!(decoded.ref_uuid, original.ref_uuid);
    }

    #[test]
    fn release_response_roundtrips() {
        let original = ReleaseResponse {
            envelope_version: ENVELOPE_VERSION,
            released: true,
        };
        let decoded = roundtrip(&original);
        assert_eq!(decoded.envelope_version, original.envelope_version);
        assert!(decoded.released);
    }
}
