//! Envelope v1↔v2 forward-compatibility for the B2 wire additions.
//!
//! `ENVELOPE_VERSION` went 1→2 with `ServerToWorker::InputResponse`
//! *appended* (after `Reject`). Postcard encodes an enum's variant as a
//! varint discriminant index, so appending a variant must not shift the
//! indices of the existing ones — an "old" payload (any pre-existing
//! variant) must still decode unchanged, and the new variant must
//! round-trip.

use blazen_controlplane::protocol::{
    CancelInstruction, ENVELOPE_VERSION, InputResponse, RespondToInputRequest, ServerToWorker,
};
use uuid::Uuid;

#[test]
fn input_response_variant_round_trips() {
    let run_id = Uuid::new_v4();
    let frame = ServerToWorker::InputResponse(InputResponse {
        envelope_version: ENVELOPE_VERSION,
        run_id,
        request_id: "req-123".into(),
        response_json: b"{\"approved\":true}".to_vec(),
    });
    let bytes = postcard::to_allocvec(&frame).expect("encode");
    let decoded: ServerToWorker = postcard::from_bytes(&bytes).expect("decode");
    match decoded {
        ServerToWorker::InputResponse(r) => {
            assert_eq!(r.run_id, run_id);
            assert_eq!(r.request_id, "req-123");
            assert_eq!(r.response_json, b"{\"approved\":true}");
        }
        other => panic!("expected InputResponse, got {other:?}"),
    }
}

#[test]
fn existing_variant_indices_unchanged() {
    // A `Cancel` frame is an existing (pre-v2) variant. Appending
    // `InputResponse` after `Reject` must leave `Cancel`'s discriminant
    // index untouched, so this still decodes as `Cancel`.
    let run_id = Uuid::new_v4();
    let frame = ServerToWorker::Cancel(CancelInstruction {
        envelope_version: ENVELOPE_VERSION,
        run_id,
    });
    let bytes = postcard::to_allocvec(&frame).expect("encode");
    let decoded: ServerToWorker = postcard::from_bytes(&bytes).expect("decode");
    match decoded {
        ServerToWorker::Cancel(c) => assert_eq!(c.run_id, run_id),
        other => panic!("expected Cancel, got {other:?}"),
    }
}

#[test]
fn respond_to_input_request_round_trips() {
    let run_id = Uuid::new_v4();
    let req = RespondToInputRequest {
        envelope_version: ENVELOPE_VERSION,
        run_id,
        request_id: "req-abc".into(),
        response_json: b"\"hello\"".to_vec(),
    };
    let bytes = postcard::to_allocvec(&req).expect("encode");
    let decoded: RespondToInputRequest = postcard::from_bytes(&bytes).expect("decode");
    assert_eq!(decoded.run_id, run_id);
    assert_eq!(decoded.request_id, "req-abc");
    assert_eq!(decoded.response_json, b"\"hello\"");
}
