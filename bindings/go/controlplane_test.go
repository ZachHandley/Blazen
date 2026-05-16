package blazen

// Smoke tests for the control-plane UniFFI surface. These do not require
// a running control-plane server — they exercise the type-construction
// path through the generated bindings and confirm the link table picked
// up the new symbols. End-to-end RPC behaviour is covered by the Rust
// integration suite in blazen-controlplane.

import (
	"strings"
	"testing"

	uniffiblazen "github.com/zachhandley/Blazen/bindings/go/internal/uniffi/blazen"
)

// TestControlPlaneRecordTypesConstruct: the generated record types can
// be constructed by the Go code without panicking. The fields exist as
// expected and the discriminated-union enum is well-formed.
func TestControlPlaneRecordTypesConstruct(t *testing.T) {
	cap := uniffiblazen.ControlPlaneWorkerCapability{
		Kind:    "workflow:hello",
		Version: 1,
	}
	if cap.Kind != "workflow:hello" || cap.Version != 1 {
		t.Fatalf("capability fields lost: %+v", cap)
	}

	mode := uniffiblazen.ControlPlaneAdmissionModeFixed
	if mode != uniffiblazen.ControlPlaneAdmissionModeFixed {
		t.Fatal("admission mode constant mismatch")
	}

	maxInFlight := uint32(4)
	admission := uniffiblazen.ControlPlaneAdmission{
		Mode:        mode,
		MaxInFlight: &maxInFlight,
		TotalMb:     nil,
	}
	if admission.MaxInFlight == nil || *admission.MaxInFlight != 4 {
		t.Fatalf("max_in_flight lost: %+v", admission)
	}

	req := uniffiblazen.ControlPlaneSubmitRequest{
		WorkflowName:    "summarize",
		InputJson:       `{"text":"hi"}`,
		WorkflowVersion: nil,
		RequiredTags:    []string{"region=us-west"},
		IdempotencyKey:  nil,
		DeadlineMs:      nil,
		WaitForWorker:   true,
	}
	if req.WorkflowName != "summarize" || !req.WaitForWorker {
		t.Fatalf("submit request fields lost: %+v", req)
	}

	status := uniffiblazen.ControlPlaneRunStatusPending
	if status != uniffiblazen.ControlPlaneRunStatusPending {
		t.Fatal("run status constant mismatch")
	}
}

// TestControlPlaneClientConnectRejectsBadEndpoint: the blocking
// constructor surfaces transport errors as the typed BlazenException /
// ControlPlaneError variant for malformed URIs. Tests the error path
// of the FFI without needing a server.
func TestControlPlaneClientConnectRejectsBadEndpoint(t *testing.T) {
	ensureInit()
	_, err := uniffiblazen.ControlPlaneClientConnectBlocking("not a uri")
	if err == nil {
		t.Fatal("expected bad endpoint to be rejected")
	}
	msg := err.Error()
	if !strings.Contains(strings.ToLower(msg), "transport") &&
		!strings.Contains(strings.ToLower(msg), "invalid") {
		t.Fatalf("expected transport/invalid error, got %q", msg)
	}
}

// TestControlPlaneWorkerConstructValidatesEndpoint: the blocking worker
// constructor rejects a malformed endpoint URI without needing a server
// to connect to.
func TestControlPlaneWorkerConstructValidatesEndpoint(t *testing.T) {
	ensureInit()
	_, err := uniffiblazen.ControlPlaneWorkerNewBlocking(
		"not a uri",
		"node-test",
		[]uniffiblazen.ControlPlaneWorkerCapability{
			{Kind: "workflow:hello", Version: 1},
		},
	)
	if err == nil {
		t.Fatal("expected bad endpoint URI to be rejected")
	}
	if !strings.Contains(strings.ToLower(err.Error()), "transport") &&
		!strings.Contains(strings.ToLower(err.Error()), "invalid") {
		t.Fatalf("expected transport/invalid error, got %q", err.Error())
	}
}
