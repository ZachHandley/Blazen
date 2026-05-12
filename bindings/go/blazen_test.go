package blazen

// Smoke tests for the Blazen Go binding. These exercise the foundational
// FFI link end-to-end: version reporting, idempotent runtime init, and a
// round-trip through a single-step workflow that proves the Rust ->
// Go callback ABI is wired correctly. They live in package blazen so they
// can reach unexported helpers if needed.

import (
	"context"
	"encoding/json"
	"regexp"
	"testing"
	"time"
)

// TestVersion: Version() returns a semver-shaped non-empty string.
func TestVersion(t *testing.T) {
	v := Version()
	if v == "" {
		t.Fatal("Version() returned empty string")
	}
	if !regexp.MustCompile(`^\d+\.\d+\.\d+`).MatchString(v) {
		t.Errorf("Version() = %q; expected semver-shaped prefix", v)
	}
}

// TestInitIdempotent: Init() can be called multiple times without panicking.
func TestInitIdempotent(t *testing.T) {
	Init()
	Init()
	Init()
}

type echoHandler struct{}

func (echoHandler) Invoke(_ context.Context, e Event) (StepOutput, error) {
	// StartEvent serialises as {"data": <user-payload>}; unwrap to get to the message field.
	var envelope struct {
		Data struct {
			Message string `json:"message"`
		} `json:"data"`
	}
	if err := json.Unmarshal([]byte(e.DataJSON), &envelope); err != nil {
		return nil, err
	}
	// StopEvent serialises as {"result": <stop-payload>}; nest the echo under "result".
	out, err := json.Marshal(map[string]any{"result": map[string]string{"echo": envelope.Data.Message}})
	if err != nil {
		return nil, err
	}
	return NewStepOutputSingle(Event{EventType: "blazen::StopEvent", DataJSON: string(out)}), nil
}

// TestEchoWorkflow: a one-step workflow round-trips a JSON payload end-to-end.
func TestEchoWorkflow(t *testing.T) {
	builder, err := NewWorkflowBuilder("echo-test").
		Step("echo", []string{"blazen::StartEvent"}, []string{"blazen::StopEvent"}, echoHandler{})
	if err != nil {
		t.Fatalf("Step registration failed: %v", err)
	}
	wf, err := builder.Build()
	if err != nil {
		t.Fatalf("Build failed: %v", err)
	}
	defer wf.Close()

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	result, err := wf.Run(ctx, map[string]string{"message": "hello"})
	if err != nil {
		t.Fatalf("Run failed: %v", err)
	}
	if result.Event.EventType != "blazen::StopEvent" {
		t.Errorf("expected blazen::StopEvent, got %q", result.Event.EventType)
	}
	// StopEvent wraps the handler's emitted payload under a "result" key
	// (matches the Rust StopEvent { result: <payload> } shape).
	var wrapper struct {
		Result map[string]string `json:"result"`
	}
	if err := json.Unmarshal([]byte(result.Event.DataJSON), &wrapper); err != nil {
		t.Fatalf("decode result: %v", err)
	}
	if wrapper.Result["echo"] != "hello" {
		t.Errorf("expected echo=hello, got %q (wrapper=%+v)", wrapper.Result["echo"], wrapper)
	}
}
