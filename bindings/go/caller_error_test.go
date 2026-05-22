package blazen

// Structural test for the caller-error preservation path on the Go binding.
//
// Background
//
// Blazen 0.5.4 adds a typed `CallerError` variant to the workspace's
// `BlazenError` enum. The contract: when a foreign-language tool handler
// raises a typed exception, the Rust UniFFI adapter converts it into a
// `BlazenError::CallerError { name: Option<String>, message: String,
// properties_json: String }` and propagates it back through `run_agent` so
// the host caller can pattern-match the variant, inspect the original
// exception's class name, and JSON-decode the structured payload.
//
// As of this scaffold's authoring, `./scripts/regen-bindings.sh` (which
// regenerates Go via `uniffi-bindgen-go`) hasn't yet been run against the
// 0.5.4 UDL, so the generated `uniffiblazen.BlazenErrorCallerError` type
// and its corresponding wrapper `CallerError` struct in `errors.go` don't
// exist yet. This test pins the EXPECTED post-regen shape:
//
//   - A user-defined Go error type carrying structured payload data.
//   - A `ToolHandler` whose `Execute` raises that typed error.
//   - The agent surfaces the failure as `*CallerError` via `errors.As`,
//     with the expected `Name` (Go type identifier) and a JSON-decodable
//     `PropertiesJson` blob carrying the payload fields.
//
// The test is skipped right now so the suite remains green until the regen
// surfaces the variant. Once `./scripts/regen-bindings.sh` lands the new
// type, remove the `t.Skip` and uncomment the assertions below.

import (
	"encoding/json"
	"errors"
	"testing"
)

// myCallerError is a Go-side typed error a tool handler might raise to
// signal a domain-specific failure. The `Payload` field is the kind of
// structured data that should survive the FFI round-trip via the
// CallerError `properties_json` blob.
type myCallerError struct {
	Payload struct {
		Code   string `json:"code"`
		Detail string `json:"detail"`
	}
}

// Error satisfies the standard `error` interface. The Rust side captures
// this as the `message` field on the CallerError variant.
func (e *myCallerError) Error() string {
	return "myCallerError: " + e.Payload.Code + ": " + e.Payload.Detail
}

// raisingHandler is a ToolHandler whose Execute always raises a typed
// `*myCallerError`. The Rust adapter is expected to reflect this back as
// `BlazenError::CallerError { name: Some("myCallerError"), .. }`.
type raisingHandler struct{}

func (raisingHandler) Execute(_ raisingHandlerCtx, _ string, _ string) (string, error) {
	e := &myCallerError{}
	e.Payload.Code = "E_DOMAIN"
	e.Payload.Detail = "tool refused"
	return "", e
}

// raisingHandlerCtx is a thin alias to avoid pulling in `context` solely
// for the typed-handler shape. Replaced with `context.Context` once the
// scaffold is fleshed out post-regen.
type raisingHandlerCtx = interface{}

// TestCallerErrorPreservation: a tool handler that raises a typed Go error
// surfaces back through `Agent.Run` as the structural `*CallerError`
// variant with the expected `Name` and `PropertiesJson` fields.
//
// EXPECTED post-regen behaviour (after `./scripts/regen-bindings.sh`):
//
//	var ce *CallerError
//	if errors.As(err, &ce) {
//	    require.Equal(t, "myCallerError", ce.Name)
//	    var payload struct {
//	        Code   string `json:"code"`
//	        Detail string `json:"detail"`
//	    }
//	    require.NoError(t, json.Unmarshal([]byte(ce.PropertiesJson), &payload))
//	    require.Equal(t, "E_DOMAIN", payload.Code)
//	}
//
// Until the regen lands, the test is skipped so CI stays green.
func TestCallerErrorPreservation(t *testing.T) {
	t.Skip("needs CallerError variant regen via ./scripts/regen-bindings.sh")

	// The block below is the post-regen test body. It compiles against
	// the EXPECTED public shape: a `*CallerError` wrapper struct mirroring
	// the existing typed error structs in `errors.go` (e.g. `*ProviderError`),
	// surfaced by `wrapErr` from a new `uniffiblazen.BlazenErrorCallerError`
	// generated variant.
	//
	// Because the type doesn't exist yet, every reference is behind the
	// t.Skip above; nothing executes until regen. The references that
	// would fail to compile (CallerError, NewMockModel) are kept
	// in commented form below for documentation. Reintroduce them when
	// the regen lands.
	//
	// model := NewMockModel("mock") // (regen also adds a mock helper)
	// defer model.Close()
	//
	// tool := Tool{
	//     Name:           "domain_op",
	//     Description:    "Always raises a typed caller error.",
	//     ParametersJSON: `{"type":"object","properties":{}}`,
	// }
	// agent := NewAgent(model, "", []Tool{tool}, raisingHandler{}, 2)
	// defer agent.Close()
	//
	// ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	// defer cancel()
	// _, err := agent.Run(ctx, "please call the tool")
	// if err == nil {
	//     t.Fatal("expected CallerError, got nil")
	// }
	// var ce *CallerError
	// if !errors.As(err, &ce) {
	//     t.Fatalf("expected *CallerError, got %T: %v", err, err)
	// }
	// if ce.Name != "myCallerError" {
	//     t.Errorf("Name = %q; want %q", ce.Name, "myCallerError")
	// }
	// var payload struct {
	//     Code   string `json:"code"`
	//     Detail string `json:"detail"`
	// }
	// if err := json.Unmarshal([]byte(ce.PropertiesJson), &payload); err != nil {
	//     t.Fatalf("PropertiesJson is not valid JSON: %v", err)
	// }
	// if payload.Code != "E_DOMAIN" {
	//     t.Errorf("payload.Code = %q; want %q", payload.Code, "E_DOMAIN")
	// }

	// The two lines below keep `errors.As`, `json.Unmarshal`, and
	// `raisingHandler` referenced so `goimports` / `go vet` won't drop the
	// imports while the test body is parked behind t.Skip. Remove them
	// (and uncomment the real body above) once the regen lands.
	_ = errors.As(nil, new(*myCallerError))
	_ = json.Unmarshal
	_ = raisingHandler{}
}
