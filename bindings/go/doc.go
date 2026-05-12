// Package blazen provides idiomatic Go bindings for the Blazen LLM
// orchestration framework, wrapping the UniFFI-generated FFI surface from
// the upstream `blazen-uniffi` Rust crate.
//
// # Overview
//
// Blazen is a Rust workflow / agent / pipeline engine for LLM-driven
// applications. The Go package mirrors the same surface area that the
// Python (`blazen-py`) and Node (`blazen-node`) bindings expose, but with
// idiomatic Go ergonomics: typed errors, channels for streaming, plain
// value structs for records, opaque pointer handles with `Close()` and
// finalizers for native resources, and `context.Context` on every async
// call.
//
// # Quick start
//
// Build a workflow with foreign-implemented step handlers:
//
//	type GreetHandler struct{}
//
//	func (g *GreetHandler) Invoke(ctx context.Context, event blazen.Event) (blazen.StepOutput, error) {
//	    var in struct{ Name string `json:"name"` }
//	    _ = json.Unmarshal([]byte(event.DataJSON), &in)
//	    out, _ := json.Marshal(map[string]string{"result": "Hello, " + in.Name})
//	    return blazen.NewStepOutputSingle(blazen.Event{
//	        EventType: "StopEvent",
//	        DataJSON:  string(out),
//	    }), nil
//	}
//
//	wf, _ := blazen.NewWorkflowBuilder("greeter").
//	    Step("greet", []string{"StartEvent"}, []string{"StopEvent"}, &GreetHandler{}).
//	    Build()
//	defer wf.Close()
//
//	result, err := wf.Run(context.Background(), map[string]string{"name": "Zach"})
//
// # Native library
//
// The package links statically against `libblazen_uniffi.a` shipped under
// `internal/clib/<GOOS>_<GOARCH>/`. Today only `linux_amd64` is wired up;
// other targets are added in Phase H of the bindings rollout.
package blazen
