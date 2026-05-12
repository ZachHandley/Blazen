// Package main is a runnable hello-world example for the Blazen Go
// bindings.
//
// It builds the minimal possible workflow — a single `greet` step that
// pulls a `name` field out of the incoming StartEvent and emits a
// StopEvent carrying a friendly greeting — runs it once with a fixed
// input, prints the terminal payload, and shuts the runtime down. The
// example mirrors the equivalent Swift / Kotlin / Ruby "hello workflow"
// demos so users can compare bindings side-by-side.
//
// Run it from the bindings/go directory:
//
//	go run ./examples/hello_workflow
//
// The native library is linked in via cgo from
// `bindings/go/internal/clib/<GOOS>_<GOARCH>/libblazen_uniffi.a`, so no
// additional environment setup is required on supported targets.
package main

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"time"

	blazen "github.com/zachhandley/Blazen/bindings/go"
)

// GreetHandler is a [blazen.StepHandler] that reads `{"name": "..."}`
// from the incoming event payload and emits a StopEvent whose payload is
// `{"result": "Hello, <name>!"}`.
//
// The handler is intentionally synchronous and allocation-light: it is
// the smallest possible foreign-implemented step useful for sanity-
// checking the install and the FFI boundary.
type GreetHandler struct{}

// greetInput describes the StartEvent payload the handler consumes.
type greetInput struct {
	// Name is the value the greeting is interpolated with. When the field
	// is missing or empty the handler falls back to "world" so the
	// example never produces an empty greeting.
	Name string `json:"name"`
}

// greetOutput describes the StopEvent payload the handler produces.
type greetOutput struct {
	// Result is the rendered greeting string.
	Result string `json:"result"`
}

// Invoke implements [blazen.StepHandler]. It is invoked by the workflow
// engine each time an event whose type is listed in the step's `accepts`
// set arrives (here: just `StartEvent`). Any error returned here is
// translated into a [blazen.ValidationError] on the Rust side so the
// host workflow can surface it via [blazen.Workflow.Run].
func (GreetHandler) Invoke(_ context.Context, event blazen.Event) (blazen.StepOutput, error) {
	var in greetInput
	if err := json.Unmarshal([]byte(event.DataJSON), &in); err != nil {
		return nil, fmt.Errorf("decode StartEvent payload: %w", err)
	}
	name := in.Name
	if name == "" {
		name = "world"
	}

	payload, err := json.Marshal(greetOutput{Result: "Hello, " + name + "!"})
	if err != nil {
		return nil, fmt.Errorf("encode StopEvent payload: %w", err)
	}

	return blazen.NewStepOutputSingle(blazen.Event{
		EventType: "StopEvent",
		DataJSON:  string(payload),
	}), nil
}

// main wires the workflow, runs it, and prints the terminal event.
//
// The exit code is 0 on success and 1 on any error; errors are
// classified through the typed-error hierarchy so users can see the
// shape of error handling the Go binding expects.
func main() {
	// Eagerly initialise the native runtime. The builder will lazily call
	// this on first use too, but calling it explicitly here costs nothing
	// and gives users a single, obvious place to look when diagnosing
	// startup problems.
	blazen.Init()
	defer blazen.Shutdown()

	fmt.Printf("blazen version: %s\n", blazen.Version())

	wf, err := buildWorkflow()
	if err != nil {
		exitWithClassifiedError("build workflow", err)
	}
	defer wf.Close()

	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	result, err := wf.Run(ctx, map[string]string{"name": "Zach"})
	if err != nil {
		exitWithClassifiedError("run workflow", err)
	}

	fmt.Printf("event:    %s\n", result.Event.EventType)
	fmt.Printf("payload:  %s\n", result.Event.DataJSON)
	fmt.Printf("tokens:   in=%d out=%d\n", result.TotalInputTokens, result.TotalOutputTokens)
	fmt.Printf("cost USD: %.6f\n", result.TotalCostUSD)
}

// buildWorkflow assembles the one-step "greeter" workflow.
//
// Splitting this out of [main] keeps the error-handling at the call site
// linear and lets the example illustrate the canonical builder pattern
// (each [blazen.WorkflowBuilder.Step] call returns an updated builder
// plus an error that must be checked before continuing).
func buildWorkflow() (*blazen.Workflow, error) {
	builder := blazen.NewWorkflowBuilder("greeter")
	builder, err := builder.Step(
		"greet",
		[]string{"StartEvent"},
		[]string{"StopEvent"},
		GreetHandler{},
	)
	if err != nil {
		return nil, fmt.Errorf("register greet step: %w", err)
	}
	wf, err := builder.Build()
	if err != nil {
		return nil, fmt.Errorf("build greeter workflow: %w", err)
	}
	return wf, nil
}

// exitWithClassifiedError prints a diagnostic that classifies err
// against the [blazen.Error] hierarchy and exits the process with status
// 1.
//
// The happy path of this example never reaches this function — it
// exists to show callers how to disambiguate the typed errors the Go
// binding returns. In a real application you would dispatch on these
// variants to decide whether to retry, surface a user-facing message,
// or escalate.
func exitWithClassifiedError(stage string, err error) {
	kind := classifyError(err)
	fmt.Fprintf(os.Stderr, "%s failed [%s]: %v\n", stage, kind, err)
	os.Exit(1)
}

// classifyError walks the typed-error variants the Go binding can
// return and reports the first one that matches. The order mirrors how
// frequently each variant shows up in practice — most workflow failures
// are validation or workflow errors, network/provider failures come
// next, and the catch-alls (cancelled / internal) sit at the bottom.
func classifyError(err error) string {
	var (
		authErr     *blazen.AuthError
		rateErr     *blazen.RateLimitError
		timeoutErr  *blazen.TimeoutError
		validErr    *blazen.ValidationError
		policyErr   *blazen.ContentPolicyError
		unsuppErr   *blazen.UnsupportedError
		computeErr  *blazen.ComputeError
		mediaErr    *blazen.MediaError
		providerErr *blazen.ProviderError
		wfErr       *blazen.WorkflowError
		toolErr     *blazen.ToolError
		peerErr     *blazen.PeerError
		persistErr  *blazen.PersistError
		promptErr   *blazen.PromptError
		memoryErr   *blazen.MemoryError
		cacheErr    *blazen.CacheError
		cancelErr   *blazen.CancelledError
		internalErr *blazen.InternalError
	)
	switch {
	case errors.As(err, &validErr):
		return "ValidationError"
	case errors.As(err, &wfErr):
		return "WorkflowError"
	case errors.As(err, &authErr):
		return "AuthError"
	case errors.As(err, &rateErr):
		return fmt.Sprintf("RateLimitError(retry_after_ms=%d)", rateErr.RetryAfterMs)
	case errors.As(err, &timeoutErr):
		return fmt.Sprintf("TimeoutError(elapsed_ms=%d)", timeoutErr.ElapsedMs)
	case errors.As(err, &providerErr):
		return fmt.Sprintf("ProviderError(kind=%s)", providerErr.Kind)
	case errors.As(err, &policyErr):
		return "ContentPolicyError"
	case errors.As(err, &unsuppErr):
		return "UnsupportedError"
	case errors.As(err, &computeErr):
		return "ComputeError"
	case errors.As(err, &mediaErr):
		return "MediaError"
	case errors.As(err, &toolErr):
		return "ToolError"
	case errors.As(err, &peerErr):
		return fmt.Sprintf("PeerError(kind=%s)", peerErr.Kind)
	case errors.As(err, &persistErr):
		return "PersistError"
	case errors.As(err, &promptErr):
		return fmt.Sprintf("PromptError(kind=%s)", promptErr.Kind)
	case errors.As(err, &memoryErr):
		return fmt.Sprintf("MemoryError(kind=%s)", memoryErr.Kind)
	case errors.As(err, &cacheErr):
		return fmt.Sprintf("CacheError(kind=%s)", cacheErr.Kind)
	case errors.As(err, &cancelErr):
		return "CancelledError"
	case errors.As(err, &internalErr):
		return "InternalError"
	case errors.Is(err, context.Canceled):
		return "context.Canceled"
	case errors.Is(err, context.DeadlineExceeded):
		return "context.DeadlineExceeded"
	default:
		return "untyped"
	}
}
