package blazen

import (
	"context"
	"encoding/json"
	"fmt"
	"runtime"
	"sync"
	"time"

	uniffiblazen "github.com/zachhandley/Blazen/bindings/go/internal/uniffi/blazen"
)

// Event is a single message crossing the FFI boundary between the
// workflow engine and a foreign step handler.
//
// EventType is a free-form class name (e.g. "StartEvent", "StopEvent",
// "MyCustomEvent"). DataJSON is the JSON-encoded payload — handlers are
// expected to unmarshal it into a domain-specific Go type.
//
// The JSON tags use snake_case so the struct round-trips cleanly through
// JSON pipelines that interoperate with the other Blazen bindings.
type Event struct {
	EventType string `json:"event_type"`
	DataJSON  string `json:"data_json"`
}

// toFFI converts the Go-facing Event to the generated FFI record.
//
// The generated bindgen named the second field `DataJson` (PascalCase Json,
// not JSON); this converter is the single place that knows about that
// naming quirk.
func (e Event) toFFI() uniffiblazen.Event {
	return uniffiblazen.Event{
		EventType: e.EventType,
		DataJson:  e.DataJSON,
	}
}

// eventFromFFI converts a generated FFI Event into the package-public
// form. See [Event.toFFI] for the naming rationale.
func eventFromFFI(e uniffiblazen.Event) Event {
	return Event{
		EventType: e.EventType,
		DataJSON:  e.DataJson,
	}
}

// StepOutput is the sum type returned by a [StepHandler]: zero, one, or
// many events to publish back into the workflow's event queue.
//
// Construct values via [NewStepOutputNone], [NewStepOutputSingle], or
// [NewStepOutputMultiple] rather than implementing this interface in user
// code — the interface is sealed so the wrapper can guarantee a closed set
// of variants for FFI conversion.
type StepOutput interface {
	stepOutput()
}

// StepOutputNone signals that the step performed work but produced no
// downstream event.
type StepOutputNone struct{}

func (StepOutputNone) stepOutput() {}

// StepOutputSingle is the common case: the step produced exactly one
// event.
type StepOutputSingle struct {
	Event Event
}

func (StepOutputSingle) stepOutput() {}

// StepOutputMultiple represents a fan-out: the step produced several
// events at once.
type StepOutputMultiple struct {
	Events []Event
}

func (StepOutputMultiple) stepOutput() {}

// NewStepOutputNone returns a StepOutput indicating the step produced no
// event.
func NewStepOutputNone() StepOutput { return StepOutputNone{} }

// NewStepOutputSingle returns a StepOutput carrying a single event.
func NewStepOutputSingle(e Event) StepOutput { return StepOutputSingle{Event: e} }

// NewStepOutputMultiple returns a StepOutput carrying multiple events.
//
// A nil or empty slice is permitted; the workflow engine treats it
// identically to [StepOutputNone].
func NewStepOutputMultiple(events []Event) StepOutput {
	return StepOutputMultiple{Events: events}
}

// stepOutputToFFI converts the public StepOutput into the generated FFI
// sum type. An unknown concrete type collapses to [StepOutputNone] —
// callers cannot construct other variants because the interface is sealed.
func stepOutputToFFI(o StepOutput) uniffiblazen.StepOutput {
	switch v := o.(type) {
	case StepOutputNone:
		return uniffiblazen.StepOutputNone{}
	case StepOutputSingle:
		return uniffiblazen.StepOutputSingle{Event: v.Event.toFFI()}
	case StepOutputMultiple:
		ffiEvents := make([]uniffiblazen.Event, len(v.Events))
		for i, ev := range v.Events {
			ffiEvents[i] = ev.toFFI()
		}
		return uniffiblazen.StepOutputMultiple{Events: ffiEvents}
	default:
		return uniffiblazen.StepOutputNone{}
	}
}

// StepHandler is the foreign-implementable interface registered against a
// workflow step. The engine calls Invoke whenever an event whose type
// matches the step's accepts list arrives.
//
// The ctx argument is the wrapper-managed [context.Context] supplied at
// handler-registration time (see [WorkflowBuilder.Step]); it is never the
// per-call context of [Workflow.Run] because UniFFI's callback ABI does
// not propagate the run-time context across the FFI boundary. Handlers
// that need cancellation tied to the active run should consult external
// state rather than relying on this ctx.
type StepHandler interface {
	Invoke(ctx context.Context, event Event) (StepOutput, error)
}

// stepHandlerAdapter satisfies the generated [uniffiblazen.StepHandler]
// interface and forwards calls to the user-supplied [StepHandler].
//
// The adapter holds [context.Background] because UniFFI does not pass a
// per-invocation context across the FFI boundary; handlers see a fresh
// background context on every call.
type stepHandlerAdapter struct {
	inner StepHandler
}

// Invoke implements the generated FFI StepHandler interface. Errors
// returned by the user handler are funnelled through [unwrapToValidation]
// so the Rust side receives a typed [*uniffiblazen.BlazenError] rather
// than an opaque foreign error.
func (a *stepHandlerAdapter) Invoke(event uniffiblazen.Event) (uniffiblazen.StepOutput, error) {
	out, err := a.inner.Invoke(context.Background(), eventFromFFI(event))
	if err != nil {
		return uniffiblazen.StepOutputNone{}, unwrapToValidation(err)
	}
	return stepOutputToFFI(out), nil
}

// WorkflowResult is the outcome of a workflow run. Event carries the
// terminal payload (typically a "StopEvent"). The token counters and
// TotalCostUSD are aggregated across every LLM step that ran during the
// workflow; they are zero when no LLM activity occurred.
type WorkflowResult struct {
	Event             Event
	TotalInputTokens  uint64
	TotalOutputTokens uint64
	TotalCostUSD      float64
}

// workflowResultFromFFI converts the generated FFI record into the
// package-public form, renaming TotalCostUsd to TotalCostUSD to match Go
// naming conventions for acronyms.
func workflowResultFromFFI(r uniffiblazen.WorkflowResult) WorkflowResult {
	return WorkflowResult{
		Event:             eventFromFFI(r.Event),
		TotalInputTokens:  r.TotalInputTokens,
		TotalOutputTokens: r.TotalOutputTokens,
		TotalCostUSD:      r.TotalCostUsd,
	}
}

// initOnce lazily initialises the embedded native runtime the first time a
// workflow operation needs it. Calling Init multiple times is safe, but
// gating it through sync.Once keeps the hot path branch-free.
var initOnce sync.Once

// ensureInit performs a one-time Init() of the underlying native runtime.
// Every public entry point that touches the FFI calls this so users never
// have to remember to call Init() explicitly.
func ensureInit() {
	initOnce.Do(func() {
		uniffiblazen.Init()
	})
}

// Workflow is a built workflow ready to run. Obtain one via
// [WorkflowBuilder.Build]. Call [Workflow.Close] when finished to release
// the underlying native handle; a finalizer is attached as a safety net,
// but explicit Close is preferred for predictable resource release.
type Workflow struct {
	inner *uniffiblazen.Workflow
	once  sync.Once
}

// Run executes the workflow with the supplied input value, which is
// JSON-marshalled into the StartEvent payload. Passing nil sends the
// JSON literal `null`.
//
// The Rust side of the run is blocking from Go's perspective. To honour
// ctx cancellation Run launches the FFI call on a background goroutine
// and waits on ctx.Done(); when ctx fires first the function returns
// ctx.Err() immediately. The workflow itself keeps running on the Rust
// side until it finishes naturally — cancellation propagation into the
// Rust runtime is a known gap pending an upstream UniFFI feature.
func (w *Workflow) Run(ctx context.Context, input any) (*WorkflowResult, error) {
	payload, err := json.Marshal(input)
	if err != nil {
		return nil, &ValidationError{Message: fmt.Sprintf("marshal workflow input: %s", err)}
	}
	return w.RunJSON(ctx, string(payload))
}

// RunJSON is the lower-level form of [Workflow.Run] that accepts a raw
// JSON string. Use this when the input is already a JSON document and the
// extra marshal/unmarshal of [Workflow.Run] would be wasteful.
//
// ctx cancellation semantics match [Workflow.Run] — the caller is
// unblocked on cancel, but the Rust run continues in the background.
func (w *Workflow) RunJSON(ctx context.Context, inputJSON string) (*WorkflowResult, error) {
	if w.inner == nil {
		return nil, &ValidationError{Message: "workflow has been closed"}
	}

	type runResult struct {
		res uniffiblazen.WorkflowResult
		err error
	}
	done := make(chan runResult, 1)
	go func() {
		res, err := w.inner.Run(inputJSON)
		done <- runResult{res: res, err: err}
	}()

	select {
	case <-ctx.Done():
		return nil, ctx.Err()
	case r := <-done:
		if r.err != nil {
			return nil, wrapErr(r.err)
		}
		out := workflowResultFromFFI(r.res)
		return &out, nil
	}
}

// StepNames returns the workflow's step names in registration order.
func (w *Workflow) StepNames() []string {
	if w.inner == nil {
		return nil
	}
	return w.inner.StepNames()
}

// Close releases the underlying native handle. It is safe to call Close
// multiple times and from multiple goroutines; subsequent calls are
// no-ops.
func (w *Workflow) Close() {
	w.once.Do(func() {
		if w.inner != nil {
			w.inner.Destroy()
			w.inner = nil
		}
	})
}

// WorkflowBuilder configures a workflow before it is built. Methods
// return the receiver so calls can be chained; any error short-circuits
// the chain by returning a nil builder alongside the wrapped error.
type WorkflowBuilder struct {
	inner *uniffiblazen.WorkflowBuilder
}

// NewWorkflowBuilder starts a new builder with the given workflow name.
// The name is surfaced in telemetry, traces, and error messages.
func NewWorkflowBuilder(name string) *WorkflowBuilder {
	ensureInit()
	return &WorkflowBuilder{inner: uniffiblazen.NewWorkflowBuilder(name)}
}

// Step registers a handler for the given step.
//
// name is a unique identifier within the workflow. accepts lists the
// event types that should trigger this step (e.g. {"StartEvent"}); emits
// declares every event type the handler may return so the engine can
// validate routing ahead of time.
//
// The returned builder is the receiver, enabling fluent chaining; on
// failure the builder is nil and the error is wrapped via [wrapErr].
func (b *WorkflowBuilder) Step(name string, accepts, emits []string, handler StepHandler) (*WorkflowBuilder, error) {
	if b == nil || b.inner == nil {
		return nil, &ValidationError{Message: "workflow builder has already been built or is nil"}
	}
	if handler == nil {
		return nil, &ValidationError{Message: "step handler must not be nil"}
	}
	adapter := &stepHandlerAdapter{inner: handler}
	next, err := b.inner.Step(name, accepts, emits, adapter)
	if err != nil {
		return nil, wrapErr(err)
	}
	b.inner = next
	return b, nil
}

// StepTimeout sets the per-step timeout. Steps that run longer than d
// are aborted by the engine. The duration is rounded down to whole
// milliseconds; negative values are clamped to zero.
func (b *WorkflowBuilder) StepTimeout(d time.Duration) (*WorkflowBuilder, error) {
	if b == nil || b.inner == nil {
		return nil, &ValidationError{Message: "workflow builder has already been built or is nil"}
	}
	next, err := b.inner.StepTimeoutMs(durationToMillis(d))
	if err != nil {
		return nil, wrapErr(err)
	}
	b.inner = next
	return b, nil
}

// Timeout sets the workflow-wide timeout. The whole run aborts after d
// elapses, regardless of which step is currently active. The duration is
// rounded down to whole milliseconds; negative values are clamped to
// zero.
func (b *WorkflowBuilder) Timeout(d time.Duration) (*WorkflowBuilder, error) {
	if b == nil || b.inner == nil {
		return nil, &ValidationError{Message: "workflow builder has already been built or is nil"}
	}
	next, err := b.inner.TimeoutMs(durationToMillis(d))
	if err != nil {
		return nil, wrapErr(err)
	}
	b.inner = next
	return b, nil
}

// Build finalises the workflow. The builder is consumed by this call;
// subsequent invocations return a [ValidationError]. The returned
// [Workflow] owns a native handle and should be closed with
// [Workflow.Close] when no longer needed.
func (b *WorkflowBuilder) Build() (*Workflow, error) {
	if b == nil || b.inner == nil {
		return nil, &ValidationError{Message: "workflow builder has already been built or is nil"}
	}
	inner, err := b.inner.Build()
	if err != nil {
		return nil, wrapErr(err)
	}
	b.inner = nil
	wf := &Workflow{inner: inner}
	// Safety net: if the caller forgets Close, the finalizer reclaims the
	// underlying native handle on GC. Close() clears the finalizer via
	// the underlying Destroy().
	runtime.SetFinalizer(wf, func(w *Workflow) { w.Close() })
	return wf, nil
}

// durationToMillis converts a Go time.Duration to the uint64 millisecond
// units expected by the FFI. Negative durations clamp to zero rather than
// underflow.
func durationToMillis(d time.Duration) uint64 {
	if d <= 0 {
		return 0
	}
	return uint64(d / time.Millisecond)
}

// Compile-time interface checks: catch generator drift early. If the FFI
// renames or retypes the StepHandler trait, these declarations will fail
// to build.
var (
	_ uniffiblazen.StepHandler = (*stepHandlerAdapter)(nil)
	_ error                    = (*ValidationError)(nil)
)
