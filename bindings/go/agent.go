package blazen

import (
	"context"
	"runtime"
	"sync"

	uniffiblazen "github.com/zachhandley/Blazen/bindings/go/internal/uniffi/blazen"
)

// ToolHandler is implemented by callers to execute the tools an agent
// requests. The agent loop calls Execute when the model emits a tool
// call, then feeds the returned string back to the model as the
// tool-result message on the next iteration.
//
// argumentsJSON is the model's JSON-encoded arguments object; the
// returned string MUST be valid JSON (use "null" for tools with no
// useful return value). Returning a non-nil error aborts the agent loop
// and the error message is surfaced verbatim to the foreign caller.
//
// The ctx argument is [context.Background] in this release because the
// UniFFI callback ABI does not propagate the run-time context across
// the FFI boundary; handlers that need cancellation tied to the active
// run should consult external state rather than relying on this ctx.
type ToolHandler interface {
	Execute(ctx context.Context, toolName string, argumentsJSON string) (string, error)
}

// ToolHandlerFunc adapts a bare function into a [ToolHandler]. Use this
// when a full method receiver is overkill — typically when the handler
// closes over a small amount of state in a constructor.
type ToolHandlerFunc func(ctx context.Context, toolName string, argumentsJSON string) (string, error)

// Execute satisfies the [ToolHandler] interface by invoking the
// underlying function value.
func (f ToolHandlerFunc) Execute(ctx context.Context, toolName string, argumentsJSON string) (string, error) {
	return f(ctx, toolName, argumentsJSON)
}

// AgentResult is the outcome of a completed agent loop. FinalMessage is
// the model's terminal textual response. Iterations counts the number
// of LLM round-trips the loop executed. ToolCallCount is the total
// number of tool invocations dispatched. TotalUsage aggregates token
// counters across every completion call. TotalCostUSD is the summed
// per-iteration USD cost (zero when the provider did not report cost
// data — the wire format does not distinguish "zero" from "unknown").
type AgentResult struct {
	FinalMessage  string
	Iterations    uint32
	ToolCallCount uint32
	TotalUsage    TokenUsage
	TotalCostUSD  float64
}

// agentResultFromFFI converts the generated FFI record into the
// package-public form, renaming TotalCostUsd to TotalCostUSD to match
// Go naming conventions for acronyms.
func agentResultFromFFI(r uniffiblazen.AgentResult) AgentResult {
	return AgentResult{
		FinalMessage:  r.FinalMessage,
		Iterations:    r.Iterations,
		ToolCallCount: r.ToolCallCount,
		TotalUsage:    tokenUsageFromFFI(r.TotalUsage),
		TotalCostUSD:  r.TotalCostUsd,
	}
}

// toolHandlerAdapter satisfies the generated [uniffiblazen.ToolHandler]
// interface and forwards calls to the user-supplied [ToolHandler].
//
// The adapter holds [context.Background] because UniFFI does not pass a
// per-invocation context across the FFI boundary; handlers see a fresh
// background context on every call.
type toolHandlerAdapter struct {
	inner ToolHandler
}

// Execute implements the generated FFI ToolHandler interface. Errors
// returned by the user handler are funnelled through
// [unwrapToValidation] so the Rust side receives a typed
// [*uniffiblazen.BlazenError] rather than an opaque foreign error;
// [BlazenError::Tool] would be more semantically precise but the
// validation helper is the established cross-binding convention for
// surfacing foreign-side failures and keeps parity with the workflow
// step handler adapter.
func (a *toolHandlerAdapter) Execute(toolName, argumentsJson string) (string, error) {
	out, err := a.inner.Execute(context.Background(), toolName, argumentsJson)
	if err != nil {
		return "", unwrapToValidation(err)
	}
	return out, nil
}

// Agent runs an LLM tool-call loop. The loop alternates between calling
// the completion model and executing the tools the model selects until
// the model produces a final response (no tool calls) or the iteration
// budget is exhausted.
//
// Construct one via [NewAgent], drive it with [Agent.Run] or
// [Agent.RunBlocking], and release the underlying native handle via
// [Agent.Close]. A finalizer is attached as a safety net, but explicit
// Close is preferred for predictable resource release.
//
// Reuse a single Agent across calls when configuration is stable — the
// underlying native handle holds the model, tool catalogue, and
// iteration budget that would otherwise be re-allocated per call.
type Agent struct {
	inner *uniffiblazen.Agent
	once  sync.Once
}

// NewAgent constructs an agent over the given completion model.
//
// systemPrompt is the system message prepended to every turn; pass the
// empty string to omit. tools is the catalogue exposed to the model
// (the Name on each Tool must match the names the handler dispatches
// on). handler executes the tools the model picks. maxIterations is a
// safety cap on the LLM round-trip count; the loop terminates with the
// model's last message if exceeded.
//
// model and handler must be non-nil; both are programmer errors that
// panic immediately rather than returning an error, mirroring the
// upstream FFI contract that [uniffiblazen.NewAgent] is infallible.
func NewAgent(model *CompletionModel, systemPrompt string, tools []Tool, handler ToolHandler, maxIterations uint32) *Agent {
	ensureInit()
	if model == nil || model.inner == nil {
		panic("blazen.NewAgent: model must not be nil or closed")
	}
	if handler == nil {
		panic("blazen.NewAgent: handler must not be nil")
	}
	adapter := &toolHandlerAdapter{inner: handler}
	inner := uniffiblazen.NewAgent(
		model.inner,
		optString(systemPrompt),
		toolSliceToFFI(tools),
		adapter,
		maxIterations,
	)
	a := &Agent{inner: inner}
	// Safety net: if the caller forgets Close, the finalizer reclaims the
	// underlying native handle on GC. Close() clears the finalizer via the
	// underlying Destroy().
	runtime.SetFinalizer(a, func(a *Agent) { a.Close() })
	return a
}

// Run drives the agent loop with userInput as the initial user message.
//
// The Rust side of the run is blocking from Go's perspective. To honour
// ctx cancellation Run launches the FFI call on a background goroutine
// and waits on ctx.Done(); when ctx fires first the function returns
// ctx.Err() immediately. The agent itself keeps running on the Rust
// side until it finishes naturally — cancellation propagation into the
// Rust runtime is a known gap pending an upstream UniFFI feature.
func (a *Agent) Run(ctx context.Context, userInput string) (*AgentResult, error) {
	if a.inner == nil {
		return nil, &ValidationError{Message: "agent has been closed"}
	}

	type runResult struct {
		res uniffiblazen.AgentResult
		err error
	}
	done := make(chan runResult, 1)
	go func() {
		res, err := a.inner.Run(userInput)
		done <- runResult{res: res, err: err}
	}()

	select {
	case <-ctx.Done():
		return nil, ctx.Err()
	case r := <-done:
		if r.err != nil {
			return nil, wrapErr(r.err)
		}
		out := agentResultFromFFI(r.res)
		return &out, nil
	}
}

// RunBlocking is the synchronous variant of [Agent.Run]. It does not
// accept a [context.Context] and blocks the calling goroutine until the
// loop terminates. Prefer [Agent.Run] in long-running services where
// cancellation matters; use this in short scripts or main() functions
// where the async wiring is overkill.
func (a *Agent) RunBlocking(userInput string) (*AgentResult, error) {
	if a.inner == nil {
		return nil, &ValidationError{Message: "agent has been closed"}
	}
	res, err := a.inner.RunBlocking(userInput)
	if err != nil {
		return nil, wrapErr(err)
	}
	out := agentResultFromFFI(res)
	return &out, nil
}

// Close releases the underlying native handle. It is safe to call Close
// multiple times and from multiple goroutines; subsequent calls are
// no-ops.
func (a *Agent) Close() {
	a.once.Do(func() {
		if a.inner != nil {
			a.inner.Destroy()
			a.inner = nil
		}
	})
}

// Compile-time interface check: catch generator drift early. If the FFI
// renames or retypes the ToolHandler trait, this declaration will fail
// to build.
var _ uniffiblazen.ToolHandler = (*toolHandlerAdapter)(nil)
