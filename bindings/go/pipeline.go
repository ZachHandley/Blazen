package blazen

import (
	"context"
	"encoding/json"
	"fmt"
	"runtime"
	"sync"
	"time"

	uniffiblazen "github.com/zorpxinc/blazen-go/internal/uniffi/blazen"
)

// Pipeline is a validated, runnable composition of one or more workflows.
// A pipeline runs its stages sequentially (one workflow at a time), or in
// parallel within a Parallel stage that fans out to several workflows.
//
// Obtain a Pipeline via [PipelineBuilder.Build]. Call [Pipeline.Close]
// when finished to release the underlying native handle; a finalizer is
// attached as a safety net but explicit Close is preferred for
// predictable resource release.
type Pipeline struct {
	inner *uniffiblazen.Pipeline
	once  sync.Once
}

// Run executes the pipeline with the supplied input value, which is
// JSON-marshalled into the StartEvent payload delivered to the first
// stage. Passing nil sends the JSON literal `null`.
//
// The Rust side of the run is blocking from Go's perspective. To honour
// ctx cancellation Run launches the FFI call on a background goroutine
// and waits on ctx.Done(); when ctx fires first the function returns
// ctx.Err() immediately. The pipeline itself keeps running on the Rust
// side until it finishes naturally — cancellation propagation into the
// Rust runtime is a known gap pending an upstream UniFFI feature.
func (p *Pipeline) Run(ctx context.Context, input any) (*WorkflowResult, error) {
	payload, err := json.Marshal(input)
	if err != nil {
		return nil, &ValidationError{Message: fmt.Sprintf("marshal pipeline input: %s", err)}
	}
	return p.RunJSON(ctx, string(payload))
}

// RunJSON is the lower-level form of [Pipeline.Run] that accepts a raw
// JSON string. Use this when the input is already a JSON document and
// the extra marshal/unmarshal of [Pipeline.Run] would be wasteful.
//
// ctx cancellation semantics match [Pipeline.Run] — the caller is
// unblocked on cancel, but the Rust run continues in the background.
func (p *Pipeline) RunJSON(ctx context.Context, inputJSON string) (*WorkflowResult, error) {
	if p.inner == nil {
		return nil, &ValidationError{Message: "pipeline has been closed"}
	}

	type runResult struct {
		res uniffiblazen.WorkflowResult
		err error
	}
	done := make(chan runResult, 1)
	go func() {
		res, err := p.inner.Run(inputJSON)
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

// StageNames returns the pipeline's stage names in registration order.
// A Parallel stage contributes a single entry (the stage name passed to
// [PipelineBuilder.Parallel]), not one entry per branch.
func (p *Pipeline) StageNames() []string {
	if p.inner == nil {
		return nil
	}
	return p.inner.StageNames()
}

// Close releases the underlying native handle. It is safe to call Close
// multiple times and from multiple goroutines; subsequent calls are
// no-ops.
func (p *Pipeline) Close() {
	p.once.Do(func() {
		if p.inner != nil {
			p.inner.Destroy()
			p.inner = nil
		}
	})
}

// PipelineBuilder configures a pipeline before it is built. Methods
// return the receiver so calls can be chained; any error short-circuits
// the chain by returning a nil builder alongside the wrapped error.
type PipelineBuilder struct {
	inner *uniffiblazen.PipelineBuilder
}

// NewPipelineBuilder starts a new builder with the given pipeline name.
// The name is surfaced in telemetry, traces, and error messages.
func NewPipelineBuilder(name string) *PipelineBuilder {
	ensureInit()
	return &PipelineBuilder{inner: uniffiblazen.NewPipelineBuilder(name)}
}

// Stage appends a sequential stage that runs the given workflow. The
// stage name is surfaced in telemetry and must be unique within the
// pipeline; the engine validates uniqueness at [PipelineBuilder.Build]
// time.
//
// The workflow must not have been closed. Closing the workflow after it
// has been added is a programmer error: the pipeline takes ownership of
// the underlying native handle when Build succeeds.
func (b *PipelineBuilder) Stage(name string, w *Workflow) (*PipelineBuilder, error) {
	if b == nil || b.inner == nil {
		return nil, &ValidationError{Message: "pipeline builder has already been built or is nil"}
	}
	if w == nil || w.inner == nil {
		return nil, &ValidationError{Message: "workflow must not be nil or closed"}
	}
	next, err := b.inner.Stage(name, w.inner)
	if err != nil {
		return nil, wrapErr(err)
	}
	b.inner = next
	return b, nil
}

// AddWorkflow appends a sequential stage that runs the given workflow,
// using the workflow's own name as the stage name. It is the shorthand
// counterpart of [PipelineBuilder.Stage].
func (b *PipelineBuilder) AddWorkflow(w *Workflow) (*PipelineBuilder, error) {
	if b == nil || b.inner == nil {
		return nil, &ValidationError{Message: "pipeline builder has already been built or is nil"}
	}
	if w == nil || w.inner == nil {
		return nil, &ValidationError{Message: "workflow must not be nil or closed"}
	}
	next, err := b.inner.AddWorkflow(w.inner)
	if err != nil {
		return nil, wrapErr(err)
	}
	b.inner = next
	return b, nil
}

// Parallel appends a fan-out stage that runs several workflows
// concurrently under a single stage name. branchNames labels each branch
// for telemetry and must have the same length as workflows.
//
// When waitAll is true the stage completes only once every branch has
// finished; when false the stage completes as soon as the first branch
// finishes and the remaining branches are cancelled by the engine.
func (b *PipelineBuilder) Parallel(name string, branchNames []string, workflows []*Workflow, waitAll bool) (*PipelineBuilder, error) {
	if b == nil || b.inner == nil {
		return nil, &ValidationError{Message: "pipeline builder has already been built or is nil"}
	}
	if len(branchNames) != len(workflows) {
		return nil, &ValidationError{Message: fmt.Sprintf("parallel: branchNames length %d does not match workflows length %d", len(branchNames), len(workflows))}
	}
	inners := make([]*uniffiblazen.Workflow, len(workflows))
	for i, w := range workflows {
		if w == nil || w.inner == nil {
			return nil, &ValidationError{Message: fmt.Sprintf("parallel: workflow at index %d is nil or closed", i)}
		}
		inners[i] = w.inner
	}
	next, err := b.inner.Parallel(name, branchNames, inners, waitAll)
	if err != nil {
		return nil, wrapErr(err)
	}
	b.inner = next
	return b, nil
}

// TimeoutPerStage sets the per-stage timeout. Stages that run longer
// than d are aborted by the engine. The duration is rounded down to
// whole milliseconds; negative values are clamped to zero.
func (b *PipelineBuilder) TimeoutPerStage(d time.Duration) (*PipelineBuilder, error) {
	if b == nil || b.inner == nil {
		return nil, &ValidationError{Message: "pipeline builder has already been built or is nil"}
	}
	next, err := b.inner.TimeoutPerStageMs(durationToMillis(d))
	if err != nil {
		return nil, wrapErr(err)
	}
	b.inner = next
	return b, nil
}

// TotalTimeout sets the pipeline-wide timeout. The whole run aborts
// after d elapses, regardless of which stage is currently active. The
// duration is rounded down to whole milliseconds; negative values are
// clamped to zero.
func (b *PipelineBuilder) TotalTimeout(d time.Duration) (*PipelineBuilder, error) {
	if b == nil || b.inner == nil {
		return nil, &ValidationError{Message: "pipeline builder has already been built or is nil"}
	}
	next, err := b.inner.TotalTimeoutMs(durationToMillis(d))
	if err != nil {
		return nil, wrapErr(err)
	}
	b.inner = next
	return b, nil
}

// Build finalises the pipeline. The builder is consumed by this call;
// subsequent invocations return a [ValidationError]. The returned
// [Pipeline] owns a native handle and should be closed with
// [Pipeline.Close] when no longer needed.
func (b *PipelineBuilder) Build() (*Pipeline, error) {
	if b == nil || b.inner == nil {
		return nil, &ValidationError{Message: "pipeline builder has already been built or is nil"}
	}
	inner, err := b.inner.Build()
	if err != nil {
		return nil, wrapErr(err)
	}
	b.inner = nil
	p := &Pipeline{inner: inner}
	// Safety net: if the caller forgets Close, the finalizer reclaims the
	// underlying native handle on GC. Close() clears the finalizer via
	// the underlying Destroy().
	runtime.SetFinalizer(p, func(pl *Pipeline) { pl.Close() })
	return p, nil
}
