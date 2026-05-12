package blazen

import (
	"context"

	uniffiblazen "github.com/zachhandley/Blazen/bindings/go/internal/uniffi/blazen"
)

// BatchItem is the per-request outcome inside a [BatchResult]'s Responses
// slice. It is a sealed sum type: only [BatchItemSuccess] and
// [BatchItemFailure] satisfy it, matching the FFI's two-variant enum.
//
// Type-switch on the concrete variant to consume an outcome:
//
//	switch v := item.(type) {
//	case blazen.BatchItemSuccess:
//	    use(v.Response)
//	case blazen.BatchItemFailure:
//	    log(v.ErrorMessage)
//	}
type BatchItem interface {
	batchItem()
}

// BatchItemSuccess is the [BatchItem] variant carrying a completed
// [CompletionResponse]. Slot ordering in [BatchResult.Responses] matches
// the input request slice passed to [CompleteBatch].
type BatchItemSuccess struct {
	Response CompletionResponse
}

func (BatchItemSuccess) batchItem() {}

// BatchItemFailure is the [BatchItem] variant carrying a per-request
// error. ErrorMessage mirrors the Display form of the underlying
// BlazenError on the Rust side; the structured variant is flattened to a
// string at the FFI boundary because nested-enum errors do not survive
// every UniFFI target language. Callers wanting typed errors should
// dispatch requests individually via [CompletionModel.Complete].
type BatchItemFailure struct {
	ErrorMessage string
}

func (BatchItemFailure) batchItem() {}

// BatchResult is the aggregated outcome of a [CompleteBatch] /
// [CompleteBatchBlocking] call.
//
// Responses is parallel to the input requests slice: Responses[i]
// corresponds to requests[i]. Per-request errors appear as
// [BatchItemFailure] entries and do NOT cause [CompleteBatch] itself to
// return a Go-level error — only batch-level validation failures do
// (e.g. a malformed ParametersJSON / ResponseFormatJSON that fails to
// reach the provider at all).
//
// TotalUsage and TotalCostUSD aggregate only the successful responses;
// failed slots contribute zero. A TotalCostUSD of 0.0 is ambiguous
// between "no cost incurred" and "no provider reported cost data" — the
// wire format does not distinguish the two.
type BatchResult struct {
	Responses    []BatchItem
	TotalUsage   TokenUsage
	TotalCostUSD float64
}

// batchItemFromFFI converts the generated FFI BatchItem sum type into
// the wrapper's sealed interface. Unknown variants are downgraded to a
// [BatchItemFailure] with an explanatory message — this is defensive and
// should never trip in practice unless the generated bindings drift
// ahead of this wrapper.
func batchItemFromFFI(item uniffiblazen.BatchItem) BatchItem {
	switch v := item.(type) {
	case uniffiblazen.BatchItemSuccess:
		return BatchItemSuccess{Response: completionResponseFromFFI(v.Response)}
	case uniffiblazen.BatchItemFailure:
		return BatchItemFailure{ErrorMessage: v.ErrorMessage}
	default:
		return BatchItemFailure{ErrorMessage: "unknown batch item variant"}
	}
}

// batchResultFromFFI converts the generated FFI BatchResult into the
// wrapper form, translating each per-request slot and renaming the
// generated TotalCostUsd field to the idiomatic Go TotalCostUSD.
func batchResultFromFFI(r uniffiblazen.BatchResult) *BatchResult {
	items := make([]BatchItem, len(r.Responses))
	for i, ffiItem := range r.Responses {
		items[i] = batchItemFromFFI(ffiItem)
	}
	return &BatchResult{
		Responses:    items,
		TotalUsage:   tokenUsageFromFFI(r.TotalUsage),
		TotalCostUSD: r.TotalCostUsd,
	}
}

// CompleteBatch dispatches a batch of completion requests through the
// supplied model with a bounded concurrency cap.
//
// maxConcurrency = 0 means unlimited (every request dispatched in
// parallel, bounded only by the provider's own rate limits). For most
// callers, 4-8 is a reasonable starting point.
//
// Honours ctx cancellation by detaching from the wait: if ctx fires
// before the batch completes, the function returns ctx.Err() and the
// Rust-side batch continues to completion in the background.
// Cancellation propagation into the runtime is a known gap pending an
// upstream UniFFI feature.
//
// A non-nil error return indicates a batch-level failure (e.g. the
// model handle is closed, or every request failed input validation
// before dispatch). Per-request failures surface as [BatchItemFailure]
// entries inside the returned [BatchResult] and do not produce an
// error here.
func CompleteBatch(ctx context.Context, model *CompletionModel, requests []CompletionRequest, maxConcurrency uint32) (*BatchResult, error) {
	ensureInit()
	if model == nil || model.inner == nil {
		return nil, &ValidationError{Message: "model is nil or closed"}
	}
	ffiReqs := make([]uniffiblazen.CompletionRequest, len(requests))
	for i, r := range requests {
		ffiReqs[i] = r.toFFI()
	}
	type batchOutcome struct {
		res uniffiblazen.BatchResult
		err error
	}
	done := make(chan batchOutcome, 1)
	go func() {
		res, err := uniffiblazen.CompleteBatch(model.inner, ffiReqs, maxConcurrency)
		done <- batchOutcome{res: res, err: err}
	}()
	select {
	case <-ctx.Done():
		return nil, ctx.Err()
	case r := <-done:
		if r.err != nil {
			return nil, wrapErr(r.err)
		}
		return batchResultFromFFI(r.res), nil
	}
}

// CompleteBatchBlocking is the synchronous form of [CompleteBatch]. It
// does not accept a [context.Context] and blocks the calling goroutine
// until the entire batch completes. Prefer [CompleteBatch] in
// long-running services where cancellation matters; use this in short
// scripts or main() functions where the async wiring is overkill.
//
// The semantics of maxConcurrency, the [BatchResult] return value, and
// the distinction between batch-level errors (returned here) versus
// per-request failures (carried inside [BatchResult.Responses]) are
// identical to [CompleteBatch].
func CompleteBatchBlocking(model *CompletionModel, requests []CompletionRequest, maxConcurrency uint32) (*BatchResult, error) {
	ensureInit()
	if model == nil || model.inner == nil {
		return nil, &ValidationError{Message: "model is nil or closed"}
	}
	ffiReqs := make([]uniffiblazen.CompletionRequest, len(requests))
	for i, r := range requests {
		ffiReqs[i] = r.toFFI()
	}
	res, err := uniffiblazen.CompleteBatchBlocking(model.inner, ffiReqs, maxConcurrency)
	if err != nil {
		return nil, wrapErr(err)
	}
	return batchResultFromFFI(res), nil
}
