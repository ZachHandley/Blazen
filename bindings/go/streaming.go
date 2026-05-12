package blazen

import (
	"context"
	"sync"

	uniffiblazen "github.com/zachhandley/Blazen/bindings/go/internal/uniffi/blazen"
)

// StreamChunk is a single incremental delta from a streaming completion.
//
// ContentDelta is the text delta accumulated since the previous chunk and
// is the empty string when the chunk carries only tool-call deltas.
// ToolCalls is a tool-call snapshot for this chunk; consumers should
// replace, not append, on each delivery. IsFinal is a UI hint marking the
// last content-bearing chunk before the terminal [StreamDoneEvent] —
// [StreamDoneEvent] remains the authoritative completion signal.
type StreamChunk struct {
	ContentDelta string
	ToolCalls    []ToolCall
	IsFinal      bool
}

// streamChunkFromFFI converts an FFI StreamChunk into the wrapper form.
func streamChunkFromFFI(c uniffiblazen.StreamChunk) StreamChunk {
	return StreamChunk{
		ContentDelta: c.ContentDelta,
		ToolCalls:    toolCallSliceFromFFI(c.ToolCalls),
		IsFinal:      c.IsFinal,
	}
}

// StreamEvent is one element delivered on the channel returned by
// [Stream]. The interface is sealed — switch on the concrete type to
// inspect the payload:
//
//	for ev := range stream {
//	    switch e := ev.(type) {
//	    case *blazen.StreamChunkEvent:
//	        // incremental text / tool-call delta in e.Chunk
//	    case *blazen.StreamDoneEvent:
//	        // successful end — e.FinishReason and e.Usage are final
//	    case *blazen.StreamErrorEvent:
//	        // failure — e.Err is already wrapped via the package error types
//	    }
//	}
//
// Exactly one of [StreamDoneEvent] or [StreamErrorEvent] is delivered as
// the terminal event, after which the channel is closed.
type StreamEvent interface {
	streamEvent()
}

// StreamChunkEvent carries an incremental [StreamChunk] from the model.
type StreamChunkEvent struct {
	Chunk StreamChunk
}

// streamEvent marks StreamChunkEvent as a sealed [StreamEvent] variant.
func (*StreamChunkEvent) streamEvent() {}

// StreamDoneEvent is the terminal event delivered on a successful run.
//
// FinishReason mirrors the provider-reported reason (e.g. "stop",
// "tool_calls", "length") and is the empty string when the provider did
// not report one. Usage is the aggregated token usage for the stream;
// counters are zero for providers that do not surface usage mid-stream.
type StreamDoneEvent struct {
	FinishReason string
	Usage        TokenUsage
}

// streamEvent marks StreamDoneEvent as a sealed [StreamEvent] variant.
func (*StreamDoneEvent) streamEvent() {}

// StreamErrorEvent is the terminal event delivered on a failed run. Err
// is already converted to a typed package error via the same machinery
// used by the non-streaming entry points, so callers can switch on the
// concrete error type with [errors.As].
type StreamErrorEvent struct {
	Err error
}

// streamEvent marks StreamErrorEvent as a sealed [StreamEvent] variant.
func (*StreamErrorEvent) streamEvent() {}

// streamChannelBuffer is the buffer depth of the channel returned by
// [Stream]. A modest buffer prevents the FFI thread from blocking on a
// slow consumer for typical text-token rates while still bounding memory
// growth if the consumer stalls entirely.
const streamChannelBuffer = 16

// channelSink bridges the FFI's three-callback [uniffiblazen.CompletionStreamSink]
// trait into a single Go channel of [StreamEvent].
//
// The sink owns the send side of ch and is responsible for closing it
// exactly once. The done channel signals "the terminal event has been
// delivered" to any cooperating goroutine (ctx watcher or the caller of
// CompleteStreaming). The once guard ensures the channel is closed at
// most once even when callbacks race.
type channelSink struct {
	ch   chan<- StreamEvent
	done chan struct{}
	once sync.Once
}

// finish performs the single-shot terminal-event delivery. It sends ev on
// the channel (unless the slot is already closed by a racing caller),
// closes the channel, and signals done. Subsequent calls are no-ops.
func (s *channelSink) finish(ev StreamEvent) {
	s.once.Do(func() {
		s.ch <- ev
		close(s.ch)
		close(s.done)
	})
}

// OnChunk delivers an incremental chunk to the consumer. The select also
// watches s.done so that a terminal event delivered concurrently (e.g.
// from the ctx watcher) does not deadlock this callback against a closed
// channel; in that case the chunk is dropped silently and an aborting
// error is returned so the Rust stream stops sending.
func (s *channelSink) OnChunk(chunk uniffiblazen.StreamChunk) error {
	select {
	case <-s.done:
		// Channel already closed by a terminal event — tell the Rust
		// stream to stop by returning a typed BlazenError.
		return unwrapToValidation(errStreamConsumerGone)
	case s.ch <- &StreamChunkEvent{Chunk: streamChunkFromFFI(chunk)}:
		return nil
	}
}

// OnDone delivers the successful-completion terminal event.
func (s *channelSink) OnDone(finishReason string, usage uniffiblazen.TokenUsage) error {
	s.finish(&StreamDoneEvent{
		FinishReason: finishReason,
		Usage:        tokenUsageFromFFI(usage),
	})
	return nil
}

// OnError delivers the failed-completion terminal event. The incoming
// FFI error is converted to a typed package error via [wrapErr] before it
// reaches the consumer.
func (s *channelSink) OnError(uniffiErr *uniffiblazen.BlazenError) error {
	s.finish(&StreamErrorEvent{Err: wrapErr(uniffiErr)})
	return nil
}

// errStreamConsumerGone is the sentinel surfaced to the Rust side when a
// chunk arrives after the channel has already been terminated (e.g.
// because ctx was cancelled). It exists only to give Rust a reason to
// stop streaming — consumers never see it because the channel is already
// closed at that point.
var errStreamConsumerGone = &ValidationError{Message: "stream consumer no longer receiving"}

// Stream drives a streaming completion. The returned channel receives
// [StreamChunkEvent] values as the model emits them, then exactly one of
// [StreamDoneEvent] (success) or [StreamErrorEvent] (failure) as the
// terminal event, and is then closed.
//
// The model handle must be alive for the duration of the stream — do not
// call [CompletionModel.Close] until the channel has been drained.
//
// If ctx is cancelled the channel receives a [StreamErrorEvent] carrying
// ctx.Err() and is closed; the underlying Rust stream observes the
// closure on its next chunk delivery and tears down cooperatively, but
// cancellation propagation is best-effort and the provider request keeps
// running until it finishes naturally.
//
// Always drain the channel. A buffered consumer that stops reading
// before the terminal event will prevent the background goroutine from
// exiting and pin the model handle in memory.
func Stream(ctx context.Context, model *CompletionModel, req CompletionRequest) <-chan StreamEvent {
	ensureInit()
	ch := make(chan StreamEvent, streamChannelBuffer)

	if model == nil || model.inner == nil {
		// Fast-fail path: deliver a single terminal error and close.
		go func() {
			ch <- &StreamErrorEvent{Err: &ValidationError{Message: "completion model has been closed"}}
			close(ch)
		}()
		return ch
	}

	sink := &channelSink{
		ch:   ch,
		done: make(chan struct{}),
	}

	// ctx watcher: races the sink's terminal callbacks to deliver
	// ctx.Err() if the caller cancels before the stream completes.
	go func() {
		select {
		case <-ctx.Done():
			sink.finish(&StreamErrorEvent{Err: ctx.Err()})
		case <-sink.done:
			// Terminal event already delivered by OnDone/OnError.
		}
	}()

	// FFI driver: blocks on CompleteStreaming, which itself returns only
	// when the sink has been fully drained on the Rust side. Sink
	// callbacks have already closed the channel by the time this returns
	// successfully; the early-failure (request validation) path needs an
	// explicit finish so the consumer is not left waiting forever.
	go func() {
		if err := uniffiblazen.CompleteStreaming(model.inner, req.toFFI(), sink); err != nil {
			sink.finish(&StreamErrorEvent{Err: wrapErr(err)})
		}
	}()

	return ch
}

// Compile-time interface check: catch generator drift early. If the FFI
// renames or retypes the CompletionStreamSink trait, this declaration
// will fail to build.
var _ uniffiblazen.CompletionStreamSink = (*channelSink)(nil)
