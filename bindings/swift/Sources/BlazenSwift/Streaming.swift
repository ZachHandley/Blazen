import Foundation
import UniFFIBlazen

/// A single chunk from a streaming chat completion. Re-exported under the
/// canonical Blazen namespace.
public typealias StreamChunk = UniFFIBlazen.StreamChunk

/// A terminal stream event, delivered as the final element of a streaming
/// completion's `AsyncThrowingStream` so callers see token usage and the
/// provider-reported finish reason in the same async-for loop they used
/// for content chunks.
///
/// The full chunk sequence is:
///
/// ```text
///   StreamEvent.chunk(...)
///   StreamEvent.chunk(...)
///   ...
///   StreamEvent.done(finishReason:usage:)
/// ```
///
/// On failure the stream finishes by throwing instead — the `done` event
/// is only emitted on the happy path.
public enum StreamEvent: Sendable, Equatable {
    /// An incremental content/tool-call chunk from the model.
    case chunk(StreamChunk)
    /// The terminal completion signal. Emitted exactly once after the
    /// last `chunk` on a successful run.
    case done(finishReason: String, usage: TokenUsage)
}

public extension CompletionModel {
    /// Run a streaming completion, surfacing each chunk and the terminal
    /// `done` signal through an `AsyncThrowingStream`.
    ///
    /// Iterate with `for try await event in model.completeStream(request)`:
    ///
    /// ```swift
    /// for try await event in model.completeStream(request) {
    ///     switch event {
    ///     case .chunk(let chunk):
    ///         print(chunk.contentDelta, terminator: "")
    ///     case .done(let reason, _):
    ///         print("\n[done: \(reason)]")
    ///     }
    /// }
    /// ```
    ///
    /// Provider failures and sink-side errors are surfaced by the stream
    /// throwing — they are not emitted as a final `.done` event. Cancel
    /// the iterator (e.g. by breaking out of the `for-await` loop or
    /// cancelling the enclosing `Task`) to tear the stream down early;
    /// the underlying Tokio task observes the cancellation and shuts down
    /// cooperatively.
    func completeStream(
        _ request: CompletionRequest
    ) -> AsyncThrowingStream<StreamEvent, Error> {
        let model = self
        return AsyncThrowingStream(StreamEvent.self, bufferingPolicy: .unbounded) { continuation in
            let sink = AsyncStreamSink(continuation: continuation)
            let task = Task<Void, Never> {
                do {
                    try await completeStreaming(model: model, request: request, sink: sink)
                    // Sink already finished the continuation on `onDone` /
                    // `onError`; this is the no-error fall-through.
                } catch {
                    // `completeStreaming` itself can throw before any chunk
                    // has been dispatched (e.g. malformed request). Surface
                    // those failures the same way the sink would.
                    sink.finish(throwing: wrap(error))
                }
            }
            continuation.onTermination = { _ in
                task.cancel()
            }
        }
    }
}

/// Sink-side adapter that bridges UniFFI's `CompletionStreamSink`
/// callbacks into an `AsyncThrowingStream` continuation.
///
/// Mutation of the finish flag is funneled through an inner actor so the
/// three callbacks (which may race in pathological provider behaviour)
/// can never finish the continuation twice. The continuation itself is
/// thread-safe — `AsyncThrowingStream.Continuation` is `Sendable` and
/// documented as safe to call from any context.
final class AsyncStreamSink: CompletionStreamSink, @unchecked Sendable {
    private let continuation: AsyncThrowingStream<StreamEvent, Error>.Continuation
    private let state = FinishState()

    init(continuation: AsyncThrowingStream<StreamEvent, Error>.Continuation) {
        self.continuation = continuation
    }

    func onChunk(chunk: StreamChunk) async throws {
        if await state.isFinished { return }
        continuation.yield(.chunk(chunk))
    }

    func onDone(finishReason: String, usage: TokenUsage) async throws {
        guard await state.markFinished() else { return }
        continuation.yield(.done(finishReason: finishReason, usage: usage))
        continuation.finish()
    }

    func onError(err: BlazenError) async throws {
        guard await state.markFinished() else { return }
        continuation.finish(throwing: err)
    }

    /// Idempotent failure-finish. Used by the surrounding `Task` when
    /// `completeStreaming` itself throws before the sink runs. Safe to
    /// call from sync contexts because the `Task` we spawn does not need
    /// the finish flag to be exclusive on its own thread.
    func finish(throwing error: Error) {
        Task {
            if await state.markFinished() {
                continuation.finish(throwing: error)
            }
        }
    }
}

/// Tiny `actor` guarding the single-shot finish flag so the sink can
/// race-safely decide which callback (one of `onDone`, `onError`, or the
/// surrounding `Task` failure path) gets to actually close the
/// continuation.
private actor FinishState {
    private var finished = false

    /// True once any callback has signalled completion.
    var isFinished: Bool { finished }

    /// Returns `true` exactly once — the first caller that transitions
    /// the flag from `false` to `true`. All subsequent callers see
    /// `false` and skip the continuation mutation.
    func markFinished() -> Bool {
        guard !finished else { return false }
        finished = true
        return true
    }
}
