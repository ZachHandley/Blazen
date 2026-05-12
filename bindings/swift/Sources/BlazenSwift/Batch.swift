import Foundation
import UniFFIBlazen

/// Per-request outcome within a `BatchResult`. `case success(_:)` carries
/// the completion response; `case failure(errorMessage:)` carries the
/// flattened error message string (the structured `BlazenError` does not
/// survive nesting inside a UniFFI enum cleanly across all four target
/// languages, so the message is flattened to a string at the wire).
public typealias BatchItem = UniFFIBlazen.BatchItem

/// Outcome of a `completeBatch` call. `responses` is one slot per input
/// request in registration order; `totalUsage` and `totalCostUsd`
/// aggregate only the successful responses.
public typealias BatchResult = UniFFIBlazen.BatchResult

/// Run a batch of completion requests with bounded concurrency.
///
/// - Parameters:
///   - model: the completion model to drive (one provider / one model id;
///     dispatch across providers from caller code instead).
///   - requests: the requests to send, in order.
///   - maxConcurrency: hard cap on in-flight requests. `0` means
///     unlimited (every request dispatched in parallel).
///
/// Returns a `BatchResult` with per-request outcomes and aggregated
/// usage / cost. Individual request failures appear as `BatchItem.failure`
/// in the same slot — they do not cause this function itself to throw.
///
/// Throws `BlazenError.Validation` if any input request fails to convert
/// to the upstream wire format (typically a malformed
/// `parameters_json` / `response_format_json` payload).
public func completeBatch(
    model: CompletionModel,
    requests: [CompletionRequest],
    maxConcurrency: UInt32 = 8
) async throws -> BatchResult {
    try await UniFFIBlazen.completeBatch(
        model: model,
        requests: requests,
        maxConcurrency: maxConcurrency
    )
}
