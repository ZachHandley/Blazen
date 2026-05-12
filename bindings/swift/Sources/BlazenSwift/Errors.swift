import Foundation
import UniFFIBlazen

/// Re-export of the UniFFI-generated error type under the canonical Blazen
/// namespace.
///
/// The underlying `BlazenError` (defined in `UniFFIBlazen`) is a flat enum
/// whose every variant carries a `message: String` payload. UniFFI's
/// `flat_error` representation does not survive the round-trip from Rust
/// with the rich structured fields the Node binding's error classes expose
/// (status codes, retry-after hints, request ids, etc.) — those details are
/// folded into the message string by the underlying conversion before they
/// cross the FFI boundary.
///
/// `BlazenError` is `LocalizedError` so `error.localizedDescription` is
/// available in error chains, and `Equatable`/`Hashable` so it composes
/// with Swift Testing helpers and `Set`/`Dictionary` lookups.
public typealias BlazenError = UniFFIBlazen.BlazenError

/// Map any thrown `Error` into its canonical Blazen representation.
///
/// `Error` values that originated from the UniFFI layer are already
/// `BlazenError`s and pass through unchanged. Anything else (Swift
/// `CancellationError`, foreign `URLError`, host-language exceptions) is
/// folded into `BlazenError.Internal` with the underlying error's
/// description as the message — never lose detail by collapsing into a
/// bare "unknown" string.
///
/// `wrap` exists primarily for higher-level adapters in this package (the
/// `AsyncThrowingStream` bridge in `Streaming.swift`, the agent loop, etc.)
/// that need a uniform error surface even when the failure originates on
/// the Swift side.
public func wrap(_ error: Error) -> BlazenError {
    if let blazen = error as? BlazenError {
        return blazen
    }
    if error is CancellationError {
        return .Cancelled(message: "cancelled")
    }
    return .Internal(message: String(describing: error))
}

/// Convenience: the human-readable message attached to a `BlazenError`,
/// regardless of which variant it is. Use this when surfacing errors
/// directly to a user — `error.localizedDescription` is also available via
/// `LocalizedError` and is functionally equivalent on these variants.
public extension BlazenError {
    /// Plain-text message carried inside the error variant. Equivalent to
    /// `localizedDescription` but typed as `String` (not `String?`).
    var message: String {
        switch self {
        case let .Auth(message), let .RateLimit(message), let .Timeout(message),
             let .Validation(message), let .ContentPolicy(message), let .Unsupported(message),
             let .Compute(message), let .Media(message), let .Provider(message),
             let .Workflow(message), let .Tool(message), let .Peer(message),
             let .Persist(message), let .Prompt(message), let .Memory(message),
             let .Cache(message), let .Cancelled(message), let .Internal(message):
            return message
        }
    }
}
