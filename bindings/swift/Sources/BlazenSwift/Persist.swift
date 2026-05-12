import Foundation
import UniFFIBlazen

/// A workflow-checkpoint store handle. Build via
/// `Persist.redb(path:)` (embedded file-backed) or
/// `Persist.valkey(url:ttlSeconds:)` (Redis/Valkey).
public typealias CheckpointStore = UniFFIBlazen.CheckpointStore

/// A snapshot of a workflow's state at a point in time.
public typealias WorkflowCheckpoint = UniFFIBlazen.WorkflowCheckpoint

/// A serialised representation of a queued event captured in a
/// checkpoint.
public typealias PersistedEvent = UniFFIBlazen.PersistedEvent

/// Factory namespace for checkpoint store backends.
public enum Persist {
    /// Build an embedded redb-backed checkpoint store rooted at `path`.
    /// The database file is created on demand; re-opening an existing
    /// file is safe.
    public static func redb(path: String) throws -> CheckpointStore {
        try newRedbCheckpointStore(path: path)
    }

    /// Build a Redis / Valkey-backed checkpoint store connected to
    /// `url` (`redis://host:port/db` or `rediss://` for TLS). When
    /// `ttlSeconds` is set, every saved checkpoint auto-expires after
    /// that many seconds.
    public static func valkey(url: String, ttlSeconds: UInt64? = nil) throws -> CheckpointStore {
        try newValkeyCheckpointStore(url: url, ttlSeconds: ttlSeconds)
    }
}
