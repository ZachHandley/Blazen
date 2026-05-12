package dev.zorpx.blazen

import kotlinx.serialization.Serializable

/**
 * Snapshot of a workflow's persisted state at a point in time.
 *
 * Checkpoints round-trip through Blazen's checkpoint store (redb on disk
 * or Valkey/Redis over the network) so a workflow run can be resumed
 * across restarts. `eventsJson` is the JSON-encoded replay log; `stepName`
 * is the last-completed step.
 */
@Serializable
public data class WorkflowCheckpoint(
    val workflowId: String,
    val stepName: String,
    val eventsJson: String,
    val createdAtMs: Long,
)

/**
 * A single event entry in a persisted workflow's history.
 *
 * Used by the checkpoint store's history-iteration API to stream the
 * replay log lazily.
 */
@Serializable
public data class PersistedEvent(
    val eventType: String,
    val dataJson: String,
    val emittedAtMs: Long,
)
