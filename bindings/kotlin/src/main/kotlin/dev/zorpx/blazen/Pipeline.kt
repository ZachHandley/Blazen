package dev.zorpx.blazen

import kotlinx.serialization.Serializable

/**
 * Per-stage configuration record for a Blazen [Pipeline].
 *
 * Pipelines are linear chains of stages, each consuming one event type and
 * emitting another. The Rust core builds a workflow under the hood; this
 * record describes a single stage on the wire.
 *
 * Once the UniFFI proc-macro surface for Pipeline lands in the cdylib,
 * [PipelineStage] is what the generated `pipelineBuilder.stage(...)` call
 * will accept.
 */
@Serializable
public data class PipelineStage(
    val name: String,
    val accepts: String,
    val emits: String,
)

/**
 * Lightweight description of an assembled pipeline (returned post-build).
 *
 * Carries only the metadata that is meaningful outside the Rust runtime —
 * the live pipeline handle itself is opaque and lives behind the
 * generated FFI surface in [`dev.zorpx.blazen.uniffi`][dev.zorpx.blazen.uniffi].
 */
@Serializable
public data class PipelineDescription(
    val name: String,
    val stages: List<PipelineStage>,
)
