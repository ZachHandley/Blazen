package dev.zorpx.blazen

/**
 * Sealed exception hierarchy raised by the Blazen Kotlin binding.
 *
 * Mirrors the Rust-side `BlazenError` enum in
 * `crates/blazen-uniffi/src/errors.rs`. Each subclass corresponds to one
 * variant; `kind`-tagged variants (`Provider`, `Peer`, `Prompt`, `Memory`,
 * `Cache`) carry the discriminator as a property.
 *
 * Callers branch on these via Kotlin's exhaustive `when (e) is ...` on a
 * `try { ... } catch (e: BlazenException) { ... }` block.
 */
public sealed class BlazenException(
    message: String,
    cause: Throwable? = null,
) : RuntimeException(message, cause) {
    /** Authentication / credentials failure. */
    public class Auth(message: String) : BlazenException(message)

    /** Rate limit exceeded. `retryAfterMs` is set when the provider hinted one. */
    public class RateLimit(message: String, public val retryAfterMs: Long?) : BlazenException(message)

    /** Operation timed out before the provider responded. */
    public class Timeout(message: String, public val elapsedMs: Long) : BlazenException(message)

    /** Input validation failed (bad schema, missing required field, etc.). */
    public class Validation(message: String) : BlazenException(message)

    /** Content policy violation. */
    public class ContentPolicy(message: String) : BlazenException(message)

    /** Operation unsupported on this platform / build / provider. */
    public class Unsupported(message: String) : BlazenException(message)

    /** Compute (CPU/GPU/accelerator) failure or OOM. */
    public class Compute(message: String) : BlazenException(message)

    /** Media decode / encode / transcode error. */
    public class Media(message: String) : BlazenException(message)

    /**
     * Provider / backend error.
     *
     * `kind` identifies the specific backend and failure mode (e.g.
     * `"LlamaCppModelLoad"`, `"OpenAIHttp"`). Other fields mirror the
     * structured payload from upstream's HTTP-aware provider errors.
     */
    public class Provider(
        public val kind: String,
        message: String,
        public val providerName: String?,
        public val status: Int?,
        public val endpoint: String?,
        public val requestId: String?,
        public val detail: String?,
        public val retryAfterMs: Long?,
    ) : BlazenException(message)

    /** Workflow execution error (step panic, deadlock, missing context). */
    public class Workflow(message: String) : BlazenException(message)

    /** Tool / function-call error during LLM agent execution. */
    public class Tool(message: String) : BlazenException(message)

    /** Distributed peer-to-peer error. `kind`: `"Encode"`, `"Transport"`, etc. */
    public class Peer(public val kind: String, message: String) : BlazenException(message)

    /** Persistence layer error (redb / valkey checkpoint store). */
    public class Persist(message: String) : BlazenException(message)

    /** Prompt template error. `kind`: `"MissingVariable"`, `"NotFound"`, etc. */
    public class Prompt(public val kind: String, message: String) : BlazenException(message)

    /** Memory subsystem error. `kind`: `"NoEmbedder"`, `"Embedding"`, etc. */
    public class Memory(public val kind: String, message: String) : BlazenException(message)

    /** Model-cache / download error. `kind`: `"Download"`, `"CacheDir"`, `"Io"`. */
    public class Cache(public val kind: String, message: String) : BlazenException(message)

    /** Operation was cancelled. */
    public class Cancelled : BlazenException("cancelled")

    /** Fallback for errors that don't fit any other variant. */
    public class Internal(message: String) : BlazenException(message)
}
