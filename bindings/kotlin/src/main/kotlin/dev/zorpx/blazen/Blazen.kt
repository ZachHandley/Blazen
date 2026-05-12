package dev.zorpx.blazen

/**
 * Top-level entry point for the Blazen Kotlin binding.
 *
 * Blazen is a Rust LLM orchestration framework; this module surfaces the
 * native runtime through Mozilla UniFFI + JNA. The package
 * [`dev.zorpx.blazen.uniffi`][dev.zorpx.blazen.uniffi] holds the generated
 * FFI glue (do not call directly); this package provides the idiomatic
 * Kotlin API on top of it.
 *
 * ## UniFFI surface coverage
 *
 * The companion native library at
 * `crates/blazen-uniffi/target/release/libblazen_uniffi.so` exposes only the
 * UDL-declared surface today (the [version] function). The rich
 * `#[uniffi::export]` proc-macro surface (`Workflow`, `Pipeline`,
 * `CompletionModel`, streaming, agent, batch, etc.) does **not** yet appear
 * in the cdylib's exported metadata sections, so the bindgen cannot emit
 * Kotlin glue for it. Once the upstream crate is rebuilt with those
 * symbols retained, regenerate `blazen.kt` via
 * `scripts/regen-bindings.sh kotlin` and add idiomatic wrappers next to
 * [Errors] and the value records in [Workflow], [Pipeline], [LLM].
 */
public object Blazen {
    /**
     * The native `blazen-uniffi` crate version baked into the loaded
     * shared library. Useful for diagnosing skew between this Kotlin
     * module and the `libblazen_uniffi.{so,dylib,dll}` resolved by JNA at
     * runtime.
     */
    public val version: String
        get() = dev.zorpx.blazen.uniffi.version()
}
