# Blazen for Kotlin / JVM

Native JVM binding for the [Blazen](https://github.com/ZachHandley/Blazen) event-driven AI workflow engine. The Rust core is compiled to a per-arch shared library (`libblazen_uniffi.{so,dylib,dll}`) and exposed to Kotlin via UniFFI + JNA, so workflows, pipelines, LLM providers, and agents run in-process against actual Rust -- no HTTP shim, no subprocess.

## Status

This is a v0 binding. JVM only (Android target is not wired). Maven Central publishing is pending -- consume via composite build or the Forgejo Maven package described below.

The hand-written wrapper sources under `src/main/kotlin/dev/zorpx/blazen/{Workflow,Pipeline,LLM,...}.kt` are scaffolding. They declare value records that mirror the generated types, but do not yet bridge to the live FFI surface. **For now, import directly from the UniFFI-generated package `dev.zorpx.blazen.uniffi.*`** -- that is what the test suite (`WorkflowTest.kt`) and every working call path in this binding actually use.

## Install

Maven Central publishing has not landed yet. Two options today:

**Composite build (recommended while iterating):**

```kotlin
// settings.gradle.kts
includeBuild("../path/to/Blazen/bindings/kotlin")

// build.gradle.kts
dependencies {
    implementation("dev.zorpx.blazen:blazen-kotlin:0.1.0")
}
```

**Forgejo Maven package** (URL TBD -- once the Forgejo release workflow publishes the jar, point your `repositories { maven { url = uri("...") } }` block at it and depend on the same coordinates).

## Hello workflow

```kotlin
import dev.zorpx.blazen.uniffi.Event
import dev.zorpx.blazen.uniffi.StepHandler
import dev.zorpx.blazen.uniffi.StepOutput
import dev.zorpx.blazen.uniffi.WorkflowBuilder
import kotlinx.coroutines.runBlocking

fun main() = runBlocking {
    val handler = object : StepHandler {
        override suspend fun invoke(event: Event): StepOutput {
            // StartEvent wire shape: {"data": {"message": "hello"}}
            val payload = """{"result":{"echo":"hello"}}"""
            return StepOutput.Single(
                Event(eventType = "blazen::StopEvent", dataJson = payload),
            )
        }
    }

    val builder = WorkflowBuilder("echo")
    val workflow = builder.step(
        name = "echo",
        accepts = listOf("blazen::StartEvent"),
        emits = listOf("blazen::StopEvent"),
        handler = handler,
    ).build()

    try {
        val result = workflow.run("""{"message":"hello"}""")
        println(result.event.eventType)  // blazen::StopEvent
        println(result.event.dataJson)   // {"result":{"echo":"hello"}}
    } finally {
        workflow.close()
        builder.close()
    }
}
```

`Workflow`, `WorkflowBuilder`, and most FFI handles implement `AutoCloseable`; pair every constructor with `.close()` (or wrap in `use { }`) so the native handle is released.

## Async

`StepHandler.invoke` is a `suspend fun`, so it composes with kotlinx.coroutines out of the box -- launch from any `CoroutineScope`, call other suspend functions, await `Deferred`s. The FFI runs on uniffi-rs's tokio runtime under the hood, so the bridge is genuinely async: returning from a handler does not block a JVM thread waiting on Rust.

Use `Workflow.runBlocking(inputJson)` from non-coroutine contexts if you need a synchronous entry point.

## Error handling

Every error the Rust core can raise surfaces as a variant of the sealed `BlazenException` class. Variants carry structured fields (status codes, `retryAfterMs`, request ids, provider kind discriminators), so callers branch on type instead of parsing messages:

```kotlin
import dev.zorpx.blazen.uniffi.BlazenException

try {
    val result = workflow.run(inputJson)
} catch (e: BlazenException) {
    when (e) {
        is BlazenException.RateLimit -> Thread.sleep(e.retryAfterMs?.toLong() ?: 1_000)
        is BlazenException.Auth      -> rotateApiKey()
        is BlazenException.Provider  -> log.warn("provider ${e.provider} failed: ${e.kind}")
        is BlazenException.Workflow  -> log.error("workflow failure", e)
        else                         -> throw e
    }
}
```

## Native library loading

JNA selects `libblazen_uniffi.{so,dylib,dll}` from `src/main/resources/<jna-platform>/` based on the running JVM's OS + architecture (`linux-x86-64`, `linux-aarch64`, `darwin-aarch64`, `win32-x86-64`, etc.). Linux x86_64 and Linux aarch64 ship out of the box; other platforms come from `scripts/build-uniffi-lib.sh` or the Forgejo release artefacts. Touching any FFI symbol (for example `Blazen.version`) triggers JNA's classpath scan and loads the shared library.

## Where to go from here

- [Kotlin quickstart](https://blazen.dev/docs/guides/kotlin/quickstart)
- [LLM providers](https://blazen.dev/docs/guides/kotlin/llm)
- [Streaming events](https://blazen.dev/docs/guides/kotlin/streaming)
- [Agent loop](https://blazen.dev/docs/guides/kotlin/agent)

## License

MPL-2.0 -- see the [root LICENSE](../../LICENSE) for the full text.
