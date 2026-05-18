package dev.zorpx.blazen

import dev.zorpx.blazen.uniffi.BlazenException
import org.junit.jupiter.api.Disabled
import org.junit.jupiter.api.Test

/**
 * Structural test for the caller-error preservation path on the Kotlin
 * binding.
 *
 * ## Background
 *
 * Blazen 0.5.4 adds a typed `CallerError` variant to the workspace's
 * [BlazenException] sealed class. The contract: when a foreign-language
 * tool handler throws a typed Kotlin exception, the Rust UniFFI adapter
 * converts it into a `BlazenError::CallerError { name: Option<String>,
 * message: String, propertiesJson: String }` and propagates it back
 * through `run(userInput:)` so the host caller can sealed-class
 * pattern-match the variant, inspect the original exception's class name,
 * and JSON-decode the structured payload.
 *
 * As of this scaffold's authoring, `./scripts/regen-bindings.sh` (which
 * regenerates Kotlin via the local `uniffi-bindgen`) hasn't yet been run
 * against the 0.5.4 UDL, so the generated `BlazenException.CallerError`
 * subclass doesn't exist in `uniffi/blazen.kt`. This test pins the
 * EXPECTED post-regen shape:
 *
 *   1. A typed Kotlin exception a tool handler can throw.
 *   2. A [dev.zorpx.blazen.uniffi.ToolHandler] that throws that typed
 *      exception.
 *   3. The agent surfaces the failure as a
 *      `BlazenException.CallerError(name, message, propertiesJson)`
 *      whose `name == "MyCallerError"` and a JSON-decodable
 *      `propertiesJson` blob carrying the payload.
 *
 * The test is `@Disabled` right now so the suite remains green until the
 * regen surfaces the variant. Once `./scripts/regen-bindings.sh` lands
 * the new subclass, remove the annotation and uncomment the real
 * assertions below.
 */
@Disabled("needs CallerError variant regen via ./scripts/regen-bindings.sh")
class CallerErrorTest {
    /**
     * A typed Kotlin exception a tool handler might throw to signal a
     * domain-specific failure. The constructor-arg fields are the kind of
     * structured data that should survive the FFI round-trip via the
     * CallerError `propertiesJson` blob.
     */
    private class MyCallerError(
        val code: String,
        val detail: String,
    ) : RuntimeException("$code: $detail")

    /**
     * Mirrors the canonical post-regen test body. The references below
     * (`BlazenException.CallerError`, a mock model factory, the `Agent`
     * top-level constructor) don't yet exist in the generated bindings;
     * they're the EXPECTED post-regen API surface this test pins.
     *
     * Once the regen lands:
     *
     *   1. Remove the `@Disabled` class-level annotation.
     *   2. Uncomment the body below.
     *   3. The `BlazenException.CallerError` import + check should
     *      compile and pass.
     */
    @Test
    fun `tool handler caller error is preserved`() {
        // The block below is the post-regen test body. It compiles
        // against the EXPECTED public shape: a `BlazenException.CallerError`
        // sealed-class subtype mirroring the existing
        // `BlazenException.Provider` variant (with its own structured
        // accessors), surfaced by the agent's `run` path on tool-handler
        // failure.
        //
        // val model = Blazen.mockCompletionModel(modelId = "mock")  // regen helper
        // val tool = Tool(
        //     name = "domain_op",
        //     description = "Always throws a typed caller error.",
        //     parametersJson = """{"type":"object","properties":{}}""",
        // )
        // val handler = object : dev.zorpx.blazen.uniffi.ToolHandler {
        //     override suspend fun execute(
        //         toolName: String,
        //         argumentsJson: String,
        //     ): String = throw MyCallerError(code = "E_DOMAIN", detail = "tool refused")
        // }
        // val agent = dev.zorpx.blazen.uniffi.Agent(
        //     model = model,
        //     systemPrompt = null,
        //     tools = listOf(tool),
        //     toolHandler = handler,
        //     maxIterations = 2u,
        // )
        // val ex = assertThrows<BlazenException.CallerError> {
        //     runBlocking { agent.run("please call the tool") }
        // }
        // assertEquals("MyCallerError", ex.name)
        // val payload = Json.parseToJsonElement(ex.propertiesJson).jsonObject
        // assertEquals("E_DOMAIN", payload["code"]?.jsonPrimitive?.content)
        // assertEquals("tool refused", payload["detail"]?.jsonPrimitive?.content)

        // Keep `MyCallerError` and `BlazenException` referenced so unused
        // imports don't get pruned while parked behind @Disabled.
        val sentinel: BlazenException = BlazenException.Validation("scaffold")
        check(sentinel.message == "scaffold")
        val err = MyCallerError(code = "E_DOMAIN", detail = "tool refused")
        check(err.code == "E_DOMAIN")
    }
}
