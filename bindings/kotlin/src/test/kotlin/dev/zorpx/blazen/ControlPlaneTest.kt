package dev.zorpx.blazen

import dev.zorpx.blazen.uniffi.BlazenException
import dev.zorpx.blazen.uniffi.ControlPlaneAdmission
import dev.zorpx.blazen.uniffi.ControlPlaneAdmissionMode
import dev.zorpx.blazen.uniffi.ControlPlaneClient
import dev.zorpx.blazen.uniffi.ControlPlaneRunStatus
import dev.zorpx.blazen.uniffi.ControlPlaneSubmitRequest
import dev.zorpx.blazen.uniffi.ControlPlaneWorker
import dev.zorpx.blazen.uniffi.ControlPlaneWorkerCapability
import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Assertions.assertNotNull
import org.junit.jupiter.api.Assertions.assertThrows
import org.junit.jupiter.api.Assertions.assertTrue
import org.junit.jupiter.api.Test

/**
 * Smoke + shape tests for the control-plane UniFFI surface. These do
 * not require a running control-plane server — they exercise the
 * type-construction path and the error path of the FFI without
 * standing up a transport.
 */
class ControlPlaneTest {
    @Test
    fun `worker capability constructs and keeps fields`() {
        val cap = ControlPlaneWorkerCapability(kind = "workflow:hello", version = 1u)
        assertEquals("workflow:hello", cap.kind)
        assertEquals(1u, cap.version)
    }

    @Test
    fun `admission with fixed mode keeps max in flight`() {
        val admission = ControlPlaneAdmission(
            mode = ControlPlaneAdmissionMode.FIXED,
            maxInFlight = 4u,
            totalMb = null,
        )
        assertEquals(ControlPlaneAdmissionMode.FIXED, admission.mode)
        assertEquals(4u, admission.maxInFlight)
        assertEquals(null, admission.totalMb)
    }

    @Test
    fun `admission with reactive mode keeps both nullable fields null`() {
        val admission = ControlPlaneAdmission(
            mode = ControlPlaneAdmissionMode.REACTIVE,
            maxInFlight = null,
            totalMb = null,
        )
        assertEquals(ControlPlaneAdmissionMode.REACTIVE, admission.mode)
        assertEquals(null, admission.maxInFlight)
        assertEquals(null, admission.totalMb)
    }

    @Test
    fun `submit request constructs with all required fields`() {
        val req = ControlPlaneSubmitRequest(
            workflowName = "summarize",
            inputJson = """{"text":"hello"}""",
            workflowVersion = null,
            requiredTags = listOf("region=us-west"),
            idempotencyKey = "dedupe-1",
            deadlineMs = 60_000UL,
            waitForWorker = true,
        )
        assertEquals("summarize", req.workflowName)
        assertTrue(req.waitForWorker)
        assertEquals(listOf("region=us-west"), req.requiredTags)
    }

    @Test
    fun `run status enum has all five variants`() {
        // Sanity-check the wire enum surface — if a new variant was
        // dropped or renamed accidentally, this fails to compile.
        val statuses = listOf(
            ControlPlaneRunStatus.PENDING,
            ControlPlaneRunStatus.RUNNING,
            ControlPlaneRunStatus.COMPLETED,
            ControlPlaneRunStatus.FAILED,
            ControlPlaneRunStatus.CANCELLED,
        )
        assertEquals(5, statuses.size)
    }

    @Test
    fun `client blocking constructor surfaces transport error on bad URI`() {
        val err = assertThrows(BlazenException::class.java) {
            ControlPlaneClient.connectBlocking("not a uri")
        }
        assertNotNull(err)
        val msg = (err.message ?: "").lowercase()
        assertTrue(
            msg.contains("transport") || msg.contains("invalid") || msg.contains("uri"),
            "expected transport/invalid/uri in error message, got '$msg'",
        )
    }

    @Test
    fun `worker blocking constructor validates endpoint URI eagerly`() {
        val err = assertThrows(BlazenException::class.java) {
            ControlPlaneWorker.newBlocking(
                endpoint = "not a uri",
                nodeId = "node-test",
                capabilities = listOf(
                    ControlPlaneWorkerCapability("workflow:hello", 1u),
                ),
            )
        }
        assertNotNull(err)
        val msg = (err.message ?: "").lowercase()
        assertTrue(
            msg.contains("transport") || msg.contains("invalid") || msg.contains("uri"),
            "expected transport/invalid/uri in error message, got '$msg'",
        )
    }
}
