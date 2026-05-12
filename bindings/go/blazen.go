package blazen

import (
	uniffiblazen "github.com/zorpxinc/blazen-go/internal/uniffi/blazen"
)

// Init initializes the Blazen runtime.
//
// The first call sets up the embedded Tokio runtime that drives every
// async operation across the FFI boundary, configures default
// tracing-subscriber output (controlled by RUST_LOG), and prepares the
// native side for use. Subsequent calls are no-ops, so it is safe to
// call from multiple goroutines or initialisation paths.
//
// Most package functions call Init internally, so callers usually do not
// need to invoke it explicitly — but doing so eagerly avoids the small
// startup cost on the first real request.
func Init() {
	uniffiblazen.Init()
}

// Version returns the version of the embedded blazen-uniffi native
// library. Useful for diagnosing version-skew between the Go module and
// the bundled native lib.
func Version() string {
	return uniffiblazen.Version()
}

// Shutdown flushes telemetry exporters and any other shutdown work that
// the native runtime needs before the host process exits. This is
// idempotent and safe to call from a defer in a long-running program.
//
// If no telemetry exporters were initialised (InitLangfuse / InitOtlp /
// InitPrometheus), Shutdown is a no-op.
func Shutdown() {
	uniffiblazen.ShutdownTelemetry()
}
