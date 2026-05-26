//go:build linux && arm64

package blazen

// cgo linker directives for the bundled libblazen_uniffi.a static archive
// (Linux aarch64). The archive must exist at
// `internal/clib/linux_arm64/libblazen_uniffi.a` — produced by
// `scripts/build-uniffi-lib.sh linux_arm64`.

// The -Wl,allow-multiple-definition LDFLAG (passed via the cgo line
// below) tolerates duplicate ggml symbols vendored independently by
// whisper-rs-sys, llama-cpp-sys-2, and diffusion-rs-sys. See
// cgo_link.go for the full rationale.

/*
#cgo LDFLAGS: -L${SRCDIR}/internal/clib/linux_arm64 -lblazen_uniffi -ldl -lm -lpthread -lstdc++ -Wl,--allow-multiple-definition
*/
import "C"
