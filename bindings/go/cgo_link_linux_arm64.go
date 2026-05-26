//go:build linux && arm64

package blazen

// cgo linker directives for the bundled libblazen_uniffi.a static archive
// (Linux aarch64). The archive must exist at
// `internal/clib/linux_arm64/libblazen_uniffi.a` — produced by
// `scripts/build-uniffi-lib.sh linux_arm64`.

// The -Wl,allow-multiple-definition LDFLAG (passed via the cgo line
// below) tolerates duplicate symbols inside libblazen_uniffi.a. The
// historic ggml triple-vendoring is now fixed by the Phase 3
// blazen-ggml-sys work — see cgo_link.go for the full rationale and
// the remaining protobuf vendoring follow-up that still needs the flag.

/*
#cgo LDFLAGS: -L${SRCDIR}/internal/clib/linux_arm64 -lblazen_uniffi -ldl -lm -lpthread -lstdc++ -Wl,--allow-multiple-definition
*/
import "C"
