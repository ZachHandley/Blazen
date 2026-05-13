//go:build linux && arm64

package blazen

// cgo linker directives for the bundled libblazen_uniffi.a static archive
// (Linux aarch64). The archive must exist at
// `internal/clib/linux_arm64/libblazen_uniffi.a` — produced by
// `scripts/build-uniffi-lib.sh linux_arm64`.

// #cgo LDFLAGS: -L${SRCDIR}/internal/clib/linux_arm64 -lblazen_uniffi -ldl -lm -lpthread -lstdc++
import "C"
