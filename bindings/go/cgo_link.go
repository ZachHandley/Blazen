//go:build linux && amd64

package blazen

// cgo linker directives for the bundled libblazen_uniffi.a static archive.
//
// The corresponding archive must exist at
// `internal/clib/linux_amd64/libblazen_uniffi.a`. Phase H of the binding
// rollout adds matching prebuilt archives for darwin_arm64, darwin_amd64,
// linux_arm64, and windows_amd64 alongside their own build tags.

// #cgo LDFLAGS: -L${SRCDIR}/internal/clib/linux_amd64 -lblazen_uniffi -ldl -lm -lpthread -lstdc++
import "C"
