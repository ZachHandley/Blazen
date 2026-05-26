//go:build linux && amd64

package blazen

// cgo linker directives for the bundled libblazen_uniffi.a static archive.
//
// The corresponding archive must exist at
// `internal/clib/linux_amd64/libblazen_uniffi.a`. Phase H of the binding
// rollout adds matching prebuilt archives for darwin_arm64, darwin_amd64,
// linux_arm64, and windows_amd64 alongside their own build tags.

// The -Wl,allow-multiple-definition LDFLAG (passed via the cgo line
// below) tolerates duplicate ggml symbols vendored independently by
// whisper-rs-sys, llama-cpp-sys-2, and diffusion-rs-sys. All three
// bundle the same upstream ggml C sources, so the duplicates are
// functionally equivalent and the linker is free to pick either copy.

/*
#cgo LDFLAGS: -L${SRCDIR}/internal/clib/linux_amd64 -lblazen_uniffi -ldl -lm -lpthread -lstdc++ -Wl,--allow-multiple-definition
*/
import "C"
