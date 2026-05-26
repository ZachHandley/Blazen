//go:build linux && amd64

package blazen

// cgo linker directives for the bundled libblazen_uniffi.a static archive.
//
// The corresponding archive must exist at
// `internal/clib/linux_amd64/libblazen_uniffi.a`. Phase H of the binding
// rollout adds matching prebuilt archives for darwin_arm64, darwin_amd64,
// linux_arm64, and windows_amd64 alongside their own build tags.

// The -Wl,allow-multiple-definition LDFLAG (passed via the cgo line
// below) tolerates duplicate symbols inside libblazen_uniffi.a. As of
// the Phase 3 ggml-dedup work (blazen-ggml-sys + the three [patch.crates-io]
// forks of whisper-rs-sys / diffusion-rs-sys / llama-cpp-sys-2), the
// historic ggml triple-vendoring is fixed — those three -sys crates all
// link against the single blazen-ggml-sys install tree via their new
// `system-ggml`/`system-ggml-static` cargo features.
//
// The flag is still needed because libblazen_uniffi.a also contains a
// statically-bundled protobuf (pulled in by the `distributed` /
// `controlplane` features through `tonic` → `prost` → `protobuf-c++`),
// and wire_format_lite.cc.o gets compiled into the archive via multiple
// translation-unit paths. Each translation-unit instance produces
// `multiple definition of google::protobuf::internal::WireFormatLite::*`
// at the final cgo static link. Dropping `--allow-multiple-definition`
// without first deduplicating that protobuf vendoring is a build break.
// Tracked as a Phase 3 follow-up.

/*
#cgo LDFLAGS: -L${SRCDIR}/internal/clib/linux_amd64 -lblazen_uniffi -ldl -lm -lpthread -lstdc++ -Wl,--allow-multiple-definition
*/
import "C"
