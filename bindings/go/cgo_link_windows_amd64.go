//go:build windows && amd64

package blazen

// cgo linker directives for the bundled blazen_uniffi.lib static archive
// (Windows x86_64 MSVC). The archive must exist at
// `internal/clib/windows_amd64/blazen_uniffi.lib` — produced by
// `scripts/build-uniffi-lib.sh windows_amd64`.
//
// The Windows system libraries listed below mirror the
// `cargo:rustc-link-lib` directives that the Rust build emits for the
// MSVC target (Win32 sockets, secure-channel/crypto, user env, ntdll).
// If the upstream dep graph adds a new native-static dependency, append
// it here and re-run `scripts/build-uniffi-lib.sh windows_amd64`.

// #cgo LDFLAGS: -L${SRCDIR}/internal/clib/windows_amd64 -lblazen_uniffi -lws2_32 -luserenv -lbcrypt -lncrypt -lntdll -ladvapi32 -lcrypt32 -lsecur32 -lkernel32 -liphlpapi
import "C"
