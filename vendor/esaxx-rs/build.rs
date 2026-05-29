// Blazen patch of esaxx-rs 0.1.10's build.rs. Upstream hardcodes
// `.static_crt(true)` on both arms, which forces `/MT` (static CRT) on
// Windows MSVC. Rust's MSVC default for cdylib targets is `/MD` (dynamic
// CRT), as is the pyke ort-sys prebuilt and every cmake-driven sys crate
// in the build graph. The hardcoded `/MT` here was producing
//   LNK2038 'RuntimeLibrary' mismatch: MT_StaticRelease vs MD_DynamicRelease
// on every Windows MSVC `cdylib` consumer of tokenizers' esaxx_fast
// feature (so: blazen-uniffi, blazen-cabi via tokenizers transitively
// from ct2rs/candle-core/...).
//
// The fix is to drop the hardcoded `.static_crt(true)` and let cc-rs
// honor `CARGO_CFG_TARGET_FEATURE` (`+crt-static` → /MT, `-crt-static`
// → /MD). esaxx-rs has no business forcing the CRT — every other cc-rs
// crate in the ecosystem follows the consumer's choice.

#[cfg(feature = "cpp")]
#[cfg(not(target_os = "macos"))]
fn main() {
    cc::Build::new()
        .cpp(true)
        .flag("-std=c++11")
        .file("src/esaxx.cpp")
        .include("src")
        .compile("esaxx");
}

#[cfg(feature = "cpp")]
#[cfg(target_os = "macos")]
fn main() {
    cc::Build::new()
        .cpp(true)
        .flag("-std=c++11")
        .flag("-stdlib=libc++")
        .file("src/esaxx.cpp")
        .include("src")
        .compile("esaxx");
}

#[cfg(not(feature = "cpp"))]
fn main() {}
