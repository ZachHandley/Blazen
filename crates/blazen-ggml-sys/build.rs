//! Build script for `blazen-ggml-sys`.
//!
//! cmake-builds the vendored `ggml-org/ggml` source tree at `vendor/ggml/`
//! once, installs the artifacts into `${OUT_DIR}/install/`, and exposes
//! the install prefix via `cargo:prefix` / `cargo:include` / `cargo:lib`
//! so downstream `-sys` build scripts can pick them up through cargo's
//! `DEP_GGML_*` env vars.
//!
//! Modeled on `qts_ggml_sys`'s build.rs (which itself follows the
//! upstream `ggml-config.cmake` consumer contract).

use std::env;
use std::path::{Path, PathBuf};

fn main() {
    let target = env::var("TARGET").expect("TARGET");
    let out_dir = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR"));
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR"));

    let ggml_root = manifest_dir.join("vendor/ggml");
    let install_prefix = out_dir.join("install");

    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=vendor/ggml/CMakeLists.txt");
    println!("cargo:rerun-if-env-changed=GGML_BLAS_VENDOR");
    println!("cargo:rerun-if-env-changed=BLA_VENDOR");
    println!("cargo:rerun-if-env-changed=VULKAN_SDK");
    validate_features(&target);

    let mut cfg = cmake::Config::new(&ggml_root);
    cfg.profile("Release");
    cfg.define("BUILD_SHARED_LIBS", "OFF");
    cfg.define("GGML_STATIC", "ON");
    cfg.define("GGML_BUILD_EXAMPLES", "OFF");
    cfg.define("GGML_BUILD_TESTS", "OFF");

    if target.contains("msvc") {
        // Force the dynamic CRT (/MD). The ort-sys prebuilt and Rust's cdylib
        // default are /MD; the cmake C++ libs otherwise build /MT, which
        // collides at link time (LNK2038 RuntimeLibrary mismatch + LNK2005
        // duplicate std::locale symbols). CMP0091=NEW is required because these
        // CMakeLists predate the abstraction and would otherwise ignore the var.
        cfg.define("CMAKE_POLICY_DEFAULT_CMP0091", "NEW");
        cfg.define("CMAKE_MSVC_RUNTIME_LIBRARY", "MultiThreadedDLL");
        cfg.static_crt(false);
    }

    // The vendored ggml header has been patched to set GGML_MAX_NAME=128
    // (instead of the upstream default 64) so stable-diffusion.cpp — which
    // requires the constant to be at least 128 — can find_package this
    // build and have its static_assert pass. See
    // vendor/ggml/include/ggml.h around line 229 for the patch.
    // Install layout: `${install_prefix}/{lib,include,lib/cmake/ggml}` so
    // downstream `find_package(ggml REQUIRED)` resolves against our build.
    cfg.define("CMAKE_INSTALL_PREFIX", &install_prefix);

    // Force `lib/` (not `lib64/`). ggml's CMakeLists.txt does
    // `include(GNUInstallDirs)`, which on RHEL-based 64-bit hosts
    // (manylinux_2_28, RHEL/Rocky/Fedora) defaults CMAKE_INSTALL_LIBDIR
    // to `lib64`. Downstream code in this build.rs (and the install
    // paths the three -sys consumers reach into via DEP_GGML_LIB)
    // assume `lib/`. Pin it explicitly so the install layout is
    // identical across Debian/Ubuntu/manylinux/Alpine hosts.
    cfg.define("CMAKE_INSTALL_LIBDIR", "lib");

    if feature_enabled("native") {
        cfg.define("GGML_NATIVE", "ON");
    } else {
        cfg.define("GGML_NATIVE", "OFF");
    }

    // Metal is Apple-only; silently skip on non-Apple targets.
    if feature_enabled("metal") && target.contains("apple") {
        cfg.define("GGML_METAL", "ON");
        cfg.define("GGML_METAL_EMBED_LIBRARY", "ON");
    } else {
        cfg.define("GGML_METAL", "OFF");
    }

    cfg.define("GGML_BLAS", "OFF");

    // Force-disable GPU backends — we don't expose cargo features for
    // them today (see Cargo.toml). Setting these explicitly prevents
    // ggml's CMakeLists.txt defaults from auto-detecting host toolchains.
    cfg.define("GGML_CUDA", "OFF");
    cfg.define("GGML_VULKAN", "OFF");
    cfg.define("GGML_HIP", "OFF");

    // `cmake::Config::build()` runs cmake + cmake --build + cmake
    // --install (the install step honors CMAKE_INSTALL_PREFIX).
    let dst = cfg.build();

    let lib_dir = install_prefix.join("lib");
    let include_dir = install_prefix.join("include");

    // Copy ggml's internal `ggml-ext.h` into the install/include tree.
    // llama.cpp (when consuming us via LLAMA_USE_SYSTEM_GGML=ON) directly
    // `#include "../src/ggml-ext.h"` from `src/llama.cpp` + `src/llama-model.cpp`
    // — that include is satisfied by llama.cpp's own internal path
    // resolution when ggml is built as a subproject, but when we install
    // ggml standalone, cmake's `install(FILES ...)` only copies the
    // `include/*.h` public surface and skips this private header. Copy
    // it manually so downstream cmake consumers can `find_package(ggml)`
    // and still get this header on the include path via
    // `${ggml_INCLUDE_DIRS}`.
    let private_header = ggml_root.join("src/ggml-ext.h");
    if private_header.exists() {
        std::fs::copy(&private_header, include_dir.join("ggml-ext.h"))
            .expect("install ggml-ext.h to include dir");
    }

    // Cargo metadata that downstream `-sys` build scripts read via
    // `DEP_GGML_PREFIX` / `DEP_GGML_INCLUDE` / `DEP_GGML_LIB`.
    println!("cargo:prefix={}", install_prefix.display());
    println!("cargo:include={}", include_dir.display());
    println!("cargo:lib={}", lib_dir.display());

    println!("cargo:rustc-link-search=native={}", lib_dir.display());

    // Direct consumers (this crate as a Rust dep) get the static libs
    // linked in. The `-sys` crates that go through cmake's
    // `find_package(ggml)` get the same libs via the install tree.
    println!("cargo:rustc-link-lib=static=ggml");
    if feature_enabled("metal") && target.contains("apple") {
        println!("cargo:rustc-link-lib=static=ggml-metal");
    }
    // GPU-backend link lines intentionally omitted — see Cargo.toml.

    println!("cargo:rustc-link-lib=static=ggml-cpu");
    println!("cargo:rustc-link-lib=static=ggml-base");

    if feature_enabled("metal") && target.contains("apple") {
        println!("cargo:rustc-link-lib=framework=Metal");
        println!("cargo:rustc-link-lib=framework=MetalKit");
        println!("cargo:rustc-link-lib=framework=Foundation");
    }
    if target.contains("apple") {
        println!("cargo:rustc-link-lib=framework=Accelerate");
    }

    if target.contains("apple") {
        println!("cargo:rustc-link-lib=c++");
    } else if !(target.contains("windows") && target.contains("msvc")) {
        println!("cargo:rustc-link-lib=stdc++");
    }

    if target.contains("linux") {
        println!("cargo:rustc-link-lib=gomp");
        println!("cargo:rustc-link-lib=pthread");
        println!("cargo:rustc-link-lib=m");
        println!("cargo:rustc-link-lib=dl");
    }

    let _ = dst; // silence unused
}

fn feature_enabled(name: &str) -> bool {
    env::var(format!(
        "CARGO_FEATURE_{}",
        name.to_ascii_uppercase().replace('-', "_")
    ))
    .is_ok()
}

#[allow(dead_code)]
fn map_feature_cmake(cfg: &mut cmake::Config, feature: &str, cmake_opt: &str) {
    if feature_enabled(feature) {
        cfg.define(cmake_opt, "ON");
    }
}

fn validate_features(target: &str) {
    if feature_enabled("metal") && !target.contains("apple") {
        println!(
            "cargo:warning=blazen-ggml-sys: `metal` feature ignored on non-Apple target ({target})"
        );
    }
}

// Tell rustc about the `Path` import even though we don't use it directly.
const _: fn() = || {
    let _ = Path::new("");
};
