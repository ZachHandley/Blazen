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
    // Install layout: `${install_prefix}/{lib,include,lib/cmake/ggml}` so
    // downstream `find_package(ggml REQUIRED)` resolves against our build.
    cfg.define("CMAKE_INSTALL_PREFIX", &install_prefix);

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

    if feature_enabled("blas") {
        cfg.define("GGML_BLAS", "ON");
        if let Some(blas_vendor) = env::var("GGML_BLAS_VENDOR")
            .ok()
            .or_else(|| env::var("BLA_VENDOR").ok())
            .filter(|v| !v.trim().is_empty())
        {
            cfg.define("BLA_VENDOR", &blas_vendor);
        }
        if target.contains("apple") {
            cfg.define("GGML_ACCELERATE", "ON");
        }
    } else {
        cfg.define("GGML_BLAS", "OFF");
    }

    map_feature_cmake(&mut cfg, "cuda", "GGML_CUDA");
    map_feature_cmake(&mut cfg, "vulkan", "GGML_VULKAN");
    map_feature_cmake(&mut cfg, "hip", "GGML_HIP");
    map_feature_cmake(&mut cfg, "musa", "GGML_MUSA");
    map_feature_cmake(&mut cfg, "opencl", "GGML_OPENCL");
    map_feature_cmake(&mut cfg, "rpc", "GGML_RPC");
    map_feature_cmake(&mut cfg, "sycl", "GGML_SYCL");
    map_feature_cmake(&mut cfg, "webgpu", "GGML_WEBGPU");
    map_feature_cmake(&mut cfg, "openvino", "GGML_OPENVINO");
    map_feature_cmake(&mut cfg, "hexagon", "GGML_HEXAGON");
    map_feature_cmake(&mut cfg, "cann", "GGML_CANN");
    map_feature_cmake(&mut cfg, "zendnn", "GGML_ZENDNN");
    map_feature_cmake(&mut cfg, "zdnn", "GGML_ZDNN");
    map_feature_cmake(&mut cfg, "virtgpu", "GGML_VIRTGPU");

    // `cmake::Config::build()` runs cmake + cmake --build + cmake
    // --install (the install step honors CMAKE_INSTALL_PREFIX).
    let dst = cfg.build();

    let lib_dir = install_prefix.join("lib");
    let include_dir = install_prefix.join("include");

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
    if feature_enabled("cuda") {
        println!("cargo:rustc-link-lib=static=ggml-cuda");
    }
    if feature_enabled("vulkan") {
        println!("cargo:rustc-link-lib=static=ggml-vulkan");
        emit_vulkan_loader_links(&target);
    }
    if feature_enabled("hip") {
        println!("cargo:rustc-link-lib=static=ggml-hip");
    }
    if feature_enabled("musa") {
        println!("cargo:rustc-link-lib=static=ggml-musa");
    }
    if feature_enabled("opencl") {
        println!("cargo:rustc-link-lib=static=ggml-opencl");
    }
    if feature_enabled("blas") {
        println!("cargo:rustc-link-lib=static=ggml-blas");
    }
    if feature_enabled("rpc") {
        println!("cargo:rustc-link-lib=static=ggml-rpc");
    }
    if feature_enabled("sycl") {
        println!("cargo:rustc-link-lib=static=ggml-sycl");
    }
    if feature_enabled("webgpu") {
        println!("cargo:rustc-link-lib=static=ggml-webgpu");
    }
    if feature_enabled("openvino") {
        println!("cargo:rustc-link-lib=static=ggml-openvino");
    }
    if feature_enabled("hexagon") {
        println!("cargo:rustc-link-lib=static=ggml-hexagon");
    }
    if feature_enabled("cann") {
        println!("cargo:rustc-link-lib=static=ggml-cann");
    }
    if feature_enabled("zendnn") {
        println!("cargo:rustc-link-lib=static=ggml-zendnn");
    }
    if feature_enabled("zdnn") {
        println!("cargo:rustc-link-lib=static=ggml-zdnn");
    }
    if feature_enabled("virtgpu") {
        println!("cargo:rustc-link-lib=static=ggml-virtgpu");
    }

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
    if feature_enabled("cuda") && target.contains("apple") {
        panic!("blazen-ggml-sys: `cuda` feature is not supported on Apple targets");
    }
}

fn emit_vulkan_loader_links(target: &str) {
    for dir in vulkan_search_dirs(target) {
        if dir.exists() {
            println!("cargo:rustc-link-search=native={}", dir.display());
        }
    }
    let lib = if target.contains("apple") {
        "dylib=vulkan"
    } else if target.contains("windows") {
        "vulkan-1"
    } else {
        "vulkan"
    };
    println!("cargo:rustc-link-lib={lib}");
}

fn vulkan_search_dirs(target: &str) -> Vec<PathBuf> {
    let mut dirs = Vec::new();
    if let Ok(sdk) = env::var("VULKAN_SDK") {
        let sdk = PathBuf::from(sdk);
        if target.contains("windows") {
            dirs.push(sdk.join("Lib"));
        } else {
            dirs.push(sdk.join("lib"));
            if target.contains("apple") {
                dirs.push(sdk.join("macOS").join("lib"));
            }
        }
    }
    if target.contains("apple") {
        dirs.push(PathBuf::from("/opt/homebrew/lib"));
        dirs.push(PathBuf::from("/usr/local/lib"));
    }
    dirs
}

// Tell rustc about the `Path` import even though we don't use it directly.
const _: fn() = || {
    let _ = Path::new("");
};
