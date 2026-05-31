// Forked from upstream diffusion-rs-sys 0.1.19 — keep clippy quiet on
// upstream patterns the Blazen workspace's lint policy normally flags.
#![allow(clippy::all)]

use std::{
    env,
    fs::{create_dir_all, read_dir},
    path::{Path, PathBuf},
};

use cmake::Config;
use fs_extra::dir;

// Inspired by https://github.com/tazz4843/whisper-rs/blob/master/sys/build.rs

// Blazen fork: GPU-backend toolchain auto-detection. When a GPU cargo
// feature is enabled but the matching SDK/toolchain is absent on the host
// (e.g. building `cuda`/`hipblas`/`vulkan`/`sycl` on a Mac), gracefully
// skip that backend with a `cargo:warning=` instead of panicking. When the
// toolchain IS present, behavior is byte-identical to before.
//
// These helpers are only referenced from `#[cfg(feature = "...")]` GPU
// blocks, so in a default (CPU/Metal) build they go unused — silence the
// resulting dead_code warning.
#[allow(dead_code)]
fn cmd_ok(cmd: &str, arg: &str) -> bool {
    std::process::Command::new(cmd)
        .arg(arg)
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}
#[allow(dead_code)]
fn any_env_dir(vars: &[&str]) -> bool {
    vars.iter().any(|v| {
        std::env::var(v)
            .map(|p| !p.is_empty() && std::path::Path::new(&p).exists())
            .unwrap_or(false)
    })
}
#[allow(dead_code)]
fn cuda_present() -> bool {
    cmd_ok("nvcc", "--version")
        || any_env_dir(&["CUDA_PATH", "CUDA_HOME", "CUDA_TOOLKIT_ROOT_DIR"])
}
#[allow(dead_code)]
fn rocm_present() -> bool {
    cmd_ok("hipcc", "--version")
        || any_env_dir(&["HIP_PATH", "ROCM_PATH"])
        || std::path::Path::new("/opt/rocm").exists()
}
#[allow(dead_code)]
fn vulkan_present() -> bool {
    any_env_dir(&["VULKAN_SDK"]) || cmd_ok("glslc", "--version")
}
#[allow(dead_code)]
fn sycl_present() -> bool {
    any_env_dir(&["ONEAPI_ROOT"])
}

fn main() {
    // Link C++ standard library
    let target = env::var("TARGET").unwrap();
    if let Some(cpp_stdlib) = get_cpp_link_stdlib(&target) {
        println!("cargo:rustc-link-lib=dylib={cpp_stdlib}");
    }

    println!("cargo:rerun-if-changed=wrapper.h");

    // Copy stable-diffusion code into the build script directory
    let out = PathBuf::from(env::var("OUT_DIR").unwrap());
    let diffusion_root = out.join("stable-diffusion.cpp/");

    if !diffusion_root.exists() {
        create_dir_all(&diffusion_root).unwrap();
        dir::copy("./stable-diffusion.cpp", &out, &Default::default()).unwrap_or_else(|e| {
            panic!(
                "Failed to copy stable-diffusion sources into {}: {}",
                diffusion_root.display(),
                e
            )
        });
    }

    // Bindgen
    let bindings = bindgen::Builder::default()
        .header("wrapper.h")
        .clang_arg("-I./stable-diffusion.cpp")
        .clang_arg("-I./stable-diffusion.cpp/ggml/include")
        .rustified_non_exhaustive_enum(".*")
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
        .generate()
        .unwrap()
        .write_to_file(out.join("bindings.rs"));

    if let Err(e) = bindings {
        println!("cargo:warning=Unable to generate bindings: {e}");
        println!("cargo:warning=Using bundled bindings.rs, which may be out of date");
    }

    // stop if we're on docs.rs
    if env::var("DOCS_RS").is_ok() {
        return;
    }

    // Configure cmake for building
    let mut config = Config::new(&diffusion_root);

    if target.contains("msvc") {
        config.generator("Ninja");
        config.define("CMAKE_BUILD_TYPE", "Release");
        config.define("CMAKE_C_COMPILER", "cl.exe");
        config.define("CMAKE_CXX_COMPILER", "cl.exe");
        config.define("CMAKE_CXX_FLAGS", "'/bigobj'");
        // Force the dynamic CRT (/MD). The ort-sys prebuilt and Rust's cdylib
        // default are /MD; stable-diffusion.cpp otherwise builds /MT, which
        // collides at link time (LNK2038 RuntimeLibrary mismatch + LNK2005
        // duplicate std::locale symbols). CMP0091=NEW is required because the
        // CMakeLists predates the abstraction and would otherwise ignore the
        // runtime-library var.
        config.define("CMAKE_POLICY_DEFAULT_CMP0091", "NEW");
        config.define("CMAKE_MSVC_RUNTIME_LIBRARY", "MultiThreadedDLL");
        config.static_crt(false);
    }
    config
        .profile("Release")
        .define("SD_BUILD_SHARED_LIBS", "OFF")
        .define("SD_BUILD_EXAMPLES", "OFF")
        .define("SD_BUILD_SERVER", "OFF")
        .define("GGML_OPENMP", "OFF")
        .very_verbose(true)
        .pic(true);

    // Blazen fork: route ggml through blazen-ggml-sys when the
    // `system-ggml` cargo feature is on. Upstream stable-diffusion.cpp's
    // cmake exposes `SD_USE_SYSTEM_GGML` which makes its build call
    // `find_package(ggml REQUIRED)` instead of compiling the in-tree
    // ggml subtree. We point that find_package at the blazen-ggml-sys
    // install prefix via CMAKE_PREFIX_PATH (DEP_GGML_PREFIX is set by
    // cargo because blazen-ggml-sys declares `links = "ggml"` and is
    // declared as both a regular dep AND a build-dep in Cargo.toml).
    #[cfg(feature = "system-ggml")]
    {
        let ggml_prefix = env::var("DEP_GGML_PREFIX").expect(
            "DEP_GGML_PREFIX must be set when `system-ggml` feature is on (provided by blazen-ggml-sys via its `links = \"ggml\"` Cargo manifest entry)",
        );
        println!(
            "cargo:warning=blazen-diffusion-sys: routing ggml via blazen-ggml-sys at {ggml_prefix}"
        );
        config.define("SD_USE_SYSTEM_GGML", "ON");
        config.define("CMAKE_PREFIX_PATH", &ggml_prefix);
        println!("cargo:rerun-if-env-changed=DEP_GGML_PREFIX");
    }

    //Enable cmake feature flags
    #[cfg(feature = "cuda")]
    if cuda_present() {
        println!("cargo:rerun-if-env-changed=CUDA_PATH");
        println!("cargo:rustc-link-lib=cublas");
        println!("cargo:rustc-link-lib=cudart");
        println!("cargo:rustc-link-lib=cublasLt");
        println!("cargo:rustc-link-lib=cuda");

        if target.contains("msvc") {
            let cuda_path = PathBuf::from(
                env::var("CUDA_PATH").expect("CUDA toolchain detected but CUDA_PATH env variable is not set"),
            )
            .join("lib/x64");
            println!("cargo:rustc-link-search={}", cuda_path.display());
        } else {
            println!("cargo:rustc-link-lib=culibos");
            println!("cargo:rustc-link-search=/usr/local/cuda/lib64");
            println!("cargo:rustc-link-search=/usr/local/cuda/lib64/stubs");
            println!("cargo:rustc-link-search=/opt/cuda/lib64");
            println!("cargo:rustc-link-search=/opt/cuda/lib64/stubs");
        }

        config.define("SD_CUDA", "ON");
        if let Ok(target) = env::var("CUDA_COMPUTE_CAP") {
            config.define("CUDA_COMPUTE_CAP", target);
        }
    } else {
        println!("cargo:warning=feature `cuda` enabled but CUDA toolchain not detected — skipping CUDA backend.");
    }

    #[cfg(feature = "hipblas")]
    if rocm_present() {
        println!("cargo:rerun-if-env-changed=HIP_PATH");
        println!("cargo:rustc-link-lib=hipblas");
        println!("cargo:rustc-link-lib=rocblas");
        println!("cargo:rustc-link-lib=amdhip64");

        config.generator("Ninja");
        config.define("CMAKE_C_COMPILER", "clang");
        config.define("CMAKE_CXX_COMPILER", "clang++");
        config.define("CMAKE_BUILD_WITH_INSTALL_RPATH", "ON");
        config.define("CMAKE_POSITION_INDEPENDENT_CODE", "ON");
        let hip_lib_path = if target.contains("msvc") {
            let hip_path = env::var("HIP_PATH").expect("Missing HIP_PATH env variable");
            PathBuf::from(hip_path).join("lib")
        } else {
            let hip_path = match env::var("HIP_PATH") {
                Ok(path) => PathBuf::from(path),
                Err(_) => PathBuf::from("/opt/rocm"),
            };
            hip_path.join("lib")
        };
        println!("cargo:rustc-link-search={}", hip_lib_path.display());

        config.define("SD_HIPBLAS", "ON");
        if let Ok(target) = env::var("GFX_NAME") {
            config.define("AMDGPU_TARGETS", &target);
            config.define("GPU_TARGETS", target);
        }
    } else {
        println!("cargo:warning=feature `hipblas` enabled but ROCm/HIP toolchain not detected — skipping HIPBLAS backend.");
    }

    #[cfg(feature = "metal")]
    {
        config.define("SD_METAL", "ON");
        println!("cargo:rustc-link-lib=framework=Foundation");
        println!("cargo:rustc-link-lib=framework=Metal");
        println!("cargo:rustc-link-lib=framework=MetalKit");
    }

    #[cfg(feature = "vulkan")]
    if vulkan_present() {
        let vulkan_path = env::var("VULKAN_SDK").map(|path| PathBuf::from(path));
        if target.contains("msvc") {
            println!("cargo:rerun-if-env-changed=VULKAN_SDK");
            println!("cargo:rustc-link-lib=vulkan-1");

            let vulkan_lib_path = vulkan_path
                .expect("Please install Vulkan SDK and ensure that VULKAN_SDK env variable is set")
                .join("Lib");
            println!("cargo:rustc-link-search={}", vulkan_lib_path.display());
        } else {
            if let Ok(vulkan_path) = vulkan_path {
                let vulkan_lib_path = vulkan_path.join("lib");
                println!("cargo:rustc-link-search={}", vulkan_lib_path.display());
            }
            if target.contains("darwin") {
                println!("cargo:rustc-link-search=/usr/local/lib");
            }
            println!("cargo:rustc-link-lib=vulkan");
        }
        config.define("SD_VULKAN", "ON");
    } else {
        println!("cargo:warning=feature `vulkan` enabled but Vulkan SDK not detected — skipping Vulkan backend.");
    }

    #[cfg(feature = "sycl")]
    if sycl_present() {
        env::var("ONEAPI_ROOT").expect("Please load the oneAPi environment before building. See https://github.com/ggerganov/llama.cpp/blob/master/docs/backend/SYCL.md");
        let sycl_lib_path = PathBuf::from(env::var("ONEAPI_ROOT").unwrap()).join("mkl/latest/lib");
        println!("cargo:rustc-link-search={}", sycl_lib_path.display());

        println!("cargo:rustc-link-lib=static=mkl_sycl");
        println!("cargo:rustc-link-lib=static=mkl_core");
        println!("cargo:rustc-link-lib=static=mkl_scalapack_ilp64");
        println!("cargo:rustc-link-lib=static=mkl_intel_ilp64");
        println!("cargo:rustc-link-lib=static=mkl_blacs_intelmpi_ilp64");
        println!("cargo:rustc-link-lib=static=mkl_tbb_thread");

        println!("cargo:rustc-link-lib=tbb");
        println!("cargo:rustc-link-lib=OpenCL");
        println!("cargo:rustc-link-lib=svml");
        println!("cargo:rustc-link-lib=imf");
        println!("cargo:rustc-link-lib=intlc");
        println!("cargo:rustc-link-lib=ur_loader");
        println!("cargo:rustc-link-lib=m");
        println!("cargo-rustc-link-lib=dl");
        println!("cargo:rustc-link-lib=sycl");
        println!("cargo:rustc-link-lib=dnnl");

        if target.contains("msvc") {
            config.generator("Ninja");
            config.define("CMAKE_C_COMPILER", "cl");
            config.define("CMAKE_CXX_COMPILER", "icx");
        } else {
            config.define("CMAKE_C_COMPILER", "icx");
            config.define("CMAKE_CXX_COMPILER", "icpx");
        }
        config.define("SD_SYCL", "ON");
    } else {
        println!("cargo:warning=feature `sycl` enabled but oneAPI/SYCL toolchain not detected — skipping SYCL backend.");
    }

    // Build stable-diffusion
    let destination = config.build();

    add_link_search_path(&out.join("lib")).unwrap();
    add_link_search_path(&out.join("build")).unwrap();
    add_link_search_path(&out).unwrap();

    println!("cargo:rustc-link-search=native={}", destination.display());
    println!("cargo:rustc-link-lib=static=stable-diffusion");
    // When system-ggml is on, ggml comes from blazen-ggml-sys's emit —
    // skip re-emitting our own ggml link lines to avoid duplicate
    // symbols / link-order weirdness.
    #[cfg(not(feature = "system-ggml"))]
    {
        println!("cargo:rustc-link-lib=static=ggml-base");
        println!("cargo:rustc-link-lib=static=ggml-cpu");
    }

    if target.contains("apple") {
        println!("cargo:rustc-link-lib=framework=Accelerate");
    }

    // Only emit the backend-specific ggml link line when the backend was
    // actually enabled above (toolchain present). When a GPU feature is on
    // but its toolchain was missing, the SD_* cmake flag was skipped and
    // cmake never built the matching `ggml-<backend>` static lib, so linking
    // against it would fail — guard with the same `*_present()` check.
    #[cfg(all(feature = "cuda", not(feature = "system-ggml")))]
    if cuda_present() {
        println!("cargo:rustc-link-lib=static=ggml-cuda");
    }

    #[cfg(all(feature = "hipblas", not(feature = "system-ggml")))]
    if rocm_present() {
        println!("cargo:rustc-link-lib=static=ggml-hip");
    }

    #[cfg(all(feature = "metal", not(feature = "system-ggml")))]
    println!("cargo:rustc-link-lib=static=ggml-metal");

    #[cfg(all(feature = "vulkan", not(feature = "system-ggml")))]
    if vulkan_present() {
        println!("cargo:rustc-link-lib=static=ggml-vulkan");
    }

    #[cfg(all(feature = "sycl", not(feature = "system-ggml")))]
    if sycl_present() {
        println!("cargo:rustc-link-lib=static=ggml-sycl");
    }
}

fn add_link_search_path(dir: &Path) -> std::io::Result<()> {
    if dir.is_dir() {
        println!("cargo:rustc-link-search={}", dir.display());
        for entry in read_dir(dir)? {
            add_link_search_path(&entry?.path())?;
        }
    }
    Ok(())
}

// From https://github.com/alexcrichton/cc-rs/blob/fba7feded71ee4f63cfe885673ead6d7b4f2f454/src/lib.rs#L2462
fn get_cpp_link_stdlib(target: &str) -> Option<&'static str> {
    if target.contains("msvc") {
        None
    } else if target.contains("apple") || target.contains("freebsd") || target.contains("openbsd") {
        Some("c++")
    } else if target.contains("android") {
        Some("c++_shared")
    } else {
        Some("stdc++")
    }
}
