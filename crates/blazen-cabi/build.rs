// Build script for blazen-cabi: invokes cbindgen on every cargo build so the
// committed C header at `bindings/ruby/ext/blazen/blazen.h` stays in sync
// with the Rust extern "C" surface. The audit-bindings-drift CI job will
// fail if the regenerated header differs from what's committed.

use std::env;
use std::path::PathBuf;

fn main() {
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR"));
    let config_path = manifest_dir.join("cbindgen.toml");

    // Header lives under bindings/ruby/ext/blazen/ (committed, drift-audited).
    // From crates/blazen-cabi -> ../../bindings/ruby/ext/blazen/blazen.h
    let workspace_root = manifest_dir
        .parent()
        .and_then(|p| p.parent())
        .expect("crate must live two dirs deep from workspace root");
    let header_path = workspace_root.join("bindings/ruby/ext/blazen/blazen.h");

    if let Some(parent) = header_path.parent() {
        std::fs::create_dir_all(parent).expect("create header parent dir");
    }

    let config = cbindgen::Config::from_file(&config_path).expect("read cbindgen.toml");

    let bindings = cbindgen::Builder::new()
        .with_crate(&manifest_dir)
        .with_config(config)
        .generate();

    match bindings {
        Ok(b) => {
            b.write_to_file(&header_path);
        }
        Err(e) => {
            // Don't fail the build on header-generation errors during early
            // bootstrapping — print a warning so the developer sees it.
            // Phase-9 drift audit will still catch missing symbols.
            println!(
                "cargo:warning=cbindgen failed to generate {}: {e}",
                header_path.display()
            );
        }
    }

    // The fastembed embedding backend statically links ONNX Runtime, which
    // calls OpenMP runtime symbols. The final cdylib must carry a load-time
    // dep on the matching OpenMP runtime so those symbols resolve at dlopen
    // time — without this the Ruby FFI load fails with e.g.
    // `undefined symbol: GOMP_barrier` on linux or `___kmpc_fork_call` on
    // macOS.
    //
    // The runtime name differs by platform: glibc/linux uses GNU OpenMP
    // (`libgomp`), macOS uses LLVM OpenMP (`libomp` from homebrew). musl and
    // windows targets bundle their OpenMP statically inside the ort prebuilt
    // tarball, so no extra link line is needed there.
    if env::var_os("CARGO_FEATURE_FASTEMBED").is_some() {
        let target = env::var("TARGET").unwrap_or_default();
        if target.contains("linux") && !target.contains("musl") {
            println!("cargo:rustc-link-lib=dylib=gomp");
        } else if target.contains("apple") {
            // homebrew on aarch64 mac vs Intel mac
            println!("cargo:rustc-link-search=/opt/homebrew/opt/libomp/lib");
            println!("cargo:rustc-link-search=/usr/local/opt/libomp/lib");
            println!("cargo:rustc-link-lib=dylib=omp");
        }
    }

    println!("cargo:rerun-if-changed=src");
    println!("cargo:rerun-if-changed=cbindgen.toml");
    println!("cargo:rerun-if-changed=build.rs");
}
