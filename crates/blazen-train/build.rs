fn main() -> Result<(), Box<dyn std::error::Error>> {
    // The proto-generated `allreduce_pb` module and the gRPC server / client
    // that consume it are gated on the `distributed` feature in `lib.rs`
    // and only built on native targets — tonic itself does not compile for
    // wasm32-wasi*. Skip codegen on wasm so we don't pull `protoc` onto a
    // host that can't link the result anyway. Mirrors the pattern in
    // `crates/blazen-controlplane/build.rs`.
    let target = std::env::var("TARGET").unwrap_or_default();
    if target.starts_with("wasm32-") {
        return Ok(());
    }

    // Only compile the proto when the `distributed` feature is active —
    // the cfg-gate in `lib.rs` is what consumers see, and `tonic-prost-build`
    // is an optional build-dep so its absence here would be a hard error
    // otherwise. Detect via the cargo-emitted feature env var.
    #[cfg(feature = "distributed")]
    {
        tonic_prost_build::configure()
            .build_server(true)
            .build_client(true)
            .compile_protos(&["proto/blazen_allreduce.proto"], &["proto/"])?;
    }
    Ok(())
}
