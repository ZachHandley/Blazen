fn main() -> Result<(), Box<dyn std::error::Error>> {
    // The proto-generated `pb` module and the gRPC server / client that
    // consume it are gated to native targets in `lib.rs` (tonic does not
    // compile for wasm32-wasi*). Skip codegen on wasm targets so we don't
    // pull `tonic-prost-build` -> `protoc` requirements onto a host that
    // can't link the result anyway.
    let target = std::env::var("TARGET").unwrap_or_default();
    if target.starts_with("wasm32-") {
        return Ok(());
    }

    tonic_prost_build::configure()
        .build_server(cfg!(feature = "server"))
        .build_client(cfg!(feature = "client"))
        .compile_protos(&["proto/blazen_controlplane.proto"], &["proto/"])?;
    Ok(())
}
