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

    // Model server proto (PR5) — same envelope shape (PostcardRequest /
    // PostcardResponse) but lives in its own `blazen.modelserver.v1`
    // proto package so the generated tonic service trait doesn't collide
    // with `BlazenControlPlane`. Always emit both server and client code
    // when the matching feature is on so the host binary (which may flip
    // either flag on) gets the symbols it needs.
    if cfg!(any(feature = "model-server", feature = "model-client")) {
        tonic_prost_build::configure()
            .build_server(cfg!(feature = "model-server"))
            .build_client(cfg!(feature = "model-client"))
            .compile_protos(&["proto/blazen_modelserver.proto"], &["proto/"])?;
    }
    Ok(())
}
