fn main() -> Result<(), Box<dyn std::error::Error>> {
    tonic_prost_build::configure()
        .build_server(cfg!(feature = "server"))
        .build_client(cfg!(feature = "client"))
        .compile_protos(&["proto/blazen_peer.proto"], &["proto/"])?;
    Ok(())
}
