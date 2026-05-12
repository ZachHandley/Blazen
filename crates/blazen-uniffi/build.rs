fn main() {
    uniffi_build::generate_scaffolding("src/blazen.udl")
        .expect("failed to generate UniFFI scaffolding from src/blazen.udl");
}
