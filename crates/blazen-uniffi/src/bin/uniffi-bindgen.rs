// Local entrypoint for the uniffi-bindgen CLI. Per the upstream uniffi-rs
// pattern, the CLI is built from each project rather than installed globally —
// this guarantees the bindgen version always matches the runtime version
// embedded in the cdylib (avoiding subtle scaffolding-format drift).
//
// Run via `cargo run --bin uniffi-bindgen -p blazen-uniffi -- <args>` or via
// `scripts/regen-bindings.sh` which wraps it.

fn main() {
    uniffi::uniffi_bindgen_main();
}
