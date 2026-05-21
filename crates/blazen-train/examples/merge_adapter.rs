//! Single-adapter merge CLI smoke example.
//!
//! Folds a PEFT LoRA adapter into a base safetensors file and writes a
//! plain (non-LoRA) merged shard.
//!
//! Usage:
//!
//! ```text
//! cargo run -p blazen-train --example merge_adapter -- \
//!     <base.safetensors> <adapter_model.safetensors> <output.safetensors> [scale]
//! ```
//!
//! `scale` defaults to `1.0` (PEFT-canonical "full strength"). The
//! adapter's intrinsic `alpha / r` is read from the sibling
//! `adapter_config.json` and folded in regardless.
//!
//! Deliberately no `clap` dependency — this is a smoke example, not a
//! shipped CLI.

#[cfg(feature = "engine")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use std::path::PathBuf;

    let args: Vec<String> = std::env::args().collect();
    if args.len() < 4 || args.len() > 5 {
        eprintln!(
            "usage: {} <base.safetensors> <adapter_model.safetensors> <output.safetensors> [scale]",
            args.first().map_or("merge_adapter", String::as_str)
        );
        std::process::exit(2);
    }

    let base = PathBuf::from(&args[1]);
    let adapter = PathBuf::from(&args[2]);
    let output = PathBuf::from(&args[3]);
    let scale: f32 = if let Some(s) = args.get(4) {
        s.parse().map_err(|e| format!("invalid scale '{s}': {e}"))?
    } else {
        1.0
    };

    println!(
        "merging adapter\n  base    : {}\n  adapter : {}\n  output  : {}\n  scale   : {scale}",
        base.display(),
        adapter.display(),
        output.display()
    );

    blazen_train::merge_lora_into_base(&base, &adapter, &output, scale)?;

    println!("ok — wrote merged safetensors to {}", output.display());
    Ok(())
}

#[cfg(not(feature = "engine"))]
fn main() {
    eprintln!(
        "merge_adapter example requires the 'engine' feature: \
         cargo run -p blazen-train --example merge_adapter --features engine -- ..."
    );
    std::process::exit(2);
}
