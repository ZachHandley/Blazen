mod templates;

use clap::{Parser, Subcommand, ValueEnum};
use std::fs;
use std::path::Path;

#[derive(Parser)]
#[command(name = "blazen", about = "Blazen CLI tools")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Generate a BLAZEN.md usage guide in the current directory
    Init {
        /// Target language for the guide
        #[arg(short, long, default_value = "rust")]
        lang: Lang,

        /// Overwrite existing BLAZEN.md
        #[arg(short, long)]
        force: bool,
    },
}

#[derive(Clone, ValueEnum)]
enum Lang {
    Rust,
    Python,
    Typescript,
}

fn main() {
    let cli = Cli::parse();

    match cli.command {
        Commands::Init { lang, force } => {
            let path = Path::new("BLAZEN.md");

            if path.exists() && !force {
                eprintln!("BLAZEN.md already exists. Use --force to overwrite.");
                std::process::exit(1);
            }

            let content = match lang {
                Lang::Rust => templates::rust::TEMPLATE,
                Lang::Python => templates::python::TEMPLATE,
                Lang::Typescript => templates::typescript::TEMPLATE,
            };

            fs::write(path, content).expect("Failed to write BLAZEN.md");
            println!("Created BLAZEN.md");
        }
    }
}
