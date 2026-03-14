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
    /// Scaffold Blazen project files in the current directory
    Init {
        #[command(subcommand)]
        target: InitTarget,
    },
}

#[derive(Subcommand)]
enum InitTarget {
    /// Generate a BLAZEN.md workflow usage guide
    Workflow {
        /// Target language for the guide
        #[arg(short, long, default_value = "rust")]
        lang: Lang,

        /// Overwrite existing BLAZEN.md
        #[arg(short, long)]
        force: bool,
    },

    /// Scaffold a multi-workflow pipeline project
    Pipeline {
        /// Target language for the guide
        #[arg(short, long, default_value = "rust")]
        lang: Lang,

        /// Overwrite existing BLAZEN.md
        #[arg(short, long)]
        force: bool,
    },

    /// Scaffold a prompt registry with example YAML templates
    Prompts {
        /// Overwrite existing files
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

fn write_file(path: &Path, content: &str, force: bool) -> bool {
    if path.exists() && !force {
        eprintln!(
            "{} already exists. Use --force to overwrite.",
            path.display()
        );
        return false;
    }
    fs::write(path, content).unwrap_or_else(|e| {
        eprintln!("Failed to write {}: {e}", path.display());
        std::process::exit(1);
    });
    println!("Created {}", path.display());
    true
}

fn main() {
    let cli = Cli::parse();

    match cli.command {
        Commands::Init { target } => match target {
            InitTarget::Workflow { lang, force } => {
                let content = match lang {
                    Lang::Rust => templates::rust::TEMPLATE,
                    Lang::Python => templates::python::TEMPLATE,
                    Lang::Typescript => templates::typescript::TEMPLATE,
                };

                write_file(Path::new("BLAZEN.md"), content, force);
            }

            InitTarget::Pipeline { lang, force } => {
                let content = match lang {
                    Lang::Rust => templates::pipeline_rust::TEMPLATE,
                    Lang::Python => templates::pipeline_python::TEMPLATE,
                    Lang::Typescript => templates::pipeline_typescript::TEMPLATE,
                };

                write_file(Path::new("BLAZEN.md"), content, force);
            }

            InitTarget::Prompts { force } => {
                let blazen_md = Path::new("BLAZEN.md");
                let prompts_dir = Path::new("prompts");
                let example_yaml = prompts_dir.join("example.yaml");

                write_file(blazen_md, templates::prompts::TEMPLATE, force);

                if !prompts_dir.exists() {
                    fs::create_dir_all(prompts_dir).unwrap_or_else(|e| {
                        eprintln!("Failed to create prompts/ directory: {e}");
                        std::process::exit(1);
                    });
                    println!("Created prompts/");
                }

                write_file(&example_yaml, templates::prompts::EXAMPLE_YAML, force);
            }
        },
    }
}
