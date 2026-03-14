//! # `Blazen` Prompts
//!
//! Provides reusable prompt template management with `{{variable}}`
//! interpolation, versioning, and file I/O for the `Blazen` AI workflow
//! engine.
//!
//! Templates are organised in a [`PromptRegistry`] that supports named
//! lookups, version pinning, and batch rendering into
//! [`CompletionRequest`](blazen_llm::CompletionRequest) messages.
//!
//! ## Quick start
//!
//! ```
//! use std::collections::HashMap;
//! use blazen_prompts::{PromptTemplate, PromptRegistry};
//! use blazen_prompts::template::TemplateRole;
//!
//! // Create a template
//! let template = PromptTemplate::new(
//!     "summarise",
//!     TemplateRole::System,
//!     "Summarise the following {{doc_type}} in {{style}} style.",
//! );
//!
//! // Register and render
//! let mut registry = PromptRegistry::new();
//! registry.register(template);
//!
//! let mut vars = HashMap::new();
//! vars.insert("doc_type".to_owned(), "article".to_owned());
//! vars.insert("style".to_owned(), "concise".to_owned());
//!
//! let message = registry.render("summarise", &vars).unwrap();
//! ```
//!
//! ## File I/O
//!
//! Prompt templates can be loaded from and saved to YAML or JSON files:
//!
//! ```no_run
//! use blazen_prompts::PromptRegistry;
//!
//! // Load all prompt files from a directory
//! let registry = PromptRegistry::from_dir("./prompts").unwrap();
//!
//! // Save to a single file
//! registry.to_file("./prompts/all.yaml").unwrap();
//! ```

pub mod error;
pub mod format;
pub mod registry;
pub mod template;

pub use error::PromptError;
pub use format::PromptFile;
pub use registry::PromptRegistry;
pub use template::{PromptTemplate, TemplateRole};
