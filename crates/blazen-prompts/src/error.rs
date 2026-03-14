//! Error types for the prompt template system.

/// Errors that can occur when working with prompt templates and registries.
#[derive(Debug, thiserror::Error)]
pub enum PromptError {
    /// A required template variable was not provided during rendering.
    #[error("missing variable '{name}' in template '{template}'")]
    MissingVariable {
        /// The name of the template that references the variable.
        template: String,
        /// The name of the missing variable.
        name: String,
    },

    /// The requested prompt was not found in the registry.
    #[error("prompt '{name}' not found")]
    NotFound {
        /// The name of the prompt that was not found.
        name: String,
    },

    /// The requested version of a prompt was not found in the registry.
    #[error("prompt '{name}' version '{version}' not found")]
    VersionNotFound {
        /// The name of the prompt.
        name: String,
        /// The version that was not found.
        version: String,
    },

    /// An I/O error occurred while reading or writing prompt files.
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    /// A YAML parsing or serialization error occurred.
    #[error("YAML parse error: {0}")]
    Yaml(#[from] serde_yaml::Error),

    /// A JSON parsing or serialization error occurred.
    #[error("JSON parse error: {0}")]
    Json(#[from] serde_json::Error),

    /// A validation error with a descriptive message.
    #[error("validation error: {0}")]
    Validation(String),
}
