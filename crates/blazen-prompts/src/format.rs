//! File format types for serializing and deserializing prompt collections.
//!
//! Prompt files use a simple wrapper struct that holds a list of templates:
//!
//! ```yaml
//! prompts:
//!   - name: analyze_image
//!     version: "1.0"
//!     role: system
//!     description: "System prompt for image analysis"
//!     template: |
//!       You are an expert image analyst. Analyze the {{image_type}} image
//!       focusing on {{focus_areas}}.
//!     metadata:
//!       author: zach
//! ```

use serde::{Deserialize, Serialize};

use crate::template::PromptTemplate;

/// A serializable collection of prompt templates.
///
/// This is the top-level structure used when reading/writing prompt template
/// files in YAML or JSON format.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PromptFile {
    /// The prompt templates in this file.
    pub prompts: Vec<PromptTemplate>,
}

impl PromptFile {
    /// Initialize all templates after deserialization.
    ///
    /// This populates the cached `variables` field on each template since it
    /// is `#[serde(skip)]` and not included in serialized output.
    pub fn init(&mut self) {
        for template in &mut self.prompts {
            template.init();
        }
    }
}
