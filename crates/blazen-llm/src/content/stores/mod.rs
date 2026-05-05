//! Built-in [`ContentStore`] implementations.
//!
//! - [`InMemoryContentStore`] — default; ephemeral, in-process.
//! - [`LocalFileContentStore`] — persists to a configured directory.
//!   Native targets only.
//! - [`CustomContentStore`] — wraps user-supplied async callbacks for
//!   plugging into S3 / GCS / your own backend.
//!
//! Provider-file stores (`OpenAI` Files / Anthropic Files / Gemini Files /
//! fal.ai storage) ship in a follow-up wave.
//!
//! [`ContentStore`]: super::store::ContentStore

pub mod anthropic_files;
pub mod custom;
pub mod fal_storage;
pub mod gemini_files;
pub mod in_memory;
#[cfg(not(target_arch = "wasm32"))]
pub mod local_file;
pub mod openai_files;

pub use anthropic_files::AnthropicFilesStore;
pub use custom::{CustomContentStore, CustomContentStoreBuilder};
pub use fal_storage::FalStorageStore;
pub use gemini_files::GeminiFilesStore;
pub use in_memory::InMemoryContentStore;
#[cfg(not(target_arch = "wasm32"))]
pub use local_file::LocalFileContentStore;
pub use openai_files::OpenAiFilesStore;
