//! Content store and content-kind subsystem.
//!
//! This module is the home of the "what is this thing and where do I get it"
//! plumbing for multimodal content that flows through Blazen conversations.
//! It exposes:
//!
//! - [`ContentKind`] — the taxonomy used by tool-input declarations and the
//!   content store to classify any blob of bytes / external reference.
//! - [`detect`] / [`detect_from_bytes`] / [`detect_from_path`] — magic-
//!   number-based file type detection, with extension and MIME fallbacks.
//!
//! Subsequent waves add the [`ContentStore`] trait, [`ContentHandle`], and
//! built-in store implementations (in-memory, local-file, provider-file).
//!
//! [`ContentStore`]: store::ContentStore
//! [`ContentHandle`]: handle::ContentHandle

pub mod detect;
pub mod handle;
pub mod kind;
pub mod render;
pub mod store;
pub mod stores;
pub mod tool_input;
pub mod visibility;

#[cfg(not(target_arch = "wasm32"))]
pub use detect::detect_from_path;
pub use detect::{detect, detect_from_bytes};
pub use handle::ContentHandle;
pub use kind::ContentKind;
pub use store::{ContentBody, ContentHint, ContentMetadata, ContentStore, DynContentStore};
#[cfg(not(target_arch = "wasm32"))]
pub use stores::LocalFileContentStore;
pub use stores::{
    AnthropicFilesStore, CustomContentStore, CustomContentStoreBuilder, FalStorageStore,
    GeminiFilesStore, InMemoryContentStore, OpenAiFilesStore,
};
