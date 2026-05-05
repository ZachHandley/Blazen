//! Python bindings for the `blazen_llm::content` subsystem.
//!
//! Exposes [`PyContentKind`] / [`PyContentHandle`] / [`PyContentStore`]
//! plus the tool-input schema builders. Wired into the `blazen` Python
//! module from [`crate::lib`].

pub mod handle;
pub mod kind;
pub mod store;
pub mod tool_input;

pub use handle::PyContentHandle;
pub use kind::PyContentKind;
pub use store::PyContentStore;
