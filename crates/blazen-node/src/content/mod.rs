//! Node bindings for the Blazen content subsystem.
//!
//! Mirrors [`blazen_llm::content`]. See module docs there for the design.
//!
//! Exposed surface:
//! - [`JsContentKind`] — string-enum mirror of `ContentKind`.
//! - [`JsContentHandle`] — plain-object mirror of `ContentHandle`.
//! - [`JsContentStore`] — class wrapping `Arc<dyn ContentStore>` with static
//!   factories for every built-in implementation.
//! - Tool-input schema builders ([`tool_input::image_input`] et al.) as
//!   module-level napi functions.

pub mod handle;
pub mod kind;
pub mod store;
pub mod tool_input;

pub use handle::JsContentHandle;
pub use kind::JsContentKind;
pub use store::{JsContentMetadata, JsContentStore, PutOptions};
