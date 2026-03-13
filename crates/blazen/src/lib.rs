//! # `Blazen`
//!
//! A high-performance, event-driven workflow framework for Rust.
//!
//! `Blazen` models computation as a directed graph of *steps* connected by
//! typed *events*. The runtime maintains an internal event queue, routing
//! events to matching step handlers and spawning them concurrently until a
//! [`StopEvent`] terminates the workflow.
//!
//! ## Quick start
//!
//! ```rust,ignore
//! use blazen::prelude::*;
//!
//! #[derive(Debug, Clone, Serialize, Deserialize, Event)]
//! struct GreetEvent { name: String }
//!
//! #[step]
//! async fn greet(event: StartEvent, _ctx: Context) -> Result<StopEvent, WorkflowError> {
//!     let name = event.data["name"].as_str().unwrap_or("World");
//!     Ok(StopEvent { result: serde_json::json!({ "greeting": format!("Hello, {}!", name) }) })
//! }
//!
//! #[tokio::main]
//! async fn main() -> anyhow::Result<()> {
//!     let workflow = WorkflowBuilder::new("greeter")
//!         .step(greet_registration())
//!         .build()?;
//!     let result = workflow.run(serde_json::json!({ "name": "Zach" })).await?.result().await?;
//!     println!("{}", result.to_json());
//!     Ok(())
//! }
//! ```
//!
//! ## Feature flags
//!
//! | Feature | Description |
//! |---------|-------------|
//! | `llm` (default) | LLM provider integrations |
//! | `openai` | OpenAI + OpenAI-compatible providers |
//! | `anthropic` | Anthropic Claude provider |
//! | `persist` | Checkpoint storage via `redb` |
//! | `all` | Everything |

// ---------------------------------------------------------------------------
// Core re-exports
// ---------------------------------------------------------------------------

// Events: traits and built-in types.
pub use blazen_events::{
    AnyEvent, DynamicEvent, Event, EventEnvelope, StartEvent, StopEvent, intern_event_type,
};

// Macros: derive and attribute macros.
pub use blazen_macros::{Event, step};

// Core: workflow engine types.
pub use blazen_core::{
    Context, Result, StepFn, StepOutput, StepRegistration, Workflow, WorkflowBuilder,
    WorkflowError, WorkflowHandler,
};

// Serde: needed by derive(Event) expansion and examples.
pub use serde;
pub use serde_json;

// ---------------------------------------------------------------------------
// Optional modules
// ---------------------------------------------------------------------------

/// LLM provider integrations (requires `llm` feature).
#[cfg(feature = "llm")]
pub mod llm {
    pub use blazen_llm::*;
}

/// Checkpoint persistence (requires `persist` feature).
#[cfg(feature = "persist")]
pub mod persist {
    pub use blazen_persist::*;
}

// Top-level re-exports of key persist types for convenience.
#[cfg(feature = "persist")]
pub use blazen_persist::{CheckpointStore, WorkflowCheckpoint};

// ---------------------------------------------------------------------------
// Prelude
// ---------------------------------------------------------------------------

/// Convenient wildcard import for common types and traits.
///
/// ```rust,ignore
/// use blazen::prelude::*;
/// ```
pub mod prelude {
    // Event traits and built-in events.
    pub use blazen_events::{AnyEvent, Event, StartEvent, StopEvent};

    // Derive and attribute macros.
    pub use blazen_macros::{Event, step};

    // Core workflow types.
    pub use blazen_core::{
        Context, Result, StepOutput, StepRegistration, Workflow, WorkflowBuilder, WorkflowError,
        WorkflowHandler,
    };

    // Serde derives (needed for #[derive(Event)] to work).
    pub use serde::{Deserialize, Serialize};

    // JSON value construction.
    pub use serde_json;
}
