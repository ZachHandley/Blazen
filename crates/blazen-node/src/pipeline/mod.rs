//! Node bindings for the `blazen-pipeline` crate.

pub mod builder;
pub mod event;
pub mod handler;
#[allow(clippy::module_inception)]
pub mod pipeline;
pub mod snapshot;
pub mod stage;

pub use builder::JsPipelineBuilder;
pub use event::JsPipelineEvent;
pub use handler::JsPipelineHandler;
pub use pipeline::JsPipeline;
pub use snapshot::{JsActiveWorkflowSnapshot, JsPipelineResult, JsPipelineSnapshot, JsStageResult};
pub use stage::{JsJoinStrategy, JsParallelStage, JsStage};
