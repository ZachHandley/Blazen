//! WASM bindings for the [`blazen_pipeline`] crate.
//!
//! Mirrors the layout of the Python (`crates/blazen-py/src/pipeline`) and
//! Node (`crates/blazen-node/src/pipeline`) bindings: a typed JS-facing
//! [`Pipeline`](pipeline::WasmPipeline) class with a fluent
//! [`PipelineBuilder`](builder::WasmPipelineBuilder), plus
//! [`Stage`](stage::WasmStage) / [`ParallelStage`](stage::WasmParallelStage)
//! wrappers and a [`PipelineHandler`](handler::WasmPipelineHandler) for
//! awaiting results, streaming events, pausing, and aborting.
//!
//! # v1 limitations
//!
//! - `Stage::input_mapper` and `Stage::condition` are not exposed. The
//!   underlying [`InputMapperFn`](blazen_pipeline::InputMapperFn) and
//!   [`ConditionFn`](blazen_pipeline::ConditionFn) are synchronous Rust
//!   closures (`Fn(&PipelineState) -> Value` / `... -> bool`), and the
//!   pipeline execution loop calls them from within a Tokio task on a
//!   non-`Send` `wasm_bindgen_futures::JsFuture`-incompatible code path.
//!   Wiring a JS callback into a sync closure on wasm32 would require
//!   either blocking on a JS future inside a sync context (impossible on
//!   wasm32, single-threaded) or extending `blazen_pipeline` with async-fn
//!   support upstream.
//! - `PipelineHandler::snapshot()` mirrors the upstream stub and currently
//!   returns a `ChannelClosed` error; this is preserved here so JS callers
//!   see the same shape as native Rust callers.

pub mod builder;
pub mod error;
pub mod event;
pub mod handler;
#[allow(clippy::module_inception)]
pub mod pipeline;
pub mod snapshot;
pub mod stage;
pub mod state;

pub use builder::WasmPipelineBuilder;
pub use error::pipeline_err;
pub use event::WasmPipelineEvent;
pub use handler::WasmPipelineHandler;
pub use pipeline::WasmPipeline;
pub use snapshot::{
    WasmActiveWorkflowSnapshot, WasmPipelineResult, WasmPipelineSnapshot, WasmStageResult,
};
pub use stage::{WasmJoinStrategy, WasmParallelStage, WasmStage, WasmStageKind};
pub use state::WasmPipelineState;
