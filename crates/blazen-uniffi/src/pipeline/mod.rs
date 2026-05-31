//! Pipeline surface for the UniFFI bindings.
//!
//! A pipeline composes one or more [`Workflow`](crate::workflow::Workflow)s
//! into an ordered run. The first stage's input is the JSON payload passed to
//! [`Pipeline::run`](pipeline::Pipeline::run); each subsequent stage receives
//! the previous stage's output as its `StartEvent` payload.
//!
//! ## Real engine, full parity
//!
//! Unlike the earlier thin re-implementation, this module is a thin FFI
//! wrapper over the real [`blazen_pipeline`] orchestrator. Each uniffi
//! [`Workflow`](crate::workflow::Workflow) holds an `Arc<blazen_core::Workflow>`
//! inner; since `blazen_core::Workflow` is `Clone` (Arc-backed step registry),
//! we obtain an owned core workflow per stage with a cheap clone — there is no
//! borrow obstacle to feeding it into a `blazen_pipeline::Stage`.
//!
//! This gives the uniffi [`PipelineHandler`](handler::PipelineHandler) the same
//! ergonomic control surface as the workflow handler:
//! `result`/`pause`/`resume_in_place`/`snapshot`/`abort`/`progress`/
//! `respond_to_input`/`usage_total`/`cost_total_usd`/`stream_events`, plus the
//! [`run`](pipeline::Pipeline::run) result shorthand and
//! [`start`](pipeline::Pipeline::start) handle path.
//!
//! ## Example (Go)
//!
//! ```go,ignore
//! ingest, _ := blazen.NewWorkflowBuilder("ingest").Step(...).Build()
//! enrich, _ := blazen.NewWorkflowBuilder("enrich").Step(...).Build()
//!
//! pipe, _ := blazen.NewPipelineBuilder("etl").
//!     AddWorkflow(ingest).
//!     AddWorkflow(enrich).
//!     TotalTimeoutMs(60_000).
//!     Build()
//!
//! result, err := pipe.Run(`{"source":"s3://..."}`)
//! ```

pub mod builder;
pub mod handler;
#[allow(clippy::module_inception)]
pub mod pipeline;

pub use builder::PipelineBuilder;
pub use handler::{PipelineEvent, PipelineEventSink, PipelineHandler};
pub use pipeline::Pipeline;
