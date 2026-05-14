//! Tonic service trait implementation for `BlazenControlPlane`.
//!
//! This file wires the proto-generated service trait to handler
//! functions in [`super::session`], [`super::rpc`], and
//! [`super::subscribe`]. Each handler decodes the postcard payload from
//! the incoming `PostcardRequest`, calls into the shared state, and
//! encodes the response.

use std::pin::Pin;
use std::sync::Arc;

use tokio_stream::Stream;
use tonic::{Request, Response, Status, Streaming};

use crate::pb::{
    PostcardRequest, PostcardResponse, blazen_control_plane_server::BlazenControlPlane,
};

use super::SharedState;

/// Concrete implementation of the `BlazenControlPlane` gRPC service.
///
/// Built from an [`Arc<SharedState>`] so the same registry / queue /
/// admission instance is shared across every concurrent RPC. Cloning
/// the service is cheap — it only bumps the `Arc`.
#[derive(Clone)]
pub struct ControlPlaneService {
    pub(crate) shared: Arc<SharedState>,
}

impl ControlPlaneService {
    /// Build a fresh service from the given shared state.
    #[must_use]
    pub fn new(shared: Arc<SharedState>) -> Self {
        Self { shared }
    }
}

/// Stream type alias matching the shape `tonic` expects from every
/// server-streaming RPC on this service (boxed `dyn Stream + Send`).
pub type PostcardResponseStream =
    Pin<Box<dyn Stream<Item = Result<PostcardResponse, Status>> + Send>>;

#[tonic::async_trait]
impl BlazenControlPlane for ControlPlaneService {
    type WorkerSessionStream = PostcardResponseStream;
    type SubscribeRunEventsStream = PostcardResponseStream;
    type SubscribeAllStream = PostcardResponseStream;

    async fn worker_session(
        &self,
        request: Request<Streaming<PostcardRequest>>,
    ) -> Result<Response<Self::WorkerSessionStream>, Status> {
        super::session::handle_worker_session(self.shared.clone(), request.into_inner()).await
    }

    async fn submit_workflow(
        &self,
        request: Request<PostcardRequest>,
    ) -> Result<Response<PostcardResponse>, Status> {
        super::rpc::handle_submit_workflow(&self.shared, request.into_inner()).await
    }

    async fn cancel_workflow(
        &self,
        request: Request<PostcardRequest>,
    ) -> Result<Response<PostcardResponse>, Status> {
        super::rpc::handle_cancel_workflow(&self.shared, request.into_inner()).await
    }

    async fn describe_workflow(
        &self,
        request: Request<PostcardRequest>,
    ) -> Result<Response<PostcardResponse>, Status> {
        super::rpc::handle_describe_workflow(&self.shared, request.into_inner()).await
    }

    async fn list_workers(
        &self,
        request: Request<PostcardRequest>,
    ) -> Result<Response<PostcardResponse>, Status> {
        super::rpc::handle_list_workers(&self.shared, request.into_inner()).await
    }

    async fn drain_worker(
        &self,
        request: Request<PostcardRequest>,
    ) -> Result<Response<PostcardResponse>, Status> {
        super::rpc::handle_drain_worker(&self.shared, request.into_inner()).await
    }

    async fn subscribe_run_events(
        &self,
        request: Request<PostcardRequest>,
    ) -> Result<Response<Self::SubscribeRunEventsStream>, Status> {
        super::subscribe::handle_subscribe_run_events(self.shared.clone(), request.into_inner())
            .await
    }

    async fn subscribe_all(
        &self,
        request: Request<PostcardRequest>,
    ) -> Result<Response<Self::SubscribeAllStream>, Status> {
        super::subscribe::handle_subscribe_all(self.shared.clone(), request.into_inner()).await
    }
}
