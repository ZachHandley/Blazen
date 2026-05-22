//! Multi-GPU / multi-node distributed training via gRPC ring-AllReduce.
//!
//! Today `blazen-train::Trainer` runs on a single device. This module
//! adds an [`AllReduceTrainer`] wrapper that lets N workers form a
//! logical ring: every worker computes local gradients, then participates
//! in a ring-AllReduce that sums the per-parameter gradient tensors
//! across the ring before the optimizer step.
//!
//! ## Algorithm
//!
//! Classic Baidu / NCCL ring-AllReduce:
//!
//! - N workers indexed `0..N`. Each holds an identical-shape gradient
//!   tensor `G` divided into `N` equal chunks.
//! - **Reduce-scatter** (N-1 steps): at step `s`, worker `i` sends chunk
//!   `(i - s) mod N` to worker `(i + 1) mod N` and receives chunk
//!   `(i - s - 1) mod N` from worker `(i - 1) mod N`, accumulating the
//!   received chunk into its local copy. After `N-1` steps, worker `i`
//!   holds the fully reduced chunk `(i + 1) mod N`.
//! - **All-gather** (N-1 steps): each worker forwards its reduced chunk
//!   around the ring. After `N-1` steps, every worker has the full sum.
//!
//! Total bandwidth per worker is `2 * (N-1) / N * |G|`, independent of
//! `N` for large `N` — the asymptotic-optimal bandwidth bound for
//! gradient averaging.
//!
//! ## Transport
//!
//! [`transport::RingTransport`] abstracts the inter-worker bytes pipe.
//! Two concrete impls ship in phase 1:
//!
//! - [`transport::InMemoryRingTransport`] — `tokio::sync::mpsc` channels
//!   between in-process workers. Used by every unit test in this module.
//! - [`transport::GrpcRingTransport`] — tonic clients to next / prev
//!   workers using the `blazen.allreduce.v1` proto service defined in
//!   `proto/blazen_allreduce.proto`.
//!
//! ## Integration
//!
//! [`trainer::AllReduceTrainer`] wraps a [`crate::Trainer`]. Its `step`
//! method delegates the forward / backward to the inner trainer, then
//! takes the resulting `GradStore`, AllReduces each trainable param's
//! gradient (averaging by `world_size`), and reinserts the averaged
//! gradients before the optimizer step. Phase 2 extends this to the
//! `QloraTrainer`, `GrpoTrainer`, and `PpoTrainer` wrappers, and exposes
//! the wrapper through the Python / Node / WASM bindings.

pub mod all_reduce_grpo;
pub mod all_reduce_ppo;
pub mod allreduce;
pub mod ring;
pub mod trainer;
pub mod transport;

/// Generated tonic/prost types for `blazen.allreduce.v1`.
#[cfg(feature = "distributed")]
#[allow(clippy::all, clippy::pedantic)]
pub mod pb {
    tonic::include_proto!("blazen.allreduce.v1");
}

pub use all_reduce_grpo::AllReduceGrpoTrainer;
pub use all_reduce_ppo::AllReducePpoTrainer;
pub use allreduce::ring_all_reduce;
pub use ring::{RingConfig, RingTopology, parse_peer_list};
pub use trainer::AllReduceTrainer;
pub use transport::{InMemoryRingTransport, RingTransport};

#[cfg(feature = "distributed")]
pub use transport::grpc::{GrpcRingTransport, spawn_allreduce_server};
