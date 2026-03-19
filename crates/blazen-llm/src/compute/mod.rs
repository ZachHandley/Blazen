//! Compute platform abstraction for async job-based media generation.
//!
//! This module provides a unified trait system for compute-heavy media
//! generation providers (fal.ai, Replicate, `RunPod`, etc.) that expose a
//! submit-poll-retrieve workflow for GPU workloads including image generation,
//! video synthesis, audio production, transcription, and 3D model creation.
//!
//! ## Architecture
//!
//! [`ComputeProvider`] is the base trait modelling the async job lifecycle:
//! submit, poll status, retrieve results, and cancel. Media-specific traits
//! ([`ImageGeneration`], [`VideoGeneration`], [`AudioGeneration`],
//! [`Transcription`], [`ThreeDGeneration`]) extend this with typed request
//! and response types for each modality.
//!
//! ## Example
//!
//! ```rust,no_run
//! use blazen_llm::compute::{ComputeProvider, ComputeRequest, ImageGeneration, ImageRequest};
//!
//! # async fn example(provider: impl ImageGeneration) -> Result<(), blazen_llm::BlazenError> {
//! // High-level: generate an image with the typed API.
//! let request = ImageRequest::new("a cat in space")
//!     .with_size(1024, 1024)
//!     .with_count(2);
//! let result = provider.generate_image(request).await?;
//! for image in &result.images {
//!     println!("url: {:?}, size: {}x{}", image.media.url, image.width.unwrap_or(0), image.height.unwrap_or(0));
//! }
//!
//! // Low-level: submit arbitrary JSON and wait for the result.
//! let request = ComputeRequest {
//!     model: "fal-ai/flux/dev".into(),
//!     input: serde_json::json!({ "prompt": "a cat in space" }),
//!     webhook: None,
//! };
//! let result = provider.run(request).await?;
//! println!("output: {}", result.output);
//! # Ok(())
//! # }
//! ```

pub mod job;
pub mod requests;
pub mod results;
pub mod traits;

pub use job::*;
pub use requests::*;
pub use results::*;
pub use traits::*;
