//! Bridge between [`blazen_3d::backends::triposr::TripoSrBackend`] and
//! the [`ThreeDGeneration`](crate::compute::ThreeDGeneration) trait.
//!
//! Lives here (not in `blazen-3d`) because `blazen-llm` is downstream of
//! `blazen-3d` -- putting the impl in `blazen-3d` would form a dep cycle.
//!
//! [`ComputeProvider::submit`] returns [`BlazenError::Unsupported`] --
//! 3D generation in this stack is synchronous and does not use the
//! asynchronous job-queue surface.

use std::time::Instant;

use async_trait::async_trait;
use base64::Engine as _;
use base64::engine::general_purpose::STANDARD as BASE64;
use blazen_3d::backends::triposr::TripoSrBackend;

use crate::compute::{
    ComputeProvider, ComputeRequest, ComputeResult, JobHandle, JobStatus, ThreeDGeneration,
    ThreeDRequest, ThreeDResult,
};
use crate::error::BlazenError;
use crate::media::{Generated3DModel, MediaOutput, MediaType};
use crate::types::RequestTiming;

const PROVIDER_ID: &str = "blazen-3d-triposr";

// ---------------------------------------------------------------------------
// ComputeProvider
// ---------------------------------------------------------------------------

#[async_trait]
impl ComputeProvider for TripoSrBackend {
    fn provider_id(&self) -> &str {
        PROVIDER_ID
    }

    async fn submit(&self, _request: ComputeRequest) -> Result<JobHandle, BlazenError> {
        Err(BlazenError::unsupported(
            "blazen-3d-triposr runs synchronously and does not use the \
             ComputeRequest job API; call `ThreeDGeneration::generate_3d` directly instead",
        ))
    }

    async fn status(&self, _job: &JobHandle) -> Result<JobStatus, BlazenError> {
        Err(BlazenError::unsupported(
            "blazen-3d-triposr does not expose a job queue -- inference is synchronous",
        ))
    }

    async fn result(&self, _job: JobHandle) -> Result<ComputeResult, BlazenError> {
        Err(BlazenError::unsupported(
            "blazen-3d-triposr does not expose a job queue -- inference is synchronous",
        ))
    }

    async fn cancel(&self, _job: &JobHandle) -> Result<(), BlazenError> {
        Err(BlazenError::unsupported(
            "blazen-3d-triposr inference is synchronous and cannot be cancelled",
        ))
    }
}

// ---------------------------------------------------------------------------
// ThreeDGeneration
// ---------------------------------------------------------------------------

#[async_trait]
impl ThreeDGeneration for TripoSrBackend {
    async fn generate_3d(&self, request: ThreeDRequest) -> Result<ThreeDResult, BlazenError> {
        // TripoSR is image-to-3D only. A text-only prompt cannot drive
        // it; reject early with a clear message so callers don't pay the
        // weight-load cost only to fail at inference time.
        let image_url = request.image_url.as_deref().ok_or_else(|| {
            BlazenError::validation(
                "TripoSR requires `image_url` (image-to-3D); text-only prompts are not supported",
            )
        })?;

        // mesh_resolution opt-in via the parameters object; default to
        // 256 to match the upstream TripoSR reference.
        let mesh_resolution = request
            .parameters
            .get("mesh_resolution")
            .and_then(serde_json::Value::as_u64)
            .and_then(|n| usize::try_from(n).ok())
            .unwrap_or(256);

        let started = Instant::now();

        // Fetch the image (HTTP/HTTPS URL or data: URI) and decode it
        // into raw RGB at (height, width, 3) -- exactly what
        // `TripoSrPipeline::image_to_glb` expects.
        let image_bytes = fetch_image_bytes(image_url).await?;
        let dyn_image = image::load_from_memory(&image_bytes)
            .map_err(|e| BlazenError::provider(PROVIDER_ID, format!("image decode failed: {e}")))?;
        let rgb = dyn_image.to_rgb8();
        let (width, height) = (rgb.width(), rgb.height());
        let rgb_bytes = rgb.into_raw();

        // The pipeline is synchronous candle inference (CPU- or
        // CUDA-bound); offload to the blocking pool so we don't park a
        // tokio worker through the whole forward pass.
        let backend = self.clone();
        let glb_bytes = tokio::task::spawn_blocking(move || {
            backend
                .pipeline()
                .image_to_glb(&rgb_bytes, width, height, mesh_resolution)
        })
        .await
        .map_err(|e| BlazenError::provider(PROVIDER_ID, format!("task join failed: {e}")))?
        .map_err(|e| BlazenError::provider(PROVIDER_ID, format!("inference failed: {e}")))?;

        let elapsed = started.elapsed();
        let total_ms = u64::try_from(elapsed.as_millis()).unwrap_or(u64::MAX);
        let timing = RequestTiming {
            queue_ms: None,
            execution_ms: Some(total_ms),
            total_ms: Some(total_ms),
        };

        let glb_size = u64::try_from(glb_bytes.len()).unwrap_or(u64::MAX);
        let mut media = MediaOutput::from_base64(BASE64.encode(&glb_bytes), MediaType::Glb);
        media.file_size = Some(glb_size);

        let model = Generated3DModel {
            media,
            vertex_count: None,
            face_count: None,
            has_textures: false,
            has_animations: false,
        };

        Ok(ThreeDResult {
            models: vec![model],
            timing,
            cost: None,
            usage: None,
            metadata: serde_json::json!({
                "provider": PROVIDER_ID,
                "model": "triposr",
                "image_url": image_url,
                "mesh_resolution": mesh_resolution,
            }),
        })
    }
}

// ---------------------------------------------------------------------------
// Image fetch
// ---------------------------------------------------------------------------

/// Fetch the bytes behind `image_url`. Supports `http://`, `https://`,
/// and RFC 2397 `data:` URIs (both `base64` and percent-encoded).
async fn fetch_image_bytes(image_url: &str) -> Result<Vec<u8>, BlazenError> {
    if let Some(payload) = image_url.strip_prefix("data:") {
        return decode_data_uri(payload);
    }

    if image_url.starts_with("http://") || image_url.starts_with("https://") {
        let response = reqwest::get(image_url).await.map_err(|e| {
            BlazenError::provider(PROVIDER_ID, format!("failed to fetch image_url: {e}"))
        })?;
        let status = response.status();
        if !status.is_success() {
            return Err(BlazenError::provider(
                PROVIDER_ID,
                format!("image_url fetch returned HTTP {status}"),
            ));
        }
        let bytes = response.bytes().await.map_err(|e| {
            BlazenError::provider(PROVIDER_ID, format!("failed to read image body: {e}"))
        })?;
        return Ok(bytes.to_vec());
    }

    Err(BlazenError::validation(format!(
        "unsupported image_url scheme: {image_url}; expected http://, https://, or data:"
    )))
}

/// Decode the payload of a `data:` URI (everything after the leading
/// `data:`). Handles `data:<mime>;base64,<b64>` and
/// `data:<mime>,<percent-encoded>` shapes.
fn decode_data_uri(payload: &str) -> Result<Vec<u8>, BlazenError> {
    let (header, body) = payload.split_once(',').ok_or_else(|| {
        BlazenError::validation("malformed data: URI -- missing ',' separator before payload")
    })?;
    if header
        .split(';')
        .any(|tok| tok.eq_ignore_ascii_case("base64"))
    {
        BASE64
            .decode(body)
            .map_err(|e| BlazenError::validation(format!("data: URI base64 decode failed: {e}")))
    } else {
        Ok(percent_decode(body))
    }
}

/// Minimal RFC 3986 percent-decoder for `data:` URI bodies. Invalid
/// `%XX` sequences are kept verbatim so callers see "the literal `%`
/// followed by two chars" rather than a silent corruption.
fn percent_decode(input: &str) -> Vec<u8> {
    let bytes = input.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'%' && i + 2 < bytes.len() {
            let hi = (bytes[i + 1] as char).to_digit(16);
            let lo = (bytes[i + 2] as char).to_digit(16);
            if let (Some(hi), Some(lo)) = (hi, lo) {
                #[allow(clippy::cast_possible_truncation)]
                out.push(((hi << 4) | lo) as u8);
                i += 3;
                continue;
            }
        }
        out.push(bytes[i]);
        i += 1;
    }
    out
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// `data:` URI decoder accepts plain `base64` and percent-encoded
    /// shapes and surfaces structured errors on garbage input. No
    /// pipeline / weights required.
    #[test]
    fn data_uri_base64_round_trips() {
        let payload = "image/png;base64,SGVsbG8="; // "Hello"
        let bytes = decode_data_uri(payload).expect("decode base64 data uri");
        assert_eq!(bytes, b"Hello");
    }

    #[test]
    fn data_uri_percent_round_trips() {
        let payload = "text/plain,Hello%20%41"; // "Hello A"
        let bytes = decode_data_uri(payload).expect("decode percent data uri");
        assert_eq!(bytes, b"Hello A");
    }

    #[test]
    fn data_uri_missing_comma_rejected() {
        let err = decode_data_uri("image/png;base64").unwrap_err();
        assert!(matches!(err, BlazenError::Validation { .. }));
    }

    #[tokio::test]
    async fn unsupported_scheme_is_validation_error() {
        let err = fetch_image_bytes("ftp://example.com/img.png")
            .await
            .unwrap_err();
        assert!(matches!(err, BlazenError::Validation { .. }));
    }

    /// End-to-end `generate_3d` rejects text-only requests before any
    /// pipeline work. Ignored because constructing a `TripoSrBackend`
    /// requires loaded weights; the validation path itself is exercised
    /// indirectly by `unsupported_scheme_is_validation_error` above.
    #[tokio::test]
    #[ignore = "requires TripoSR weights to construct a TripoSrBackend"]
    async fn text_only_request_returns_validation_error() {
        // Documentation-only: with real weights, this would call
        // `backend.generate_3d(ThreeDRequest::new("a duck"))` and assert
        // the returned error matches `BlazenError::Validation { .. }`
        // with a message mentioning `image_url`.
    }
}
