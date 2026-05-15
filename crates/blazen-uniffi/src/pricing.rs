//! UniFFI exports for the pricing registry.
//!
//! Surface intentionally minimal: only the runtime *refresh* path is
//! exposed for now. Cost computation flows through the existing
//! `CompletionResponse.usage.cost_usd` field, which is computed from the
//! global registry — calling [`refresh_pricing`] at app startup populates
//! that registry with the latest catalog from blazen.dev so cost numbers
//! reflect current pricing for the ~1600+ models the build-time baked
//! baseline doesn't carry.

use crate::errors::{BlazenError, BlazenResult};

/// Refresh the pricing registry from a remote catalog. `url` defaults to
/// the blazen.dev Cloudflare Worker, which mirrors models.dev plus live
/// OpenRouter / Together pricing on a daily cron.
///
/// Returns the number of entries registered. Misses still return `null`
/// from `compute_cost`; no automatic retry / cache layer beyond the
/// global registry.
#[uniffi::export(async_runtime = "tokio")]
pub async fn refresh_pricing(url: Option<String>) -> BlazenResult<u32> {
    let target = url.unwrap_or_else(|| blazen_llm::DEFAULT_PRICING_URL.to_owned());
    let count = blazen_llm::refresh_default_with_url(&target)
        .await
        .map_err(|e| BlazenError::Internal {
            message: e.to_string(),
        })?;
    Ok(u32::try_from(count).unwrap_or(u32::MAX))
}
