//! AMD `ROCm` backend. Shells out to `rocm-smi --showmeminfo vram --json`
//! and parses the per-device entries. Degrades to an empty vector if the
//! binary is absent (so this backend is harmless to enable on hosts
//! without AMD GPUs).
//!
//! `rocm-smi` JSON shape (abbreviated):
//! ```json
//! {
//!   "card0": {
//!     "GPU ID": "0x740c",
//!     "Card series": "AMD Radeon RX 7900 XTX",
//!     "VRAM Total Memory (B)": "25753026560",
//!     "VRAM Total Used Memory (B)": "172716032"
//!   }
//! }
//! ```

use crate::types::{GpuInfo, ProbeError};
#[cfg(target_os = "linux")]
use crate::types::GpuVendor;
#[cfg(target_os = "linux")]
use std::collections::BTreeMap;

#[cfg(target_os = "linux")]
const BYTES_PER_MB: u64 = 1024 * 1024;

/// Probe AMD GPUs via `rocm-smi --json`.
///
/// # Errors
/// Returns `ProbeError::RocmSmi` if the binary is on PATH but exits
/// non-zero in an unexpected way. A missing `rocm-smi` returns Ok(empty).
/// Returns `ProbeError::Json` if the binary's output cannot be parsed.
#[cfg(target_os = "linux")]
pub async fn probe() -> Result<Vec<GpuInfo>, ProbeError> {
    let output = match tokio::process::Command::new("rocm-smi")
        .args(["--showmeminfo", "vram", "--json"])
        .output()
        .await
    {
        Ok(o) => o,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
            tracing::debug!("rocm-smi not on PATH; reporting zero AMD GPUs");
            return Ok(Vec::new());
        }
        Err(e) => return Err(ProbeError::RocmSmi(e.to_string())),
    };

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        tracing::debug!(stderr = %stderr, "rocm-smi exited non-zero; reporting zero AMD GPUs");
        return Ok(Vec::new());
    }

    let raw: BTreeMap<String, serde_json::Value> = serde_json::from_slice(&output.stdout)?;

    let mut out = Vec::new();
    for (idx, (card_key, entry)) in raw.iter().enumerate() {
        if !card_key.starts_with("card") {
            continue;
        }
        let total_bytes: u64 = entry
            .get("VRAM Total Memory (B)")
            .and_then(|v| v.as_str())
            .and_then(|s| s.parse().ok())
            .unwrap_or(0);
        let used_bytes: u64 = entry
            .get("VRAM Total Used Memory (B)")
            .and_then(|v| v.as_str())
            .and_then(|s| s.parse().ok())
            .unwrap_or(0);

        if total_bytes == 0 {
            continue;
        }

        let name = entry
            .get("Card series")
            .and_then(|v| v.as_str())
            .map_or_else(|| format!("AMD {card_key}"), str::to_string);
        let uuid = entry
            .get("GPU ID")
            .and_then(|v| v.as_str())
            .map(str::to_string);

        let index = u32::try_from(idx).unwrap_or(u32::MAX);
        out.push(GpuInfo {
            vendor: GpuVendor::Amd,
            index,
            name,
            uuid,
            total_vram_mb: total_bytes / BYTES_PER_MB,
            free_vram_mb: Some(total_bytes.saturating_sub(used_bytes) / BYTES_PER_MB),
            used_vram_mb: Some(used_bytes / BYTES_PER_MB),
        });
    }

    Ok(out)
}

/// No-op stub for non-Linux builds. Returns an empty vector.
///
/// # Errors
/// Infallible on this build target.
#[cfg(not(target_os = "linux"))]
#[allow(clippy::unused_async)]
pub async fn probe() -> Result<Vec<GpuInfo>, ProbeError> {
    Ok(Vec::new())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn probe_degrades_gracefully_without_rocm() {
        let result = probe().await;
        assert!(
            result.is_ok(),
            "probe must not error when rocm-smi is absent"
        );
    }
}
