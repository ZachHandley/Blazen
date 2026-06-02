//! macOS Metal backend. Shells out to `system_profiler
//! SPDisplaysDataType -json` and emits one [`GpuInfo`] per detected
//! display adapter. Apple Silicon does not expose free-VRAM through any
//! public API — `free_vram_mb` is therefore always `None`. Discrete
//! NVIDIA / AMD GPUs on Intel Macs still report only total VRAM here
//! (use the NVML/ROCm backends instead if those GPUs are CUDA/ROCm-
//! accessible).

#[cfg(target_os = "macos")]
use crate::types::GpuVendor;
use crate::types::{GpuInfo, ProbeError};

/// Probe macOS Metal GPUs via `system_profiler`.
///
/// # Errors
/// Returns `ProbeError::SystemProfiler` if the binary is on PATH but
/// exits non-zero. A missing `system_profiler` returns Ok(empty).
/// Returns `ProbeError::Json` if the binary's output cannot be parsed.
#[cfg(target_os = "macos")]
pub async fn probe() -> Result<Vec<GpuInfo>, ProbeError> {
    let output = match tokio::process::Command::new("system_profiler")
        .args(["SPDisplaysDataType", "-json"])
        .output()
        .await
    {
        Ok(o) => o,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
            tracing::debug!("system_profiler missing; reporting zero Metal GPUs");
            return Ok(Vec::new());
        }
        Err(e) => return Err(ProbeError::SystemProfiler(e.to_string())),
    };

    if !output.status.success() {
        return Err(ProbeError::SystemProfiler(format!(
            "system_profiler exited {}",
            output.status
        )));
    }

    let parsed: serde_json::Value = serde_json::from_slice(&output.stdout)?;
    let entries = parsed
        .get("SPDisplaysDataType")
        .and_then(|v| v.as_array())
        .cloned()
        .unwrap_or_default();

    let mut out = Vec::with_capacity(entries.len());
    for (idx, entry) in entries.into_iter().enumerate() {
        let name = entry
            .get("sppci_model")
            .and_then(|v| v.as_str())
            .map(str::to_string)
            .unwrap_or_else(|| format!("Metal #{idx}"));

        // sppci_vendor on Apple Silicon is "sppci_vendor_Apple"; Intel
        // Macs see "Intel", "ATI" (AMD), or "NVIDIA".
        let vendor_raw = entry
            .get("sppci_vendor")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_ascii_lowercase();
        let vendor = if vendor_raw.contains("apple") {
            GpuVendor::AppleMetal
        } else if vendor_raw.contains("amd") || vendor_raw.contains("ati") {
            GpuVendor::Amd
        } else if vendor_raw.contains("nvidia") {
            GpuVendor::Nvidia
        } else if vendor_raw.contains("intel") {
            GpuVendor::Intel
        } else {
            GpuVendor::Unknown
        };

        // VRAM lives under either "spdisplays_vram" (string like "16 GB")
        // for discrete GPUs OR isn't present at all on Apple Silicon
        // (where the GPU shares unified memory). For Apple Silicon we
        // synthesize total_vram_mb from total system RAM later if the
        // caller chooses; here we just parse what's actually printed.
        let total_vram_mb = parse_vram_string(
            entry
                .get("spdisplays_vram")
                .and_then(|v| v.as_str())
                .or_else(|| entry.get("spdisplays_vram_shared").and_then(|v| v.as_str()))
                .unwrap_or(""),
        );

        out.push(GpuInfo {
            vendor,
            index: idx as u32,
            name,
            uuid: None,
            total_vram_mb,
            free_vram_mb: None, // Metal does not expose this publicly.
            used_vram_mb: None,
        });
    }

    Ok(out)
}

/// No-op stub for non-macOS builds. Returns an empty vector.
///
/// # Errors
/// Infallible on this build target.
#[cfg(not(target_os = "macos"))]
#[allow(clippy::unused_async)]
pub async fn probe() -> Result<Vec<GpuInfo>, ProbeError> {
    Ok(Vec::new())
}

#[cfg_attr(not(target_os = "macos"), allow(dead_code))]
fn parse_vram_string(raw: &str) -> u64 {
    // Examples: "16 GB", "8 GB", "1536 MB", "" (Apple Silicon shared).
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return 0;
    }
    let mut parts = trimmed.split_whitespace();
    let n: u64 = parts.next().and_then(|s| s.parse().ok()).unwrap_or(0);
    let unit = parts.next().unwrap_or("MB").to_ascii_uppercase();
    match unit.as_str() {
        "GB" => n * 1024,
        "TB" => n * 1024 * 1024,
        _ => n,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_vram_handles_common_units() {
        assert_eq!(parse_vram_string("16 GB"), 16 * 1024);
        assert_eq!(parse_vram_string("1536 MB"), 1536);
        assert_eq!(parse_vram_string("1 TB"), 1024 * 1024);
        assert_eq!(parse_vram_string(""), 0);
        assert_eq!(parse_vram_string("garbage"), 0);
    }

    #[tokio::test]
    async fn probe_degrades_gracefully_off_macos() {
        let result = probe().await;
        assert!(result.is_ok());
    }
}
