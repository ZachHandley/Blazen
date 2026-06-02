//! NVIDIA NVML backend. Compiled only when both the `nvml` feature is
//! enabled AND the target is Linux. NVML is the same API `nvidia-smi`
//! uses, so on a healthy NVIDIA driver install this is the
//! lowest-overhead per-device free-VRAM source available.
//!
//! Tests behind the `nvml-live` feature exercise real hardware and should
//! only be enabled by a developer on an NVIDIA-equipped host.

#[cfg(all(target_os = "linux", feature = "nvml"))]
use crate::types::GpuVendor;
use crate::types::{GpuInfo, ProbeError};

/// Probe NVIDIA GPUs via NVML. On non-Linux targets or builds without
/// the `nvml` feature, this is a no-op returning an empty vector.
///
/// # Errors
/// Returns `ProbeError::NvmlQuery` if NVML is reachable but a per-device
/// memory query fails (rare — usually only on a misbehaving driver).
#[cfg(all(target_os = "linux", feature = "nvml"))]
pub async fn probe() -> Result<Vec<GpuInfo>, ProbeError> {
    tokio::task::spawn_blocking(probe_blocking)
        .await
        .map_err(|e| ProbeError::NvmlQuery(format!("join: {e}")))?
}

/// No-op stub for non-Linux / non-`nvml` builds. Returns an empty vector.
///
/// # Errors
/// Infallible on this build target; preserved to match the live signature.
#[cfg(not(all(target_os = "linux", feature = "nvml")))]
#[allow(clippy::unused_async)]
pub async fn probe() -> Result<Vec<GpuInfo>, ProbeError> {
    Ok(Vec::new())
}

#[cfg(all(target_os = "linux", feature = "nvml"))]
fn probe_blocking() -> Result<Vec<GpuInfo>, ProbeError> {
    use nvml_wrapper::Nvml;

    const BYTES_PER_MB: u64 = 1024 * 1024;

    let nvml = match Nvml::init() {
        Ok(n) => n,
        Err(e) => {
            // Driver absent / loading failed / etc. — treat as "no NVIDIA
            // GPUs visible" rather than a fatal error.
            tracing::debug!(error = %e, "NVML init failed; reporting zero NVIDIA GPUs");
            return Ok(Vec::new());
        }
    };

    let count = nvml
        .device_count()
        .map_err(|e| ProbeError::NvmlQuery(e.to_string()))?;

    let mut out = Vec::with_capacity(count as usize);
    for idx in 0..count {
        let device = match nvml.device_by_index(idx) {
            Ok(d) => d,
            Err(e) => {
                tracing::warn!(index = idx, error = %e, "NVML device_by_index failed; skipping");
                continue;
            }
        };

        let name = device.name().unwrap_or_else(|_| format!("NVIDIA #{idx}"));
        let uuid = device.uuid().ok();
        let mem = device
            .memory_info()
            .map_err(|e| ProbeError::NvmlQuery(e.to_string()))?;

        out.push(GpuInfo {
            vendor: GpuVendor::Nvidia,
            index: idx,
            name,
            uuid,
            total_vram_mb: mem.total / BYTES_PER_MB,
            free_vram_mb: Some(mem.free / BYTES_PER_MB),
            used_vram_mb: Some(mem.used / BYTES_PER_MB),
        });
    }

    Ok(out)
}

#[cfg(all(test, feature = "nvml-live"))]
mod live_tests {
    use super::*;

    #[tokio::test]
    async fn probe_returns_nonzero_total_vram_on_nvidia() {
        let gpus = probe().await.expect("probe must succeed on NVIDIA host");
        assert!(!gpus.is_empty(), "expected at least one NVIDIA GPU");
        for gpu in &gpus {
            assert!(
                gpu.total_vram_mb > 0,
                "GPU {} reported 0 total VRAM",
                gpu.index
            );
            assert!(gpu.free_vram_mb.is_some(), "NVML must expose free VRAM");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn probe_degrades_gracefully_without_nvml() {
        // Without the `nvml-live` feature enabled OR on a non-NVIDIA host,
        // the probe must return Ok with an empty vec — never panic, never
        // error. This is the contract every backend honors so the merged
        // snapshot stays consistent.
        let result = probe().await;
        assert!(result.is_ok(), "probe must not error in absence of NVML");
    }
}
