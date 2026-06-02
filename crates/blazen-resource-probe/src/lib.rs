//! Cross-platform GPU / CPU / RAM probing for Blazen workers.
//!
//! The probe produces a [`ProbedResources`] snapshot every interval; workers
//! use the latest snapshot to source the real `AdmissionSnapshot`
//! (`vram_free_mb`, `in_flight_vram_mb`) that the control-plane scheduler
//! consumes. Backends degrade gracefully — a missing NVML, absent
//! `rocm-smi` binary, or non-Darwin platform returns an empty GPU vector
//! rather than failing the whole snapshot.
//!
//! Backends:
//! - `backend_nvml` — NVIDIA via NVML (feature-gated `nvml`, Linux only).
//! - `backend_rocm` — AMD `ROCm` via `rocm-smi --json` shell-out (Linux).
//! - `backend_macos` — Apple Metal via `system_profiler SPDisplaysDataType
//!   -json` shell-out (macOS).
//! - `backend_sysinfo` — CPU + RAM via `sysinfo` (all platforms).

pub mod backend_macos;
pub mod backend_nvml;
pub mod backend_rocm;
pub mod backend_sysinfo;
pub mod probe;
pub mod types;

pub use probe::{Probe, ProbeConfig, ProbeHandle};
pub use types::{GpuInfo, GpuVendor, ProbeError, ProbedResources};

/// Probe every backend once and merge the results. The CPU+RAM probe is
/// always run; GPU backends run if their platform/feature conditions hold.
#[must_use = "the returned snapshot is the whole point of calling probe_once"]
pub async fn probe_once() -> ProbedResources {
    let (cpu_cores, total_ram_mb, free_ram_mb) = backend_sysinfo::probe_cpu_ram();

    let mut gpus = Vec::new();

    #[cfg(all(target_os = "linux", feature = "nvml"))]
    {
        if let Ok(nvml_gpus) = backend_nvml::probe().await {
            gpus.extend(nvml_gpus);
        }
    }

    #[cfg(target_os = "linux")]
    {
        if let Ok(rocm_gpus) = backend_rocm::probe().await {
            gpus.extend(rocm_gpus);
        }
    }

    #[cfg(target_os = "macos")]
    {
        if let Ok(metal_gpus) = backend_macos::probe().await {
            gpus.extend(metal_gpus);
        }
    }

    ProbedResources {
        gpus,
        cpu_cores,
        total_ram_mb,
        free_ram_mb,
        last_probed_at: chrono::Utc::now(),
    }
}
