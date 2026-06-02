use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GpuVendor {
    Nvidia,
    Amd,
    Intel,
    AppleMetal,
    Unknown,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct GpuInfo {
    pub vendor: GpuVendor,
    pub index: u32,
    pub name: String,
    pub uuid: Option<String>,
    pub total_vram_mb: u64,
    /// `None` when the platform does not expose free-VRAM (notably macOS
    /// Apple Silicon — Metal does not expose this via any public API).
    pub free_vram_mb: Option<u64>,
    pub used_vram_mb: Option<u64>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ProbedResources {
    pub gpus: Vec<GpuInfo>,
    pub cpu_cores: u32,
    pub total_ram_mb: u64,
    pub free_ram_mb: u64,
    pub last_probed_at: chrono::DateTime<chrono::Utc>,
}

impl ProbedResources {
    /// Total free VRAM across all GPUs whose backend exposes it. GPUs that
    /// report `None` (e.g. Apple Silicon) are skipped — the sum reflects
    /// only the share we can actually account for.
    #[must_use]
    pub fn total_free_vram_mb(&self) -> u64 {
        self.gpus.iter().filter_map(|g| g.free_vram_mb).sum()
    }

    /// Total VRAM across all GPUs (this number is always known).
    #[must_use]
    pub fn total_vram_mb(&self) -> u64 {
        self.gpus.iter().map(|g| g.total_vram_mb).sum()
    }

    /// `true` if every probed GPU exposes free-VRAM. Lets callers decide
    /// whether `total_free_vram_mb` is meaningful vs. an under-count.
    #[must_use]
    pub fn free_vram_is_complete(&self) -> bool {
        !self.gpus.is_empty() && self.gpus.iter().all(|g| g.free_vram_mb.is_some())
    }
}

#[derive(Debug, thiserror::Error)]
pub enum ProbeError {
    #[error("NVML init failed: {0}")]
    NvmlInit(String),
    #[error("NVML query failed: {0}")]
    NvmlQuery(String),
    #[error("rocm-smi shell-out failed: {0}")]
    RocmSmi(String),
    #[error("system_profiler shell-out failed: {0}")]
    SystemProfiler(String),
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    #[error("json parse: {0}")]
    Json(#[from] serde_json::Error),
}
