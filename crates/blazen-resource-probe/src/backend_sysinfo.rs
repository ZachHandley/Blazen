use sysinfo::System;

const BYTES_PER_MB: u64 = 1024 * 1024;

/// Synchronous CPU+RAM probe — sysinfo's refresh is non-blocking but
/// allocates; we still keep this sync because callers run it from inside
/// `probe_once()` which is itself an async wrapper.
///
/// Returns `(cpu_cores, total_ram_mb, free_ram_mb)`. Cores are *logical*
/// cores (matching what tokio sees); free RAM uses sysinfo's "available"
/// number which excludes reclaimable cache (the more honest figure).
#[must_use]
pub fn probe_cpu_ram() -> (u32, u64, u64) {
    let mut sys = System::new();
    sys.refresh_memory();
    sys.refresh_cpu_list(sysinfo::CpuRefreshKind::nothing());

    let cpu_cores = u32::try_from(sys.cpus().len()).unwrap_or(u32::MAX);
    let total_ram_mb = sys.total_memory() / BYTES_PER_MB;
    let free_ram_mb = sys.available_memory() / BYTES_PER_MB;

    (cpu_cores, total_ram_mb, free_ram_mb)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cpu_and_ram_probed_on_all_platforms() {
        let (cores, total_mb, free_mb) = probe_cpu_ram();
        assert!(cores > 0, "expected at least one logical CPU");
        assert!(total_mb > 0, "expected non-zero total RAM");
        assert!(free_mb <= total_mb, "free RAM cannot exceed total");
    }
}
