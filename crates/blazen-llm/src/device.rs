//! Hardware device selection for compute backends.

use serde::{Deserialize, Serialize};
use std::fmt;

/// Selects which hardware device to target for model execution.
///
/// `Remote { endpoint }` is for external-engine proxy backends
/// (`blazen-llm-vllm`, `blazen-llm-ollama`, ...) where the actual hardware
/// lives in another process — typically another host. The proxy provider
/// surfaces the upstream server's URL as the "device"; the
/// [`Pool`] conversion collapses it to [`Pool::Remote`] so the
/// `ModelManager`'s host RAM and per-GPU VRAM budgets are not double-counted
/// against the proxy.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
#[cfg_attr(feature = "tsify", derive(tsify_next::Tsify))]
#[cfg_attr(feature = "tsify", tsify(into_wasm_abi, from_wasm_abi))]
#[serde(rename_all = "snake_case")]
pub enum Device {
    /// Run on the CPU (default).
    #[default]
    Cpu,
    /// Run on an NVIDIA CUDA GPU at the given device index.
    Cuda(usize),
    /// Run on Apple Silicon GPU (Metal).
    Metal,
    /// Run on a Vulkan-capable GPU at the given device index.
    Vulkan(usize),
    /// Run on an AMD `ROCm` GPU at the given device index.
    Rocm(usize),
    /// Run on a *remote* inference engine identified by its base URL
    /// (e.g. `"http://vllm-1.cluster.internal:8000"`). Used by proxy
    /// providers; not a target the local runtime can dispatch onto.
    Remote { endpoint: String },
}

impl Device {
    /// Bytes this device's model placement should count against any
    /// host-side budget. Returns `0` for [`Self::Remote`] — proxy
    /// providers never charge the local [`ModelManager`] for weights
    /// that physically live in another process.
    ///
    /// All other variants return `0` here too: the on-device VRAM /
    /// host-RAM footprint comes from the model itself
    /// ([`LocalModel::memory_bytes`](crate::traits::LocalModel::memory_bytes)),
    /// not the device tag. The method exists so callers can ask the
    /// device alone "should I count this?" without consulting the model.
    #[must_use]
    #[allow(clippy::unused_self)] // signature is forward-looking; today every variant returns 0
    pub fn memory_bytes(&self) -> u64 {
        // Every variant currently returns 0: the on-device footprint
        // comes from the model itself (`LocalModel::memory_bytes`), not
        // the device tag. The method exists so callers can ask the
        // device alone "should I count this?" — for `Remote` the answer
        // is unambiguously no, and the parity with the other arms makes
        // future per-device adjustments a one-line change.
        let _ = self;
        0
    }
}

impl Device {
    /// Auto-detect the device from the `BLAZEN_DEVICE` environment variable.
    ///
    /// Falls back to [`Device::Cpu`] when the variable is unset or empty.
    ///
    /// # Errors
    ///
    /// Returns [`BlazenError`](crate::error::BlazenError) if the environment
    /// variable is set but contains an unparseable value.
    pub fn from_env() -> Result<Self, crate::error::BlazenError> {
        match std::env::var("BLAZEN_DEVICE") {
            Ok(val) if !val.is_empty() => Self::parse(&val),
            _ => Ok(Self::Cpu),
        }
    }

    /// Parse a device specifier string.
    ///
    /// Accepted formats (case-insensitive):
    ///
    /// | Input        | Result          |
    /// |--------------|-----------------|
    /// | `"cpu"`      | `Device::Cpu`   |
    /// | `"cuda"`     | `Device::Cuda(0)` |
    /// | `"cuda:0"`   | `Device::Cuda(0)` |
    /// | `"cuda:3"`   | `Device::Cuda(3)` |
    /// | `"metal"`    | `Device::Metal` |
    /// | `"vulkan:1"` | `Device::Vulkan(1)` |
    /// | `"rocm:0"`   | `Device::Rocm(0)` |
    /// | `"remote:http://host:8000"` | `Device::Remote { endpoint: "http://host:8000".into() }` |
    ///
    /// # Errors
    ///
    /// Returns [`BlazenError`](crate::error::BlazenError) when the string does
    /// not match any recognised device specifier.
    pub fn parse(s: &str) -> Result<Self, crate::error::BlazenError> {
        let s = s.trim();
        // `Remote` carries a URL that must survive case and contains its
        // own `:` separator, so handle it before the lowercased split.
        if let Some(rest) = s
            .strip_prefix("remote:")
            .or_else(|| s.strip_prefix("Remote:"))
            .or_else(|| s.strip_prefix("REMOTE:"))
        {
            if rest.is_empty() {
                return Err(crate::error::BlazenError::validation(
                    "remote device requires an endpoint URL: \"remote:<url>\"",
                ));
            }
            return Ok(Self::Remote {
                endpoint: rest.to_string(),
            });
        }
        let lower = s.to_ascii_lowercase();

        if let Some((name, idx)) = lower.split_once(':') {
            let index = idx.parse::<usize>().map_err(|_| {
                crate::error::BlazenError::validation(format!(
                    "invalid device index in \"{s}\": expected a non-negative integer"
                ))
            })?;
            match name {
                "cuda" => Ok(Self::Cuda(index)),
                "vulkan" => Ok(Self::Vulkan(index)),
                "rocm" => Ok(Self::Rocm(index)),
                _ => Err(crate::error::BlazenError::validation(format!(
                    "unknown device \"{s}\": expected cpu, cuda, metal, vulkan, or rocm"
                ))),
            }
        } else {
            match lower.as_str() {
                "cpu" => Ok(Self::Cpu),
                "cuda" => Ok(Self::Cuda(0)),
                "metal" => Ok(Self::Metal),
                "vulkan" => Ok(Self::Vulkan(0)),
                "rocm" => Ok(Self::Rocm(0)),
                _ => Err(crate::error::BlazenError::validation(format!(
                    "unknown device \"{s}\": expected cpu, cuda, metal, vulkan, or rocm"
                ))),
            }
        }
    }
}

impl fmt::Display for Device {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Cpu => f.write_str("cpu"),
            Self::Cuda(idx) => write!(f, "cuda:{idx}"),
            Self::Metal => f.write_str("metal"),
            Self::Vulkan(idx) => write!(f, "vulkan:{idx}"),
            Self::Rocm(idx) => write!(f, "rocm:{idx}"),
            Self::Remote { endpoint } => write!(f, "remote:{endpoint}"),
        }
    }
}

/// Memory pool identifier used by the model manager for per-device budgeting.
///
/// Collapses the finer-grained [`Device`] enum (which distinguishes `CUDA` from
/// `Vulkan` from `ROCm`) into the two memory regions that actually matter for a
/// budget: host RAM and GPU VRAM (indexed when there are multiple GPUs).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[cfg_attr(feature = "tsify", derive(tsify_next::Tsify))]
#[cfg_attr(feature = "tsify", tsify(into_wasm_abi, from_wasm_abi))]
#[serde(rename_all = "snake_case")]
pub enum Pool {
    /// Host RAM pool.
    Cpu,
    /// GPU VRAM pool at the given device index. Metal collapses to `Gpu(0)`.
    Gpu(usize),
    /// Off-host pool — the memory lives in another process / host (remote
    /// inference engine reached over HTTP). Proxy backends report this so
    /// the `ModelManager` knows to skip local budget bookkeeping for them.
    Remote,
}

impl From<&Device> for Pool {
    fn from(d: &Device) -> Self {
        match d {
            Device::Cpu => Self::Cpu,
            Device::Cuda(n) | Device::Vulkan(n) | Device::Rocm(n) => Self::Gpu(*n),
            Device::Metal => Self::Gpu(0),
            Device::Remote { .. } => Self::Remote,
        }
    }
}

impl From<Device> for Pool {
    fn from(d: Device) -> Self {
        Self::from(&d)
    }
}

impl fmt::Display for Pool {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Cpu => f.write_str("cpu"),
            Self::Gpu(idx) => write!(f, "gpu:{idx}"),
            Self::Remote => f.write_str("remote"),
        }
    }
}

#[cfg(test)]
// The env-var tests below wrap `std::env::{set_var, remove_var}` in
// `unsafe` blocks (required since Rust 2024). The workspace-wide
// `unsafe_code = "warn"` lint flags them; allow it in test scope only.
#[allow(unsafe_code)]
mod tests {
    use super::*;

    // -- parse ---------------------------------------------------------------

    #[test]
    fn parse_cpu() {
        assert_eq!(Device::parse("cpu").unwrap(), Device::Cpu);
    }

    #[test]
    fn parse_cuda_bare() {
        assert_eq!(Device::parse("cuda").unwrap(), Device::Cuda(0));
    }

    #[test]
    fn parse_cuda_zero() {
        assert_eq!(Device::parse("cuda:0").unwrap(), Device::Cuda(0));
    }

    #[test]
    fn parse_cuda_nonzero() {
        assert_eq!(Device::parse("cuda:3").unwrap(), Device::Cuda(3));
    }

    #[test]
    fn parse_metal() {
        assert_eq!(Device::parse("metal").unwrap(), Device::Metal);
    }

    #[test]
    fn parse_vulkan() {
        assert_eq!(Device::parse("vulkan:0").unwrap(), Device::Vulkan(0));
    }

    #[test]
    fn parse_vulkan_bare() {
        assert_eq!(Device::parse("vulkan").unwrap(), Device::Vulkan(0));
    }

    #[test]
    fn parse_rocm() {
        assert_eq!(Device::parse("rocm:1").unwrap(), Device::Rocm(1));
    }

    #[test]
    fn parse_rocm_bare() {
        assert_eq!(Device::parse("rocm").unwrap(), Device::Rocm(0));
    }

    // -- case insensitivity --------------------------------------------------

    #[test]
    fn parse_case_insensitive_cpu() {
        assert_eq!(Device::parse("CPU").unwrap(), Device::Cpu);
    }

    #[test]
    fn parse_case_insensitive_cuda() {
        assert_eq!(Device::parse("Cuda:0").unwrap(), Device::Cuda(0));
    }

    #[test]
    fn parse_case_insensitive_metal() {
        assert_eq!(Device::parse("METAL").unwrap(), Device::Metal);
    }

    #[test]
    fn parse_case_insensitive_vulkan() {
        assert_eq!(Device::parse("VULKAN:2").unwrap(), Device::Vulkan(2));
    }

    #[test]
    fn parse_case_insensitive_rocm() {
        assert_eq!(Device::parse("RoCm:0").unwrap(), Device::Rocm(0));
    }

    // -- invalid inputs ------------------------------------------------------

    #[test]
    fn parse_invalid_name() {
        assert!(Device::parse("tpu").is_err());
    }

    #[test]
    fn parse_invalid_index() {
        assert!(Device::parse("cuda:abc").is_err());
    }

    #[test]
    fn parse_empty() {
        assert!(Device::parse("").is_err());
    }

    #[test]
    fn parse_unknown_with_index() {
        assert!(Device::parse("tpu:0").is_err());
    }

    // -- Display round-trip --------------------------------------------------

    #[test]
    fn display_roundtrip_cpu() {
        let d = Device::Cpu;
        assert_eq!(Device::parse(&d.to_string()).unwrap(), d);
    }

    #[test]
    fn display_roundtrip_cuda() {
        let d = Device::Cuda(2);
        assert_eq!(Device::parse(&d.to_string()).unwrap(), d);
    }

    #[test]
    fn display_roundtrip_metal() {
        let d = Device::Metal;
        assert_eq!(Device::parse(&d.to_string()).unwrap(), d);
    }

    #[test]
    fn display_roundtrip_vulkan() {
        let d = Device::Vulkan(1);
        assert_eq!(Device::parse(&d.to_string()).unwrap(), d);
    }

    #[test]
    fn display_roundtrip_rocm() {
        let d = Device::Rocm(0);
        assert_eq!(Device::parse(&d.to_string()).unwrap(), d);
    }

    // -- Default -------------------------------------------------------------

    #[test]
    fn default_is_cpu() {
        assert_eq!(Device::default(), Device::Cpu);
    }

    // -- from_env ------------------------------------------------------------

    #[test]
    fn from_env_unset() {
        // Remove the var so the fallback to Cpu fires.
        // This test may be flaky in parallel if another test sets the var
        // concurrently, but that is acceptable for unit tests.
        unsafe { std::env::remove_var("BLAZEN_DEVICE") };
        assert_eq!(Device::from_env().unwrap(), Device::Cpu);
    }

    #[test]
    fn from_env_set() {
        unsafe { std::env::set_var("BLAZEN_DEVICE", "metal") };
        assert_eq!(Device::from_env().unwrap(), Device::Metal);
        unsafe { std::env::remove_var("BLAZEN_DEVICE") };
    }

    #[test]
    fn from_env_invalid() {
        unsafe { std::env::set_var("BLAZEN_DEVICE", "bad_device") };
        assert!(Device::from_env().is_err());
        unsafe { std::env::remove_var("BLAZEN_DEVICE") };
    }

    // -- Pool ----------------------------------------------------------------

    #[test]
    fn pool_from_cpu_device() {
        assert_eq!(Pool::from(&Device::Cpu), Pool::Cpu);
    }

    #[test]
    fn pool_from_cuda_device_preserves_index() {
        assert_eq!(Pool::from(&Device::Cuda(3)), Pool::Gpu(3));
    }

    #[test]
    fn pool_from_metal_device() {
        assert_eq!(Pool::from(&Device::Metal), Pool::Gpu(0));
    }

    #[test]
    fn pool_from_vulkan_device_preserves_index() {
        assert_eq!(Pool::from(&Device::Vulkan(2)), Pool::Gpu(2));
    }

    #[test]
    fn pool_from_rocm_device_preserves_index() {
        assert_eq!(Pool::from(&Device::Rocm(1)), Pool::Gpu(1));
    }

    #[test]
    fn pool_display_cpu() {
        assert_eq!(Pool::Cpu.to_string(), "cpu");
    }

    #[test]
    fn pool_display_gpu() {
        assert_eq!(Pool::Gpu(2).to_string(), "gpu:2");
    }

    #[test]
    fn pool_serde_roundtrip_cpu() {
        let p = Pool::Cpu;
        let json = serde_json::to_string(&p).unwrap();
        assert_eq!(json, "\"cpu\"");
        assert_eq!(serde_json::from_str::<Pool>(&json).unwrap(), p);
    }

    #[test]
    fn pool_serde_roundtrip_gpu() {
        let p = Pool::Gpu(2);
        let json = serde_json::to_string(&p).unwrap();
        assert_eq!(serde_json::from_str::<Pool>(&json).unwrap(), p);
    }

    // -- Remote (proxy-backend device) --------------------------------------

    #[test]
    fn parse_remote_http_url() {
        let d = Device::parse("remote:http://vllm.local:8000").unwrap();
        assert_eq!(
            d,
            Device::Remote {
                endpoint: "http://vllm.local:8000".into()
            }
        );
    }

    #[test]
    fn parse_remote_https_url_preserves_path() {
        let d = Device::parse("remote:https://api.example.com/v1").unwrap();
        assert_eq!(
            d,
            Device::Remote {
                endpoint: "https://api.example.com/v1".into()
            }
        );
    }

    #[test]
    fn parse_remote_missing_endpoint_errors() {
        assert!(Device::parse("remote:").is_err());
    }

    #[test]
    fn display_remote_roundtrips_via_parse() {
        let d = Device::Remote {
            endpoint: "http://h:8000".into(),
        };
        assert_eq!(Device::parse(&d.to_string()).unwrap(), d);
    }

    #[test]
    fn pool_from_remote_device() {
        let d = Device::Remote {
            endpoint: "http://h:8000".into(),
        };
        assert_eq!(Pool::from(&d), Pool::Remote);
    }

    #[test]
    fn remote_device_memory_bytes_is_zero() {
        let d = Device::Remote {
            endpoint: "http://h:8000".into(),
        };
        assert_eq!(d.memory_bytes(), 0);
    }

    #[test]
    fn pool_display_remote() {
        assert_eq!(Pool::Remote.to_string(), "remote");
    }
}
