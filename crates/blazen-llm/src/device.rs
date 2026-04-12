//! Hardware device selection for compute backends.

use serde::{Deserialize, Serialize};
use std::fmt;

/// Selects which hardware device to target for model execution.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
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
    ///
    /// # Errors
    ///
    /// Returns [`BlazenError`](crate::error::BlazenError) when the string does
    /// not match any recognised device specifier.
    pub fn parse(s: &str) -> Result<Self, crate::error::BlazenError> {
        let s = s.trim();
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
}
