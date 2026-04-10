//! Model quantization format selection.
//!
//! [`Quantization`] represents the most common weight quantization formats
//! used by GGUF model files and GPU-based quantisation schemes (GPTQ, AWQ).

use serde::{Deserialize, Serialize};

use crate::error::BlazenError;

/// Quantization format for a model's weights.
///
/// Covers IEEE floating-point formats, GGML k-quant levels, and the two
/// most popular GPU quantisation schemes. The default is [`Q4_K_M`](Self::Q4_K_M)
/// -- the most widely used GGUF quantization for consumer hardware.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[cfg_attr(feature = "tsify", derive(tsify_next::Tsify))]
#[cfg_attr(feature = "tsify", tsify(into_wasm_abi, from_wasm_abi))]
pub enum Quantization {
    /// Full 32-bit IEEE 754 floating-point.
    #[serde(rename = "f32")]
    F32,
    /// Half-precision 16-bit IEEE 754 floating-point.
    #[serde(rename = "f16")]
    F16,
    /// Brain floating-point 16-bit.
    #[serde(rename = "bf16")]
    BF16,
    /// 8-bit quantization (GGML).
    #[serde(rename = "q8_0")]
    Q8_0,
    /// 6-bit k-quant.
    #[serde(rename = "q6_k")]
    Q6K,
    /// 5-bit k-quant (medium).
    #[serde(rename = "q5_k_m")]
    Q5KM,
    /// 5-bit k-quant (small).
    #[serde(rename = "q5_k_s")]
    Q5KS,
    /// 4-bit k-quant (medium) -- the most common choice for consumer hardware.
    #[serde(rename = "q4_k_m")]
    #[default]
    Q4KM,
    /// 4-bit k-quant (small).
    #[serde(rename = "q4_k_s")]
    Q4KS,
    /// 3-bit k-quant (medium).
    #[serde(rename = "q3_k_m")]
    Q3KM,
    /// 2-bit k-quant.
    #[serde(rename = "q2_k")]
    Q2K,
    /// GPTQ 4-bit GPU quantization.
    #[serde(rename = "gptq-4bit")]
    Gptq4Bit,
    /// AWQ 4-bit GPU quantization.
    #[serde(rename = "awq-4bit")]
    Awq4Bit,
}

impl Quantization {
    /// All variants in descending precision order.
    const ALL: [Self; 13] = [
        Self::F32,
        Self::F16,
        Self::BF16,
        Self::Q8_0,
        Self::Q6K,
        Self::Q5KM,
        Self::Q5KS,
        Self::Q4KM,
        Self::Q4KS,
        Self::Q3KM,
        Self::Q2K,
        Self::Gptq4Bit,
        Self::Awq4Bit,
    ];

    /// Returns the canonical GGUF-format string used in model file names.
    ///
    /// # Examples
    ///
    /// ```
    /// use blazen_llm::Quantization;
    ///
    /// assert_eq!(Quantization::Q4KM.as_gguf_str(), "q4_k_m");
    /// assert_eq!(Quantization::Gptq4Bit.as_gguf_str(), "gptq-4bit");
    /// ```
    #[must_use]
    pub const fn as_gguf_str(&self) -> &'static str {
        match self {
            Self::F32 => "f32",
            Self::F16 => "f16",
            Self::BF16 => "bf16",
            Self::Q8_0 => "q8_0",
            Self::Q6K => "q6_k",
            Self::Q5KM => "q5_k_m",
            Self::Q5KS => "q5_k_s",
            Self::Q4KM => "q4_k_m",
            Self::Q4KS => "q4_k_s",
            Self::Q3KM => "q3_k_m",
            Self::Q2K => "q2_k",
            Self::Gptq4Bit => "gptq-4bit",
            Self::Awq4Bit => "awq-4bit",
        }
    }

    /// Parse a quantization string, accepting the canonical GGUF names plus
    /// common aliases. Matching is case-insensitive.
    ///
    /// # Accepted formats
    ///
    /// | Canonical      | Aliases (case-insensitive)     |
    /// |----------------|-------------------------------|
    /// | `q4_k_m`       | `Q4_K_M`, `q4km`, `Q4KM`     |
    /// | `q5_k_m`       | `q5km`, `Q5KM`                |
    /// | `q5_k_s`       | `q5ks`, `Q5KS`                |
    /// | `q4_k_s`       | `q4ks`, `Q4KS`                |
    /// | `q3_k_m`       | `q3km`, `Q3KM`                |
    /// | `q6_k`         | `q6k`, `Q6K`                  |
    /// | `q2_k`         | `q2k`, `Q2K`                  |
    /// | `gptq-4bit`    | `gptq4`, `gptq4bit`           |
    /// | `awq-4bit`     | `awq4`, `awq4bit`             |
    ///
    /// # Errors
    ///
    /// Returns [`BlazenError::Validation`] when the string does not match any
    /// known quantization format.
    pub fn parse(s: &str) -> Result<Self, BlazenError> {
        let lower = s.to_lowercase();
        match lower.as_str() {
            "f32" => Ok(Self::F32),
            "f16" => Ok(Self::F16),
            "bf16" => Ok(Self::BF16),
            "q8_0" | "q80" => Ok(Self::Q8_0),
            "q6_k" | "q6k" => Ok(Self::Q6K),
            "q5_k_m" | "q5km" => Ok(Self::Q5KM),
            "q5_k_s" | "q5ks" => Ok(Self::Q5KS),
            "q4_k_m" | "q4km" => Ok(Self::Q4KM),
            "q4_k_s" | "q4ks" => Ok(Self::Q4KS),
            "q3_k_m" | "q3km" => Ok(Self::Q3KM),
            "q2_k" | "q2k" => Ok(Self::Q2K),
            "gptq-4bit" | "gptq4" | "gptq4bit" => Ok(Self::Gptq4Bit),
            "awq-4bit" | "awq4" | "awq4bit" => Ok(Self::Awq4Bit),
            _ => Err(BlazenError::Validation {
                field: Some("quantization".into()),
                message: format!(
                    "unknown quantization format {s:?}; expected one of: {}",
                    Self::ALL
                        .iter()
                        .map(Self::as_gguf_str)
                        .collect::<Vec<_>>()
                        .join(", ")
                ),
            }),
        }
    }
}

impl std::fmt::Display for Quantization {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_gguf_str())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_all_canonical_strings() {
        let cases = [
            ("f32", Quantization::F32),
            ("f16", Quantization::F16),
            ("bf16", Quantization::BF16),
            ("q8_0", Quantization::Q8_0),
            ("q6_k", Quantization::Q6K),
            ("q5_k_m", Quantization::Q5KM),
            ("q5_k_s", Quantization::Q5KS),
            ("q4_k_m", Quantization::Q4KM),
            ("q4_k_s", Quantization::Q4KS),
            ("q3_k_m", Quantization::Q3KM),
            ("q2_k", Quantization::Q2K),
            ("gptq-4bit", Quantization::Gptq4Bit),
            ("awq-4bit", Quantization::Awq4Bit),
        ];
        for (input, expected) in cases {
            assert_eq!(
                Quantization::parse(input).unwrap(),
                expected,
                "failed to parse canonical string {input:?}"
            );
        }
    }

    #[test]
    fn parse_case_insensitive() {
        assert_eq!(Quantization::parse("Q4_K_M").unwrap(), Quantization::Q4KM);
        assert_eq!(Quantization::parse("F16").unwrap(), Quantization::F16);
        assert_eq!(Quantization::parse("BF16").unwrap(), Quantization::BF16);
        assert_eq!(Quantization::parse("Q8_0").unwrap(), Quantization::Q8_0);
        assert_eq!(
            Quantization::parse("GPTQ-4BIT").unwrap(),
            Quantization::Gptq4Bit
        );
        assert_eq!(
            Quantization::parse("AWQ-4BIT").unwrap(),
            Quantization::Awq4Bit
        );
    }

    #[test]
    fn parse_common_aliases() {
        // No-separator aliases
        assert_eq!(Quantization::parse("q4km").unwrap(), Quantization::Q4KM);
        assert_eq!(Quantization::parse("Q4KM").unwrap(), Quantization::Q4KM);
        assert_eq!(Quantization::parse("q5km").unwrap(), Quantization::Q5KM);
        assert_eq!(Quantization::parse("q5ks").unwrap(), Quantization::Q5KS);
        assert_eq!(Quantization::parse("q4ks").unwrap(), Quantization::Q4KS);
        assert_eq!(Quantization::parse("q3km").unwrap(), Quantization::Q3KM);
        assert_eq!(Quantization::parse("q6k").unwrap(), Quantization::Q6K);
        assert_eq!(Quantization::parse("q2k").unwrap(), Quantization::Q2K);

        // GPU-quant shorthands
        assert_eq!(
            Quantization::parse("gptq4").unwrap(),
            Quantization::Gptq4Bit
        );
        assert_eq!(
            Quantization::parse("gptq4bit").unwrap(),
            Quantization::Gptq4Bit
        );
        assert_eq!(Quantization::parse("awq4").unwrap(), Quantization::Awq4Bit);
        assert_eq!(
            Quantization::parse("awq4bit").unwrap(),
            Quantization::Awq4Bit
        );
    }

    #[test]
    fn parse_invalid_returns_error() {
        let bad = ["", "q1_0", "int8", "fp16", "ggml", "nonsense"];
        for input in bad {
            assert!(
                Quantization::parse(input).is_err(),
                "expected error for {input:?}"
            );
        }
    }

    #[test]
    fn display_roundtrip_all_variants() {
        for variant in Quantization::ALL {
            let display = variant.to_string();
            assert_eq!(display, variant.as_gguf_str());
            assert_eq!(
                Quantization::parse(&display).unwrap(),
                variant,
                "display round-trip failed for {variant:?}"
            );
        }
    }

    #[test]
    fn default_is_q4_k_m() {
        assert_eq!(Quantization::default(), Quantization::Q4KM);
    }

    #[test]
    fn as_gguf_str_all_variants() {
        let expected = [
            (Quantization::F32, "f32"),
            (Quantization::F16, "f16"),
            (Quantization::BF16, "bf16"),
            (Quantization::Q8_0, "q8_0"),
            (Quantization::Q6K, "q6_k"),
            (Quantization::Q5KM, "q5_k_m"),
            (Quantization::Q5KS, "q5_k_s"),
            (Quantization::Q4KM, "q4_k_m"),
            (Quantization::Q4KS, "q4_k_s"),
            (Quantization::Q3KM, "q3_k_m"),
            (Quantization::Q2K, "q2_k"),
            (Quantization::Gptq4Bit, "gptq-4bit"),
            (Quantization::Awq4Bit, "awq-4bit"),
        ];
        for (variant, gguf) in expected {
            assert_eq!(
                variant.as_gguf_str(),
                gguf,
                "wrong gguf str for {variant:?}"
            );
        }
    }

    #[test]
    fn serde_roundtrip() {
        for variant in Quantization::ALL {
            let json = serde_json::to_string(&variant).unwrap();
            let parsed: Quantization = serde_json::from_str(&json).unwrap();
            assert_eq!(parsed, variant, "serde round-trip failed for {variant:?}");
            // Serde repr should match the GGUF string
            assert_eq!(
                json,
                format!("\"{}\"", variant.as_gguf_str()),
                "serde repr mismatch for {variant:?}"
            );
        }
    }
}
