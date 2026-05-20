//! Per-architecture trainable model wrappers.
//!
//! Each submodule mirrors `candle_transformers::models::<arch>`'s forward
//! pass but substitutes [`crate::lora::LoraLinear`] at every site listed
//! in [`crate::config::LoraConfig::target_modules`] (q/k/v/o by default,
//! optionally gate/up/down for MLP LoRA).
//!
//! Wave 2A populates `qwen2`, 2B populates `llama`, 2C populates `mistral`.

pub mod llama;
pub mod mistral;
pub mod qwen2;
