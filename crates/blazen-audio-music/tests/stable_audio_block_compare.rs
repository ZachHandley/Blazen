//! StableAudio tensor-dump comparison harness (Rust side).
//!
//! The companion Python harness at `tests/python/stable_audio_dump.py`
//! writes one `.bin` + `.meta.json` pair per intermediate tensor of the
//! reference `stabilityai/stable-audio-open-small` model. This file
//! provides the Rust-side primitives that the future StableAudio candle
//! port will call to assert block-by-block parity against those dumps.
//!
//! The helpers are intentionally **runtime-only**: there is no actual
//! model code here yet, and this file compiles cleanly without the
//! `musicgen` feature (candle is pulled in as a `[dev-dependencies]`
//! entry on `blazen-audio-music`). If the dump directory is missing,
//! `load_dump` panics at runtime with a clear instruction to regenerate.
//!
//! Tolerances expected by the caller:
//! * FP32: `1e-5`
//! * BF16: `1e-3`

#![allow(dead_code)]
// This is a test-support harness, not production code. The pedantic
// lints (Debug-in-panic, missing # Panics docs, must_use, etc.) add
// noise without value here -- panics with clear context are exactly
// the right behavior for a diff-against-reference test helper.
#![allow(clippy::pedantic, clippy::nursery)]

use std::fs;
use std::path::{Path, PathBuf};

use candle_core::{DType, Device, Shape, Tensor};
use serde_json::Value;

/// Resolve the dump directory the same way the Python harness does:
/// honor `$BLAZEN_STABLEAUDIO_DUMP_DIR` first, fall back to
/// `~/.cache/blazen-stableaudio-research/dumps`.
fn dump_root() -> PathBuf {
    if let Ok(env) = std::env::var("BLAZEN_STABLEAUDIO_DUMP_DIR") {
        return PathBuf::from(env);
    }
    let home = std::env::var("HOME").unwrap_or_else(|_| ".".to_string());
    PathBuf::from(home).join(".cache/blazen-stableaudio-research/dumps")
}

fn parse_dtype(name: &str) -> DType {
    match name {
        "float32" => DType::F32,
        "float16" => DType::F16,
        "bfloat16" => DType::BF16,
        "float64" => DType::F64,
        "int64" => DType::I64,
        "uint8" => DType::U8,
        // candle has no native i32/i16/i8/bool tensor types -- the
        // upstream StableAudio path doesn't use them at the boundaries
        // we dump, so loudly refuse.
        other => panic!("Unsupported dtype `{other}` in dump meta -- extend parse_dtype if needed"),
    }
}

fn dtype_byte_size(dtype: DType) -> usize {
    match dtype {
        DType::F32 | DType::I64 => 4,
        DType::F64 => 8,
        DType::F16 | DType::BF16 => 2,
        DType::U8 | DType::U32 => 4,
        _ => panic!("dtype_byte_size: unhandled {dtype:?}"),
    }
    // Note: candle's U32/I64 widths above are nominal -- we only use F32 /
    // F16 / BF16 / F64 at the StableAudio boundaries we care about.
}

/// Load a previously-dumped tensor by its `block_name`.
///
/// Panics with a clear "run the Python dump first" message if the
/// directory or files don't exist.
pub fn load_dump(block_name: &str) -> Tensor {
    let root = dump_root();
    if !root.exists() {
        panic!(
            "StableAudio dump dir {root:?} does not exist. Regenerate with:\n  \
             pip install stable-audio-tools\n  \
             python tests/python/stable_audio_dump.py"
        );
    }

    let bin_path = root.join(format!("{block_name}.bin"));
    let meta_path = root.join(format!("{block_name}.meta.json"));
    if !bin_path.exists() || !meta_path.exists() {
        panic!(
            "StableAudio dump `{block_name}` missing ({bin_path:?} / {meta_path:?}). \
             Regenerate with `python tests/python/stable_audio_dump.py`."
        );
    }

    load_dump_from_paths(&bin_path, &meta_path, block_name)
}

fn load_dump_from_paths(bin_path: &Path, meta_path: &Path, block_name: &str) -> Tensor {
    let meta_text =
        fs::read_to_string(meta_path).unwrap_or_else(|e| panic!("read {meta_path:?}: {e}"));
    let meta: Value =
        serde_json::from_str(&meta_text).unwrap_or_else(|e| panic!("parse {meta_path:?}: {e}"));

    let dtype_name = meta["dtype"]
        .as_str()
        .unwrap_or_else(|| panic!("{block_name}: meta.dtype is not a string"));
    let dtype = parse_dtype(dtype_name);

    let shape_vals = meta["shape"]
        .as_array()
        .unwrap_or_else(|| panic!("{block_name}: meta.shape is not an array"));
    let shape: Vec<usize> = shape_vals
        .iter()
        .map(|v| {
            v.as_u64()
                .unwrap_or_else(|| panic!("{block_name}: shape entry not u64: {v}"))
                as usize
        })
        .collect();
    let shape = Shape::from(shape);

    let raw = fs::read(bin_path).unwrap_or_else(|e| panic!("read {bin_path:?}: {e}"));

    let device = Device::Cpu;
    match dtype {
        DType::F32 => {
            let n = raw.len() / 4;
            let mut buf = Vec::<f32>::with_capacity(n);
            for chunk in raw.chunks_exact(4) {
                buf.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
            }
            Tensor::from_vec(buf, shape, &device)
                .unwrap_or_else(|e| panic!("{block_name}: build F32 tensor: {e}"))
        }
        DType::F64 => {
            let n = raw.len() / 8;
            let mut buf = Vec::<f64>::with_capacity(n);
            for chunk in raw.chunks_exact(8) {
                let mut arr = [0u8; 8];
                arr.copy_from_slice(chunk);
                buf.push(f64::from_le_bytes(arr));
            }
            Tensor::from_vec(buf, shape, &device)
                .unwrap_or_else(|e| panic!("{block_name}: build F64 tensor: {e}"))
        }
        DType::I64 => {
            let n = raw.len() / 8;
            let mut buf = Vec::<i64>::with_capacity(n);
            for chunk in raw.chunks_exact(8) {
                let mut arr = [0u8; 8];
                arr.copy_from_slice(chunk);
                buf.push(i64::from_le_bytes(arr));
            }
            Tensor::from_vec(buf, shape, &device)
                .unwrap_or_else(|e| panic!("{block_name}: build I64 tensor: {e}"))
        }
        DType::F16 | DType::BF16 => {
            // Build a U8 tensor of the raw bytes and reinterpret -- candle
            // supports building F16/BF16 from raw bytes via from_raw_buffer.
            Tensor::from_raw_buffer(&raw, dtype, shape.dims(), &device)
                .unwrap_or_else(|e| panic!("{block_name}: build {dtype:?} tensor: {e}"))
        }
        DType::U8 => Tensor::from_vec(raw, shape, &device)
            .unwrap_or_else(|e| panic!("{block_name}: build U8 tensor: {e}")),
        other => panic!("{block_name}: unsupported runtime dtype {other:?}"),
    }
}

/// Element-wise comparison. On mismatch, prints max-abs-diff plus the
/// first five offending positions, then panics.
pub fn assert_tensor_close(actual: &Tensor, expected: &Tensor, tol: f32, block: &str) {
    assert_eq!(
        actual.shape(),
        expected.shape(),
        "{block}: shape mismatch (actual {:?} vs expected {:?})",
        actual.shape(),
        expected.shape(),
    );
    assert_eq!(
        actual.dtype(),
        expected.dtype(),
        "{block}: dtype mismatch ({:?} vs {:?})",
        actual.dtype(),
        expected.dtype(),
    );

    let a = actual
        .to_dtype(DType::F32)
        .and_then(|t| t.flatten_all())
        .and_then(|t| t.to_vec1::<f32>())
        .unwrap_or_else(|e| panic!("{block}: collect actual: {e}"));
    let e = expected
        .to_dtype(DType::F32)
        .and_then(|t| t.flatten_all())
        .and_then(|t| t.to_vec1::<f32>())
        .unwrap_or_else(|e| panic!("{block}: collect expected: {e}"));

    assert_eq!(a.len(), e.len(), "{block}: element count mismatch");

    let mut max_diff = 0.0f32;
    let mut offenders: Vec<(usize, f32, f32, f32)> = Vec::new();
    for (i, (av, ev)) in a.iter().zip(e.iter()).enumerate() {
        let d = (av - ev).abs();
        if d > max_diff {
            max_diff = d;
        }
        if d > tol && offenders.len() < 5 {
            offenders.push((i, *av, *ev, d));
        }
    }

    if max_diff > tol {
        let mut msg =
            format!("{block}: tensors differ beyond tol={tol:.3e} (max_abs_diff={max_diff:.6e})\n");
        for (i, av, ev, d) in offenders {
            msg.push_str(&format!(
                "  [{i}] actual={av:+.6e}  expected={ev:+.6e}  abs_diff={d:.3e}\n"
            ));
        }
        panic!("{msg}");
    }
}

// ---------------------------------------------------------------------------
// Self-test: confirms the helper compiles and the missing-dump path panics
// with the documented message. Skipped (returns Ok) when the dump dir is
// actually present -- the model port's tests will exercise the happy path.
// ---------------------------------------------------------------------------

#[test]
fn missing_dump_dir_panics_with_clear_message() {
    if dump_root().exists() {
        // Dumps are already present on this machine -- skip the
        // negative-path check so we don't disturb real artifacts.
        return;
    }
    let result = std::panic::catch_unwind(|| load_dump("definitely_does_not_exist_block_name"));
    let err = result.expect_err("expected panic when dump dir is missing");
    let msg = err
        .downcast_ref::<String>()
        .map(String::as_str)
        .or_else(|| err.downcast_ref::<&'static str>().copied())
        .unwrap_or("");
    assert!(
        msg.contains("stable_audio_dump.py"),
        "panic message should reference the dump script, got: {msg}"
    );
}
