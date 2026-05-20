//! Gradient clipping by global L2 norm.
//!
//! Standard recipe used by HuggingFace `transformers` and most fine-tuning
//! workflows: sum the squared L2 norms of every parameter's gradient, take
//! the square root for the global norm, and if it exceeds `max_norm` scale
//! every gradient tensor by `max_norm / global_norm`.
//!
//! Candle's `GradStore` exposes `remove` + `insert`, so we can mutate the
//! grads in place by removing the tensor, scaling it, and re-inserting.

use candle_core::backprop::GradStore;
use candle_core::{Result, Var};

/// Compute the global L2 norm of the gradients across all `vars`.
///
/// Returns `Ok(0.0)` when no grad is present for any var (e.g. a forward
/// pass that didn't touch any trainable param). Missing per-var grads
/// contribute zero. Each grad is upcast to f32 for the norm reduction so
/// bf16/f16 grads don't overflow the accumulator.
///
/// # Errors
///
/// Returns the underlying `candle_core::Error` if a grad tensor cannot be
/// cast to f32 or its squared L2 cannot be reduced (e.g. device failure).
pub fn global_grad_norm(grads: &GradStore, vars: &[&Var]) -> Result<f32> {
    let mut total_sq: f64 = 0.0;
    for var in vars {
        let Some(grad) = grads.get(var) else {
            continue;
        };
        let grad_f32 = grad.to_dtype(candle_core::DType::F32)?;
        let sq_sum = grad_f32.sqr()?.sum_all()?.to_scalar::<f32>()?;
        total_sq += f64::from(sq_sum);
    }
    // Why: total_sq is the sum of squared f32 grads; sqrt fits comfortably
    // back into f32 — overflow only if a single grad already exceeded
    // f32::MAX, which would have failed upstream.
    #[allow(clippy::cast_possible_truncation)]
    let norm = total_sq.sqrt() as f32;
    Ok(norm)
}

/// Scale all per-`var` gradients in `grads` so that the global L2 norm is at
/// most `max_norm`. Returns the pre-clip global norm so callers can log it.
///
/// No-op when the global norm is at or below `max_norm`, or when `max_norm`
/// is non-finite / non-positive (defensive: a misconfigured clip should not
/// silently zero the grads).
///
/// # Errors
///
/// Returns the underlying `candle_core::Error` if the norm computation or
/// per-grad scaling op fails (dtype/device mismatch, OOM, etc.).
pub fn clip_grad_norm(grads: &mut GradStore, vars: &[&Var], max_norm: f32) -> Result<f32> {
    let total = global_grad_norm(grads, vars)?;
    if !max_norm.is_finite() || max_norm <= 0.0 || total <= max_norm || total == 0.0 {
        return Ok(total);
    }
    let scale = f64::from(max_norm) / f64::from(total);
    for var in vars {
        let Some(grad) = grads.remove(var.as_tensor()) else {
            continue;
        };
        let scaled = (grad * scale)?;
        grads.insert(var.as_tensor(), scaled);
    }
    Ok(total)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{Device, Tensor};

    fn make_var(data: &[f32], device: &Device) -> Var {
        Var::from_tensor(&Tensor::new(data, device).unwrap()).unwrap()
    }

    // Why: GradStore::new() is private; the only way to get one is via
    // loss.backward(). We construct a tiny graph whose gradient w.r.t. each
    // var is a controlled tensor, then overwrite via remove+insert when we
    // need exact shapes/values for the assertion.
    fn make_grad_store(var: &Var, target_grad: &Tensor) -> candle_core::backprop::GradStore {
        let loss = (var.as_tensor() * 1.0).unwrap().sum_all().unwrap();
        let mut store = loss.backward().unwrap();
        store.remove(var.as_tensor());
        store.insert(var.as_tensor(), target_grad.clone());
        store
    }

    #[test]
    fn global_grad_norm_with_no_grads_returns_zero() {
        let device = Device::Cpu;
        let v = make_var(&[1.0_f32, 2.0, 3.0], &device);
        let untouched = make_var(&[0.0_f32; 3], &device);
        let g = Tensor::new(&[0.0_f32, 0.0, 0.0], &device).unwrap();
        let store = make_grad_store(&v, &g);
        let norm = global_grad_norm(&store, &[&untouched]).unwrap();
        assert!((norm - 0.0).abs() < 1e-6);
    }

    #[test]
    fn clip_grad_norm_no_op_when_under_threshold() {
        let device = Device::Cpu;
        let v = make_var(&[0.0_f32; 3], &device);
        let g = Tensor::new(&[3.0_f32, 0.0, 4.0], &device).unwrap();
        let mut store = make_grad_store(&v, &g);

        let pre = clip_grad_norm(&mut store, &[&v], 10.0).unwrap();
        assert!((pre - 5.0).abs() < 1e-5);

        let after = store.get(v.as_tensor()).unwrap();
        let after_vec = after.to_vec1::<f32>().unwrap();
        assert!((after_vec[0] - 3.0).abs() < 1e-6);
        assert!((after_vec[1] - 0.0).abs() < 1e-6);
        assert!((after_vec[2] - 4.0).abs() < 1e-6);
    }

    #[test]
    fn clip_grad_norm_scales_when_over_threshold() {
        let device = Device::Cpu;
        let var_a = make_var(&[0.0_f32; 2], &device);
        let var_b = make_var(&[0.0_f32; 2], &device);
        let g_a = Tensor::new(&[3.0_f32, 0.0], &device).unwrap();
        let g_b = Tensor::new(&[0.0_f32, 4.0], &device).unwrap();
        let mut store = make_grad_store(&var_a, &g_a);
        store.remove(var_b.as_tensor());
        store.insert(var_b.as_tensor(), g_b);

        let max = 2.5_f32;
        let pre = clip_grad_norm(&mut store, &[&var_a, &var_b], max).unwrap();
        assert!((pre - 5.0).abs() < 1e-5);

        let post = global_grad_norm(&store, &[&var_a, &var_b]).unwrap();
        assert!((post - max).abs() < 1e-5);

        let out_a = store
            .get(var_a.as_tensor())
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        let out_b = store
            .get(var_b.as_tensor())
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        let expected_scale = max / 5.0;
        assert!((out_a[0] - 3.0 * expected_scale).abs() < 1e-5);
        assert!((out_b[1] - 4.0 * expected_scale).abs() < 1e-5);
    }
}
