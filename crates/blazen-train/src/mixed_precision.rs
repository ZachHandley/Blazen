//! Mixed-precision training helpers. bf16 forward + fp32 master weights.
//!
//! Pattern:
//! 1. Hold a fp32 "master" copy of trainable params (the source of truth).
//! 2. Cast master weights to bf16 for the forward pass each step.
//! 3. Backward computes bf16 grads.
//! 4. Cast grads to fp32 and apply to the master weights via the optimizer.
//!
//! `TrainConfig.mixed_precision = MixedPrecision::None` skips this entirely
//! and trains in fp32; the wrapper is only constructed when bf16 is on.

use candle_core::{DType, Device, Tensor, Var};

use crate::error::BlazenTrainError;

/// A pair of (master fp32 var, working bf16 var). The optimizer updates the
/// master; before each forward, [`Self::sync_to_working`] rebuilds the working
/// view as a fresh bf16 cast of the master.
#[derive(Debug)]
pub struct MixedPrecisionVar {
    /// fp32 source of truth — the optimizer steps on this.
    pub master: Var,
    /// bf16 mirror used in the forward pass.
    pub working: Var,
    device: Device,
}

impl MixedPrecisionVar {
    /// Initialize from an fp32 tensor. The master is built directly from
    /// `initial` (cast to fp32 if needed) and the working copy is the bf16
    /// version of that.
    ///
    /// # Errors
    ///
    /// Returns a `BlazenTrainError::Candle` if the dtype cast, device move,
    /// or `Var` construction fails.
    pub fn new(initial: &Tensor, device: &Device) -> Result<Self, BlazenTrainError> {
        let master_t = initial.to_dtype(DType::F32)?.to_device(device)?;
        let master = Var::from_tensor(&master_t)?;
        let working_t = master_t.to_dtype(DType::BF16)?;
        let working = Var::from_tensor(&working_t)?;
        Ok(Self {
            master,
            working,
            device: device.clone(),
        })
    }

    /// Sync master -> working with bf16 cast. Call before each forward.
    ///
    /// Why: `Var::set` byte-copies storage and rejects cross-dtype writes, so
    /// we rebuild the working `Var` from a fresh cast of the master tensor.
    ///
    /// # Errors
    ///
    /// Returns a `BlazenTrainError::Candle` if the bf16 cast, device move,
    /// or `Var` construction fails.
    pub fn sync_to_working(&mut self) -> Result<(), BlazenTrainError> {
        let bf16 = self.master.as_tensor().to_dtype(DType::BF16)?;
        let bf16 = bf16.to_device(&self.device)?;
        self.working = Var::from_tensor(&bf16)?;
        Ok(())
    }

    /// Sync working_grad -> master_grad with fp32 cast. Call before optimizer
    /// step. The optimizer expects fp32 grads to match the fp32 master.
    ///
    /// # Errors
    ///
    /// Returns a `BlazenTrainError::Candle` if the fp32 cast or device move
    /// fails.
    pub fn upcast_gradient(&self, grad: &Tensor) -> Result<Tensor, BlazenTrainError> {
        let upcast = grad.to_dtype(DType::F32)?.to_device(&self.device)?;
        Ok(upcast)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mixed_precision_var_master_is_fp32_working_is_bf16() {
        let device = Device::Cpu;
        let initial = Tensor::new(&[1.0_f32, 2.0, 3.0], &device).unwrap();
        let mpv = MixedPrecisionVar::new(&initial, &device).unwrap();
        assert_eq!(mpv.master.dtype(), DType::F32);
        assert_eq!(mpv.working.dtype(), DType::BF16);
    }

    #[test]
    fn sync_to_working_preserves_values_within_bf16_precision() {
        let device = Device::Cpu;
        let initial = Tensor::new(&[1.0_f32, 0.5, -0.25, 17.0], &device).unwrap();
        let mut mpv = MixedPrecisionVar::new(&initial, &device).unwrap();

        let updated = Tensor::new(&[2.5_f32, -1.5, 0.125, 32.0], &device).unwrap();
        mpv.master.set(&updated).unwrap();
        mpv.sync_to_working().unwrap();

        let back = mpv
            .working
            .as_tensor()
            .to_dtype(DType::F32)
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        let expected = updated.to_vec1::<f32>().unwrap();
        // Why: bf16 has ~3 decimal digits of mantissa precision; a 0.5% tolerance
        // is safe for values in this range while catching outright sync bugs.
        for (b, e) in back.iter().zip(expected.iter()) {
            let denom = e.abs().max(1.0);
            assert!((b - e).abs() / denom < 5e-3, "back={b} expected={e}");
        }
    }

    #[test]
    fn upcast_gradient_returns_fp32() {
        let device = Device::Cpu;
        let initial = Tensor::new(&[0.0_f32; 4], &device).unwrap();
        let mpv = MixedPrecisionVar::new(&initial, &device).unwrap();

        let bf16_grad = Tensor::new(&[0.5_f32, -0.25, 1.0, 0.125], &device)
            .unwrap()
            .to_dtype(DType::BF16)
            .unwrap();
        let up = mpv.upcast_gradient(&bf16_grad).unwrap();
        assert_eq!(up.dtype(), DType::F32);
        assert_eq!(up.dims(), bf16_grad.dims());

        let v = up.to_vec1::<f32>().unwrap();
        assert!((v[0] - 0.5).abs() < 1e-3);
        assert!((v[1] + 0.25).abs() < 1e-3);
        assert!((v[2] - 1.0).abs() < 1e-3);
        assert!((v[3] - 0.125).abs() < 1e-3);
    }
}
