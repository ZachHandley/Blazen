//! Ring-AllReduce kernel.
//!
//! Implements the classic Baidu / NCCL ring-AllReduce algorithm for a
//! single dense tensor (typically a parameter's gradient). See the
//! [module docs](super) for the algorithmic walkthrough.
//!
//! The tensor is flattened to f32, chunked into `world_size` slices, and
//! the reduce-scatter + all-gather phases pass each chunk around the
//! ring once. After the algorithm completes, every worker holds the
//! same fully-summed tensor.
//!
//! The kernel is device-agnostic — chunks are materialized on the host
//! via `Tensor::to_vec1::<f32>`, summed in f32, and copied back via
//! `Tensor::from_slice`. A future phase can specialize this to GPU-
//! resident NCCL collectives where available; the trait boundary stays
//! the same.

use candle_core::{DType, Device, Tensor};

use crate::error::BlazenTrainError;

use super::ring::RingTopology;
use super::transport::RingTransport;

/// Run a ring-AllReduce on `grad`, replacing it in-place with the
/// element-wise sum across every worker in the ring.
///
/// `tag` is mixed into the `(step, chunk_id)` keys exchanged on the
/// transport so multiple AllReduce calls within the same optimizer step
/// (one per parameter) don't collide.
///
/// # Errors
/// - [`BlazenTrainError::Distributed`] for transport or shape failures.
/// - [`BlazenTrainError::Candle`] for tensor op failures.
pub async fn ring_all_reduce(
    grad: &mut Tensor,
    topology: &RingTopology,
    transport: &dyn RingTransport,
    tag: u32,
) -> Result<(), BlazenTrainError> {
    if topology.world_size <= 1 {
        return Ok(());
    }

    let orig_shape = grad.dims().to_vec();
    let orig_dtype = grad.dtype();
    let device = grad.device().clone();
    let numel: usize = orig_shape.iter().product();

    // Flatten to a 1-D f32 buffer on the host. f32 keeps the math
    // identical regardless of the source dtype (bf16/f16/f32 grads all
    // accumulate in f32, matching the convention used by `grad_clip`).
    let flat = grad
        .to_dtype(DType::F32)?
        .flatten_all()?
        .to_vec1::<f32>()
        .map_err(|e| BlazenTrainError::Distributed(format!("grad to_vec1 failed: {e}")))?;

    // Pad to the next multiple of world_size so every chunk has equal
    // length. The pad slice is sliced off again at the end.
    let world = topology.world_size;
    let padded_len = numel.div_ceil(world) * world;
    let mut buf = flat;
    buf.resize(padded_len, 0.0);
    let chunk_len = padded_len / world;

    // -----------------------------------------------------------------
    // Reduce-scatter: N-1 rounds. At round s, worker i sends chunk
    // (i - s) mod N to next and receives chunk (i - s - 1) mod N from
    // prev, accumulating into its local copy.
    // -----------------------------------------------------------------
    for s in 0..(world - 1) {
        let send_chunk_idx = (topology.rank + world - s) % world;
        let recv_chunk_idx = (topology.rank + world - s - 1) % world;

        let send_payload = chunk_bytes(&buf, send_chunk_idx, chunk_len);
        // Encode (tag, phase, step) into chunk_id so the receiver's
        // (step, chunk_id) key is unique within the optimizer step.
        // phase 0 = reduce-scatter, phase 1 = all-gather.
        let send_step = encode_step(tag, 0, u32::try_from(s).unwrap_or(0));
        let chunk_id = u32::try_from(send_chunk_idx).unwrap_or(0);
        transport.send(send_step, chunk_id, send_payload).await?;

        let recv_step = encode_step(tag, 0, u32::try_from(s).unwrap_or(0));
        let recv_chunk_id = u32::try_from(recv_chunk_idx).unwrap_or(0);
        let recv_payload = transport.recv(recv_step, recv_chunk_id).await?;
        let recv_floats = bytes_to_floats(&recv_payload, chunk_len)?;

        let base = recv_chunk_idx * chunk_len;
        for j in 0..chunk_len {
            buf[base + j] += recv_floats[j];
        }
    }

    // -----------------------------------------------------------------
    // All-gather: N-1 rounds. At round s, worker i sends chunk
    // (i + 1 - s) mod N to next and receives chunk (i - s) mod N from
    // prev, overwriting its local copy.
    // -----------------------------------------------------------------
    for s in 0..(world - 1) {
        let send_chunk_idx = (topology.rank + world + 1 - s) % world;
        let recv_chunk_idx = (topology.rank + world - s) % world;

        let send_payload = chunk_bytes(&buf, send_chunk_idx, chunk_len);
        let send_step = encode_step(tag, 1, u32::try_from(s).unwrap_or(0));
        let chunk_id = u32::try_from(send_chunk_idx).unwrap_or(0);
        transport.send(send_step, chunk_id, send_payload).await?;

        let recv_step = encode_step(tag, 1, u32::try_from(s).unwrap_or(0));
        let recv_chunk_id = u32::try_from(recv_chunk_idx).unwrap_or(0);
        let recv_payload = transport.recv(recv_step, recv_chunk_id).await?;
        let recv_floats = bytes_to_floats(&recv_payload, chunk_len)?;

        let base = recv_chunk_idx * chunk_len;
        buf[base..base + chunk_len].copy_from_slice(&recv_floats);
    }

    // Trim the pad, rebuild the tensor on the original device + dtype.
    buf.truncate(numel);
    let reduced = Tensor::from_slice(&buf, (numel,), &device)?
        .reshape(orig_shape.as_slice())?
        .to_dtype(orig_dtype)?;
    *grad = reduced;
    Ok(())
}

/// Average a tensor in-place by `world_size` (used after AllReduce-sum
/// to recover the mean gradient that matches a global-batch baseline).
///
/// # Errors
/// Forwards candle errors from the affine multiply.
pub fn average_by(grad: &mut Tensor, world_size: usize) -> Result<(), BlazenTrainError> {
    if world_size <= 1 {
        return Ok(());
    }
    // world_size is the number of workers in the ring — realistic values
    // are 2..8192, well within f64 mantissa precision.
    #[allow(clippy::cast_precision_loss)]
    let scale = 1.0_f64 / world_size as f64;
    *grad = grad.affine(scale, 0.0)?;
    Ok(())
}

fn chunk_bytes(buf: &[f32], chunk_idx: usize, chunk_len: usize) -> Vec<u8> {
    let base = chunk_idx * chunk_len;
    let slice = &buf[base..base + chunk_len];
    let mut out = Vec::with_capacity(chunk_len * 4);
    for f in slice {
        out.extend_from_slice(&f.to_le_bytes());
    }
    out
}

fn bytes_to_floats(bytes: &[u8], expected_len: usize) -> Result<Vec<f32>, BlazenTrainError> {
    if bytes.len() != expected_len * 4 {
        return Err(BlazenTrainError::Distributed(format!(
            "chunk size mismatch: got {} bytes, expected {} ({}× f32)",
            bytes.len(),
            expected_len * 4,
            expected_len,
        )));
    }
    let mut out = Vec::with_capacity(expected_len);
    for i in 0..expected_len {
        let off = i * 4;
        let arr = [bytes[off], bytes[off + 1], bytes[off + 2], bytes[off + 3]];
        out.push(f32::from_le_bytes(arr));
    }
    Ok(out)
}

/// Pack `(tag, phase, round)` into a single u32 step key so multiple
/// concurrent AllReduce calls (one per param within an optimizer step)
/// can share one transport without their (step, chunk_id) keys colliding.
///
/// Layout: top 16 bits = tag, next 4 bits = phase (0 or 1), low 12 bits
/// = round. Caps: 65536 distinct tags, 4096 rounds per phase — well past
/// any realistic gradient count or ring size.
fn encode_step(tag: u32, phase: u32, round: u32) -> u32 {
    ((tag & 0xFFFF) << 16) | ((phase & 0xF) << 12) | (round & 0xFFF)
}

/// Build a `Tensor` of shape `[len]` on CPU from a flat f32 slice.
/// Helper for tests so we don't pull `candle_core` into every test
/// module that consumes the AllReduce kernel.
///
/// # Errors
/// Forwards candle errors.
pub fn tensor_from_flat(data: &[f32], device: &Device) -> Result<Tensor, BlazenTrainError> {
    Ok(Tensor::from_slice(data, (data.len(),), device)?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::distributed::ring::RingTopology;
    use crate::distributed::transport::InMemoryRingTransport;
    use std::sync::Arc;

    fn make_topology(rank: usize, world: usize) -> RingTopology {
        let peers: Vec<String> = (0..world).map(|i| format!("rank{i}:1")).collect();
        RingTopology::new(rank, peers).expect("topology")
    }

    async fn run_allreduce_world(grads: Vec<Vec<f32>>) -> Vec<Vec<f32>> {
        let world = grads.len();
        let ring = InMemoryRingTransport::ring(world);
        let device = Device::Cpu;

        let mut handles = Vec::with_capacity(world);
        for (rank, g) in grads.into_iter().enumerate() {
            let transport: Arc<InMemoryRingTransport> = ring[rank].clone();
            let topo = make_topology(rank, world);
            let device = device.clone();
            handles.push(tokio::spawn(async move {
                let mut t = tensor_from_flat(&g, &device).expect("tensor");
                ring_all_reduce(&mut t, &topo, transport.as_ref(), 0)
                    .await
                    .expect("allreduce");
                t.to_vec1::<f32>().expect("to_vec1")
            }));
        }
        let mut out = Vec::with_capacity(world);
        for h in handles {
            out.push(h.await.expect("join"));
        }
        out
    }

    #[tokio::test]
    async fn ring_all_reduce_world_2_matches_sum() {
        let grads = vec![vec![1.0_f32, 2.0, 3.0, 4.0], vec![5.0_f32, 6.0, 7.0, 8.0]];
        let out = run_allreduce_world(grads).await;
        let expected = vec![6.0_f32, 8.0, 10.0, 12.0];
        for got in &out {
            assert_eq!(got, &expected);
        }
    }

    #[tokio::test]
    async fn ring_all_reduce_world_4_matches_sum() {
        // 4 workers, each with a length-8 gradient. Expected sum is
        // element-wise [10, 14, 18, 22, 26, 30, 34, 38] (chosen so each
        // worker contributes 1+2+3+4 = 10 to the first cell, etc.).
        let grads = vec![
            vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            vec![2.0_f32, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
            vec![3.0_f32, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
            vec![4.0_f32, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0],
        ];
        let out = run_allreduce_world(grads).await;
        let expected: Vec<f32> = (0..8_u32)
            .map(|i| {
                // safe: i is small enough to fit f32 exactly
                #[allow(clippy::cast_precision_loss)]
                let base = (i + 1) as f32;
                base + (base + 1.0) + (base + 2.0) + (base + 3.0)
            })
            .collect();
        for got in &out {
            assert_eq!(got, &expected);
        }
    }

    #[tokio::test]
    async fn ring_all_reduce_average_divides_by_world_size() {
        let grads = vec![vec![2.0_f32, 4.0, 6.0, 8.0], vec![4.0_f32, 8.0, 12.0, 16.0]];
        let world = grads.len();
        let mut out = run_allreduce_world(grads).await;
        for got in &mut out {
            // Apply average separately to confirm the helper matches
            // dividing by world_size.
            let device = Device::Cpu;
            let mut t = tensor_from_flat(got, &device).expect("tensor");
            average_by(&mut t, world).expect("avg");
            *got = t.to_vec1::<f32>().expect("to_vec1");
        }
        let expected = vec![3.0_f32, 6.0, 9.0, 12.0];
        for got in &out {
            assert_eq!(got, &expected);
        }
    }

    #[tokio::test]
    async fn ring_all_reduce_handles_uneven_chunk_size() {
        // length-5 gradient with world_size=3 → padded to 6, last cell
        // dropped on the way out. The pad zeroes contribute nothing to
        // the sum so the visible result still matches element-wise sum.
        let grads = vec![
            vec![1.0_f32, 2.0, 3.0, 4.0, 5.0],
            vec![10.0_f32, 20.0, 30.0, 40.0, 50.0],
            vec![100.0_f32, 200.0, 300.0, 400.0, 500.0],
        ];
        let out = run_allreduce_world(grads).await;
        let expected = vec![111.0_f32, 222.0, 333.0, 444.0, 555.0];
        for got in &out {
            assert_eq!(got.len(), 5);
            assert_eq!(got, &expected);
        }
    }

    #[tokio::test]
    async fn ring_all_reduce_world_1_is_noop() {
        let device = Device::Cpu;
        let mut t = tensor_from_flat(&[1.0, 2.0, 3.0], &device).expect("tensor");
        let topo = make_topology(0, 1);
        let ring = InMemoryRingTransport::ring(1);
        ring_all_reduce(&mut t, &topo, ring[0].as_ref(), 0)
            .await
            .expect("noop");
        assert_eq!(t.to_vec1::<f32>().expect("to_vec1"), vec![1.0, 2.0, 3.0]);
    }

    #[tokio::test]
    async fn ring_all_reduce_world_3_preserves_dtype_f32() {
        let grads = vec![
            vec![1.5_f32, -2.0, 3.25],
            vec![0.5_f32, 1.0, -1.25],
            vec![-1.0_f32, 2.0, 4.0],
        ];
        let out = run_allreduce_world(grads).await;
        let expected = [1.0_f32, 1.0, 6.0];
        for got in &out {
            for (a, b) in got.iter().zip(expected.iter()) {
                assert!((a - b).abs() < 1e-5, "got {a}, want {b}");
            }
        }
    }
}
