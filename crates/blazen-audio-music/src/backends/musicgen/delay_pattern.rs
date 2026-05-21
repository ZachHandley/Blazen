//! MusicGen's "delay pattern" token interleaver.
//!
//! MusicGen's EnCodec backbone produces 4 parallel codebook streams per
//! audio frame. The decoder must emit one frame per autoregressive step,
//! but EnCodec's codebooks are *causal in time per stream*: codebook 1
//! depends on codebook 0, codebook 2 on codebooks 0+1, etc. To preserve
//! that ordering at generation time, the published Meta audiocraft
//! implementation uses a *delay pattern*:
//!
//! * Codebook 0 is generated at every step starting at step 0.
//! * Codebook 1 is delayed by 1 step (emits a BOS at step 0, then real
//!   tokens from step 1 onward).
//! * Codebook 2 is delayed by 2 steps.
//! * Codebook 3 is delayed by 3 steps.
//!
//! For a target of `T` audio frames we therefore run the decoder for
//! `T + (num_codebooks - 1)` autoregressive steps. After generation we
//! *undelay* the matrix to recover the actual EnCodec code grid.
//!
//! This module owns two pure functions over `Vec<Vec<u32>>` so they can
//! be unit-tested without a model in the loop.

/// Build the delay-pattern *prefix* matrix that initialises the
/// generation buffer: shape `[num_codebooks][num_codebooks]` where
/// `buf[cb][t] = bos` if `t < cb`, else "unfilled" (callers populate the
/// rest during the autoregressive loop). The buffer returned here is
/// just the BOS prefix.
///
/// Example for `num_codebooks=4`, `bos=2048`:
///
/// ```text
/// cb0: [ , , , ]      // no BOS prefix
/// cb1: [2048, , , ]
/// cb2: [2048, 2048, , ]
/// cb3: [2048, 2048, 2048, ]
/// ```
///
/// The returned `Vec<Vec<u32>>` has exactly the BOS entries (length `cb`
/// per codebook `cb`), so callers can `push` real tokens onto each row
/// during generation.
#[must_use]
pub fn bos_prefix(num_codebooks: usize, bos: u32) -> Vec<Vec<u32>> {
    (0..num_codebooks).map(|cb| vec![bos; cb]).collect()
}

/// Per-step "should I sample this codebook yet?" predicate.
///
/// At absolute step `step` (0-indexed), codebook `cb` is *active* iff
/// `step >= cb`. Before that the delay-pattern keeps the codebook
/// frozen at BOS.
#[must_use]
pub const fn codebook_active(step: usize, cb: usize) -> bool {
    step >= cb
}

/// Undelay the per-codebook token streams into a flat EnCodec code grid.
///
/// `delayed[cb]` has length `T + num_codebooks - 1` where the first `cb`
/// entries are BOS prefix tokens. We strip those, then truncate every
/// stream to the common length `T` (the shortest *real* stream) so the
/// EnCodec decoder sees `[num_codebooks, T]` rectangular codes.
///
/// Returns `(num_codebooks, T)` as a row-major `Vec<u32>` plus the
/// effective `T`.
///
/// # Errors
///
/// Returns `None` if any stream is shorter than its codebook's BOS prefix
/// (i.e. generation aborted before that codebook activated).
#[must_use]
pub fn undelay(delayed: &[Vec<u32>]) -> Option<(Vec<u32>, usize)> {
    let num_codebooks = delayed.len();
    if num_codebooks == 0 {
        return Some((Vec::new(), 0));
    }
    // Per-codebook stripped lengths.
    let mut stripped_lens = Vec::with_capacity(num_codebooks);
    for (cb, row) in delayed.iter().enumerate() {
        if row.len() < cb {
            return None;
        }
        stripped_lens.push(row.len() - cb);
    }
    let t = stripped_lens.iter().copied().min().unwrap_or(0);
    let mut flat = Vec::with_capacity(num_codebooks * t);
    for (cb, row) in delayed.iter().enumerate() {
        // Skip the `cb` BOS prefix tokens, take `t` real tokens.
        flat.extend(row.iter().skip(cb).take(t).copied());
    }
    Some((flat, t))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bos_prefix_has_triangular_shape() {
        let p = bos_prefix(4, 2048);
        assert_eq!(p.len(), 4);
        assert_eq!(p[0].len(), 0);
        assert_eq!(p[1], vec![2048]);
        assert_eq!(p[2], vec![2048, 2048]);
        assert_eq!(p[3], vec![2048, 2048, 2048]);
    }

    #[test]
    fn codebook_active_obeys_delay() {
        // step 0: only cb0 is active.
        assert!(codebook_active(0, 0));
        assert!(!codebook_active(0, 1));
        // step 3: cb0..cb3 all active.
        for cb in 0..4 {
            assert!(codebook_active(3, cb), "step=3 cb={cb}");
        }
        // step 2: cb0, cb1, cb2 active; cb3 not.
        assert!(codebook_active(2, 0));
        assert!(codebook_active(2, 1));
        assert!(codebook_active(2, 2));
        assert!(!codebook_active(2, 3));
    }

    #[test]
    fn undelay_strips_prefix_and_truncates() {
        // Simulated 4-codebook output for T=3 real frames. The delayed
        // matrix runs for T + (num_codebooks - 1) = 6 steps total.
        // After undelay each row should be length 3.
        let delayed = vec![
            // cb0: 6 real tokens (no prefix)
            vec![10, 11, 12, 13, 14, 15],
            // cb1: 1 BOS + 5 real
            vec![2048, 20, 21, 22, 23, 24],
            // cb2: 2 BOS + 4 real
            vec![2048, 2048, 30, 31, 32, 33],
            // cb3: 3 BOS + 3 real
            vec![2048, 2048, 2048, 40, 41, 42],
        ];
        let (flat, t) = undelay(&delayed).expect("undelay should succeed");
        assert_eq!(t, 3, "common stripped length is 3");
        assert_eq!(flat.len(), 4 * 3);
        // Row-major: cb0[0..3], cb1[0..3], cb2[0..3], cb3[0..3].
        assert_eq!(&flat[0..3], &[10, 11, 12]);
        assert_eq!(&flat[3..6], &[20, 21, 22]);
        assert_eq!(&flat[6..9], &[30, 31, 32]);
        assert_eq!(&flat[9..12], &[40, 41, 42]);
    }

    #[test]
    fn undelay_handles_empty_input() {
        let (flat, t) = undelay(&[]).expect("empty input is valid");
        assert!(flat.is_empty());
        assert_eq!(t, 0);
    }

    #[test]
    fn undelay_rejects_short_prefix() {
        // cb3 needs 3 BOS prefix tokens; supply only 2.
        let delayed = vec![
            vec![1, 2, 3],
            vec![2048, 4, 5],
            vec![2048, 2048, 6],
            vec![2048, 2048],
        ];
        assert!(undelay(&delayed).is_none());
    }

    #[test]
    fn undelay_truncates_to_shortest_real_stream() {
        // cb1 only got 2 real tokens; cb0 got 4. After undelay common T=2.
        let delayed = vec![
            vec![10, 11, 12, 13],
            vec![2048, 20, 21],
            vec![2048, 2048, 30, 31, 32],
            vec![2048, 2048, 2048, 40, 41],
        ];
        let (flat, t) = undelay(&delayed).expect("undelay");
        assert_eq!(t, 2);
        assert_eq!(&flat[0..2], &[10, 11]);
        assert_eq!(&flat[2..4], &[20, 21]);
        assert_eq!(&flat[4..6], &[30, 31]);
        assert_eq!(&flat[6..8], &[40, 41]);
    }
}
