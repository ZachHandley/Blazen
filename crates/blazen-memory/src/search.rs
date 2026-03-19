//! Similarity computation helpers.
//!
//! These functions normalize raw distance metrics into 0.0..=1.0 similarity
//! scores where 1.0 means identical.

use crate::error::{MemoryError, Result};

// ---------------------------------------------------------------------------
// Text-level SimHash similarity (u64, from elid::simhash)
// ---------------------------------------------------------------------------

/// Compute similarity between two 64-bit text `SimHashes`.
///
/// Uses Hamming distance normalized over 64 bits:
///   similarity = 1.0 - (`hamming_distance` / 64.0)
///
/// Returns a value in 0.0..=1.0.
#[must_use]
pub fn compute_text_simhash_similarity(a: u64, b: u64) -> f64 {
    let distance = (a ^ b).count_ones();
    1.0 - (f64::from(distance) / 64.0)
}

// ---------------------------------------------------------------------------
// Embedding-level SimHash similarity (u128, from elid::embeddings::simhash_128)
// ---------------------------------------------------------------------------

/// Compute similarity between two 128-bit embedding `SimHashes`.
///
/// Uses Hamming distance normalized over 128 bits:
///   similarity = 1.0 - (`hamming_distance` / 128.0)
///
/// Returns a value in 0.0..=1.0.
#[must_use]
pub fn compute_embedding_simhash_similarity(a: u128, b: u128) -> f64 {
    let distance = (a ^ b).count_ones();
    1.0 - (f64::from(distance) / 128.0)
}

// ---------------------------------------------------------------------------
// ELID string similarity (via hamming_distance on ELID payloads)
// ---------------------------------------------------------------------------

/// Compute similarity between two ELID strings.
///
/// Decodes both ELIDs and computes the Hamming distance between their
/// Mini128 payloads, then normalizes to 0.0..=1.0.
///
/// Returns `Err` if either string is not a valid Mini128 ELID.
pub fn compute_elid_similarity(a: &str, b: &str) -> Result<f64> {
    let elid_a =
        elid::embeddings::types::Elid::from_string(a.to_string()).map_err(MemoryError::Elid)?;
    let elid_b =
        elid::embeddings::types::Elid::from_string(b.to_string()).map_err(MemoryError::Elid)?;

    let distance = elid::embeddings::hamming_distance(&elid_a, &elid_b)?;

    // Mini128 uses 128-bit hashes so max distance is 128.
    Ok(1.0 - (f64::from(distance) / 128.0))
}

/// Parse a hex-encoded 128-bit `SimHash` back to u128.
///
/// Returns `None` if the string is not a valid 32-character hex string.
#[must_use]
pub fn simhash_from_hex(hex: &str) -> Option<u128> {
    u128::from_str_radix(hex, 16).ok()
}

/// Encode a u128 `SimHash` as a zero-padded hex string.
#[must_use]
pub fn simhash_to_hex(hash: u128) -> String {
    format!("{hash:032x}")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_text_simhash_similarity_identical() {
        assert!(
            (compute_text_simhash_similarity(0xDEAD_BEEF, 0xDEAD_BEEF) - 1.0).abs() < f64::EPSILON
        );
    }

    #[test]
    fn test_text_simhash_similarity_opposite() {
        let a: u64 = 0;
        let b: u64 = !0;
        assert!((compute_text_simhash_similarity(a, b) - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_text_simhash_similarity_one_bit() {
        let a: u64 = 0;
        let b: u64 = 1;
        let sim = compute_text_simhash_similarity(a, b);
        let expected = 1.0 - (1.0 / 64.0);
        assert!((sim - expected).abs() < f64::EPSILON);
    }

    #[test]
    fn test_embedding_simhash_similarity_identical() {
        let h: u128 = 0xDEAD_BEEF_CAFE_BABE;
        assert!((compute_embedding_simhash_similarity(h, h) - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_embedding_simhash_similarity_opposite() {
        let a: u128 = 0;
        let b: u128 = !0;
        assert!((compute_embedding_simhash_similarity(a, b) - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_simhash_hex_roundtrip() {
        let original: u128 = 0x0123456789ABCDEF_FEDCBA9876543210;
        let hex = simhash_to_hex(original);
        let recovered = simhash_from_hex(&hex).unwrap();
        assert_eq!(original, recovered);
    }

    #[test]
    fn test_simhash_hex_zero_padded() {
        let small: u128 = 1;
        let hex = simhash_to_hex(small);
        assert_eq!(hex.len(), 32, "hex should always be 32 chars");
        assert_eq!(hex, "00000000000000000000000000000001");
    }

    #[test]
    fn test_simhash_from_hex_invalid() {
        assert!(simhash_from_hex("not-a-hex").is_none());
        assert!(simhash_from_hex("").is_none());
    }
}
