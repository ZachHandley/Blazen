//! Magic-number-based content type detection.
//!
//! Provides a small public API for classifying bytes or files into a
//! [`ContentKind`] plus optional MIME hint. The implementation prefers the
//! `infer` crate (gated behind the `content-detect` feature) for magic-byte
//! sniffing, and falls back to file-extension hints when bytes don't match
//! any known signature or when the feature is disabled.
//!
//! Detection order:
//! 1. Magic-number sniff via `infer` (covers most images, audio, video,
//!    archives, fonts, common documents).
//! 2. Filename-extension hint (covers 3D / CAD / code / data formats that
//!    `infer` does not recognize, plus fallback for ambiguous bytes).
//! 3. MIME hint, if the caller provided one.
//! 4. [`ContentKind::Other`].

use std::path::Path;

use super::kind::ContentKind;

/// Detect [`ContentKind`] and optional MIME from a byte slice.
///
/// Returns `(kind, Some(mime))` when the magic-number signature matched,
/// otherwise `(ContentKind::Other, None)`.
#[must_use]
pub fn detect_from_bytes(bytes: &[u8]) -> (ContentKind, Option<String>) {
    detect_from_bytes_impl(bytes)
}

#[cfg(feature = "content-detect")]
fn detect_from_bytes_impl(bytes: &[u8]) -> (ContentKind, Option<String>) {
    if bytes.is_empty() {
        return (ContentKind::Other, None);
    }
    if let Some(kind) = infer::get(bytes) {
        let mime = kind.mime_type().to_owned();
        let category = ContentKind::from_mime(&mime);
        return (category, Some(mime));
    }
    (ContentKind::Other, None)
}

#[cfg(not(feature = "content-detect"))]
fn detect_from_bytes_impl(_bytes: &[u8]) -> (ContentKind, Option<String>) {
    (ContentKind::Other, None)
}

/// Detect [`ContentKind`] and optional MIME from a filesystem path.
///
/// Reads up to 8 KiB from the front of the file for the magic-number sniff,
/// then falls back to the file's extension. Returns `(ContentKind::Other,
/// None)` if the file cannot be opened.
#[cfg(not(target_arch = "wasm32"))]
#[must_use]
pub fn detect_from_path(path: &Path) -> (ContentKind, Option<String>) {
    use std::fs::File;
    use std::io::Read;

    // Try magic-byte sniff first.
    let mut header = [0u8; 8 * 1024];
    let read_len = File::open(path)
        .and_then(|mut f| f.read(&mut header))
        .ok()
        .unwrap_or(0);
    if read_len > 0 {
        let (kind, mime) = detect_from_bytes(&header[..read_len]);
        if kind != ContentKind::Other {
            return (kind, mime);
        }
    }

    // Fall back to extension.
    if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
        let kind = ContentKind::from_extension(ext);
        if kind != ContentKind::Other {
            return (kind, None);
        }
    }
    (ContentKind::Other, None)
}

/// Detect [`ContentKind`] and optional MIME using a chain of signals: an
/// explicit MIME hint, then the bytes' magic numbers, then the filename's
/// extension. The first signal that resolves to a non-`Other` kind wins.
///
/// Use this when you have several weak signals (e.g. an HTTP response body
/// with a `Content-Type` header and a filename) and want a single best-effort
/// classification.
#[must_use]
pub fn detect(
    bytes: Option<&[u8]>,
    mime_hint: Option<&str>,
    filename: Option<&str>,
) -> (ContentKind, Option<String>) {
    if let Some(mime) = mime_hint {
        let kind = ContentKind::from_mime(mime);
        if kind != ContentKind::Other {
            return (kind, Some(mime.to_owned()));
        }
    }
    if let Some(bytes) = bytes {
        let (kind, mime) = detect_from_bytes(bytes);
        if kind != ContentKind::Other {
            return (kind, mime);
        }
    }
    if let Some(name) = filename
        && let Some(ext) = Path::new(name).extension().and_then(|e| e.to_str())
    {
        let kind = ContentKind::from_extension(ext);
        if kind != ContentKind::Other {
            return (kind, mime_hint.map(str::to_owned));
        }
    }
    (ContentKind::Other, mime_hint.map(str::to_owned))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_bytes_are_other() {
        let (k, m) = detect_from_bytes(&[]);
        assert_eq!(k, ContentKind::Other);
        assert!(m.is_none());
    }

    #[cfg(feature = "content-detect")]
    #[test]
    fn detect_png_signature() {
        // 8-byte PNG header + minimal IHDR-like data so infer can recognize it.
        let png = [
            0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A, 0x00, 0x00, 0x00, 0x0D,
        ];
        let (k, m) = detect_from_bytes(&png);
        assert_eq!(k, ContentKind::Image);
        assert_eq!(m.as_deref(), Some("image/png"));
    }

    #[cfg(feature = "content-detect")]
    #[test]
    fn detect_pdf_signature() {
        let pdf = b"%PDF-1.7\n%\xE2\xE3\xCF\xD3\n";
        let (k, _m) = detect_from_bytes(pdf);
        assert_eq!(k, ContentKind::Document);
    }

    #[test]
    fn mime_hint_overrides_other_for_unknown_bytes() {
        let (k, m) = detect(Some(&[0u8; 4]), Some("image/png"), None);
        assert_eq!(k, ContentKind::Image);
        assert_eq!(m.as_deref(), Some("image/png"));
    }

    #[test]
    fn extension_hint_resolves_3d_when_bytes_unknown() {
        // Bytes won't sniff to anything; .glb extension drives 3D classification.
        let (k, _m) = detect(Some(&[0u8; 4]), None, Some("model.glb"));
        assert_eq!(k, ContentKind::ThreeDModel);
    }

    #[test]
    fn extension_for_cad() {
        let (k, _m) = detect(None, None, Some("part.step"));
        assert_eq!(k, ContentKind::Cad);
    }

    #[test]
    fn no_signals_yield_other() {
        let (k, _m) = detect(None, None, None);
        assert_eq!(k, ContentKind::Other);
    }
}
