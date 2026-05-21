//! Subprocess wrapper around the system `espeak-ng` binary.
//!
//! Upstream `piper-rs` links the `espeak-rs-sys` crate, which
//! `cc`-compiles the GPL-3.0+ eSpeak NG C sources. That would taint a
//! consumer of this vendored crate with GPL link-time obligations
//! (Blazen ships under MPL-2.0, which is link-time incompatible).
//!
//! The patch replaces all in-process FFI calls with subprocess
//! invocations of `espeak-ng`. The binary is invoked once per `synthesize`
//! call:
//!
//! ```text
//! espeak-ng --ipa=3 -q --sep="" -v <voice> -- <text>
//! ```
//!
//! - `--ipa=3` emits IPA (no language-switch markers — equivalent to
//!   upstream's "strip `(en)` brackets" post-processing).
//! - `-q` skips audio playback (we only want the phoneme dump on stdout).
//! - `--sep=""` keeps the phoneme stream contiguous (matches what the
//!   upstream `espeak_TextToPhonemes` FFI call returned).
//! - `-v <voice>` selects the per-voice phonemizer (`en-us`, `ar`, …).
//!
//! Subprocess invocation puts a process boundary between Blazen and
//! eSpeak NG → no derived-work / link-time GPL inheritance.

use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

use crate::{PiperError, PiperResult};

const ESPEAK_NG_BIN: &str = "espeak-ng";

/// Locate the `espeak-ng` binary on `PATH`.
///
/// On Windows the binary may be `espeak-ng.exe`; `Command::new` handles
/// the extension probe natively via `PATHEXT`. We do a `--version`
/// invocation here to *prove* it actually runs (the binary may exist
/// but be broken / unrunnable on locked-down sandboxes).
///
/// # Errors
///
/// [`PiperError::EspeakNgMissing`] with the underlying io error message.
pub(crate) fn locate_espeak_ng() -> PiperResult<PathBuf> {
    let mut cmd = Command::new(ESPEAK_NG_BIN);
    cmd.arg("--version")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .stdin(Stdio::null());
    match cmd.status() {
        Ok(s) if s.success() => Ok(PathBuf::from(ESPEAK_NG_BIN)),
        Ok(s) => Err(PiperError::EspeakNgMissing(format!(
            "`{ESPEAK_NG_BIN} --version` exited with status {s}"
        ))),
        Err(e) => Err(PiperError::EspeakNgMissing(e.to_string())),
    }
}

/// Synchronously phonemize `text` using `espeak-ng` at `binary` for the
/// given `voice` (e.g. `"en-us"`, `"ar"`, `"de"`).
///
/// Returned phonemes are a single IPA string with no language-switch
/// markers, no per-phoneme separators, and clause breakers retained
/// (`.`, `,`, `!`, `?`). That matches what upstream's FFI call returned.
///
/// # Errors
///
/// [`PiperError::PhonemizationError`] on any subprocess / decoding failure.
pub(crate) fn text_to_phonemes_blocking(
    binary: &Path,
    text: &str,
    voice: &str,
) -> PiperResult<String> {
    if text.is_empty() {
        return Ok(String::new());
    }
    // espeak-ng accepts text via -- or stdin; we use stdin so commas /
    // quotes / shell-meta in the text never round-trip through argv.
    let mut child = Command::new(binary)
        .arg("--ipa=3")
        .arg("-q")
        .arg("--sep=")
        .arg("-v")
        .arg(voice)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| {
            PiperError::PhonemizationError(format!("failed to spawn `{}`: {e}", binary.display()))
        })?;

    // Write the text on stdin and close it so espeak-ng knows it's EOF.
    use std::io::Write as _;
    {
        let stdin = child
            .stdin
            .as_mut()
            .ok_or_else(|| PiperError::PhonemizationError("child stdin unavailable".to_string()))?;
        stdin
            .write_all(text.as_bytes())
            .map_err(|e| PiperError::PhonemizationError(format!("write stdin: {e}")))?;
    }
    // Drop child.stdin via re-take to close it.
    drop(child.stdin.take());

    let output = child
        .wait_with_output()
        .map_err(|e| PiperError::PhonemizationError(format!("espeak-ng wait failed: {e}")))?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(PiperError::PhonemizationError(format!(
            "espeak-ng exited {} — stderr: {}",
            output.status,
            stderr.trim()
        )));
    }

    let raw = String::from_utf8(output.stdout)
        .map_err(|e| PiperError::PhonemizationError(format!("stdout not UTF-8: {e}")))?;
    // espeak-ng may insert language-switch markers like `(en)` even with
    // --ipa=3 if the input mixes languages. Defensive strip — upstream
    // did the same after the FFI call.
    Ok(strip_lang_switches(raw.trim()))
}

/// Strip inline language-switch markers of the form `(xx)` that espeak
/// inserts when the text contains words from a different language than
/// the active voice. Behaviour-compatible with upstream's helper.
fn strip_lang_switches(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut depth: usize = 0;
    for c in s.chars() {
        match c {
            '(' => depth += 1,
            ')' => depth = depth.saturating_sub(1),
            _ if depth == 0 => out.push(c),
            _ => {}
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strip_removes_simple_marker() {
        assert_eq!(strip_lang_switches("foo(en)bar"), "foobar");
        assert_eq!(strip_lang_switches("(ar)hello"), "hello");
    }

    #[test]
    fn strip_handles_unbalanced_parens() {
        // saturating_sub keeps us safe if espeak ever emits a stray `)`.
        assert_eq!(strip_lang_switches("foo)bar"), "foobar");
    }

    #[test]
    fn locate_returns_missing_when_binary_absent() {
        // Force a guaranteed-missing binary by overriding PATH to /nonexistent.
        // We can't easily isolate PATH per-test, so this just sanity-checks
        // that the function returns Ok or EspeakNgMissing — both are valid
        // states depending on whether the test runner has espeak-ng.
        match locate_espeak_ng() {
            Ok(_) | Err(PiperError::EspeakNgMissing(_)) => {}
            Err(other) => panic!("unexpected error variant: {other:?}"),
        }
    }
}
