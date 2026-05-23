//! Whisper BPE tokenization (Wave F.2.2) — **no Rust-side implementation
//! required**.
//!
//! The upstream `ct2rs` 0.9 crate already owns the entire Whisper
//! tokenization surface internally. Specifically, [`ct2rs::Whisper::new`]
//! loads `tokenizer.json` from the model directory into a private
//! `hf::Tokenizer` field, and [`ct2rs::Whisper::generate`]:
//!
//! 1. Builds the `<|startoftranscript|> <|lang|> <|transcribe|> [<|notimestamps|>]`
//!    prompt string itself before handing it to the C++ runtime.
//! 2. Runs encoder/decoder inference inside `CTranslate2`.
//! 3. Decodes the resulting token-id sequence back to a `String` via the
//!    private `hf::Tokenizer` and returns `Vec<String>` directly.
//!
//! In other words, no raw token ids ever cross the `ct2rs` API boundary
//! on the Whisper path — the public surface is text-in, text-out. The
//! crate-level [`ct2rs::Tokenizer`] re-export targets the separate
//! `Translator` / `Generator` code path and is **not** the Whisper
//! tokenizer; instantiating it here would be both redundant and wrong
//! (it would load a second copy of the same `tokenizer.json` with no
//! caller).
//!
//! Therefore Wave F.2.2 contributes no public surface. Wave F.2.3
//! (the [`super::decoder`] module wrapping [`ct2rs::Whisper`]) and
//! Wave F.2.4 (the [`super::pipeline`] module) both consume
//! [`ct2rs::Whisper::generate`] directly and rely on its built-in
//! tokenizer; no `super::tokenizer::*` import is needed by either
//! downstream wave.
//!
//! # Future-extension notes
//!
//! If we ever need to expose Whisper-internal token ids to callers —
//! e.g. for **word-level timestamps**, **logit biasing**, **custom
//! suppress-tokens beyond what [`ct2rs::WhisperOptions`] already
//! provides**, or **streaming partial-decode** — we would need to fork
//! `ct2rs::Whisper::generate` to expose the raw `GenerationStepResult`
//! token stream and load a separate `tokenizers::Tokenizer` here for
//! incremental decode. None of those features is in scope for Wave F.2,
//! so this module intentionally stays empty rather than ship dead code.
