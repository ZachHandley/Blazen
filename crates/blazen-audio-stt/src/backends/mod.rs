//! Concrete [`SttBackend`](crate::SttBackend) implementations.
//!
//! Each backend is gated behind its own Cargo feature so the workspace
//! `cargo check` stays fast and downstream binaries only pay for the
//! engines they actually use.
//!
//! | Module                      | Feature       | Status   |
//! |-----------------------------|---------------|----------|
//! | [`whispercpp`]              | `whispercpp`  | active   |
//! | [`candle`]                  | `candle`      | active   |

#[cfg(feature = "candle")]
pub mod candle;

#[cfg(feature = "whispercpp")]
pub mod whispercpp;
