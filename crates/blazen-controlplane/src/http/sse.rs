//! Server-Sent-Events helpers shared across REST routes.
//!
//! `OpenAI` streaming uses a fixed envelope:
//!
//! ```text
//! data: { ... json ... }\n\n
//! data: { ... json ... }\n\n
//! data: [DONE]\n\n
//! ```
//!
//! The `[DONE]` literal is not a valid JSON object — clients must
//! string-match it. [`finish_event`] returns the matching axum
//! [`axum::response::sse::Event`].
//!
//! For internal streams that carry typed Rust values (e.g.
//! [`crate::model_protocol::StreamCompleteChunk`]), [`json_event`]
//! serialises the value via `serde_json` and wraps the resulting string
//! in a `data:` event.

use axum::response::sse::Event;
use serde::Serialize;

/// Build a `data: <json>\n\n` SSE event from any `serde::Serialize` value.
///
/// # Errors
///
/// Returns a [`serde_json::Error`] when the value cannot be serialised.
pub fn json_event<T: Serialize>(value: &T) -> Result<Event, serde_json::Error> {
    let body = serde_json::to_string(value)?;
    Ok(Event::default().data(body))
}

/// The terminal `data: [DONE]\n\n` event every OpenAI-style stream must
/// emit before closing.
pub fn finish_event() -> Event {
    Event::default().data("[DONE]")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn event_debug(ev: &Event) -> String {
        // axum's Event doesn't expose its body directly; rely on the
        // Debug impl, which prints the byte buffer with backslash-
        // escaped quotes. Strip the backslashes before substring
        // matching so the expected JSON literal stays readable.
        let s = format!("{ev:?}");
        s.replace('\\', "")
    }

    #[test]
    fn json_event_carries_payload() {
        #[derive(serde::Serialize)]
        struct Tiny {
            x: u32,
        }
        let ev = json_event(&Tiny { x: 7 }).unwrap();
        let dbg = event_debug(&ev);
        assert!(dbg.contains("\"x\":7"), "debug repr: {dbg}");
    }

    #[test]
    fn finish_event_is_done_literal() {
        let ev = finish_event();
        let dbg = event_debug(&ev);
        assert!(dbg.contains("[DONE]"), "debug repr: {dbg}");
    }
}
