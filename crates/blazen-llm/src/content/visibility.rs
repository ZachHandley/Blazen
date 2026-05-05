//! Make content handles visible to the model.
//!
//! When a conversation carries multimodal content via [`ContentHandle`]
//! references (either explicitly via tool inputs or implicitly via
//! auto-registered user-message attachments), the model needs to know
//! what handle ids exist so it can pass them as tool arguments.
//!
//! This module provides two complementary mechanisms:
//!
//! 1. [`collect_visible_handles`] — walk a slice of [`ChatMessage`] and
//!    return every [`ContentHandle`] that appears in any source position.
//! 2. [`build_handle_directory_system_note`] — render the collected
//!    handles into a system-prompt note describing the available
//!    references, ready to prepend or insert into the conversation.
//!
//! The agent runner is the natural place to call both: collect handles
//! before dispatch, render the directory, and inject it into the request.
//!
//! [`ContentHandle`]: super::handle::ContentHandle
//! [`ChatMessage`]: crate::types::ChatMessage

use std::collections::BTreeMap;

use super::handle::ContentHandle;
use crate::types::{ChatMessage, ContentPart, ImageSource, LlmPayload, MessageContent};

/// Walk every message and return every distinct [`ContentHandle`] referenced
/// by an [`ImageSource::Handle`] in user / assistant / tool content or in
/// tool-result `LlmPayload::Parts` overrides.
///
/// Handles are deduplicated by [`ContentHandle::id`] and returned in
/// first-seen order.
#[must_use]
pub fn collect_visible_handles(messages: &[ChatMessage]) -> Vec<ContentHandle> {
    let mut by_id: BTreeMap<String, ContentHandle> = BTreeMap::new();
    let mut order: Vec<String> = Vec::new();

    let mut record = |handle: &ContentHandle| {
        if !by_id.contains_key(&handle.id) {
            order.push(handle.id.clone());
            by_id.insert(handle.id.clone(), handle.clone());
        }
    };

    for msg in messages {
        visit_content(&msg.content, &mut record);
        if let Some(tool_result) = &msg.tool_result
            && let Some(payload) = &tool_result.llm_override
        {
            visit_payload(payload, &mut record);
        }
    }

    order
        .into_iter()
        .filter_map(|id| by_id.remove(&id))
        .collect()
}

fn visit_content(content: &MessageContent, record: &mut impl FnMut(&ContentHandle)) {
    match content {
        MessageContent::Text(_) => {}
        MessageContent::Image(img) => visit_source(&img.source, record),
        MessageContent::Parts(parts) => {
            for part in parts {
                visit_part(part, record);
            }
        }
    }
}

fn visit_part(part: &ContentPart, record: &mut impl FnMut(&ContentHandle)) {
    match part {
        ContentPart::Text { .. } => {}
        ContentPart::Image(img) => visit_source(&img.source, record),
        ContentPart::Audio(a) => visit_source(&a.source, record),
        ContentPart::Video(v) => visit_source(&v.source, record),
        ContentPart::File(f) => visit_source(&f.source, record),
    }
}

fn visit_payload(payload: &LlmPayload, record: &mut impl FnMut(&ContentHandle)) {
    if let LlmPayload::Parts { parts } = payload {
        for part in parts {
            visit_part(part, record);
        }
    }
}

fn visit_source(source: &ImageSource, record: &mut impl FnMut(&ContentHandle)) {
    if let ImageSource::Handle { handle } = source {
        record(handle);
    }
}

/// Render a list of handles into a human-readable system-note paragraph
/// the model can read. Returns `None` when `handles` is empty.
///
/// The text is intentionally compact — one bullet per handle — so it
/// doesn't dominate the prompt budget when many handles are in scope.
#[must_use]
pub fn build_handle_directory_system_note(handles: &[ContentHandle]) -> Option<String> {
    if handles.is_empty() {
        return None;
    }
    let mut out = String::with_capacity(64 + handles.len() * 64);
    out.push_str(
        "The following content references are available in this conversation. \
         Pass the quoted id verbatim as a tool argument to attach the content:\n",
    );
    for h in handles {
        out.push_str("- \"");
        out.push_str(&h.id);
        out.push_str("\": ");
        out.push_str(h.kind.as_str());
        if let Some(mime) = &h.mime_type {
            out.push_str(" (");
            out.push_str(mime);
            if let Some(size) = h.byte_size {
                out.push_str(", ");
                out.push_str(&format_size(size));
            }
            out.push(')');
        } else if let Some(size) = h.byte_size {
            out.push_str(" (");
            out.push_str(&format_size(size));
            out.push(')');
        }
        if let Some(name) = &h.display_name {
            out.push_str(" — \"");
            out.push_str(name);
            out.push('"');
        }
        out.push('\n');
    }
    Some(out)
}

/// One-call request preparation: resolve every `Handle`-source against
/// the store, then prepend a system note describing any handles still
/// referenced (typically from before-resolution callers, or from sources
/// the store could not fully materialize). Returns the number of handles
/// resolved.
///
/// Intended for the agent runner. If you only need one half, call
/// [`crate::types::CompletionRequest::resolve_handles_with`] or
/// [`build_handle_directory_system_note`] directly.
///
/// # Errors
///
/// Returns whatever
/// [`CompletionRequest::resolve_handles_with`](crate::types::CompletionRequest::resolve_handles_with)
/// returns.
pub async fn prepare_request_with_store(
    request: &mut crate::types::CompletionRequest,
    store: &dyn super::store::ContentStore,
) -> Result<usize, crate::error::BlazenError> {
    // Snapshot handles first so the directory note describes them by id
    // even after resolution rewrites the source variant.
    let handles = collect_visible_handles(&request.messages);
    let resolved = request.resolve_handles_with(store).await?;

    if let Some(note) = build_handle_directory_system_note(&handles) {
        request
            .messages
            .insert(0, crate::types::ChatMessage::system(note));
    }

    Ok(resolved)
}

fn format_size(bytes: u64) -> String {
    const KIB: u64 = 1024;
    const MIB: u64 = KIB * 1024;
    const GIB: u64 = MIB * 1024;
    // byte sizes for display only; precision loss is fine for KiB/MiB/GiB rendering
    #[allow(clippy::cast_precision_loss)]
    if bytes >= GIB {
        format!("{:.1} GiB", bytes as f64 / GIB as f64)
    } else if bytes >= MIB {
        format!("{:.1} MiB", bytes as f64 / MIB as f64)
    } else if bytes >= KIB {
        format!("{:.1} KiB", bytes as f64 / KIB as f64)
    } else {
        format!("{bytes} B")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::content::handle::ContentHandle;
    use crate::content::kind::ContentKind;
    use crate::types::{ChatMessage, ContentPart, ImageContent, ImageSource, MessageContent, Role};

    fn user_with_handle(h: &ContentHandle) -> ChatMessage {
        ChatMessage {
            role: Role::User,
            content: MessageContent::Parts(vec![ContentPart::Image(ImageContent {
                source: ImageSource::Handle { handle: h.clone() },
                media_type: h.mime_type.clone(),
            })]),
            tool_call_id: None,
            name: None,
            tool_calls: Vec::new(),
            tool_result: None,
        }
    }

    #[test]
    fn empty_messages_yield_no_handles() {
        let handles = collect_visible_handles(&[]);
        assert!(handles.is_empty());
    }

    #[test]
    fn handles_are_deduplicated_in_first_seen_order() {
        let h1 = ContentHandle::new("h_a", ContentKind::Image);
        let h2 = ContentHandle::new("h_b", ContentKind::Audio);
        let messages = vec![
            user_with_handle(&h1),
            user_with_handle(&h2),
            user_with_handle(&h1),
        ];
        let handles = collect_visible_handles(&messages);
        assert_eq!(handles.len(), 2);
        assert_eq!(handles[0].id, "h_a");
        assert_eq!(handles[1].id, "h_b");
    }

    #[test]
    fn directory_note_is_none_when_empty() {
        assert!(build_handle_directory_system_note(&[]).is_none());
    }

    #[test]
    fn directory_note_lists_handles_with_kind_and_mime() {
        let h = ContentHandle::new("h_abc", ContentKind::Image)
            .with_mime_type("image/png")
            .with_byte_size(2048)
            .with_display_name("vacation.png");
        let note = build_handle_directory_system_note(&[h]).unwrap();
        assert!(note.contains("\"h_abc\""));
        assert!(note.contains("image"));
        assert!(note.contains("image/png"));
        assert!(note.contains("2.0 KiB"));
        assert!(note.contains("vacation.png"));
    }

    #[tokio::test]
    async fn prepare_request_resolves_and_injects_system_note() {
        use crate::content::store::{ContentBody, ContentHint, ContentStore};
        use crate::content::stores::InMemoryContentStore;
        use crate::types::CompletionRequest;

        let store = InMemoryContentStore::new();
        let h = store
            .put(
                ContentBody::Url("https://example.com/x.png".into()),
                ContentHint::default()
                    .with_kind(ContentKind::Image)
                    .with_mime_type("image/png")
                    .with_display_name("x.png"),
            )
            .await
            .unwrap();
        let mut req = CompletionRequest::new(vec![user_with_handle(&h)]);
        let n = prepare_request_with_store(&mut req, &store).await.unwrap();
        assert_eq!(n, 1);
        // System note should have been prepended.
        assert!(matches!(req.messages[0].role, Role::System));
        let note = req.messages[0]
            .content
            .as_text()
            .expect("system text content");
        assert!(note.contains(&h.id));
        assert!(note.contains("image/png"));
        // Handle source resolved to URL.
        let MessageContent::Parts(parts) = &req.messages[1].content else {
            panic!()
        };
        let ContentPart::Image(img) = &parts[0] else {
            panic!()
        };
        assert!(matches!(img.source, ImageSource::Url { .. }));
    }

    #[test]
    fn handles_inside_tool_result_payload_are_collected() {
        use crate::types::{LlmPayload, ToolOutput};
        let h = ContentHandle::new("h_in_tool", ContentKind::Audio);
        let msg = ChatMessage::tool_result(
            "call_1",
            "speak",
            ToolOutput::with_override(
                serde_json::json!({}),
                LlmPayload::Parts {
                    parts: vec![ContentPart::Audio(crate::types::AudioContent {
                        source: ImageSource::Handle { handle: h.clone() },
                        media_type: Some("audio/wav".into()),
                        duration_seconds: None,
                    })],
                },
            ),
        );
        let handles = collect_visible_handles(&[msg]);
        assert_eq!(handles.len(), 1);
        assert_eq!(handles[0].id, "h_in_tool");
    }
}
