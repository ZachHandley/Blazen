#![cfg(target_arch = "wasm32")]

//! Unit tests for the WASM `WasmLlmPayload` / `WasmContentPart` -> core
//! conversion path. Verifies the cross-layer `kind`/`type` discriminator
//! bug from 0.5.3 stays fixed: JS callers provide `partType` on inner
//! parts; the napi layer converts to the core `ContentPart` (which uses
//! `tag = "type"` internally).

use blazen_wasm_sdk::agent_types::{
    WasmContentPart, WasmImageContent, WasmImageSource, WasmLlmPayload,
};
use wasm_bindgen_test::{wasm_bindgen_test, wasm_bindgen_test_configure};

wasm_bindgen_test_configure!(run_in_browser);

#[wasm_bindgen_test]
fn parts_text_variant_round_trips() {
    let wasm_payload = WasmLlmPayload {
        kind: "parts".into(),
        text: None,
        value: None,
        parts: Some(vec![WasmContentPart {
            part_type: "text".into(),
            text: Some("summary".into()),
            image: None,
            file: None,
            audio: None,
            video: None,
        }]),
        provider: None,
    };

    let core_payload: blazen_llm::types::LlmPayload = wasm_payload.try_into().unwrap();
    match core_payload {
        blazen_llm::types::LlmPayload::Parts { parts } => {
            assert_eq!(parts.len(), 1);
            match &parts[0] {
                blazen_llm::types::ContentPart::Text { text } => {
                    assert_eq!(text, "summary");
                }
                _ => panic!("expected ContentPart::Text"),
            }
        }
        _ => panic!("expected LlmPayload::Parts"),
    }
}

#[wasm_bindgen_test]
fn parts_image_variant_round_trips() {
    let wasm_payload = WasmLlmPayload {
        kind: "parts".into(),
        text: None,
        value: None,
        parts: Some(vec![WasmContentPart {
            part_type: "image".into(),
            text: None,
            image: Some(WasmImageContent {
                source: WasmImageSource {
                    source_type: "url".into(),
                    url: Some("https://example.com/x.png".into()),
                    data: None,
                },
                media_type: Some("image/png".into()),
            }),
            file: None,
            audio: None,
            video: None,
        }]),
        provider: None,
    };

    let core_payload: blazen_llm::types::LlmPayload = wasm_payload.try_into().unwrap();
    match core_payload {
        blazen_llm::types::LlmPayload::Parts { parts } => {
            assert_eq!(parts.len(), 1);
            match &parts[0] {
                blazen_llm::types::ContentPart::Image(img) => {
                    assert_eq!(img.media_type.as_deref(), Some("image/png"));
                }
                _ => panic!("expected ContentPart::Image"),
            }
        }
        _ => panic!("expected LlmPayload::Parts"),
    }
}

#[wasm_bindgen_test]
fn text_variant_works() {
    let payload = WasmLlmPayload {
        kind: "text".into(),
        text: Some("hello".into()),
        value: None,
        parts: None,
        provider: None,
    };
    let core: blazen_llm::types::LlmPayload = payload.try_into().unwrap();
    match core {
        blazen_llm::types::LlmPayload::Text { text } => assert_eq!(text, "hello"),
        _ => panic!("expected Text variant"),
    }
}

#[wasm_bindgen_test]
fn unknown_kind_returns_error() {
    let payload = WasmLlmPayload {
        kind: "bogus".into(),
        text: None,
        value: None,
        parts: None,
        provider: None,
    };
    let result: Result<blazen_llm::types::LlmPayload, String> = payload.try_into();
    assert!(result.is_err());
    let err = result.unwrap_err();
    assert!(
        err.contains("bogus"),
        "error should mention the unknown kind: {err}"
    );
}
