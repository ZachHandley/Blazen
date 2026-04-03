//! Lightweight WASM-compatible workflow context.
//!
//! [`WasmContext`] mirrors the API of `blazen_core::Context` but is designed
//! for `wasm32-unknown-unknown` where there is no tokio runtime and no
//! `Send + Sync` requirement.
//!
//! The struct uses `Rc<WasmContextInner>` so that cloning is cheap — the
//! event loop keeps one handle while a clone is passed to JS via
//! `JsValue::from(ctx.clone())`.

use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

use wasm_bindgen::prelude::*;

// ---------------------------------------------------------------------------
// TypeScript type augmentation
// ---------------------------------------------------------------------------

#[wasm_bindgen(typescript_custom_section)]
const TS_STATE_VALUE: &str = r#"
/** A value that can be stored in the workflow context state map. */
export type StateValue =
  | string
  | number
  | boolean
  | null
  | Uint8Array
  | StateValue[]
  | { [key: string]: StateValue };
"#;

// ---------------------------------------------------------------------------
// WasmStateEntry
// ---------------------------------------------------------------------------

/// Internal discriminated storage for context state values.
enum WasmStateEntry {
    /// Any JS value stored as-is.
    Value(JsValue),
    /// Raw binary data.
    Bytes(Vec<u8>),
}

// ---------------------------------------------------------------------------
// UUID v4 generator (Math.random-based, no external crate)
// ---------------------------------------------------------------------------

/// Generate a UUID v4 string using `js_sys::Math::random()`.
///
/// Follows RFC 4122 section 4.4: variant bits `10xx` and version nibble `4`.
fn generate_uuid_v4() -> String {
    let mut bytes = [0u8; 16];
    for byte in &mut bytes {
        *byte = (js_sys::Math::random() * 256.0) as u8;
    }

    // Set version 4 (bits 4-7 of byte 6).
    bytes[6] = (bytes[6] & 0x0F) | 0x40;
    // Set variant 1 (bits 6-7 of byte 8).
    bytes[8] = (bytes[8] & 0x3F) | 0x80;

    format!(
        "{:02x}{:02x}{:02x}{:02x}-{:02x}{:02x}-{:02x}{:02x}-{:02x}{:02x}-{:02x}{:02x}{:02x}{:02x}{:02x}{:02x}",
        bytes[0], bytes[1], bytes[2], bytes[3],
        bytes[4], bytes[5],
        bytes[6], bytes[7],
        bytes[8], bytes[9],
        bytes[10], bytes[11], bytes[12], bytes[13], bytes[14], bytes[15],
    )
}

// ---------------------------------------------------------------------------
// WasmContextInner
// ---------------------------------------------------------------------------

/// Shared inner state behind `Rc`.
struct WasmContextInner {
    workflow_name: String,
    run_id: String,
    state: RefCell<HashMap<String, WasmStateEntry>>,
    event_queue: RefCell<Vec<JsValue>>,
}

// ---------------------------------------------------------------------------
// WasmContext
// ---------------------------------------------------------------------------

/// A lightweight workflow context for the WASM runtime.
///
/// Cheaply clonable via `Rc` — the event loop keeps one handle while
/// a clone is converted to `JsValue` for the JS step handler.
#[wasm_bindgen(js_name = "Context")]
pub struct WasmContext {
    inner: Rc<WasmContextInner>,
}

impl Clone for WasmContext {
    fn clone(&self) -> Self {
        Self {
            inner: Rc::clone(&self.inner),
        }
    }
}

// ---------------------------------------------------------------------------
// Internal (non-wasm_bindgen) API
// ---------------------------------------------------------------------------

impl WasmContext {
    /// Create a new context for the given workflow.
    ///
    /// A UUID v4 `run_id` is generated automatically.
    pub(crate) fn new(workflow_name: String) -> Self {
        Self {
            inner: Rc::new(WasmContextInner {
                workflow_name,
                run_id: generate_uuid_v4(),
                state: RefCell::new(HashMap::new()),
                event_queue: RefCell::new(Vec::new()),
            }),
        }
    }

    /// Drain all queued events, returning them and leaving the queue empty.
    pub(crate) fn drain_events(&self) -> Vec<JsValue> {
        self.inner.event_queue.borrow_mut().drain(..).collect()
    }
}

// ---------------------------------------------------------------------------
// Public wasm_bindgen API
// ---------------------------------------------------------------------------

#[wasm_bindgen(js_class = "Context")]
impl WasmContext {
    /// Store a value in the context state map.
    ///
    /// If `value` is a `Uint8Array`, it is stored as raw bytes internally.
    /// All other JS values are stored as-is.
    #[wasm_bindgen]
    pub fn set(&self, key: String, value: JsValue) {
        let entry = if value.is_instance_of::<js_sys::Uint8Array>() {
            let arr: js_sys::Uint8Array = value.unchecked_into();
            WasmStateEntry::Bytes(arr.to_vec())
        } else {
            WasmStateEntry::Value(value)
        };
        self.inner.state.borrow_mut().insert(key, entry);
    }

    /// Retrieve a value from the context state map.
    ///
    /// - `Bytes` entries are returned as `Uint8Array`.
    /// - `Value` entries are returned as the original `JsValue`.
    /// - Missing keys return `JsValue::NULL`.
    #[wasm_bindgen]
    pub fn get(&self, key: String) -> JsValue {
        let state = self.inner.state.borrow();
        match state.get(&key) {
            Some(WasmStateEntry::Value(v)) => v.clone(),
            Some(WasmStateEntry::Bytes(b)) => js_sys::Uint8Array::from(b.as_slice()).into(),
            None => JsValue::NULL,
        }
    }

    /// Store raw binary data under the given key.
    #[wasm_bindgen(js_name = "setBytes")]
    pub fn set_bytes(&self, key: String, data: js_sys::Uint8Array) {
        self.inner
            .state
            .borrow_mut()
            .insert(key, WasmStateEntry::Bytes(data.to_vec()));
    }

    /// Retrieve raw binary data previously stored under the given key.
    ///
    /// Returns a `Uint8Array` if the key exists and was stored as bytes,
    /// otherwise returns `null`.
    #[wasm_bindgen(js_name = "getBytes")]
    pub fn get_bytes(&self, key: String) -> JsValue {
        let state = self.inner.state.borrow();
        match state.get(&key) {
            Some(WasmStateEntry::Bytes(b)) => js_sys::Uint8Array::from(b.as_slice()).into(),
            _ => JsValue::NULL,
        }
    }

    /// Push an event onto the internal event queue.
    #[wasm_bindgen(js_name = "sendEvent")]
    pub fn send_event(&self, event: JsValue) {
        self.inner.event_queue.borrow_mut().push(event);
    }

    /// No-op in the WASM runtime (API compatibility).
    #[wasm_bindgen(js_name = "writeEventToStream")]
    pub fn write_event_to_stream(&self, _event: JsValue) {}

    /// Return the unique run ID for this workflow execution.
    #[wasm_bindgen(js_name = "runId")]
    pub fn run_id(&self) -> String {
        self.inner.run_id.clone()
    }

    /// The workflow name.
    #[wasm_bindgen(getter, js_name = "workflowName")]
    pub fn workflow_name(&self) -> String {
        self.inner.workflow_name.clone()
    }
}
