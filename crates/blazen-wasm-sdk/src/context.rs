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

#[wasm_bindgen(typescript_custom_section)]
const TS_BLAZEN_STATE_META: &str = r#"
export interface BlazenStateMeta {
  transient?: string[];
  storeBy?: Record<string, { save(key: string, value: any, ctx: any): void; load(key: string, ctx: any): any }>;
}
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
// BlazenState helpers (non-wasm_bindgen)
// ---------------------------------------------------------------------------

/// JS property name constants used by the `BlazenState` protocol.
mod blazen_state_keys {
    pub const MARKER: &str = "__blazen_state__";
    pub const META_SUFFIX: &str = ".__blazen_meta__";
    pub const META_PROP: &str = "meta";
    pub const TRANSIENT: &str = "transient";
    pub const STORE_BY: &str = "storeBy";
    pub const CLASS_NAME: &str = "className";
    pub const FIELDS: &str = "fields";
    pub const RESTORE: &str = "restore";
}

/// Metadata extracted from a `BlazenState` object's constructor.
struct BlazenStateMeta {
    /// The JS value of the `transient` array (or `UNDEFINED`).
    transient_arr: JsValue,
    /// Pre-parsed set of transient field names for fast lookup.
    transient_set: std::collections::HashSet<String>,
    /// The `storeBy` object (or `UNDEFINED`).
    store_by: JsValue,
    /// The resolved class name.
    class_name: JsValue,
    /// The restore function name (or `UNDEFINED`).
    restore_fn_name: JsValue,
}

/// Read a property from `meta` if it is an object, otherwise return `UNDEFINED`.
fn meta_prop(meta: &JsValue, prop: &str) -> JsValue {
    if meta.is_object() {
        js_sys::Reflect::get(meta, &JsValue::from_str(prop)).unwrap_or(JsValue::UNDEFINED)
    } else {
        JsValue::UNDEFINED
    }
}

/// Look up a `FieldStore` in `store_by` for the given field name.
fn field_store(store_by: &JsValue, field_name_val: &JsValue) -> Option<JsValue> {
    if store_by.is_object() {
        js_sys::Reflect::get(store_by, field_name_val)
            .ok()
            .filter(JsValue::is_object)
    } else {
        None
    }
}

impl BlazenStateMeta {
    /// Extract metadata from a `BlazenState` JS value's constructor.
    fn from_value(value: &JsValue) -> Self {
        let constructor = js_sys::Reflect::get(value, &JsValue::from_str("constructor"))
            .unwrap_or(JsValue::UNDEFINED);

        let meta = if constructor.is_object() || constructor.is_function() {
            js_sys::Reflect::get(
                &constructor,
                &JsValue::from_str(blazen_state_keys::META_PROP),
            )
            .unwrap_or(JsValue::UNDEFINED)
        } else {
            JsValue::UNDEFINED
        };

        let transient_arr = meta_prop(&meta, blazen_state_keys::TRANSIENT);

        let transient_set: std::collections::HashSet<String> =
            if transient_arr.is_instance_of::<js_sys::Array>() {
                let arr: &js_sys::Array = transient_arr.unchecked_ref();
                arr.iter().filter_map(|v| v.as_string()).collect()
            } else {
                std::collections::HashSet::new()
            };

        let store_by = meta_prop(&meta, blazen_state_keys::STORE_BY);

        let class_name: JsValue = if meta.is_object() {
            js_sys::Reflect::get(&meta, &JsValue::from_str(blazen_state_keys::CLASS_NAME))
                .ok()
                .filter(JsValue::is_string)
                .unwrap_or_else(|| {
                    js_sys::Reflect::get(&constructor, &JsValue::from_str("name"))
                        .unwrap_or(JsValue::from_str("BlazenState"))
                })
        } else {
            js_sys::Reflect::get(&constructor, &JsValue::from_str("name"))
                .unwrap_or(JsValue::from_str("BlazenState"))
        };

        let restore_fn_name = meta_prop(&meta, blazen_state_keys::RESTORE);

        Self {
            transient_arr,
            transient_set,
            store_by,
            class_name,
            restore_fn_name,
        }
    }
}

impl WasmContext {
    /// Returns `true` if `value` is a JS object carrying the `__blazen_state__`
    /// marker property set to a truthy value.
    fn is_blazen_state(value: &JsValue) -> bool {
        if !value.is_object() {
            return false;
        }
        let marker_key = JsValue::from_str(blazen_state_keys::MARKER);
        js_sys::Reflect::get(value, &marker_key)
            .map(|v| v.is_truthy())
            .unwrap_or(false)
    }

    /// Decompose a `BlazenState` object and store each field individually.
    ///
    /// Stores metadata at `{key}.__blazen_meta__` so that [`get`] can
    /// reconstruct the object later.
    fn set_blazen_state(&self, key: &str, value: &JsValue) {
        let sm = BlazenStateMeta::from_value(value);

        // Iterate Object.keys(value), skipping the marker and transient fields.
        let obj_keys = js_sys::Object::keys(value.unchecked_ref::<js_sys::Object>());
        let fields_arr = js_sys::Array::new();
        let ctx_js: JsValue = self.clone().into();

        for i in 0..obj_keys.length() {
            let field_name_val = obj_keys.get(i);
            let Some(field_name) = field_name_val.as_string() else {
                continue;
            };

            if field_name == blazen_state_keys::MARKER || sm.transient_set.contains(&field_name) {
                continue;
            }

            fields_arr.push(&field_name_val);

            let field_value =
                js_sys::Reflect::get(value, &field_name_val).unwrap_or(JsValue::UNDEFINED);
            let field_key = format!("{key}.{field_name}");

            if let Some(store_obj) = field_store(&sm.store_by, &field_name_val) {
                // Call store.save(fieldKey, fieldValue, ctx)
                let save_fn = js_sys::Reflect::get(&store_obj, &JsValue::from_str("save"))
                    .unwrap_or(JsValue::UNDEFINED);
                if save_fn.is_function() {
                    let save: &js_sys::Function = save_fn.unchecked_ref();
                    let _ = save.call3(
                        &store_obj,
                        &JsValue::from_str(&field_key),
                        &field_value,
                        &ctx_js,
                    );
                }
            } else {
                self.set(field_key, field_value);
            }
        }

        self.persist_blazen_meta(key, &sm, &fields_arr);
    }

    /// Write the `__blazen_meta__` entry for a decomposed `BlazenState`.
    fn persist_blazen_meta(&self, key: &str, sm: &BlazenStateMeta, fields_arr: &js_sys::Array) {
        let meta_obj = js_sys::Object::new();
        let _ = js_sys::Reflect::set(
            &meta_obj,
            &JsValue::from_str(blazen_state_keys::CLASS_NAME),
            &sm.class_name,
        );
        let _ = js_sys::Reflect::set(
            &meta_obj,
            &JsValue::from_str(blazen_state_keys::FIELDS),
            fields_arr,
        );
        let _ = js_sys::Reflect::set(
            &meta_obj,
            &JsValue::from_str(blazen_state_keys::TRANSIENT),
            &sm.transient_arr,
        );
        if sm.store_by.is_object() {
            let _ = js_sys::Reflect::set(
                &meta_obj,
                &JsValue::from_str(blazen_state_keys::STORE_BY),
                &sm.store_by,
            );
        }
        if sm.restore_fn_name.is_string() {
            let _ = js_sys::Reflect::set(
                &meta_obj,
                &JsValue::from_str(blazen_state_keys::RESTORE),
                &sm.restore_fn_name,
            );
        }

        let meta_key = format!("{key}{}", blazen_state_keys::META_SUFFIX);
        self.set(meta_key, meta_obj.into());
    }

    /// Attempt to reconstruct a `BlazenState` object from its decomposed
    /// fields.  Returns `None` if there is no `__blazen_meta__` entry for
    /// `key`.
    fn get_blazen_state(&self, key: &str) -> Option<JsValue> {
        let meta_key = format!("{key}{}", blazen_state_keys::META_SUFFIX);

        // Read the metadata entry.  We must drop the borrow before calling
        // `self.get()` recursively.
        let meta_val = {
            let state = self.inner.state.borrow();
            match state.get(&meta_key) {
                Some(WasmStateEntry::Value(v)) => v.clone(),
                _ => return None,
            }
        };

        if !meta_val.is_object() {
            return None;
        }

        let fields = js_sys::Reflect::get(&meta_val, &JsValue::from_str(blazen_state_keys::FIELDS))
            .unwrap_or(JsValue::UNDEFINED);
        let fields_arr: &js_sys::Array = fields.dyn_ref::<js_sys::Array>()?;

        let store_by =
            js_sys::Reflect::get(&meta_val, &JsValue::from_str(blazen_state_keys::STORE_BY))
                .unwrap_or(JsValue::UNDEFINED);

        let result = js_sys::Object::new();
        let ctx_js: JsValue = self.clone().into();

        for i in 0..fields_arr.length() {
            let field_name_val = fields_arr.get(i);
            let Some(field_name) = field_name_val.as_string() else {
                continue;
            };

            let field_key = format!("{key}.{field_name}");

            let field_value = if let Some(store_obj) = field_store(&store_by, &field_name_val) {
                let load_fn = js_sys::Reflect::get(&store_obj, &JsValue::from_str("load"))
                    .unwrap_or(JsValue::UNDEFINED);
                if load_fn.is_function() {
                    let load: &js_sys::Function = load_fn.unchecked_ref();
                    load.call2(&store_obj, &JsValue::from_str(&field_key), &ctx_js)
                        .unwrap_or(JsValue::UNDEFINED)
                } else {
                    self.get(field_key)
                }
            } else {
                self.get(field_key)
            };

            let _ = js_sys::Reflect::set(&result, &field_name_val, &field_value);
        }

        // Set the __blazen_state__ marker on the reconstructed object.
        let _ = js_sys::Reflect::set(
            &result,
            &JsValue::from_str(blazen_state_keys::MARKER),
            &JsValue::TRUE,
        );

        // If a restore function name was saved, call it on the result object.
        let restore_name =
            js_sys::Reflect::get(&meta_val, &JsValue::from_str(blazen_state_keys::RESTORE))
                .unwrap_or(JsValue::UNDEFINED);
        if let Some(name) = restore_name.as_string() {
            let restore_fn = js_sys::Reflect::get(&result, &JsValue::from_str(&name))
                .unwrap_or(JsValue::UNDEFINED);
            if restore_fn.is_function() {
                let func: &js_sys::Function = restore_fn.unchecked_ref();
                let _ = func.call0(&result);
            }
        }

        Some(result.into())
    }
}

// ---------------------------------------------------------------------------
// Public wasm_bindgen API
// ---------------------------------------------------------------------------

#[wasm_bindgen(js_class = "Context")]
impl WasmContext {
    /// Store a value in the context state map.
    ///
    /// If `value` carries the `__blazen_state__` marker, the object is
    /// decomposed: each field is stored individually and metadata is persisted
    /// so that [`get`] can reconstruct it.
    ///
    /// If `value` is a `Uint8Array`, it is stored as raw bytes internally.
    /// All other JS values are stored as-is.
    #[wasm_bindgen]
    pub fn set(&self, key: String, value: JsValue) {
        // BlazenState decomposition takes priority.
        if Self::is_blazen_state(&value) {
            self.set_blazen_state(&key, &value);
            return;
        }

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
    /// If `{key}.__blazen_meta__` exists the value is a decomposed
    /// `BlazenState` — the object is reconstructed from its individual fields
    /// and returned.
    ///
    /// - `Bytes` entries are returned as `Uint8Array`.
    /// - `Value` entries are returned as the original `JsValue`.
    /// - Missing keys return `JsValue::NULL`.
    #[wasm_bindgen]
    pub fn get(&self, key: String) -> JsValue {
        // Attempt BlazenState reconstruction first.
        if let Some(reconstructed) = self.get_blazen_state(&key) {
            return reconstructed;
        }

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
