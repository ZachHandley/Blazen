//! Runtime registry of JS classes that extend `Error`, plus a builder API on
//! [`crate::Error`] for constructing typed JS error instances from Rust.
//!
//! ## Why this exists
//!
//! napi-rs lacks a way to throw a Rust [`Error`] as an instance of a
//! user-defined JS subclass of `Error`. The existing workaround in bindings
//! (encoding the class name + structured fields as sentinels in the error
//! message, then parsing them in a post-build JS shim) is brittle and bloats
//! the generated JS.
//!
//! This module lets you:
//! 1. At module init, register named JS classes that extend `Error` (or
//!    another previously-registered class) via [`register_error_class`].
//! 2. Anywhere in Rust, construct an [`Error`] tagged with a class name and
//!    structured fields via [`Error::with_class`] /
//!    [`Error::with_field`].
//!
//! When napi-rs converts the [`Error`] back to JS, the conversion path looks
//! up the class in the registry, constructs `new Class(reason, props)`, and
//! throws that instance. The JS caller sees `e instanceof MyError === true`
//! with structured fields as own properties.
//!
//! ## Class shape
//!
//! Registered classes are constructed by a single shared factory function so
//! the layout is consistent:
//!
//! ```js
//! class Name extends Parent {
//!   constructor(message, props) {
//!     super(message);
//!     this.name = "Name";
//!     if (props) Object.assign(this, props);
//!   }
//! }
//! ```
//!
//! ## Threading
//!
//! [`register_error_class`] must be called from the JS thread (typically
//! inside `#[napi::module_init]`). The registry is process-global; if you
//! register the same name twice, the second call replaces the first.
//!
//! The materialization path ([`Error::with_class`] → conversion to JS) runs
//! on whichever thread napi-rs is dispatching from; the registry mutex
//! serializes access to the class refs but does not touch any JS-thread-only
//! handles outside the conversion call.

use std::collections::HashMap;
use std::ffi::CString;
use std::ptr;
use std::sync::{Mutex, OnceLock};

use crate::bindgen_runtime::Object;
use crate::{check_status, sys, Env, Error, Result, Status};

// ----------------------------------------------------------------------------
// Field value type — what gets attached to the JS instance as own properties.
// ----------------------------------------------------------------------------

/// A structured field value attached to an error instance via
/// [`Error::with_field`]. Each variant maps to a JS primitive (or null).
#[derive(Debug, Clone, PartialEq)]
pub enum ErrorFieldValue {
  Null,
  Bool(bool),
  I64(i64),
  F64(f64),
  String(String),
}

impl From<bool> for ErrorFieldValue {
  fn from(v: bool) -> Self {
    Self::Bool(v)
  }
}

impl From<i32> for ErrorFieldValue {
  fn from(v: i32) -> Self {
    Self::I64(v as i64)
  }
}

impl From<i64> for ErrorFieldValue {
  fn from(v: i64) -> Self {
    Self::I64(v)
  }
}

impl From<u32> for ErrorFieldValue {
  fn from(v: u32) -> Self {
    Self::I64(v as i64)
  }
}

impl From<u16> for ErrorFieldValue {
  fn from(v: u16) -> Self {
    Self::I64(v as i64)
  }
}

impl From<u64> for ErrorFieldValue {
  fn from(v: u64) -> Self {
    // Saturate at i64::MAX to avoid silent truncation. Error fields are typed
    // for HTTP statuses / retry-after / counts, not for full-range u64.
    Self::I64(v.try_into().unwrap_or(i64::MAX))
  }
}

impl From<f32> for ErrorFieldValue {
  fn from(v: f32) -> Self {
    Self::F64(v as f64)
  }
}

impl From<f64> for ErrorFieldValue {
  fn from(v: f64) -> Self {
    Self::F64(v)
  }
}

impl From<String> for ErrorFieldValue {
  fn from(v: String) -> Self {
    Self::String(v)
  }
}

impl From<&str> for ErrorFieldValue {
  fn from(v: &str) -> Self {
    Self::String(v.to_owned())
  }
}

impl<T: Into<ErrorFieldValue>> From<Option<T>> for ErrorFieldValue {
  fn from(v: Option<T>) -> Self {
    match v {
      Some(inner) => inner.into(),
      None => Self::Null,
    }
  }
}

// ----------------------------------------------------------------------------
// Registry
// ----------------------------------------------------------------------------

/// One entry in the class registry: the `napi_ref` keeping the JS class alive
/// across GC cycles, plus the `napi_env` it belongs to (used at materialization
/// time to dereference the ref in the right context).
struct ClassRef {
  reference: sys::napi_ref,
  #[allow(dead_code)]
  env: sys::napi_env,
}

// SAFETY: napi_ref and napi_env are opaque pointer aliases (`*mut c_void`).
// At the Rust level they're trivially Send + Sync; the actual thread-safety
// invariant is that we only call into the napi APIs on the JS thread that
// owns the env. The registry's `Mutex` serializes HashMap access, and the
// materialize path runs inside `ToNapiValue::to_napi_value` which is itself
// only called on the JS thread.
unsafe impl Send for ClassRef {}
unsafe impl Sync for ClassRef {}

static REGISTRY: OnceLock<Mutex<HashMap<&'static str, ClassRef>>> = OnceLock::new();

fn registry() -> &'static Mutex<HashMap<&'static str, ClassRef>> {
  REGISTRY.get_or_init(|| Mutex::new(HashMap::new()))
}

// ----------------------------------------------------------------------------
// Registration
// ----------------------------------------------------------------------------

/// Register a JS class extending `Error` (or another previously-registered
/// class) under `name`. Subsequent calls to [`Error::with_class`] with the same
/// name materialize an instance of this class at throw time.
///
/// Must be called from the JS thread, typically inside `#[napi::module_init]`.
///
/// If `parent` is `Some(name)` but the parent has not yet been registered, an
/// error is returned. Register parents before children.
///
/// # Errors
/// Returns an error if any underlying napi call fails (script evaluation,
/// reference creation, etc.) or if the parent name is given but not yet
/// registered.
pub fn register_error_class(
  env: &Env,
  name: &'static str,
  parent: Option<&'static str>,
) -> Result<()> {
  let raw_env = env.raw();

  // Resolve parent class — either a previously-registered class, or
  // globalThis.Error if no parent name was given.
  let parent_value = if let Some(parent_name) = parent {
    let reg = registry().lock().unwrap_or_else(|e| e.into_inner());
    let parent_ref = reg.get(parent_name).ok_or_else(|| {
      Error::new(
        Status::InvalidArg,
        format!("parent error class `{parent_name}` not yet registered; register parents before children"),
      )
    })?;
    let mut p = ptr::null_mut();
    check_status!(
      unsafe { sys::napi_get_reference_value(raw_env, parent_ref.reference, &mut p) },
      "register_error_class: deref parent class reference failed"
    )?;
    p
  } else {
    let mut global = ptr::null_mut();
    check_status!(
      unsafe { sys::napi_get_global(raw_env, &mut global) },
      "register_error_class: get globalThis failed"
    )?;
    let key = CString::new("Error").expect("Error literal contains no NUL");
    let mut err = ptr::null_mut();
    check_status!(
      unsafe { sys::napi_get_named_property(raw_env, global, key.as_ptr(), &mut err) },
      "register_error_class: get globalThis.Error failed"
    )?;
    err
  };

  // Resolve the class-factory function. On hosts that allow dynamic code
  // generation (Node native, wasi-node, browsers via wasm-runtime), we
  // build it inline via `napi_run_script`. On hosts that block eval
  // (Cloudflare workerd's WASI runtime), the host predefines
  // `globalThis.__blazenErrorClassFactory` with the same shape and we
  // pick it up here instead. See
  // `crates/blazen-node/scripts/post-build.mjs` Section 4 for the
  // workerd-side definition.
  let mut global_for_factory = ptr::null_mut();
  check_status!(
    unsafe { sys::napi_get_global(raw_env, &mut global_for_factory) },
    "register_error_class: get globalThis failed (factory lookup)"
  )?;
  let host_factory_key =
    CString::new("__blazenErrorClassFactory").expect("literal contains no NUL");
  let mut host_factory = ptr::null_mut();
  check_status!(
    unsafe {
      sys::napi_get_named_property(
        raw_env,
        global_for_factory,
        host_factory_key.as_ptr(),
        &mut host_factory,
      )
    },
    "register_error_class: probe globalThis.__blazenErrorClassFactory failed"
  )?;
  let mut host_factory_type: sys::napi_valuetype = 0;
  check_status!(
    unsafe { sys::napi_typeof(raw_env, host_factory, &mut host_factory_type) },
    "register_error_class: typeof host factory failed"
  )?;

  let factory_fn = if host_factory_type == sys::ValueType::napi_function {
    host_factory
  } else {
    // No host-provided factory — fall back to building it inline.
    let factory_src = "(function(parent, name){return class extends parent{constructor(message,props){super(message);this.name=name;if(props)Object.assign(this,props);}};})";

    let mut script_str = ptr::null_mut();
    check_status!(
      unsafe {
        sys::napi_create_string_utf8(
          raw_env,
          factory_src.as_ptr().cast(),
          factory_src.len() as isize,
          &mut script_str,
        )
      },
      "register_error_class: create factory script string failed"
    )?;

    let mut compiled = ptr::null_mut();
    check_status!(
      unsafe { sys::napi_run_script(raw_env, script_str, &mut compiled) },
      "register_error_class: run factory script failed"
    )?;
    compiled
  };

  // Build the name string to pass as the second argument.
  let mut name_value = ptr::null_mut();
  check_status!(
    unsafe {
      sys::napi_create_string_utf8(raw_env, name.as_ptr().cast(), name.len() as isize, &mut name_value)
    },
    "register_error_class: create name string failed"
  )?;

  // Call the factory with (parent, name) → returns the class constructor.
  let mut undefined = ptr::null_mut();
  check_status!(
    unsafe { sys::napi_get_undefined(raw_env, &mut undefined) },
    "register_error_class: get undefined failed"
  )?;
  let args: [sys::napi_value; 2] = [parent_value, name_value];
  let mut class_value = ptr::null_mut();
  check_status!(
    unsafe {
      sys::napi_call_function(
        raw_env,
        undefined,
        factory_fn,
        args.len(),
        args.as_ptr(),
        &mut class_value,
      )
    },
    "register_error_class: call factory failed"
  )?;

  // Pin the class so it survives GC.
  let mut class_ref = ptr::null_mut();
  check_status!(
    unsafe { sys::napi_create_reference(raw_env, class_value, 1, &mut class_ref) },
    "register_error_class: create class reference failed"
  )?;

  registry().lock().unwrap_or_else(|e| e.into_inner()).insert(
    name,
    ClassRef {
      reference: class_ref,
      env: raw_env,
    },
  );

  Ok(())
}

/// Bind a previously-registered class onto the given exports object as a
/// named property. This lets JS callers do `require('mymod').MyError`
/// and use `instanceof` against the same class object the napi runtime
/// constructs at throw time.
///
/// # Errors
/// Returns an error if the class was never registered, or any underlying
/// napi call fails.
pub fn export_class_to(env: &Env, exports: &Object<'_>, name: &str) -> Result<()> {
  let raw_env = env.raw();
  let class_value = {
    let reg = registry().lock().unwrap_or_else(|e| e.into_inner());
    let entry = reg.get(name).ok_or_else(|| {
      Error::new(
        Status::InvalidArg,
        format!("error class `{name}` not registered; cannot export"),
      )
    })?;
    let mut v = ptr::null_mut();
    check_status!(
      unsafe { sys::napi_get_reference_value(raw_env, entry.reference, &mut v) },
      "export_class_to: deref class reference failed"
    )?;
    v
  };

  let key = CString::new(name).map_err(|e| {
    Error::new(
      Status::InvalidArg,
      format!("export_class_to: invalid name `{name}`: {e}"),
    )
  })?;
  // SAFETY: `exports.0.value` is a valid napi_value (an Object), `class_value`
  // was just produced by `napi_get_reference_value` on a live reference, and
  // both are owned by `raw_env`. We're inside the same JS thread that owns
  // `env`, which is required for any napi call on these handles.
  check_status!(
    unsafe { sys::napi_set_named_property(raw_env, exports.0.value, key.as_ptr(), class_value) },
    "export_class_to: napi_set_named_property failed"
  )?;
  Ok(())
}

// ----------------------------------------------------------------------------
// Materialization (called from `ToNapiValue::to_napi_value` for Error<S>)
// ----------------------------------------------------------------------------

/// Construct a JS instance of the named class with `reason` as the message and
/// `fields` attached as own properties. Returns the resulting `napi_value`.
///
/// # Safety
/// `env` must be a valid `napi_env` on the JS thread. Caller is responsible
/// for the lifetime of the returned value (typically it's used as a throw
/// value or stored in a `napi_ref`).
pub(crate) unsafe fn materialize_error_instance(
  env: sys::napi_env,
  class_name: &str,
  reason: &str,
  fields: &[(&'static str, ErrorFieldValue)],
) -> Result<sys::napi_value> {
  // Resolve the class reference, then drop the lock before any further JS
  // calls so we don't hold the mutex across napi calls that could re-enter
  // (e.g. via finalizers running on this thread).
  let class_value = {
    let reg = registry().lock().unwrap_or_else(|e| e.into_inner());
    let entry = reg.get(class_name).ok_or_else(|| {
      Error::new(
        Status::InvalidArg,
        format!("error class `{class_name}` not registered; call register_error_class at module init"),
      )
    })?;
    let mut v = ptr::null_mut();
    check_status!(
      unsafe { sys::napi_get_reference_value(env, entry.reference, &mut v) },
      "materialize_error_instance: deref class reference failed"
    )?;
    v
  };

  // Build the reason string.
  let mut reason_value = ptr::null_mut();
  check_status!(
    unsafe {
      sys::napi_create_string_utf8(env, reason.as_ptr().cast(), reason.len() as isize, &mut reason_value)
    },
    "materialize_error_instance: create reason string failed"
  )?;

  // Build the props object (or undefined when fields are empty).
  let mut props_value = ptr::null_mut();
  if fields.is_empty() {
    check_status!(
      unsafe { sys::napi_get_undefined(env, &mut props_value) },
      "materialize_error_instance: get undefined failed"
    )?;
  } else {
    check_status!(
      unsafe { sys::napi_create_object(env, &mut props_value) },
      "materialize_error_instance: create props object failed"
    )?;
    for (field_name, field_value) in fields {
      let napi_field_value = unsafe { field_to_napi_value(env, field_value)? };
      let field_key = CString::new(*field_name).map_err(|e| {
        Error::new(
          Status::InvalidArg,
          format!("invalid field name `{field_name}`: {e}"),
        )
      })?;
      check_status!(
        unsafe {
          sys::napi_set_named_property(env, props_value, field_key.as_ptr(), napi_field_value)
        },
        "materialize_error_instance: set_named_property failed"
      )?;
    }
  }

  // new Class(reason, props)
  let args: [sys::napi_value; 2] = [reason_value, props_value];
  let mut instance = ptr::null_mut();
  check_status!(
    unsafe { sys::napi_new_instance(env, class_value, args.len(), args.as_ptr(), &mut instance) },
    "materialize_error_instance: napi_new_instance failed"
  )?;

  Ok(instance)
}

unsafe fn field_to_napi_value(
  env: sys::napi_env,
  value: &ErrorFieldValue,
) -> Result<sys::napi_value> {
  let mut out = ptr::null_mut();
  match value {
    ErrorFieldValue::Null => check_status!(
      unsafe { sys::napi_get_null(env, &mut out) },
      "field_to_napi_value: get_null failed"
    )?,
    ErrorFieldValue::Bool(b) => check_status!(
      unsafe { sys::napi_get_boolean(env, *b, &mut out) },
      "field_to_napi_value: get_boolean failed"
    )?,
    ErrorFieldValue::I64(i) => check_status!(
      unsafe { sys::napi_create_int64(env, *i, &mut out) },
      "field_to_napi_value: create_int64 failed"
    )?,
    ErrorFieldValue::F64(f) => check_status!(
      unsafe { sys::napi_create_double(env, *f, &mut out) },
      "field_to_napi_value: create_double failed"
    )?,
    ErrorFieldValue::String(s) => check_status!(
      unsafe { sys::napi_create_string_utf8(env, s.as_ptr().cast(), s.len() as isize, &mut out) },
      "field_to_napi_value: create_string failed"
    )?,
  };
  Ok(out)
}
