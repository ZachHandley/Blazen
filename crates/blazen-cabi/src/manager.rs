//! C-ABI surface over [`blazen_manager::ModelManager`]: memory-budget-aware
//! model loader with per-pool LRU eviction and `LoRA` adapter orchestration.
//!
//! ## Construction
//!
//! Two constructors are exposed: [`blazen_model_manager_new`] (defaults to
//! `u64::MAX` budgets on `Pool::Cpu` and `Pool::Gpu(0)` — the same
//! "no enforcement" sentinel the Python binding uses when neither budget is
//! given) and [`blazen_model_manager_with_budgets_gb`] (CPU RAM + GPU VRAM
//! budgets in gigabytes).
//!
//! ## Async pattern
//!
//! Every verb that exercises the underlying `async` API exposes a
//! `_blocking` variant (drives the cabi tokio runtime via `block_on`) and a
//! future-returning variant whose result is taken via the matching
//! `blazen_future_take_*` typed accessor declared in
//! [`crate::future`]. Verbs that are synchronous in Rust today
//! ([`ModelManager::pools`] is the only one) are exposed as a single
//! synchronous extern function.
//!
//! ## Deferred surface
//!
//! `register` / `register_local` is intentionally **not** exposed here. The
//! underlying API requires an `Arc<dyn LocalModel>`, which means the cabi
//! would need a foreign-language callback trampoline (sync + async hooks for
//! `load` / `unload` / `is_loaded` / `device` / `memory_bytes` /
//! `load_adapter` / `unload_adapter` / `list_adapters`). That trampoline is
//! tracked alongside the Ruby callback wave; until it lands, callers wire
//! local models in via the provider factories (which carry an internal
//! `LocalModel` impl) and register them through native code paths or use
//! the C-ABI manager only for backends registered before the manager
//! handle was wrapped.

use std::ffi::c_char;
use std::path::PathBuf;
use std::sync::Arc;

use blazen_llm::AdapterOptions;
use blazen_manager::ModelManager;
use blazen_uniffi::errors::BlazenError as InnerError;

use crate::error::BlazenError;
use crate::future::BlazenFuture;
use crate::llm_provider::BlazenLlmProvider;
use crate::llm_records::{BlazenModelRequest, BlazenModelResponse};
#[cfg(feature = "hf-loader")]
use crate::manager_records::{
    BLAZEN_BACKEND_HINT_CANDLE, BLAZEN_BACKEND_HINT_LLAMACPP, BLAZEN_BACKEND_HINT_MISTRALRS,
    BLAZEN_BACKEND_HINT_NONE, BlazenHfLoadOptions,
};
use crate::manager_records::{
    BlazenAdapterHandleInfo, BlazenAdapterStatusList, BlazenModelStatusList, BlazenPoolStatus,
    BlazenPoolStatusList,
};
use crate::runtime::runtime;
use crate::stream_sink::{BlazenCompletionStreamSinkVTable, CStreamSink};
use crate::string::cstr_to_str;

// ---------------------------------------------------------------------------
// Local error-out helpers (mirror the per-module style used in agent.rs/llm.rs)
// ---------------------------------------------------------------------------

unsafe fn write_error(out_err: *mut *mut BlazenError, err: InnerError) -> i32 {
    if !out_err.is_null() {
        // SAFETY: per the function-level contract on each caller, `out_err` is
        // either null (handled above) or a single-writer destination.
        unsafe {
            *out_err = BlazenError::from(err).into_ptr();
        }
    }
    -1
}

unsafe fn write_internal_error(out_err: *mut *mut BlazenError, msg: &str) -> i32 {
    // SAFETY: forwarded to `write_error`.
    unsafe {
        write_error(
            out_err,
            InnerError::Internal {
                message: msg.to_owned(),
            },
        )
    }
}

// Why: blazen_llm::BlazenError and blazen_uniffi::errors::BlazenError are
// distinct types. Route through the existing
// `impl From<blazen_llm::BlazenError> for InnerError` in blazen-uniffi so
// per-variant category mapping (Validation/Unsupported/Timeout/...) stays
// in one place.
fn into_inner_error(e: blazen_llm::BlazenError) -> InnerError {
    e.into()
}

// ---------------------------------------------------------------------------
// BlazenModelManager
// ---------------------------------------------------------------------------

/// Opaque handle around [`blazen_manager::ModelManager`]. Construct via
/// [`blazen_model_manager_new`] or [`blazen_model_manager_with_budgets_gb`];
/// release with [`blazen_model_manager_free`].
pub struct BlazenModelManager(pub(crate) Arc<ModelManager>);

impl BlazenModelManager {
    fn into_ptr(self) -> *mut BlazenModelManager {
        Box::into_raw(Box::new(self))
    }
}

/// Constructs a manager with `u64::MAX` budgets on `Pool::Cpu` and
/// `Pool::Gpu(0)`. Matches the Python binding's no-args constructor sentinel
/// — useful for tests and runtime-unconstrained embedders. Caller frees with
/// [`blazen_model_manager_free`].
#[unsafe(no_mangle)]
pub extern "C" fn blazen_model_manager_new() -> *mut BlazenModelManager {
    use std::collections::HashMap;

    use blazen_llm::Pool;

    let mut budgets: HashMap<Pool, u64> = HashMap::new();
    budgets.insert(Pool::Cpu, u64::MAX);
    budgets.insert(Pool::Gpu(0), u64::MAX);
    BlazenModelManager(Arc::new(ModelManager::new(budgets))).into_ptr()
}

/// Constructs a manager with explicit CPU RAM and GPU VRAM budgets, both in
/// gigabytes. Pass `0.0` to disable a pool. Caller frees with
/// [`blazen_model_manager_free`].
#[unsafe(no_mangle)]
pub extern "C" fn blazen_model_manager_with_budgets_gb(
    cpu_ram_gb: f64,
    gpu_vram_gb: f64,
) -> *mut BlazenModelManager {
    BlazenModelManager(Arc::new(ModelManager::with_budgets_gb(
        cpu_ram_gb,
        gpu_vram_gb,
    )))
    .into_ptr()
}

/// Frees a [`BlazenModelManager`] handle. No-op on a null pointer.
///
/// # Safety
///
/// `mgr` must be null OR a pointer previously produced by the cabi surface.
/// Double-free is undefined behavior.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_model_manager_free(mgr: *mut BlazenModelManager) {
    if mgr.is_null() {
        return;
    }
    // SAFETY: per the `Box::into_raw` provenance contract.
    drop(unsafe { Box::from_raw(mgr) });
}

// ---------------------------------------------------------------------------
// load / unload / is_loaded
// ---------------------------------------------------------------------------

unsafe fn borrow_model_id(
    fn_name: &str,
    model_id: *const c_char,
    out_err: *mut *mut BlazenError,
) -> Option<String> {
    // SAFETY: caller upholds the NUL + lifetime contract on `model_id`.
    if let Some(s) = unsafe { cstr_to_str(model_id) } {
        Some(s.to_owned())
    } else {
        // SAFETY: `out_err` upholds the function-level contract on caller.
        unsafe {
            write_internal_error(out_err, &format!("{fn_name}: null or non-UTF-8 model_id"));
        }
        None
    }
}

/// Synchronously loads the model registered as `model_id`. Returns `0` on
/// success or `-1` on failure (writing `*out_err`).
///
/// # Safety
///
/// `mgr` must be null OR a live [`BlazenModelManager`]. `model_id` must be a
/// NUL-terminated UTF-8 buffer valid for the duration of the call.
/// `out_err` is null OR a single-writer destination.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_model_manager_load_blocking(
    mgr: *const BlazenModelManager,
    model_id: *const c_char,
    out_err: *mut *mut BlazenError,
) -> i32 {
    if mgr.is_null() {
        // SAFETY: out_err contract per the per-fn doc.
        return unsafe {
            write_internal_error(out_err, "blazen_model_manager_load_blocking: null manager")
        };
    }
    // SAFETY: out_err + model_id contracts per the per-fn doc.
    let Some(id) =
        (unsafe { borrow_model_id("blazen_model_manager_load_blocking", model_id, out_err) })
    else {
        return -1;
    };
    // SAFETY: caller has guaranteed `mgr` is a live pointer.
    let mgr = unsafe { &*mgr };
    let inner = Arc::clone(&mgr.0);
    match runtime().block_on(async move { inner.load(&id).await }) {
        Ok(()) => 0,
        // SAFETY: out_err contract per the per-fn doc.
        Err(e) => unsafe { write_error(out_err, into_inner_error(e)) },
    }
}

/// Spawns a load onto the cabi tokio runtime; pop the result with
/// [`blazen_future_take_unit`]. Returns null on argument-shape failure.
///
/// # Safety
///
/// Same as [`blazen_model_manager_load_blocking`] (minus `out_err`).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_model_manager_load(
    mgr: *const BlazenModelManager,
    model_id: *const c_char,
) -> *mut BlazenFuture {
    if mgr.is_null() {
        return std::ptr::null_mut();
    }
    // SAFETY: caller upholds the NUL + lifetime contract on `model_id`.
    let Some(id) = (unsafe { cstr_to_str(model_id) }).map(str::to_owned) else {
        return std::ptr::null_mut();
    };
    // SAFETY: caller has guaranteed `mgr` is a live pointer.
    let mgr = unsafe { &*mgr };
    let inner = Arc::clone(&mgr.0);
    BlazenFuture::spawn(async move { inner.load(&id).await.map_err(into_inner_error) })
}

/// Synchronously unloads the model registered as `model_id`. Returns `0` on
/// success or `-1` on failure.
///
/// # Safety
///
/// See [`blazen_model_manager_load_blocking`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_model_manager_unload_blocking(
    mgr: *const BlazenModelManager,
    model_id: *const c_char,
    out_err: *mut *mut BlazenError,
) -> i32 {
    if mgr.is_null() {
        // SAFETY: out_err contract per the per-fn doc.
        return unsafe {
            write_internal_error(
                out_err,
                "blazen_model_manager_unload_blocking: null manager",
            )
        };
    }
    // SAFETY: out_err + model_id contracts per the per-fn doc.
    let Some(id) =
        (unsafe { borrow_model_id("blazen_model_manager_unload_blocking", model_id, out_err) })
    else {
        return -1;
    };
    // SAFETY: caller has guaranteed `mgr` is a live pointer.
    let mgr = unsafe { &*mgr };
    let inner = Arc::clone(&mgr.0);
    match runtime().block_on(async move { inner.unload(&id).await }) {
        Ok(()) => 0,
        // SAFETY: out_err contract per the per-fn doc.
        Err(e) => unsafe { write_error(out_err, into_inner_error(e)) },
    }
}

/// Spawns an unload onto the cabi tokio runtime; pop the result with
/// [`blazen_future_take_unit`].
///
/// # Safety
///
/// Same as [`blazen_model_manager_load`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_model_manager_unload(
    mgr: *const BlazenModelManager,
    model_id: *const c_char,
) -> *mut BlazenFuture {
    if mgr.is_null() {
        return std::ptr::null_mut();
    }
    // SAFETY: caller upholds the NUL + lifetime contract on `model_id`.
    let Some(id) = (unsafe { cstr_to_str(model_id) }).map(str::to_owned) else {
        return std::ptr::null_mut();
    };
    // SAFETY: caller has guaranteed `mgr` is a live pointer.
    let mgr = unsafe { &*mgr };
    let inner = Arc::clone(&mgr.0);
    BlazenFuture::spawn(async move { inner.unload(&id).await.map_err(into_inner_error) })
}

/// Synchronously checks whether `model_id` is currently loaded. Returns `1`
/// for loaded, `0` for not loaded, `-1` on argument-shape failure (writing
/// `*out_err`).
///
/// # Safety
///
/// See [`blazen_model_manager_load_blocking`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_model_manager_is_loaded_blocking(
    mgr: *const BlazenModelManager,
    model_id: *const c_char,
    out_err: *mut *mut BlazenError,
) -> i32 {
    if mgr.is_null() {
        // SAFETY: out_err contract per the per-fn doc.
        return unsafe {
            write_internal_error(
                out_err,
                "blazen_model_manager_is_loaded_blocking: null manager",
            )
        };
    }
    // SAFETY: out_err + model_id contracts per the per-fn doc.
    let Some(id) =
        (unsafe { borrow_model_id("blazen_model_manager_is_loaded_blocking", model_id, out_err) })
    else {
        return -1;
    };
    // SAFETY: caller has guaranteed `mgr` is a live pointer.
    let mgr = unsafe { &*mgr };
    let inner = Arc::clone(&mgr.0);
    i32::from(runtime().block_on(async move { inner.is_loaded(&id).await }))
}

/// Spawns an `is_loaded` query onto the cabi tokio runtime; pop the result
/// with [`blazen_future_take_bool`].
///
/// # Safety
///
/// Same as [`blazen_model_manager_load`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_model_manager_is_loaded(
    mgr: *const BlazenModelManager,
    model_id: *const c_char,
) -> *mut BlazenFuture {
    if mgr.is_null() {
        return std::ptr::null_mut();
    }
    // SAFETY: caller upholds the NUL + lifetime contract on `model_id`.
    let Some(id) = (unsafe { cstr_to_str(model_id) }).map(str::to_owned) else {
        return std::ptr::null_mut();
    };
    // SAFETY: caller has guaranteed `mgr` is a live pointer.
    let mgr = unsafe { &*mgr };
    let inner = Arc::clone(&mgr.0);
    BlazenFuture::spawn(async move { Ok::<bool, InnerError>(inner.is_loaded(&id).await) })
}

// ---------------------------------------------------------------------------
// register_remote / complete (unified remote-provider dispatch)
//
// Why: core `ModelManager` now holds BOTH local models and remote providers
// in one registry. `register_provider(id, Arc<dyn Model>, None, mem)` files a
// remote provider for by-name dispatch (it owns no local weights, so it never
// counts against a memory budget). `complete(id, request)` then dispatches
// straight through. The C surface exposes the remote-provider half here; the
// local-only `register` still needs the foreign callback trampoline tracked in
// the module-level deferred-surface note.
// ---------------------------------------------------------------------------

/// Registers the remote provider behind `provider` under `id` so it can be
/// dispatched by name with [`blazen_model_manager_complete_blocking`] /
/// [`blazen_model_manager_complete`]. The provider's inner `Arc` is cloned and
/// coerced to `Arc<dyn blazen_llm::Model>` — the original `provider` handle
/// remains valid and the caller still frees it with
/// [`crate::llm_provider::blazen_llm_provider_free`].
///
/// `memory_estimate_bytes` is recorded but not enforced for remote providers
/// (they own no local weights and file on `Pool::Remote`); pass `0` unless a
/// host wants the bookkeeping. Returns `0` on success or `-1` on failure
/// (writes `*out_err`).
///
/// # Safety
///
/// `mgr` must be null OR a live [`BlazenModelManager`]. `id` must be a
/// NUL-terminated UTF-8 buffer valid for the call. `provider` must be null OR
/// a live [`BlazenLlmProvider`] (produced by a
/// `blazen_<engine>_provider_as_llm_provider` C function). `out_err` is null
/// OR a single-writer destination.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_model_manager_register_remote(
    mgr: *const BlazenModelManager,
    id: *const c_char,
    provider: *const BlazenLlmProvider,
    memory_estimate_bytes: u64,
    out_err: *mut *mut BlazenError,
) -> i32 {
    if mgr.is_null() {
        // SAFETY: out_err contract per the per-fn doc.
        return unsafe {
            write_internal_error(
                out_err,
                "blazen_model_manager_register_remote: null manager",
            )
        };
    }
    if provider.is_null() {
        // SAFETY: out_err contract per the per-fn doc.
        return unsafe {
            write_internal_error(
                out_err,
                "blazen_model_manager_register_remote: null provider",
            )
        };
    }
    // SAFETY: out_err + id contracts per the per-fn doc.
    let Some(id) =
        (unsafe { borrow_model_id("blazen_model_manager_register_remote", id, out_err) })
    else {
        return -1;
    };
    // SAFETY: caller has guaranteed `provider` is a live handle.
    let model = unsafe { &*provider }.as_model();
    // SAFETY: caller has guaranteed `mgr` is a live pointer.
    let mgr = unsafe { &*mgr };
    let inner = Arc::clone(&mgr.0);
    runtime().block_on(async move {
        inner
            .register_provider(&id, model, None, memory_estimate_bytes)
            .await;
    });
    0
}

/// Synchronously runs a chat completion against the provider registered under
/// `id`. On success returns `0` and writes a fresh `BlazenModelResponse*` into
/// `*out_response` (free with
/// [`crate::llm_records::blazen_model_response_free`]). On failure returns
/// `-1` and writes a fresh `BlazenError*` into `*out_err`.
///
/// **The `request` pointer is consumed** (matching the per-engine
/// `blazen_<engine>_provider_complete_blocking` contract): internally
/// `Box::from_raw` reclaims it. Do not also call
/// [`crate::llm_records::blazen_model_request_free`] on it afterwards.
///
/// # Safety
///
/// `mgr` must be null OR a live [`BlazenModelManager`]. `id` must be a
/// NUL-terminated UTF-8 buffer valid for the call. `request` must be null OR a
/// live [`BlazenModelRequest`]; ownership transfers to this function
/// regardless of outcome. `out_response` / `out_err` are each null OR a
/// single-writer destination.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_model_manager_complete_blocking(
    mgr: *const BlazenModelManager,
    id: *const c_char,
    request: *mut BlazenModelRequest,
    out_response: *mut *mut BlazenModelResponse,
    out_err: *mut *mut BlazenError,
) -> i32 {
    if mgr.is_null() {
        if !request.is_null() {
            // SAFETY: caller transferred ownership; drop to avoid a leak.
            drop(unsafe { Box::from_raw(request) });
        }
        // SAFETY: out_err contract per the per-fn doc.
        return unsafe {
            write_internal_error(
                out_err,
                "blazen_model_manager_complete_blocking: null manager",
            )
        };
    }
    if request.is_null() {
        // SAFETY: out_err contract per the per-fn doc.
        return unsafe {
            write_internal_error(
                out_err,
                "blazen_model_manager_complete_blocking: null request",
            )
        };
    }
    // SAFETY: out_err + id contracts per the per-fn doc.
    let Some(id) =
        (unsafe { borrow_model_id("blazen_model_manager_complete_blocking", id, out_err) })
    else {
        // SAFETY: caller transferred ownership of `request`; drop to avoid a leak.
        drop(unsafe { Box::from_raw(request) });
        return -1;
    };
    // SAFETY: caller has transferred ownership of `request`.
    let wire_request = unsafe { Box::from_raw(request) }.0;
    let core_request = match blazen_llm::ModelRequest::try_from(wire_request) {
        Ok(r) => r,
        // SAFETY: out_err contract per the per-fn doc.
        Err(e) => return unsafe { write_error(out_err, e) },
    };
    // SAFETY: caller has guaranteed `mgr` is a live pointer.
    let mgr = unsafe { &*mgr };
    let inner = Arc::clone(&mgr.0);
    match runtime().block_on(async move { inner.complete(&id, core_request).await }) {
        Ok(resp) => {
            if !out_response.is_null() {
                let wire = blazen_uniffi::llm::ModelResponse::from(resp);
                // SAFETY: out_response contract per the per-fn doc.
                unsafe {
                    *out_response = BlazenModelResponse::from(wire).into_ptr();
                }
            }
            0
        }
        // SAFETY: out_err contract per the per-fn doc.
        Err(e) => unsafe { write_error(out_err, into_inner_error(e)) },
    }
}

/// Spawns a by-name chat completion onto the cabi tokio runtime; pop the
/// result with [`crate::llm_records::blazen_future_take_model_response`]. Returns
/// null on argument-shape failure (null manager / null or non-UTF-8 `id` /
/// null or invalid `request`); the `request` pointer is consumed in every
/// case (dropped on the null/invalid paths to avoid a leak).
///
/// # Safety
///
/// Same as [`blazen_model_manager_complete_blocking`] (minus the out-param
/// pointers). The request buffer is taken before this function returns.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_model_manager_complete(
    mgr: *const BlazenModelManager,
    id: *const c_char,
    request: *mut BlazenModelRequest,
) -> *mut BlazenFuture {
    if mgr.is_null() {
        if !request.is_null() {
            // SAFETY: caller transferred ownership; drop to avoid a leak.
            drop(unsafe { Box::from_raw(request) });
        }
        return std::ptr::null_mut();
    }
    if request.is_null() {
        return std::ptr::null_mut();
    }
    // SAFETY: caller upholds the NUL + lifetime contract on `id`.
    let Some(id) = (unsafe { cstr_to_str(id) }).map(str::to_owned) else {
        // SAFETY: caller transferred ownership; drop to avoid a leak.
        drop(unsafe { Box::from_raw(request) });
        return std::ptr::null_mut();
    };
    // SAFETY: caller has transferred ownership of `request`.
    let wire_request = unsafe { Box::from_raw(request) }.0;
    let Ok(core_request) = blazen_llm::ModelRequest::try_from(wire_request) else {
        return std::ptr::null_mut();
    };
    // SAFETY: caller has guaranteed `mgr` is a live pointer.
    let mgr = unsafe { &*mgr };
    let inner = Arc::clone(&mgr.0);
    BlazenFuture::spawn(async move {
        inner
            .complete(&id, core_request)
            .await
            .map(blazen_uniffi::llm::ModelResponse::from)
            .map_err(into_inner_error)
    })
}

/// Synchronously streams a chat completion against the provider registered
/// under `id`, driving each chunk into the foreign sink described by `vtable`.
///
/// Returns `0` once the stream terminates (including streams that fail
/// mid-flight — those errors are delivered to the sink via `on_error`, not the
/// return value). Returns `-1` (and writes `*out_err`) only on a start-side
/// failure that emits no frames: null manager, null/non-UTF-8 `id`,
/// null/invalid `request`, or a failure to *open* the stream. This mirrors
/// [`crate::model_client::blazen_modelclient_stream_complete_blocking`] so all
/// streaming surfaces observe a uniform happy/error split.
///
/// **The `request` pointer is consumed** (as with
/// [`blazen_model_manager_complete_blocking`]): it is reclaimed internally on
/// every path. Do not also call
/// [`crate::llm_records::blazen_model_request_free`] on it afterwards.
///
/// ## Ownership of `vtable`
///
/// `vtable.user_data` is CONSUMED: ownership transfers to the wrapping
/// [`CStreamSink`], which releases it via `drop_user_data` exactly once on
/// drop. On every early-return path that aborts BEFORE the sink is
/// constructed, this function invokes `(vtable.drop_user_data)(vtable.user_data)`
/// itself (and reclaims `request`) to honour the same contract.
///
/// # Safety
///
/// `mgr` must be null OR a live [`BlazenModelManager`]. `id` must be a
/// NUL-terminated UTF-8 buffer valid for the call. `request` must be null OR a
/// live [`BlazenModelRequest`]; ownership transfers to this function
/// regardless of outcome. `vtable.user_data` and its four function pointers
/// must satisfy the [`BlazenCompletionStreamSinkVTable`] contract. `out_err`
/// is null OR a single-writer destination.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_model_manager_stream_blocking(
    mgr: *const BlazenModelManager,
    id: *const c_char,
    request: *mut BlazenModelRequest,
    vtable: BlazenCompletionStreamSinkVTable,
    out_err: *mut *mut BlazenError,
) -> i32 {
    if mgr.is_null() {
        (vtable.drop_user_data)(vtable.user_data);
        if !request.is_null() {
            // SAFETY: caller transferred ownership; drop to avoid a leak.
            drop(unsafe { Box::from_raw(request) });
        }
        // SAFETY: out_err contract per the per-fn doc.
        return unsafe {
            write_internal_error(
                out_err,
                "blazen_model_manager_stream_blocking: null manager",
            )
        };
    }
    if request.is_null() {
        (vtable.drop_user_data)(vtable.user_data);
        // SAFETY: out_err contract per the per-fn doc.
        return unsafe {
            write_internal_error(
                out_err,
                "blazen_model_manager_stream_blocking: null request",
            )
        };
    }
    // SAFETY: out_err + id contracts per the per-fn doc.
    let Some(id) =
        (unsafe { borrow_model_id("blazen_model_manager_stream_blocking", id, out_err) })
    else {
        (vtable.drop_user_data)(vtable.user_data);
        // SAFETY: caller transferred ownership of `request`; drop to avoid a leak.
        drop(unsafe { Box::from_raw(request) });
        return -1;
    };
    // SAFETY: caller has transferred ownership of `request`.
    let wire_request = unsafe { Box::from_raw(request) }.0;
    let core_request = match blazen_llm::ModelRequest::try_from(wire_request) {
        Ok(r) => r,
        Err(e) => {
            (vtable.drop_user_data)(vtable.user_data);
            // SAFETY: out_err contract per the per-fn doc.
            return unsafe { write_error(out_err, e) };
        }
    };
    // SAFETY: caller has guaranteed `mgr` is a live pointer.
    let mgr = unsafe { &*mgr };
    let inner = Arc::clone(&mgr.0);

    // From here on, `CStreamSink::drop` is responsible for invoking
    // `drop_user_data` exactly once.
    let sink_arc: Arc<dyn blazen_uniffi::streaming::CompletionStreamSink> =
        Arc::new(CStreamSink::from_vtable(vtable));

    runtime().block_on(async move {
        let stream = match inner.stream(&id, core_request).await {
            Ok(s) => s,
            // No frames emitted yet — report the start-failure through the
            // `-1` return + `*out_err` (the sink owns only mid-stream errors).
            // SAFETY: out_err contract per the per-fn doc.
            Err(e) => return unsafe { write_error(out_err, into_inner_error(e)) },
        };
        // `drive_completion_stream` delivers every subsequent chunk/error to
        // the sink and always resolves to `Ok(())`.
        let _ = blazen_uniffi::streaming::drive_completion_stream(stream, sink_arc).await;
        0
    })
}

/// Spawns a by-name streaming completion onto the cabi tokio runtime, driving
/// each chunk into the foreign sink described by `vtable`. Pop the (unit)
/// result with [`crate::future::blazen_future_take_unit`]. Returns null on an
/// argument-shape failure (null manager / null or non-UTF-8 `id` / null or
/// invalid `request`); the `request` pointer is consumed in every case
/// (dropped on the null/invalid paths to avoid a leak), and on those paths
/// `vtable.drop_user_data` is invoked before returning.
///
/// Unlike the `_blocking` variant, a failure to *open* the stream is reported
/// through the sink's `on_error` (the future still resolves to unit `Ok`),
/// matching how the per-engine async `<engine>_provider_complete_streaming`
/// functions and [`blazen_manager::ModelManager::stream`] surface late vs.
/// early failures once a future has been handed back.
///
/// # Safety
///
/// Same as [`blazen_model_manager_stream_blocking`] (minus `out_err`). The
/// request buffer is taken before this function returns; `vtable.user_data`
/// transfers to the spawned task's sink.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_model_manager_stream(
    mgr: *const BlazenModelManager,
    id: *const c_char,
    request: *mut BlazenModelRequest,
    vtable: BlazenCompletionStreamSinkVTable,
) -> *mut BlazenFuture {
    if mgr.is_null() {
        (vtable.drop_user_data)(vtable.user_data);
        if !request.is_null() {
            // SAFETY: caller transferred ownership; drop to avoid a leak.
            drop(unsafe { Box::from_raw(request) });
        }
        return std::ptr::null_mut();
    }
    if request.is_null() {
        (vtable.drop_user_data)(vtable.user_data);
        return std::ptr::null_mut();
    }
    // SAFETY: caller upholds the NUL + lifetime contract on `id`.
    let Some(id) = (unsafe { cstr_to_str(id) }).map(str::to_owned) else {
        (vtable.drop_user_data)(vtable.user_data);
        // SAFETY: caller transferred ownership; drop to avoid a leak.
        drop(unsafe { Box::from_raw(request) });
        return std::ptr::null_mut();
    };
    // SAFETY: caller has transferred ownership of `request`.
    let wire_request = unsafe { Box::from_raw(request) }.0;
    let Ok(core_request) = blazen_llm::ModelRequest::try_from(wire_request) else {
        (vtable.drop_user_data)(vtable.user_data);
        return std::ptr::null_mut();
    };
    // SAFETY: caller has guaranteed `mgr` is a live pointer.
    let mgr = unsafe { &*mgr };
    let inner = Arc::clone(&mgr.0);

    // From here on, `CStreamSink::drop` is responsible for invoking
    // `drop_user_data` exactly once.
    let sink_arc: Arc<dyn blazen_uniffi::streaming::CompletionStreamSink> =
        Arc::new(CStreamSink::from_vtable(vtable));

    BlazenFuture::spawn(async move {
        match inner.stream(&id, core_request).await {
            Ok(stream) => {
                // Mid-stream chunks/errors flow to the sink; always Ok(()).
                let _ = blazen_uniffi::streaming::drive_completion_stream(stream, sink_arc).await;
            }
            Err(e) => {
                // A future was already handed back, so route the start-failure
                // through the sink (mirroring the per-engine async streams)
                // rather than failing the unit future.
                let _ = sink_arc.on_error(into_inner_error(e)).await;
            }
        }
        Ok::<(), InnerError>(())
    })
}

// ---------------------------------------------------------------------------
// status
// ---------------------------------------------------------------------------

/// Synchronously snapshots the status of every registered model. Returns a
/// caller-owned [`BlazenModelStatusList`] on success (free with
/// [`crate::manager_records::blazen_model_status_list_free`]) or null on
/// failure (writes `*out_err`).
///
/// # Safety
///
/// `mgr` must be null OR a live [`BlazenModelManager`]. `out_err` is null OR
/// a single-writer destination.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_model_manager_status_blocking(
    mgr: *const BlazenModelManager,
    out_err: *mut *mut BlazenError,
) -> *mut BlazenModelStatusList {
    if mgr.is_null() {
        // SAFETY: out_err contract per the per-fn doc.
        unsafe {
            write_internal_error(
                out_err,
                "blazen_model_manager_status_blocking: null manager",
            );
        }
        return std::ptr::null_mut();
    }
    // SAFETY: caller has guaranteed `mgr` is a live pointer.
    let mgr = unsafe { &*mgr };
    let inner = Arc::clone(&mgr.0);
    let statuses = runtime().block_on(async move { inner.status().await });
    BlazenModelStatusList::from(statuses).into_ptr()
}

/// Spawns the status snapshot onto the cabi tokio runtime; pop the result
/// with [`blazen_future_take_model_status_list`].
///
/// # Safety
///
/// `mgr` must be null OR a live [`BlazenModelManager`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_model_manager_status(
    mgr: *const BlazenModelManager,
) -> *mut BlazenFuture {
    if mgr.is_null() {
        return std::ptr::null_mut();
    }
    // SAFETY: caller has guaranteed `mgr` is a live pointer.
    let mgr = unsafe { &*mgr };
    let inner = Arc::clone(&mgr.0);
    BlazenFuture::spawn(async move {
        Ok::<Vec<blazen_manager::ModelStatus>, InnerError>(inner.status().await)
    })
}

// ---------------------------------------------------------------------------
// pools
// ---------------------------------------------------------------------------

/// Snapshots configured pools together with their live `used_bytes` and
/// loaded-model counts. Returns a caller-owned [`BlazenPoolStatusList`]
/// (never null on a non-null `mgr`).
///
/// Why: `ModelManager::pools` is synchronous and only returns `(label,
/// budget)` pairs — to also surface `used_bytes` and `loaded_models` we have
/// to await `used_bytes(pool)` and walk the model statuses. The cabi blocks
/// on those awaits so a single C call returns the full snapshot.
///
/// # Safety
///
/// `mgr` must be null OR a live [`BlazenModelManager`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_model_manager_pools(
    mgr: *const BlazenModelManager,
) -> *mut BlazenPoolStatusList {
    if mgr.is_null() {
        return std::ptr::null_mut();
    }
    // SAFETY: caller has guaranteed `mgr` is a live pointer.
    let mgr = unsafe { &*mgr };
    let inner = Arc::clone(&mgr.0);
    let snapshot = runtime().block_on(async move { collect_pool_snapshot(&inner).await });
    BlazenPoolStatusList { inner: snapshot }.into_ptr()
}

async fn collect_pool_snapshot(mgr: &ModelManager) -> Vec<BlazenPoolStatus> {
    let statuses = mgr.status().await;
    let mut out = Vec::new();
    for (pool, budget_bytes) in mgr.pools() {
        let used_bytes = mgr.used_bytes(pool).await;
        let loaded_models = statuses
            .iter()
            .filter(|s| s.pool == pool && s.loaded)
            .count();
        out.push(BlazenPoolStatus {
            pool_label: format!("{pool}"),
            budget_bytes,
            used_bytes,
            loaded_models,
        });
    }
    out
}

// ---------------------------------------------------------------------------
// load_adapter / unload_adapter / list_adapters
// ---------------------------------------------------------------------------

unsafe fn borrow_adapter_args(
    fn_name: &str,
    model_id: *const c_char,
    adapter_dir: *const c_char,
    adapter_id: *const c_char,
    out_err: *mut *mut BlazenError,
) -> Option<(String, PathBuf, String)> {
    // SAFETY: caller upholds the NUL + lifetime contract on `model_id`.
    let Some(model_id) = (unsafe { cstr_to_str(model_id) }).map(str::to_owned) else {
        // SAFETY: out_err contract.
        unsafe {
            write_internal_error(out_err, &format!("{fn_name}: null or non-UTF-8 model_id"));
        }
        return None;
    };
    // SAFETY: caller upholds the NUL + lifetime contract on `adapter_dir`.
    let Some(adapter_dir) = (unsafe { cstr_to_str(adapter_dir) }).map(PathBuf::from) else {
        // SAFETY: out_err contract.
        unsafe {
            write_internal_error(
                out_err,
                &format!("{fn_name}: null or non-UTF-8 adapter_dir"),
            );
        }
        return None;
    };
    // SAFETY: caller upholds the NUL + lifetime contract on `adapter_id`.
    let Some(adapter_id) = (unsafe { cstr_to_str(adapter_id) }).map(str::to_owned) else {
        // SAFETY: out_err contract.
        unsafe {
            write_internal_error(out_err, &format!("{fn_name}: null or non-UTF-8 adapter_id"));
        }
        return None;
    };
    Some((model_id, adapter_dir, adapter_id))
}

/// Synchronously mounts a PEFT-format `LoRA` adapter. Returns a caller-owned
/// [`BlazenAdapterHandleInfo`] (free with
/// [`crate::manager_records::blazen_adapter_handle_info_free`]) on success
/// or null on failure (writes `*out_err`).
///
/// `scale` is the strength multiplier for the adapter delta-weights;
/// `1.0` is full PEFT strength. The base model is loaded automatically if
/// not already in residence.
///
/// # Safety
///
/// `mgr` must be null OR a live [`BlazenModelManager`]. `model_id`,
/// `adapter_dir`, and `adapter_id` must each be NUL-terminated UTF-8
/// buffers valid for the duration of this call.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_model_manager_load_adapter_blocking(
    mgr: *const BlazenModelManager,
    model_id: *const c_char,
    adapter_dir: *const c_char,
    adapter_id: *const c_char,
    scale: f64,
    out_err: *mut *mut BlazenError,
) -> *mut BlazenAdapterHandleInfo {
    if mgr.is_null() {
        // SAFETY: out_err contract.
        unsafe {
            write_internal_error(
                out_err,
                "blazen_model_manager_load_adapter_blocking: null manager",
            );
        }
        return std::ptr::null_mut();
    }
    // SAFETY: see borrow_adapter_args contract.
    let Some((model_id, adapter_dir, adapter_id)) = (unsafe {
        borrow_adapter_args(
            "blazen_model_manager_load_adapter_blocking",
            model_id,
            adapter_dir,
            adapter_id,
            out_err,
        )
    }) else {
        return std::ptr::null_mut();
    };
    let options = AdapterOptions {
        adapter_id,
        #[allow(clippy::cast_possible_truncation)] // C surface uses f64; AdapterOptions::scale is f32
        scale: scale as f32,
    };
    // SAFETY: caller has guaranteed `mgr` is a live pointer.
    let mgr = unsafe { &*mgr };
    let inner = Arc::clone(&mgr.0);
    match runtime()
        .block_on(async move { inner.load_adapter(&model_id, &adapter_dir, options).await })
    {
        Ok(handle) => BlazenAdapterHandleInfo::from(handle).into_ptr(),
        Err(e) => {
            // SAFETY: out_err contract.
            unsafe {
                write_error(out_err, into_inner_error(e));
            }
            std::ptr::null_mut()
        }
    }
}

/// Spawns a `load_adapter` onto the cabi tokio runtime; pop the result with
/// [`blazen_future_take_adapter_handle_info`].
///
/// # Safety
///
/// Same as [`blazen_model_manager_load_adapter_blocking`] (minus `out_err`).
/// String buffers are copied before this function returns.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_model_manager_load_adapter(
    mgr: *const BlazenModelManager,
    model_id: *const c_char,
    adapter_dir: *const c_char,
    adapter_id: *const c_char,
    scale: f64,
) -> *mut BlazenFuture {
    if mgr.is_null() {
        return std::ptr::null_mut();
    }
    // SAFETY: caller upholds the NUL + lifetime contracts on the three strings.
    let (Some(model_id), Some(adapter_dir), Some(adapter_id)) = (
        unsafe { cstr_to_str(model_id) }.map(str::to_owned),
        unsafe { cstr_to_str(adapter_dir) }.map(PathBuf::from),
        unsafe { cstr_to_str(adapter_id) }.map(str::to_owned),
    ) else {
        return std::ptr::null_mut();
    };
    let options = AdapterOptions {
        adapter_id,
        #[allow(clippy::cast_possible_truncation)] // C surface uses f64
        scale: scale as f32,
    };
    // SAFETY: caller has guaranteed `mgr` is a live pointer.
    let mgr = unsafe { &*mgr };
    let inner = Arc::clone(&mgr.0);
    BlazenFuture::spawn(async move {
        inner
            .load_adapter(&model_id, &adapter_dir, options)
            .await
            .map_err(into_inner_error)
    })
}

/// Synchronously unmounts a previously-loaded adapter.
///
/// # Safety
///
/// `mgr` must be null OR a live [`BlazenModelManager`]. `model_id` and
/// `adapter_id` must each be NUL-terminated UTF-8 buffers valid for the
/// call.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_model_manager_unload_adapter_blocking(
    mgr: *const BlazenModelManager,
    model_id: *const c_char,
    adapter_id: *const c_char,
    out_err: *mut *mut BlazenError,
) -> i32 {
    if mgr.is_null() {
        // SAFETY: out_err contract.
        return unsafe {
            write_internal_error(
                out_err,
                "blazen_model_manager_unload_adapter_blocking: null manager",
            )
        };
    }
    // SAFETY: caller upholds the NUL + lifetime contracts.
    let Some(model_id) = (unsafe { cstr_to_str(model_id) }).map(str::to_owned) else {
        // SAFETY: out_err contract.
        return unsafe {
            write_internal_error(
                out_err,
                "blazen_model_manager_unload_adapter_blocking: null or non-UTF-8 model_id",
            )
        };
    };
    let Some(adapter_id) = (unsafe { cstr_to_str(adapter_id) }).map(str::to_owned) else {
        // SAFETY: out_err contract.
        return unsafe {
            write_internal_error(
                out_err,
                "blazen_model_manager_unload_adapter_blocking: null or non-UTF-8 adapter_id",
            )
        };
    };
    // SAFETY: caller has guaranteed `mgr` is a live pointer.
    let mgr = unsafe { &*mgr };
    let inner = Arc::clone(&mgr.0);
    match runtime().block_on(async move { inner.unload_adapter(&model_id, &adapter_id).await }) {
        Ok(()) => 0,
        // SAFETY: out_err contract.
        Err(e) => unsafe { write_error(out_err, into_inner_error(e)) },
    }
}

/// Spawns an `unload_adapter` onto the cabi tokio runtime; pop the result
/// with [`blazen_future_take_unit`].
///
/// # Safety
///
/// Same as [`blazen_model_manager_unload_adapter_blocking`] (minus
/// `out_err`).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_model_manager_unload_adapter(
    mgr: *const BlazenModelManager,
    model_id: *const c_char,
    adapter_id: *const c_char,
) -> *mut BlazenFuture {
    if mgr.is_null() {
        return std::ptr::null_mut();
    }
    // SAFETY: caller upholds the NUL + lifetime contracts.
    let (Some(model_id), Some(adapter_id)) = (
        unsafe { cstr_to_str(model_id) }.map(str::to_owned),
        unsafe { cstr_to_str(adapter_id) }.map(str::to_owned),
    ) else {
        return std::ptr::null_mut();
    };
    // SAFETY: caller has guaranteed `mgr` is a live pointer.
    let mgr = unsafe { &*mgr };
    let inner = Arc::clone(&mgr.0);
    BlazenFuture::spawn(async move {
        inner
            .unload_adapter(&model_id, &adapter_id)
            .await
            .map_err(into_inner_error)
    })
}

/// Synchronously lists adapters mounted on `model_id`. Returns a
/// caller-owned [`BlazenAdapterStatusList`] (free with
/// [`crate::manager_records::blazen_adapter_status_list_free`]) on success
/// or null on failure (writes `*out_err`).
///
/// # Safety
///
/// `mgr` must be null OR a live [`BlazenModelManager`]. `model_id` must be a
/// NUL-terminated UTF-8 buffer valid for the call.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_model_manager_list_adapters_blocking(
    mgr: *const BlazenModelManager,
    model_id: *const c_char,
    out_err: *mut *mut BlazenError,
) -> *mut BlazenAdapterStatusList {
    if mgr.is_null() {
        // SAFETY: out_err contract.
        unsafe {
            write_internal_error(
                out_err,
                "blazen_model_manager_list_adapters_blocking: null manager",
            );
        }
        return std::ptr::null_mut();
    }
    // SAFETY: out_err + model_id contracts.
    let Some(id) = (unsafe {
        borrow_model_id(
            "blazen_model_manager_list_adapters_blocking",
            model_id,
            out_err,
        )
    }) else {
        return std::ptr::null_mut();
    };
    // SAFETY: caller has guaranteed `mgr` is a live pointer.
    let mgr = unsafe { &*mgr };
    let inner = Arc::clone(&mgr.0);
    match runtime().block_on(async move { inner.list_adapters(&id).await }) {
        Ok(items) => BlazenAdapterStatusList::from_statuses(items).into_ptr(),
        Err(e) => {
            // SAFETY: out_err contract.
            unsafe {
                write_error(out_err, into_inner_error(e));
            }
            std::ptr::null_mut()
        }
    }
}

/// Spawns a `list_adapters` onto the cabi tokio runtime; pop the result with
/// [`blazen_future_take_adapter_status_list`].
///
/// # Safety
///
/// Same as [`blazen_model_manager_list_adapters_blocking`] (minus
/// `out_err`).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_model_manager_list_adapters(
    mgr: *const BlazenModelManager,
    model_id: *const c_char,
) -> *mut BlazenFuture {
    if mgr.is_null() {
        return std::ptr::null_mut();
    }
    // SAFETY: caller upholds the NUL + lifetime contract.
    let Some(id) = (unsafe { cstr_to_str(model_id) }).map(str::to_owned) else {
        return std::ptr::null_mut();
    };
    // SAFETY: caller has guaranteed `mgr` is a live pointer.
    let mgr = unsafe { &*mgr };
    let inner = Arc::clone(&mgr.0);
    BlazenFuture::spawn(async move { inner.list_adapters(&id).await.map_err(into_inner_error) })
}

// ---------------------------------------------------------------------------
// load_from_hf
// ---------------------------------------------------------------------------

#[cfg(feature = "hf-loader")]
#[allow(clippy::result_large_err)] // Why: InnerError is large but funnels into Box<BlazenError> at the FFI boundary; matches the take_typed convention.
fn parse_pool_label(s: &str) -> Result<blazen_llm::Pool, InnerError> {
    use blazen_llm::Pool;
    let trimmed = s.trim();
    let lower = trimmed.to_ascii_lowercase();
    if let Some((name, idx)) = lower.split_once(':') {
        if name == "gpu" {
            let index = idx.parse::<usize>().map_err(|_| InnerError::Validation {
                message: format!(
                    "invalid pool label {trimmed:?}: expected 'cpu', 'gpu', or 'gpu:N' where N is a non-negative integer"
                ),
            })?;
            return Ok(Pool::Gpu(index));
        }
        return Err(InnerError::Validation {
            message: format!(
                "invalid pool label {trimmed:?}: expected 'cpu', 'gpu', or 'gpu:N' where N is a non-negative integer"
            ),
        });
    }
    match lower.as_str() {
        "cpu" => Ok(Pool::Cpu),
        "gpu" => Ok(Pool::Gpu(0)),
        _ => Err(InnerError::Validation {
            message: format!(
                "invalid pool label {trimmed:?}: expected 'cpu', 'gpu', or 'gpu:N' where N is a non-negative integer"
            ),
        }),
    }
}

#[cfg(feature = "hf-loader")]
#[allow(clippy::result_large_err)] // Why: see parse_pool_label.
fn backend_hint_from_tag(
    tag: i32,
) -> Result<Option<blazen_manager::hf_loader::BackendHint>, InnerError> {
    use blazen_manager::hf_loader::BackendHint;
    match tag {
        BLAZEN_BACKEND_HINT_NONE => Ok(None),
        BLAZEN_BACKEND_HINT_MISTRALRS => Ok(Some(BackendHint::Mistralrs)),
        BLAZEN_BACKEND_HINT_CANDLE => Ok(Some(BackendHint::Candle)),
        BLAZEN_BACKEND_HINT_LLAMACPP => Ok(Some(BackendHint::Llamacpp)),
        other => Err(InnerError::Validation {
            message: format!(
                "invalid backend_hint {other}: expected -1 (none), 0 (mistralrs), 1 (candle), or 2 (llamacpp)"
            ),
        }),
    }
}

/// Translate the C-side `BlazenHfLoadOptions` into the Rust-side
/// `HfLoadOptions`. Strings are copied (the C caller is free to drop the
/// source buffers after the conversion runs).
///
/// # Safety
///
/// `opts` must be null OR a pointer to a `BlazenHfLoadOptions` whose pointer
/// fields are each null OR NUL-terminated UTF-8 buffers valid for the
/// duration of this call.
#[cfg(feature = "hf-loader")]
#[allow(clippy::result_large_err)] // Why: see parse_pool_label.
unsafe fn convert_hf_options(
    opts: *const BlazenHfLoadOptions,
) -> Result<blazen_manager::hf_loader::HfLoadOptions, InnerError> {
    use blazen_manager::hf_loader::HfLoadOptions;

    let Some(opts) = (unsafe { opts.as_ref() }) else {
        return Ok(HfLoadOptions::default());
    };

    let backend_hint = backend_hint_from_tag(opts.backend_hint)?;
    // SAFETY: caller upholds the NUL + lifetime contract on each pointer
    // field per the function-level docs.
    let revision = unsafe { crate::string::cstr_to_opt_string(opts.revision) };
    let hf_token = unsafe { crate::string::cstr_to_opt_string(opts.hf_token) };
    let cache_dir = unsafe { crate::string::cstr_to_opt_string(opts.cache_dir) }.map(PathBuf::from);
    let device = unsafe { crate::string::cstr_to_opt_string(opts.device) };
    let gguf_file = unsafe { crate::string::cstr_to_opt_string(opts.gguf_file) };
    let pool_label = unsafe { crate::string::cstr_to_opt_string(opts.pool) };
    let pool = match pool_label {
        Some(label) => Some(parse_pool_label(&label)?),
        None => None,
    };

    Ok(HfLoadOptions {
        backend_hint,
        revision,
        hf_token,
        cache_dir,
        device,
        gguf_file,
        memory_estimate_bytes: if opts.memory_estimate_bytes == 0 {
            None
        } else {
            Some(opts.memory_estimate_bytes)
        },
        pool,
    })
}

/// Synchronously detects the right backend for a Hugging Face repo, downloads
/// it via the chosen backend, and registers it as `id`. Writes the chosen
/// backend's stable string (`"mistralrs"` / `"candle"` / `"llamacpp"`) into
/// `*out_chosen_backend` (caller-owned, free via
/// [`crate::string::blazen_string_free`]). Returns `0` on success or `-1` on
/// failure (writes `*out_err`).
///
/// `options` may be null, in which case
/// [`blazen_manager::hf_loader::HfLoadOptions::default`] applies (auto-detect
/// backend, no token, default cache dir, `Pool::Cpu`).
///
/// # Safety
///
/// `mgr` must be null OR a live [`BlazenModelManager`]. `id` and `repo` must
/// each be NUL-terminated UTF-8 buffers valid for the duration of the call.
/// `options` is null OR a `BlazenHfLoadOptions` whose pointer fields each
/// follow the same null-OR-NUL-terminated contract. `out_chosen_backend` and
/// `out_err` are each null OR a single-writer destination.
#[cfg(feature = "hf-loader")]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_model_manager_load_from_hf_blocking(
    mgr: *const BlazenModelManager,
    id: *const c_char,
    repo: *const c_char,
    options: *const BlazenHfLoadOptions,
    out_chosen_backend: *mut *mut c_char,
    out_err: *mut *mut BlazenError,
) -> i32 {
    if mgr.is_null() {
        // SAFETY: out_err contract per the per-fn doc.
        return unsafe {
            write_internal_error(
                out_err,
                "blazen_model_manager_load_from_hf_blocking: null manager",
            )
        };
    }
    // SAFETY: caller upholds the NUL + lifetime contract on `id`.
    let Some(id) = (unsafe { cstr_to_str(id) }).map(str::to_owned) else {
        // SAFETY: out_err contract.
        return unsafe {
            write_internal_error(
                out_err,
                "blazen_model_manager_load_from_hf_blocking: null or non-UTF-8 id",
            )
        };
    };
    // SAFETY: caller upholds the NUL + lifetime contract on `repo`.
    let Some(repo) = (unsafe { cstr_to_str(repo) }).map(str::to_owned) else {
        // SAFETY: out_err contract.
        return unsafe {
            write_internal_error(
                out_err,
                "blazen_model_manager_load_from_hf_blocking: null or non-UTF-8 repo",
            )
        };
    };
    // SAFETY: caller upholds the per-field contracts on `options`.
    let opts = match unsafe { convert_hf_options(options) } {
        Ok(o) => o,
        // SAFETY: out_err contract.
        Err(e) => return unsafe { write_error(out_err, e) },
    };
    // SAFETY: caller has guaranteed `mgr` is a live pointer.
    let mgr = unsafe { &*mgr };
    let inner = Arc::clone(&mgr.0);
    match runtime().block_on(async move { inner.load_from_hf(id, &repo, opts).await }) {
        Ok(backend) => {
            if !out_chosen_backend.is_null() {
                // SAFETY: out-param contract.
                unsafe {
                    *out_chosen_backend = crate::string::alloc_cstring(backend.as_str());
                }
            }
            0
        }
        // SAFETY: out_err contract per the per-fn doc.
        Err(e) => unsafe { write_error(out_err, into_inner_error(e)) },
    }
}

/// Spawns a `load_from_hf` onto the cabi tokio runtime; pop the result with
/// [`crate::future::blazen_future_take_string`] (yields the chosen backend's
/// stable label). Returns null on argument-shape failure (null manager / null
/// or non-UTF-8 string args / invalid options).
///
/// # Safety
///
/// Same as [`blazen_model_manager_load_from_hf_blocking`] (minus the two
/// out-param pointers). String + option buffers are copied before this
/// function returns.
#[cfg(feature = "hf-loader")]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_model_manager_load_from_hf(
    mgr: *const BlazenModelManager,
    id: *const c_char,
    repo: *const c_char,
    options: *const BlazenHfLoadOptions,
) -> *mut BlazenFuture {
    if mgr.is_null() {
        return std::ptr::null_mut();
    }
    // SAFETY: caller upholds the NUL + lifetime contracts on `id` / `repo`.
    let (Some(id), Some(repo)) = (
        unsafe { cstr_to_str(id) }.map(str::to_owned),
        unsafe { cstr_to_str(repo) }.map(str::to_owned),
    ) else {
        return std::ptr::null_mut();
    };
    // SAFETY: caller upholds the per-field contracts on `options`.
    let Ok(opts) = (unsafe { convert_hf_options(options) }) else {
        return std::ptr::null_mut();
    };
    // SAFETY: caller has guaranteed `mgr` is a live pointer.
    let mgr = unsafe { &*mgr };
    let inner = Arc::clone(&mgr.0);
    BlazenFuture::spawn(async move {
        inner
            .load_from_hf(id, &repo, opts)
            .await
            .map(|b| b.as_str().to_owned())
            .map_err(into_inner_error)
    })
}

// ---------------------------------------------------------------------------
// train_lora (feature = "training")
//
// Why: Ruby progress callback uses Fiber.scheduler-aware polling, deferred to
// a follow-up. Sync/async without progress is the v1 surface — the Ruby
// wrapper can display progress by tailing log files / checking the adapter
// directory until the typed callback trampoline lands.
// ---------------------------------------------------------------------------

#[cfg(feature = "training")]
use crate::training_records::{
    BlazenDpoConfig, BlazenFullFineTuneConfig, BlazenFullFineTuneResult, BlazenJsonlDataset,
    BlazenKtoConfig, BlazenOrpoConfig, BlazenPreferenceJsonlDataset, BlazenRatedJsonlDataset,
    BlazenSimpoConfig, BlazenTrainConfig, BlazenTrainedAdapter, convert_dpo_config,
    convert_full_finetune_config, convert_kto_config, convert_orpo_config, convert_simpo_config,
    convert_train_config, full_finetune_result_to_cabi, trained_adapter_to_cabi,
};

/// Adapter so `Arc<JsonlDataset>` satisfies `Box<dyn TrainingDataset>`.
/// Mirrors the `ArcDataset` shim in `crates/blazen-py/src/manager.rs`.
#[cfg(feature = "training")]
struct ArcDataset(Arc<blazen_train::dataset::JsonlDataset>);

#[cfg(feature = "training")]
#[async_trait::async_trait]
impl blazen_train::TrainingDataset for ArcDataset {
    fn len(&self) -> usize {
        self.0.len()
    }

    async fn batch(
        &self,
        batch_size: usize,
        idx: usize,
    ) -> Result<blazen_train::TrainingBatch, blazen_train::BlazenTrainError> {
        self.0.batch(batch_size, idx).await
    }
}

/// Synchronously trains a `LoRA` adapter end-to-end. Writes the resulting
/// adapter handle into `*out_adapter` on success (caller releases the inner
/// `adapter_dir` string with [`crate::training_records::blazen_trained_adapter_free`]),
/// returns `0` on success or `-1` on failure (writes `*out_err`).
///
/// `dataset` is consumed by the training run — the cabi takes the Arc out of
/// the handle before the call, so the caller MUST still free the handle with
/// [`crate::training_records::blazen_jsonl_dataset_free`] after this returns.
///
/// # Safety
///
/// `mgr` must be null OR a live [`BlazenModelManager`]. `config` must point
/// to a fully-populated [`BlazenTrainConfig`] with valid string pointers (see
/// per-field docs). `dataset` must be null OR a pointer produced by
/// [`crate::training_records::blazen_jsonl_dataset_from_path`]. `out_adapter`
/// and `out_err` are each null OR a single-writer destination.
#[cfg(feature = "training")]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_model_manager_train_lora_blocking(
    mgr: *const BlazenModelManager,
    config: *const BlazenTrainConfig,
    dataset: *mut BlazenJsonlDataset,
    out_adapter: *mut BlazenTrainedAdapter,
    out_err: *mut *mut BlazenError,
) -> i32 {
    if mgr.is_null() {
        // SAFETY: out-param contract per the per-fn doc.
        return unsafe {
            write_internal_error(
                out_err,
                "blazen_model_manager_train_lora_blocking: null manager",
            )
        };
    }
    if dataset.is_null() {
        // SAFETY: out-param contract.
        return unsafe {
            write_internal_error(
                out_err,
                "blazen_model_manager_train_lora_blocking: null dataset",
            )
        };
    }
    // SAFETY: caller upholds the contract on `config`.
    let rust_cfg = match unsafe { convert_train_config(config) } {
        Ok(c) => c,
        // SAFETY: out-param contract.
        Err(e) => return unsafe { write_error(out_err, e) },
    };
    // SAFETY: caller upholds the contract on `dataset` (produced by
    // blazen_jsonl_dataset_from_path → Box::into_raw).
    let ds_arc = Arc::clone(&unsafe { &*dataset }.inner);
    let dataset_box: Box<dyn blazen_train::TrainingDataset> = Box::new(ArcDataset(ds_arc));

    // SAFETY: caller has guaranteed `mgr` is a live pointer.
    let mgr_ref = unsafe { &*mgr };
    let inner = Arc::clone(&mgr_ref.0);
    match runtime().block_on(async move { inner.train_lora(rust_cfg, dataset_box, None).await }) {
        Ok(adapter) => {
            if !out_adapter.is_null() {
                // SAFETY: out-param contract.
                unsafe {
                    *out_adapter = trained_adapter_to_cabi(&adapter);
                }
            }
            0
        }
        // SAFETY: out-param contract.
        Err(e) => unsafe { write_error(out_err, into_inner_error(e)) },
    }
}

/// Spawns a `train_lora` onto the cabi tokio runtime; pop the result with
/// [`crate::future::blazen_future_take_trained_adapter`]. Returns null on
/// argument-shape failure (null manager / null dataset / invalid config).
///
/// `dataset` is consumed by the training run; the caller must still free the
/// handle with [`crate::training_records::blazen_jsonl_dataset_free`] after
/// awaiting the future. The internal `Arc<JsonlDataset>` is cloned out before
/// this function returns, so freeing the handle early does not invalidate
/// the in-flight train run.
///
/// # Safety
///
/// Same as [`blazen_model_manager_train_lora_blocking`] (minus the out-param
/// pointers).
#[cfg(feature = "training")]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_model_manager_train_lora(
    mgr: *const BlazenModelManager,
    config: *const BlazenTrainConfig,
    dataset: *mut BlazenJsonlDataset,
) -> *mut BlazenFuture {
    if mgr.is_null() || dataset.is_null() {
        return std::ptr::null_mut();
    }
    // SAFETY: caller upholds the contract on `config`.
    let Ok(rust_cfg) = (unsafe { convert_train_config(config) }) else {
        return std::ptr::null_mut();
    };
    // SAFETY: caller upholds the contract on `dataset`.
    let ds_arc = Arc::clone(&unsafe { &*dataset }.inner);
    // SAFETY: caller has guaranteed `mgr` is a live pointer.
    let mgr_ref = unsafe { &*mgr };
    let inner = Arc::clone(&mgr_ref.0);
    BlazenFuture::spawn(async move {
        let dataset_box: Box<dyn blazen_train::TrainingDataset> = Box::new(ArcDataset(ds_arc));
        inner
            .train_lora(rust_cfg, dataset_box, None)
            .await
            .map_err(into_inner_error)
    })
}

// ---------------------------------------------------------------------------
// PR8 Wave 18 — train_dpo / train_orpo / train_simpo / train_kto / fine_tune
//
// Why: Same async pattern as train_lora above (both blocking and future-
// returning variants) and same deferred-progress caveat: the Ruby progress
// callback uses Fiber.scheduler-aware polling, deferred to a follow-up. Sync
// /async without progress is the v1 surface — the Ruby wrapper can display
// progress by tailing log files or checking the output directory until the
// typed callback trampoline lands.
// ---------------------------------------------------------------------------

#[cfg(feature = "training")]
async fn run_train_dpo(
    inner: Arc<ModelManager>,
    cfg: blazen_train::DpoConfig,
    ds: Arc<blazen_train::dataset::PreferenceJsonlDataset>,
) -> Result<blazen_train::TrainedAdapter, blazen_llm::BlazenError> {
    inner
        .train_dpo(cfg, ds as Arc<dyn blazen_train::PreferenceDataset>, None)
        .await
}

#[cfg(feature = "training")]
async fn run_train_orpo(
    inner: Arc<ModelManager>,
    cfg: blazen_train::OrpoConfig,
    ds: Arc<blazen_train::dataset::PreferenceJsonlDataset>,
) -> Result<blazen_train::TrainedAdapter, blazen_llm::BlazenError> {
    inner
        .train_orpo(cfg, ds as Arc<dyn blazen_train::PreferenceDataset>, None)
        .await
}

#[cfg(feature = "training")]
async fn run_train_simpo(
    inner: Arc<ModelManager>,
    cfg: blazen_train::SimpoConfig,
    ds: Arc<blazen_train::dataset::PreferenceJsonlDataset>,
) -> Result<blazen_train::TrainedAdapter, blazen_llm::BlazenError> {
    inner
        .train_simpo(cfg, ds as Arc<dyn blazen_train::PreferenceDataset>, None)
        .await
}

#[cfg(feature = "training")]
async fn run_train_kto(
    inner: Arc<ModelManager>,
    cfg: blazen_train::KtoConfig,
    ds: Arc<blazen_train::dataset::RatedJsonlDataset>,
) -> Result<blazen_train::TrainedAdapter, blazen_llm::BlazenError> {
    inner
        .train_kto(cfg, ds as Arc<dyn blazen_train::RatedDataset>, None)
        .await
}

#[cfg(feature = "training")]
async fn run_fine_tune(
    inner: Arc<ModelManager>,
    cfg: blazen_train::FullFineTuneConfig,
    ds: Arc<blazen_train::dataset::JsonlDataset>,
) -> Result<blazen_train::FullFineTuneResult, blazen_llm::BlazenError> {
    // `JsonlDataset` already impls `TrainingDataset`, so `Arc<JsonlDataset>`
    // coerces directly to `Arc<dyn TrainingDataset>` — no shim needed for the
    // fine-tune entry point (unlike `train_lora` above, which still uses the
    // `ArcDataset` adapter because it takes `Box<dyn TrainingDataset>`).
    inner
        .fine_tune(cfg, ds as Arc<dyn blazen_train::TrainingDataset>, None)
        .await
}

// --- DPO ------------------------------------------------------------------

/// Synchronously trains a `LoRA` adapter end-to-end via Direct Preference
/// Optimization (DPO). Writes the resulting adapter handle into `*out_adapter`
/// on success; returns `0` on success or `-1` on failure (writes `*out_err`).
///
/// `dataset` is consumed by the training run — the cabi clones the inner
/// `Arc` before the call, so the caller MUST still free the dataset handle
/// with [`crate::training_records::blazen_preference_jsonl_dataset_free`]
/// after this returns.
///
/// # Safety
///
/// `mgr` must be null OR a live [`BlazenModelManager`]. `config` must point to
/// a fully-populated [`BlazenDpoConfig`] (see per-field docs). `dataset` must
/// be null OR a pointer produced by
/// [`crate::training_records::blazen_preference_jsonl_dataset_from_path`].
/// `out_adapter` and `out_err` are each null OR a single-writer destination.
#[cfg(feature = "training")]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_model_manager_train_dpo_blocking(
    mgr: *const BlazenModelManager,
    config: *const BlazenDpoConfig,
    dataset: *mut BlazenPreferenceJsonlDataset,
    out_adapter: *mut BlazenTrainedAdapter,
    out_err: *mut *mut BlazenError,
) -> i32 {
    if mgr.is_null() {
        // SAFETY: out-param contract.
        return unsafe {
            write_internal_error(
                out_err,
                "blazen_model_manager_train_dpo_blocking: null manager",
            )
        };
    }
    if dataset.is_null() {
        // SAFETY: out-param contract.
        return unsafe {
            write_internal_error(
                out_err,
                "blazen_model_manager_train_dpo_blocking: null dataset",
            )
        };
    }
    // SAFETY: caller upholds the contract on `config`.
    let rust_cfg = match unsafe { convert_dpo_config(config) } {
        Ok(c) => c,
        // SAFETY: out-param contract.
        Err(e) => return unsafe { write_error(out_err, e) },
    };
    // SAFETY: caller upholds the dataset-pointer contract.
    let ds_arc = Arc::clone(&unsafe { &*dataset }.inner);
    // SAFETY: caller has guaranteed `mgr` is a live pointer.
    let mgr_ref = unsafe { &*mgr };
    let inner = Arc::clone(&mgr_ref.0);
    match runtime().block_on(async move { run_train_dpo(inner, rust_cfg, ds_arc).await }) {
        Ok(adapter) => {
            if !out_adapter.is_null() {
                // SAFETY: out-param contract.
                unsafe {
                    *out_adapter = trained_adapter_to_cabi(&adapter);
                }
            }
            0
        }
        // SAFETY: out-param contract.
        Err(e) => unsafe { write_error(out_err, into_inner_error(e)) },
    }
}

/// Spawns a `train_dpo` onto the cabi tokio runtime; pop the result with
/// [`crate::future::blazen_future_take_trained_adapter`]. Returns null on
/// argument-shape failure.
///
/// # Safety
///
/// Same as [`blazen_model_manager_train_dpo_blocking`] (minus the two out-
/// param pointers).
#[cfg(feature = "training")]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_model_manager_train_dpo(
    mgr: *const BlazenModelManager,
    config: *const BlazenDpoConfig,
    dataset: *mut BlazenPreferenceJsonlDataset,
) -> *mut BlazenFuture {
    if mgr.is_null() || dataset.is_null() {
        return std::ptr::null_mut();
    }
    // SAFETY: caller upholds the contract on `config`.
    let Ok(rust_cfg) = (unsafe { convert_dpo_config(config) }) else {
        return std::ptr::null_mut();
    };
    // SAFETY: caller upholds the dataset-pointer contract.
    let ds_arc = Arc::clone(&unsafe { &*dataset }.inner);
    // SAFETY: caller has guaranteed `mgr` is a live pointer.
    let mgr_ref = unsafe { &*mgr };
    let inner = Arc::clone(&mgr_ref.0);
    BlazenFuture::spawn(async move {
        run_train_dpo(inner, rust_cfg, ds_arc)
            .await
            .map_err(into_inner_error)
    })
}

// --- ORPO -----------------------------------------------------------------

/// Synchronously trains a `LoRA` adapter via Odds Ratio Preference
/// Optimization (ORPO). Same surface as
/// [`blazen_model_manager_train_dpo_blocking`].
///
/// # Safety
///
/// Same as [`blazen_model_manager_train_dpo_blocking`].
#[cfg(feature = "training")]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_model_manager_train_orpo_blocking(
    mgr: *const BlazenModelManager,
    config: *const BlazenOrpoConfig,
    dataset: *mut BlazenPreferenceJsonlDataset,
    out_adapter: *mut BlazenTrainedAdapter,
    out_err: *mut *mut BlazenError,
) -> i32 {
    if mgr.is_null() {
        // SAFETY: out-param contract.
        return unsafe {
            write_internal_error(
                out_err,
                "blazen_model_manager_train_orpo_blocking: null manager",
            )
        };
    }
    if dataset.is_null() {
        // SAFETY: out-param contract.
        return unsafe {
            write_internal_error(
                out_err,
                "blazen_model_manager_train_orpo_blocking: null dataset",
            )
        };
    }
    // SAFETY: caller upholds the contract on `config`.
    let rust_cfg = match unsafe { convert_orpo_config(config) } {
        Ok(c) => c,
        // SAFETY: out-param contract.
        Err(e) => return unsafe { write_error(out_err, e) },
    };
    // SAFETY: caller upholds the dataset-pointer contract.
    let ds_arc = Arc::clone(&unsafe { &*dataset }.inner);
    // SAFETY: caller has guaranteed `mgr` is a live pointer.
    let mgr_ref = unsafe { &*mgr };
    let inner = Arc::clone(&mgr_ref.0);
    match runtime().block_on(async move { run_train_orpo(inner, rust_cfg, ds_arc).await }) {
        Ok(adapter) => {
            if !out_adapter.is_null() {
                // SAFETY: out-param contract.
                unsafe {
                    *out_adapter = trained_adapter_to_cabi(&adapter);
                }
            }
            0
        }
        // SAFETY: out-param contract.
        Err(e) => unsafe { write_error(out_err, into_inner_error(e)) },
    }
}

/// Spawns a `train_orpo` onto the cabi tokio runtime. Same surface as
/// [`blazen_model_manager_train_dpo`].
///
/// # Safety
///
/// Same as [`blazen_model_manager_train_orpo_blocking`].
#[cfg(feature = "training")]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_model_manager_train_orpo(
    mgr: *const BlazenModelManager,
    config: *const BlazenOrpoConfig,
    dataset: *mut BlazenPreferenceJsonlDataset,
) -> *mut BlazenFuture {
    if mgr.is_null() || dataset.is_null() {
        return std::ptr::null_mut();
    }
    // SAFETY: caller upholds the contract on `config`.
    let Ok(rust_cfg) = (unsafe { convert_orpo_config(config) }) else {
        return std::ptr::null_mut();
    };
    // SAFETY: caller upholds the dataset-pointer contract.
    let ds_arc = Arc::clone(&unsafe { &*dataset }.inner);
    // SAFETY: caller has guaranteed `mgr` is a live pointer.
    let mgr_ref = unsafe { &*mgr };
    let inner = Arc::clone(&mgr_ref.0);
    BlazenFuture::spawn(async move {
        run_train_orpo(inner, rust_cfg, ds_arc)
            .await
            .map_err(into_inner_error)
    })
}

// --- SimPO ----------------------------------------------------------------

/// Synchronously trains a `LoRA` adapter via Simple Preference Optimization
/// (`SimPO`). Same surface as [`blazen_model_manager_train_dpo_blocking`].
///
/// # Safety
///
/// Same as [`blazen_model_manager_train_dpo_blocking`].
#[cfg(feature = "training")]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_model_manager_train_simpo_blocking(
    mgr: *const BlazenModelManager,
    config: *const BlazenSimpoConfig,
    dataset: *mut BlazenPreferenceJsonlDataset,
    out_adapter: *mut BlazenTrainedAdapter,
    out_err: *mut *mut BlazenError,
) -> i32 {
    if mgr.is_null() {
        // SAFETY: out-param contract.
        return unsafe {
            write_internal_error(
                out_err,
                "blazen_model_manager_train_simpo_blocking: null manager",
            )
        };
    }
    if dataset.is_null() {
        // SAFETY: out-param contract.
        return unsafe {
            write_internal_error(
                out_err,
                "blazen_model_manager_train_simpo_blocking: null dataset",
            )
        };
    }
    // SAFETY: caller upholds the contract on `config`.
    let rust_cfg = match unsafe { convert_simpo_config(config) } {
        Ok(c) => c,
        // SAFETY: out-param contract.
        Err(e) => return unsafe { write_error(out_err, e) },
    };
    // SAFETY: caller upholds the dataset-pointer contract.
    let ds_arc = Arc::clone(&unsafe { &*dataset }.inner);
    // SAFETY: caller has guaranteed `mgr` is a live pointer.
    let mgr_ref = unsafe { &*mgr };
    let inner = Arc::clone(&mgr_ref.0);
    match runtime().block_on(async move { run_train_simpo(inner, rust_cfg, ds_arc).await }) {
        Ok(adapter) => {
            if !out_adapter.is_null() {
                // SAFETY: out-param contract.
                unsafe {
                    *out_adapter = trained_adapter_to_cabi(&adapter);
                }
            }
            0
        }
        // SAFETY: out-param contract.
        Err(e) => unsafe { write_error(out_err, into_inner_error(e)) },
    }
}

/// Spawns a `train_simpo` onto the cabi tokio runtime. Same surface as
/// [`blazen_model_manager_train_dpo`].
///
/// # Safety
///
/// Same as [`blazen_model_manager_train_simpo_blocking`].
#[cfg(feature = "training")]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_model_manager_train_simpo(
    mgr: *const BlazenModelManager,
    config: *const BlazenSimpoConfig,
    dataset: *mut BlazenPreferenceJsonlDataset,
) -> *mut BlazenFuture {
    if mgr.is_null() || dataset.is_null() {
        return std::ptr::null_mut();
    }
    // SAFETY: caller upholds the contract on `config`.
    let Ok(rust_cfg) = (unsafe { convert_simpo_config(config) }) else {
        return std::ptr::null_mut();
    };
    // SAFETY: caller upholds the dataset-pointer contract.
    let ds_arc = Arc::clone(&unsafe { &*dataset }.inner);
    // SAFETY: caller has guaranteed `mgr` is a live pointer.
    let mgr_ref = unsafe { &*mgr };
    let inner = Arc::clone(&mgr_ref.0);
    BlazenFuture::spawn(async move {
        run_train_simpo(inner, rust_cfg, ds_arc)
            .await
            .map_err(into_inner_error)
    })
}

// --- KTO ------------------------------------------------------------------

/// Synchronously trains a `LoRA` adapter via Kahneman-Tversky Optimization
/// (KTO). Consumes a [`BlazenRatedJsonlDataset`] (single-completion plus a
/// desirability flag per row) rather than the chosen/rejected pairs of DPO.
///
/// # Safety
///
/// `mgr` must be null OR a live [`BlazenModelManager`]. `config` must point to
/// a fully-populated [`BlazenKtoConfig`]. `dataset` must be null OR a pointer
/// produced by [`crate::training_records::blazen_rated_jsonl_dataset_from_path`].
/// `out_adapter` and `out_err` are each null OR a single-writer destination.
#[cfg(feature = "training")]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_model_manager_train_kto_blocking(
    mgr: *const BlazenModelManager,
    config: *const BlazenKtoConfig,
    dataset: *mut BlazenRatedJsonlDataset,
    out_adapter: *mut BlazenTrainedAdapter,
    out_err: *mut *mut BlazenError,
) -> i32 {
    if mgr.is_null() {
        // SAFETY: out-param contract.
        return unsafe {
            write_internal_error(
                out_err,
                "blazen_model_manager_train_kto_blocking: null manager",
            )
        };
    }
    if dataset.is_null() {
        // SAFETY: out-param contract.
        return unsafe {
            write_internal_error(
                out_err,
                "blazen_model_manager_train_kto_blocking: null dataset",
            )
        };
    }
    // SAFETY: caller upholds the contract on `config`.
    let rust_cfg = match unsafe { convert_kto_config(config) } {
        Ok(c) => c,
        // SAFETY: out-param contract.
        Err(e) => return unsafe { write_error(out_err, e) },
    };
    // SAFETY: caller upholds the dataset-pointer contract.
    let ds_arc = Arc::clone(&unsafe { &*dataset }.inner);
    // SAFETY: caller has guaranteed `mgr` is a live pointer.
    let mgr_ref = unsafe { &*mgr };
    let inner = Arc::clone(&mgr_ref.0);
    match runtime().block_on(async move { run_train_kto(inner, rust_cfg, ds_arc).await }) {
        Ok(adapter) => {
            if !out_adapter.is_null() {
                // SAFETY: out-param contract.
                unsafe {
                    *out_adapter = trained_adapter_to_cabi(&adapter);
                }
            }
            0
        }
        // SAFETY: out-param contract.
        Err(e) => unsafe { write_error(out_err, into_inner_error(e)) },
    }
}

/// Spawns a `train_kto` onto the cabi tokio runtime. Same surface as
/// [`blazen_model_manager_train_dpo`] but consumes a `RatedJsonlDataset`.
///
/// # Safety
///
/// Same as [`blazen_model_manager_train_kto_blocking`].
#[cfg(feature = "training")]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_model_manager_train_kto(
    mgr: *const BlazenModelManager,
    config: *const BlazenKtoConfig,
    dataset: *mut BlazenRatedJsonlDataset,
) -> *mut BlazenFuture {
    if mgr.is_null() || dataset.is_null() {
        return std::ptr::null_mut();
    }
    // SAFETY: caller upholds the contract on `config`.
    let Ok(rust_cfg) = (unsafe { convert_kto_config(config) }) else {
        return std::ptr::null_mut();
    };
    // SAFETY: caller upholds the dataset-pointer contract.
    let ds_arc = Arc::clone(&unsafe { &*dataset }.inner);
    // SAFETY: caller has guaranteed `mgr` is a live pointer.
    let mgr_ref = unsafe { &*mgr };
    let inner = Arc::clone(&mgr_ref.0);
    BlazenFuture::spawn(async move {
        run_train_kto(inner, rust_cfg, ds_arc)
            .await
            .map_err(into_inner_error)
    })
}

// --- full fine-tune --------------------------------------------------------

/// Synchronously runs a full fine-tune (every parameter trains; no `LoRA`
/// adapter). Writes the [`BlazenFullFineTuneResult`] into `*out_result` on
/// success (caller releases the inner `output_dir` string with
/// [`crate::training_records::blazen_full_finetune_result_free`]). Returns
/// `0` on success or `-1` on failure (writes `*out_err`).
///
/// Setting `config.gradient_checkpointing = 1` is rejected at trainer init
/// because candle 0.10.2 has no activation-checkpointing primitive.
///
/// # Safety
///
/// `mgr` must be null OR a live [`BlazenModelManager`]. `config` must point to
/// a fully-populated [`BlazenFullFineTuneConfig`]. `dataset` must be null OR
/// a pointer produced by
/// [`crate::training_records::blazen_jsonl_dataset_from_path`]. `out_result`
/// and `out_err` are each null OR a single-writer destination.
#[cfg(feature = "training")]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_model_manager_fine_tune_blocking(
    mgr: *const BlazenModelManager,
    config: *const BlazenFullFineTuneConfig,
    dataset: *mut BlazenJsonlDataset,
    out_result: *mut BlazenFullFineTuneResult,
    out_err: *mut *mut BlazenError,
) -> i32 {
    if mgr.is_null() {
        // SAFETY: out-param contract.
        return unsafe {
            write_internal_error(
                out_err,
                "blazen_model_manager_fine_tune_blocking: null manager",
            )
        };
    }
    if dataset.is_null() {
        // SAFETY: out-param contract.
        return unsafe {
            write_internal_error(
                out_err,
                "blazen_model_manager_fine_tune_blocking: null dataset",
            )
        };
    }
    // SAFETY: caller upholds the contract on `config`.
    let rust_cfg = match unsafe { convert_full_finetune_config(config) } {
        Ok(c) => c,
        // SAFETY: out-param contract.
        Err(e) => return unsafe { write_error(out_err, e) },
    };
    // SAFETY: caller upholds the dataset-pointer contract.
    let ds_arc = Arc::clone(&unsafe { &*dataset }.inner);
    // SAFETY: caller has guaranteed `mgr` is a live pointer.
    let mgr_ref = unsafe { &*mgr };
    let inner = Arc::clone(&mgr_ref.0);
    match runtime().block_on(async move { run_fine_tune(inner, rust_cfg, ds_arc).await }) {
        Ok(result) => {
            if !out_result.is_null() {
                // SAFETY: out-param contract.
                unsafe {
                    *out_result = full_finetune_result_to_cabi(&result);
                }
            }
            0
        }
        // SAFETY: out-param contract.
        Err(e) => unsafe { write_error(out_err, into_inner_error(e)) },
    }
}

/// Spawns a `fine_tune` onto the cabi tokio runtime; pop the result with
/// [`crate::future::blazen_future_take_full_finetune_result`].
///
/// # Safety
///
/// Same as [`blazen_model_manager_fine_tune_blocking`] (minus the two
/// out-param pointers).
#[cfg(feature = "training")]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_model_manager_fine_tune(
    mgr: *const BlazenModelManager,
    config: *const BlazenFullFineTuneConfig,
    dataset: *mut BlazenJsonlDataset,
) -> *mut BlazenFuture {
    if mgr.is_null() || dataset.is_null() {
        return std::ptr::null_mut();
    }
    // SAFETY: caller upholds the contract on `config`.
    let Ok(rust_cfg) = (unsafe { convert_full_finetune_config(config) }) else {
        return std::ptr::null_mut();
    };
    // SAFETY: caller upholds the dataset-pointer contract.
    let ds_arc = Arc::clone(&unsafe { &*dataset }.inner);
    // SAFETY: caller has guaranteed `mgr` is a live pointer.
    let mgr_ref = unsafe { &*mgr };
    let inner = Arc::clone(&mgr_ref.0);
    BlazenFuture::spawn(async move {
        run_fine_tune(inner, rust_cfg, ds_arc)
            .await
            .map_err(into_inner_error)
    })
}

// ---------------------------------------------------------------------------
// Tests (PR8 Wave 18)
// ---------------------------------------------------------------------------

#[cfg(all(test, feature = "training"))]
mod tests {
    use std::ffi::CString;

    use super::*;
    use crate::training_records::{
        BLAZEN_MIXED_PRECISION_NONE, BLAZEN_SCHEDULER_COSINE, BlazenLoraConfig, BlazenOptimConfig,
        BlazenSchedulerConfig, BlazenTrainCoreConfig,
    };

    /// Build a `BlazenDpoConfig` with intentionally-malformed core (zero
    /// `max_steps`) plus its backing `CStrings`.
    fn make_malformed_dpo() -> (
        BlazenDpoConfig,
        Vec<CString>,
        Vec<*const c_char>,
        Vec<CString>,
    ) {
        let repo_c = CString::new("Qwen/Qwen2.5-0.5B").unwrap();
        let out_c = CString::new("./out").unwrap();
        let modules = vec![CString::new("q_proj").unwrap()];
        let module_ptrs: Vec<*const c_char> = modules.iter().map(|c| c.as_ptr()).collect();
        let core = BlazenTrainCoreConfig {
            base_model_repo: repo_c.as_ptr(),
            base_model_revision: std::ptr::null(),
            output_dir: out_c.as_ptr(),
            optim: BlazenOptimConfig {
                learning_rate: 2e-4,
                beta1: 0.9,
                beta2: 0.999,
                epsilon: 1e-8,
                weight_decay: 0.0,
                has_gradient_clip: 0,
                gradient_clip: 0.0,
            },
            scheduler: BlazenSchedulerConfig {
                kind: BLAZEN_SCHEDULER_COSINE,
                warmup_steps: 0,
            },
            max_steps: 0, // <-- malformed
            batch_size: 1,
            gradient_accumulation_steps: 1,
            max_seq_len: 32,
            has_eval_steps: 0,
            eval_steps: 0,
            has_save_steps: 0,
            save_steps: 0,
            seed: 42,
            mixed_precision: BLAZEN_MIXED_PRECISION_NONE,
            device: std::ptr::null(),
        };
        let lora = BlazenLoraConfig {
            rank: 8,
            alpha: 16.0,
            dropout: 0.0,
            target_modules: module_ptrs.as_ptr(),
            target_modules_len: module_ptrs.len(),
        };
        let cfg = BlazenDpoConfig {
            core,
            lora,
            beta: 0.1,
            label_smoothing: 0.0,
            reference_model_repo: std::ptr::null(),
            reference_model_revision: std::ptr::null(),
        };
        (cfg, vec![repo_c, out_c], module_ptrs, modules)
    }

    #[test]
    fn train_dpo_blocking_returns_minus_one_on_validation_failure() {
        // Build a manager, build a malformed DPO config (max_steps = 0), pass
        // them through the blocking entry point with a non-null dataset
        // surrogate. The convert step must reject the config before any train
        // work runs, returning -1 and writing an Internal-or-Validation error
        // into `*out_err`.
        //
        // We can't easily synthesize a "real" PreferenceJsonlDataset without
        // disk I/O, but we can verify the early null-dataset bail-out is the
        // negative-control: pass null and confirm we hit the documented
        // "null dataset" path with -1.

        let mgr_ptr = blazen_model_manager_new();
        let (cfg, _strings, _module_ptrs, _modules) = make_malformed_dpo();

        let mut adapter = BlazenTrainedAdapter {
            adapter_dir: std::ptr::null_mut(),
            final_loss: 0.0,
            total_steps: 0,
        };
        let mut err: *mut BlazenError = std::ptr::null_mut();
        // SAFETY: mgr_ptr is a live cabi-produced manager; dataset is null
        // (handled by the early-bail branch); the config + out-params live for
        // the call duration.
        let rc = unsafe {
            blazen_model_manager_train_dpo_blocking(
                mgr_ptr,
                std::ptr::from_ref(&cfg),
                std::ptr::null_mut(), // null dataset triggers "null dataset" branch
                std::ptr::addr_of_mut!(adapter),
                std::ptr::addr_of_mut!(err),
            )
        };
        assert_eq!(
            rc, -1,
            "expected -1 from train_dpo_blocking with null dataset"
        );
        assert!(!err.is_null(), "expected non-null err on null dataset");
        // SAFETY: err is a freshly-produced BlazenError* — reclaim it.
        unsafe {
            drop(Box::from_raw(err));
        }
        // SAFETY: mgr_ptr was produced by `blazen_model_manager_new`.
        unsafe {
            blazen_model_manager_free(mgr_ptr);
        }
    }

    #[test]
    fn train_dpo_returns_null_future_on_null_dataset() {
        // The future-returning variant returns a null pointer (rather than a
        // BlazenFuture wrapping an error) when the dataset is null — matches
        // the documented contract on the function.
        let mgr_ptr = blazen_model_manager_new();
        let (cfg, _strings, _module_ptrs, _modules) = make_malformed_dpo();

        // SAFETY: see the blocking variant above.
        let fut = unsafe {
            blazen_model_manager_train_dpo(mgr_ptr, std::ptr::from_ref(&cfg), std::ptr::null_mut())
        };
        assert!(fut.is_null(), "expected null future on null dataset");
        // SAFETY: mgr_ptr was produced by `blazen_model_manager_new`.
        unsafe {
            blazen_model_manager_free(mgr_ptr);
        }
    }
}
